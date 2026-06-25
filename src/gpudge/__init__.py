# src/gpudge/__init__.py
"""gpudge -- lightweight GPU-only Mann-Whitney U DGE."""
from __future__ import annotations

import math
import os
import warnings
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from typing import Literal

import anndata as ad
import numpy as np
import polars as pl
import scipy.sparse as sp
import torch

from ._csr_dense import HAS_NUMBA, csr_row_sums, csr_rows_col_range_to_dense
from ._ingest import ALL_OTHERS, LEGACY_ALL_OTHERS as _LEGACY_ALL_OTHERS, ingest
from ._means import group_means
from ._mwu import _rank_with_ties, _tie_term_per_gene
from ._fdr import bh_per_group
from ._output import DEFAULT_OUTPUT_COLUMNS, assemble_dataframe
from ._stream import (
    _auto_gene_chunk_size,
    run_gene_chunks_with_recovery,
)
from ._shard_stream import group_chunk_stats

try:
    __version__ = _pkg_version("gpudge")
except PackageNotFoundError:
    # Package is being imported from source tree without an installed
    # distribution (e.g. during dev with PYTHONPATH=src). Fall back to a
    # placeholder rather than crashing.
    __version__ = "0+unknown"

# Warn once per process if the numba fast path is unavailable. Reduces the
# 3x scipy-only slowdown to a visible signal so users can opt into [fast].
if not HAS_NUMBA:
    warnings.warn(
        "gpudge: numba is not installed; falling back to scipy "
        "for sparse CSR row slicing (~3x slower on cell line 2-scale inputs). "
        "Install with `pip install gpudge[fast]` (or "
        "`uv sync --extra fast`) to enable the numba kernel.",
        stacklevel=2,
    )

__all__ = ["de", "ALL_OTHERS", "MeanCalc", "__version__"]

MeanCalc = Literal["arithmetic", "geometric"]


class _Removed:
    """Sentinel for a removed parameter that still wants a helpful error."""


_REMOVED = _Removed()


def _row_col_slice_np(
    X,
    rows,
    col_start: int,
    col_stop: int,
    *,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Dense (n_rows, cols) float32 numpy array (sparse or dense X).

    Delegates to ``csr_rows_col_range_to_dense`` which uses a one-pass
    numba kernel for sparse CSR when ``numba`` is installed (the ``[fast]``
    extra), and falls back to scipy's two-step slice + toarray otherwise.
    For dense X, returns ``np.ascontiguousarray`` of the slice.

    ``out``: optional pre-allocated float32 buffer to write into. Forwarded
    to the underlying CSR fast path; ignored on non-CSR / no-numba paths.
    """
    return csr_rows_col_range_to_dense(
        X, rows, col_start, col_stop, out=out)


def _refmode_chunk_keep(
    start, stop, ch,
    arith_target_acc, arith_ref_acc, other_target_acc, other_ref_acc,
    counts, ref_label_idx, group_libtot, target_sum,
    min_mean_value, min_total_value, min_cpm_cell, min_cpm_bulk,
    keep_genes_arr,
):
    from ._filter import combined_keep_mask
    n_groups = arith_target_acc.shape[0]
    sl = slice(start, stop)
    if target_sum is not None:
        # main holds the normalized unit; cpm filter unit = main * 1e6/target_sum
        from ._normalize import cpm_rescale_factor
        f = cpm_rescale_factor(target_sum)
        unscaled_t = other_target_acc[:, sl] if other_target_acc is not None else None
        unscaled_r = other_ref_acc[sl] if other_ref_acc is not None else None
        scaled_t = arith_target_acc[:, sl] * f
        scaled_r = arith_ref_acc[sl] * f
    else:
        unscaled_t = arith_target_acc[:, sl]
        unscaled_r = arith_ref_acc[sl]
        scaled_t = other_target_acc[:, sl] if other_target_acc is not None else None
        scaled_r = other_ref_acc[sl] if other_ref_acc is not None else None

    n_ref = counts[ref_label_idx]
    filters = []
    if min_mean_value is not None:
        filters.append((unscaled_t, unscaled_r, float(min_mean_value)))
    if min_total_value is not None:
        filters.append((unscaled_t * counts[:, None],
                        unscaled_r * float(n_ref), float(min_total_value)))
    if min_cpm_cell is not None:
        filters.append((scaled_t, scaled_r, float(min_cpm_cell)))
    if min_cpm_bulk is not None:
        libtot_safe = np.where(group_libtot == 0, 1.0, group_libtot)
        ref_libtot = group_libtot[ref_label_idx]
        ref_libtot_safe = ref_libtot if ref_libtot != 0 else 1.0
        bulk_t = (unscaled_t * counts[:, None]) / libtot_safe[:, None] * 1e6
        bulk_r = (unscaled_r * float(n_ref)) / ref_libtot_safe * 1e6
        filters.append((bulk_t, bulk_r, float(min_cpm_bulk)))
    return combined_keep_mask(
        n_groups, ch, filters=filters,
        keep_genes=(keep_genes_arr[sl] if keep_genes_arr is not None else None))


def _all_others_chunk_keep(
    start, stop, ch,
    arith_np, other_target_acc, counts_np, rest_count_safe,
    group_libtot, target_sum,
    min_mean_value, min_total_value, min_cpm_cell, min_cpm_bulk,
    keep_genes_arr,
):
    from ._filter import combined_keep_mask
    n_groups = arith_np.shape[0]
    sl = slice(start, stop)
    if target_sum is not None:
        from ._normalize import cpm_rescale_factor
        f = cpm_rescale_factor(target_sum)
        unscaled_t = other_target_acc[:, sl] if other_target_acc is not None else None
        scaled_t = arith_np * f
    else:
        unscaled_t = arith_np
        scaled_t = other_target_acc[:, sl] if other_target_acc is not None else None

    def _rest_mean(per_group):  # (sum_all - sum_g) / rest_count, per group
        sum_per_group = per_group * counts_np[:, None]
        sum_all = sum_per_group.sum(axis=0)
        return (sum_all[None, :] - sum_per_group) / rest_count_safe[:, None]

    filters = []
    if min_mean_value is not None:
        filters.append((unscaled_t, _rest_mean(unscaled_t), float(min_mean_value)))
    if min_total_value is not None:
        tot_t = unscaled_t * counts_np[:, None]
        rest_tot = tot_t.sum(axis=0)[None, :] - tot_t
        filters.append((tot_t, rest_tot, float(min_total_value)))
    if min_cpm_cell is not None:
        filters.append((scaled_t, _rest_mean(scaled_t), float(min_cpm_cell)))
    if min_cpm_bulk is not None:
        libtot_safe = np.where(group_libtot == 0, 1.0, group_libtot)
        rest_libtot = group_libtot.sum() - group_libtot
        rest_libtot_safe = np.where(rest_libtot == 0, 1.0, rest_libtot)
        tot_t = unscaled_t * counts_np[:, None]
        rest_tot = tot_t.sum(axis=0)[None, :] - tot_t
        bulk_t = tot_t / libtot_safe[:, None] * 1e6
        bulk_r = rest_tot / rest_libtot_safe[:, None] * 1e6
        filters.append((bulk_t, bulk_r, float(min_cpm_bulk)))
    return combined_keep_mask(
        n_groups, ch, filters=filters,
        keep_genes=(keep_genes_arr[sl] if keep_genes_arr is not None else None))


def de(
    adata: ad.AnnData | None = None,
    *,
    shard_archive: str | os.PathLike | None = None,
    groupby: str | None = None,
    reference: str | ad.AnnData | None = None,   # str | ALL_OTHERS sentinel | ad.AnnData | None
    mean_calc: MeanCalc = "arithmetic",
    epsilon: float = 1e-9,
    min_feature_filter=_REMOVED,
    gpu_gene_chunk_size: int | None = None,
    oom_recovery: bool = True,
    densify_input: bool = False,
    cpm_normalize: bool = False,
    normalize_target_sum: float | int | str | None = None,
    output_columns: dict[str, str] | None = None,
    filter_gene_min_mean_value: float | None = None,
    filter_gene_min_total_value: float | None = None,
    filter_gene_min_cpm_cell: float | None = None,
    filter_gene_min_cpm_bulk: float | None = None,
    keep_genes: np.ndarray | None = None,
) -> pl.DataFrame:
    """Per-(target, feature) differential expression on GPU.

    GPU-only Mann–Whitney U with per-group BH-FDR and optional opt-in
    per-gene expression filters. All transformations (CPM, log1p, etc.)
    are the caller's responsibility unless a ``filter_gene_*`` or
    ``cpm_normalize`` parameter handles them inline.

    Parameters
    ----------
    adata : anndata.AnnData
        Single-cell expression matrix. Dense or sparse CSR X is accepted;
        sparse X is streamed to GPU per gene-chunk.
    groupby : str
        Column in ``adata.obs`` that defines the groups (e.g. guide identity).
    reference : str | anndata.AnnData | None
        Name of the reference group in ``adata.obs[groupby]``, OR the
        ``ALL_OTHERS`` sentinel (``"__all_others__"``) for 1-vs-rest
        comparisons. The pre-v0.1 spelling ``"all_others"`` is still
        accepted with a ``DeprecationWarning`` and will be removed in
        a future release; pass ``ALL_OTHERS`` (or the new string) instead.
        When streaming (``shard_archive=``), ``reference`` may instead be an
        ``AnnData`` external control pool (Mode 2). If the archive also
        designates its own reference shard, the external pool wins and the
        archive's reference shard is ignored (a ``UserWarning`` is emitted).
    mean_calc : {"arithmetic", "geometric"}, default "arithmetic"
        How ``target_mean`` and ``ref_mean`` (and the log2 fold change derived
        from them) are computed. Independent of any active gene filter, which
        always uses arithmetic means.
    epsilon : float, default 1e-9
        Pseudocount inside ``log2((target_mean + epsilon) / (ref_mean + epsilon))``.
        Default matches ``scanpy.tl.rank_genes_groups``.
    filter_gene_min_mean_value : float | None, default None
        Keep a ``(target, gene)`` row if the per-group arithmetic mean of
        ``adata.X`` **as supplied** — in the target group OR the reference
        group — exceeds this threshold. Unit-agnostic: the filter operates
        on whatever units ``adata.X`` carries (counts, CPM, log1p-CPM, …);
        no warning is emitted. ``None`` = filter off. A negative threshold
        is treated as keep-all (every gene passes). ``0.0`` drops genes whose
        per-group mean is zero or negative in both the target and reference.
    filter_gene_min_total_value : float | None, default None
        Like ``filter_gene_min_mean_value``, but the threshold is applied to
        the per-group **sum** (mean × cell count) of ``adata.X`` as supplied.
        Unit-agnostic; no warning. Same ``None``/negative/0.0 semantics.
    filter_gene_min_cpm_cell : float | None, default None
        Keep a ``(target, gene)`` row if the mean of per-cell CPM — computed
        as ``(gene_count_in_cell / cell_library_size) × 1e6`` — in the target group
        OR the reference group exceeds this threshold. **Assumes ``adata.X``
        contains raw counts.** Emits a ``UserWarning`` once (per ``de()``
        call) if ``adata.X`` contains fractional or negative values, or if
        any cell has a negative library size. Same ``None``/negative/0.0
        semantics.
    filter_gene_min_cpm_bulk : float | None, default None
        Keep a ``(target, gene)`` row if the **pooled bulk CPM** —
        ``Σcounts / Σlibsize × 1e6`` over all cells in the group — in the
        target group OR the reference group exceeds this threshold. **Assumes
        raw counts.** Emits the same ``UserWarning`` as
        ``filter_gene_min_cpm_cell`` on non-integer/negative X. Same
        ``None``/negative/0.0 semantics.
    keep_genes : np.ndarray | None, default None
        A per-gene boolean mask of dtype ``np.bool_``, length ``n_vars``,
        aligned to ``adata.var_names``. When provided it is AND-combined with
        any active ``filter_gene_*`` filters: only genes where
        ``keep_genes[i]`` is ``True`` can survive. Use this as an escape
        hatch when ``adata.X`` is not in raw counts and you want to supply a
        pre-computed inclusion mask instead of (or in addition to) the
        ``filter_gene_*`` thresholds.

    gpu_gene_chunk_size : int | None, default None
        Number of genes per GPU pass. ``None`` auto-picks from free device
        memory. Smaller values reduce GPU memory but increase per-chunk
        overhead.
    oom_recovery : bool, default True
        If True, a CUDA OOM while processing a gene-chunk halves
        ``gpu_gene_chunk_size`` (to a floor of 64, or half a smaller explicit
        chunk) and retries, logging
        each downshift — for both auto and explicit chunk sizes, and for both
        the literal-``reference`` and ``ALL_OTHERS`` (one-vs-rest) paths. If
        False, the first OOM raises; an explicit ``gpu_gene_chunk_size`` is then
        honored exactly (use False for benchmarking, where a labeled chunk must
        be that chunk or an error). Results are identical regardless of chunk
        size.
    densify_input : bool, default False
        If True and ``adata.X`` is sparse, **mutate ``adata.X`` in place** to a
        dense numpy array before the chunk loop (i.e. ``adata.X =
        adata.X.toarray()``). The sparse matrix is dropped after this point.
        Trades n_cells × n_genes × 4 bytes of host RAM (~153 GB steady-state
        for cell line 2; up to ~310 GB peak during the in-place sparse→dense swap) for
        ~30-40% faster per-chunk per-group slicing (numpy fancy indexing
        instead of repeated CSR slicing). The caller must be OK with the
        sparse → dense replacement; pass ``adata.copy()`` first to preserve
        the original. Note: just setting ``adata.X = ...`` without dropping
        the sparse first (e.g. holding both in separate variables) makes this
        slower not faster, because both representations coexist; we do the
        replacement inside de() so the rebind drops the sparse refcount to 0.
    cpm_normalize : bool, default False
        If True, normalize each cell to 1e6 total counts on the fly, inside
        the chunk loop. Row sums are computed once over the full X before
        the loop; each per-chunk slice is then multiplied by ``1e6 /
        row_sum`` on the GPU after upload. Matches the result of
        ``scanpy.pp.normalize_total(adata, target_sum=1e6)`` but does not
        mutate ``adata.X``. Use when you want to feed raw counts and skip
        the upfront ``normalize_total`` pass.
    normalize_target_sum : float | int | str | None, default None
        On-the-fly per-cell library-size normalization, applied like
        ``cpm_normalize`` (inside the chunk loop, **without** mutating
        ``adata.X``). ``None`` = off. A positive number normalizes each cell so
        its total counts equal that value — equivalent to
        ``scanpy.pp.normalize_total(adata, target_sum=N)``. The string
        ``"median"`` normalizes to the median of per-cell total counts over
        cells with a positive total — scanpy's default
        ``normalize_total(target_sum=None)``. ``cpm_normalize=True`` is exactly
        ``normalize_target_sum=1e6``; **only one** of the two may be set (else
        ``ValueError``). Note the naming nuance: scanpy spells "use the median"
        as ``target_sum=None``, whereas here ``None`` means "off" and the median
        is requested with the explicit string ``"median"``.
    output_columns : dict[str, str] | None, default None
        If provided, output only these columns (the dict keys), renamed to
        the dict values. Keys must be from the default output column set:
        ``target``, ``feature``, ``target_mean``, ``ref_mean``,
        ``target_ncells``, ``ref_ncells``, ``log2_fold_change``,
        ``p_value``, ``test_statistic``, ``p_adj``.

    Returns
    -------
    polars.DataFrame
        Long-format table with one row per (target, feature) pair that
        survives all active gene filters (if any). Columns are the defaults
        above unless ``output_columns`` is provided.

    Raises
    ------
    RuntimeError
        If no CUDA device is available.
    NotImplementedError
        If ``reference=ALL_OTHERS`` is combined with
        ``mean_calc='geometric'`` (mixed-mean log fold change is
        unsupported).
    ValueError
        If ``groupby`` is not in ``adata.obs``, or ``reference`` is not a
        value in ``adata.obs[groupby]`` (and not the ``ALL_OTHERS``
        sentinel).
    KeyError
        If ``output_columns`` contains a key not present in the default
        output schema.

    Notes
    -----
    **Filtering is opt-in.** By default (all ``filter_gene_*`` and
    ``keep_genes`` are ``None``) no gene is dropped before the Mann–Whitney
    U test. Pass one or more ``filter_gene_*`` thresholds or a ``keep_genes``
    mask to restrict the gene set.

    **Filter semantics.** All active filters AND-combine: a
    ``(target, gene)`` row survives only if it clears every active filter.
    Within each filter the criterion is target-group OR reference-group
    (either passing is sufficient). ``None`` turns a filter off; a negative
    threshold is an explicit keep-all for that filter; ``0.0`` keeps only
    strictly-positive genes.

    **BH-FDR scope.** Benjamini–Hochberg FDR correction is computed per
    target group over the set of genes that survive all active filters.
    As a result, ``p_adj`` values depend on which genes pass the filter:
    changing the filter changes the multiple-testing universe, so ``p_adj``
    may increase or decrease relative to an unfiltered run. This is expected
    behaviour, not a bug — it is **not** equivalent to formal
    independent-filtering (which guarantees non-inflation).
    """
    # --- streaming vs in-memory dispatch (cheap, archive-free checks first) ---
    _streaming = shard_archive is not None
    if (adata is None) == (shard_archive is None):
        raise ValueError(
            "de(): provide exactly one of adata= or shard_archive= "
            "(got both or neither)."
        )
    if isinstance(reference, ad.AnnData) and not _streaming:
        raise ValueError(
            "de(): an AnnData reference= is only supported with shard_archive= "
            "(streaming mode); for in-memory de() pass a reference group label."
        )
    # --- shared input validation (BOTH paths, before any GPU work or archive
    #     open) so the streaming and in-memory paths reject identical inputs. ---
    if min_feature_filter is not _REMOVED:
        raise ValueError(
            "min_feature_filter was removed. Use filter_gene_min_mean_value= "
            "for a mean gate on adata.X as supplied. If you previously combined "
            "it with cpm_normalize=True (which filtered on CPM), use "
            "filter_gene_min_cpm_cell= instead."
        )
    # Accept the pre-v0.1 sentinel value with a deprecation warning. This MUST
    # run before the streaming dispatch so the ALL_OTHERS guard below also
    # catches the legacy spelling (unsupported with shard_archive=); otherwise
    # reference='all_others' would skip the warning + NotImplementedError and
    # fall through to a misleading "not among the reference labels" error — and,
    # worst case, a real group literally named 'all_others' would be used as a
    # literal reference. reference may be an AnnData (streaming external pool),
    # so only the str spelling is remapped.
    if isinstance(reference, str) and reference == _LEGACY_ALL_OTHERS:
        warnings.warn(
            f"reference={_LEGACY_ALL_OTHERS!r} is deprecated; pass the "
            f"ALL_OTHERS constant (or the string {ALL_OTHERS!r}) instead. "
            "The legacy spelling will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        reference = ALL_OTHERS
    # Fail fast on bad inputs BEFORE any GPU work (cheap, clear errors).
    if mean_calc not in ("arithmetic", "geometric"):
        raise ValueError(
            f"mean_calc must be 'arithmetic' or 'geometric', got {mean_calc!r}."
        )
    if epsilon < 0:
        raise ValueError(f"epsilon must be >= 0, got {epsilon!r}.")
    if cpm_normalize and normalize_target_sum is not None:
        raise ValueError(
            "only one of cpm_normalize / normalize_target_sum may be set "
            "(cpm_normalize=True is equivalent to normalize_target_sum=1e6)."
        )
    if output_columns is not None:
        unknown = [k for k in output_columns if k not in DEFAULT_OUTPUT_COLUMNS]
        if unknown:
            raise KeyError(
                "output_columns keys not present in the de() output schema: "
                f"{unknown}. Valid keys: {list(DEFAULT_OUTPUT_COLUMNS)}"
            )
        dests = list(output_columns.values())
        if len(set(dests)) != len(dests):
            raise ValueError(
                f"output_columns maps multiple keys to the same name: {dests}."
            )

    if _streaming:
        if densify_input:
            raise ValueError(
                "de(): densify_input is not supported with shard_archive= "
                "(X is never fully resident when streaming)."
            )
        if isinstance(reference, str) and reference == ALL_OTHERS:
            raise NotImplementedError(
                f"de(): reference=ALL_OTHERS ({ALL_OTHERS!r}) is not supported with "
                "shard_archive= (1-vs-rest needs global ranks over all cells)."
            )
        if not torch.cuda.is_available():
            raise RuntimeError(
                "gpudge requires a CUDA GPU; torch.cuda.is_available() returned False"
            )
        from ._shard_stream import stream_de
        return stream_de(
            shard_archive,
            groupby=groupby, reference=reference,
            mean_calc=mean_calc, epsilon=epsilon,
            gpu_gene_chunk_size=gpu_gene_chunk_size, oom_recovery=oom_recovery,
            cpm_normalize=cpm_normalize,
            normalize_target_sum=normalize_target_sum,
            output_columns=output_columns,
            filter_gene_min_mean_value=filter_gene_min_mean_value,
            filter_gene_min_total_value=filter_gene_min_total_value,
            filter_gene_min_cpm_cell=filter_gene_min_cpm_cell,
            filter_gene_min_cpm_bulk=filter_gene_min_cpm_bulk,
            keep_genes=keep_genes, device=torch.device("cuda"),
        )
    # ---- in-memory path below ----
    if groupby is None or reference is None:
        raise ValueError("de(): in-memory mode requires groupby= and reference=.")
    if reference == ALL_OTHERS and mean_calc == "geometric":
        raise NotImplementedError(
            f"reference={ALL_OTHERS!r} with mean_calc='geometric' is not "
            "supported (would mix geometric target means with arithmetic "
            f"rest means). Use mean_calc='arithmetic' for {ALL_OTHERS!r}, "
            "or pick a literal reference group for geometric."
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "gpudge requires a CUDA GPU; "
            "torch.cuda.is_available() returned False"
        )
    device = torch.device("cuda")

    from ._filter import (
        _row_scale_needs, validate_keep_genes, x_has_noncount_signal,
    )
    from ._normalize import resolve_target_sum
    keep_genes_arr = (validate_keep_genes(keep_genes, int(adata.n_vars))
                      if keep_genes is not None else None)
    _median_requested = (isinstance(normalize_target_sum, str)
                         and normalize_target_sum == "median")
    # Row sums are needed for the median target, for cpm filters, and for any
    # active normalization scale. Compute them first (cheap), then resolve the
    # target_sum (which consumes them only for the median).
    _cpm_filter_active = (filter_gene_min_cpm_cell is not None
                          or filter_gene_min_cpm_bulk is not None)
    _count_filter_active = (filter_gene_min_mean_value is not None
                            or filter_gene_min_total_value is not None
                            or filter_gene_min_cpm_bulk is not None)
    # We may need row sums BEFORE we know target_sum (the median needs them);
    # and once target_sum is known, scale_main decides the scale tensor. Resolve
    # in two steps: (a) decide if row sums are needed at all, (b) resolve, then
    # (c) decide if the scale tensor is needed.
    # (normalize_target_sum is not None covers the 'median' case; _cpm_filter_active
    #  covers cpm_bulk — both are subsets, so no separate terms.)
    _need_row_sums_for_resolve = (
        bool(cpm_normalize)
        or (normalize_target_sum is not None)
        or _cpm_filter_active)
    _row_sums_np_early = (csr_row_sums(adata.X)
                          if _need_row_sums_for_resolve else None)
    target_sum = resolve_target_sum(
        cpm_normalize=cpm_normalize,
        normalize_target_sum=normalize_target_sum,
        row_sums=_row_sums_np_early)
    _scale_main = target_sum is not None
    # Extra-unit needs (generalized from cpm_normalize -> _scale_main). At most
    # one extra is ever required (the cpm filter unit, when normalization is
    # active, derives from the main mean by a constant — see _normalize).
    _need_unscaled_extra = _count_filter_active and _scale_main
    _need_scaled_extra = (filter_gene_min_cpm_cell is not None) and not _scale_main
    _need_row_sums_np, _need_row_scales_t = _row_scale_needs(
        _scale_main, filter_gene_min_cpm_cell, filter_gene_min_cpm_bulk,
        _median_requested)

    state = ingest(adata, groupby=groupby, reference=reference)
    n_groups = len(state.unique_labels)
    labels_t = torch.from_numpy(state.labels).to(device)

    # Auto-pick gene chunk size from free GPU memory if not provided.
    # Heuristic adopted from pdex's default_gene_chunk_size: working memory is
    # dominated by ref-cell ranking buffers (~24 bytes per ref cell per gene:
    # float32 values + int64 sort indices + float64 ranks + workspace), capped
    # at 16 GB and 20% of free GPU memory. For all_others, ranks cover every
    # cell (1-vs-rest semantics), so use n_cells as the budget basis instead.
    #
    # In ref-mode we also allocate per-chunk GPU accumulators of shape
    # (n_groups, ch) × float64. Count: arithmetic / U / p (= 3) for
    # mean_calc='arithmetic', plus a 4th (target_mean) for 'geometric'.
    # Fold that into bytes_per_gene so the heuristic doesn't blow past
    # free GPU memory on datasets with many groups (e.g., cell line 2 with
    # 4672 target guides adds ~110-150K bytes/gene, comparable to the
    # ranking term for typical ref sizes).
    if gpu_gene_chunk_size is None:
        free, _ = torch.cuda.mem_get_info(device)
        if state.ref_label == ALL_OTHERS:
            budget_n = state.n_cells
        else:
            counts = np.bincount(state.labels, minlength=n_groups)
            budget_n = int(counts[state.ref_label_idx])
        gpu_gene_chunk_size = _auto_gene_chunk_size(
            free_bytes=free,
            budget_n=budget_n,
            n_groups=n_groups,
            mean_calc=mean_calc,
            n_genes=state.n_genes,
            ref_mode=state.ref_label != ALL_OTHERS,
        )

    n_genes = state.n_genes
    if state.ref_label == ALL_OTHERS:
        ref_mean_acc = np.zeros((n_groups, n_genes), dtype=np.float64)
    else:
        ref_mean_acc = np.zeros(n_genes, dtype=np.float64)
    target_mean_acc = np.zeros((n_groups, n_genes), dtype=np.float64)
    # Arithmetic means for all groups — always needed for the filter, regardless
    # of mean_calc.  For mean_calc="arithmetic" this is the same array as
    # target_mean_acc, but we keep them separate for clarity.
    arith_target_acc = np.zeros((n_groups, n_genes), dtype=np.float64)
    arith_ref_acc = np.zeros(n_genes, dtype=np.float64)
    keep_mask_acc = np.zeros((n_groups, n_genes), dtype=bool)
    U_acc = np.zeros((n_groups, n_genes), dtype=np.float64)
    p_acc = np.ones((n_groups, n_genes), dtype=np.float64)

    # Per-group mean in the OTHER unit (vs the X-units arith_*_acc), for the
    # filters whose unit the test path did not produce. Allocated lazily.
    other_target_acc = (
        np.zeros((n_groups, n_genes), dtype=np.float64)
        if (_need_unscaled_extra or _need_scaled_extra) else None
    )
    other_ref_acc = (
        np.zeros(n_genes, dtype=np.float64)
        if (_need_unscaled_extra or _need_scaled_extra) else None
    )
    # Per-group cell counts (needed by total/bulk filters; also used post-loop).
    counts = np.bincount(state.labels, minlength=n_groups)

    # Optionally drop the sparse matrix in favor of a dense one. Must rebind
    # adata.X (not just hold a local reference) so the sparse refcount goes
    # to zero — keeping both representations alive at cell line 2 scale costs 310 GB
    # host and triggers severe paging.
    if densify_input and sp.issparse(adata.X):
        warnings.warn(
            "densify_input=True: replacing adata.X (sparse) with a dense "
            "numpy array in place. The caller's AnnData is mutated; pass "
            "adata.copy() first to preserve sparsity.",
            UserWarning,
            stacklevel=2,
        )
        adata.X = adata.X.toarray()

    # Inline CPM/target-sum scaling. `csr_row_sums` handles both sparse-CSR and
    # dense X uniformly; scipy's `sum(axis=1)` alone is unusably slow on
    # narrow integer dtypes (50× slowdown on uint16 — 292s out of a 392s
    # de() call before the fix).
    # _need_row_sums_np implies _need_row_sums_for_resolve (every disjunct of the
    # former is a subset of the latter), so _row_sums_np_early is always present
    # here — no recompute fallback needed.
    if _need_row_sums_np:
        row_sums_np = _row_sums_np_early
    else:
        row_sums_np = None
    if _need_row_scales_t:
        row_sums_safe = np.where(row_sums_np == 0, 1.0, row_sums_np)
        # Numerator is target_sum when normalization scales the main unit;
        # otherwise 1e6 (the scale tensor then only feeds the cpm filter extra).
        _scale_num = target_sum if _scale_main else 1.0e6
        row_scales = torch.from_numpy(
            (_scale_num / row_sums_safe).astype(np.float32)).to(device)
    else:
        row_scales = None

    # Single non-count warning for cpm_* filters (raw-counts assumption): fire
    # once if a sampled value is fractional/negative OR any row sum < 0.
    if _cpm_filter_active:
        _noncount = x_has_noncount_signal(adata.X)
        if not _noncount and row_sums_np is not None:
            _noncount = bool((row_sums_np < 0).any())
        if _noncount:
            warnings.warn(
                "adata.X does not look like raw counts (non-integer or negative "
                "values); the filter_gene_min_cpm_* filters assume raw counts. "
                "If X is not counts, pass a precomputed keep_genes mask instead.",
                UserWarning, stacklevel=2)

    # Per-group library totals Σ_i L_i (only bulk CPM needs them; row_sums_np is
    # guaranteed present whenever a cpm filter is active, so this is never None
    # when cpm_bulk is requested).
    if filter_gene_min_cpm_bulk is not None and row_sums_np is not None:
        group_libtot = np.bincount(
            state.labels, weights=row_sums_np, minlength=n_groups).astype(np.float64)
    else:
        group_libtot = None

    if state.ref_label != ALL_OTHERS:
        X_host = adata.X
        # Pre-compute per-group row indices once (avoids repeated
        # np.flatnonzero inside the gene-chunk loop). One stable argsort +
        # boundary split instead of n_groups full-array scans; stable sort keeps
        # each group's rows in ascending order (== flatnonzero). (ultrareview perf.)
        _order = np.argsort(state.labels, kind="stable")
        _bounds = np.searchsorted(state.labels[_order], np.arange(n_groups + 1))
        group_to_rows = [_order[_bounds[g]:_bounds[g + 1]]
                         for g in range(n_groups)]
        ref_rows = group_to_rows[state.ref_label_idx]
        n_ref = len(ref_rows)
        if row_scales is not None:
            ref_rows_t = torch.from_numpy(
                ref_rows.astype(np.int64)).to(device)
            group_rows_t = [
                torch.from_numpy(g_rows.astype(np.int64)).to(device)
                for g_rows in group_to_rows
            ]
        else:
            ref_rows_t = None
            group_rows_t = None

    else:
        X_host = adata.X

    if state.ref_label == ALL_OTHERS:
        # all_others path: 1-vs-rest, needs global ranks across all cells.
        # Wrapped in the OOM-recovery driver (gpudge#27): the chunk body slices
        # [start:stop] to the GPU itself, so a downshifted retry re-slices a
        # narrower block. Per-gene accumulators are written by absolute index,
        # so re-processing a sub-range overwrites those genes identically.
        #
        # 1-vs-rest constants that depend only on the group labels (not the gene
        # chunk) are computed ONCE here instead of inside every chunk call.
        # (ultrareview perf.)
        counts_t = torch.zeros(n_groups, dtype=torch.float64, device=device)
        counts_t.index_add_(0, labels_t.long(),
                            torch.ones(state.n_cells, dtype=torch.float64,
                                       device=device))
        m_t = counts_t
        N_t = torch.tensor(float(state.n_cells), dtype=torch.float64,
                           device=device)
        n_rest = N_t - m_t
        u_offset = (m_t * (m_t + 1) / 2)[:, None]
        mn = (m_t * n_rest)[:, None]
        mu = mn / 2
        base_var = mn * (N_t + 1) / 12
        counts_np = np.bincount(state.labels, minlength=n_groups)
        _rest_count = state.n_cells - counts_np
        rest_count_safe = np.where(_rest_count == 0, 1, _rest_count)

        def _process_gene_chunk_ao(start, stop):
            ch_genes = stop - start
            if sp.issparse(adata.X):
                block = (adata.X[:, start:stop].tocsc()
                         .toarray().astype(np.float32, copy=False))
            else:
                block = np.ascontiguousarray(adata.X[:, start:stop],
                                             dtype=np.float32)
            X_chunk = torch.from_numpy(block).to(device, non_blocking=True)  # UNSCALED

            # Other-unit per-group mean (no division; from the unscaled block):
            if other_target_acc is not None:
                if _scale_main:
                    other_unit = group_means(X_chunk, labels_t, n_groups,
                                             kind="arithmetic")
                else:
                    other_unit = group_means(X_chunk * row_scales.unsqueeze(1),
                                             labels_t, n_groups, kind="arithmetic")
                other_target_acc[:, start:stop] = other_unit.cpu().numpy()
                del other_unit

            # Test/reported path: scale iff target_sum is active.
            if _scale_main:
                X_chunk = X_chunk * row_scales.unsqueeze(1)

            arith = group_means(X_chunk, labels_t, n_groups, kind="arithmetic")
            arith_np = arith.cpu().numpy()

            # all_others only supports arithmetic mean_calc (enforced above)
            out_means = arith_np

            ranks, tie_term = _rank_with_ties(X_chunk)
            rank_sums = torch.zeros((n_groups, ch_genes), dtype=torch.float64,
                                    device=device)
            rank_sums.index_add_(0, labels_t.long(), ranks)
            U = rank_sums - u_offset
            tie_corr = mn * tie_term[None, :] / (12 * N_t * (N_t - 1))
            var = (base_var - tie_corr).clamp_min(
                torch.finfo(torch.float64).tiny)
            numerator = (U - mu).abs() - 0.5
            numerator = numerator.clamp_min(0.0)
            z = numerator / var.sqrt()
            p = torch.erfc(z / math.sqrt(2.0))
            U_chunk = U.cpu().numpy()
            p_chunk = p.cpu().numpy()

            # rest-mean: the global gene sum = Σ_g mean_g·count_g (reuse the
            # per-group sums) instead of recasting the full (n_cells, ch) chunk
            # to f64 — drops a multi-GB transient. (ultrareview perf.)
            sum_per_group = arith_np * counts_np[:, None]
            sum_all = sum_per_group.sum(axis=0)
            rest_sum = sum_all[None, :] - sum_per_group
            ref_chunk = rest_sum / rest_count_safe[:, None]

            target_mean_acc[:, start:stop] = out_means
            ref_mean_acc[:, start:stop] = ref_chunk
            new_keep = _all_others_chunk_keep(
                start, stop, stop - start,
                arith_np, other_target_acc, counts_np, rest_count_safe,
                group_libtot, target_sum,
                filter_gene_min_mean_value, filter_gene_min_total_value,
                filter_gene_min_cpm_cell, filter_gene_min_cpm_bulk,
                keep_genes_arr,
            )
            keep_mask_acc[:, start:stop] = new_keep
            U_acc[:, start:stop] = U_chunk
            p_acc[:, start:stop] = p_chunk

        run_gene_chunks_with_recovery(
            n_genes, gpu_gene_chunk_size, _process_gene_chunk_ao,
            oom_recovery=oom_recovery)
    else:
        # ref-mode path: per gene chunk, densify ref ONCE then loop per group.
        # GPU memory per chunk = n_ref × chunk + m × chunk per group (not
        # n_cells × chunk), which scales to cell line 2-size datasets.
        #
        # Both ref and per-group means are computed on GPU. Each slice is
        # already uploaded for the MWU sort/searchsorted, so the mean comes
        # essentially free — no duplicate host pass.

        # Pre-allocate one pinned host buffer for the per-group slice that
        # gets reused across all (chunk × group) iterations. torch's
        # implicit .to(device) on non-pinned memory does a CPU pin+copy
        # internally — nsys 2026-05-25 attributed ~12 s of cudaMemcpyAsync
        # API wall to that step on cell line 2 (separate from the 11.7 s of
        # actual H2D transfer). With the buffer pre-pinned, .to(device,
        # non_blocking=True) skips that step.
        #
        # Sizing: max_group_rows × gpu_gene_chunk_size × 4 bytes. On cell line 2
        # that's ~7000 × ~6000 × 4 ≈ 170 MB. Pinned memory is reserved
        # for the whole de() call; freed when the function returns.
        #
        # Safe to reuse across iterations: each per-group iteration ends
        # with a .cpu() call (arith mean readout) which implicitly syncs
        # the CUDA stream, so the previous H2D is guaranteed complete
        # before the next CSR slice writes into the buffer. Double-
        # buffering for true overlap is T5 phase 2.
        max_group_rows = max(
            (len(group_to_rows[g])
             for g in range(n_groups) if g != state.ref_label_idx),
            default=0,
        )
        # Double-buffered pinned host arena: two buffers alternate per
        # target-group iteration so that iteration N+1's CSR slice and
        # async H2D can run while iteration N's GPU compute is still in
        # flight. Reuse safety is guaranteed by per-buffer CUDA events:
        # before reusing buf[k], synchronize() on event[k] waits for the
        # previous H2D queued from that buffer to complete.
        if HAS_NUMBA and sp.issparse(X_host) and X_host.format == "csr" and max_group_rows > 0:
            group_host_bufs = [
                torch.empty(max_group_rows, gpu_gene_chunk_size,
                            dtype=torch.float32, pin_memory=True),
                torch.empty(max_group_rows, gpu_gene_chunk_size,
                            dtype=torch.float32, pin_memory=True),
            ]
            group_host_bufs_np = [b.numpy() for b in group_host_bufs]
            group_h2d_events = [torch.cuda.Event(), torch.cuda.Event()]
        else:
            # Non-CSR / no-numba / no target groups: per-iteration allocation
            # (the legacy path). out= is only honoured on the numba+CSR path.
            group_host_bufs = None
            group_host_bufs_np = None
            group_h2d_events = None

        def _process_gene_chunk(start, stop):
            torch.cuda.nvtx.range_push(f"chunk_{start}")

            # --- Ref: host slice → GPU; means on GPU ---
            torch.cuda.nvtx.range_push("ref_slice")
            ref_dense = _row_col_slice_np(X_host, ref_rows, start, stop)
            torch.cuda.nvtx.range_pop()
            torch.cuda.nvtx.range_push("ref_upload")
            ref_t = torch.from_numpy(ref_dense).to(device)         # (n_ref, chunk) UNSCALED
            del ref_dense
            torch.cuda.nvtx.range_pop()

            torch.cuda.nvtx.range_push("ref_means")
            # Other-unit ref mean (no division; from the UNSCALED tensor). Only
            # materialize the unscaled f64 when a filter needs the other unit, and
            # free it BEFORE the X-units scale/cast — so the common
            # cpm_normalize=True/no-count-filter path keeps its single f64 cast
            # (no memory regression).
            if other_ref_acc is not None:
                ref_f64_un = ref_t.to(torch.float64)
                if _scale_main:
                    other_ref_acc[start:stop] = ref_f64_un.mean(dim=0).cpu().numpy()
                else:
                    rs_ref = row_scales[ref_rows_t].unsqueeze(1).to(torch.float64)
                    other_ref_acc[start:stop] = (
                        (ref_f64_un * rs_ref).mean(dim=0).cpu().numpy())
                del ref_f64_un
            # Test/reported path: scale IN PLACE iff target_sum is active, then cast ONCE.
            if _scale_main:
                ref_t.mul_(row_scales[ref_rows_t].unsqueeze(1))
            ref_f64 = ref_t.to(torch.float64)                      # X-units
            arith_ref = ref_f64.mean(dim=0).cpu().numpy()
            arith_ref_acc[start:stop] = arith_ref
            if mean_calc == "geometric":
                ref_mean_acc[start:stop] = (
                    torch.expm1(torch.log1p(ref_f64).mean(dim=0)).cpu().numpy())
            else:
                ref_mean_acc[start:stop] = arith_ref
            del ref_f64
            torch.cuda.nvtx.range_pop()

            torch.cuda.nvtx.range_push("ref_sort")
            sorted_ref = torch.sort(
                ref_t.T.contiguous(), dim=1).values                # (chunk, n_ref)
            torch.cuda.nvtx.range_pop()
            torch.cuda.nvtx.range_push("ref_tie_term")
            ref_tie_term = _tie_term_per_gene(sorted_ref)          # (chunk,)
            torch.cuda.nvtx.range_pop()
            del ref_t

            # --- Per non-ref group: host slice → GPU; means + MWU on GPU ---
            torch.cuda.nvtx.range_push("group_loop")
            ch = stop - start  # genes in this chunk (last chunk may be < gpu_gene_chunk_size)

            # Per-chunk GPU accumulators: write each target group's
            # results into a slot here, then one batched .cpu() per
            # accumulator after the loop instead of one per-group.
            # torch.zeros (not torch.empty) so the ref-label row and
            # any empty target groups read as 0 after the .cpu() copy
            # (uninitialised would leak arbitrary GPU bytes).
            arith_target_chunk = torch.zeros(
                (n_groups, ch), dtype=torch.float64, device=device)
            if other_target_acc is not None:
                other_target_chunk = torch.zeros(
                    (n_groups, ch), dtype=torch.float64, device=device)
            U_chunk = torch.zeros(
                (n_groups, ch), dtype=torch.float64, device=device)
            p_chunk = torch.zeros(
                (n_groups, ch), dtype=torch.float64, device=device)
            if mean_calc == "geometric":
                target_mean_chunk = torch.zeros(
                    (n_groups, ch), dtype=torch.float64, device=device)

            # iter_idx is the counter of *active* iterations (skipping ref +
            # empty groups). It selects which double-buffer slot to use.
            iter_idx = 0
            for g in range(n_groups):
                if g == state.ref_label_idx:
                    continue
                g_rows = group_to_rows[g]
                if len(g_rows) == 0:
                    continue
                m = len(g_rows)

                if group_host_bufs_np is not None:
                    # Double-buffer: alternate between two pinned host
                    # buffers. Before reusing buf[buf_idx], wait for the
                    # PREVIOUS H2D from that same buffer to finish (event
                    # recorded ≥1 iteration ago). The CPU host write +
                    # async H2D for this iteration can then overlap with
                    # the GPU work for the previous iteration.
                    #
                    # For the first 2 iterations, the events haven't been
                    # recorded yet; synchronize() on a never-recorded
                    # event is a no-op (PyTorch documents this).
                    buf_idx = iter_idx % 2
                    group_h2d_events[buf_idx].synchronize()
                    # Pack the slice into a CONTIGUOUS (m, ch) view of the pinned
                    # buffer's flat prefix. The plain out[:m, :ch] view is
                    # non-contiguous whenever ch < gpu_gene_chunk_size (the
                    # trailing gene-chunk, or after an OOM downshift), which makes
                    # .to(device, non_blocking=True) fall back to a synchronous
                    # staged copy; a contiguous pinned view keeps the async H2D.
                    # m*ch <= max_group_rows*gpu_gene_chunk_size, so it always
                    # fits the buffer, and the per-buffer event sync above still
                    # guards reuse (same underlying storage). (ultrareview #46)
                    packed = group_host_bufs_np[buf_idx].reshape(-1)[:m * ch].reshape(m, ch)
                    group_dense = _row_col_slice_np(
                        X_host, g_rows, start, stop, out=packed)
                    group_t = torch.from_numpy(group_dense).to(
                        device, non_blocking=True)
                    # Record AFTER queuing the H2D so a future
                    # synchronize() waits for this H2D to complete.
                    group_h2d_events[buf_idx].record()
                else:
                    group_dense = _row_col_slice_np(X_host, g_rows, start, stop)
                    group_t = torch.from_numpy(group_dense).to(device)
                    del group_dense
                _scales = (row_scales[group_rows_t[g]]
                           if row_scales is not None else None)
                arith_t, reported_t, other_t, u1, p = group_chunk_stats(
                    group_t, sorted_ref, ref_tie_term, n_ref,
                    mean_calc=mean_calc, scale_main=_scale_main,
                    group_scales=_scales, want_other=other_target_acc is not None)
                arith_target_chunk[g] = arith_t
                if other_target_acc is not None:
                    other_target_chunk[g] = other_t
                if mean_calc == "geometric":
                    target_mean_chunk[g] = reported_t
                U_chunk[g] = u1
                p_chunk[g] = p
                del group_t, arith_t, reported_t, other_t, u1, p

                iter_idx += 1

            # Batched D2H: collapses ~14k per-chunk .cpu() calls into 3-4.
            arith_target_chunk_np = arith_target_chunk.cpu().numpy()
            arith_target_acc[:, start:stop] = arith_target_chunk_np
            if other_target_acc is not None:
                other_target_acc[:, start:stop] = other_target_chunk.cpu().numpy()
                del other_target_chunk
            U_acc[:, start:stop] = U_chunk.cpu().numpy()
            p_acc[:, start:stop] = p_chunk.cpu().numpy()
            if mean_calc == "geometric":
                target_mean_acc[:, start:stop] = target_mean_chunk.cpu().numpy()
                del target_mean_chunk
            else:
                target_mean_acc[:, start:stop] = arith_target_chunk_np
            del arith_target_chunk, U_chunk, p_chunk, arith_target_chunk_np

            torch.cuda.nvtx.range_pop()  # group_loop

            new_keep = _refmode_chunk_keep(
                start, stop, ch,
                arith_target_acc, arith_ref_acc, other_target_acc, other_ref_acc,
                counts, state.ref_label_idx, group_libtot, target_sum,
                filter_gene_min_mean_value, filter_gene_min_total_value,
                filter_gene_min_cpm_cell, filter_gene_min_cpm_bulk,
                keep_genes_arr,
            )
            keep_mask_acc[:, start:stop] = new_keep

            del sorted_ref, ref_tie_term
            torch.cuda.nvtx.range_pop()  # chunk_<i>

        run_gene_chunks_with_recovery(
            n_genes, gpu_gene_chunk_size, _process_gene_chunk,
            oom_recovery=oom_recovery,
        )

    if state.ref_label == ALL_OTHERS:
        rm_b = ref_mean_acc
    else:
        rm_b = np.broadcast_to(ref_mean_acc, target_mean_acc.shape)
    log2fc = np.log2((target_mean_acc + epsilon) / (rm_b + epsilon))

    if state.ref_label == ALL_OTHERS:
        target_indices = np.arange(n_groups)
        target_labels = state.unique_labels
    else:
        keep_mask_acc[state.ref_label_idx] = False
        target_indices = np.array(
            [i for i in range(n_groups) if i != state.ref_label_idx]
        )
        target_labels = state.unique_labels[target_indices]

    keep_for_targets = keep_mask_acc[target_indices]
    tm = target_mean_acc[target_indices]
    rm = (ref_mean_acc[target_indices] if state.ref_label == ALL_OTHERS
          else ref_mean_acc)
    lfc = log2fc[target_indices]
    U_t2 = U_acc[target_indices]
    p_t2 = p_acc[target_indices]

    # counts hoisted before the chunk loop (reused by the per-gene filters).
    target_ncells = counts[target_indices]
    if state.ref_label == ALL_OTHERS:
        ref_ncells = state.n_cells - target_ncells
    else:
        ref_ncells = int(counts[state.ref_label_idx])

    flat_keep = keep_for_targets.ravel()
    # Pre-filter inside assemble_dataframe: build only the kept rows instead
    # of materialising n_target × n_features strings then dropping ~40 %.
    df = assemble_dataframe(
        target=target_labels,
        feature=adata.var_names.to_numpy(),
        target_mean=tm,
        ref_mean=rm,
        target_ncells=target_ncells,
        ref_ncells=ref_ncells,
        log2_fold_change=lfc,
        p_value=p_t2,
        test_statistic=U_t2,
        p_adj=np.zeros_like(p_t2),
        flat_keep=flat_keep,
        output_columns=None,
    )

    if df.height > 0:
        # Per-row group index, derived directly from the 2D keep mask:
        # np.nonzero on a (n_target, n_features) bool returns (row, col)
        # arrays — `row` IS the target_pos for each kept row, which is what
        # bh_per_group wants. This skips the previous
        # `keep_indices = np.flatnonzero(flat_keep); post_filter_g =
        # keep_indices // n_features` formulation that held two large int64
        # arrays simultaneously.
        n_target = len(target_labels)
        post_filter_g = np.nonzero(keep_for_targets)[0]
        g_idx = torch.from_numpy(post_filter_g)
        p_torch = df["p_value"].to_torch()
        adj = bh_per_group(p_torch, g_idx, n_target)
        df = df.with_columns(p_adj=pl.Series(adj.numpy()))

    if output_columns is None:
        return df
    # Keys + duplicate-destination validated at entry. select-THEN-rename (not
    # rename-then-select) so a destination name that shadows an unselected
    # default column can't collide on rename. (Codex review.)
    return df.select(list(output_columns)).rename(output_columns)
