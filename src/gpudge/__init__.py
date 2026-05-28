# src/gpudge/__init__.py
"""gpudge -- lightweight GPU-only Mann-Whitney U DGE."""
from __future__ import annotations

import math
import warnings
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from typing import Literal

import anndata as ad
import numpy as np
import polars as pl
import scipy.sparse as sp
import torch

from ._csr_dense import HAS_NUMBA, csr_row_sums, csr_rows_col_range_to_dense
from ._ingest import ALL_OTHERS, LEGACY_ALL_OTHERS, ingest
from ._means import group_means
from ._mwu import mwu_one_group, _rank_with_ties, _tie_term_per_gene
from ._fdr import bh_per_group
from ._output import assemble_dataframe
from ._stream import iter_gene_chunks

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


def de(
    adata: ad.AnnData,
    *,
    groupby: str,
    reference: str,
    mean_calc: MeanCalc = "arithmetic",
    epsilon: float = 1e-9,
    min_feature_filter: float = 1.0,
    gpu_gene_chunk_size: int | None = None,
    densify_input: bool = False,
    cpm_normalize: bool = False,
    output_columns: dict[str, str] | None = None,
) -> pl.DataFrame:
    """Per-(target, feature) differential expression on GPU.

    GPU-only Mann–Whitney U with per-group BH-FDR and an inline TPM filter.
    All transformations (CPM, log1p, etc.) are the caller's responsibility.

    Parameters
    ----------
    adata : anndata.AnnData
        Single-cell expression matrix. Dense or sparse CSR X is accepted;
        sparse X is streamed to GPU per gene-chunk.
    groupby : str
        Column in ``adata.obs`` that defines the groups (e.g. guide identity).
    reference : str
        Name of the reference group in ``adata.obs[groupby]``, OR the
        ``ALL_OTHERS`` sentinel (``"__all_others__"``) for 1-vs-rest
        comparisons. The pre-v0.1 spelling ``"all_others"`` is still
        accepted with a ``DeprecationWarning`` and will be removed in
        v0.1.0; pass ``ALL_OTHERS`` (or the new string) instead.
    mean_calc : {"arithmetic", "geometric"}, default "arithmetic"
        How ``target_mean`` and ``ref_mean`` (and the log2 fold change derived
        from them) are computed. Independent of the TPM filter, which always
        uses arithmetic means.
    epsilon : float, default 1e-9
        Pseudocount inside ``log2((target_mean + epsilon) / (ref_mean + epsilon))``.
        Default matches ``scanpy.tl.rank_genes_groups``.
    min_feature_filter : float, default 1.0
        CPM threshold for the inline per-(group, gene) filter. A row is kept
        if either the target-group arithmetic mean OR the reference-group
        arithmetic mean exceeds this value. Set to 0.0 to disable.
    gpu_gene_chunk_size : int | None, default None
        Number of genes per GPU pass. ``None`` auto-picks from free device
        memory. Smaller values reduce GPU memory but increase per-chunk
        overhead.
    densify_input : bool, default False
        If True and ``adata.X`` is sparse, **mutate ``adata.X`` in place** to a
        dense numpy array before the chunk loop (i.e. ``adata.X =
        adata.X.toarray()``). The sparse matrix is dropped after this point.
        Trades n_cells × n_genes × 4 bytes of host RAM (~154 GB for cell line 2) for
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
        survives ``min_feature_filter``. Columns are the defaults above
        unless ``output_columns`` is provided.

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
    """
    # Accept the pre-v0.1 sentinel value with a deprecation warning. Lets
    # existing callers keep working while we move toward the new
    # collision-resistant '__all_others__' spelling.
    if reference == LEGACY_ALL_OTHERS:
        warnings.warn(
            f"reference={LEGACY_ALL_OTHERS!r} is deprecated; pass the "
            f"ALL_OTHERS constant (or the string {ALL_OTHERS!r}) instead. "
            "The legacy spelling will be removed in v0.1.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        reference = ALL_OTHERS
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

    state = ingest(adata, groupby=groupby, reference=reference)
    n_groups = len(state.unique_labels)
    labels_t = torch.from_numpy(state.labels).to(device)

    # Auto-pick gene chunk size from free GPU memory if not provided.
    # Heuristic adopted from pdex's default_gene_chunk_size: working memory is
    # dominated by ref-cell ranking buffers (~24 bytes per ref cell per gene:
    # float32 values + int64 sort indices + float64 ranks + workspace), capped
    # at 16 GB and 18% of free GPU memory. For all_others, ranks cover every
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
            accumulator_bytes_per_gene = 0  # accumulators are ref-mode only
        else:
            counts = np.bincount(state.labels, minlength=n_groups)
            budget_n = int(counts[state.ref_label_idx])
            n_accumulators = 4 if mean_calc == "geometric" else 3
            accumulator_bytes_per_gene = 8 * n_accumulators * n_groups
        bytes_per_gene = max(budget_n * 24 + accumulator_bytes_per_gene, 1)
        budget = min(int(free * 0.18), 16 * 1024**3)
        gpu_gene_chunk_size = max(16, budget // bytes_per_gene)
        gpu_gene_chunk_size = min(int(gpu_gene_chunk_size), state.n_genes)
        if gpu_gene_chunk_size >= 64:
            gpu_gene_chunk_size = (gpu_gene_chunk_size // 64) * 64

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

    # Inline CPM scaling. Per-cell scale = 1e6 / row_sum, applied to each
    # chunk on GPU after upload. Matches scanpy.pp.normalize_total(1e6)
    # without mutating adata.X. `csr_row_sums` handles both sparse-CSR and
    # dense X uniformly; scipy's `sum(axis=1)` alone is unusably slow on
    # narrow integer dtypes (50× slowdown on uint16 — 292s out of a 392s
    # de() call before the fix).
    if cpm_normalize:
        row_sums_np = csr_row_sums(adata.X)
        row_sums_np = np.where(row_sums_np == 0, 1.0, row_sums_np)
        row_scales = torch.from_numpy(
            (1.0e6 / row_sums_np).astype(np.float32)
        ).to(device)
    else:
        row_scales = None

    if state.ref_label != ALL_OTHERS:
        X_host = adata.X
        # Pre-compute per-group row indices once (avoids repeated
        # np.flatnonzero inside the gene-chunk loop).
        group_to_rows = [np.flatnonzero(state.labels == g)
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
        # all_others path: unchanged — needs global ranks across all cells.
        for start, X_chunk in iter_gene_chunks(adata.X,
                                               chunk_size=gpu_gene_chunk_size,
                                               device=device):
            ch_genes = X_chunk.shape[1]
            stop = start + ch_genes

            if row_scales is not None:
                X_chunk = X_chunk * row_scales.unsqueeze(1)

            arith = group_means(X_chunk, labels_t, n_groups, kind="arithmetic")
            arith_np = arith.cpu().numpy()

            # all_others only supports arithmetic mean_calc (enforced above)
            out_means = arith_np

            ranks, tie_term = _rank_with_ties(X_chunk)
            rank_sums = torch.zeros((n_groups, ch_genes), dtype=torch.float64,
                                    device=device)
            rank_sums.index_add_(0, labels_t.long(), ranks)
            counts_t = torch.zeros(n_groups, dtype=torch.float64, device=device)
            counts_t.index_add_(0, labels_t.long(),
                                torch.ones(state.n_cells, dtype=torch.float64,
                                           device=device))
            m_t = counts_t
            N_t = torch.tensor(float(state.n_cells), dtype=torch.float64,
                               device=device)
            n_rest = N_t - m_t
            U = rank_sums - (m_t * (m_t + 1) / 2)[:, None]
            mn = (m_t * n_rest)[:, None]
            mu = mn / 2
            base_var = mn * (N_t + 1) / 12
            tie_corr = mn * tie_term[None, :] / (12 * N_t * (N_t - 1))
            var = (base_var - tie_corr).clamp_min(
                torch.finfo(torch.float64).tiny)
            numerator = (U - mu).abs() - 0.5
            numerator = numerator.clamp_min(0.0)
            z = numerator / var.sqrt()
            p = torch.erfc(z / math.sqrt(2.0))
            U_chunk = U.cpu().numpy()
            p_chunk = p.cpu().numpy()

            sum_all = X_chunk.to(torch.float64).sum(dim=0).cpu().numpy()
            counts_np = np.bincount(state.labels, minlength=n_groups)
            sum_per_group = arith_np * counts_np[:, None]
            rest_sum = sum_all[None, :] - sum_per_group
            rest_count = state.n_cells - counts_np
            rest_count_safe = np.where(rest_count == 0, 1, rest_count)
            ref_chunk = rest_sum / rest_count_safe[:, None]

            target_mean_acc[:, start:stop] = out_means
            ref_mean_acc[:, start:stop] = ref_chunk
            keep_chunk = ((arith_np > min_feature_filter)
                          | (ref_chunk > min_feature_filter))
            keep_mask_acc[:, start:stop] = keep_chunk
            U_acc[:, start:stop] = U_chunk
            p_acc[:, start:stop] = p_chunk
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

        for chunk_idx, start in enumerate(range(0, n_genes, gpu_gene_chunk_size)):
            stop = min(start + gpu_gene_chunk_size, n_genes)
            torch.cuda.nvtx.range_push(f"chunk_{chunk_idx}")

            # --- Ref: host slice → GPU; means on GPU ---
            torch.cuda.nvtx.range_push("ref_slice")
            ref_dense = _row_col_slice_np(X_host, ref_rows, start, stop)
            torch.cuda.nvtx.range_pop()
            torch.cuda.nvtx.range_push("ref_upload")
            ref_t = torch.from_numpy(ref_dense).to(device)         # (n_ref, chunk)
            del ref_dense
            if row_scales is not None:
                ref_t.mul_(row_scales[ref_rows_t].unsqueeze(1))
            torch.cuda.nvtx.range_pop()

            torch.cuda.nvtx.range_push("ref_means")
            ref_arith_gpu = ref_t.to(torch.float64).mean(dim=0)
            arith_ref = ref_arith_gpu.cpu().numpy()
            arith_ref_acc[start:stop] = arith_ref
            if mean_calc == "geometric":
                ref_mean_acc[start:stop] = (
                    torch.expm1(
                        torch.log1p(ref_t.to(torch.float64)).mean(dim=0)
                    ).cpu().numpy()
                )
            else:
                ref_mean_acc[start:stop] = arith_ref
            del ref_arith_gpu
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
                    group_dense = _row_col_slice_np(
                        X_host, g_rows, start, stop,
                        out=group_host_bufs_np[buf_idx])
                    group_t = torch.from_numpy(group_dense).to(
                        device, non_blocking=True)
                    # Record AFTER queuing the H2D so a future
                    # synchronize() waits for this H2D to complete.
                    group_h2d_events[buf_idx].record()
                else:
                    group_dense = _row_col_slice_np(X_host, g_rows, start, stop)
                    group_t = torch.from_numpy(group_dense).to(device)
                    del group_dense
                if row_scales is not None:
                    group_t.mul_(row_scales[group_rows_t[g]].unsqueeze(1))

                group_f64 = group_t.to(torch.float64)
                # Write into per-chunk GPU accumulators instead of
                # per-group .cpu().numpy(); this is the T4 change.
                arith_target_chunk[g] = group_f64.mean(dim=0)
                if mean_calc == "geometric":
                    target_mean_chunk[g] = torch.expm1(
                        torch.log1p(group_f64).mean(dim=0))
                del group_f64

                group_T = group_t.T.contiguous()                   # (chunk, m)
                del group_t

                u1, p = mwu_one_group(
                    sorted_ref, ref_tie_term, group_T, n_ref=n_ref)
                U_chunk[g] = u1
                p_chunk[g] = p
                del group_T, u1, p

                iter_idx += 1

            # Batched D2H: collapses ~14k per-chunk .cpu() calls into 3-4.
            arith_target_chunk_np = arith_target_chunk.cpu().numpy()
            arith_target_acc[:, start:stop] = arith_target_chunk_np
            U_acc[:, start:stop] = U_chunk.cpu().numpy()
            p_acc[:, start:stop] = p_chunk.cpu().numpy()
            if mean_calc == "geometric":
                target_mean_acc[:, start:stop] = target_mean_chunk.cpu().numpy()
                del target_mean_chunk
            else:
                target_mean_acc[:, start:stop] = arith_target_chunk_np
            del arith_target_chunk, U_chunk, p_chunk, arith_target_chunk_np

            torch.cuda.nvtx.range_pop()  # group_loop

            # Per-chunk filter: keep if target OR ref arithmetic mean > threshold
            arith_ref_slice = arith_ref_acc[start:stop]
            keep_chunk = (
                (arith_target_acc[:, start:stop] > min_feature_filter)
                | (arith_ref_slice[None, :] > min_feature_filter)
            )
            keep_mask_acc[:, start:stop] = keep_chunk

            del sorted_ref, ref_tie_term
            torch.cuda.nvtx.range_pop()  # chunk_<i>

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

    counts = np.bincount(state.labels, minlength=n_groups)
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
    unknown = [k for k in output_columns if k not in df.columns]
    if unknown:
        raise KeyError(
            f"output_columns keys not present in de() output: {unknown}. "
            f"Valid keys: {sorted(df.columns)}"
        )
    return df.rename(output_columns).select(list(output_columns.values()))
