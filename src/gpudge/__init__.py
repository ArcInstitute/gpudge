# src/gpudge/__init__.py
"""gpudge -- lightweight GPU-only Mann-Whitney U DGE."""
from __future__ import annotations

import math
import os
import warnings
from collections.abc import Callable, Iterable, Sequence
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from typing import Literal

import anndata as ad
import numpy as np
import polars as pl
import scipy.sparse as sp
import torch

from ._cell_source import CellGroup, _check_2d
from ._csr_dense import (
    HAS_NUMBA, csr_row_sums, csr_rows_col_range_to_dense, ensure_csr,
)
from ._gpu_mem import _release_gpu_memory
from ._ingest import ALL_OTHERS, LEGACY_ALL_OTHERS as _LEGACY_ALL_OTHERS, ingest
from ._lfc import lfc_base_names, lfc_column_names, normalize_lfc_spec
from ._means import group_means
from ._mwu import _rank_with_ties, _tie_term_per_gene
from ._fdr import bh_per_group
from ._output import (
    DEFAULT_OUTPUT_COLUMNS,
    assemble_dataframe,
    effect_size_from_u,
)
from ._stream import (
    _auto_gene_chunk_size,
    _pinned_buf_width,
    run_gene_chunks_with_recovery,
)
from ._shard_stream import group_chunk_stats
from ._taustar import (
    normalize_taustar_iters, normalize_taustar_se, normalize_taustar_spec,
    taustar_column_names,
)

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
        "for sparse CSR row slicing (~3x slower on CCL_2-scale inputs). "
        "Install with `pip install gpudge[fast]` (or "
        "`uv sync --extra fast`) to enable the numba kernel.",
        stacklevel=2,
    )

__all__ = ["de", "ALL_OTHERS", "CellGroup", "MeanCalc", "__version__"]

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
    archive: str | os.PathLike | None = None,
    shard_archive: str | os.PathLike | None = None,   # deprecated alias of archive=
    cell_source: Callable[[], Iterable[CellGroup]] | None = None,
    targets: Sequence[str] | np.ndarray | None = None,
    var_names: Sequence[str] | np.ndarray | None = None,
    groupby: str | None = None,
    # cell_source= mode accepts the control pool as a raw matrix. The arms are
    # spelled out rather than widened to `| Any`: an Any arm admits every type
    # to a static checker, which is no annotation at all.
    reference: str | ad.AnnData | np.ndarray | sp.spmatrix | None = None,
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
    lfc_threshold: float | Iterable[float] | None = None,
    lfc_threshold_alt: str | Iterable[str] = ("up", "down"),
    tau_star: float | Iterable[float] | None = None,
    tau_star_iters: int | None = None,
    tau_star_se: bool = False,
    stream_n_workers: int = 16,
    stream_prefetch: int = 2,
    release_gpu_memory: bool = True,
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
    archive : str | os.PathLike
        Path to a shardad archive to stream instead of passing ``adata``.
        Both layouts are accepted and dispatched automatically off the
        archive's own manifest (not its file extension): ``layout='shard'``
        and ``layout='cell'`` (``.csad``). Requires the optional ``streaming``
        extra (``shardad[cell]>=0.7.1``). Exactly one of ``adata`` /
        ``archive`` / ``cell_source`` must be given.
    shard_archive : str | os.PathLike
        Deprecated spelling of ``archive``; accepted with a
        ``DeprecationWarning`` and removed in a future release.
    cell_source : Callable[[], Iterable[CellGroup]]
        A callable returning an iterable of ``CellGroup`` -- one per target
        group -- instead of passing ``adata`` or ``archive``. Use this to feed
        gpudge cells you read yourself: a consumer already streaming an archive
        for its own aggregation would otherwise pay a second, unfusable pass
        over the payload.

        Requires ``targets``, ``var_names``, and a ``reference`` that is the
        control pool itself (an AnnData or a cells x genes matrix) -- a group
        label and ``ALL_OTHERS`` are both rejected, since there is no obs
        column to resolve them against. An AnnData reference must have
        ``var_names`` equal to ``var_names`` element-for-element. ``groupby``
        is unused: the source decides the grouping.

        **May be called more than once**, and must yield the same groups each
        time. Nothing calls it twice today, but the contract reserves it so
        ``normalize_target_sum='median'`` -- which needs a row-sums pre-pass --
        can be added later without breaking one-shot sources. That spelling
        currently raises ``NotImplementedError``; pass the target as a number.

        **Pin ``gpu_gene_chunk_size`` if your groups are large.** The automatic
        gene-chunk sizer models the target working set from the largest group's
        cell count, which a source cannot report without being drained -- so in
        this mode it sizes as if that term were zero and can pick a chunk that
        is too large. With the default ``oom_recovery=True`` that costs a
        downshift; with ``oom_recovery=False`` it fails outright.

        ``densify_input``, ``stream_n_workers`` and ``stream_prefetch`` are
        ignored in this mode.
    targets : Sequence[str]
        The ordered target labels, one per group the source yields. Defines the
        output row order, so it need not match yield order. Must be non-empty
        and free of duplicates. Every label must be yielded exactly once -- a
        missing one raises rather than emitting an all-zero row. Used only with
        ``cell_source``.
    var_names : Sequence[str]
        The gene axis, shared by every yielded group and by ``reference``. The
        gene count is ``len(var_names)``. Used only with ``cell_source``.
    groupby : str
        Column in ``adata.obs`` that defines the groups (e.g. guide identity).
    reference : str | anndata.AnnData | numpy.ndarray | scipy.sparse matrix | None
        Name of the reference group in ``adata.obs[groupby]``, OR the
        ``ALL_OTHERS`` sentinel (``"__all_others__"``) for 1-vs-rest
        comparisons. The pre-v0.1 spelling ``"all_others"`` is still
        accepted with a ``DeprecationWarning`` and will be removed in
        a future release; pass ``ALL_OTHERS`` (or the new string) instead.
        ``reference`` may instead be an ``AnnData`` external control pool: the
        pool is ranked resident-sorted on GPU with **no target-reference
        concatenation** — every group in ``adata.obs[groupby]`` is a target,
        each ranked against the pool. Supported on **both** the in-memory path
        (``adata=``) and the streaming path (``archive=``); results are
        bit-identical between the two on the same cells. ``reference.var_names``
        must equal ``adata`` / the archive gene axis in order. (Streaming only:
        if the archive also designates its own archive reference pool, the external
        pool wins and the archive reference pool is ignored with a ``UserWarning``.)
        With ``cell_source=`` the pool may also be a bare cells x genes matrix,
        and a group name / ``ALL_OTHERS`` is rejected — there is no ``obs``
        column to resolve them against.
    mean_calc : {"arithmetic", "geometric"}, default "arithmetic"
        How ``target_mean`` and ``ref_mean`` (and the log2 fold change derived
        from them) are computed. Independent of any active gene filter, which
        always uses arithmetic means.
    epsilon : float, default 1e-9
        Pseudocount inside ``log2((target_mean + epsilon) / (ref_mean + epsilon))``.
        Default matches ``scanpy.tl.rank_genes_groups``. Must be finite and >= 0.
        With ``epsilon=0``, a gene whose target and reference means are both 0
        yields ``NaN`` log2FC and a target-only (ref-mean 0) gene yields ``+inf``;
        the default ``1e-9`` avoids both. Use a gene filter, or a small positive
        epsilon, if such genes are present and you need finite log2FC.
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
    lfc_threshold : float | Iterable[float] | None, default None
        Effect-size floor in **log2 fold-change units** (τ ≥ 0, ≤ 30). When set,
        `de()` additionally reports one-sided Mann–Whitney p-values against
        the composite nulls ``H0: log2FC <= +τ`` (``up``) and
        ``H0: log2FC >= -τ`` (``down``), testing them at the rank level: the
        target is compared against ``reference * 2**(±τ)``. Any FINITE iterable
        -- list, tuple, ndarray, generator -- evaluates a whole τ grid in ONE
        pass, sharing all the expensive
        τ-independent work. The two-sided ``p_value`` / ``Ueffect`` /
        ``p_adj`` columns are always emitted unchanged. Not supported with
        ``reference=ALL_OTHERS``.

        Output columns use ``tau=<±τ>_{p,Ueffect,padj}``, where the sign of τ
        encodes direction: ``+`` is up (``H0: log2FC <= +τ``) and ``-`` is
        down (``H0: log2FC >= -τ``). Columns are emitted by ascending |τ|,
        with down (``-``) before up (``+``) at each magnitude. Each
        (τ, direction) is its own BH family. Each directional ``Ueffect`` is
        ``2A − 1`` for that direction's shifted comparison.

        **Three caveats you must read:**

        1. The test and ``Ueffect`` are rank-based while
           ``log2_fold_change`` is a mean-ratio, so their signs can flatly
           contradict each other on skewed or heavy-tailed genes — a gene can
           have negative ``Ueffect`` while its ``log2_fold_change`` is strongly
           positive. Do NOT assume the rank direction equals the mean-ratio
           direction.
        2. τ is applied as a multiplicative shift, and ``0 * 2**τ == 0``, so
           the shift moves only the nonzero entries. On high-dropout genes the
           effective floor is weaker than τ suggests; ``lfc_threshold=0.5`` is
           NOT "drop everything with |log2FC| < 0.5".
        3. p-values are a normal approximation (matching
           ``scipy.stats.mannwhitneyu(method='asymptotic')``), not exact, and
           are unreliable for groups with only a handful of cells. They are also
           **not guaranteed monotone in τ**: the tie correction changes
           discontinuously where exact target/reference ties appear or vanish,
           so a directional p can dip slightly as τ grows. The *statistic* is
           monotone; the p-value is only monotone away from tie transitions.

        The base ``Ueffect`` is the signed rank-biserial correlation / Cliff's
        delta ``2A − 1 ∈ [−1, 1]`` for the raw comparison, where its sign is
        the rank direction of change (positive means target ranks above the
        reference). Recover the probability of superiority / AUC as
        ``A = (Ueffect + 1) / 2 ∈ [0, 1]``.

        A two-sided threshold test is deliberately not offered. If you need
        one, combine as ``min(1.0, 2 * min(p_up, p_down))``. Taking plain
        ``min(p_up, p_down)``, or picking the direction from the observed sign
        of ``log2_fold_change``, is anticonservative — it is post-hoc selection
        from the same data.

        **Linear-scale assumption.** τ is applied as a ``2**τ`` multiplicative
        shift, which is correct when ranks are on a linear count/CPM scale (the
        default, or ``cpm_normalize=True``). If you pre-log-transform ``X`` the
        shift should be additive instead, and ``log2_fold_change`` is already
        inconsistent in that case — pairing log-input ranking with this API is
        discouraged.

        **Memory.** A τ grid widens the result by ``3 × len(τ) × len(alt)``
        Float64 columns — that is the dominant cost. GPU memory grows only by
        the per-(τ, direction) result accumulators (the shift is applied to the
        target transiently, so no scaled reference is ever held resident); the
        auto chunk sizer accounts for them and will shrink the gene chunk
        somewhat.
    lfc_threshold_alt : str | Iterable[str], default ("up", "down")
        Which directional tests to compute; values from ``{"up", "down"}``.
        Ignored when ``lfc_threshold is None``. Computing only one direction
        halves the added kernel work. Order does not matter — columns are always
        emitted ``down`` before ``up`` within each |τ|.

    tau_star : float | Iterable[float] | None, default None
        One-sided ``p_dir`` levels in the OPEN interval (0, 1). For each level
        ``q`` and each (target, gene), emits ``tau*_p<q>``: the **signed** log2
        shift at which the gene crosses ``p_dir = q``. The gene's direction is
        fixed once, by ``sign(Ueffect)``; the returned value is that direction's
        crossing, and its own sign is a result, not a restatement of the
        direction -- see **On sign** below.

        ``q = 0.5`` is the **Hodges-Lehmann log2 shift**, the effect size the
        rank test actually estimates, and is depth-invariant. Any smaller ``q``
        is a one-sided confidence bound on it: the largest floor the gene
        survives at that level.

        **``q = 0.05`` is a one-sided 95% bound, i.e. the endpoint of a
        two-sided 90% interval.** The endpoint matching an ``alpha = 0.05``
        two-sided call is ``q = 0.025``, and only there does
        ``tau*_p0.025 > 0`` mean "called at alpha = 0.05 with floor tau = 0".

        **On sign.** At ``q = 0.5`` -- the point estimate -- ``sign(tau*_p0.5)``
        agrees with ``sign(Ueffect)``, so unlike ``log2_fold_change`` the effect
        size cannot contradict the test. Two caveats. (a) The agreement is not
        exact at zero: the continuity correction puts the up and down levels at
        ``mu + 0.5`` and ``mu - 0.5``, giving signed ``tau*`` a half-pair
        discontinuity there, so a gene whose crossing lands on that plateau has
        ``|tau*|`` at the bisection's resolution floor with an arbitrary sign.
        The effect is zero either way, and the gap shrinks as ``m*n`` grows.
        (b) **A bound at ``q < 0.5`` may legitimately have the opposite sign to
        ``Ueffect``** -- that is what a confidence bound means. An up gene that
        is not significant at ``q`` has ``tau*_p<q> < 0``. Compare a bound to
        the ``q = 0.5`` estimate, never to zero-crossing alone.

        ``+/-inf`` is a RESULT, not an error: the confidence-bound endpoint is
        unbounded, which happens on zero-heavy genes and on genes absent from
        one side. NaN means undefined -- an empty group, an empty reference, or
        a gene that is zero on both sides. Not supported with
        ``reference=ALL_OTHERS``.

        **Input domain.** Like the log2 ratio it estimates, ``tau*`` is defined
        for **finite, non-negative** ``X`` -- the expression counts gpudge is
        built for. Negative values invert the monotonicity the bisection relies
        on and infinities do not move under a finite scaling, so neither is
        rejected but neither yields a meaningful ``tau*``. The base ``p_value``
        and ``Ueffect`` columns, being pure rank statistics, are unaffected.

        **On raw counts ``tau*`` measures almost nothing -- normalize first.**
        The modal pairwise log2 ratio is exactly 0, because ``T_i == R_j`` is
        common for small integers, so most genes land on a tie atom at zero
        rather than on a resolvable shift. Measured on a 1.27M-cell production
        archive: 87.9% of finite ``tau*`` lie within 1e-5 of zero and only
        6.0% of rows reach ``|tau*| >= 0.01``. ``normalize_target_sum``
        collapses that plateau to 0.02% and lifts usable rows to 46.8%. Use
        library-size normalization before reading ``tau*`` as an effect size.

        See ``tau_star_se`` for what the same atom does to the interval width,
        and for a standard error on the ``q = 0.5`` point estimate.

    tau_star_iters : int | None, default None
        Bisection steps per level (default 20), which puts the residual well
        under 1e-4 in log2 units. Used only when ``tau_star`` is set, but
        validated regardless -- ``de(tau_star=None, tau_star_iters=0)`` raises.

    tau_star_se : bool, default False
        Emit a standard error for the ``tau*`` point estimate. Requires
        ``tau_star``; ``tau_star_se=True`` with ``tau_star=None`` raises. Adds
        three float64 columns after the ``tau*_p<level>`` block:

        * ``tau*_lo_p0.025``, ``tau*_hi_p0.025`` -- the endpoints of the
          nominal 95% two-sided interval for the log2 shift. Each inverts a
          **one-sided** test at ``p_dir = 0.025``, so the pair has nominal
          ``1 - 2*0.025`` coverage. Nominal, not guaranteed: the kernel inverts
          a normal approximation, so finite-sample coverage on discrete counts
          is not established -- the same standing every asymptotic
          Mann-Whitney interval has.
        * ``tau*_se`` -- ``(hi - lo) / (2 * z)`` where
          ``z = Phi^-1(0.975) = 1.959964...``, in log2 units.

        **``0.5`` is added to the level set if absent**, so ``tau*_p0.5`` --
        the estimate the SE belongs to -- is always in the default output. The
        emitted level set can therefore be larger than the one requested.
        (``output_columns`` can still project it away, like any column.)

        A rank-inversion estimator has no finite-sample, tuning-free variance
        obtainable from the quantities this kernel computes, so the SE is
        backed out of an interval width. The level it is measured at is fixed
        internally and deliberately not a parameter: at ``q = 0.5`` the
        quantile is 0 and the endpoint gap is the ``+/-0.5`` continuity
        correction -- a discretization artifact, not a confidence width.

        **The interval is often strongly asymmetric**, and ``tau*_se``
        collapses it to one number by construction -- it is a normal-equivalent
        interval-width SE, not in general the sampling standard error of the
        Hodges-Lehmann estimator. Where ``hi - tau*_p0.5`` and
        ``tau*_p0.5 - lo`` differ materially, read the endpoints.

        **``tau*_se`` is ``+inf`` whenever either endpoint is unbounded, which
        is common.** Unboundedness comes from zeros, so small or zero-heavy
        target groups report ``+inf`` routinely -- consumers doing
        inverse-variance weighting should treat it as zero weight rather than
        dropping the row. The endpoints stay informative when the SE does not.
        A gene whose rank statistic no finite shift can move -- an all-zero
        target, or an all-zero reference -- reports one of three unbounded
        readings: ``(+inf, +inf)`` (unbounded above), ``(-inf, +inf)``
        (unidentified) or ``(-inf, -inf)`` (unbounded below), according to
        where its statistic falls relative to the two test levels. Which one
        is **not** determined by which side the zeros are on: ``T=[0]`` vs
        ``R=[1]`` is unidentified, while ``T=[0,0,0]`` vs ``R=[1,2,3,5]`` is
        unbounded below. NaN means undefined, matching ``tau*``: an empty
        group, an empty reference, or a gene that is zero on both sides. The
        endpoint columns are exactly the two branches of ``tau*_p0.025``
        reported separately, so those never disagree. That relation holds at
        ``q = 0.025`` only -- every other level is a different root.

        **Domain caveat, sharper than ``tau_star``'s.** The raw-counts tie atom
        quantified under ``tau_star`` does not merely flatten the point
        estimate here: that atom, not sampling variability, then dominates
        ``hi - lo``. Use library-size normalization
        (``normalize_target_sum``) before reading either ``tau*`` or
        ``tau*_se`` as an effect size.

        Not the SE of ``log2_fold_change`` -- different estimand -- and not
        comparable to a negative-binomial GLM's ``lfcSE`` beyond the shape of
        the pair. Not supported with ``reference=ALL_OTHERS`` (inherited from
        ``tau_star``).

        Costs two extra bisections for the endpoints -- ~150 s on top of a
        ~270 s three-level run at 1.27M cells -- plus one more if ``0.5`` has
        to be auto-inserted.

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
        size, with one bounded exception: the tie-correction sum is accumulated
        exactly in int64 only while the relevant axis fits the cube bound of
        2 097 151 cells, and there are **two** ways to exceed it — the reference
        tie axis itself (``n_ref``, or all cells under ``ALL_OTHERS``), or the
        POOLED sample ``n_ref + m`` even when the reference alone fits. Past
        either, the sum falls back to a shape-sensitive float64 reduction and
        ``p_value`` / ``p_adj`` may differ between chunk sizes (observed at
        1-2 ULP; not a guaranteed bound). **gpudge does not warn when this
        happens** -- the condition is data-dependent and cheap to reason about
        from the sizes above, and the check cannot be made both per-invocation
        and free in the per-(group, chunk) inner loop. If you need
        bit-reproducible p-values at that scale, pin a ``gpu_gene_chunk_size``
        that fits **and** pass ``oom_recovery=False``: that combination is what
        fixes the reduction shape, which is what makes the float64 result
        reproducible even though it is not exact. A pinned chunk alone is NOT
        enough -- under the default ``oom_recovery=True`` an OOM halves even an
        explicit chunk, and it can strike after earlier chunks have already
        succeeded, so one run can mix reduction shapes.
    densify_input : bool, default False
        If True and ``adata.X`` is sparse, **mutate ``adata.X`` in place** to a
        dense numpy array before the chunk loop (i.e. ``adata.X =
        adata.X.toarray()``). The sparse matrix is dropped after this point.
        Trades n_cells × n_genes × 4 bytes of host RAM (~153 GB steady-state
        for CCL_2; up to ~310 GB peak during the in-place sparse→dense swap) for
        ~30-40% faster per-chunk per-group slicing (numpy fancy indexing
        instead of repeated CSR slicing). The caller must be OK with the
        sparse → dense replacement; pass ``adata.copy()`` first to preserve
        the original. Note: just setting ``adata.X = ...`` without dropping
        the sparse first (e.g. holding both in separate variables) makes this
        slower not faster, because both representations coexist; we do the
        replacement inside de() so the rebind drops the sparse refcount to 0.
        Not supported together with an AnnData ``reference=`` (raises
        ``ValueError``); the external reference is ranked resident on the GPU,
        not densified in place.
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
        ``normalize_total(target_sum=None)`` as its dense/Dask branch implements
        it. Caveat: scanpy's CSR branch medians over ALL cells, empty ones
        included, so the two *can* differ when zero-total cells are present,
        and gpudge's sparse paths use CSR. That split affects scanpy
        1.11.2–1.12.3 and the 1.13.0a1 prerelease; versions before 1.11.2
        never had it. scverse/scanpy#4256 fixes it by adopting this same
        positive-cell median on the CSR branch — merged upstream and
        backported to the 1.12.x branch for 1.12.4, but in no release as of
        1.12.3. The choice of
        target is **not** neutral: in exact arithmetic it is a common scale, but
        (a) with ``epsilon > 0`` arithmetic ``log2_fold_change`` is
        target-dependent — generally negligible when both means greatly exceed
        ``epsilon``, but material for zero or near-zero means, and absent at
        ``epsilon=0``, (b) ``mean_calc="geometric"`` uses
        ``expm1(mean(log1p(x)))``, which is not scale-homogeneous and so is
        target-dependent regardless of ``epsilon``, and (c) row scales are
        applied in float32 and equal values are treated as ties, so a different
        target can create or destroy ties and move ``Ueffect``, ``p_value`` and
        ``p_adj`` — the normalization-specific case of the float32 tie
        behaviour under **Numerical precision** in Notes, which applies
        whether or not normalization is on. Non-zero reported means are in
        target-dependent units;
        ``filter_gene_min_cpm_cell`` cancels the target mathematically but can
        still flip at a float32 boundary. ``cpm_normalize=True`` is exactly
        ``normalize_target_sum=1e6``; **only one** of the two may be set (else
        ``ValueError``). Note the naming nuance: scanpy spells "use the median"
        as ``target_sum=None``, whereas here ``None`` means "off" and the median
        is requested with the explicit string ``"median"``.
    output_columns : dict[str, str] | None, default None
        If provided, output only these columns (the dict keys), renamed to
        the dict values. Keys must be from the default output column set:
        ``target``, ``feature``, ``target_mean``, ``ref_mean``,
        ``target_ncells``, ``ref_ncells``, ``log2_fold_change``,
        ``p_value``, ``Ueffect``, ``p_adj``. When ``lfc_threshold`` is set,
        directional ``tau=<±τ>_{p,Ueffect,padj}`` names are also valid
        keys; when ``tau_star`` is set, so are the ``tau*_p<level>`` names;
        when ``tau_star_se`` is set, so are ``tau*_lo_p0.025``,
        ``tau*_hi_p0.025`` and ``tau*_se``.
    stream_n_workers : int, default 16
        Streaming only (``archive=``); ignored on the in-memory path. Meaning
        depends on the archive's layout:

        * ``layout='shard'`` — CPU decode-ahead workers for
          ``iter_group_shards``. Peak host RAM scales with this (~14 GB per
          worker on CCL_2); it is the speed-vs-host-RAM dial.
        * ``layout='cell'`` — decode threads for shardad's Rust cell gather.
          Costs no extra host RAM; 16 is the measured sweet spot.

        Unused on the shard-layout GPU device-decode path, which requires
        ``stream_prefetch=0``.
    stream_prefetch : int, default 2
        Streaming only (``archive=``). Decode-ahead queue depth on
        ``layout='shard'`` (``0`` disables prefetch, the low-host-RAM
        fallback). **No effect on ``layout='cell'``**, which gathers
        synchronously.
    release_gpu_memory : bool, default True
        Return gpudge's GPU memory caches (torch's caching allocator and, if
        importable, cupy's pools) to the CUDA driver on exit, so a same-process
        caller can allocate GPU memory after de() (otherwise the pools hold ~all
        of VRAM and the caller's next cudaMalloc / cuBLAS op can OOM). Pass
        ``False`` to keep the caches resident (avoids re-allocating the resident
        reference when calling de() repeatedly in a tight loop). ``gpudge_arc#76``.

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
        output schema or, when ``lfc_threshold`` is set, the directional
        ``tau=<±τ>_{p,Ueffect,padj}`` schema.

    Notes
    -----
    **Bring-your-own cell source.** ``cell_source`` yields ``CellGroup``:

        CellGroup(label, X, rows=None)

    ``label`` is the target's label (not an index -- gpudge maps it through
    ``targets``, so yield order is free). ``rows=None`` means all rows of
    ``X``; passing indices lets you yield one shared matrix per group without
    copying, and they must be a 1-D integer array, in range, without
    duplicates (a bool mask or float array is rejected rather than silently
    cast to different cells); a *dense* ``X`` must additionally be
    C-contiguous and aligned when ``rows`` re-orders or subsets it and library
    sizes are being computed, since numpy reduces a Fortran-ordered, strided
    or unaligned slice in a different order. There is deliberately no way to
    supply library sizes: gpudge computes them with its own kernel, which is
    what keeps CPM scaling byte-identical to every other gpudge path --
    guaranteed for a target matrix that is CSR, or C-contiguous and aligned
    dense, with standard NumPy/SciPy semantics (gpudge sums what it is handed,
    so a group gathered out of a
    Fortran-ordered parent will differ in the last bits from summing that
    parent and indexing).

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

    **Numerical precision: expression is staged in float32.** Every densify
    path — the numba CSR kernel, the scipy fallback and the dense slice —
    emits float32 regardless of ``adata.X``'s dtype, so a float64 input is
    **downcast**, and float32 is what is held resident on the GPU. Reductions
    are then accumulated in float64, so the quantization loss is entirely at
    that staging step; it is invisible from the returned schema, in which
    every statistic is ``Float64``. Two consequences, neither a correctness
    bug in the statistics but both worth knowing before calibrating a
    tolerance (``gpudge_arc#115``):

    - ``log2_fold_change`` carries an **absolute** error floor on the order
      of one float32 ulp, *independent of the fold change's magnitude* — a
      float32-relative error in a group mean becomes an absolute error in
      its log, ``d(log2 FC) = (1/ln 2) · (dm/m)``. Measured against a
      construction whose true answer is zero to float64 round-off: max
      ``9.8e-08``, median ``3.0e-08``, unbiased and uncorrelated with
      ``|log2_fold_change|``. The same contrast on a float64 CPU backend
      (scanpy) is ``3e-15``, so a tolerance calibrated there will not hold
      here: two gpudge runs cannot be asserted equal below ~``1e-7``
      absolute, at any fold change.
    - ``Ueffect``, ``p_value`` and ``p_adj`` inherit float32 **tie**
      behaviour, because the Mann–Whitney sort ranks the float32 tensor and
      equal values are treated as ties. This channel is **not** bounded by
      the argument above: quantization can create or destroy a tie, so a
      gene can cross a significance threshold for purely numerical reasons.
      A downstream top-K selection that filters on ``p_adj`` before ranking
      is therefore not stable to float32 granularity, however large the
      effect-size gaps are.
    """
    # --- archive= / shard_archive= alias, before anything reads either ---
    if shard_archive is not None:
        if archive is not None:
            raise ValueError(
                "de(): pass only archive=; shard_archive= is the deprecated "
                "spelling of the same parameter (got both)."
            )
        warnings.warn(
            "de(shard_archive=...) is deprecated; use de(archive=...). The new "
            "spelling takes both shard-layout and cell-layout archives. The old "
            "one will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        archive = shard_archive
    # --- streaming vs in-memory dispatch (cheap, archive-free checks first) ---
    _streaming = archive is not None
    _byo = cell_source is not None
    _modes_set = [name for name, on in (("adata=", adata is not None),
                                        ("archive=", archive is not None),
                                        ("cell_source=", _byo)) if on]
    if len(_modes_set) != 1:
        raise ValueError(
            "de(): provide exactly one of adata= or archive= or cell_source= "
            f"(got {', '.join(_modes_set) if _modes_set else 'none'})."
        )
    # EARLY, and deliberately so: de()'s ALL_OTHERS feature guards below fire
    # on `isinstance(reference, str) and reference == ALL_OTHERS`, so without
    # this a BYO caller passing ALL_OTHERS with tau_star=/lfc_threshold= would
    # get "tau_star is not supported with ALL_OTHERS" -- true, but beside the
    # point. Rejecting the reference TYPE here makes `reference` provably
    # non-str in BYO mode, so those guards can never fire and neither is
    # touched.
    if _byo and (reference is None or isinstance(reference, str)):
        raise ValueError(
            "de(cell_source=...) requires reference= to be the control pool "
            "itself -- an AnnData or a cells x genes matrix. A group label "
            "(and the ALL_OTHERS sentinel) has no obs column to resolve "
            f"against here; got {type(reference).__name__}."
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
    # catches the legacy spelling (unsupported with archive=); otherwise
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
    if not math.isfinite(epsilon) or epsilon < 0:
        raise ValueError(f"epsilon must be a finite value >= 0, got {epsilon!r}.")
    lfc_combos = normalize_lfc_spec(lfc_threshold, lfc_threshold_alt)
    taustar_se = normalize_taustar_se(tau_star_se)
    taustar_levels = normalize_taustar_spec(tau_star, taustar_se=taustar_se)
    # Validated unconditionally: only USED when tau_star is set, but a nonsense
    # value is a caller error either way. The docstring says "used only when",
    # NOT "ignored" -- de(tau_star=None, tau_star_iters=0) does raise.
    taustar_iters = normalize_taustar_iters(tau_star_iters)
    if taustar_levels is not None and isinstance(reference, str) \
            and reference == ALL_OTHERS:
        # NotImplementedError, matching the lfc_threshold guard --
        # NOT ValueError. The constraint is "same error shape as
        # lfc_threshold", and that shape is NotImplementedError.
        raise NotImplementedError(
            f"tau_star is not supported with reference=ALL_OTHERS "
            f"({ALL_OTHERS!r}). The one-vs-rest path ranks all cells jointly, "
            f"so there is no per-reference distribution to shift.")
    if lfc_combos is not None and isinstance(reference, str) \
            and reference == ALL_OTHERS:
        raise NotImplementedError(
            f"lfc_threshold is not supported with reference=ALL_OTHERS "
            f"({ALL_OTHERS!r}). The one-vs-rest path ranks all cells jointly, "
            f"which has no per-reference distribution to shift by 2**tau. Use "
            f"an explicit reference (a group name or a control AnnData).")
    if cpm_normalize and normalize_target_sum is not None:
        raise ValueError(
            "only one of cpm_normalize / normalize_target_sum may be set "
            "(cpm_normalize=True is equivalent to normalize_target_sum=1e6)."
        )
    if output_columns is not None:
        if not output_columns:
            raise ValueError(
                "output_columns must be a non-empty dict mapping default column "
                "names to output names, or None (got an empty dict)."
            )
        allowed = tuple(DEFAULT_OUTPUT_COLUMNS) + tuple(
            (lfc_column_names(lfc_combos) if lfc_combos else [])
            + (taustar_column_names(taustar_levels, taustar_se)
               if taustar_levels else []))
        unknown = [k for k in output_columns if k not in allowed]
        if unknown:
            raise KeyError(
                "output_columns keys not present in the de() output schema: "
                f"{unknown}. Valid keys: {list(allowed)}"
            )
        dests = list(output_columns.values())
        if len(set(dests)) != len(dests):
            raise ValueError(
                f"output_columns maps multiple keys to the same name: {dests}."
            )

    def _finish(result):
        # Return gpudge's GPU caches to the driver on a successful return (unless
        # the caller opted out) so a same-process caller can allocate GPU memory
        # after de(); torch/cupy pools otherwise hold ~all of VRAM. #76
        if release_gpu_memory:
            _release_gpu_memory()
        return result

    if _byo:
        if not callable(cell_source):
            raise TypeError(
                "de(): cell_source must be a callable returning an iterable "
                f"of CellGroup; got {type(cell_source).__name__}."
            )
        if targets is None or var_names is None:
            raise ValueError(
                "de(cell_source=...) requires targets= (the ordered target "
                "labels) and var_names= (the gene axis)."
            )
        _targets = np.asarray(targets, dtype=str)
        _var_names = np.asarray(var_names, dtype=str)
        if _targets.ndim != 1 or _targets.size == 0:
            raise ValueError(
                "de(): targets must be a non-empty 1-D sequence of labels.")
        if _var_names.ndim != 1 or _var_names.size == 0:
            raise ValueError(
                "de(): var_names must be a non-empty 1-D sequence of names.")
        if np.unique(_targets).size != _targets.size:
            raise ValueError(
                "de(): targets contains duplicate labels; each target must "
                "appear exactly once.")
        if groupby is not None:
            raise ValueError(
                "de(): groupby= is not used with cell_source= -- the source "
                "decides the grouping. Pass the labels via targets=.")
        if isinstance(reference, ad.AnnData):
            # Order, not just count: a permuted reference would otherwise
            # return confidently wrong per-gene results. Matches the existing
            # external-reference check for de(adata=, reference=<AnnData>).
            if not np.array_equal(np.asarray(reference.var_names, dtype=str),
                                  _var_names):
                raise ValueError(
                    "reference AnnData var_names do not match var_names "
                    "(element-for-element, not just in length); align the "
                    "reference to the gene axis you are passing.")
            _ref_X = reference.X
        else:
            _ref_X = reference
        # BEFORE the .shape reads below, and before the CUDA check, so a
        # malformed reference fails fast on CPU with a clear message rather
        # than as an AttributeError inside cell_source_de.
        _check_2d(_ref_X, "reference")
        if int(_ref_X.shape[0]) == 0:
            raise ValueError(
                "reference has 0 cells; an external reference pool must be "
                "non-empty.")
        if int(_ref_X.shape[1]) != int(_var_names.size):
            raise ValueError(
                f"reference has {int(_ref_X.shape[1])} genes but var_names has "
                f"{int(_var_names.size)}; the reference and targets must share "
                "the gene axis.")
        if isinstance(normalize_target_sum, str) \
                and normalize_target_sum == "median":
            raise NotImplementedError(
                "normalize_target_sum='median' is not supported with "
                "cell_source= yet: the median needs a row-sums pre-pass over "
                "every target group before DE can start. Pass the library-size "
                "target as a number instead (e.g. normalize_target_sum=1e6, "
                "or cpm_normalize=True).")
        if not torch.cuda.is_available():
            raise RuntimeError(
                "gpudge requires a CUDA GPU; "
                "torch.cuda.is_available() returned False"
            )
        from . import _refpool
        return _finish(_refpool.cell_source_de(
            cell_source, targets=_targets, var_names=_var_names,
            reference=_ref_X, mean_calc=mean_calc, epsilon=epsilon,
            gpu_gene_chunk_size=gpu_gene_chunk_size, oom_recovery=oom_recovery,
            cpm_normalize=cpm_normalize,
            normalize_target_sum=normalize_target_sum,
            output_columns=output_columns, lfc_combos=lfc_combos,
            taustar_levels=taustar_levels, taustar_iters=taustar_iters,
            taustar_se=taustar_se,
            filter_gene_min_mean_value=filter_gene_min_mean_value,
            filter_gene_min_total_value=filter_gene_min_total_value,
            filter_gene_min_cpm_cell=filter_gene_min_cpm_cell,
            filter_gene_min_cpm_bulk=filter_gene_min_cpm_bulk,
            keep_genes=keep_genes, device=torch.device("cuda"),
        ))

    if _streaming:
        for _name, _val in (("stream_n_workers", stream_n_workers),
                            ("stream_prefetch", stream_prefetch)):
            if not isinstance(_val, (int, np.integer)) or isinstance(_val, bool):
                raise TypeError(
                    f"{_name} must be an int, got {type(_val).__name__}."
                )
        if stream_n_workers < 1:
            raise ValueError(
                f"stream_n_workers must be >= 1, got {stream_n_workers}."
            )
        if stream_prefetch < 0:
            raise ValueError(
                f"stream_prefetch must be >= 0 (0 disables prefetch), got {stream_prefetch}."
            )
        if densify_input:
            raise ValueError(
                "de(): densify_input is not supported with archive= "
                "(X is never fully resident when streaming)."
            )
        if isinstance(reference, str) and reference == ALL_OTHERS:
            raise NotImplementedError(
                f"de(): reference=ALL_OTHERS ({ALL_OTHERS!r}) is not supported with "
                "archive= (1-vs-rest needs global ranks over all cells)."
            )
        if not torch.cuda.is_available():
            raise RuntimeError(
                "gpudge requires a CUDA GPU; torch.cuda.is_available() returned False"
            )
        from ._shard_stream import stream_de
        return _finish(stream_de(
            archive,
            groupby=groupby, reference=reference,
            mean_calc=mean_calc, epsilon=epsilon,
            gpu_gene_chunk_size=gpu_gene_chunk_size, oom_recovery=oom_recovery,
            cpm_normalize=cpm_normalize,
            normalize_target_sum=normalize_target_sum,
            output_columns=output_columns,
            lfc_combos=lfc_combos,
            taustar_levels=taustar_levels, taustar_iters=taustar_iters,
            taustar_se=taustar_se,
            filter_gene_min_mean_value=filter_gene_min_mean_value,
            filter_gene_min_total_value=filter_gene_min_total_value,
            filter_gene_min_cpm_cell=filter_gene_min_cpm_cell,
            filter_gene_min_cpm_bulk=filter_gene_min_cpm_bulk,
            keep_genes=keep_genes, stream_n_workers=stream_n_workers,
            stream_prefetch=stream_prefetch, device=torch.device("cuda"),
        ))
    # ---- in-memory path below ----
    if groupby is None or reference is None:
        raise ValueError("de(): in-memory mode requires groupby= and reference=.")
    # In-memory external reference pool: a separate control AnnData ranked
    # resident-sorted on GPU with NO target-reference concat (the same core the
    # streaming Mode-2 path runs -> results are bit-identical). Cheap invariants
    # validated here, before the CUDA check, so they fail fast on CPU too.
    if isinstance(reference, ad.AnnData):
        if densify_input:
            raise ValueError(
                "de(): densify_input=True is not supported with an AnnData "
                "reference= (the external reference is ranked resident on the "
                "GPU, not densified in place; matches the streaming path)."
            )
        if reference.n_vars != adata.n_vars:
            raise ValueError(
                f"reference AnnData has {reference.n_vars} genes but adata has "
                f"{adata.n_vars}; the reference and targets must share the gene "
                "axis."
            )
        if not np.array_equal(np.asarray(reference.var_names),
                              np.asarray(adata.var_names)):
            raise ValueError(
                "reference AnnData var_names do not match adata's gene axis "
                "order; align the reference to adata.var_names."
            )
        if reference.n_obs == 0:
            raise ValueError(
                "reference AnnData has 0 cells; an external reference pool must "
                "be non-empty."
            )
        if not torch.cuda.is_available():
            raise RuntimeError(
                "gpudge requires a CUDA GPU; "
                "torch.cuda.is_available() returned False"
            )
        from ._refpool import inmem_external_ref_de
        return _finish(inmem_external_ref_de(
            adata, groupby=groupby, reference=reference,
            mean_calc=mean_calc, epsilon=epsilon,
            gpu_gene_chunk_size=gpu_gene_chunk_size, oom_recovery=oom_recovery,
            cpm_normalize=cpm_normalize,
            normalize_target_sum=normalize_target_sum,
            output_columns=output_columns,
            lfc_combos=lfc_combos,
            taustar_levels=taustar_levels, taustar_iters=taustar_iters,
            taustar_se=taustar_se,
            filter_gene_min_mean_value=filter_gene_min_mean_value,
            filter_gene_min_total_value=filter_gene_min_total_value,
            filter_gene_min_cpm_cell=filter_gene_min_cpm_cell,
            filter_gene_min_cpm_bulk=filter_gene_min_cpm_bulk,
            keep_genes=keep_genes, device=torch.device("cuda"),
        ))
    # Guard non-str reference BEFORE any `reference == ...` comparison: a
    # list/array reference would otherwise raise the opaque "truth value of an
    # array is ambiguous" error. ALL_OTHERS is itself a str, so isinstance
    # covers the sentinel too. (AnnData is handled above.)
    if not isinstance(reference, str):
        raise ValueError(
            "de(): in-memory reference= must be a group-label string, the "
            f"ALL_OTHERS sentinel (a string), or an AnnData control pool; got "
            f"{type(reference).__name__}."
        )
    if reference == ALL_OTHERS and mean_calc == "geometric":
        raise NotImplementedError(
            f"reference={ALL_OTHERS!r} with mean_calc='geometric' is not "
            "supported (would mix geometric target means with arithmetic "
            f"rest means). Use mean_calc='arithmetic' for {ALL_OTHERS!r}, "
            "or pick a literal reference group for geometric."
        )
    # PREFLIGHT, before the CUDA probe, the CSR coercion, the row-sums pass and
    # every host accumulator: densify_input cannot be honoured on a view, and
    # rejecting it late meant the caller paid all of that first. (codex review)
    if densify_input and getattr(adata, "is_view", False) and sp.issparse(adata.X):
        raise ValueError(
            "densify_input=True is not supported when adata is an AnnData "
            "view: assigning to a view's .X writes through to the parent "
            "instead of rebinding, so the dense array is paid for and then "
            "discarded. Pass adata.copy() (or adata.to_memory()) instead.")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "gpudge requires a CUDA GPU; "
            "torch.cuda.is_available() returned False"
        )
    device = torch.device("cuda")

    # A non-CSR sparse adata.X (typically CSC from an upstream concat/cache)
    # would silently drop to single-threaded scipy slicing in the numba fast
    # paths (csr_row_sums / csr_rows_col_range_to_dense). Coerce once, in place,
    # to canonical CSR (matches the densify_input in-place contract below); one
    # UserWarning. Restricted to the literal-reference path: the ALL_OTHERS
    # densify (all cells per chunk via adata.X[:, s:e].tocsc()) never touches the
    # numba CSR kernel, so coercing CSC there buys nothing and would only add a
    # per-chunk CSR->CSC round-trip — that path is a separately-scoped follow-up
    # (#66). Leaving ALL_OTHERS untouched keeps it byte-for-byte as before.
    # Coercion target depends on whether ``adata`` can actually be rebound.
    #
    # MATERIALIZED AnnData: assign back to ``adata.X``, as the #66 design
    # specifies. That is deliberate and load-bearing for MEMORY — it drops the
    # caller's CSC/COO refcount so only one sparse encoding is live, and
    # densify_input then frees that too. Holding a local CSR copy instead would
    # keep BOTH encodings resident for the whole run and can turn a large
    # supported CSC input into a host OOM (spec
    # docs/superpowers/specs/2026-07-02-noncsr-sparse-coerce-csr-design.md).
    #
    # VIEW: rebinding is impossible. ``view.X = ...`` writes the value back
    # THROUGH into the parent and re-reads the parent slice, so the coercion
    # silently vanished (adata.X stayed a SparseCSCMatrixView) while ensure_csr's
    # warning said it had happened — dropping every gather tile onto scipy
    # slicing, the regression #66 added this line to prevent. A view therefore
    # keeps a local CSR and accepts that the parent's encoding stays alive; that
    # is the unavoidable cost of a view, not a regression on the common path.
    #
    # Assign ONLY when ensure_csr actually returned a different object: it passes
    # an already-CSR X through unchanged, and assigning it back on a view would
    # run a full O(n_obs x n_var) sparse scatter into the parent for nothing.
    # (ultrareview 2026-08; view/materialized split per the codex review.)
    if reference != ALL_OTHERS:
        # stacklevel=3: de() -> ensure_csr -> warn points at the user's de() call.
        _coerced = ensure_csr(adata.X, name="adata.X", stacklevel=3)
        if _coerced is adata.X:
            _X_host = adata.X                      # already CSR: nothing to do
        elif getattr(adata, "is_view", False):
            _X_host = _coerced                     # cannot rebind a view
        else:
            adata.X = _coerced                     # in-place: frees the CSC
            _X_host = adata.X
        # MUST drop this local. Once _X_host is established, `_coerced` is a
        # second strong reference to the sparse matrix that lives until de()
        # returns — so densify_input's "the sparse matrix is dropped after this
        # point" contract would silently fail to free anything, on BOTH the
        # already-CSR and the coerced-materialized paths. Pinned by
        # test_densify_input_releases_the_sparse_matrix. (codex review round 4.)
        del _coerced
    else:
        _X_host = adata.X

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
    _row_sums_np_early = (csr_row_sums(_X_host)
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
    # free GPU memory on datasets with many groups (e.g., CCL_2 with
    # 4672 target guides adds ~110-150K bytes/gene, comparable to the
    # ranking term for typical ref sizes).
    n_combos = 0 if lfc_combos is None else len(lfc_combos)
    # tau* accumulator ROWS, not levels: tau_star_se appends lo/hi/se. The
    # sizer's `n_levels` parameter has always been a row count (it multiplies
    # accumulator bytes); the name is kept because tests/test_stream.py passes
    # it by keyword in 15 places.
    n_taustar_rows = (0 if taustar_levels is None
                      else len(taustar_column_names(taustar_levels, taustar_se)))
    if gpu_gene_chunk_size is None:
        free, _ = torch.cuda.mem_get_info(device)
        max_group_rows = 0
        if state.ref_label == ALL_OTHERS:
            budget_n = state.n_cells
        else:
            counts = np.bincount(state.labels, minlength=n_groups)
            budget_n = int(counts[state.ref_label_idx])
            # UNCONDITIONAL: the sizer models the Phase-1 target tile whenever it
            # knows the tile height, so withholding the height here would leave
            # the base literal-reference path with the very OOM the un-gating in
            # _stream.py fixes. Gating this on `n_combos or n_taustar_rows` was
            # the caller half of the same bug. (codex review, ultrareview 2026-08)
            max_group_rows = int(
                np.delete(counts, state.ref_label_idx).max(initial=0))
        gpu_gene_chunk_size = _auto_gene_chunk_size(
            free_bytes=free, budget_n=budget_n, n_groups=n_groups,
            mean_calc=mean_calc, n_genes=state.n_genes,
            ref_mode=state.ref_label != ALL_OTHERS,
            n_combos=n_combos, n_levels=n_taustar_rows,
            max_group_rows=max_group_rows)

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
    dir_U_acc = (np.zeros((n_combos, n_groups, n_genes), dtype=np.float64)
                 if n_combos else None)
    dir_p_acc = (np.ones((n_combos, n_groups, n_genes), dtype=np.float64)
                 if n_combos else None)
    taustar_acc = (np.empty((n_taustar_rows, n_groups, n_genes), dtype=np.float64)
                   if taustar_levels is not None else None)

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
    # to zero — keeping both representations alive at CCL_2 scale costs 310 GB
    # host and triggers severe paging.
    if densify_input and sp.issparse(_X_host):
        # A view cannot satisfy this parameter's contract. Assigning to a view's
        # .X writes through to the parent rather than rebinding, so the dense
        # array would be built, converted to COO, scattered into the parent and
        # then discarded — the caller pays the full dense allocation (~310 GB peak
        # at CCL_2 scale, per the docstring) and the chunk loop still takes the
        # sparse branch. Fail loudly instead of burning the memory for nothing.
        # (ultrareview 2026-08)
        # Unreachable: the preflight above rejects views before any work. Kept
        # so a future refactor that drops the preflight cannot silently restore
        # the write-through no-op.
        assert not getattr(adata, "is_view", False), \
            "densify_input on a view should have been rejected at the preflight"
        warnings.warn(
            "densify_input=True: replacing adata.X (sparse) with a dense "
            "numpy array in place. The caller's AnnData is mutated; pass "
            "adata.copy() first to preserve sparsity.",
            UserWarning,
            stacklevel=2,
        )
        adata.X = _X_host.toarray()
        _X_host = adata.X          # rebind succeeded (not a view): re-read it

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
        _noncount = x_has_noncount_signal(_X_host)
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
        X_host = _X_host
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
        X_host = _X_host

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
            if sp.issparse(_X_host):
                block = (_X_host[:, start:stop].tocsc()
                         .toarray().astype(np.float32, copy=False))
            else:
                block = np.ascontiguousarray(_X_host[:, start:stop],
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
            # clamp_min(1.0): for a 1-cell input N_t==1 makes 12*N_t*(N_t-1)==0;
            # mn is also 0 there, so without the clamp tie_corr is 0/0 == NaN.
            # Clamped, tie_corr==0 and p degrades to the graceful ~1.0 sentinel
            # (numerator clamps to 0 below). (L1)
            tie_corr = mn * tie_term[None, :] / (12 * N_t * (N_t - 1)).clamp_min(1.0)
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
        # n_cells × chunk), which scales to CCL_2-size datasets.
        #
        # Both ref and per-group means are computed on GPU. Each slice is
        # already uploaded for the MWU sort/searchsorted, so the mean comes
        # essentially free — no duplicate host pass.

        # Pre-allocate one pinned host buffer for the per-group slice that
        # gets reused across all (chunk × group) iterations. torch's
        # implicit .to(device) on non-pinned memory does a CPU pin+copy
        # internally — nsys 2026-05-25 attributed ~12 s of cudaMemcpyAsync
        # API wall to that step on CCL_2 (separate from the 11.7 s of
        # actual H2D transfer). With the buffer pre-pinned, .to(device,
        # non_blocking=True) skips that step.
        #
        # Sizing: max_group_rows × gpu_gene_chunk_size × 4 bytes. On CCL_2
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
            # Cap the pinned buffer width at n_genes: a user-pinned
            # gpu_gene_chunk_size above n_genes would over-allocate page-locked
            # host memory (the gene-chunk loop below never emits a tile wider
            # than n_genes). The loop step stays the raw gpu_gene_chunk_size. #80b
            _pinned_w = _pinned_buf_width(gpu_gene_chunk_size, n_genes)
            group_host_bufs = [
                torch.empty(max_group_rows, _pinned_w,
                            dtype=torch.float32, pin_memory=True),
                torch.empty(max_group_rows, _pinned_w,
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
            # NaN (not 0.0) so untouched rows — the skipped ref row and any
            # empty target group — read as the documented NaN p sentinel rather
            # than being written into p_acc as p=0.0 (maximally significant).
            # Active groups overwrite their row below. (L2; latent: np.unique
            # guarantees observed groups are non-empty, ref row is dropped.)
            p_chunk = torch.full(
                (n_groups, ch), float("nan"), dtype=torch.float64, device=device)
            if lfc_combos is not None:
                dir_U_chunk = torch.zeros(
                    (n_combos, n_groups, ch), dtype=torch.float64, device=device)
                dir_p_chunk = torch.full(
                    (n_combos, n_groups, ch), float("nan"),
                    dtype=torch.float64, device=device)
            if taustar_levels is not None:
                taustar_chunk = torch.full(
                    (n_taustar_rows, n_groups, ch), float("nan"),
                    dtype=torch.float64, device=device)
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
                (arith_t, reported_t, other_t, u1, p,
                 dir_u1, dir_p, taustar_t) = group_chunk_stats(
                    group_t, sorted_ref, ref_tie_term, n_ref,
                    mean_calc=mean_calc, scale_main=_scale_main,
                    group_scales=_scales,
                    want_other=other_target_acc is not None,
                    lfc_combos=lfc_combos,
                    taustar_levels=taustar_levels,
                    taustar_iters=taustar_iters,
                    taustar_se=taustar_se)
                arith_target_chunk[g] = arith_t
                if other_target_acc is not None:
                    other_target_chunk[g] = other_t
                if mean_calc == "geometric":
                    target_mean_chunk[g] = reported_t
                U_chunk[g] = u1
                p_chunk[g] = p
                if lfc_combos is not None:
                    dir_U_chunk[:, g] = dir_u1
                    dir_p_chunk[:, g] = dir_p
                if taustar_levels is not None:
                    taustar_chunk[:, g] = taustar_t
                del (
                    group_t, arith_t, reported_t, other_t, u1, p, dir_u1,
                    dir_p, taustar_t,
                )

                iter_idx += 1

            # Batched D2H: collapses ~14k per-chunk .cpu() calls into 3-4.
            arith_target_chunk_np = arith_target_chunk.cpu().numpy()
            arith_target_acc[:, start:stop] = arith_target_chunk_np
            if other_target_acc is not None:
                other_target_acc[:, start:stop] = other_target_chunk.cpu().numpy()
                del other_target_chunk
            U_acc[:, start:stop] = U_chunk.cpu().numpy()
            p_acc[:, start:stop] = p_chunk.cpu().numpy()
            if lfc_combos is not None:
                dir_U_acc[:, :, start:stop] = dir_U_chunk.cpu().numpy()
                dir_p_acc[:, :, start:stop] = dir_p_chunk.cpu().numpy()
                del dir_U_chunk, dir_p_chunk
            if taustar_levels is not None:
                taustar_acc[:, :, start:stop] = taustar_chunk.cpu().numpy()
                del taustar_chunk
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
    extra_columns = None
    if lfc_combos is not None:
        extra_columns = {}
        for k, (p_name, u_name, _q_name) in enumerate(lfc_base_names(lfc_combos)):
            extra_columns[p_name] = dir_p_acc[k][target_indices]
            extra_columns[u_name] = effect_size_from_u(
                dir_U_acc[k][target_indices], target_ncells, ref_ncells)
    if taustar_levels is not None:
        extra_columns = extra_columns or {}
        for k, name in enumerate(
                taustar_column_names(taustar_levels, taustar_se)):
            extra_columns[name] = taustar_acc[k][target_indices]

    df = assemble_dataframe(
        target=target_labels,
        feature=adata.var_names.to_numpy(),
        target_mean=tm,
        ref_mean=rm,
        target_ncells=target_ncells,
        ref_ncells=ref_ncells,
        log2_fold_change=lfc,
        p_value=p_t2,
        test_statistic=effect_size_from_u(
            U_t2, target_ncells, ref_ncells),
        p_adj=np.zeros_like(p_t2),
        flat_keep=flat_keep,
        extra_columns=extra_columns,
        output_columns=None,
    )

    n_target = len(target_labels)
    if df.height > 0:
        # Per-row group index, derived directly from the 2D keep mask:
        # np.nonzero on a (n_target, n_features) bool returns (row, col)
        # arrays — `row` IS the target_pos for each kept row, which is what
        # bh_per_group wants. This skips the previous
        # `keep_indices = np.flatnonzero(flat_keep); post_filter_g =
        # keep_indices // n_features` formulation that held two large int64
        # arrays simultaneously.
        post_filter_g = np.nonzero(keep_for_targets)[0]
        g_idx = torch.from_numpy(post_filter_g)
        adj = bh_per_group(df["p_value"].to_torch(), g_idx, n_target)
        new_cols = [pl.Series("p_adj", adj.numpy())]
        if lfc_combos is not None:
            # Each (tau, direction) is its own family of hypotheses.
            for p_name, _u_name, q_name in lfc_base_names(lfc_combos):
                q = bh_per_group(df[p_name].to_torch(), g_idx, n_target)
                new_cols.append(pl.Series(q_name, q.numpy()))
        df = df.with_columns(new_cols)
    elif lfc_combos is not None:
        # ZERO ROWS still needs every directional p_adj to EXIST and be typed,
        # or the schema differs from a populated result and from
        # empty_output_frame.
        df = df.with_columns([
            pl.Series(q_name, [], dtype=pl.Float64)
            for _p, _u, q_name in lfc_base_names(lfc_combos)
        ])

    # Final explicit projection pins the CANONICAL column order. Without it the
    # p_adj columns land after ALL the p/Ueffect columns, giving
    # p1,es1,p2,es2,q1,q2 instead of the required p1,es1,q1,p2,es2,q2.
    if lfc_combos is not None or taustar_levels is not None:
        df = df.select(
            list(DEFAULT_OUTPUT_COLUMNS)
            + (lfc_column_names(lfc_combos) if lfc_combos else [])
            + (taustar_column_names(taustar_levels, taustar_se)
               if taustar_levels else []))

    if output_columns is None:
        return _finish(df)
    # Keys + duplicate-destination validated at entry. select-THEN-rename (not
    # rename-then-select) so a destination name that shadows an unselected
    # default column can't collide on rename. (Codex review.)
    return _finish(df.select(list(output_columns)).rename(output_columns))
