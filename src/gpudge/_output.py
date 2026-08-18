# src/gpudge/_output.py
"""Assemble the per-(target, feature) DataFrame with optional rename/select."""
from __future__ import annotations

import contextlib

import numpy as np
import polars as pl

DEFAULT_OUTPUT_COLUMNS = (
    "target", "feature",
    "target_mean", "ref_mean",
    "target_ncells", "ref_ncells",
    "log2_fold_change",
    "p_value", "Ueffect", "p_adj",
)

# Canonical typed schema of the de() output. The dtypes mirror what
# assemble_dataframe() produces from its numpy inputs (str -> String,
# {target,ref}_ncells int64 -> Int64, the rest float64 -> Float64); the
# test suite pins this against a real assemble_dataframe output so the two
# can't drift. Kept here (next to assemble_dataframe) as the single source
# of truth so callers building an empty result don't hand-maintain a copy.
_OUTPUT_SCHEMA: dict[str, pl.DataType] = {
    "target": pl.String,
    "feature": pl.String,
    "target_mean": pl.Float64,
    "ref_mean": pl.Float64,
    "target_ncells": pl.Int64,
    "ref_ncells": pl.Int64,
    "log2_fold_change": pl.Float64,
    "p_value": pl.Float64,
    "Ueffect": pl.Float64,
    "p_adj": pl.Float64,
}


def effect_size_from_u(u, target_ncells, ref_ncells):
    """Rank-biserial correlation (Cliff's delta) ``2A − 1`` from Mann–Whitney U1.

    ``A = U1 / (m·n) = P(target > ref) + 0.5·P(target == ref)`` (probability of
    superiority / AUC); ``2A − 1 ∈ [−1, 1]`` is the signed rank effect size — its
    sign is the RANK direction of change (positive = target ranks above the
    reference). ``u`` is ``(n_targets, n_genes)``; ``target_ncells`` is
    ``(n_targets,)``; ``ref_ncells`` is a scalar (ref-mode) or ``(n_targets,)``
    (all-others). Returns float64. Yields NaN where ``m·n == 0`` (a degenerate
    empty group / empty reference), matching the NaN p-value sentinel.
    """
    u = np.asarray(u, dtype=np.float64)
    m = np.asarray(target_ncells, dtype=np.float64)
    n = np.asarray(ref_ncells, dtype=np.float64)
    denom = (m * n)[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        eff = 2.0 * u / denom - 1.0
    return np.where(denom > 0, eff, np.nan)


def log2_ratio(target_mean, ref_mean, epsilon: float) -> np.ndarray:
    """``log2((target_mean + epsilon) / (ref_mean + epsilon))``, silencing only
    the DOCUMENTED degeneracies.

    ``epsilon=0`` is a supported input whose outcomes ``de()`` promises: a gene
    zero in both groups yields NaN, a one-sided gene ±inf. Producing a
    documented value must not also emit an unsuppressed numpy RuntimeWarning --
    which under ``-W error::RuntimeWarning`` turns the documented path into a
    crash, and took four of this repo's own tests with it.

    The suppression is CONDITIONAL, and that is the point. It applies only when
    ``epsilon == 0`` -- the only setting under which the documented degeneracies
    arise -- AND both means are finite and non-negative.

    That predicate is a DOMAIN GUARD, not a proof of completeness, and one gap
    is known: with ``epsilon == 0`` and two strictly positive finite means whose
    quotient UNDERFLOWS (``float64.tiny / float64.max``), ``log2(0)`` warns and
    this helper silences it even though neither mean is zero. Unreachable from
    ``de()`` -- its float64 means come from float32-staged data, and dividing by
    even the largest possible cell count leaves float64 range to spare -- so it
    is documented rather than closed. Do not restate this function as
    "silences only the documented cases" without closing it. (codex review.)

    gpudge accepts arbitrary X, so the clauses each earn their place:

    * a centered or log-transformed input can carry a NEGATIVE mean, and
      ``log2`` of a negative ratio is undefined rather than documented;
    * an INFINITE mean gives ``inf/inf`` → NaN, equally undefined, and ``inf``
      would otherwise pass a bare non-negativity test (codex review);
    * a NaN mean makes both reductions NaN, and every comparison against NaN is
      False, so it keeps its diagnostics too.

    The ``epsilon`` test is not merely a fast path, though it is that too (the
    default ``epsilon=1e-9`` pays no scan, ~87 ms at CCL_2 shape). It is load
    bearing: with a positive epsilon the DENOMINATOR is ``>= epsilon``, but the
    QUOTIENT can still reach zero by underflow -- ``1e-300 / 3.4e38`` is 0.0,
    and ``log2(0)`` warns -- so suppressing on a positive epsilon would hide a
    real diagnostic. Pass the UNBROADCAST arrays: ``ref_mean`` is ``(n_genes,)``
    on the literal-reference path, and scanning the broadcast view instead would
    cost a full ``(n_groups, n_genes)`` pass.
    """
    tm = np.asarray(target_mean)
    rm = np.asarray(ref_mean)
    # `+ 0.0` normalizes a signed zero: `epsilon=-0.0` passes de()'s
    # `isfinite and >= 0` validation, and `x / (-0.0 + -0.0)` is -inf where the
    # docstring promises +inf for a target-only gene.
    epsilon = epsilon + 0.0
    documented_only = epsilon == 0 and (
        tm.size == 0 or rm.size == 0
        # min() >= 0 excludes negatives and -inf; isfinite(max()) excludes
        # +inf. NaN fails both, which is the intent.
        or (tm.min() >= 0 and rm.min() >= 0
            and np.isfinite(tm.max()) and np.isfinite(rm.max())))
    ctx = (np.errstate(divide="ignore", invalid="ignore") if documented_only
           else contextlib.nullcontext())
    with ctx:
        return np.log2((tm + epsilon) / (rm + epsilon))


def empty_output_frame(output_columns: dict[str, str] | None = None,
                       extra_names: list[str] | None = None) -> pl.DataFrame:
    """Zero-row DataFrame carrying the canonical de() output schema (correctly
    typed, **not** all-Null columns), honouring ``output_columns`` the same way
    ``assemble_dataframe`` does.

    ``extra_names`` are directional (lfc_threshold) column names; they are
    appended as Float64 so an empty result has the identical schema to a
    non-empty one with the same tau grid. Building from empty lists instead
    yields Null columns that mismatch / error downstream on concat.
    """
    schema = dict(_OUTPUT_SCHEMA)
    for name in (extra_names or ()):
        schema[name] = pl.Float64
    df = pl.DataFrame(schema=schema)
    if output_columns is None:
        return df
    return df.select(list(output_columns)).rename(output_columns)


def assemble_dataframe(
    *,
    target: np.ndarray,             # (n_guides,) str
    feature: np.ndarray,            # (n_genes,) str
    target_mean: np.ndarray,        # (n_guides, n_genes)
    ref_mean: np.ndarray,           # (n_guides, n_genes) or (n_genes,)
    target_ncells: np.ndarray,      # (n_guides,)
    ref_ncells: int | np.ndarray,   # int (ref mode) or (n_guides,) (all_others)
    log2_fold_change: np.ndarray,   # (n_guides, n_genes)
    p_value: np.ndarray,
    test_statistic: np.ndarray,
    p_adj: np.ndarray,
    flat_keep: np.ndarray | None = None,  # (n_guides*n_genes,) bool — pre-filter
    extra_columns: dict[str, np.ndarray] | None = None,  # (n_guides, n_genes) each
    output_columns: dict[str, str] | None = None,
) -> pl.DataFrame:
    n_guides, n_genes = target_mean.shape

    rm_arr = np.asarray(ref_mean)
    if rm_arr.ndim == 1 and rm_arr.shape != (n_genes,):
        raise ValueError(
            f"ref_mean has shape {rm_arr.shape}; 1D ref_mean must be "
            f"({n_genes},) to match target_mean.shape[1]."
        )

    # Validate flat_keep eagerly so the size check fires before either
    # branch, and short-circuit an all-True mask to the unfiltered path.
    # (An unfiltered run -- no filter_gene_* set -- produces such a mask; with
    # it, the filtered path would allocate full-length guide_idx/gene_idx int64
    # arrays and do large fancy-indexing for no row reduction. np.all is O(n)
    # but short-circuits on the first False — the worst case (all-True) is
    # exactly the case where we save much more than the scan costs.
    if flat_keep is not None:
        fk = np.asarray(flat_keep)
        if fk.size != n_guides * n_genes:
            raise ValueError(
                f"flat_keep.size={fk.size} does not match target_mean shape "
                f"{(n_guides, n_genes)} (expected {n_guides * n_genes}). "
                "Pass a mask raveled from the same (n_guides, n_genes) layout."
            )
        if fk.all():
            flat_keep = None

    if flat_keep is None:
        # Unfiltered path: build columns directly via np.repeat/np.tile / ravel
        # without materialising explicit (guide_idx, gene_idx) index arrays —
        # on CCL_2 those would be ~700 MB each (int64 × 86.6M).
        target_col = np.repeat(np.asarray(target), n_genes)
        feature_col = np.tile(np.asarray(feature), n_guides)
        target_ncells_col = np.repeat(np.asarray(target_ncells), n_genes)
        if np.isscalar(ref_ncells):
            ref_ncells_col = np.full(
                n_guides * n_genes, ref_ncells, dtype=np.int64
            )
        else:
            ref_ncells_col = np.repeat(np.asarray(ref_ncells), n_genes)
        if rm_arr.ndim == 1:
            ref_mean_col = np.broadcast_to(
                rm_arr, (n_guides, n_genes)
            ).ravel().copy()
        else:
            ref_mean_col = rm_arr.ravel()
        target_mean_col = np.asarray(target_mean).ravel()
        log2_fold_change_col = np.asarray(log2_fold_change).ravel()
        p_value_col = np.asarray(p_value).ravel()
        test_statistic_col = np.asarray(test_statistic).ravel()
        p_adj_col = np.asarray(p_adj).ravel()
        extra_cols = {k: np.asarray(v).ravel()
                      for k, v in (extra_columns or {}).items()}
    else:
        # nsys 2026-05-25 attributed ~6.7 s of de() wall to assemble_dataframe,
        # dominated by polars `new_str` on the 86.6M-row target/feature
        # columns. Building only the kept rows (~53M on CCL_2) cuts string
        # materialisation + Arrow conversion proportionally and lets us drop
        # the downstream `.filter(...)` pass.
        #
        # `fk` was validated and bound above; 2D nonzero gives
        # (guide_idx, gene_idx) directly — no flat keep_indices intermediate,
        # so peak memory is two int64 arrays of length n_keep instead of three.
        guide_idx, gene_idx = np.nonzero(fk.reshape(n_guides, n_genes))
        if rm_arr.ndim == 1:
            ref_mean_col = rm_arr[gene_idx]
        else:
            ref_mean_col = rm_arr[guide_idx, gene_idx]
        if np.isscalar(ref_ncells):
            ref_ncells_col = np.full(guide_idx.size, ref_ncells, dtype=np.int64)
        else:
            ref_ncells_col = np.asarray(ref_ncells)[guide_idx]
        target_col = np.asarray(target)[guide_idx]
        feature_col = np.asarray(feature)[gene_idx]
        target_ncells_col = np.asarray(target_ncells)[guide_idx]
        target_mean_col = np.asarray(target_mean)[guide_idx, gene_idx]
        log2_fold_change_col = np.asarray(log2_fold_change)[guide_idx, gene_idx]
        p_value_col = np.asarray(p_value)[guide_idx, gene_idx]
        test_statistic_col = np.asarray(test_statistic)[guide_idx, gene_idx]
        p_adj_col = np.asarray(p_adj)[guide_idx, gene_idx]
        extra_cols = {k: np.asarray(v)[guide_idx, gene_idx]
                      for k, v in (extra_columns or {}).items()}

    # polars infers a column's dtype from its VALUES: a 0-length object-dtype
    # numpy array yields pl.Object, where a non-empty one yields pl.String.
    # `adata.var_names.to_numpy()` and the ingest label array are both object
    # dtype, so a fully-filtered (or target-less) result would carry a schema
    # that differs from a populated result's AND from empty_output_frame's --
    # the exact mismatch-on-concat failure empty_output_frame exists to avoid.
    # Cast only in the empty case: a non-empty object array already infers
    # String, so the hot path (86.6M rows on CCL_2) is untouched. Both columns
    # are row-length, so one size check covers them.
    if target_col.size == 0:
        target_col = target_col.astype(np.str_)
        feature_col = feature_col.astype(np.str_)

    columns: dict[str, np.ndarray] = {
        "target":            target_col,
        "feature":           feature_col,
        "target_mean":       target_mean_col,
        "ref_mean":          ref_mean_col,
        "target_ncells":     target_ncells_col,
        "ref_ncells":        ref_ncells_col,
        "log2_fold_change":  log2_fold_change_col,
        "p_value":           p_value_col,
        "Ueffect":           test_statistic_col,
        "p_adj":             p_adj_col,
    }

    for k, v in extra_cols.items():
        if k in columns:
            raise KeyError(
                f"extra_columns key {k!r} shadows a default output column; "
                f"directional column names must not collide with "
                f"{sorted(columns)}.")
        columns[k] = v

    if output_columns is None:
        ordered = list(DEFAULT_OUTPUT_COLUMNS) + list(extra_cols)
        return pl.DataFrame({k: columns[k] for k in ordered})

    out: dict[str, np.ndarray] = {}
    for src, dst in output_columns.items():
        if src not in columns:
            raise KeyError(
                f"output_columns key {src!r} not in available columns: "
                f"{sorted(columns)}"
            )
        out[dst] = columns[src]
    return pl.DataFrame(out)
