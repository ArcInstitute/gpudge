# src/gpudge/_output.py
"""Assemble the per-(target, feature) DataFrame with optional rename/select."""
from __future__ import annotations

import numpy as np
import polars as pl

DEFAULT_OUTPUT_COLUMNS = (
    "target", "feature",
    "target_mean", "ref_mean",
    "target_ncells", "ref_ncells",
    "log2_fold_change",
    "p_value", "test_statistic", "p_adj",
)


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
    # `min_feature_filter=0.0` produces such a mask; with it, the filtered
    # path would allocate full-length guide_idx/gene_idx int64 arrays and
    # do large fancy-indexing for no row reduction. np.all is O(n) but
    # short-circuits on the first False — the worst case (all-True) is
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
        # on cell line 2 those would be ~700 MB each (int64 × 86.6M).
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
    else:
        # nsys 2026-05-25 attributed ~6.7 s of de() wall to assemble_dataframe,
        # dominated by polars `new_str` on the 86.6M-row target/feature
        # columns. Building only the kept rows (~53M on cell line 2) cuts string
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

    columns: dict[str, np.ndarray] = {
        "target":            target_col,
        "feature":           feature_col,
        "target_mean":       target_mean_col,
        "ref_mean":          ref_mean_col,
        "target_ncells":     target_ncells_col,
        "ref_ncells":        ref_ncells_col,
        "log2_fold_change":  log2_fold_change_col,
        "p_value":           p_value_col,
        "test_statistic":    test_statistic_col,
        "p_adj":             p_adj_col,
    }

    if output_columns is None:
        return pl.DataFrame({k: columns[k] for k in DEFAULT_OUTPUT_COLUMNS})

    out: dict[str, np.ndarray] = {}
    for src, dst in output_columns.items():
        if src not in columns:
            raise KeyError(
                f"output_columns key {src!r} not in available columns: "
                f"{sorted(columns)}"
            )
        out[dst] = columns[src]
    return pl.DataFrame(out)
