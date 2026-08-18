# src/gpudge/_filter.py
"""Per-(target, gene) keep mask: AND of active expression filters + keep_genes.

Shared by de()'s two chunk loops (ref-mode and ALL_OTHERS):
  - within each filter: keep if target-group OR reference-group clears threshold
  - across filters: AND (a row must clear every active filter)
  - keep_genes (per-gene bool mask) is ANDed in, broadcast over targets
  - a NEGATIVE threshold is an explicit keep-all for that filter (robust to
    unit-agnostic *_value quantities that can be negative)
  - no active filters and keep_genes=None -> keep everything
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def _one_filter_mask(target_q, ref_q, threshold):
    """(target_q > thr) | (ref_q > thr); negative thr -> explicit all-True."""
    if threshold < 0.0:
        return np.ones(target_q.shape, dtype=bool)
    ref_b = ref_q if ref_q.ndim == 2 else ref_q[None, :]
    return (target_q > threshold) | (ref_b > threshold)


def combined_keep_mask(
    n_targets: int,
    n_genes: int,
    *,
    filters: list[tuple[np.ndarray, np.ndarray, float]],
    keep_genes: np.ndarray | None = None,
) -> np.ndarray:
    """AND each active filter's (target OR ref) mask, then AND keep_genes.

    ``filters`` holds only ACTIVE filters as ``(target_q, ref_q, threshold)``
    (the caller skips any whose threshold is None). ``target_q`` is
    ``(n_targets, n_genes)``; ``ref_q`` is ``(n_genes,)`` (broadcast) or
    ``(n_targets, n_genes)``. Returns a ``(n_targets, n_genes)`` bool mask.
    """
    keep = np.ones((n_targets, n_genes), dtype=bool)
    for target_q, ref_q, threshold in filters:
        keep &= _one_filter_mask(target_q, ref_q, threshold)
    if keep_genes is not None:
        keep &= keep_genes[None, :]
    return keep


def validate_keep_genes(keep_genes, n_genes: int) -> np.ndarray:
    """Return a validated bool array of length n_genes, else raise ValueError.

    Requires dtype EXACTLY boolean (no silent cast of int/str/object masks).
    """
    arr = np.asarray(keep_genes)
    if arr.ndim != 1:
        raise ValueError(
            f"keep_genes must be 1-D; got {arr.ndim}-D shape {arr.shape}."
        )
    if arr.shape[0] != n_genes:
        raise ValueError(
            f"keep_genes length {arr.shape[0]} != n_genes {n_genes}."
        )
    if arr.dtype != np.bool_:
        raise ValueError(
            f"keep_genes must be a boolean array (dtype np.bool_); got "
            f"{arr.dtype}. Cast explicitly (e.g. mask.astype(bool)) if intended."
        )
    return arr


def x_has_noncount_signal(X, *, k: int = 100_000, tol: float = 1e-6) -> bool:
    """True if a sample of X is fractional or negative (a non-raw-count signal).

    Sparse: samples ``X.data`` (the nonzero values). Dense: samples flat
    positions (NOT ``np.ndarray.data`` — that is a raw memory buffer, not the
    values). Sampling uses ``np.linspace(0, n-1, k)`` indices so the span
    ALWAYS includes the first and last element (a head prefix could miss late
    fractional/negative values).

    NEVER materialises a copy of the dense input. ``np.ravel`` defaults to
    ``order='C'`` and therefore COPIES anything that is not already C-contiguous
    — a whole-matrix transient (~40 GB for a 500k x 20k f32 group) inside a
    function whose entire job is to read at most ``k`` values. ``order='K'``
    returns a view for C- and F-contiguous input; genuinely strided input is
    sampled through ``unravel_index`` with no flattening at all. Note
    ``_check_sliceable_layout`` deliberately declines to normalise F-ordered
    input, so non-C-contiguous dense really does reach here. The sample values
    are unchanged for C-contiguous input; for other layouts the sampled
    POSITIONS differ (K/strided order vs imposed row-major), which cannot change
    the boolean result's meaning — it is a fractional/negative signal over a
    spread-out sample either way. (ultrareview 2026-08)
    """
    if sp.issparse(X):
        data = X.data
    else:
        Xa = np.asarray(X)
        n_elem = Xa.size
        if n_elem == 0:
            return False
        if Xa.flags["C_CONTIGUOUS"] or Xa.flags["F_CONTIGUOUS"]:
            data = np.ravel(Xa, order="K")          # view, never a copy
        else:
            idx = np.linspace(0, n_elem - 1, min(k, n_elem), dtype=np.int64)
            rc = np.unravel_index(idx, Xa.shape)
            sample = Xa[rc]
            if (sample < 0).any():
                return True
            return bool(np.any(np.abs(sample - np.rint(sample)) > tol))
    n = data.shape[0]
    if n == 0:
        return False
    idx = np.linspace(0, n - 1, min(k, n), dtype=np.int64)
    sample = data[idx]
    if (sample < 0).any():
        return True
    return bool(np.any(np.abs(sample - np.rint(sample)) > tol))


def _row_scale_needs(scale_main, min_cpm_cell, min_cpm_bulk, median_requested):
    """What CPM/normalization precompute de() needs: (need_row_sums_np,
    need_row_scales_t).

      - ``need_row_scales_t`` — the GPU per-cell scale tensor (``row_scales`` =
        ``numerator/L``) plus the per-group row-index tensors. Needed to scale
        the main unit (``scale_main``) or to form the per-cell-CPM "scaled" unit
        for the filter_gene_min_cpm_cell gate.
      - ``need_row_sums_np`` — the CPU per-cell library sizes. Needed to build
        ``row_scales`` AND for the per-group library totals (``group_libtot``)
        the bulk-CPM filter consumes AND to compute the median target sum.

    ``scale_main`` is ``target_sum is not None`` (normalization scales the test/
    reported unit). ``median_requested`` is ``normalize_target_sum == 'median'``;
    it forces the row-sum precompute even when nothing else needs it.
    """
    need_row_scales_t = bool(scale_main) or (min_cpm_cell is not None)
    need_row_sums_np = (need_row_scales_t or (min_cpm_bulk is not None)
                        or bool(median_requested))
    return need_row_sums_np, need_row_scales_t
