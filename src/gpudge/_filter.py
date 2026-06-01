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

    Sparse: samples ``X.data`` (the nonzero values). Dense: samples
    ``np.ravel(X)`` (NOT ``np.ndarray.data`` — that is a raw memory buffer, not
    the values). Sampling uses ``np.linspace(0, n-1, k)`` indices so the span
    ALWAYS includes the first and last element (a head prefix could miss late
    fractional/negative values).
    """
    if sp.issparse(X):
        data = X.data
    else:
        data = np.ravel(np.asarray(X))
    n = data.shape[0]
    if n == 0:
        return False
    idx = np.linspace(0, n - 1, min(k, n), dtype=np.int64)
    sample = data[idx]
    if (sample < 0).any():
        return True
    return bool(np.any(np.abs(sample - np.rint(sample)) > tol))


def _row_scale_needs(cpm_normalize, min_cpm_cell, min_cpm_bulk):
    """What CPM precompute de() needs: (need_row_sums_np, need_row_scales_t).

    Two distinct needs, deliberately NOT conflated:
      - ``need_row_scales_t`` — the GPU per-cell scale tensor (``row_scales`` =
        1e6/L) plus the per-group row-index tensors (``ref_rows_t`` /
        ``group_rows_t``). Used ONLY to CPM-scale the MWU/report tensor
        (``cpm_normalize``) or to form the per-cell-CPM "scaled" unit
        (``filter_gene_min_cpm_cell``).
      - ``need_row_sums_np`` — the CPU per-cell library sizes. Needed to build
        ``row_scales`` AND for the per-group library totals (``group_libtot``)
        that pooled-bulk CPM (``filter_gene_min_cpm_bulk``) uses.

    A ``filter_gene_min_cpm_bulk``-only run therefore needs the CPU row sums but
    NOT the GPU scale/index tensors — avoiding that allocation on large data.
    """
    need_row_scales_t = bool(cpm_normalize) or (min_cpm_cell is not None)
    need_row_sums_np = need_row_scales_t or (min_cpm_bulk is not None)
    return need_row_sums_np, need_row_scales_t
