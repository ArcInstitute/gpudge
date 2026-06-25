"""Pure resolution of the (cpm_normalize, normalize_target_sum) knobs into a
single internal target_sum, plus the cpm-filter rescale constant. GPU-free and
side-effect-free so it is unit-testable without CUDA; shared by the in-memory
and streaming de() paths."""
from __future__ import annotations

import numbers

import numpy as np

_MEDIAN = "median"


def resolve_target_sum(*, cpm_normalize, normalize_target_sum, row_sums):
    """Return the resolved per-cell normalization target as a float, or None
    for "no normalization".

    ``cpm_normalize=True`` is exactly ``normalize_target_sum=1e6``; only one of
    the two may be active. ``normalize_target_sum`` may be a positive number or
    the string ``"median"`` (scanpy's ``target_sum=None`` behaviour: the median
    of per-cell total counts over cells with a positive total). ``row_sums`` is
    the per-cell library-size array; it is required (and only consumed) when
    ``"median"`` is requested.
    """
    if cpm_normalize and normalize_target_sum is not None:
        raise ValueError(
            "only one of cpm_normalize / normalize_target_sum may be set "
            "(cpm_normalize=True is equivalent to normalize_target_sum=1e6)."
        )
    if cpm_normalize:
        return 1.0e6
    if normalize_target_sum is None:
        return None
    if isinstance(normalize_target_sum, str):
        if normalize_target_sum != _MEDIAN:
            raise ValueError(
                f"normalize_target_sum string must be {_MEDIAN!r}, got "
                f"{normalize_target_sum!r}."
            )
        if row_sums is None:
            raise ValueError(
                "normalize_target_sum='median' requires per-cell row_sums."
            )
        positive = np.asarray(row_sums)[np.asarray(row_sums) > 0]
        if positive.size == 0:
            raise ValueError(
                "normalize_target_sum='median': no cell has positive total "
                "counts; cannot compute a median target."
            )
        return float(np.median(positive))
    # numeric
    if isinstance(normalize_target_sum, numbers.Real) and not isinstance(
            normalize_target_sum, bool):
        val = float(normalize_target_sum)
        if not np.isfinite(val) or val <= 0:
            raise ValueError(
                f"normalize_target_sum must be a finite positive number, "
                f"got {val!r}."
            )
        return val
    raise ValueError(
        "normalize_target_sum must be None, a positive number, or 'median'; "
        f"got {type(normalize_target_sum).__name__} {normalize_target_sum!r}."
    )


def cpm_rescale_factor(target_sum):
    """Constant that converts a target_sum-normalized mean into a 1e6-CPM mean:
    ``cpm_mean = norm_mean * cpm_rescale_factor(target_sum)``. Used by the
    filter_gene_min_cpm_cell gate when normalization is active."""
    return 1.0e6 / float(target_sum)
