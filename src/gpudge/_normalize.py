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
    the string ``"median"``: the median of per-cell total counts **over cells
    with a positive total**. ``row_sums`` is the per-cell library-size array; it
    is required (and only consumed) when ``"median"`` is requested.

    On ``"median"`` vs scanpy: this matches scanpy's *dense/Dask* branch and its
    internal ``_compute_nnz_median`` helper. Its *CSR* branch instead medians
    over ALL cells, empty ones included (plain ``np.median(counts_per_cell)``)
    — so the two *can* differ when zero-total cells are present. They need not:
    row sums ``[0, 10, 10, 20]`` give 10 either way. Since gpudge's sparse paths
    use CSR, a caller comparing against scanpy on the same object may see a
    different target. That is **not** a purely cosmetic difference — see the
    ``de()`` docstring for what moves.

    That CSR/dense split affects scanpy 1.11.2–1.12.3 and the 1.13.0a1
    prerelease (cut before the fix merged); versions before 1.11.2 never had
    it, using the positive-cell rule on every path. The fix —
    scverse/scanpy#4256, which puts the CSR branch on ``_compute_nnz_median``
    too, converging on the rule implemented here — is merged on scanpy
    ``main`` and backported to the ``1.12.x`` branch, targeted at 1.12.4, but
    **no scanpy release carries it as of 1.12.3**. Pinned by
    ``tests/test_scanpy_median_contract.py``.
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
