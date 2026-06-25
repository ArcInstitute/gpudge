"""Tests for the optional numba CSR-row gather + scipy fallback."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from gpudge._csr_dense import (
    HAS_NUMBA,
    csr_rows_col_range_to_dense,
)


def _reference(X, rows, col_start, col_stop):
    """The scipy two-step expression we're matching."""
    return X[rows, col_start:col_stop].toarray().astype(np.float32, copy=False)


@pytest.mark.parametrize(
    "shape,density,seed",
    [
        ((1, 1), 1.0, 0),
        ((10, 20), 0.3, 1),
        ((100, 50), 0.1, 2),
        ((500, 200), 0.05, 3),
    ],
)
def test_matches_scipy_random(shape, density, seed):
    rng = np.random.default_rng(seed)
    n_rows, n_cols = shape
    dense = (rng.random(shape, dtype=np.float32)
             * (rng.random(shape) < density)).astype(np.float32)
    X = csr_matrix(dense)

    row_subsets = [
        np.arange(n_rows, dtype=np.int64),
        np.array([0], dtype=np.int64),
        rng.permutation(n_rows)[: max(1, n_rows // 3)].astype(np.int64),
    ]
    col_ranges = [
        (0, n_cols),
        (0, max(1, n_cols // 2)),
    ]
    if n_cols > 4:
        col_ranges.append((1, n_cols - 1))

    for rows in row_subsets:
        for cs, ce in col_ranges:
            got = csr_rows_col_range_to_dense(X, rows, cs, ce)
            want = _reference(X, rows, cs, ce)
            assert got.shape == want.shape, (
                f"shape mismatch rows={rows} cols=[{cs},{ce})")
            assert got.dtype == np.float32
            np.testing.assert_array_equal(got, want)


def test_duplicate_rows():
    rng = np.random.default_rng(7)
    X = csr_matrix(rng.random((20, 10), dtype=np.float32))
    rows = np.array([3, 3, 7, 0, 7, 0], dtype=np.int64)
    got = csr_rows_col_range_to_dense(X, rows, 0, 10)
    want = _reference(X, rows, 0, 10)
    np.testing.assert_array_equal(got, want)


def test_dense_input_passthrough():
    rng = np.random.default_rng(11)
    X = rng.random((10, 20), dtype=np.float32)
    rows = np.array([1, 5, 9], dtype=np.int64)
    got = csr_rows_col_range_to_dense(X, rows, 2, 12)
    expected = np.ascontiguousarray(X[rows, 2:12], dtype=np.float32)
    np.testing.assert_array_equal(got, expected)


@pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed; fast path inactive")
def test_numba_kernel_is_exercised():
    """When numba is available, the result must still match scipy."""
    rng = np.random.default_rng(42)
    dense = (rng.random((400, 300), dtype=np.float32)
             * (rng.random((400, 300)) < 0.05)).astype(np.float32)
    X = csr_matrix(dense)
    rows = rng.permutation(400)[:50].astype(np.int64)
    got = csr_rows_col_range_to_dense(X, rows, 50, 250)
    want = _reference(X, rows, 50, 250)
    np.testing.assert_array_equal(got, want)


@pytest.mark.skipif(not HAS_NUMBA, reason="out= path is numba-only")
def test_out_buffer_writes_into_view():
    """Passing a pre-allocated buffer makes the kernel write directly
    into the [:m, :n_cols] view, untouched outside. The returned array
    must be that view (not a fresh allocation) so the caller can keep
    the buffer pinned across iterations.
    """
    rng = np.random.default_rng(13)
    dense = (rng.random((200, 100), dtype=np.float32)
             * (rng.random((200, 100)) < 0.1)).astype(np.float32)
    X = csr_matrix(dense)
    rows = rng.permutation(200)[:40].astype(np.int64)
    # Over-sized buffer (max-row, max-col) with a sentinel outside the view.
    buf = np.full((100, 200), -7.0, dtype=np.float32)
    got = csr_rows_col_range_to_dense(X, rows, 10, 90, out=buf)
    # 1. content is correct for the active view
    want = _reference(X, rows, 10, 90)
    np.testing.assert_array_equal(got, want)
    # 2. returned array IS the view of the same buffer (no fresh alloc)
    assert got.base is buf
    assert got.shape == (40, 80)
    # 3. cells outside the active sub-view kept the sentinel
    assert (buf[40:, :] == -7.0).all()   # rows past m
    assert (buf[:40, 80:] == -7.0).all() # cols past n_cols
    # 4. cells inside the active sub-view that were "stale -7" before are
    #    correctly cleared (the kernel zeroes the view before writing).
    assert not (got == -7.0).any()


@pytest.mark.skipif(not HAS_NUMBA, reason="out= path is numba-only")
def test_out_buffer_rejects_bad_dtype_or_shape():
    rng = np.random.default_rng(14)
    X = csr_matrix(rng.random((10, 10), dtype=np.float32))
    rows = np.array([0, 1, 2], dtype=np.int64)
    # Wrong dtype
    with pytest.raises(ValueError, match=r"out\.dtype="):
        csr_rows_col_range_to_dense(
            X, rows, 0, 5, out=np.empty((3, 5), dtype=np.float64))
    # Buffer too small (rows)
    with pytest.raises(ValueError, match=r"out shape"):
        csr_rows_col_range_to_dense(
            X, rows, 0, 5, out=np.empty((2, 5), dtype=np.float32))
    # Buffer too small (cols)
    with pytest.raises(ValueError, match=r"out shape"):
        csr_rows_col_range_to_dense(
            X, rows, 0, 5, out=np.empty((3, 4), dtype=np.float32))


@pytest.mark.parametrize(
    "data_dtype,indices_dtype",
    [
        ("uint16", "int32"),     # h5ad_compression 'u16'
        ("uint16", "uint16"),    # h5ad_compression 'u16u16'
        ("int16",  "int32"),     # signed narrow data
        ("uint8",  "int32"),     # 8-bit counts
    ],
)
def test_narrow_dtypes_match_float32(data_dtype, indices_dtype):
    """uint16/int16 X.data + uint16 X.indices must produce float32 dense
    identical to the float32-baseline CSR. Mimics the in-memory state of
    a u16-loaded h5ad from the h5ad_compression session.
    """
    rng = np.random.default_rng(123)
    # Random count-like matrix with small integer values to fit uint16/uint8.
    counts = (rng.integers(0, 200, size=(200, 150))
              * (rng.random((200, 150)) < 0.1)).astype("int32")
    X_f32 = csr_matrix(counts.astype(np.float32))

    X_narrow = X_f32.copy()
    X_narrow.data = X_narrow.data.astype(data_dtype)
    X_narrow.indices = X_narrow.indices.astype(indices_dtype)

    rows = rng.permutation(200)[:60].astype(np.int64)
    got = csr_rows_col_range_to_dense(X_narrow, rows, 10, 130)
    want = _reference(X_f32, rows, 10, 130)
    assert got.dtype == np.float32
    np.testing.assert_array_equal(got, want)


# --- csr_row_sums (used by cpm_normalize; numba kernel + scipy/dense fallback) ---

def test_csr_row_sums_matches_dense_float32():
    from gpudge._csr_dense import csr_row_sums
    rng = np.random.default_rng(0)
    dense = rng.integers(0, 5, size=(6, 4)).astype(np.float32)
    dense[2] = 0.0  # a zero row
    got = csr_row_sums(csr_matrix(dense))
    assert got.dtype == np.float64
    np.testing.assert_allclose(got, dense.sum(axis=1))


def test_csr_row_sums_uint16_casts_to_float64():
    # narrow-dtype CSR: result must accumulate in float64 (scipy sum(axis=1) is
    # pathologically slow on uint16 — the perf reason csr_row_sums exists). Path
    # is numba when installed, scipy otherwise; both must give the same answer.
    from gpudge._csr_dense import csr_row_sums
    dense = np.array([[1, 2, 0], [0, 0, 0], [65000, 1, 2]], dtype=np.uint16)
    got = csr_row_sums(csr_matrix(dense))
    assert got.dtype == np.float64
    np.testing.assert_allclose(got, dense.astype(np.float64).sum(axis=1))


def test_csr_row_sums_dense_fallback():
    from gpudge._csr_dense import csr_row_sums
    dense = np.arange(12, dtype=np.float32).reshape(3, 4)
    got = csr_row_sums(dense)  # dense ndarray -> scipy .sum(axis=1) fallback
    np.testing.assert_allclose(got, dense.sum(axis=1))


# --- ultrareview regressions: non-canonical CSR + row bounds (numba path) ---

@pytest.mark.skipif(not HAS_NUMBA, reason="duplicate-index summing differs only on the numba path")
def test_noncanonical_duplicate_indices_are_summed():
    """A CSR with duplicate column indices within a row (non-canonical) must be
    SUMMED, matching scipy's .toarray(). The numba kernel previously did
    last-write-wins, so on the [fast] path the result silently differed from the
    scipy fallback. Regression for the ultrareview _csr_dense finding."""
    # Row 0: col 1 twice (2.0 + 3.0 = 5.0) + col 3 once (4.0). Row 1: col 0 twice.
    data = np.array([2.0, 3.0, 4.0, 1.0, 1.0], dtype=np.float32)
    indices = np.array([1, 1, 3, 0, 0], dtype=np.int32)
    indptr = np.array([0, 3, 5], dtype=np.int32)
    X = csr_matrix((data, indices, indptr), shape=(2, 4))
    assert not X.has_canonical_format            # genuinely non-canonical
    rows = np.array([0, 1], dtype=np.int64)
    want = X.toarray().astype(np.float32)        # scipy sums duplicates
    got = csr_rows_col_range_to_dense(X, rows, 0, 4)
    np.testing.assert_array_equal(got, want)
    # the pre-allocated (pinned) out= variant must sum too
    buf = np.full((4, 4), -1.0, dtype=np.float32)
    got2 = csr_rows_col_range_to_dense(X, rows, 0, 4, out=buf)
    np.testing.assert_array_equal(got2, want)


@pytest.mark.skipif(not HAS_NUMBA, reason="bounds guard protects the boundscheck=False numba kernel")
def test_out_of_range_row_raises():
    """Row indices outside [0, n_rows) must raise IndexError (scipy's behaviour)
    rather than reading indptr out of bounds under boundscheck=False. Regression
    for the ultrareview _csr_dense finding."""
    X = csr_matrix(np.eye(5, dtype=np.float32))
    with pytest.raises(IndexError, match="row index"):
        csr_rows_col_range_to_dense(X, np.array([0, 5], dtype=np.int64), 0, 5)
    with pytest.raises(IndexError, match="row index"):
        csr_rows_col_range_to_dense(X, np.array([-1], dtype=np.int64), 0, 5)
