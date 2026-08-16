# tests/test_csr_dense_gpu.py
"""Focused unit tests for the device CSR densify + row-sum helpers.

All needs_cuda (they build cupy device matrices): they skip on a CPU box and run
on an H100. They mirror the numba host kernels bit-for-bit on integer-count data
and cover the contiguous fast path, the empty-rows guard, the non-contiguous
fancy-index fallback, and the float64 row sums.
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from conftest import needs_cuda


@needs_cuda
def test_device_densify_matches_numba_contiguous_rows():
    cxs = pytest.importorskip("cupyx.scipy.sparse",
                              reason="requires cupy (gpudge[streaming-gpu])")
    from gpudge._csr_dense import csr_rows_col_range_to_dense
    from gpudge._csr_dense_gpu import cupy_csr_rows_col_range_to_torch
    rng = np.random.default_rng(0)
    X = sp.csr_matrix(rng.integers(0, 50, size=(30, 20)).astype(np.float32))
    Xd = cxs.csr_matrix(X)                       # upload to device
    rows = np.arange(5, 18, dtype=np.int64)      # contiguous (streaming group)
    host = csr_rows_col_range_to_dense(X, rows, 3, 11)          # numba float32
    dev = cupy_csr_rows_col_range_to_torch(Xd, rows, 3, 11)
    assert str(dev.dtype) == "torch.float32"
    assert dev.is_cuda
    np.testing.assert_array_equal(host, dev.cpu().numpy())


@needs_cuda
def test_device_densify_empty_rows():
    import torch

    cxs = pytest.importorskip("cupyx.scipy.sparse",
                              reason="requires cupy (gpudge[streaming-gpu])")
    from gpudge._csr_dense_gpu import cupy_csr_rows_col_range_to_torch
    X = sp.csr_matrix(np.arange(30, dtype=np.float32).reshape(6, 5))
    Xd = cxs.csr_matrix(X)
    out = cupy_csr_rows_col_range_to_torch(Xd, np.empty(0, dtype=np.int64), 1, 4)
    assert out.shape == (0, 3)
    assert out.dtype == torch.float32
    assert out.is_cuda


@needs_cuda
def test_device_densify_noncontiguous_rows_matches_numba():
    # rows out of order + a duplicate -> exercises the fancy-index fallback (not
    # the contiguous slice). Both host numba + device gather rows independently,
    # so duplicates/order match exactly.
    cxs = pytest.importorskip("cupyx.scipy.sparse",
                              reason="requires cupy (gpudge[streaming-gpu])")
    from gpudge._csr_dense import csr_rows_col_range_to_dense
    from gpudge._csr_dense_gpu import cupy_csr_rows_col_range_to_torch
    rng = np.random.default_rng(2)
    X = sp.csr_matrix(rng.integers(0, 40, size=(20, 12)).astype(np.float32))
    Xd = cxs.csr_matrix(X)
    rows = np.array([9, 2, 2, 15], dtype=np.int64)
    host = csr_rows_col_range_to_dense(X, rows, 1, 8)
    dev = cupy_csr_rows_col_range_to_torch(Xd, rows, 1, 8)
    np.testing.assert_array_equal(host, dev.cpu().numpy())


@needs_cuda
def test_device_row_sums_match_numba_float64():
    cxs = pytest.importorskip("cupyx.scipy.sparse",
                              reason="requires cupy (gpudge[streaming-gpu])")
    from gpudge._csr_dense import csr_row_sums
    from gpudge._csr_dense_gpu import cupy_csr_row_sums
    rng = np.random.default_rng(1)
    X = sp.csr_matrix(rng.integers(0, 500, size=(40, 25)).astype(np.float32))
    Xd = cxs.csr_matrix(X)
    host = csr_row_sums(X)                        # numba float64
    dev = cupy_csr_row_sums(Xd)
    assert dev.dtype == np.float64
    np.testing.assert_array_equal(host, dev)      # exact (integer counts)
