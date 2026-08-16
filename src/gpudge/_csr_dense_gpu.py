# src/gpudge/_csr_dense_gpu.py
"""Device-side CSR densify + row sums for the streaming DE device-decode path.

Imported LAZILY (only after ``is_cupy_csr(...)`` / ``_should_device_decode(...)``
is True), so CPU-only / no-cupy environments never import cupy. Mirrors the numba
host kernels in ``_csr_dense.py`` bit-for-bit on integer-count shard data: uint16
counts are exact in float32, and row sums are exact in float64 regardless of
reduction order (a shard row sum <= n_genes * 65535 << 2**53).
"""
from __future__ import annotations

import numpy as np
import torch


def cupy_csr_rows_col_range_to_torch(cupy_csr, rows, col_start, col_stop):
    """Dense ``(len(rows), col_stop-col_start)`` float32 torch tensor on device.

    Zero-copy DLPack hand-off from a cupy dense array. ``rows`` is a host int
    ndarray; a streaming group is a contiguous slice within its shard, so the
    fast major-axis slice is used when ``rows`` is contiguous (else fancy row
    indexing as a fallback).
    """
    import cupy as cp

    m = int(rows.shape[0])
    n_cols = int(col_stop) - int(col_start)
    # Match the device the shard's CSR actually lives on (robust under multi-GPU;
    # the non-empty result inherits this device via DLPack).
    device = torch.device("cuda", int(cupy_csr.data.device.id))
    if m == 0:
        return torch.empty((0, n_cols), dtype=torch.float32, device=device)
    r0 = int(rows[0])
    r1 = int(rows[-1]) + 1
    # Strict contiguity: r1-r0==m is necessary but not sufficient — np.diff==1 also
    # rules out duplicates / permutations that would make the [r0:r1] slice return
    # the wrong rows. gpudge's streaming groups are np.arange slices so this always
    # holds; the strict check keeps the helper correct if reused with other rows.
    if (r1 - r0) == m and bool((np.diff(rows) == 1).all()):
        block = cupy_csr[r0:r1, int(col_start):int(col_stop)]   # major-axis fast path
    else:                                                # arbitrary rows fallback
        block = cupy_csr[cp.asarray(rows)][:, int(col_start):int(col_stop)]
    dense = block.toarray()                              # device ndarray
    if dense.dtype != cp.float32:
        dense = dense.astype(cp.float32)
    # Pass the cupy array itself (not a raw capsule): torch.from_dlpack uses the
    # public __dlpack__ protocol, the version-robust zero-copy path (torch>=2.5).
    return torch.from_dlpack(dense)                      # zero-copy, on device


def cupy_csr_row_sums(cupy_csr) -> np.ndarray:
    """Per-row sums as a host float64 ndarray, bit-identical to numba csr_row_sums.

    Sums in float64 on device (integer counts sum EXACTLY in float64 regardless
    of reduction order), then copies the small ``(n_rows,)`` result to host.
    """
    import cupy as cp

    sums = cupy_csr.astype(cp.float64).sum(axis=1)       # (n_rows, 1) f64 on device
    return cp.asnumpy(sums).ravel().astype(np.float64, copy=False)
