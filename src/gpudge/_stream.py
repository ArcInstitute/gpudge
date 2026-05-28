# src/gpudge/_stream.py
"""Iterate (start_col, dense_torch_tensor_on_device) chunks across genes.

Accepts dense numpy ndarray or scipy.sparse CSR. For sparse, the conversion to
dense happens per-chunk on the GPU side so the full dense matrix is never on
the host.
"""
from __future__ import annotations

from typing import Iterator
import numpy as np
import scipy.sparse as sp
import torch


def iter_gene_chunks(
    X,                          # ndarray (cells, genes) or CSR
    *,
    chunk_size: int,
    device: str | torch.device = "cuda",
) -> Iterator[tuple[int, torch.Tensor]]:
    """Yield (start_col, chunk_tensor) for col-blocks of size `chunk_size`."""
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
    n_cells, n_genes = X.shape
    for start in range(0, n_genes, chunk_size):
        stop = min(start + chunk_size, n_genes)
        if sp.issparse(X):
            # Slice CSR by columns → CSC for efficiency, then to dense
            block_csc = X[:, start:stop].tocsc()
            t = torch.from_numpy(block_csc.toarray().astype(np.float32, copy=False))
        else:
            t = torch.from_numpy(np.ascontiguousarray(X[:, start:stop],
                                                     dtype=np.float32))
        yield start, t.to(device, non_blocking=True)
