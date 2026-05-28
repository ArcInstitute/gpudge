# tests/test_stream.py
import numpy as np
import pytest
import scipy.sparse as sp
import torch
from gpudge._stream import iter_gene_chunks
from conftest import needs_cuda


@needs_cuda
def test_dense_chunks_reassemble_full_matrix():
    X = np.arange(60, dtype=np.float32).reshape(5, 12)
    chunks = list(iter_gene_chunks(X, chunk_size=4, device="cuda"))
    got = torch.cat([c for _, c in chunks], dim=1).cpu().numpy()
    np.testing.assert_array_equal(got, X)
    starts = [s for s, _ in chunks]
    assert starts == [0, 4, 8]


@needs_cuda
def test_sparse_chunks_reassemble_full_matrix():
    X_dense = np.arange(60, dtype=np.float32).reshape(5, 12)
    X = sp.csr_matrix(X_dense)
    chunks = list(iter_gene_chunks(X, chunk_size=5, device="cuda"))
    got = torch.cat([c for _, c in chunks], dim=1).cpu().numpy()
    np.testing.assert_array_equal(got, X_dense)


@needs_cuda
def test_chunk_size_exceeds_n_genes_yields_one_chunk():
    X = np.zeros((3, 5), dtype=np.float32)
    chunks = list(iter_gene_chunks(X, chunk_size=100, device="cuda"))
    assert len(chunks) == 1
