# tests/test_means.py
import numpy as np
import pytest
import torch
from gpudge._means import group_means
from conftest import needs_cuda


def _gold_arith(X, labels, n_groups):
    out = np.zeros((n_groups, X.shape[1]), dtype=np.float64)
    for g in range(n_groups):
        m = labels == g
        if m.sum() == 0:
            continue
        out[g] = X[m].mean(axis=0)
    return out


def _gold_geom(X, labels, n_groups):
    out = np.zeros((n_groups, X.shape[1]), dtype=np.float64)
    for g in range(n_groups):
        m = labels == g
        if m.sum() == 0:
            continue
        out[g] = np.expm1(np.log1p(X[m]).mean(axis=0))
    return out


@needs_cuda
def test_arith_means_match_numpy_dense():
    rng = np.random.default_rng(0)
    n_cells, n_genes, n_groups = 200, 50, 5
    X = rng.exponential(1.0, (n_cells, n_genes)).astype(np.float32)
    labels = rng.integers(0, n_groups, n_cells).astype(np.int32)
    X_t = torch.from_numpy(X).cuda()
    labels_t = torch.from_numpy(labels).cuda()
    got = group_means(X_t, labels_t, n_groups, kind="arithmetic").cpu().numpy()
    np.testing.assert_allclose(got, _gold_arith(X, labels, n_groups), rtol=1e-5, atol=1e-6)


@needs_cuda
def test_geom_means_match_numpy_dense():
    rng = np.random.default_rng(0)
    n_cells, n_genes, n_groups = 200, 50, 5
    X = rng.exponential(1.0, (n_cells, n_genes)).astype(np.float32)
    labels = rng.integers(0, n_groups, n_cells).astype(np.int32)
    X_t = torch.from_numpy(X).cuda()
    labels_t = torch.from_numpy(labels).cuda()
    got = group_means(X_t, labels_t, n_groups, kind="geometric").cpu().numpy()
    np.testing.assert_allclose(got, _gold_geom(X, labels, n_groups), rtol=1e-4, atol=1e-5)


@needs_cuda
def test_unknown_kind_raises():
    X = torch.zeros((10, 5), device="cuda")
    labels = torch.zeros(10, dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match="kind"):
        group_means(X, labels, 1, kind="harmonic")
