# tests/test_mwu.py
import numpy as np
import pytest
import torch
from scipy.stats import mannwhitneyu
from gpudge._mwu import mwu_ref
from conftest import needs_cuda


def _scipy_per_gene_ref(X: np.ndarray, labels: np.ndarray, n_groups: int,
                        ref_idx: int):
    """Per-(group, gene) scipy MWU (two-sided, with ties)."""
    n_genes = X.shape[1]
    U = np.zeros((n_groups, n_genes), dtype=np.float64)
    p = np.ones((n_groups, n_genes), dtype=np.float64)
    ref_mask = labels == ref_idx
    ref_X = X[ref_mask]
    for g in range(n_groups):
        if g == ref_idx:
            continue
        gm = labels == g
        if gm.sum() == 0:
            continue
        gX = X[gm]
        for j in range(n_genes):
            res = mannwhitneyu(gX[:, j], ref_X[:, j], alternative="two-sided",
                               method="asymptotic")
            U[g, j] = res.statistic
            p[g, j] = res.pvalue
    return U, p


@needs_cuda
def test_mwu_ref_matches_scipy_on_synthetic():
    """Equivalence: torch MWU == scipy MWU on a small synthetic example."""
    rng = np.random.default_rng(0)
    n_cells, n_genes, n_groups = 200, 25, 4
    X = rng.negative_binomial(2, 0.5, (n_cells, n_genes)).astype(np.float32)
    labels = rng.integers(0, n_groups, n_cells).astype(np.int32)
    ref_idx = 0

    X_t = torch.from_numpy(X).cuda()
    labels_t = torch.from_numpy(labels).cuda()
    U_got, p_got = mwu_ref(X_t, labels_t, n_groups, ref_idx=ref_idx)
    U_got = U_got.cpu().numpy(); p_got = p_got.cpu().numpy()

    U_exp, p_exp = _scipy_per_gene_ref(X, labels, n_groups, ref_idx)

    # ref row is zeros/ones; ignore it
    mask = np.arange(n_groups) != ref_idx
    np.testing.assert_allclose(U_got[mask], U_exp[mask], rtol=0, atol=0.5)
    # p-values: relative agreement to ~1% (Normal approximation, ties)
    np.testing.assert_allclose(p_got[mask], p_exp[mask], rtol=1e-3, atol=1e-6)


@needs_cuda
def test_mwu_ref_ignores_ref_row():
    """Output for ref_idx row should be sentinel (NaN p, U=0)."""
    rng = np.random.default_rng(1)
    n_cells, n_genes, n_groups, ref_idx = 100, 10, 3, 1
    X = rng.exponential(1.0, (n_cells, n_genes)).astype(np.float32)
    labels = rng.integers(0, n_groups, n_cells).astype(np.int32)
    X_t = torch.from_numpy(X).cuda()
    labels_t = torch.from_numpy(labels).cuda()
    U, p = mwu_ref(X_t, labels_t, n_groups, ref_idx=ref_idx)
    assert torch.isnan(p[ref_idx]).all()
