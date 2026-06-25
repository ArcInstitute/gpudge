# tests/test_mwu.py
import numpy as np
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
    U_got = U_got.cpu().numpy()
    p_got = p_got.cpu().numpy()

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


# --- _tie_term_per_gene: Σ(t^3 - t) tie correction (GPU-free; CPU tensors) ---

def test_tie_term_per_gene_known_runs():
    from gpudge._mwu import _tie_term_per_gene
    # one gene; sorted run lengths 3,1,2 -> (27-3)+(1-1)+(8-2) = 24+0+6 = 30
    got = _tie_term_per_gene(torch.tensor([[0., 0., 0., 1., 2., 2.]]))
    assert got.shape == (1,)
    assert float(got[0]) == 30.0


def test_tie_term_per_gene_no_ties_is_zero():
    from gpudge._mwu import _tie_term_per_gene
    assert float(_tie_term_per_gene(torch.tensor([[1., 2., 3., 4.]]))[0]) == 0.0


def test_tie_term_per_gene_all_tied():
    from gpudge._mwu import _tie_term_per_gene
    # one run of 4 -> 4^3 - 4 = 60
    assert float(_tie_term_per_gene(torch.tensor([[5., 5., 5., 5.]]))[0]) == 60.0


def test_tie_term_per_gene_multi_gene():
    from gpudge._mwu import _tie_term_per_gene
    got = _tie_term_per_gene(torch.tensor([[0., 0., 1., 1.],    # 6 + 6 = 12
                                           [3., 3., 3., 3.]]))  # one run -> 60
    assert got.shape == (2,)
    assert [float(x) for x in got] == [12.0, 60.0]


def test_tie_term_per_gene_empty_inputs():
    # (0, k) and (n, 0) must return zeros, not raise (gpudge#ultrareview/Codex).
    from gpudge._mwu import _tie_term_per_gene
    assert _tie_term_per_gene(torch.empty(0, 5)).shape == (0,)
    assert [float(x) for x in _tie_term_per_gene(torch.empty(3, 0))] == [0.0, 0.0, 0.0]
