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


def _assert_mwu_ref_matches_scipy(device):
    """Equivalence: torch MWU == scipy MWU on a small synthetic example.
    mwu_ref reads X.device, so this runs identically on CPU and CUDA."""
    rng = np.random.default_rng(0)
    n_cells, n_genes, n_groups = 200, 25, 4
    X = rng.negative_binomial(2, 0.5, (n_cells, n_genes)).astype(np.float32)
    labels = rng.integers(0, n_groups, n_cells).astype(np.int32)
    ref_idx = 0

    X_t = torch.from_numpy(X).to(device)
    labels_t = torch.from_numpy(labels).to(device)
    U_got, p_got = mwu_ref(X_t, labels_t, n_groups, ref_idx=ref_idx)
    U_got = U_got.cpu().numpy()
    p_got = p_got.cpu().numpy()

    U_exp, p_exp = _scipy_per_gene_ref(X, labels, n_groups, ref_idx)

    # ref row is zeros/ones; ignore it
    mask = np.arange(n_groups) != ref_idx
    np.testing.assert_allclose(U_got[mask], U_exp[mask], rtol=0, atol=0.5)
    # p-values: relative agreement to ~1% (Normal approximation, ties)
    np.testing.assert_allclose(p_got[mask], p_exp[mask], rtol=1e-3, atol=1e-6)


def test_mwu_ref_matches_scipy_cpu():
    """CPU-runnable equivalence so the core numerics (tie/continuity/variance)
    are exercised on the CPU-only CI, not only the @needs_cuda path (M1)."""
    _assert_mwu_ref_matches_scipy("cpu")


@needs_cuda
def test_mwu_ref_matches_scipy_cuda():
    _assert_mwu_ref_matches_scipy("cuda")


def _assert_mwu_ref_sentinel(device):
    """ref_idx row is the documented sentinel: U == 0 and p == NaN."""
    rng = np.random.default_rng(1)
    n_cells, n_genes, n_groups, ref_idx = 100, 10, 3, 1
    X = rng.exponential(1.0, (n_cells, n_genes)).astype(np.float32)
    labels = rng.integers(0, n_groups, n_cells).astype(np.int32)
    X_t = torch.from_numpy(X).to(device)
    labels_t = torch.from_numpy(labels).to(device)
    U, p = mwu_ref(X_t, labels_t, n_groups, ref_idx=ref_idx)
    assert torch.isnan(p[ref_idx]).all()
    assert (U[ref_idx] == 0).all()   # N4: U=0 half of the sentinel was untested


def test_mwu_ref_sentinel_cpu():
    _assert_mwu_ref_sentinel("cpu")


@needs_cuda
def test_mwu_ref_sentinel_cuda():
    _assert_mwu_ref_sentinel("cuda")


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


def test_tie_term_per_gene_multi_block(monkeypatch):
    """L10: force the block loop (block < n_genes) and assert it matches the
    single-block result. Normally block >= n_genes (the 64M-element budget), so
    the loop always ran once and the stride/run_id-reset/out-slice path was
    never exercised."""
    import gpudge._mwu as _mwu
    rng = np.random.default_rng(7)
    x = torch.from_numpy(rng.integers(0, 3, (6, 8)).astype(np.float64))  # tie-heavy
    x, _ = torch.sort(x, dim=1)                       # function expects sorted rows
    full = _mwu._tie_term_per_gene(x)                 # single block (default budget)
    monkeypatch.setattr(_mwu, "_TIE_BLOCK_ELEMS", 8)  # 8 // k(=8) -> 1 gene/block
    multi = _mwu._tie_term_per_gene(x)                # multi-block path (6 blocks)
    assert torch.allclose(full, multi)


def test_tie_term_per_gene_empty_inputs():
    # (0, k) and (n, 0) must return zeros, not raise (gpudge#ultrareview/Codex).
    from gpudge._mwu import _tie_term_per_gene
    assert _tie_term_per_gene(torch.empty(0, 5)).shape == (0,)
    assert [float(x) for x in _tie_term_per_gene(torch.empty(3, 0))] == [0.0, 0.0, 0.0]


def test_helpers_reproduce_mwu_one_group_exactly():
    """The extracted helpers must recompose into mwu_one_group bit-for-bit."""
    import torch
    from gpudge._mwu import (
        _bounds, _cross_tie_delta, _p_two_sided, _selfties,
        _sorted_and_selfties, _tie_term_per_gene, _u1_against, mwu_one_group,
    )
    rng = np.random.default_rng(7)
    n_genes, n_ref, m = 12, 40, 17
    ref = rng.negative_binomial(2, 0.4, (n_genes, n_ref)).astype(np.float32)
    grp = rng.negative_binomial(2, 0.4, (n_genes, m)).astype(np.float32)
    sorted_ref = torch.from_numpy(np.sort(ref, axis=1))
    group_T = torch.from_numpy(grp)
    ref_tie = _tie_term_per_gene(sorted_ref)

    u1_exp, p_exp = mwu_one_group(sorted_ref, ref_tie, group_T, n_ref=n_ref)

    group_sorted, gc, run_start = _sorted_and_selfties(group_T)
    u1_got = _u1_against(sorted_ref, group_T)
    tie = ref_tie + _cross_tie_delta(sorted_ref, group_sorted, gc, run_start)
    p_got = _p_two_sided(u1_got, tie, m, n_ref)

    assert torch.equal(u1_got, u1_exp)
    assert torch.equal(p_got, p_exp)

    # _selfties on an already-sorted block == the composition's tail
    gc2, rs2 = _selfties(group_sorted)
    assert torch.equal(gc2, gc) and torch.equal(rs2, run_start)

    # _bounds on same-dtype inputs is exactly torch.searchsorted
    lo, hi = _bounds(sorted_ref, group_T)
    assert torch.equal(lo, torch.searchsorted(sorted_ref, group_T, right=False))
    assert torch.equal(hi, torch.searchsorted(sorted_ref, group_T, right=True))


def _bounds_cases():
    """(ref32, values64, label) covering the whole float32 domain + the edges."""
    import numpy as np
    f32 = np.float32
    rng = np.random.default_rng(17)
    out = []
    # random SIGNED finite float32 bit patterns (spec 3.4a domain = either
    # sign), random tau in [-30, 30], both sides. Magnitudes are drawn below the
    # inf/nan exponent (0x7F800000) and given a random sign bit, so no NaN can
    # appear -- NaN violates searchsorted's sorted-sequence contract and has no
    # MWU ordering.
    for trial in range(8):
        def _signed(shape):
            mag = rng.integers(0, 0x7F800000, shape, dtype=np.uint32)
            sign = rng.integers(0, 2, shape, dtype=np.uint32) << 31
            return (mag | sign).view(np.float32)
        ref = _signed((2, 4000))
        tgt = _signed((2, 500))
        assert not np.isnan(ref).any() and not np.isnan(tgt).any()
        s = 2.0 ** (rng.random() * 60 - 30)
        out.append((np.sort(ref, axis=1).copy(), tgt.astype(np.float64) * s,
                    f"signed bit patterns #{trial} s={s:.3e}"))
    # signed zeros, MIN and MAX subnormal, min normal, max finite, +/-inf
    tiny = np.finfo(np.float32).tiny
    max_subnormal = np.nextafter(f32(tiny), f32(0.0))       # largest subnormal
    edge = np.array([[-np.inf, -1.0, -0.0, 0.0, f32(1.4e-45), max_subnormal,
                      tiny, 1.0, 2.0, f32(3.4028235e38),
                      np.inf]], dtype=np.float32)
    edge = np.sort(edge, axis=1)
    for tau in (0.0, 0.25, 1.0, 30.0):
        for s in (2.0 ** -tau, 2.0 ** tau):
            out.append((edge, edge.astype(np.float64) * s, f"edges tau={tau} s={s:g}"))
    # exact ties, and values straddling q on both sides
    ties = np.sort(rng.integers(1, 200, (1, 300)).astype(np.float32), axis=1)
    out.append((ties, ties[:, ::7].astype(np.float64), "exact ties, s=1"))
    q = ties[:, ::7].astype(np.float64)
    out.append((ties, np.nextafter(q, np.inf), "just above q"))
    out.append((ties, np.nextafter(q, -np.inf), "just below q"))
    # the spec 3.2b boundary counterexample
    r, t = f32(2.5243635e-05), f32(3.569989e-05)
    out.append((np.full((1, 100), r, dtype=np.float32),
                np.full((1, 100), t, dtype=np.float32).astype(np.float64) * (2.0 ** -0.5),
                "spec 3.2b boundary counterexample"))
    return out


def _assert_bounds_matches_native(device):
    """_bounds must be bit-identical to the native (UPCASTING) mixed-dtype
    torch.searchsorted. The native call is the oracle, not the implementation:
    it copies the whole boundary to float64 (0.05 ms -> 65.26 ms on a 160 MB
    reference), which is exactly what _bounds exists to avoid."""
    import torch
    from gpudge._mwu import _bounds
    for ref_np, val_np, label in _bounds_cases():
        ref = torch.from_numpy(ref_np).to(device)
        val = torch.from_numpy(val_np).to(device)
        lo, hi = _bounds(ref, val)
        assert torch.equal(lo, torch.searchsorted(ref, val, right=False)), label
        assert torch.equal(hi, torch.searchsorted(ref, val, right=True)), label


def test_bounds_matches_native_mixed_dtype_cpu():
    _assert_bounds_matches_native("cpu")


@needs_cuda
def test_bounds_matches_native_mixed_dtype_cuda():
    """Run on CUDA too -- this is a claim about rounding, and the CPU and CUDA
    searchsorted kernels are independent implementations."""
    _assert_bounds_matches_native("cuda")
