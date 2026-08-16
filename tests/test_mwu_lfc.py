# tests/test_mwu_lfc.py
"""Directional (lfc_threshold) MWU kernel: scipy parity, ties, float64 scaling.

The shift is applied to the TARGET (spec 3.2a) in FLOAT64 (spec 3.2b), so every
oracle here scales the target, leaves the reference alone, and works in float64.
"""
import numpy as np
import pytest
import torch
from scipy.stats import mannwhitneyu

from gpudge._lfc import lfc_scale_factor
from gpudge._mwu import (
    _cross_tie_delta, _p_one_sided, _selfties, _sorted_and_selfties,
    _tie_term_per_gene, _u1_against, mwu_one_group, mwu_one_group_lfc,
)


def _fixture(seed=3, n_genes=8, n_ref=45, m=19):
    rng = np.random.default_rng(seed)
    ref = rng.negative_binomial(2, 0.4, (n_genes, n_ref)).astype(np.float32)
    grp = rng.negative_binomial(2, 0.4, (n_genes, m)).astype(np.float32)
    sorted_ref = torch.from_numpy(np.sort(ref, axis=1))
    return ref, grp, sorted_ref, torch.from_numpy(grp), n_ref, m


def test_base_output_is_bit_identical_to_mwu_one_group():
    ref, grp, sorted_ref, group_T, n_ref, m = _fixture()
    ref_tie = _tie_term_per_gene(sorted_ref)
    u1_exp, p_exp = mwu_one_group(sorted_ref, ref_tie, group_T, n_ref=n_ref)
    u1, p, du, dp = mwu_one_group_lfc(
        sorted_ref, ref_tie, group_T, n_ref=n_ref,
        lfc_combos=((0.25, "up"), (0.25, "down")))
    assert torch.equal(u1, u1_exp)
    assert torch.equal(p, p_exp)
    assert du.shape == (2, ref.shape[0])
    assert dp.shape == (2, ref.shape[0])


@pytest.mark.parametrize("tau", [0.0, 0.25, 0.5, 1.0])
def test_matches_scipy_one_sided_per_direction(tau):
    """Oracle: everything in float64, matching spec 3.2b.

    The kernel promotes the target to float64 and multiplies by a float64
    factor, so the oracle must do exactly the same. scipy computes its rank sums
    and tie correction in the INPUT dtype: float32 inputs agree to only ~1.8e-7
    relative, float64 to ~7.7e-16 (measured). The factor comes from
    lfc_scale_factor so oracle and kernel cannot drift by an ULP.
    """
    ref, grp, sorted_ref, group_T, n_ref, m = _fixture()
    ref_tie = _tie_term_per_gene(sorted_ref)
    _, _, _, dp = mwu_one_group_lfc(
        sorted_ref, ref_tie, group_T, n_ref=n_ref,
        lfc_combos=((tau, "up"), (tau, "down")))
    s_up = lfc_scale_factor(tau, "up")
    s_dn = lfc_scale_factor(tau, "down")
    ref64 = ref.astype(np.float64)
    grp64 = grp.astype(np.float64)
    for j in range(ref.shape[0]):
        up_exp = mannwhitneyu(grp64[j] * s_up, ref64[j],
                              alternative="greater", use_continuity=True,
                              method="asymptotic").pvalue
        dn_exp = mannwhitneyu(grp64[j] * s_dn, ref64[j],
                              alternative="less", use_continuity=True,
                              method="asymptotic").pvalue
        assert dp[0, j].item() == pytest.approx(up_exp, rel=1e-12, abs=1e-300)
        assert dp[1, j].item() == pytest.approx(dn_exp, rel=1e-12, abs=1e-300)


def test_tau_zero_two_one_sided_recombine_to_two_sided():
    """clamp(2*min(p_up, p_down), max=1) == the base two-sided p, exactly.

    greaterAbs is deliberately NOT exposed, but this identity validates the
    one-sided machinery against the already-scipy-validated two-sided path. It
    is EXACT only because the factor at tau=0 is exactly 1.0 AND U1 is summed
    over group_T (base layout), not group_sorted.
    """
    ref, grp, sorted_ref, group_T, n_ref, m = _fixture()
    ref_tie = _tie_term_per_gene(sorted_ref)
    _, p_two, _, dp = mwu_one_group_lfc(
        sorted_ref, ref_tie, group_T, n_ref=n_ref,
        lfc_combos=((0.0, "up"), (0.0, "down")))
    recombined = (2.0 * torch.minimum(dp[0], dp[1])).clamp_max(1.0)
    assert torch.equal(recombined, p_two)


def test_tau_zero_recombination_extreme_tail_within_one_ulp():
    """Complete separation drives 0.5*erfc toward the subnormal range, where
    the exact identity above can lose an ULP. Allow one ULP there."""
    n_genes, n_ref, m = 2, 900, 900
    ref = np.zeros((n_genes, n_ref), dtype=np.float32)
    grp = np.full((n_genes, m), 1000.0, dtype=np.float32)
    sorted_ref = torch.from_numpy(ref)
    ref_tie = _tie_term_per_gene(sorted_ref)
    _, p_two, _, dp = mwu_one_group_lfc(
        sorted_ref, ref_tie, torch.from_numpy(grp), n_ref=n_ref,
        lfc_combos=((0.0, "up"), (0.0, "down")))
    recombined = (2.0 * torch.minimum(dp[0], dp[1])).clamp_max(1.0)
    np.testing.assert_allclose(recombined.numpy(), p_two.numpy(),
                               rtol=2 ** -52, atol=5e-324)


def test_U1_monotone_in_tau():
    """Monotonicity of the STATISTIC, not the p-value.

    U1 is non-increasing in tau for `up` (larger tau scales the target further
    down) and non-decreasing for `down`. The p-value is NOT monotone: the tie
    correction changes discontinuously, and with T = R = [4, 4] p_up is 1.0 at
    tau=0 but 0.98480859 for ANY tau > 0 (measured). See the next test.
    """
    ref, grp, sorted_ref, group_T, n_ref, m = _fixture()
    ref_tie = _tie_term_per_gene(sorted_ref)
    taus = [0.0, 0.25, 0.5, 1.0, 2.0]
    for direction, cmp in (("up", np.less_equal), ("down", np.greater_equal)):
        combos = tuple((t, direction) for t in taus)
        _, _, du, _ = mwu_one_group_lfc(sorted_ref, ref_tie, group_T,
                                        n_ref=n_ref, lfc_combos=combos)
        for i in range(len(taus) - 1):
            assert np.all(cmp(du[i + 1].numpy(), du[i].numpy())), direction


def test_p_is_NOT_monotone_in_tau_at_a_tie_transition():
    """Pins the known counterexample so nobody 'fixes' it or promises monotone
    p in the docstring. T = R = [4, 4]: at tau=0 every value ties, the variance
    collapses to the clamp and p_up == 1.0; at any tau > 0 the tie vanishes,
    U1 drops 2 -> 0 and p_up == 0.98480859. p DECREASES as tau grows."""
    sorted_ref = torch.tensor([[4.0, 4.0]], dtype=torch.float32)
    ref_tie = _tie_term_per_gene(sorted_ref)
    group_T = torch.tensor([[4.0, 4.0]], dtype=torch.float32)
    _, _, du, dp = mwu_one_group_lfc(
        sorted_ref, ref_tie, group_T, n_ref=2,
        lfc_combos=((0.0, "up"), (1e-6, "up")))
    assert du[0].item() == 2.0 and du[1].item() == 0.0
    assert dp[0].item() == pytest.approx(1.0, abs=1e-15)
    assert dp[1].item() == pytest.approx(0.98480859, rel=1e-7)
    assert dp[1].item() < dp[0].item()          # NOT monotone


# ---- spec 3.2b / 3.4a: float64 scaling, and what it buys ------------------

def _merging_float32_row(fac32, seed=11):
    """A row of adjacent float32 pairs that provably merge under float32 `fac32`."""
    f32 = np.float32
    rng = np.random.default_rng(seed)
    vals = []
    for exp in range(-8, 9):
        cand = f32(np.ldexp(1.0, exp) * (1.0 + rng.random(20000)))
        nxt = np.nextafter(cand, f32(np.inf)).astype(np.float32)
        hit = (cand != nxt) & ((cand * fac32) == (nxt * fac32))
        for v in cand[hit][:3]:
            vals.extend([v, np.nextafter(v, f32(np.inf))])
    assert len(vals) >= 20, "fixture failed to find merging pairs"
    return np.sort(np.array(vals, dtype=np.float32))[None, :]


def test_float64_scaling_is_injective_where_float32_is_not():
    """THE test that fails if anyone reverts spec 3.2b to float32 scaling.

    Float32 scaling collapses this adversarial row (measured: 102 -> 51 distinct
    at tau=0.5, tie term 0 -> 306). Float64 scaling cannot merge adjacent float32
    values at all: they differ by >= 2**-24 relative and float64 rounding
    perturbs by <= 2**-53.
    """
    s = lfc_scale_factor(0.5, "down")                    # 2**+0.5, float64
    row = _merging_float32_row(np.float32(s))
    t32 = torch.from_numpy(row)
    n_unscaled = len(torch.unique(t32))
    assert len(torch.unique(t32 * np.float32(s))) < n_unscaled, \
        "fixture is not adversarial for float32 scaling"
    assert len(torch.unique(t32.to(torch.float64) * s)) == n_unscaled


@pytest.mark.parametrize("tau", [0.25, 0.5, 0.7, 3.3, 30.0])
@pytest.mark.parametrize("direction", ["up", "down"])
def test_target_selfties_are_tau_invariant_under_float64_scaling(tau, direction):
    """Pins spec 3.4a/3.4c -- what licenses computing gc/run_start ONCE. If this
    ever fails they must be recomputed per combo inside the kernel."""
    row = _merging_float32_row(np.float32(2.0 ** 0.5))
    gs64 = torch.from_numpy(row).to(torch.float64)
    gc0, rs0 = _selfties(gs64)
    gc1, rs1 = _selfties(gs64 * lfc_scale_factor(tau, direction))
    assert torch.equal(gc0, gc1) and torch.equal(rs0, rs1)


def test_kernel_on_the_adversarial_float32_fixture_matches_a_float64_oracle():
    """Drives the adversarial fixture through the KERNEL, not just _selfties --
    a kernel that mishandled the scaled target would pass a helper-only test."""
    row = _merging_float32_row(np.float32(2.0 ** 0.5))
    grp = row                                   # (1, k) target
    ref = np.sort(row[0][::3])[None, :].copy()  # a coarser reference from the same values
    sorted_ref = torch.from_numpy(ref)
    ref_tie = _tie_term_per_gene(sorted_ref)
    n_ref = ref.shape[1]
    tau = 0.5
    _, _, _, dp = mwu_one_group_lfc(
        sorted_ref, ref_tie, torch.from_numpy(grp), n_ref=n_ref,
        lfc_combos=((tau, "up"), (tau, "down")))
    g64, r64 = grp[0].astype(np.float64), ref[0].astype(np.float64)
    up = mannwhitneyu(g64 * lfc_scale_factor(tau, "up"), r64,
                      alternative="greater", use_continuity=True,
                      method="asymptotic").pvalue
    dn = mannwhitneyu(g64 * lfc_scale_factor(tau, "down"), r64,
                      alternative="less", use_continuity=True,
                      method="asymptotic").pvalue
    assert dp[0, 0].item() == pytest.approx(up, rel=1e-12, abs=1e-300)
    assert dp[1, 0].item() == pytest.approx(dn, rel=1e-12, abs=1e-300)


def test_kernel_never_calls_tie_term_per_gene(monkeypatch):
    """Structural pin for spec 3.4a: the reference is NEVER rebuilt, so the
    directional kernel must not touch _tie_term_per_gene at all.

    A value-comparison test is NOT sufficient here: on an integer-count fixture
    an accidental _tie_term_per_gene(sorted_ref * f) returns the same value and
    would pass silently.
    """
    import gpudge._mwu as mwu
    ref, grp, sorted_ref, group_T, n_ref, m = _fixture()
    ref_tie = _tie_term_per_gene(sorted_ref)          # computed BEFORE the patch

    def _boom(*a, **k):
        raise AssertionError(
            "mwu_one_group_lfc called _tie_term_per_gene -- the reference must "
            "never be rescaled or its tie term rebuilt (spec 3.4a)")

    monkeypatch.setattr(mwu, "_tie_term_per_gene", _boom)
    mwu.mwu_one_group_lfc(sorted_ref, ref_tie, group_T, n_ref=n_ref,
                          lfc_combos=((0.4, "up"), (0.4, "down")))


def test_directional_variance_uses_THE_PASSED_ref_tie_term():
    """Complements the monkeypatch test: perturbing ref_tie_term must move every
    directional p, proving it is actually in the variance (and not, say,
    silently replaced by a locally derived one)."""
    ref, grp, sorted_ref, group_T, n_ref, m = _fixture()
    ref_tie = _tie_term_per_gene(sorted_ref)
    combos = ((0.4, "up"), (0.4, "down"))
    _, _, _, dp0 = mwu_one_group_lfc(sorted_ref, ref_tie, group_T,
                                     n_ref=n_ref, lfc_combos=combos)
    _, _, _, dp1 = mwu_one_group_lfc(sorted_ref, ref_tie + 1000.0, group_T,
                                     n_ref=n_ref, lfc_combos=combos)
    # per COMBO and per GENE, not merely "something changed"
    for k in range(len(combos)):
        assert torch.all(dp0[k] != dp1[k]), k


def test_directional_result_equals_a_from_scratch_recomputation():
    """Variance end-to-end (spec 5.6f), asserted bit-for-bit."""
    ref, grp, sorted_ref, group_T, n_ref, m = _fixture()
    ref_tie = _tie_term_per_gene(sorted_ref)
    tau, direction = 0.4, "up"
    _, _, du, dp = mwu_one_group_lfc(sorted_ref, ref_tie, group_T,
                                     n_ref=n_ref,
                                     lfc_combos=((tau, direction),))
    s = lfc_scale_factor(tau, direction)
    group_sorted, gc, run_start = _sorted_and_selfties(group_T)
    exp_u1 = _u1_against(sorted_ref, group_T.to(torch.float64) * s)
    exp_tie = ref_tie + _cross_tie_delta(
        sorted_ref, group_sorted.to(torch.float64) * s, gc, run_start)
    exp_p = _p_one_sided(exp_u1, exp_tie, m, n_ref)
    assert torch.equal(du[0], exp_u1)
    assert torch.equal(dp[0], exp_p)


@pytest.mark.parametrize("tau", [0.25, 0.5, 1.0, 2.0, 7.0])
def test_sorting_commutes_with_positive_scaling(tau):
    """Pins spec 3.4c -- what licenses `gs64 * s` with NO re-sort. If this ever
    fails, the scaled sorted target is silently unsorted and searchsorted
    returns garbage."""
    _, _, _, group_T, _, _ = _fixture()
    g64 = group_T.to(torch.float64)
    for direction in ("up", "down"):
        s = lfc_scale_factor(tau, direction)
        a = torch.sort(g64 * s, dim=1).values
        b = torch.sort(g64, dim=1).values * s
        assert torch.equal(a, b), (tau, direction)


def test_cross_tie_mass_survives_the_scaling():
    """Pins spec 3.4b: 0 * s == 0, so the zero-tie mass is LARGE, not
    negligible. A future 'optimise it away as ~0' must fail here."""
    n_genes, n_ref, m = 4, 60, 25
    rng = np.random.default_rng(11)
    ref = (rng.random((n_genes, n_ref)) > 0.7).astype(np.float32) * \
        rng.integers(1, 9, (n_genes, n_ref)).astype(np.float32)
    grp = (rng.random((n_genes, m)) > 0.7).astype(np.float32) * \
        rng.integers(1, 9, (n_genes, m)).astype(np.float32)
    sorted_ref = torch.from_numpy(np.sort(ref, axis=1))
    gs, gc, rs = _sorted_and_selfties(torch.from_numpy(grp))
    s = lfc_scale_factor(0.5, "up")
    delta = _cross_tie_delta(sorted_ref, gs.to(torch.float64) * s, gc, rs)
    assert float(delta.sum()) > 0.0


def test_float32_tie_boundary_is_resolved_correctly():
    """Spec 3.2b's counterexample, as a regression fixture.

    r and t are chosen so that float32(r * 2**0.5) == t -- a float32
    REFERENCE-scaling implementation reports a tie there. Exact arithmetic says
    t < r * 2**0.5, so the correct answer is 'no tie', and the float64 target
    scaling must produce U1 == 0 for the up direction (no target value at or
    above the shifted reference), not the m*n_ref/2 a spurious tie would give.
    """
    r, t = np.float32(2.5243635e-05), np.float32(3.569989e-05)
    assert np.float32(r * np.float32(2.0 ** 0.5)) == t      # the f32 trap
    assert float(t) < float(r) * 2.0 ** 0.5                 # the truth
    ref = np.full((1, 100), r, dtype=np.float32)
    grp = np.full((1, 100), t, dtype=np.float32)
    sorted_ref = torch.from_numpy(ref)
    ref_tie = _tie_term_per_gene(sorted_ref)
    _, _, du, dp = mwu_one_group_lfc(
        sorted_ref, ref_tie, torch.from_numpy(grp), n_ref=100,
        lfc_combos=((0.5, "up"),))
    assert du[0, 0].item() == 0.0            # NOT 5000.0 (the spurious tie)
    assert dp[0, 0].item() > 0.5             # emphatically not significant


def test_degenerate_groups_return_zeros_and_nan():
    n_genes, n_ref = 5, 12
    rng = np.random.default_rng(2)
    ref = rng.negative_binomial(2, 0.4, (n_genes, n_ref)).astype(np.float32)
    sorted_ref = torch.from_numpy(np.sort(ref, axis=1))
    ref_tie = _tie_term_per_gene(sorted_ref)
    empty = torch.zeros((n_genes, 0), dtype=torch.float32)
    u1, p, du, dp = mwu_one_group_lfc(
        sorted_ref, ref_tie, empty, n_ref=n_ref,
        lfc_combos=((0.5, "up"), (0.5, "down")))
    assert torch.all(u1 == 0) and torch.all(torch.isnan(p))
    assert du.shape == (2, n_genes) and dp.shape == (2, n_genes)
    assert torch.all(du == 0) and torch.all(torch.isnan(dp))


def test_n_ref_zero_returns_zeros_and_nan():
    n_genes, m = 3, 7
    empty_ref = torch.zeros((n_genes, 0), dtype=torch.float32)
    ref_tie = _tie_term_per_gene(empty_ref)
    grp = torch.ones((n_genes, m), dtype=torch.float32)
    u1, p, du, dp = mwu_one_group_lfc(
        empty_ref, ref_tie, grp, n_ref=0,
        lfc_combos=((0.5, "up"), (0.5, "down")))
    assert torch.all(u1 == 0) and torch.all(torch.isnan(p))
    assert torch.all(du == 0) and torch.all(torch.isnan(dp))
