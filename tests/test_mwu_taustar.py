# tests/test_mwu_taustar.py
"""tau* kernel: the constant-tie invariant, reachability limits, and the
order-statistic identity.

Every oracle here scales the TARGET and leaves the reference alone (spec 3.1),
and works in float64.
"""
import math

import numpy as np
import pytest
import torch
from scipy.stats import mannwhitneyu

from gpudge._mwu import (
    _cross_tie_delta, _cross_tie_generic, _sorted_and_selfties,
    _tie_term_per_gene, _u1_against, _u1_reach_limits, _z_from_level, _zero_counts,
    mwu_one_group, mwu_one_group_taustar,
)
from gpudge._taustar import TAUSTAR_SE_LEVEL


def _fixture(seed=5, n_genes=6, n_ref=50, m=17, p_zero=0.4):
    """Zero-inflated counts, the regime the zero-tie mass argument is about."""
    rng = np.random.default_rng(seed)
    R = rng.negative_binomial(3, 0.4, (n_genes, n_ref)).astype(np.float32)
    T = rng.negative_binomial(3, 0.4, (n_genes, m)).astype(np.float32)
    R[rng.random(R.shape) < p_zero] = 0.0
    T[rng.random(T.shape) < p_zero] = 0.0
    sorted_ref = torch.from_numpy(np.sort(R, axis=1))
    return sorted_ref, torch.from_numpy(T)


def test_z_from_level_matches_the_standard_normal_quantile():
    assert _z_from_level(0.5) == pytest.approx(0.0, abs=1e-12)
    assert _z_from_level(0.025) == pytest.approx(1.959963985, abs=1e-8)
    assert _z_from_level(0.05) == pytest.approx(1.644853627, abs=1e-8)


@pytest.mark.parametrize("q", [1e-8, 1e-16, 1e-17, 1e-20, 1e-100, 1e-300])
def test_z_from_level_stays_finite_and_exact_deep_in_the_tail(q):
    """normalize_taustar_spec accepts the whole OPEN interval (0, 1), so the
    quantile must hold up across it.

    The natural-looking `sqrt(2) * erfinv(1 - 2q)` does not: it evaluates the
    argument at `1 - 2q`, which is already wrong in the 3rd significant figure
    at q = 1e-16 (8.2095 vs 8.2221) and rounds to exactly 1.0 for q <= 1e-17,
    where erfinv returns +inf. The crossing level then becomes +/-inf and EVERY
    gene resolves to a signed infinity -- silently, on a level the validator
    called legal. Pinned against scipy, which is a test-only dependency here.
    """
    from scipy.stats import norm
    z = _z_from_level(q)
    assert math.isfinite(z)
    assert z == pytest.approx(float(norm.isf(q)), rel=1e-12)


def test_zero_counts_agree_with_numpy():
    sorted_ref, group_T = _fixture()
    group_sorted, _gc, _rs = _sorted_and_selfties(group_T)
    n0, m0 = _zero_counts(sorted_ref, group_sorted)
    np.testing.assert_array_equal(
        n0.numpy(), (sorted_ref.numpy() == 0).sum(axis=1))
    np.testing.assert_array_equal(
        m0.numpy(), (group_T.numpy() == 0).sum(axis=1))


def test_tie_generic_equals_cross_tie_delta_at_a_generic_shift():
    """Spec 3.4: off the coincidence set the cross-tie term does not move."""
    sorted_ref, group_T = _fixture()
    group_sorted, gc, run_start = _sorted_and_selfties(group_T)
    n0, _m0 = _zero_counts(sorted_ref, group_sorted)
    generic = _cross_tie_generic(group_sorted, gc, run_start, n0)

    gs64 = group_sorted.to(torch.float64)
    for delta in (0.31415926535, -0.2718281828, 1.61803398875):
        measured = _cross_tie_delta(
            sorted_ref, gs64 * math.pow(2.0, -delta), gc, run_start)
        torch.testing.assert_close(generic, measured, rtol=0, atol=0)


def test_tie_generic_differs_from_the_value_at_delta_zero():
    """delta = 0 is NOT generic: at s == 1.0 raw target values coincide with
    raw reference values in bulk. Getting this wrong silently corrupts sigma.

    DETERMINISTIC fixture, not the random one: the assertion needs at least one
    NONZERO target value that also appears in the reference, and a random draw
    only makes that overwhelmingly likely, not certain. Here target 4 and 8 both
    appear in the reference, so the two terms must differ by construction.
    """
    sorted_ref = torch.from_numpy(
        np.sort(np.array([0.0, 1.0, 4.0, 8.0], dtype=np.float32))[None, :])
    group_T = torch.from_numpy(np.array([[0.0, 4.0, 8.0]], dtype=np.float32))
    group_sorted, gc, run_start = _sorted_and_selfties(group_T)
    n0, _m0 = _zero_counts(sorted_ref, group_sorted)
    generic = _cross_tie_generic(group_sorted, gc, run_start, n0)
    at_zero = _cross_tie_delta(
        sorted_ref, group_sorted.to(torch.float64), gc, run_start)
    # The zero run contributes identically to both (0 * s == 0); the 4 and 8
    # runs contribute only at delta = 0, each rc=1/gc=1 -> 3*1*1*2 = 6.
    assert float(at_zero[0] - generic[0]) == pytest.approx(12.0)


def test_reach_limits_bracket_the_measured_u1():
    sorted_ref, group_T = _fixture()
    group_sorted, _gc, _rs = _sorted_and_selfties(group_T)
    n0, m0 = _zero_counts(sorted_ref, group_sorted)
    m, n_ref = group_T.shape[1], sorted_ref.shape[1]
    u1_min, u1_max = _u1_reach_limits(m, m0, n_ref, n0)

    gT64 = group_T.to(torch.float64)
    for delta in (-40.0, -3.0, 0.0, 3.0, 40.0):
        u1 = _u1_against(sorted_ref, gT64 * math.pow(2.0, -delta))
        assert bool((u1 >= u1_min - 1e-9).all())
        assert bool((u1 <= u1_max + 1e-9).all())


def test_reach_limits_are_attained_in_the_limit():
    sorted_ref, group_T = _fixture()
    group_sorted, _gc, _rs = _sorted_and_selfties(group_T)
    n0, m0 = _zero_counts(sorted_ref, group_sorted)
    m, n_ref = group_T.shape[1], sorted_ref.shape[1]
    u1_min, u1_max = _u1_reach_limits(m, m0, n_ref, n0)
    gT64 = group_T.to(torch.float64)
    torch.testing.assert_close(
        _u1_against(sorted_ref, gT64 * math.pow(2.0, -200.0)), u1_min)
    torch.testing.assert_close(
        _u1_against(sorted_ref, gT64 * math.pow(2.0, 200.0)), u1_max)


def test_limits_coincide_exactly_when_target_or_reference_is_all_zero():
    """Spec 3.5: n0*(m-m0) == n*(m-m0) has exactly two solutions."""
    n_ref, m = 12, 5
    # all-zero target
    sorted_ref = torch.from_numpy(
        np.sort(np.arange(n_ref, dtype=np.float32))[None, :])
    group_T = torch.zeros((1, m), dtype=torch.float32)
    gs, _gc, _rs = _sorted_and_selfties(group_T)
    n0, m0 = _zero_counts(sorted_ref, gs)
    lo, hi = _u1_reach_limits(m, m0, n_ref, n0)
    assert float(lo) == float(hi)
    # all-zero reference
    sorted_ref = torch.zeros((1, n_ref), dtype=torch.float32)
    group_T = torch.from_numpy(np.arange(1, m + 1, dtype=np.float32)[None, :])
    gs, _gc, _rs = _sorted_and_selfties(group_T)
    n0, m0 = _zero_counts(sorted_ref, gs)
    lo, hi = _u1_reach_limits(m, m0, n_ref, n0)
    assert float(lo) == float(hi)


def _u1_oracle(T_row, R_row, delta):
    """U1 at one shift, straight from the definition (spec 1)."""
    scaled = T_row.astype(np.float64) * np.exp2(-delta)
    less = (R_row[None, :] < scaled[:, None]).sum()
    equal = (R_row[None, :] == scaled[:, None]).sum()
    return float(less) + 0.5 * float(equal)


def test_base_columns_are_bit_identical_to_mwu_one_group():
    sorted_ref, group_T = _fixture()
    ref_tie = torch.zeros(sorted_ref.shape[0], dtype=torch.float64)
    n_ref = sorted_ref.shape[1]
    u1_a, p_a = mwu_one_group(sorted_ref, ref_tie, group_T, n_ref=n_ref)
    u1_b, p_b, _ts = mwu_one_group_taustar(
        sorted_ref, ref_tie, group_T, n_ref=n_ref,
        taustar_levels=(0.5, 0.05), taustar_iters=20)
    assert torch.equal(u1_a, u1_b)
    assert torch.equal(p_a, p_b)


def test_taustar_is_the_crossing_point_of_the_counting_oracle():
    """The defining property, against the level recomputed from the exposed
    primitives: U1 satisfies the level just INSIDE tau* and violates it just
    outside. Asserting only that U1 moves the right way would merely restate
    monotonicity and would pass for any tau*."""
    from gpudge._mwu import _s_sq_of

    sorted_ref, group_T = _fixture(seed=11)
    n_genes, n_ref = sorted_ref.shape
    m = group_T.shape[1]
    ref_tie = torch.zeros(n_genes, dtype=torch.float64)
    q = 0.05
    u1, _p, ts = mwu_one_group_taustar(
        sorted_ref, ref_tie, group_T, n_ref=n_ref,
        taustar_levels=(q,), taustar_iters=40)

    group_sorted, gc, run_start = _sorted_and_selfties(group_T)
    n0, _m0 = _zero_counts(sorted_ref, group_sorted)
    sigma = torch.sqrt(_s_sq_of(
        ref_tie + _cross_tie_generic(group_sorted, gc, run_start, n0),
        m, n_ref))
    z_q = _z_from_level(q)
    mu = m * n_ref / 2.0

    eps = 1e-6
    checked = 0
    for j in range(n_genes):
        t = float(ts[0, j])
        if not math.isfinite(t):
            continue
        up = bool(u1[j] >= mu)
        level = (mu + 0.5 + z_q * float(sigma[j])) if up else \
                (mu - 0.5 - z_q * float(sigma[j]))
        R_row = sorted_ref[j].numpy()
        T_row = group_T[j].numpy()
        inside = _u1_oracle(T_row, R_row, t - eps if up else t + eps)
        outside = _u1_oracle(T_row, R_row, t + eps if up else t - eps)
        if up:
            assert inside >= level
            assert outside < level
        else:
            assert inside <= level
            assert outside > level
        checked += 1
    assert checked > 0, "fixture produced no finite tau* to check"


def test_taustar_at_p_half_is_the_median_pairwise_log_ratio():
    """The Hodges-Lehmann identity (spec 1). No zeros, so every pairwise ratio
    is finite and the median is well defined."""
    rng = np.random.default_rng(2)
    n_ref, m = 60, 40
    R = rng.lognormal(0.0, 1.0, n_ref).astype(np.float32)
    T = (rng.lognormal(0.0, 1.0, m) * 2.0).astype(np.float32)
    sorted_ref = torch.from_numpy(np.sort(R)[None, :])
    group_T = torch.from_numpy(T[None, :])
    ref_tie = torch.zeros(1, dtype=torch.float64)

    _u1, _p, ts = mwu_one_group_taustar(
        sorted_ref, ref_tie, group_T, n_ref=n_ref,
        taustar_levels=(0.5,), taustar_iters=40)

    d = np.log2(T.astype(np.float64)[:, None] / R.astype(np.float64)[None, :])
    # m*n = 2400 pairs, so the continuity correction's half-pair offset is
    # negligible against this tolerance.
    assert float(ts[0, 0]) == pytest.approx(float(np.median(d)), abs=0.02)


@pytest.mark.parametrize("q", [0.5, 0.05, 0.025])
def test_scipy_brackets_the_level_across_tau_star(q):
    """Spec 5a. Assert the BRACKET -- p <= q just inside, p > q just outside --
    NOT p == q.

    U1 moves in steps of half a pair, so just inside the crossing the p-value
    sits AT OR BELOW q, usually below it by about one step's worth (equality is
    possible when a step lands exactly on the level). On this fixture that gap
    is not rounding-level: the step in p is roughly phi(z_q) * 0.5 / sd(U), and
    with sd(U) about 53 that is ~0.004 at q = 0.05 and larger near q = 0.5 where
    phi peaks -- either way an `approx(q, abs=2e-3)` assertion would be flaky.

    Evaluated just inside / just outside, never exactly AT tau*: there scipy
    measures the coincidence tie and uses the spiked variance instead of p~'s
    tie_generic (spec 3.4).
    """
    sorted_ref, group_T = _fixture(seed=7, n_genes=4, n_ref=40, m=15)
    n_genes, n_ref = sorted_ref.shape
    m = group_T.shape[1]
    ref_tie = torch.from_numpy(np.array([
        _tie_term_numpy(sorted_ref[j].numpy()) for j in range(n_genes)]))
    u1, _p, ts = mwu_one_group_taustar(
        sorted_ref, ref_tie, group_T, n_ref=n_ref,
        taustar_levels=(q,), taustar_iters=45)

    mu = m * n_ref / 2.0
    eps = 1e-5
    checked = 0
    for j in range(n_genes):
        t = float(ts[0, j])
        if not math.isfinite(t):
            continue
        up = bool(u1[j] >= mu)

        def _p_at(delta, up=up, j=j):
            scaled = group_T[j].numpy().astype(np.float64) * np.exp2(-delta)
            return mannwhitneyu(
                scaled, sorted_ref[j].numpy().astype(np.float64),
                alternative="greater" if up else "less",
                method="asymptotic", use_continuity=True).pvalue

        inside = _p_at(t - eps if up else t + eps)
        outside = _p_at(t + eps if up else t - eps)
        assert inside <= q + 1e-12
        assert outside > q
        checked += 1
    assert checked > 0, "fixture produced no finite tau* to check"


def _tie_term_numpy(values):
    """sum(t^3 - t) over the reference's tie runs, matching _tie_term_per_gene."""
    _u, counts = np.unique(values, return_counts=True)
    c = counts.astype(np.float64)
    return float((c ** 3 - c).sum())


def test_all_zero_on_both_sides_is_nan():
    sorted_ref = torch.zeros((1, 8), dtype=torch.float32)
    group_T = torch.zeros((1, 5), dtype=torch.float32)
    ref_tie = torch.zeros(1, dtype=torch.float64)
    _u1, _p, ts = mwu_one_group_taustar(
        sorted_ref, ref_tie, group_T, n_ref=8,
        taustar_levels=(0.5,), taustar_iters=20)
    assert math.isnan(float(ts[0, 0]))


def test_expressed_target_absent_reference_is_plus_inf_not_nan():
    """Spec 5d: constant U1 but a genuinely unbounded UP effect. NaN here would
    discard the strongest signal in the dataset."""
    sorted_ref = torch.zeros((1, 8), dtype=torch.float32)
    group_T = torch.full((1, 5), 7.0, dtype=torch.float32)
    ref_tie = torch.zeros(1, dtype=torch.float64)
    _u1, _p, ts = mwu_one_group_taustar(
        sorted_ref, ref_tie, group_T, n_ref=8,
        taustar_levels=(0.05,), taustar_iters=20)
    assert float(ts[0, 0]) == math.inf


def test_absent_target_expressed_reference_is_minus_inf():
    sorted_ref = torch.from_numpy(
        np.sort(np.full(8, 9.0, dtype=np.float32))[None, :])
    group_T = torch.zeros((1, 5), dtype=torch.float32)
    ref_tie = torch.zeros(1, dtype=torch.float64)
    _u1, _p, ts = mwu_one_group_taustar(
        sorted_ref, ref_tie, group_T, n_ref=8,
        taustar_levels=(0.05,), taustar_iters=20)
    assert float(ts[0, 0]) == -math.inf


def test_level_exactly_equal_to_the_reachable_minimum_is_plus_inf():
    """Spec 3.5, the equality that a STRICT rule gets wrong.

    This fixture is constructed so `level == u1_min` EXACTLY, which is the only
    case that distinguishes `level <= u1_min` from `level < u1_min`. A generic
    zero-heavy gene does not exercise it -- it satisfies the strict form too --
    so the boundary needs its own arithmetic.

    m = 1, n_ref = 3, reference [0, 0, 5], target [7], q = 0.5:
      m0 = 0, n0 = 2
      u1_min = n0*(m - m0/2) = 2
      mu     = m*n/2 = 1.5,  z_q = 0,  level = mu + 0.5 = 2.0   == u1_min
    As s -> 0+ the scaled target stays strictly positive, so it still outranks
    the two zero references: U1 -> 2, z -> 0, p_up -> 0.5 == q. So `p_up <= q`
    holds at EVERY shift and the sup is +inf. A strict `<` bisects instead and
    returns a bracket midpoint.
    """
    sorted_ref = torch.from_numpy(
        np.sort(np.array([0.0, 0.0, 5.0], dtype=np.float32))[None, :])
    group_T = torch.from_numpy(np.array([[7.0]], dtype=np.float32))
    # Reference tie term: the two zeros form one run -> 2**3 - 2 = 6.
    ref_tie = torch.tensor([6.0], dtype=torch.float64)

    group_sorted, gc, run_start = _sorted_and_selfties(group_T)
    n0, m0 = _zero_counts(sorted_ref, group_sorted)
    u1_min, _u1_max = _u1_reach_limits(1, m0, 3, n0)
    assert float(u1_min[0]) == 2.0          # the fixture is on the boundary

    _u1, _p, ts = mwu_one_group_taustar(
        sorted_ref, ref_tie, group_T, n_ref=3,
        taustar_levels=(0.5,), taustar_iters=20)
    assert float(ts[0, 0]) == math.inf


def test_coincidence_plateau_has_positive_width_and_predicted_magnitude():
    """Spec 3.4 / 5c-bis. In floating point the coincidence set has POSITIVE
    width, so `tie_generic` is not merely 'almost always' right -- it is right
    because spec 3.2 defines tau* against `p~`, whose tie term IS tie_generic.
    Assert the plateau exists with the predicted magnitude 3*rc*gc*(rc+gc)
    rather than asserting it away.

    This test deliberately checks ONLY the tie arithmetic, not a tau* value --
    the returned-value behaviour is covered by the bracket and boundary tests.
    """
    # A target value exactly equal to a reference value collides at delta = 0.
    sorted_ref = torch.from_numpy(
        np.sort(np.array([1.0, 2.0, 4.0, 4.0], dtype=np.float32))[None, :])
    group_T = torch.from_numpy(np.array([[4.0, 8.0, 16.0]], dtype=np.float32))
    group_sorted, gc, run_start = _sorted_and_selfties(group_T)
    n0, _m0 = _zero_counts(sorted_ref, group_sorted)

    generic = _cross_tie_generic(group_sorted, gc, run_start, n0)
    at_zero = _cross_tie_delta(
        sorted_ref, group_sorted.to(torch.float64), gc, run_start)
    # T=4 meets the two reference 4s: rc=2, gc=1 -> 3*rc*gc*(rc+gc) = 18.
    assert float(at_zero[0] - generic[0]) == pytest.approx(18.0)

    # A power-of-two ratio collides at an INTEGER delta, not just at zero.
    # delta = 1 -> s = 0.5 -> scaled target [2, 4, 8]. Two runs now collide:
    # 2 meets one reference 2 (rc=1, gc=1 -> 3*1*1*2 =  6) and
    # 4 meets two reference 4s (rc=2, gc=1 -> 3*2*1*3 = 18), total 24.
    at_one = _cross_tie_delta(
        sorted_ref, group_sorted.to(torch.float64) * 0.5, gc, run_start)
    assert float(at_one[0] - generic[0]) == pytest.approx(24.0)


def test_level_exactly_equal_to_the_reachable_maximum_is_minus_inf():
    """The DOWN mirror of the boundary above -- `level == u1_max` exactly.

    m = 1, n_ref = 3, reference [0, 0, 5], target [0], q = 0.5:
      m0 = 1, n0 = 2
      u1_max = (m-m0)*n_ref + m0*n0/2 = 0 + 1 = 1
      U1(0)  = 0.5 * 2 = 1 < mu = 1.5  ->  DOWN
      ref_tie = 6; the zero run has rc = 2, gc = 1 -> 27 - 3 - 6 = 18,
      so tie_generic = 24, s^2 = (3/12)*(5 - 24/12) = 0.75
      z_q = 0, level = mu - 0.5 - 0 = 1.0  ==  u1_max
    `U1 <= level` holds at every shift, so the answer is -inf. A strict `>`
    bisects instead. `dead` is False here (n0 != n_ref), so this is NOT the NaN
    path.
    """
    sorted_ref = torch.from_numpy(
        np.sort(np.array([0.0, 0.0, 5.0], dtype=np.float32))[None, :])
    group_T = torch.zeros((1, 1), dtype=torch.float32)
    ref_tie = torch.tensor([6.0], dtype=torch.float64)

    group_sorted, _gc, _rs = _sorted_and_selfties(group_T)
    n0, m0 = _zero_counts(sorted_ref, group_sorted)
    _u1_min, u1_max = _u1_reach_limits(1, m0, 3, n0)
    assert float(u1_max[0]) == 1.0          # the fixture is on the boundary

    _u1, _p, ts = mwu_one_group_taustar(
        sorted_ref, ref_tie, group_T, n_ref=3,
        taustar_levels=(0.5,), taustar_iters=20)
    assert float(ts[0, 0]) == -math.inf


@pytest.mark.parametrize("se,want_rows", [(False, 2), (True, 5)])
def test_empty_group_returns_nan_like_mwu_one_group(se, want_rows):
    """Parameterized over the SE flag because the m == 0 early return
    allocates its own tensor, on a path that never reaches the level loop.
    Reverting that allocation to `n_levels` would otherwise escape every other
    test in this file. It is caught HERE rather than at the driver: with these
    two levels a 2-row return into a 5-row accumulator slot does raise, but a
    ONE-level, SE-on caller returns a single row that torch and numpy both
    broadcast into the 4-row slot silently. So the row count is pinned at the
    kernel, where the shape is unambiguous."""
    sorted_ref, _g = _fixture()
    n_genes, n_ref = sorted_ref.shape
    ref_tie = torch.zeros(n_genes, dtype=torch.float64)
    group_T = torch.zeros((n_genes, 0), dtype=torch.float32)
    u1, p, ts = mwu_one_group_taustar(
        sorted_ref, ref_tie, group_T, n_ref=n_ref,
        taustar_levels=(0.5, 0.05), taustar_iters=20, taustar_se=se)
    assert bool((u1 == 0).all())
    assert bool(torch.isnan(p).all())
    assert ts.shape == (want_rows, n_genes)
    assert bool(torch.isnan(ts).all())


def test_levels_are_ordered_along_dim_zero():
    sorted_ref, group_T = _fixture(seed=13)
    n_genes, n_ref = sorted_ref.shape
    m = group_T.shape[1]
    ref_tie = torch.zeros(n_genes, dtype=torch.float64)
    u1, _p, ts = mwu_one_group_taustar(
        sorted_ref, ref_tie, group_T, n_ref=n_ref,
        taustar_levels=(0.05, 0.5), taustar_iters=30)
    assert ts.shape == (2, n_genes)
    # DIRECTION-AWARE, not abs(): for an up gene the 0.05 bound sits BELOW the
    # 0.5 point estimate, for a down gene ABOVE it. Comparing magnitudes is
    # WRONG whenever the interval straddles zero -- estimate +0.1 with a bound
    # of -0.3 is a legal result whose magnitudes are ordered the other way.
    is_up = (u1 >= m * n_ref / 2.0)
    finite = torch.isfinite(ts).all(dim=0)
    lo, hi = ts[0][finite], ts[1][finite]
    up = is_up[finite]
    assert bool((lo[up] <= hi[up] + 1e-9).all())
    assert bool((lo[~up] >= hi[~up] - 1e-9).all())


# --- tau_star_se ----------------------------------------------------------

_SE_LEVELS = (0.025, 0.5)

# n_ref=200/m=60, NOT the module default n_ref=50/m=17. Measured: the default
# yields 0/6 genes with both endpoints finite, so every interval invariant below
# would pass VACUOUSLY. This shape yields 6/6 with both directions represented.
_SE_FIXTURE = dict(n_ref=200, m=60)


def _run_se(levels=_SE_LEVELS, iters=30, **fx):
    """(taustar rows, u1, mu) with the SE block on.

    ref_tie is the REAL per-gene tie term, not zeros: the fixture is
    negative-binomial and heavily tied, and a zeroed term would give the kernel
    a different variance from any external oracle.
    """
    sorted_ref, group_T = _fixture(**{**_SE_FIXTURE, **fx})
    n_ref = sorted_ref.shape[1]
    ref_tie = _tie_term_per_gene(sorted_ref)
    u1, _p, ts = mwu_one_group_taustar(
        sorted_ref, ref_tie, group_T, n_ref=n_ref,
        taustar_levels=levels, taustar_iters=iters, taustar_se=True)
    mu = group_T.shape[1] * n_ref / 2.0
    return ts, u1, mu


def test_se_block_appends_exactly_three_rows():
    ts, _u1, _mu = _run_se()
    assert ts.shape[0] == len(_SE_LEVELS) + 3


def test_level_rows_are_bit_identical_with_and_without_the_se_block():
    """The release gate, at kernel level: the is_up call is the untouched
    present code path (spec 3.3). Runs on CPU, so it executes in CI."""
    sorted_ref, group_T = _fixture(**_SE_FIXTURE)
    n_ref = sorted_ref.shape[1]
    ref_tie = _tie_term_per_gene(sorted_ref)
    kw = dict(n_ref=n_ref, taustar_levels=_SE_LEVELS, taustar_iters=30)
    u_off, p_off, ts_off = mwu_one_group_taustar(
        sorted_ref, ref_tie, group_T, taustar_se=False, **kw)
    u_on, p_on, ts_on = mwu_one_group_taustar(
        sorted_ref, ref_tie, group_T, taustar_se=True, **kw)
    np.testing.assert_array_equal(u_on.numpy(), u_off.numpy())
    np.testing.assert_array_equal(p_on.numpy(), p_off.numpy())
    np.testing.assert_array_equal(
        ts_on[:len(_SE_LEVELS)].numpy(), ts_off.numpy())


def test_lo_never_exceeds_hi():
    """EXACT, not to a tolerance: spec 3.4 proves the two searches follow
    identical brackets until their first differing decision, which can only
    separate them in the correct order.

    Asserted over every NON-NaN pair, not just the finite ones. Filtering to
    finite endpoints would make the assertion blind to (+inf, -inf) -- the very
    inversion an earlier draft wrongly believed the kernel produced. Infinities
    compare correctly, so there is no reason to exclude them.
    """
    ts, _u1, _mu = _run_se()
    lo, hi = ts[2].numpy(), ts[3].numpy()
    defined = ~np.isnan(lo) & ~np.isnan(hi)
    assert defined.any()
    assert np.all(lo[defined] <= hi[defined])
    # and separately confirm the fixture reaches the finite case at all
    assert (np.isfinite(lo) & np.isfinite(hi)).any()


def test_the_interval_brackets_the_point_estimate():
    """lo(q) increases and hi(q) decreases in q, so the q=0.5 estimate lies
    inside the q=0.025 interval."""
    ts, _u1, _mu = _run_se()
    est = ts[1].numpy()          # tau*_p0.5
    lo, hi = ts[2].numpy(), ts[3].numpy()
    good = np.isfinite(lo) & np.isfinite(hi) & np.isfinite(est)
    assert good.any()
    assert np.all(lo[good] <= est[good])
    assert np.all(est[good] <= hi[good])


def test_se_is_the_half_width_over_the_quantile():
    ts, _u1, _mu = _run_se()
    lo, hi, se = ts[2].numpy(), ts[3].numpy(), ts[4].numpy()
    z = _z_from_level(TAUSTAR_SE_LEVEL)
    good = np.isfinite(lo) & np.isfinite(hi)
    np.testing.assert_allclose(se[good], (hi[good] - lo[good]) / (2 * z),
                               rtol=0, atol=0)


def test_se_is_never_negative():
    """se is deliberately NOT clamped (spec 3.4), so a negative value here is
    the intended signal of a direction/indexing regression rather than
    something a clamp would have swallowed."""
    ts, _u1, _mu = _run_se()
    se = ts[4].numpy()
    assert np.all(se[~np.isnan(se)] >= 0.0)


def test_the_one_sided_column_is_the_direction_selected_endpoint():
    """tau*_p0.025 == where(u1 >= mu, lo, hi), exactly (spec 5d)."""
    ts, u1, mu = _run_se()
    is_up = (u1 >= mu).numpy()
    want = np.where(is_up, ts[2].numpy(), ts[3].numpy())
    np.testing.assert_array_equal(ts[0].numpy(), want)


def _one_gene(t_vals, r_vals, iters=30):
    """Single hand-built gene through the SE block -> (lo, hi, se).

    The REAL reference tie term: these fixtures are mostly ties, and zeroing it
    would change the variance the levels are built from.
    """
    sorted_ref = torch.tensor([sorted(r_vals)], dtype=torch.float32)
    group_T = torch.tensor([t_vals], dtype=torch.float32)
    ref_tie = _tie_term_per_gene(sorted_ref)
    _u1, _p, ts = mwu_one_group_taustar(
        sorted_ref, ref_tie, group_T, n_ref=len(r_vals),
        taustar_levels=(0.5,), taustar_iters=iters, taustar_se=True)
    return ts[1].item(), ts[2].item(), ts[3].item()


@pytest.mark.parametrize("t,r,want", [
    # Constant U1. The band is set by c vs (L_dn, L_up), NOT by which side is
    # all-zero -- rows 1 and 2 are both all-zero targets and land in DIFFERENT
    # bands, and rows 4 and 5 are both all-zero references and likewise.
    ([0.0], [1.0], (-math.inf, math.inf)),                       # L_dn<c<L_up
    ([0.0, 0.0, 0.0], [1.0, 2.0, 3.0, 5.0], (-math.inf, -math.inf)),  # c<=L_dn
    ([0.0, 0.0], [1.0, 2.0], (-math.inf, math.inf)),             # L_dn<c<L_up
    ([3.0, 9.0], [0.0] * 4, (-math.inf, math.inf)),              # L_dn<c<L_up
    ([3.0, 9.0, 5.0, 7.0, 2.0, 8.0], [0.0] * 8,
     (math.inf, math.inf)),                                      # c>=L_up
])
def test_constant_u1_lands_in_the_band_set_by_c_not_by_which_side_is_zero(
        t, r, want):
    """Every value here was measured against the kernel. An earlier draft
    asserted 'all-zero target => (-inf,-inf)' as a rule; rows 1 and 3 disprove
    it, and row 5 disproves the mirror rule for all-zero references. Band
    membership depends on group size, because mu and sigma grow at different
    rates in m and n."""
    lo, hi, se = _one_gene(t, r)
    assert (lo, hi) == want
    assert se == math.inf


def test_zero_on_both_sides_is_nan_not_infinite():
    lo, hi, se = _one_gene([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0])
    assert math.isnan(lo) and math.isnan(hi) and math.isnan(se)


def test_se_is_infinite_when_only_the_upper_endpoint_is_unbounded():
    """Unboundedness comes from ZEROS, which pin a reachability limit -- NOT
    from separation. Zeros in the reference push u1_min up until the up-level
    falls at or below it."""
    lo, hi, se = _one_gene([1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 0.0, 1.0])
    assert math.isfinite(lo)
    assert hi == math.inf
    assert se == math.inf


def test_se_is_infinite_when_only_the_lower_endpoint_is_unbounded():
    """The mirror: zeros on the target instead."""
    lo, hi, se = _one_gene([0.0, 0.0, 0.0, 1.0], [1.0, 2.0, 3.0, 4.0])
    assert lo == -math.inf
    assert math.isfinite(hi)
    assert se == math.inf


def test_widely_separated_values_are_NOT_unbounded():
    """Regression guard on a fixture that LOOKS unbounded and is not. With
    m = n = 4 and no zeros, _u1_reach_limits is [0, 16] and both q=0.025 levels
    fall strictly inside it, so both endpoints are finite -- measured
    lo=+4.644, hi=+6.322. Pinned so nobody 'fixes' the unbounded fixtures above
    back into this shape."""
    lo, hi, se = _one_gene([50.0, 60.0, 70.0, 80.0], [1.0, 1.0, 2.0, 2.0])
    assert math.isfinite(lo) and math.isfinite(hi) and math.isfinite(se)


@pytest.mark.parametrize("side,alt", [("lo", "greater"), ("hi", "less")])
def test_se_endpoints_agree_with_scipy_in_p_space(side, alt):
    """Probe just inside and just outside a finite endpoint and confirm the
    ONE-SIDED p crosses 0.025. Deliberately p-space: an oracle that re-derives
    U1/sigma would reimplement the code under test (declined on PR #113).

    The outside probe is a STRICT inequality. `p == q` is still inside the
    significant region `p <= q`, so `0.025 <= outside` would pass on a point
    that has not actually crossed.
    """
    rng = np.random.default_rng(11)
    T = rng.negative_binomial(6, 0.3, 40).astype(np.float32) + 1.0
    R = rng.negative_binomial(4, 0.3, 60).astype(np.float32) + 1.0
    lo, hi, _se = _one_gene(list(T), list(R), iters=40)
    edge = lo if side == "lo" else hi
    assert math.isfinite(edge)
    eps = 1e-3
    # lo = sup{delta : up-test significant}, so the significant side of lo is
    # BELOW it. hi = inf{delta : down-test significant}, so the significant
    # side of hi is ABOVE it. Hence "inside" is -eps for lo and +eps for hi.
    d_in = edge - eps if side == "lo" else edge + eps
    d_out = edge + eps if side == "lo" else edge - eps

    def p_at(d):
        return mannwhitneyu(T * 2.0 ** (-d), R, alternative=alt,
                            method="asymptotic", use_continuity=True).pvalue

    assert p_at(d_in) <= TAUSTAR_SE_LEVEL < p_at(d_out)


def test_the_fixture_exercises_both_directions():
    """Guard on the tests above: every one of them asserts a property that a
    fixture of all-up genes would satisfy vacuously, and the whole point of
    the SE block is that the two directions select opposite endpoints."""
    ts, u1, mu = _run_se()
    is_up = (u1 >= mu).numpy()
    assert is_up.any() and (~is_up).any()
    assert ts.shape[0] == len(_SE_LEVELS) + 3
