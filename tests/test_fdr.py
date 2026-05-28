# tests/test_fdr.py
import numpy as np
import pytest
from scipy.stats import false_discovery_control
import torch
from gpudge._fdr import bh_per_group
from gpudge._csr_dense import HAS_NUMBA


def test_bh_per_group_matches_scipy_single_group():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 100)
    got = bh_per_group(torch.from_numpy(p),
                       group_id=torch.zeros(100, dtype=torch.int64),
                       n_groups=1).numpy()
    exp = false_discovery_control(p, method="bh")
    np.testing.assert_allclose(got, exp, rtol=1e-12)


def test_bh_per_group_independent_groups():
    rng = np.random.default_rng(1)
    p1 = rng.uniform(0, 1, 50)
    p2 = rng.uniform(0, 1, 30)
    p_all = np.concatenate([p1, p2])
    g = torch.cat([torch.zeros(50, dtype=torch.int64),
                   torch.ones(30, dtype=torch.int64)])
    got = bh_per_group(torch.from_numpy(p_all), g, n_groups=2).numpy()
    exp = np.concatenate([
        false_discovery_control(p1, method="bh"),
        false_discovery_control(p2, method="bh"),
    ])
    np.testing.assert_allclose(got, exp, rtol=1e-12)


def test_bh_per_group_handles_nan_pass_through():
    p = torch.tensor([0.01, float("nan"), 0.5])
    g = torch.tensor([0, 0, 0], dtype=torch.int64)
    got = bh_per_group(p, g, n_groups=1).numpy()
    assert np.isnan(got[1])


def test_bh_per_group_many_groups_matches_scipy_per_group():
    """T2 vectorised path: validate against scipy on many groups + mixed sizes."""
    rng = np.random.default_rng(42)
    n_groups = 50
    sizes = rng.integers(1, 500, size=n_groups)  # group sizes 1..499
    g_list = []
    p_list = []
    for gi, n in enumerate(sizes):
        g_list.append(np.full(n, gi, dtype=np.int64))
        p_list.append(rng.uniform(0, 1, n))
    g = np.concatenate(g_list)
    p = np.concatenate(p_list)
    # Shuffle so groups aren't pre-sorted (forces sort path).
    perm = rng.permutation(len(p))
    g = g[perm]
    p = p[perm]
    got = bh_per_group(torch.from_numpy(p),
                       torch.from_numpy(g),
                       n_groups=n_groups).numpy()
    # Expected: per-group scipy, scattered back via the same row layout.
    exp = np.empty_like(p)
    for gi in range(n_groups):
        m = g == gi
        exp[m] = false_discovery_control(p[m], method="bh")
    np.testing.assert_allclose(got, exp, rtol=1e-12)


def test_bh_per_group_empty_input():
    got = bh_per_group(torch.empty(0, dtype=torch.float64),
                       torch.empty(0, dtype=torch.int64),
                       n_groups=0).numpy()
    assert got.shape == (0,)


def test_bh_per_group_all_nan_input():
    p = torch.tensor([float("nan")] * 5)
    g = torch.tensor([0, 0, 1, 1, 2], dtype=torch.int64)
    got = bh_per_group(p, g, n_groups=3).numpy()
    assert np.all(np.isnan(got))


def test_bh_per_group_single_row_per_group():
    # m=1 in every group: q = p (unadjusted), clipped to [0, 1].
    p = torch.tensor([0.01, 0.5, 0.99], dtype=torch.float64)
    g = torch.tensor([0, 1, 2], dtype=torch.int64)
    got = bh_per_group(p, g, n_groups=3).numpy()
    np.testing.assert_allclose(got, [0.01, 0.5, 0.99], rtol=1e-12)


def test_bh_per_group_ties_within_group():
    # All-tied p-values: each row's BH-adjusted value should equal the
    # tied p-value (running-min from right collapses to min unadj = m*p/m = p).
    p = torch.tensor([0.5, 0.5, 0.5, 0.5], dtype=torch.float64)
    g = torch.tensor([0, 0, 0, 0], dtype=torch.int64)
    got = bh_per_group(p, g, n_groups=1).numpy()
    np.testing.assert_allclose(got, [0.5, 0.5, 0.5, 0.5], rtol=1e-12)


def test_bh_per_group_skips_empty_group_in_range():
    # n_groups=3 but only groups 0 and 2 have rows; group 1 is empty.
    p = torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64)
    g = torch.tensor([0, 2, 2], dtype=torch.int64)
    got = bh_per_group(p, g, n_groups=3).numpy()
    # Group 0 (m=1): q = 0.1. Group 2 (m=2): unadj = [0.5*2/1, 0.9*2/2]
    # = [1.0, 0.9] → running-min from right: [0.9, 0.9] → clip [0.9, 0.9].
    np.testing.assert_allclose(got, [0.1, 0.9, 0.9], rtol=1e-12)


def test_bh_per_group_inf_p_flows_through_bh():
    # ±inf are not "missing"; they flow through the BH math and clip to
    # [0, 1]. (Only NaN passes through unchanged.)
    #
    # Sorted ascending: [-inf, 0.01, 0.5, +inf]; ranks 1..4, m=4.
    #   unadj = [4*-inf/1, 4*0.01/2, 4*0.5/3, 4*+inf/4]
    #         = [-inf,     0.02,     0.667,   +inf]
    # Running-min from right then leaves the same values, and the per-
    # position clip caps them to [0, 1] → [0.0, 0.02, 0.667, 1.0].
    p = torch.tensor([0.01, float("inf"), 0.5, float("-inf")], dtype=torch.float64)
    g = torch.tensor([0, 0, 0, 0], dtype=torch.int64)
    got = bh_per_group(p, g, n_groups=1).numpy()
    np.testing.assert_allclose(got, [0.02, 1.0, 2/3, 0.0], rtol=1e-12)


def test_bh_per_group_rejects_out_of_range_group_id():
    p = torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64)
    # Negative group_id
    with pytest.raises(ValueError, match=r"group_id values must be in"):
        bh_per_group(p, torch.tensor([-1, 0, 0], dtype=torch.int64), n_groups=2)
    # group_id >= n_groups
    with pytest.raises(ValueError, match=r"group_id values must be in"):
        bh_per_group(p, torch.tensor([0, 1, 2], dtype=torch.int64), n_groups=2)
    # n_groups=0 with non-NaN rows
    with pytest.raises(ValueError, match=r"n_groups=0"):
        bh_per_group(p, torch.tensor([0, 0, 0], dtype=torch.int64), n_groups=0)


def test_bh_per_group_n_groups_zero_with_all_nan_ok():
    # n_groups=0 is fine if there are no non-NaN p rows: bh_per_group
    # returns before the bounds check (and before touching the numba
    # kernels), so the group_id values never get read.
    p = torch.tensor([float("nan"), float("nan")], dtype=torch.float64)
    # torch.zeros (not torch.empty) so this remains deterministic if a
    # future refactor decides to read group_id before the early return.
    g = torch.zeros(2, dtype=torch.int64)
    got = bh_per_group(p, g, n_groups=0).numpy()
    assert np.all(np.isnan(got))


@pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
def test_numpy_fallback_matches_numba_path():
    """The pure-numpy fallback must agree with the numba kernels on the
    same inputs (including ±inf, which modern scipy rejects). Skipped
    when numba is missing — without numba, ``bh_per_group`` itself
    dispatches to ``_bh_per_group_numpy`` and we'd be comparing it to
    itself.
    """
    from gpudge._fdr import _bh_per_group_numpy

    rng = np.random.default_rng(99)
    G = 10
    p = rng.uniform(0, 1, 500)
    g = rng.integers(0, G, 500).astype(np.int64)
    p[5] = float("inf")
    p[42] = float("-inf")

    # Numba path (via bh_per_group's full machinery)
    numba_out = bh_per_group(torch.from_numpy(p),
                              torch.from_numpy(g),
                              n_groups=G).numpy()

    # Numpy fallback path (called directly on the non-NaN subset)
    not_nan = ~np.isnan(p)
    finite_idx = np.flatnonzero(not_nan)
    numpy_out = _bh_per_group_numpy(p[finite_idx], g[finite_idx], G)
    full_numpy = np.full(p.shape, np.nan, dtype=np.float64)
    full_numpy[finite_idx] = numpy_out

    np.testing.assert_allclose(full_numpy, numba_out, rtol=1e-12,
                               equal_nan=True)
    # And the documented ±inf mapping holds in both paths.
    assert numba_out[5] == 1.0
    assert numba_out[42] == 0.0


@pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
def test_numba_kernels_directly_exercised():
    """Drive the numba kernels by hand to lock in their contract.

    Without this, the suite still passes when numba is installed but
    the kernels themselves have a regression — the rest of the tests
    go through bh_per_group, where a kernel-level bug could be masked
    by overall result-equivalence against per-group reference values
    (or by routing through the pure-numpy fallback in a future refactor).
    Asserting (1) counting-sort
    produces a valid per-group permutation and (2) the per-segment BH
    kernel writes correct q-values via the supplied order mapping
    catches kernel-only regressions early.
    """
    from gpudge._fdr import (
        _bh_per_segment_to_original,
        _counting_sort_by_group,
    )

    rng = np.random.default_rng(123)
    n, G = 200, 7
    p = rng.uniform(0, 1, n)
    g = rng.integers(0, G, n).astype(np.int64)

    # 1) counting-sort: result must be a permutation that groups rows by g.
    p_sorted, order_g, starts, stops = _counting_sort_by_group(g, p, G)
    assert p_sorted.shape == (n,)
    assert order_g.shape == (n,)
    # Permutation property
    assert sorted(order_g.tolist()) == list(range(n))
    # p_sorted[i] == p[order_g[i]]
    np.testing.assert_array_equal(p_sorted, p[order_g])
    # Each segment contains only its group's rows
    for gi in range(G):
        s, e = int(starts[gi]), int(stops[gi])
        assert np.all(g[order_g[s:e]] == gi)
    # starts/stops tile the full range
    assert starts[0] == 0
    assert stops[-1] == n

    # 2) BH kernel: q-values per row must match a per-group scipy reference.
    out = np.empty(n, dtype=np.float64)
    _bh_per_segment_to_original(p_sorted, starts, stops, order_g, out)
    exp = np.empty(n, dtype=np.float64)
    for gi in range(G):
        m = g == gi
        if m.any():
            exp[m] = false_discovery_control(p[m], method="bh")
    np.testing.assert_allclose(out, exp, rtol=1e-12)


def test_bh_per_group_mixed_nan_and_finite_per_group():
    rng = np.random.default_rng(7)
    p_list = []
    g_list = []
    for gi in range(5):
        finite = rng.uniform(0, 1, 20)
        nan_mask = rng.random(20) < 0.3
        finite[nan_mask] = np.nan
        p_list.append(finite)
        g_list.append(np.full(20, gi, dtype=np.int64))
    p = np.concatenate(p_list)
    g = np.concatenate(g_list)
    got = bh_per_group(torch.from_numpy(p),
                       torch.from_numpy(g),
                       n_groups=5).numpy()
    # Reference: per-group scipy on finite-only, scatter back; NaN passes through.
    exp = np.full_like(p, np.nan)
    for gi in range(5):
        rows = np.where(g == gi)[0]
        finite_rows = rows[np.isfinite(p[rows])]
        if finite_rows.size > 0:
            exp[finite_rows] = false_discovery_control(p[finite_rows], method="bh")
    np.testing.assert_allclose(got, exp, rtol=1e-12, equal_nan=True)
