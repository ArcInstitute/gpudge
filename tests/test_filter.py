# tests/test_filter.py
import warnings

import numpy as np
import pytest
from gpudge._filter import (
    combined_keep_mask,
    validate_keep_genes,
    x_has_noncount_signal,
    _row_scale_needs,
)


def test_single_filter_target_or_ref_1d_ref():
    tq = np.array([[0.0, 2.0, 0.0]])          # (1 target, 3 genes)
    rq = np.array([1.5, 0.0, 0.0])            # (3 genes,)
    keep = combined_keep_mask(1, 3, filters=[(tq, rq, 1.0)])
    np.testing.assert_array_equal(keep, [[True, True, False]])


def test_single_filter_2d_ref_broadcasts_per_target():
    tq = np.array([[0.0, 2.0], [3.0, 0.0]])
    rq = np.array([[1.5, 0.0], [0.0, 0.0]])
    keep = combined_keep_mask(2, 2, filters=[(tq, rq, 1.0)])
    np.testing.assert_array_equal(keep, [[True, True], [True, False]])


def test_threshold_zero_drops_zero_mean():
    tq = np.array([[0.0, 1.0]])
    rq = np.array([0.0, 0.0])
    keep = combined_keep_mask(1, 2, filters=[(tq, rq, 0.0)])
    np.testing.assert_array_equal(keep, [[False, True]])


def test_negative_threshold_keeps_all_even_with_negative_values():
    # *_value quantities can be negative (residual/centered X); a negative
    # threshold must be an explicit all-True, NOT `value > negative`.
    tq = np.array([[-5.0, -0.1]])
    rq = np.array([-9.0, -2.0])
    keep = combined_keep_mask(1, 2, filters=[(tq, rq, -1.0)])
    np.testing.assert_array_equal(keep, [[True, True]])


def test_and_across_multiple_filters():
    a_t = np.array([[5.0, 5.0, 0.0]])
    a_r = np.array([0.0, 0.0, 0.0])
    b_t = np.array([[0.0, 5.0, 5.0]])
    b_r = np.array([0.0, 0.0, 0.0])
    keep = combined_keep_mask(1, 3, filters=[(a_t, a_r, 1.0), (b_t, b_r, 1.0)])
    np.testing.assert_array_equal(keep, [[False, True, False]])


def test_no_filters_no_keep_genes_keeps_all():
    keep = combined_keep_mask(2, 3, filters=[])
    np.testing.assert_array_equal(keep, np.ones((2, 3), bool))


def test_keep_genes_anded_and_broadcast():
    tq = np.full((2, 3), 5.0)
    rq = np.zeros(3)
    kg = np.array([True, False, True])
    keep = combined_keep_mask(2, 3, filters=[(tq, rq, 1.0)], keep_genes=kg)
    np.testing.assert_array_equal(keep, [[True, False, True], [True, False, True]])


def test_keep_genes_alone_no_filters():
    kg = np.array([False, True])
    keep = combined_keep_mask(2, 2, filters=[], keep_genes=kg)
    np.testing.assert_array_equal(keep, [[False, True], [False, True]])


def test_validate_keep_genes_ok():
    kg = np.array([True, False, True])
    out = validate_keep_genes(kg, 3)
    assert out.dtype == np.bool_
    np.testing.assert_array_equal(out, kg)


@pytest.mark.parametrize("bad,n,match", [
    (np.array([1, 0, 1]), 3, "boolean"),
    (np.array(["a", "b"]), 2, "boolean"),
    (np.array([True, False]), 3, "length"),
    (np.array([[True], [False]]), 2, "1-D"),
])
def test_validate_keep_genes_rejects(bad, n, match):
    with pytest.raises(ValueError, match=match):
        validate_keep_genes(bad, n)


def test_x_has_noncount_signal_dense_fractional_last_element():
    # the LAST element is fractional; the linspace sample must include it
    x = np.ones(10_000, dtype=np.float32).reshape(100, 100)
    x[99, 99] = 2.5
    assert x_has_noncount_signal(x, k=1000) is True


def test_x_has_noncount_signal_dense_negative():
    x = np.ones((4, 4), dtype=np.float32)
    x[0, 0] = -1.0
    assert x_has_noncount_signal(x) is True


def test_x_has_noncount_signal_integer_float32_clean():
    x = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.float32)
    assert x_has_noncount_signal(x) is False


def test_x_has_noncount_signal_sparse_uses_data():
    import scipy.sparse as sp
    x = sp.csr_matrix(np.array([[0, 1.5], [2, 0]], dtype=np.float32))
    assert x_has_noncount_signal(x) is True


# _row_scale_needs(scale_main, min_cpm_cell, min_cpm_bulk, median_requested)
#   -> (need_row_sums_np, need_row_scales_t)
# row_scales_t (GPU 1e6/L tensor + per-group row-index tensors) is only used to
# normalize the test tensor (scale_main) or the per-cell-CPM unit (cpm_cell).
# cpm_bulk needs only the CPU row sums (-> group_libtot), NOT row_scales_t.
@pytest.mark.parametrize("scale_main,min_cpm_cell,min_cpm_bulk,expected", [
    (False, None, None, (False, False)),   # no CPM anything -> nothing
    (True,  None, None, (True,  True)),     # normalize scales the test tensor
    (False, 1.0,  None, (True,  True)),     # cpm_cell needs the per-cell scale
    (False, None, 1.0,  (True,  False)),    # cpm_bulk-only: row sums yes, scale NO
    (False, 1.0,  1.0,  (True,  True)),     # cell forces the scale tensor
    (True,  None, 1.0,  (True,  True)),     # normalize forces the scale tensor
    (True,  1.0,  None, (True,  True)),
    (True,  1.0,  1.0,  (True,  True)),
])
def test_row_scale_needs_truth_table(scale_main, min_cpm_cell, min_cpm_bulk,
                                     expected):
    assert _row_scale_needs(
        scale_main, min_cpm_cell, min_cpm_bulk, median_requested=False) == expected


def test_x_has_noncount_signal_never_copies_a_dense_matrix(monkeypatch):
    """The function samples at most k values; it must not materialise a copy of X.

    np.ravel defaults to order='C', so it COPIED any non-C-contiguous dense input
    in full -- ~40 GB for a 500k x 20k f32 group, inside a function whose whole
    job is to read <=100_000 values.

    This SPIES on the flatten call rather than re-deriving it: an assertion that
    calls np.ravel(X, order='K') itself tests numpy, not gpudge, and passes on the
    broken implementation (the first version of this test did exactly that --
    codex caught it). What discriminates is (a) the order actually passed, and
    (b) whether the returned buffer shares memory with the input, both observed
    from inside the call. The strided branch must not flatten at all.
    (ultrareview 2026-08; test corrected after the codex review.)
    """
    import numpy as np
    from gpudge import _filter

    base = np.arange(60_000, dtype=np.float32).reshape(300, 200)
    layouts = {
        "C": np.ascontiguousarray(base),
        "F": np.asfortranarray(base),
        "strided": base[::2, ::3],           # neither C- nor F-contiguous
    }

    calls = []
    real_ravel = np.ravel

    def ravel_spy(a, order=None):
        out = real_ravel(a) if order is None else real_ravel(a, order=order)
        calls.append((order, bool(np.shares_memory(out, a))))
        return out

    monkeypatch.setattr(np, "ravel", ravel_spy)

    for name, X in layouts.items():
        calls.clear()
        assert _filter.x_has_noncount_signal(X) is False, name
        if name == "strided":
            assert calls == [], (
                f"strided input flattened anyway: {calls} -- it must be sampled "
                f"through unravel_index with no flatten at all")
        else:
            assert len(calls) == 1, f"{name}: expected one flatten, got {calls}"
            order, shared = calls[0]
            assert order == "K", (
                f"{name}: np.ravel called with order={order!r}, not 'K' -- "
                f"order='C' copies a non-C-contiguous array in full")
            assert shared, f"{name}: the flatten returned a COPY, not a view"

    # A fractional value must still be detected in every layout, with the layout
    # PRESERVED (the earlier version rebuilt the strided case as C-contiguous, so
    # the strided branch was never exercised with a fraction).
    frac = {
        "C": np.ascontiguousarray(base),
        "F": np.asfortranarray(base),
        "strided": base.copy()[::2, ::3],
    }
    for name, Y in frac.items():
        Y[1, 1] = 0.5
        assert Y.flags["C_CONTIGUOUS"] == (name == "C"), name
        assert Y.flags["F_CONTIGUOUS"] == (name == "F"), name
        assert _filter.x_has_noncount_signal(Y) is True, name

    # Negative values too, same layouts.
    for name, Y in frac.items():
        Y[1, 1] = -3.0
        assert _filter.x_has_noncount_signal(Y) is True, name


# --- 2026-08 ultrareview (lows): the ALL_OTHERS zero-library-total guards ----
#
# `_all_others_chunk_keep` guards BOTH cpm-bulk denominators
# (`libtot_safe`, `rest_libtot_safe`). Their only observable effect is the
# absence of a 0/0 RuntimeWarning: a group with zero library total also has
# zero counts, so the guarded quantity is 0 where the unguarded one is NaN --
# and `_one_filter_mask` uses a strict `>`, under which 0 and NaN compare
# identically against every threshold >= 0 while a threshold < 0 short-circuits
# to all-True. So the keep MASK cannot distinguish them on any input, which is
# why the (GPU-gated) integration test in test_review_coverage.py pins the
# warning rather than a value. This is the CPU-runnable half of that pin.

def _bulk_keep_with_empty_rest(threshold):
    from gpudge import _all_others_chunk_keep
    ch = 6
    arith = np.array([[1.0, 2.0, 0.0, 3.0, 4.0, 5.0]])   # ONE group = everyone
    return _all_others_chunk_keep(
        0, ch, ch, arith, None,
        np.array([40.0]),          # counts
        np.array([1.0]),           # rest_count_safe (already guarded upstream)
        np.array([600.0]),         # group_libtot -> rest_libtot == 0
        None, None, None, None, threshold, None)


@pytest.mark.parametrize("threshold", [0.0, 1.0, 1e5])
def test_all_others_bulk_keep_survives_empty_rest(threshold):
    """rest_libtot == 0 must not divide by zero."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        keep = _bulk_keep_with_empty_rest(threshold)
    assert keep.shape == (1, 6)
    assert keep.dtype == bool


def test_all_others_bulk_keep_survives_a_zero_library_group():
    """The twin guard: group_libtot == 0 is the TARGET denominator."""
    from gpudge import _all_others_chunk_keep
    ch = 4
    arith = np.array([[0.0, 0.0, 0.0, 0.0],      # g0 contributes nothing
                      [1.0, 2.0, 3.0, 4.0]])
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        keep = _all_others_chunk_keep(
            0, ch, ch, arith, None,
            np.array([10.0, 30.0]),              # counts
            np.array([30.0, 10.0]),              # rest_count_safe
            np.array([0.0, 300.0]),              # group_libtot: g0 is empty
            None, None, None, None, 1.0, None)
    assert keep.shape == (2, 4)
