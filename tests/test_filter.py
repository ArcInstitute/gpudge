# tests/test_filter.py
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
    a_t = np.array([[5.0, 5.0, 0.0]]); a_r = np.array([0.0, 0.0, 0.0])
    b_t = np.array([[0.0, 5.0, 5.0]]); b_r = np.array([0.0, 0.0, 0.0])
    keep = combined_keep_mask(1, 3, filters=[(a_t, a_r, 1.0), (b_t, b_r, 1.0)])
    np.testing.assert_array_equal(keep, [[False, True, False]])


def test_no_filters_no_keep_genes_keeps_all():
    keep = combined_keep_mask(2, 3, filters=[])
    np.testing.assert_array_equal(keep, np.ones((2, 3), bool))


def test_keep_genes_anded_and_broadcast():
    tq = np.full((2, 3), 5.0); rq = np.zeros(3)
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
    x = np.ones((4, 4), dtype=np.float32); x[0, 0] = -1.0
    assert x_has_noncount_signal(x) is True


def test_x_has_noncount_signal_integer_float32_clean():
    x = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.float32)
    assert x_has_noncount_signal(x) is False


def test_x_has_noncount_signal_sparse_uses_data():
    import scipy.sparse as sp
    x = sp.csr_matrix(np.array([[0, 1.5], [2, 0]], dtype=np.float32))
    assert x_has_noncount_signal(x) is True


# _row_scale_needs(cpm_normalize, min_cpm_cell, min_cpm_bulk)
#   -> (need_row_sums_np, need_row_scales_t)
# row_scales_t (GPU 1e6/L tensor + per-group row-index tensors) is only used to
# CPM-scale the test tensor (cpm_normalize) or the per-cell-CPM unit (cpm_cell).
# cpm_bulk needs only the CPU row sums (-> group_libtot), NOT row_scales_t.
@pytest.mark.parametrize("cpm_normalize,min_cpm_cell,min_cpm_bulk,expected", [
    (False, None, None, (False, False)),   # no CPM anything -> nothing
    (True,  None, None, (True,  True)),     # cpm_normalize scales the test tensor
    (False, 1.0,  None, (True,  True)),     # cpm_cell needs the per-cell scale
    (False, None, 1.0,  (True,  False)),    # cpm_bulk-only: row sums yes, scale NO
    (False, 1.0,  1.0,  (True,  True)),     # cell forces the scale tensor
    (True,  None, 1.0,  (True,  True)),     # normalize forces the scale tensor
    (True,  1.0,  None, (True,  True)),
    (True,  1.0,  1.0,  (True,  True)),
])
def test_row_scale_needs_truth_table(cpm_normalize, min_cpm_cell, min_cpm_bulk,
                                     expected):
    assert _row_scale_needs(cpm_normalize, min_cpm_cell, min_cpm_bulk) == expected
