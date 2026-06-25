"""Unit tests for the pure normalize-target-sum resolution helpers."""
import numpy as np
import pytest

from gpudge._filter import _row_scale_needs
from gpudge._normalize import resolve_target_sum, cpm_rescale_factor


def test_both_none_returns_none():
    assert resolve_target_sum(cpm_normalize=False, normalize_target_sum=None,
                              row_sums=None) is None


def test_cpm_normalize_resolves_to_1e6():
    assert resolve_target_sum(cpm_normalize=True, normalize_target_sum=None,
                              row_sums=None) == 1e6


def test_numeric_target_passthrough():
    assert resolve_target_sum(cpm_normalize=False, normalize_target_sum=5e5,
                              row_sums=None) == 500000.0


def test_numeric_int_is_floated():
    out = resolve_target_sum(cpm_normalize=False, normalize_target_sum=10000,
                             row_sums=None)
    assert out == 10000.0
    assert isinstance(out, float)


def test_both_set_raises():
    with pytest.raises(ValueError, match="only one"):
        resolve_target_sum(cpm_normalize=True, normalize_target_sum=1e6,
                           row_sums=None)


def test_nonpositive_number_raises():
    with pytest.raises(ValueError, match="positive"):
        resolve_target_sum(cpm_normalize=False, normalize_target_sum=0,
                           row_sums=None)
    with pytest.raises(ValueError, match="positive"):
        resolve_target_sum(cpm_normalize=False, normalize_target_sum=-3.0,
                           row_sums=None)


def test_non_finite_number_raises():
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="finite"):
            resolve_target_sum(cpm_normalize=False, normalize_target_sum=bad,
                               row_sums=None)


def test_bad_string_raises():
    with pytest.raises(ValueError, match="median"):
        resolve_target_sum(cpm_normalize=False, normalize_target_sum="mean",
                           row_sums=None)


def test_median_over_positive_rows():
    # row sums 0, 2, 4, 6 -> median over {2,4,6} = 4.0 (0 excluded)
    rs = np.array([0.0, 2.0, 4.0, 6.0])
    assert resolve_target_sum(cpm_normalize=False, normalize_target_sum="median",
                              row_sums=rs) == 4.0


def test_median_requires_row_sums():
    with pytest.raises(ValueError, match="row_sums"):
        resolve_target_sum(cpm_normalize=False, normalize_target_sum="median",
                           row_sums=None)


def test_median_all_zero_raises():
    with pytest.raises(ValueError, match="positive total counts"):
        resolve_target_sum(cpm_normalize=False, normalize_target_sum="median",
                           row_sums=np.zeros(5))


def test_cpm_rescale_factor():
    assert cpm_rescale_factor(1e6) == 1.0
    assert cpm_rescale_factor(5e5) == 2.0


def test_row_scale_needs_median_forces_row_sums():
    # No scaling, no filters, but median requested -> need row sums (for median),
    # not row scales.
    need_sums, need_scales = _row_scale_needs(
        scale_main=False, min_cpm_cell=None, min_cpm_bulk=None,
        median_requested=True)
    assert need_sums is True
    assert need_scales is False


def test_row_scale_needs_scale_main():
    need_sums, need_scales = _row_scale_needs(
        scale_main=True, min_cpm_cell=None, min_cpm_bulk=None,
        median_requested=False)
    assert need_sums is True
    assert need_scales is True


def test_row_scale_needs_cpm_cell_only():
    need_sums, need_scales = _row_scale_needs(
        scale_main=False, min_cpm_cell=1.0, min_cpm_bulk=None,
        median_requested=False)
    assert need_sums is True
    assert need_scales is True


def test_row_scale_needs_none():
    need_sums, need_scales = _row_scale_needs(
        scale_main=False, min_cpm_cell=None, min_cpm_bulk=None,
        median_requested=False)
    assert need_sums is False
    assert need_scales is False
