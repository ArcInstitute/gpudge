# tests/test_taustar_naming.py
"""tau_star level validation and canonical column naming (pure, no GPU)."""
import pytest

from gpudge._taustar import (
    DEFAULT_TAUSTAR_ITERS, TAUSTAR_SE_LEVEL, normalize_taustar_iters,
    normalize_taustar_se, normalize_taustar_spec, taustar_column_names,
)


def test_none_passes_through():
    assert normalize_taustar_spec(None) is None


def test_scalar_becomes_one_level():
    assert normalize_taustar_spec(0.5) == (0.5,)


def test_levels_sort_ascending_regardless_of_input_order():
    assert normalize_taustar_spec([0.5, 0.05]) == (0.05, 0.5)
    assert normalize_taustar_spec([0.05, 0.5]) == (0.05, 0.5)


def test_column_names_use_the_p_prefix_and_g_formatting():
    assert taustar_column_names((0.05, 0.5)) == ["tau*_p0.05", "tau*_p0.5"]
    assert taustar_column_names((0.025,)) == ["tau*_p0.025"]
    assert taustar_column_names((0.5000,)) == ["tau*_p0.5"]


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5, float("nan"),
                                 float("inf")])
def test_levels_outside_the_open_unit_interval_are_rejected(bad):
    with pytest.raises(ValueError, match="tau_star"):
        normalize_taustar_spec(bad)


def test_empty_sequence_is_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        normalize_taustar_spec([])


@pytest.mark.parametrize("bad", ["0.5", b"0.5"])
def test_strings_are_rejected_not_parsed_as_a_level(bad):
    """`float | Iterable[float]` excludes str, but str is iterable and would
    otherwise be coerced into a single level by float("0.5")."""
    with pytest.raises(ValueError, match="string"):
        normalize_taustar_spec(bad)


def test_exact_duplicates_are_rejected_not_silently_deduped():
    with pytest.raises(ValueError, match="duplicate"):
        normalize_taustar_spec([0.5, 0.5])


def test_levels_colliding_after_g_formatting_are_rejected():
    # %g carries 6 significant figures.
    with pytest.raises(ValueError, match="collide"):
        normalize_taustar_spec([0.5, 0.5 + 1e-12])


def test_iters_defaults_and_validation():
    assert normalize_taustar_iters(None) == DEFAULT_TAUSTAR_ITERS
    assert normalize_taustar_iters(8) == 8
    with pytest.raises(ValueError, match="tau_star_iters"):
        normalize_taustar_iters(0)
    with pytest.raises(ValueError, match="tau_star_iters"):
        normalize_taustar_iters(-3)


def test_levels_accept_any_non_string_iterable():
    """np.ndarray is NOT a collections.abc.Sequence, so a Sequence-only test
    sends an array of levels down the SCALAR branch, where float() on it raises
    a bare TypeError. An array is a natural way to pass levels; generators and
    sets come along for free."""
    import numpy as np
    assert normalize_taustar_spec(np.array([0.5, 0.05])) == (0.05, 0.5)
    assert normalize_taustar_spec(x for x in (0.5, 0.05)) == (0.05, 0.5)
    assert normalize_taustar_spec({0.5, 0.05}) == (0.05, 0.5)


def test_numpy_scalars_are_still_treated_as_a_single_level():
    """The iterable branch must not swallow the scalar case: np.float64 and a
    0-d array are not iterable, so both stay single levels."""
    import numpy as np
    assert normalize_taustar_spec(np.float64(0.5)) == (0.5,)
    assert normalize_taustar_spec(np.array(0.5)) == (0.5,)


def test_a_failing_iterator_is_not_silently_demoted_to_a_scalar():
    """_as_seq must catch TypeError from ACQUIRING the iterator only, never
    from consuming it: a try/except wrapped around tuple(value) also swallows a
    TypeError raised mid-iteration, discarding the items already yielded and
    treating the half-consumed generator as a single level. Mirror of the same
    guard in test_lfc_naming.py -- the two _as_seq helpers stay in step (#108)."""
    def boom():
        yield 0.5
        raise TypeError("from inside the generator")

    with pytest.raises(TypeError, match="from inside the generator"):
        normalize_taustar_spec(boom())


def test_numpy_integer_iters_are_accepted():
    """operator.index() must not be so strict it rejects a numpy scalar --
    np.int64 is the natural type when the count comes from an array."""
    import numpy as np
    assert normalize_taustar_iters(np.int64(12)) == 12


@pytest.mark.parametrize("bad", [1.9, 2.0, "7", b"7", True, False])
def test_non_integer_iters_are_rejected_not_truncated(bad):
    """int(1.9) == 1 and int("7") == 7 both SUCCEED, so a plain int() coercion
    would silently collapse the bisection to a single step. That is a
    materially wrong answer, not a rounding -- 2.0 is rejected too, because
    accepting it would mean the type check depends on the fractional part.

    bool is in the list because it PASSES operator.index() (it IS an int), so
    it needs its own guard: True would otherwise mean one bisection step."""
    with pytest.raises(ValueError, match="tau_star_iters"):
        normalize_taustar_iters(bad)


# --- tau_star_se: validation ----------------------------------------------

def test_normalize_taustar_se_accepts_the_bool_singletons():
    assert normalize_taustar_se(True) is True
    assert normalize_taustar_se(False) is False


@pytest.mark.parametrize("bad", [1, 0, "true", "", None, 1.0, [], object()])
def test_normalize_taustar_se_rejects_truthy_and_falsy_non_bools(bad):
    """Truthy coercion would make tau_star_se='false' mean True."""
    with pytest.raises(ValueError, match="tau_star_se must be True or False"):
        normalize_taustar_se(bad)


def test_normalize_taustar_se_accepts_numpy_bools():
    """np.bool_ is NOT a bool subclass, but tau_star already accepts numpy
    scalar levels and tau_star_iters accepts numpy integers -- rejecting a
    numpy bool here would be gratuitously inconsistent. _taustar.py stays
    stdlib-only, so this is a `.item()` duck-test, not an isinstance check."""
    np = pytest.importorskip("numpy")
    assert normalize_taustar_se(np.True_) is True
    assert normalize_taustar_se(np.False_) is False
    assert normalize_taustar_se(np.array(True)) is True


def test_normalize_taustar_se_still_rejects_numpy_non_bools():
    """The duck-test must be precise: np.int64(1).item() is an int."""
    np = pytest.importorskip("numpy")
    with pytest.raises(ValueError, match="tau_star_se must be True or False"):
        normalize_taustar_se(np.int64(1))
    with pytest.raises(ValueError, match="tau_star_se must be True or False"):
        normalize_taustar_se(np.float64(1.0))


def test_normalize_taustar_se_rejects_one_element_vectors():
    """numpy's and torch's .item() BOTH unwrap a 1-element vector, so an
    ndim-based guard is required -- without it np.array([True]) and even
    np.array([[True]]) pass as scalars."""
    np = pytest.importorskip("numpy")
    for bad in (np.array([True]), np.array([[True]])):
        with pytest.raises(ValueError,
                           match="tau_star_se must be True or False"):
            normalize_taustar_se(bad)


def test_normalize_taustar_se_accepts_a_0d_torch_bool():
    torch = pytest.importorskip("torch")
    assert normalize_taustar_se(torch.tensor(True)) is True
    assert normalize_taustar_se(torch.tensor(False)) is False
    with pytest.raises(ValueError, match="tau_star_se must be True or False"):
        normalize_taustar_se(torch.tensor([True]))


def test_se_without_tau_star_is_an_error():
    with pytest.raises(ValueError, match="tau_star_se=True requires tau_star"):
        normalize_taustar_spec(None, taustar_se=True)


def test_none_still_passes_through_when_se_is_off():
    assert normalize_taustar_spec(None, taustar_se=False) is None


# --- tau_star_se: level set ----------------------------------------------

def test_se_forces_the_point_estimate_level_into_the_set():
    assert normalize_taustar_spec(0.05, taustar_se=True) == (0.05, 0.5)


def test_se_does_not_duplicate_an_already_requested_point_estimate():
    assert normalize_taustar_spec([0.5, 0.05], taustar_se=True) == (0.05, 0.5)


def test_se_inserts_the_point_estimate_in_ascending_position():
    assert normalize_taustar_spec([0.9, 0.05], taustar_se=True) == (0.05, 0.5, 0.9)


def test_duplicate_rejection_still_applies_under_se():
    """The 0.5 union must not swallow a genuine caller duplicate."""
    with pytest.raises(ValueError, match="duplicate"):
        normalize_taustar_spec([0.05, 0.05], taustar_se=True)


# --- tau_star_se: column names -------------------------------------------

def test_se_level_formats_to_its_documented_name_fragment():
    from gpudge._taustar import _fmt_level
    assert _fmt_level(TAUSTAR_SE_LEVEL) == "0.025"


def test_se_appends_three_columns_after_the_level_columns():
    assert taustar_column_names((0.05, 0.5), se=True) == [
        "tau*_p0.05", "tau*_p0.5",
        "tau*_lo_p0.025", "tau*_hi_p0.025", "tau*_se",
    ]


def test_column_names_are_unchanged_when_se_is_off():
    assert (taustar_column_names((0.05, 0.5), se=False)
            == taustar_column_names((0.05, 0.5))
            == ["tau*_p0.05", "tau*_p0.5"])


def test_se_row_count_is_three_more_than_the_level_count():
    """The kernel writes rows in exactly this order; the count is the contract
    the accumulators are shaped from."""
    levels = normalize_taustar_spec([0.025, 0.05], taustar_se=True)
    assert len(taustar_column_names(levels, se=True)) == len(levels) + 3
