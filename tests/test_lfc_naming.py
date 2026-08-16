# tests/test_lfc_naming.py
import numpy as np
import pytest

from gpudge._lfc import (
    lfc_base_names, lfc_column_names, lfc_scale_factor, normalize_lfc_spec,
)


def test_none_threshold_yields_none():
    assert normalize_lfc_spec(None, ("up", "down")) is None


def test_scalar_and_length_one_sequence_agree():
    a = normalize_lfc_spec(0.5, ("up", "down"))
    b = normalize_lfc_spec([0.5], ("up", "down"))
    assert a == b == ((0.5, "down"), (0.5, "up"))


def test_grid_is_sorted_ascending_regardless_of_input_order():
    got = normalize_lfc_spec([0.5, 0.0, 0.25], ("up", "down"))
    assert [t for t, _ in got] == [0.0, 0.0, 0.25, 0.25, 0.5, 0.5]
    assert [d for _, d in got] == ["down", "up"] * 3


def test_single_direction():
    assert normalize_lfc_spec([1.0], ("up",)) == ((1.0, "up"),)
    assert normalize_lfc_spec(1.0, "down") == ((1.0, "down"),)


def test_tau_grid_accepts_any_non_string_iterable():
    """np.ndarray is NOT a collections.abc.Sequence, so a Sequence-only test
    sends an array of taus down the SCALAR branch, where float() on it raises a
    bare TypeError instead of this module's ValueError (issue #108). An array is
    a natural way to pass a grid; generators and sets come along for free, and
    the set case is deterministic because the grid is sorted (and duplicates are
    rejected outright, so no ordering question survives)."""
    want = ((0.25, "up"), (0.5, "up"))
    assert normalize_lfc_spec(np.array([0.25, 0.5]), ("up",)) == want
    assert normalize_lfc_spec((t for t in (0.25, 0.5)), ("up",)) == want
    assert normalize_lfc_spec({0.25, 0.5}, ("up",)) == want


def test_numpy_inputs_normalise_to_builtin_float_and_str():
    """Tuple equality alone would pass just as happily with np.float64/np.str_,
    since both compare equal to their builtins -- so pin the exact types. These
    values become column-name fragments and DataFrame labels downstream."""
    combos = normalize_lfc_spec(np.array([0.25, 0.5]), np.array(["up", "down"]))
    assert all(type(t) is float and type(d) is str for t, d in combos)


def test_empty_ndarrays_raise_this_modules_valueerror():
    """The empty-input guards must be REACHED by an ndarray, not pre-empted by
    a bare TypeError -- the point of #108 is that arrays take the list path."""
    with pytest.raises(ValueError, match="lfc_threshold must be a number"):
        normalize_lfc_spec(np.array([]), ("up",))
    with pytest.raises(ValueError, match="lfc_threshold_alt must name at least"):
        normalize_lfc_spec(0.5, np.array([]))


def test_a_failing_iterator_is_not_silently_demoted_to_a_scalar():
    """_as_seq must catch TypeError from ACQUIRING the iterator only, never
    from consuming it. A try/except wrapped around tuple(value) also swallows a
    TypeError raised mid-iteration, discarding the items already yielded and
    treating the half-consumed generator as a single tau."""
    def boom():
        yield 0.25
        raise TypeError("from inside the generator")

    with pytest.raises(TypeError, match="from inside the generator"):
        normalize_lfc_spec(boom(), ("up",))


def test_direction_set_accepts_a_non_string_iterable():
    """lfc_threshold_alt shares _as_seq, so it had the same defect from the
    other side: an ndarray of directions stayed whole, and `d not in
    LFC_DIRECTIONS` then raised numpy's ambiguous-truth ValueError."""
    assert normalize_lfc_spec(0.5, np.array(["up", "down"])) == (
        (0.5, "down"), (0.5, "up"))


def test_numpy_scalars_are_still_treated_as_a_single_tau():
    """The iterable branch must not swallow the scalar case: np.float64 and a
    0-d array are not iterable, so both stay single taus."""
    assert normalize_lfc_spec(np.float64(0.5), ("up",)) == ((0.5, "up"),)
    assert normalize_lfc_spec(np.array(0.5), ("up",)) == ((0.5, "up"),)


def test_a_string_direction_stays_one_direction_not_its_characters():
    """_lfc must KEEP _as_seq's str branch, unlike _taustar, which rejects
    strings outright: lfc_threshold_alt genuinely takes a bare string, and
    tuple("up") would split it into ('u', 'p')."""
    assert normalize_lfc_spec(0.5, "up") == ((0.5, "up"),)


@pytest.mark.parametrize(
    "bad", [-0.1, float("nan"), float("inf"), 30.0001, 1024.0])
def test_rejects_non_finite_negative_or_out_of_range_tau(bad):
    with pytest.raises(ValueError, match="lfc_threshold"):
        normalize_lfc_spec(bad, ("up",))


def test_accepts_tau_at_the_upper_bound():
    assert normalize_lfc_spec(30.0, ("up",)) == ((30.0, "up"),)


def test_direction_order_is_canonical_not_caller_order():
    a = normalize_lfc_spec(0.5, ("down", "up"))
    b = normalize_lfc_spec(0.5, ("up", "down"))
    assert a == b == ((0.5, "down"), (0.5, "up"))


def test_negative_zero_tau_normalised():
    got = normalize_lfc_spec(-0.0, ("up",))
    assert lfc_base_names(got)[0][0] == "tau=+0_p"


def test_rejects_unknown_direction():
    with pytest.raises(ValueError, match="lfc_threshold_alt"):
        normalize_lfc_spec(0.5, ("sideways",))


def test_rejects_duplicate_direction():
    with pytest.raises(ValueError, match="lfc_threshold_alt"):
        normalize_lfc_spec(0.5, ("up", "up"))


def test_rejects_empty_direction_set():
    with pytest.raises(ValueError, match="lfc_threshold_alt"):
        normalize_lfc_spec(0.5, ())


def test_rejects_duplicate_tau():
    with pytest.raises(ValueError, match="duplicate"):
        normalize_lfc_spec([0.5, 0.5], ("up",))


def test_rejects_suffix_collision():
    """%g carries 6 significant figures, so near-identical taus can collide."""
    with pytest.raises(ValueError, match="collide"):
        normalize_lfc_spec([0.1234567, 0.12345671], ("up",))


# ---- scale factors: TARGET scaled, factor INVERTED, computed in float64 ---

def test_scale_factor_direction_mapping_is_inverted():
    """up tests T vs R*2**(+tau), implemented as (T * 2**(-tau)) vs R."""
    assert lfc_scale_factor(1.0, "up") == 0.5
    assert lfc_scale_factor(1.0, "down") == 2.0
    assert lfc_scale_factor(2.0, "up") == 0.25


def test_scale_factor_is_float64_not_float32():
    """Spec 3.2b: the scaling is done in float64. A float32 factor would
    reintroduce the tie-boundary defect AND break the gc/run_start invariance
    the kernel relies on."""
    s = lfc_scale_factor(0.3, "up")
    assert type(s) is float                       # Python float == IEEE double
    assert s == 2.0 ** -0.3
    assert s != float(np.float32(2.0 ** -0.3))    # genuinely not the f32 value


def test_scale_factor_is_never_a_reciprocal():
    """ULP pinning: 2.0**(-tau), NEVER 1.0/(2.0**tau). At float64 the two
    genuinely differ for ~27% of taus (measured), so this rule has teeth --
    at float32 they are indistinguishable and the rule would be vacuous."""
    for tau in (0.25, 0.5, 0.7, 1.3, 30.0):
        assert lfc_scale_factor(tau, "up") == 2.0 ** -tau
        assert lfc_scale_factor(tau, "down") == 2.0 ** tau
    taus = np.linspace(0.01, 30.0, 40001)
    n_differ = sum(1 for t in taus if (2.0 ** -t) != (1.0 / (2.0 ** t)))
    assert n_differ > 0.2 * taus.size, (
        f"only {n_differ}/{taus.size} taus distinguish the two spellings; "
        "the ULP-pinning rule would be untestable")


def test_scale_factor_tau_zero_is_exactly_one():
    for d in ("up", "down"):
        assert lfc_scale_factor(0.0, d) == 1.0


def test_scale_factor_rejects_bad_direction():
    with pytest.raises(ValueError, match="direction"):
        lfc_scale_factor(0.5, "sideways")


# ---- naming --------------------------------------------------------------


def test_column_name_format():
    assert lfc_base_names([
        (0.0, "up"),
        (0.5, "down"),
        (1.0, "up"),
    ]) == [
        ("tau=+0_p", "tau=+0_Ueffect", "tau=+0_padj"),
        ("tau=-0.5_p", "tau=-0.5_Ueffect", "tau=-0.5_padj"),
        ("tau=+1_p", "tau=+1_Ueffect", "tau=+1_padj"),
    ]


def test_column_names_are_ordered_and_complete():
    combos = normalize_lfc_spec([0.25], ("up", "down"))
    assert lfc_base_names(combos) == [
        ("tau=-0.25_p", "tau=-0.25_Ueffect", "tau=-0.25_padj"),
        ("tau=+0.25_p", "tau=+0.25_Ueffect", "tau=+0.25_padj"),
    ]
    assert lfc_column_names(combos) == [
        "tau=-0.25_p", "tau=-0.25_Ueffect", "tau=-0.25_padj",
        "tau=+0.25_p", "tau=+0.25_Ueffect", "tau=+0.25_padj",
    ]
