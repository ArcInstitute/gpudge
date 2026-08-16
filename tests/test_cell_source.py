"""CPU tests for de(cell_source=...) -- the public CellGroup contract, the
adapter that turns it into the internal target_source 4-tuple, and de()'s
mode validation.
"""
import warnings

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp

import gpudge
import gpudge._cell_source as _cs
import gpudge._refpool as _refpool
from gpudge._cell_source import (
    CellGroup, _as_cell_group, make_target_source,
)

TARGETS = np.asarray(["A", "B", "C"], dtype=str)


def _X(n_rows, n_genes=4, val=1.0):
    return sp.csr_matrix(np.full((n_rows, n_genes), val, dtype=np.float32))


def _source(*groups):
    def factory():
        yield from groups
    return factory


def test_cellgroup_defaults_rows_to_none():
    grp = CellGroup("TP53", _X(3))
    assert grp.label == "TP53"
    assert grp.rows is None
    assert grp._fields == ("label", "X", "rows")


def test_cellgroup_has_no_row_sums_field():
    """Deliberately cut in codex review round 1: gpudge cannot value-validate a
    caller-supplied row_sums, so accepting one re-opens the silent CPM
    divergence this API exists to close."""
    assert "row_sums" not in CellGroup._fields


def test_as_cell_group_passes_through_a_cellgroup():
    grp = CellGroup("A", _X(2))
    assert _as_cell_group(grp) is grp


@pytest.mark.parametrize("n_fields", [2, 3])
def test_as_cell_group_accepts_a_plain_tuple(n_fields):
    fields = ("A", _X(2), np.arange(2))[:n_fields]
    out = _as_cell_group(fields)
    assert isinstance(out, CellGroup)
    assert out.label == "A"


def test_as_cell_group_rejects_a_non_tuple():
    with pytest.raises(TypeError, match="CellGroup"):
        _as_cell_group("not a group")


@pytest.mark.parametrize("n_fields", [1, 4])
def test_as_cell_group_rejects_a_wrong_length_tuple(n_fields):
    with pytest.raises(TypeError, match="2-3 fields"):
        _as_cell_group(tuple(range(n_fields)))


def test_adapter_maps_labels_to_targets_order_not_yield_order():
    """The property the label-based contract buys: yield order is free."""
    src = _source(CellGroup("C", _X(1)), CellGroup("A", _X(2)),
                  CellGroup("B", _X(3)))
    got = [(g, X.shape[0]) for g, X, _r, _L in
           make_target_source(src, targets=TARGETS, n_genes=4)(False)]
    assert got == [(2, 1), (0, 2), (1, 3)]


def test_adapter_defaults_rows_to_all_rows():
    src = _source(CellGroup("A", _X(3)), CellGroup("B", _X(1)),
                  CellGroup("C", _X(1)))
    rows = [r for _g, _X, r, _L in
            make_target_source(src, targets=TARGETS, n_genes=4)(False)]
    assert np.array_equal(rows[0], np.arange(3))
    assert rows[0].dtype == np.int64


def test_adapter_computes_row_sums_when_needed():
    X = sp.csr_matrix(np.array([[1.0, 2.0, 0.0, 0.0],
                                [0.0, 0.0, 3.0, 4.0]], dtype=np.float32))
    src = _source(CellGroup("A", X), CellGroup("B", _X(1)),
                  CellGroup("C", _X(1)))
    Ls = [L for _g, _X, _r, L in
          make_target_source(src, targets=TARGETS, n_genes=4)(True)]
    assert np.allclose(Ls[0], [3.0, 7.0])
    assert Ls[0].dtype == np.float64


def test_adapter_returns_none_row_sums_when_not_needed():
    src = _source(CellGroup("A", _X(2)), CellGroup("B", _X(1)),
                  CellGroup("C", _X(1)))
    Ls = [L for _g, _X, _r, L in
          make_target_source(src, targets=TARGETS, n_genes=4)(False)]
    assert all(x is None for x in Ls)


def test_adapter_row_sums_cover_only_the_named_rows():
    """Row sums are per-row over `rows`, in rows order -- what
    _accumulate_target_group consumes (_refpool.py:196,199)."""
    shared = sp.csr_matrix(np.arange(20, dtype=np.float32).reshape(5, 4))
    src = _source(CellGroup("A", shared, np.asarray([0, 2])),
                  CellGroup("B", shared, np.asarray([1])),
                  CellGroup("C", shared, np.asarray([3, 4])))
    out = list(make_target_source(src, targets=TARGETS, n_genes=4)(True))
    assert np.array_equal(out[0][2], [0, 2])
    assert out[0][3].shape == (2,)
    assert np.allclose(out[0][3], [0 + 1 + 2 + 3, 8 + 9 + 10 + 11])


def _spy_row_sums(monkeypatch):
    """Record every matrix csr_row_sums is handed. The helper resolves the name
    as a module global at call time, so patching the module attribute works."""
    seen = []
    real = _cs.csr_row_sums

    def spy(X):
        seen.append(X)
        return real(X)

    monkeypatch.setattr(_cs, "csr_row_sums", spy)
    return seen


def test_adapter_does_not_rescan_a_shared_matrix_per_group(monkeypatch):
    """csr_row_sums(X)[rows] would be O(n_groups * nnz) on a shared matrix.
    Assert the sum is taken over the SLICE, not the whole matrix."""
    seen = _spy_row_sums(monkeypatch)
    shared = sp.csr_matrix(np.ones((100, 4), dtype=np.float32))
    src = _source(CellGroup("A", shared, np.asarray([0, 1])),
                  CellGroup("B", shared, np.asarray([2])),
                  CellGroup("C", shared, np.asarray([3])))
    list(make_target_source(src, targets=TARGETS, n_genes=4)(True))
    assert [int(m.shape[0]) for m in seen] == [2, 1, 1]


def test_adapter_takes_the_whole_matrix_when_rows_covers_it(monkeypatch):
    """The identity case must NOT pay a slice copy.

    Asserted by OBJECT IDENTITY, not row count: `X[np.arange(3)]` also has 3
    rows, so a row-count assertion would pass even if the implementation always
    sliced -- i.e. it would not test what it claims to.
    """
    seen = _spy_row_sums(monkeypatch)
    full_X = _X(3)
    src = _source(CellGroup("A", full_X, np.arange(3)),
                  CellGroup("B", _X(1)), CellGroup("C", _X(1)))
    list(make_target_source(src, targets=TARGETS, n_genes=4)(True))
    assert seen[0] is full_X


@pytest.mark.parametrize("empty", [[], (), np.empty(0, dtype=np.int64)])
def test_adapter_accepts_an_empty_rows_selector(empty):
    """A bare `[]` is float64 under np.asarray, so the dtype guard must let an
    UNTYPED empty sequence through: the core supports a zero-cell group
    (mwu_one_group returns zeros/NaN at m==0). A typed empty int array is
    accepted on its own merits."""
    src = _source(CellGroup("A", _X(3), empty), CellGroup("B", _X(1)),
                  CellGroup("C", _X(1)))
    out = list(make_target_source(src, targets=TARGETS, n_genes=4)(True))
    assert out[0][2].dtype == np.int64
    assert out[0][2].size == 0
    assert out[0][3].shape == (0,)


@pytest.mark.parametrize("bad", [np.empty(0, dtype=bool),
                                 np.empty(0, dtype=np.float64)])
def test_adapter_rejects_a_typed_empty_non_integer_selector(bad):
    """The empty-sequence shortcut must NOT swallow the dtype contract: an
    empty bool mask against a non-empty matrix is a caller error, not a
    zero-cell group."""
    src = _source(CellGroup("A", _X(3), bad))
    with pytest.raises(TypeError):
        list(make_target_source(src, targets=TARGETS, n_genes=4)(False))


def test_adapter_coerces_a_csc_target_to_csr():
    """CSR, a coerced CSC, and a dense ndarray are all
    accepted forms of CellGroup.X."""
    csc = sp.csc_matrix(np.ones((2, 4), dtype=np.float32))
    src = _source(CellGroup("A", csc), CellGroup("B", _X(1)),
                  CellGroup("C", _X(1)))
    with pytest.warns(UserWarning, match="cell_source group 'A' X"):
        out = list(make_target_source(src, targets=TARGETS, n_genes=4)(False))
    assert out[0][1].format == "csr"


def test_check_2d_reports_a_scalar_shape_as_its_own_error():
    """An object whose `shape` is a scalar: bare len() would raise
    'object of type int has no len()', naming no parameter."""
    class Weird:
        shape = 4

    src = _source(CellGroup("A", Weird()))
    with pytest.raises(TypeError, match="2-D"):
        list(make_target_source(src, targets=TARGETS, n_genes=4)(False))


@pytest.mark.parametrize("bad", [
    np.arange(4),                                    # 1-D array
    "not a matrix",
])
def test_adapter_rejects_a_non_2d_x(bad):
    """ensure_csr passes any non-scipy-sparse object through UNCHANGED
    (_csr_dense.py:170), so without an explicit check this surfaces later as an
    AttributeError or IndexError."""
    src = _source(CellGroup("A", bad))
    with pytest.raises(TypeError, match="2-D"):
        list(make_target_source(src, targets=TARGETS, n_genes=4)(False))


def test_adapter_rejects_a_2d_duck_array():
    """A duck array with a plausible 2-D shape (a dask array, a torch tensor,
    an h5py dataset) would pass a shape-only check, pass through ensure_csr
    unchanged, and fail much later on something unrelated. gpudge also cannot
    reason about its reduction order, which is load-bearing here."""
    class Duck:
        shape = (2, 4)

    src = _source(CellGroup("A", Duck()))
    with pytest.raises(TypeError, match="2-D"):
        list(make_target_source(src, targets=TARGETS, n_genes=4)(False))


def test_adapter_accepts_a_dense_ndarray_x():
    src = _source(CellGroup("A", np.ones((2, 4), dtype=np.float32)),
                  CellGroup("B", _X(1)), CellGroup("C", _X(1)))
    out = list(make_target_source(src, targets=TARGETS, n_genes=4)(True))
    assert np.allclose(out[0][3], [4.0, 4.0])


def test_adapter_accepts_unsorted_non_contiguous_rows():
    shared = sp.csr_matrix(np.arange(20, dtype=np.float32).reshape(5, 4))
    src = _source(CellGroup("A", shared, np.asarray([4, 0])),
                  CellGroup("B", _X(1)), CellGroup("C", _X(1)))
    out = list(make_target_source(src, targets=TARGETS, n_genes=4)(True))
    assert np.array_equal(out[0][2], [4, 0])
    assert np.allclose(out[0][3], [16 + 17 + 18 + 19, 0 + 1 + 2 + 3])


def test_adapter_slices_a_full_permutation_rather_than_taking_the_whole_X(
        monkeypatch):
    """`rows` covering every row OUT OF ORDER must NOT take the is_all fast
    path: a set-based or length-only check would pass, hand csr_row_sums the
    whole X, and pair sums in original order with reversed rows -- misaligning
    every library size. Asserted by object identity AND by sum order."""
    seen = _spy_row_sums(monkeypatch)
    shared = sp.csr_matrix(np.arange(12, dtype=np.float32).reshape(3, 4))
    src = _source(CellGroup("A", shared, np.asarray([2, 1, 0])),
                  CellGroup("B", _X(1)), CellGroup("C", _X(1)))
    out = list(make_target_source(src, targets=TARGETS, n_genes=4)(True))
    assert seen[0] is not shared
    assert np.array_equal(out[0][2], [2, 1, 0])
    assert np.allclose(out[0][3],
                       [8 + 9 + 10 + 11, 4 + 5 + 6 + 7, 0 + 1 + 2 + 3])


@pytest.mark.parametrize("kind", ["csr", "dense_c"])
def test_adapter_row_sums_equal_the_sibling_sum_then_slice_route(kind):
    """THE byte-identity assertion for library sizes.

    de(adata=, reference=) sums the whole matrix once and indexes the sums
    (_refpool._inmem_target_source); this mode sums the SLICE, to stay off an
    O(n_groups * nnz) rescan. Those routes must agree bit-for-bit, and
    `allclose` would not notice if they stopped -- the entire failure mode is a
    float64 low-bit one that becomes a float32 CPM-scale ULP. Wide and
    float64-valued on purpose: at 4000 genes numpy's pairwise summation blocks,
    which is where the two routes could start to differ."""
    rng = np.random.default_rng(3)
    base = rng.random((6, 4000)) * 1000.0
    X = sp.csr_matrix(base) if kind == "csr" else np.ascontiguousarray(base)
    rows = np.asarray([4, 1, 0])

    out = list(make_target_source(_source(CellGroup("A", X, rows)),
                                 targets=np.asarray(["A"]), n_genes=4000)(True))
    assert np.array_equal(out[0][3], _cs.csr_row_sums(X)[rows])


def _dense_f(seed=0, n=6, g=4000):
    return np.asfortranarray(np.random.default_rng(seed).random((n, g)) * 1000.0)


def _dense_unaligned(seed=1, n=6, g=8193):
    """A C-contiguous but UNALIGNED array: a valid ndarray over a byte buffer
    starting one byte in. numpy takes a different reduction path for it."""
    buf = bytearray(n * g * 8 + 8)
    X = np.frombuffer(buf, dtype=np.float64, offset=1, count=n * g).reshape(n, g)
    np.copyto(X, np.random.default_rng(seed).random((n, g)) * 1000.0)
    return X


# [4, 1, 0] drops a row; [5, 4, 3, 2, 1, 0] keeps them all but re-orders. BOTH
# reduce a C-contiguous slice rather than the F-ordered original, so "strict
# subset" would have been the wrong rule.
@pytest.mark.parametrize("rows", [np.asarray([4, 1, 0]),
                                  np.arange(6)[::-1].copy()])
def test_adapter_rejects_a_non_c_contiguous_dense_x_under_a_reordering_rows(
        rows):
    """A Fortran-ordered or strided dense X is one of the layouts where summing
    the slice and summing-then-slicing disagree (numpy walks them differently);
    an unaligned one is another, covered separately below. gpudge refuses rather
    than silently returning numbers that are not byte-identical to
    de(adata=, reference=). Same discipline as rejecting a bool mask: the cheap
    accommodation changes results invisibly.

    Guarded by an assertion that the divergence is real on this numpy, so the
    test cannot quietly become a tautology if numpy's reduction changes."""
    dense_f = _dense_f()
    assert not np.array_equal(_cs.csr_row_sums(dense_f)[rows],
                              _cs.csr_row_sums(dense_f[rows]))

    src = _source(CellGroup("A", dense_f, rows))
    with pytest.raises(ValueError, match="not C-contiguous"):
        list(make_target_source(src, targets=np.asarray(["A"]),
                                n_genes=4000)(True))


def test_adapter_rejects_a_c_contiguous_but_unaligned_dense_x():
    """C-contiguity is NOT sufficient. An ndarray over an offset byte buffer is
    C_CONTIGUOUS but not ALIGNED, gathering rows produces an aligned copy, and
    numpy reduces the two differently -- so this diverges while sitting inside
    a "C-contiguous dense" guarantee.

    np.ascontiguousarray is NOT the remedy here (it returns an already
    C-contiguous array unchanged, alignment and all), which is why the message
    names np.require(X, requirements=["C", "A"]) instead."""
    X = _dense_unaligned()
    rows = np.asarray([4, 1, 0])
    assert X.flags["C_CONTIGUOUS"] and not X.flags["ALIGNED"]
    assert not np.array_equal(_cs.csr_row_sums(X)[rows],
                              _cs.csr_row_sums(X[rows]))
    assert not np.ascontiguousarray(X).flags["ALIGNED"]      # remedy check

    with pytest.raises(ValueError, match="not C-contiguous and aligned"):
        list(make_target_source(_source(CellGroup("A", X, rows)),
                                targets=np.asarray(["A"]), n_genes=8193)(True))


def test_np_require_makes_an_unaligned_x_acceptable():
    """The remedy the error message names must actually work."""
    X = np.require(_dense_unaligned(), requirements=["C", "A"])
    rows = np.asarray([4, 1, 0])
    out = list(make_target_source(_source(CellGroup("A", X, rows)),
                                  targets=np.asarray(["A"]),
                                  n_genes=8193)(True))
    assert np.array_equal(out[0][3], _cs.csr_row_sums(X)[rows])


@pytest.mark.parametrize("rows", [None, np.arange(6)])
def test_adapter_accepts_a_non_c_contiguous_dense_x_that_rows_fully_covers(rows):
    """With rows=None (or a rows covering every row IN ORDER) this mode sums X
    itself, exactly as the sibling path does -- so the layout cannot matter and
    rejecting it would be gratuitous. Pinned by an EXACT comparison against
    that sibling route."""
    dense_f = _dense_f()
    out = list(make_target_source(_source(CellGroup("A", dense_f, rows)),
                                 targets=np.asarray(["A"]), n_genes=4000)(True))
    assert out[0][1] is dense_f              # no copy taken
    assert np.array_equal(out[0][3], _cs.csr_row_sums(dense_f))


def test_adapter_does_not_check_layout_when_no_row_sums_are_needed():
    """The guard exists to protect a REDUCTION. A default unnormalized run asks
    for no library sizes, so nothing reduces and there is nothing to diverge --
    rejecting there would refuse a run that cannot be wrong."""
    dense_f = _dense_f()
    out = list(make_target_source(_source(CellGroup("A", dense_f,
                                                    np.asarray([4, 1, 0]))),
                                 targets=np.asarray(["A"]),
                                 n_genes=4000)(False))
    assert out[0][3] is None


def test_adapter_accepts_an_empty_selector_against_a_non_c_contiguous_x():
    """Both row-sum routes return an empty array for an empty selector, so the
    guard must not reject one -- the contract says an empty rows is legal."""
    dense_f = _dense_f()
    out = list(make_target_source(
        _source(CellGroup("A", dense_f, np.empty(0, dtype=np.int64))),
        targets=np.asarray(["A"]), n_genes=4000)(True))
    assert out[0][3].shape == (0,)


def test_pregathering_from_a_fortran_parent_is_a_documented_parity_limit():
    """gpudge sums the matrix it is HANDED, and cannot see where it came from.

    parent_f[rows] is C-contiguous, so it is accepted with rows=None -- but its
    sums are NOT csr_row_sums(parent_f)[rows]. No guard can close this: the
    provenance is simply not in the input. This test exists so the limit is
    pinned rather than discovered, and so the docs and the error message stay
    honest about it (the message must NOT recommend pre-gathering as a way to
    preserve parity)."""
    parent_f = _dense_f(seed=5)
    rows = np.asarray([4, 1, 0])
    gathered = parent_f[rows]
    assert gathered.flags["C_CONTIGUOUS"]

    out = list(make_target_source(_source(CellGroup("A", gathered)),
                                 targets=np.asarray(["A"]), n_genes=4000)(True))
    # gpudge sums what it was given -- exactly, and that is the contract...
    assert np.array_equal(out[0][3], _cs.csr_row_sums(gathered))
    # ...but that is NOT what summing the Fortran parent and slicing gives.
    assert not np.array_equal(out[0][3], _cs.csr_row_sums(parent_f)[rows])
    # So the rejection message must not offer pre-gathering as a parity fix.
    with pytest.raises(ValueError) as exc:
        _cs._check_sliceable_layout(parent_f, rows, False, "X")
    assert "pre-gather" not in str(exc.value)


def test_adapter_leaves_a_c_contiguous_dense_x_alone():
    """The layout guard must not copy (or warn) on the normal dense case."""
    dense_c = np.ones((3, 4), dtype=np.float32)
    src = _source(CellGroup("A", dense_c, np.asarray([0, 2])),
                  CellGroup("B", _X(1)), CellGroup("C", _X(1)))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = list(make_target_source(src, targets=TARGETS, n_genes=4)(True))
    assert out[0][1] is dense_c


def test_adapter_rejects_an_unknown_label():
    src = _source(CellGroup("ZZZ", _X(1)))
    with pytest.raises(ValueError, match="'ZZZ'.*not in targets"):
        list(make_target_source(src, targets=TARGETS, n_genes=4)(False))


def test_adapter_rejects_a_duplicate_label():
    src = _source(CellGroup("A", _X(1)), CellGroup("A", _X(1)))
    with pytest.raises(ValueError, match="'A'.*more than once"):
        list(make_target_source(src, targets=TARGETS, n_genes=4)(False))


def test_adapter_rejects_a_never_yielded_target():
    """Without this the core emits a plausible U=0/p=1 row instead of failing."""
    src = _source(CellGroup("A", _X(1)))
    with pytest.raises(ValueError, match="did not yield"):
        list(make_target_source(src, targets=TARGETS, n_genes=4)(False))


def test_adapter_rejects_a_gene_axis_mismatch():
    src = _source(CellGroup("A", _X(1, n_genes=7)))
    with pytest.raises(ValueError, match="7 genes but var_names has 4"):
        list(make_target_source(src, targets=TARGETS, n_genes=4)(False))


@pytest.mark.parametrize("bad", [[-1], [3]])
def test_adapter_rejects_out_of_bounds_rows(bad):
    src = _source(CellGroup("A", _X(3), np.asarray(bad)))
    with pytest.raises(ValueError, match="out of bounds"):
        list(make_target_source(src, targets=TARGETS, n_genes=4)(False))


def test_adapter_rejects_float_rows():
    """np.asarray([1.9], dtype=np.int64) is [1] -- a DIFFERENT cell."""
    src = _source(CellGroup("A", _X(3), np.asarray([1.9, 0.0])))
    with pytest.raises(TypeError, match="integer"):
        list(make_target_source(src, targets=TARGETS, n_genes=4)(False))


def test_adapter_rejects_a_boolean_mask():
    """A mask cast to int64 becomes indices 0/1, silently selecting the wrong
    cells; point the caller at np.flatnonzero instead."""
    src = _source(CellGroup("A", _X(3), np.asarray([True, False, True])))
    with pytest.raises(TypeError, match="flatnonzero"):
        list(make_target_source(src, targets=TARGETS, n_genes=4)(False))


def test_adapter_rejects_non_1d_rows():
    src = _source(CellGroup("A", _X(4), np.arange(4).reshape(2, 2)))
    with pytest.raises(ValueError, match="1-D"):
        list(make_target_source(src, targets=TARGETS, n_genes=4)(False))


def test_adapter_rejects_duplicate_rows():
    """A repeated index would double-count that cell in every statistic."""
    src = _source(CellGroup("A", _X(3), np.asarray([1, 1])))
    with pytest.raises(ValueError, match="duplicate"):
        list(make_target_source(src, targets=TARGETS, n_genes=4)(False))


def test_adapter_warns_once_on_non_count_targets():
    """Parity with the in-memory external-ref path (_refpool.py:566-582), which
    the streaming path skips only because its shards are provably uint16."""
    frac = sp.csr_matrix(np.full((2, 4), 1.5, dtype=np.float32))
    src = _source(CellGroup("A", frac), CellGroup("B", frac),
                  CellGroup("C", frac))
    ts = make_target_source(src, targets=TARGETS, n_genes=4,
                            warn_noncount_targets=True)
    with pytest.warns(UserWarning, match="raw counts") as rec:
        list(ts(True))
    assert len(rec) == 1          # one warning, not one per group


def test_adapter_warns_at_most_once_across_two_drives():
    """The warned flag belongs to make_target_source's closure, not to a single
    target_source() call -- otherwise a second drive warns again."""
    frac = sp.csr_matrix(np.full((2, 4), 1.5, dtype=np.float32))
    src = _source(CellGroup("A", frac), CellGroup("B", frac),
                  CellGroup("C", frac))
    ts = make_target_source(src, targets=TARGETS, n_genes=4,
                            warn_noncount_targets=True)
    with pytest.warns(UserWarning, match="raw counts") as rec:
        list(ts(True))
        list(ts(True))
    assert len(rec) == 1


def test_adapter_inspects_only_the_selected_rows_for_the_warning():
    """A shared matrix may hold fractional values in rows NO group selects;
    sampling the whole matrix would warn about cells the run never touches."""
    shared = np.ones((4, 4), dtype=np.float32)
    shared[3, :] = 1.5                       # never selected below
    shared = sp.csr_matrix(shared)
    src = _source(CellGroup("A", shared, np.asarray([0])),
                  CellGroup("B", shared, np.asarray([1])),
                  CellGroup("C", shared, np.asarray([2])))
    ts = make_target_source(src, targets=TARGETS, n_genes=4,
                            warn_noncount_targets=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        list(ts(True))


def test_adapter_does_not_warn_when_not_asked():
    frac = sp.csr_matrix(np.full((2, 4), 1.5, dtype=np.float32))
    src = _source(CellGroup("A", frac), CellGroup("B", frac),
                  CellGroup("C", frac))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        list(make_target_source(src, targets=TARGETS, n_genes=4)(True))


def test_adapter_is_re_iterable():
    """The contract permits driving the source more than once."""
    calls = []

    def factory():
        calls.append(1)
        yield CellGroup("A", _X(1))
        yield CellGroup("B", _X(1))
        yield CellGroup("C", _X(1))

    ts = make_target_source(factory, targets=TARGETS, n_genes=4)
    assert len(list(ts(False))) == 3
    assert len(list(ts(False))) == 3
    assert len(calls) == 2


def _core_kw(**over):
    base = dict(targets=TARGETS, var_names=np.asarray(list("wxyz")),
                reference=_X(5), mean_calc="arithmetic", epsilon=1e-9,
                gpu_gene_chunk_size=None, oom_recovery=True,
                cpm_normalize=True, normalize_target_sum=None,
                output_columns=None, filter_gene_min_mean_value=None,
                filter_gene_min_total_value=None,
                filter_gene_min_cpm_cell=None, filter_gene_min_cpm_bulk=None,
                keep_genes=None, device="cpu")
    base.update(over)
    return base


def _abc_source():
    return _source(CellGroup("A", _X(2)), CellGroup("B", _X(1)),
                   CellGroup("C", _X(1)))


def test_cell_source_de_resolves_and_calls_the_core(monkeypatch):
    """Routing + resolution proven on CPU by stubbing the core (the pattern
    tests/test_inmem_external_ref.py:162 already uses)."""
    captured = {}

    def fake_core(**kw):
        captured.update(kw)
        captured["yielded"] = [g for g, _X, _r, _L in kw["target_source"](False)]
        return "SENTINEL"

    monkeypatch.setattr(_refpool, "refpool_de_core", fake_core)
    out = _refpool.cell_source_de(_abc_source(), **_core_kw())

    assert out == "SENTINEL"
    assert captured["yielded"] == [0, 1, 2]
    assert captured["n_genes"] == 4
    assert captured["target_sum"] == 1e6        # cpm_normalize=True resolved
    assert captured["keep_genes_arr"] is None
    assert list(captured["targets"]) == ["A", "B", "C"]
    # Passed EXPLICITLY, not left to the core's defaults -- so these assert a
    # real contract instead of passing vacuously when the key is absent.
    assert captured["uploader"] is None
    assert captured["max_group_rows"] == 0


def test_cell_source_de_drives_the_source_exactly_once(monkeypatch):
    """Nothing may re-drive the caller's source when median is not requested."""
    calls = []

    def factory():
        calls.append(1)
        yield CellGroup("A", _X(1))
        yield CellGroup("B", _X(1))
        yield CellGroup("C", _X(1))

    def fake_core(**kw):
        list(kw["target_source"](False))
        return "OK"

    monkeypatch.setattr(_refpool, "refpool_de_core", fake_core)
    assert _refpool.cell_source_de(factory, **_core_kw()) == "OK"
    assert len(calls) == 1


def _capture_target_source(monkeypatch, source, **over):
    captured = {}

    def fake_core(**kw):
        captured["ts"] = kw["target_source"]
        return None

    monkeypatch.setattr(_refpool, "refpool_de_core", fake_core)
    _refpool.cell_source_de(source, **_core_kw(**over))
    return captured["ts"]


def _frac_source():
    frac = sp.csr_matrix(np.full((2, 4), 1.5, dtype=np.float32))
    return _source(CellGroup("A", frac), CellGroup("B", frac),
                   CellGroup("C", frac))


def test_noncount_warning_is_off_without_a_cpm_filter(monkeypatch):
    ts = _capture_target_source(monkeypatch, _frac_source())
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        list(ts(True))


@pytest.mark.parametrize("flt", ["filter_gene_min_cpm_cell",
                                 "filter_gene_min_cpm_bulk"])
def test_noncount_warning_is_on_with_a_cpm_filter(monkeypatch, flt):
    ts = _capture_target_source(monkeypatch, _frac_source(), **{flt: 1.0})
    with pytest.warns(UserWarning, match="raw counts"):
        list(ts(True))


def test_cell_source_de_coerces_a_non_csr_reference(monkeypatch):
    monkeypatch.setattr(_refpool, "refpool_de_core", lambda **kw: kw["ref_X"])
    csc = sp.csc_matrix(np.ones((5, 4), dtype=np.float32))
    with pytest.warns(UserWarning, match="reference.X"):
        ref_out = _refpool.cell_source_de(
            _abc_source(), **_core_kw(reference=csc, cpm_normalize=False))
    assert ref_out.format == "csr"


def _kw(**over):
    base = dict(cell_source=_abc_source(), targets=TARGETS,
                var_names=np.asarray(list("wxyz")), reference=_X(5))
    base.update(over)
    return base


def test_cellgroup_is_public():
    assert gpudge.CellGroup is CellGroup
    assert "CellGroup" in gpudge.__all__


@pytest.mark.parametrize("over, match", [
    (dict(adata=object()), "exactly one of"),
    (dict(targets=None), "targets"),
    (dict(var_names=None), "var_names"),
    (dict(targets=np.asarray([], dtype=str)), "targets"),
    (dict(var_names=np.asarray([], dtype=str)), "var_names"),
    (dict(targets=np.asarray(["A", "A"])), "duplicate"),
    (dict(groupby="pert"), "groupby"),
    (dict(reference=None), "reference"),
    (dict(reference="A"), "reference"),
    (dict(reference=gpudge.ALL_OTHERS), "reference"),
    (dict(reference=sp.csr_matrix((0, 4), dtype=np.float32)), "0 cells"),
    (dict(reference=_X(5, n_genes=9)), "gene"),
])
def test_de_cell_source_validation(over, match):
    with pytest.raises(ValueError, match=match):
        gpudge.de(**_kw(**over))


@pytest.mark.parametrize("feature", [
    dict(tau_star=(0.5,)),
    dict(lfc_threshold=0.5),
])
def test_all_others_reports_the_byo_error_not_the_feature_error(feature):
    """de()'s ALL_OTHERS feature guards (__init__.py:747-762) fire on
    isinstance(reference, str) and would otherwise pre-empt this with
    NotImplementedError('tau_star is not supported with ALL_OTHERS') -- true,
    but beside the point: ALL_OTHERS is not a legal reference here at all."""
    with pytest.raises(ValueError, match="reference"):
        gpudge.de(**_kw(reference=gpudge.ALL_OTHERS, **feature))


def test_de_rejects_an_anndata_reference_with_a_permuted_gene_axis():
    """A count-only check would let this through and return confidently wrong
    per-gene results (the existing route checks order at __init__.py:866)."""
    names = np.asarray(list("wxyz"))
    ref = ad.AnnData(X=sp.csr_matrix(np.ones((5, 4), dtype=np.float32)))
    ref.var_names = names[::-1]
    with pytest.raises(ValueError, match="var_names"):
        gpudge.de(**_kw(reference=ref, var_names=names))


@pytest.mark.parametrize("bad", [np.arange(4), object()])
def test_de_rejects_a_non_2d_reference(bad):
    """Must raise on CPU, i.e. BEFORE de()'s CUDA check -- so the guard lives
    in de()'s BYO block, not only in cell_source_de. (A str reference is not
    in this list: it is caught earlier, by the reference-TYPE guard, with the
    more useful 'pass the control pool itself' message.)"""
    with pytest.raises(TypeError, match="2-D"):
        gpudge.de(**_kw(reference=bad))


def test_de_accepts_an_aligned_anndata_reference(monkeypatch):
    names = np.asarray(list("wxyz"))
    ref = ad.AnnData(X=sp.csr_matrix(np.ones((5, 4), dtype=np.float32)))
    ref.var_names = names
    monkeypatch.setattr(_refpool, "cell_source_de", lambda *a, **k: "OK")
    monkeypatch.setattr(gpudge.torch.cuda, "is_available", lambda: True)
    assert gpudge.de(**_kw(reference=ref, var_names=names)) == "OK"


def test_de_rejects_a_non_callable_cell_source():
    with pytest.raises(TypeError, match="cell_source"):
        gpudge.de(**_kw(cell_source=[CellGroup("A", _X(1))]))


def test_de_rejects_median_with_cell_source():
    with pytest.raises(NotImplementedError, match="median"):
        gpudge.de(**_kw(normalize_target_sum="median"))


def test_de_cell_source_routes_to_cell_source_de(monkeypatch):
    seen = {}

    def fake(cell_source, **kw):
        seen.update(kw)
        seen["cell_source"] = cell_source
        return "ROUTED"

    monkeypatch.setattr(_refpool, "cell_source_de", fake)
    monkeypatch.setattr(gpudge.torch.cuda, "is_available", lambda: True)
    assert gpudge.de(**_kw(cpm_normalize=True)) == "ROUTED"
    assert seen["cpm_normalize"] is True
    assert list(seen["targets"]) == ["A", "B", "C"]
