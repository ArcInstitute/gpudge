"""layout='cell' streaming backend (#110)."""
from __future__ import annotations

import warnings

import gpudge
import numpy as np
import pytest

from conftest import _make_synth, needs_cuda

cellstream = pytest.importorskip("cellstream", reason="requires gpudge[streaming]")

# The cell writer warns that the format is experimental on every write; that is
# a property of the format, not of anything gpudge does.
_EXPERIMENTAL = "experimental flat per-cell format"


def _write_cell(adata, path, *, group_by, reference):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*" + _EXPERIMENTAL + ".*")
        cellstream.write_sharded(adata, str(path), layout="cell",
                              group_by=group_by, reference=reference, n_workers=2)
    return str(path)


@pytest.fixture
def cell_mode1(tmp_path):
    """Cell archive WITH a reference group (comparison='ntc')."""
    adata = _make_synth(n_cells=600, n_genes=40, n_guides=8, sparse=True, seed=1)
    p = _write_cell(adata, tmp_path / "m1.csad",
                    group_by="comparison", reference="ntc")
    return p, adata


@pytest.fixture
def cell_mode2(tmp_path):
    """Guide-only cell archive WITHOUT a reference + a separate NTC AnnData."""
    full = _make_synth(n_cells=600, n_genes=40, n_guides=8, sparse=True, seed=2)
    is_ntc = full.obs["comparison"].to_numpy().astype(str) == "ntc"
    adata_g = full[~is_ntc].copy()
    adata_ntc = full[is_ntc].copy()
    p = _write_cell(adata_g, tmp_path / "m2.csad",
                    group_by="comparison", reference=None)
    return p, adata_g, adata_ntc, full


def test_cell_group_ranges_shape(cell_mode1):
    """cellstream's public group_spans() has the shape gpudge depends on, and the
    ranges tile [0, n_obs) with the reference leading. This test is the tripwire
    for cellstream changing CellStore.group_spans(), and -- because it reads a
    real archive -- for the manifest's `reference` drifting from the labels the
    spans carry."""
    from gpudge._cell_stream import _cell_group_ranges
    p, adata = cell_mode1
    store = cellstream.open(p)
    try:
        ref_labels, ranges = _cell_group_ranges(store)
        assert ref_labels == ["ntc"]
        assert ranges[0][0] == "ntc" and ranges[0][1] == 0
        pos = 0
        for _lab, start, stop in ranges:
            assert start == pos
            pos = stop
        assert pos == store.n_obs
        comp = adata.obs["comparison"].to_numpy().astype(str)
        assert {lab for lab, _, _ in ranges} == set(comp)
        for lab, start, stop in ranges:
            assert stop - start == int((comp == lab).sum())
    finally:
        store.close()


def test_cell_group_ranges_no_reference(cell_mode2):
    from gpudge._cell_stream import _cell_group_ranges
    p, adata_g, _, _ = cell_mode2
    store = cellstream.open(p)
    try:
        ref_labels, ranges = _cell_group_ranges(store)
        assert ref_labels is None
        assert {lab for lab, _, _ in ranges} == set(
            adata_g.obs["comparison"].to_numpy().astype(str))
    finally:
        store.close()


class _FakeStore:
    """Minimal CellStore stand-in for the guard tests.

    ``groups`` is a list of ``(label, start, stop)`` triples, handed back as real
    ``cellstream.GroupSpan`` values so the fake cannot drift from the type the
    store actually returns -- a plain tuple would keep passing after GroupSpan
    grew a field or stopped unpacking.
    """
    path = "<fake>"

    def __init__(self, groups, reference, n_obs):
        self._spans = [cellstream.GroupSpan(*g) for g in groups]
        self.manifest = {"reference": reference}
        self.n_obs = n_obs

    def group_spans(self):
        return list(self._spans)


@pytest.mark.parametrize("groups, reference, n_obs, match", [
    # gap between two groups
    ([("a", 0, 5), ("b", 7, 10)], None, 10, "tile"),
    # overlap
    ([("a", 0, 5), ("b", 3, 10)], None, 10, "tile"),
    # ranges stop short of n_obs
    ([("a", 0, 5)], None, 10, "cover"),
    # duplicate label -- cellstream ALLOWS these (several records can share a
    # stringified label); gpudge cannot. Not because the spans collapse:
    # _targets keeps BOTH, while _tgt_index resolves both to the LAST, so one
    # accumulator row is never written and the other is overwritten span by
    # span. See _cell_group_ranges. (Copilot review, PR #146.)
    ([("a", 0, 5), ("a", 5, 10)], None, 10, "duplicate"),
    # reference label absent from the group table
    ([("a", 0, 10)], "zzz", 10, "not in the group table"),
    # reference group is NOT leading -- cellstream's read_reference() would gather
    # [0, ref_stop) and silently pull group 'a' into the reference pool
    ([("a", 0, 5), ("ntc", 5, 10)], "ntc", 10, "leading"),
    # empty group table
    ([], None, 0, "no group table"),
])
def test_cell_group_ranges_guards(groups, reference, n_obs, match):
    from gpudge._cell_stream import _cell_group_ranges
    with pytest.raises(ValueError, match=match):
        _cell_group_ranges(_FakeStore(groups, reference, n_obs))


@pytest.mark.parametrize("manifest", [None, object(), 42],
                         ids=["none", "no-get", "not-a-mapping"])
def test_cell_group_ranges_bad_manifest_is_wrapped(manifest):
    """A `manifest` that is not a mapping lands in the SAME wrapper as a broken
    group_spans(), because one try covers both -- so the message has to name both
    surfaces. Before it did, this case reported that group_spans() had not
    returned the expected spans, blaming the call that succeeded.
    (Gemini review, PR #146.)"""
    from gpudge._cell_stream import _cell_group_ranges

    class _BadManifest:
        path = "<fake>"
        n_obs = 10

        def group_spans(self):
            return [cellstream.GroupSpan("a", 0, 10)]

    store = _BadManifest()
    store.manifest = manifest
    with pytest.raises(RuntimeError, match="manifest"):
        _cell_group_ranges(store)


def test_cell_group_ranges_without_path_still_explains():
    """A store lacking `path` must still get the explanatory error.

    Every message in _cell_group_ranges names the archive, and each one used to
    read `store.path` directly. On a store whose contract has moved far enough to
    drop it that raised a bare AttributeError instead -- in the except block it
    would additionally have masked the original exception. They all route through
    one getattr-computed local now, so the guard still speaks.
    (Gemini review, PR #146.)"""
    from gpudge._cell_stream import _cell_group_ranges

    class _NoPath:                          # deliberately no `path`
        n_obs = 10
        manifest = {"reference": None}

        def group_spans(self):              # a gap: does not tile [0, n_obs)
            return [cellstream.GroupSpan("a", 0, 4),
                    cellstream.GroupSpan("b", 6, 10)]

    with pytest.raises(ValueError, match="<unknown>.*tile"):
        _cell_group_ranges(_NoPath())


def test_cell_group_ranges_decodes_bytes_labels():
    """A bytes label must decode, not stringify to "b'ntc'".

    `str(b"ntc")` is the Python 3 str(bytes) trap: the label would reach the
    output frame as a name nobody wrote, and the reference-membership comparison
    would be made between two reprs. Upstream types labels as `str`, so this is
    contract insurance -- but BOTH sides of that comparison (the spans and the
    manifest's reference) now normalize through the same `label_str` helper, and
    this pins it. (Gemini review, PR #146.)"""
    from gpudge._cell_stream import _cell_group_ranges

    ref_labels, ranges = _cell_group_ranges(
        _FakeStore([(b"ntc", 0, 4), (b"A", 4, 10)], b"ntc", 10))
    assert ref_labels == ["ntc"]
    assert ranges == [("ntc", 0, 4), ("A", 4, 10)]


def test_cell_group_ranges_public_api_gone():
    """A cellstream that drops group_spans() must fail loudly, not silently."""
    from gpudge._cell_stream import _cell_group_ranges

    class _NoGroups:
        path = "<fake>"
        n_obs = 10
        manifest = {"reference": None}

    with pytest.raises(RuntimeError, match="group_spans"):
        _cell_group_ranges(_NoGroups())


@pytest.mark.parametrize("bad", [
    ("a", 0),                       # too few values to unpack -> ValueError
    ("a", 0, 5, 7),                 # too many -> ValueError
    ("a", 0, "five"),               # non-numeric bound -> ValueError from int()
    ("a", 0, float("inf")),         # non-finite bound -> OverflowError from int()
    None,                           # not iterable at all -> TypeError
], ids=["short", "long", "bad-bound", "inf-bound", "not-a-span"])
def test_cell_group_ranges_malformed_span_is_wrapped(bad):
    """A malformed SPAN is the same class of failure as group_spans() itself
    changing shape, and must get the same explanatory RuntimeError rather than a
    bare TypeError/ValueError escaping a list comprehension.

    Uses a raw list rather than _FakeStore. GroupSpan is a NamedTuple, so its
    annotations reject nothing -- `GroupSpan("a", 0, "five")` constructs fine --
    but the arity cases cannot be built through it at all, so the fake would
    raise in its own constructor rather than in the code under test. The guard is
    for a cellstream whose contract has moved, not for one that is behaving.

    `inf-bound` is the case that motivated widening the except clause: `int()`
    raises OverflowError, not ValueError, on a non-finite float, so before that
    widening this input escaped the wrapper entirely. (codex.)"""
    from gpudge._cell_stream import _cell_group_ranges

    class _BadSpans:
        path = "<fake>"
        n_obs = 5
        manifest = {"reference": None}

        def group_spans(self):
            return [bad]

    with pytest.raises(RuntimeError, match="group_spans"):
        _cell_group_ranges(_BadSpans())


def test_plan_batches_groups_are_never_split():
    from gpudge._cell_stream import _plan_batches
    ranges = [("a", 0, 10), ("b", 10, 25), ("c", 25, 30)]
    batches = _plan_batches(ranges, bytes_per_row=1, target_bytes=12)
    seen = []
    for row_start, row_stop, members in batches:
        assert members, "no empty batches"
        assert members[0][1] == 0
        assert members[-1][2] == row_stop - row_start
        for lab, lstart, lstop in members:
            seen.append((lab, row_start + lstart, row_start + lstop))
    assert seen == ranges                       # every group whole, in order


def test_plan_batches_respects_budget():
    from gpudge._cell_stream import _plan_batches
    ranges = [(f"g{i}", i * 5, i * 5 + 5) for i in range(10)]   # 10 groups x 5 rows
    batches = _plan_batches(ranges, bytes_per_row=1, target_bytes=12)
    assert [rs for rs, _, _ in batches] == [0, 10, 20, 30, 40]  # 2 groups per batch
    assert all(rt - rs <= 12 for rs, rt, _ in batches)


def test_plan_batches_oversized_group_gets_its_own_batch():
    from gpudge._cell_stream import _plan_batches
    ranges = [("small", 0, 2), ("huge", 2, 1002), ("tail", 1002, 1004)]
    batches = _plan_batches(ranges, bytes_per_row=1, target_bytes=10)
    labels = [[lab for lab, _, _ in members] for _, _, members in batches]
    assert ["huge"] in labels
    assert sum(len(m) for m in labels) == 3


def test_plan_batches_empty():
    from gpudge._cell_stream import _plan_batches
    assert _plan_batches([], bytes_per_row=8) == []


def test_plan_batches_target_bytes_resolved_at_call_time(monkeypatch):
    """The module budget must be patchable -- the batch-invariance GPU gate
    depends on monkeypatching it actually changing the plan."""
    from gpudge import _cell_stream as cs
    ranges = [(f"g{i}", i * 5, i * 5 + 5) for i in range(10)]
    monkeypatch.setattr(cs, "_BATCH_TARGET_BYTES", 5)
    few = cs._plan_batches(ranges, bytes_per_row=1)
    monkeypatch.setattr(cs, "_BATCH_TARGET_BYTES", 1 << 30)
    many = cs._plan_batches(ranges, bytes_per_row=1)
    assert len(few) == 10 and len(many) == 1


def test_cell_bytes_per_nnz_by_dtype():
    """float64 archives are preserved by CellStore; everything else lands as
    float32. The batch budget has to know which."""
    from gpudge._cell_stream import _cell_bytes_per_nnz
    assert _cell_bytes_per_nnz({"value_dtype_on_disk": "uint16"}) == 8
    assert _cell_bytes_per_nnz({"value_dtype_on_disk": "float32"}) == 8
    assert _cell_bytes_per_nnz({"value_dtype_on_disk": "float64"}) == 12


@pytest.fixture
def cell_zstd(tmp_path):
    """A FRACTIONAL float32 source forces the zstd codec (schema 6), not
    pfordelta -- otherwise no test ever reaches the non-pfordelta path."""
    adata = _make_synth(n_cells=200, n_genes=20, n_guides=4, sparse=True, seed=3)
    adata.X = adata.X.astype(np.float32)
    adata.X.data = adata.X.data + np.float32(0.5)
    p = _write_cell(adata, tmp_path / "z.csad",
                    group_by="comparison", reference="ntc")
    return p, adata


def test_zstd_cell_archive_backend(cell_zstd):
    """Schema 6 / zstd opens, enumerates and gathers like schema 5 -- and the
    decoded VALUES are right. Asserting only the manifest and a row count would
    pass for a decoder returning shuffled or zero-filled data."""
    from gpudge._stream_backend import open_backend
    p, adata = cell_zstd
    store = cellstream.open(p)
    try:
        assert store.manifest["codec"] == "zstd"
        assert store.manifest["schema_version"] == 6
    finally:
        store.close()
    b = open_backend(p, n_workers=2, prefetch=0)
    try:
        comp = adata.obs["comparison"].to_numpy().astype(str)
        targets, _ = b.targets()
        assert set(targets) == {c for c in set(comp) if c != "ntc"}
        ref_X, _ = b.resolve_archive_reference("comparison", None)
        np.testing.assert_array_equal(ref_X.toarray(),
                                      adata[comp == "ntc"].X.toarray())
        for g, X, rows, _Ls in b.target_source(False):
            np.testing.assert_array_equal(
                X[rows].toarray(), adata[comp == targets[g]].X.toarray())
    finally:
        b.close()


def test_open_backend_dispatches_on_manifest_not_extension(tmp_path):
    """A cell archive named .shad and a shard archive named .csad must both
    dispatch correctly -- extension is not the signal."""
    from gpudge._cell_stream import _CellBackend
    from gpudge._shard_stream import _ShardBackend
    from gpudge._stream_backend import open_backend
    adata = _make_synth(n_cells=200, n_genes=20, n_guides=4, sparse=True, seed=4)
    liar_cell = _write_cell(adata, tmp_path / "actually_cell.shad",
                            group_by="comparison", reference="ntc")
    liar_shard = str(tmp_path / "actually_shard.csad")
    cellstream.write_sharded(adata, liar_shard, format="v2", group_by="comparison",
                          reference=["ntc"], target_shard_bytes=4096, n_workers=2)
    b1 = open_backend(liar_cell, n_workers=2, prefetch=0)
    b2 = open_backend(liar_shard, n_workers=2, prefetch=0)
    try:
        assert isinstance(b1, _CellBackend)
        assert isinstance(b2, _ShardBackend)
    finally:
        b1.close()
        b2.close()


def test_cell_backend_metadata(cell_mode1):
    from gpudge._cell_stream import _CellBackend
    from gpudge._stream_backend import open_backend
    p, adata = cell_mode1
    b = open_backend(p, n_workers=2, prefetch=2)
    try:
        assert isinstance(b, _CellBackend)
        assert b.n_vars == adata.n_vars
        assert np.array_equal(b.var_names, np.asarray(adata.var_names))
        assert b.group_by == "comparison"
        assert b.has_archive_reference is True
        assert b.supports_device_decode is False
    finally:
        b.close()


def test_cell_backend_close_releases_the_store(cell_mode1):
    """CellStore owns an mmap + file handle; the backend must close it."""
    from gpudge._stream_backend import open_backend
    p, _ = cell_mode1
    b = open_backend(p, n_workers=2, prefetch=0)
    store = b._store
    b.close()
    with pytest.raises(ValueError, match="mmap closed"):
        store.gather_rows(np.arange(0, 1, dtype=np.int64))


def test_cell_backend_close_keeps_the_handle_when_close_fails(cell_mode1):
    """close() must not drop its cleanup handle on a failed close.

    cellstream's CellStore.close() raises BufferError while an mmap view is still
    exported, so clearing _store before the call would leave the backend unable
    to retry and the mmap leaked."""
    from gpudge._stream_backend import open_backend
    p, _ = cell_mode1
    b = open_backend(p, n_workers=2, prefetch=0)
    boom = [True]
    real = b._store.close

    def flaky():
        if boom[0]:
            boom[0] = False
            raise BufferError("cannot close exported pointers exist")
        real()

    b._store.close = flaky
    with pytest.raises(BufferError):
        b.close()
    assert b._store is not None, "handle discarded on a failed close"
    b.close()                                    # retry now succeeds
    assert b._store is None


def test_open_backend_closes_the_store_if_construction_fails(cell_mode1, monkeypatch):
    """A store that opens but whose backend construction then raises must not
    leak the mmap."""
    from gpudge import _cell_stream as cs
    from gpudge._stream_backend import open_backend
    p, _ = cell_mode1
    closed = []
    real_ranges = cs._cell_group_ranges

    def boom(store):
        real_ranges(store)                       # exercise the real read first
        orig = store.close
        store.close = lambda: (closed.append(1), orig())[1]
        raise ValueError("synthetic construction failure")

    monkeypatch.setattr(cs, "_cell_group_ranges", boom)
    with pytest.raises(ValueError, match="synthetic"):
        open_backend(p, n_workers=2, prefetch=0)
    assert closed == [1], "open_backend leaked the CellStore"


def test_cell_backend_targets_exclude_reference(cell_mode1):
    from gpudge._stream_backend import open_backend
    p, adata = cell_mode1
    b = open_backend(p, n_workers=2, prefetch=0)
    try:
        targets, max_group_rows = b.targets()
        comp = adata.obs["comparison"].to_numpy().astype(str)
        assert set(targets) == {c for c in set(comp) if c != "ntc"}
        assert max_group_rows == max(int((comp == t).sum()) for t in targets)
    finally:
        b.close()


def test_cell_backend_targets_mode2_includes_all(cell_mode2):
    from gpudge._stream_backend import open_backend
    p, adata_g, _, _ = cell_mode2
    b = open_backend(p, n_workers=2, prefetch=0)
    try:
        assert b.has_archive_reference is False
        targets, _ = b.targets()
        assert set(targets) == set(
            adata_g.obs["comparison"].to_numpy().astype(str))
    finally:
        b.close()


def test_cell_backend_resolve_reference_matches_adata(cell_mode1):
    from gpudge._stream_backend import open_backend
    p, adata = cell_mode1
    b = open_backend(p, n_workers=2, prefetch=0)
    try:
        ref_X, msg = b.resolve_archive_reference("comparison", None)
        comp = adata.obs["comparison"].to_numpy().astype(str)
        assert ref_X.shape == (int((comp == "ntc").sum()), adata.n_vars)
        assert msg == "ntc"
        np.testing.assert_array_equal(ref_X.toarray(),
                                      adata[comp == "ntc"].X.toarray())
    finally:
        b.close()


def test_cell_backend_unknown_reference_label_raises(cell_mode1):
    from gpudge._stream_backend import open_backend
    p, _ = cell_mode1
    b = open_backend(p, n_workers=2, prefetch=0)
    try:
        with pytest.raises(ValueError, match="not among the archive's reference"):
            b.resolve_archive_reference("comparison", "nope")
    finally:
        b.close()


def test_cell_backend_target_source_covers_every_group_once(cell_mode1):
    from gpudge._stream_backend import open_backend
    p, adata = cell_mode1
    b = open_backend(p, n_workers=2, prefetch=0)
    try:
        targets, _ = b.targets()
        comp = adata.obs["comparison"].to_numpy().astype(str)
        seen = {}
        for g, X, rows, Ls in b.target_source(True):
            assert g not in seen, "a group must be yielded exactly once"
            lab = targets[g]
            seen[g] = len(rows)
            assert len(rows) == int((comp == lab).sum())
            np.testing.assert_array_equal(
                X[rows].toarray(), adata[comp == lab].X.toarray())
            np.testing.assert_allclose(
                Ls, np.asarray(adata[comp == lab].X.sum(axis=1)).ravel())
        assert set(seen) == set(range(len(targets)))
    finally:
        b.close()


def test_cell_backend_target_row_sums_order(cell_mode1):
    from gpudge._stream_backend import open_backend
    p, adata = cell_mode1
    b = open_backend(p, n_workers=2, prefetch=0)
    try:
        targets, _ = b.targets()
        comp = adata.obs["comparison"].to_numpy().astype(str)
        expected = np.concatenate([
            np.asarray(adata[comp == lab].X.sum(axis=1)).ravel()
            for lab in targets])
        np.testing.assert_allclose(b.target_row_sums(), expected)
    finally:
        b.close()


def test_cell_backend_resolve_streaming_modes(cell_mode1, cell_mode2):
    from gpudge import _shard_stream as ss
    from gpudge._stream_backend import open_backend
    p1, adata1 = cell_mode1
    b1 = open_backend(p1, n_workers=2, prefetch=0)
    try:
        groupby, mode, ref_X, _ = ss._resolve_streaming(b1, None, None)
        assert (groupby, mode) == ("comparison", "archive_ref")
        comp = adata1.obs["comparison"].to_numpy().astype(str)
        assert ref_X.shape[0] == int((comp == "ntc").sum())
    finally:
        b1.close()

    p2, _, adata_ntc, _ = cell_mode2
    b2 = open_backend(p2, n_workers=2, prefetch=0)
    try:
        _gb, mode, ref_X, _ = ss._resolve_streaming(b2, None, adata_ntc)
        assert mode == "external_ref"
        assert ref_X.shape[0] == adata_ntc.n_obs
    finally:
        b2.close()


def test_cell_backend_external_ref_on_reference_bearing_archive_warns(cell_mode1):
    from gpudge import _shard_stream as ss
    from gpudge._stream_backend import open_backend
    p, adata = cell_mode1
    b = open_backend(p, n_workers=2, prefetch=0)
    try:
        with pytest.warns(UserWarning, match="ignored in favor of the external"):
            _gb, mode, ref_X, _ = ss._resolve_streaming(b, None, adata)
        assert mode == "external_ref"
        assert ref_X.shape[0] == adata.n_obs
    finally:
        b.close()


def test_cell_backend_groupby_mismatch_raises(cell_mode1):
    from gpudge import _shard_stream as ss
    from gpudge._stream_backend import open_backend
    p, _ = cell_mode1
    b = open_backend(p, n_workers=2, prefetch=0)
    try:
        with pytest.raises(ValueError, match="does not match the archive"):
            ss._resolve_streaming(b, "target_guide", None)
    finally:
        b.close()


def test_cell_backend_external_ref_gene_axis_mismatch_raises(cell_mode2):
    """Mode 2 gene-axis guards, reached through the CELL backend.

    The shard suite covers the shared resolver, but this coverage is owed to
    the cell backend specifically -- and it is the backend's
    n_vars/var_names that the resolver compares against."""
    from gpudge import _shard_stream as ss
    from gpudge._stream_backend import open_backend
    p, _adata_g, adata_ntc, _ = cell_mode2
    b = open_backend(p, n_workers=2, prefetch=0)
    try:
        narrow = adata_ntc[:, : adata_ntc.n_vars - 1].copy()      # wrong n_vars
        with pytest.raises(ValueError, match="genes but the archive"):
            ss._resolve_streaming(b, None, narrow)
        shuffled = adata_ntc[:, ::-1].copy()                      # right count, wrong order
        with pytest.raises(ValueError, match="var_names do not match"):
            ss._resolve_streaming(b, None, shuffled)
    finally:
        b.close()


def _write_shard(adata, path, *, group_by, reference):
    cellstream.write_sharded(adata, str(path), format="v2",
                          group_by=group_by, reference=reference,
                          target_shard_bytes=4096, n_workers=2)
    return str(path)


@pytest.fixture
def twin_mode1(tmp_path):
    """The SAME AnnData written both ways, with DEFAULT ordering.

    Default ordering was load-bearing under the pre-0.8.0 writer: its cell
    writer converted categorical sort keys with .to_numpy() (losing category
    order) where the shard writer used .values (cellstream #252), so a categorical
    sort_within_group ordered rows differently between layouts and moved the
    float64 group means. Fixed upstream in 0.8.0, below the >=0.9.0 floor. The
    fixture keeps default ordering anyway: it is what the twin-mode exactness
    promise is stated against, and pinning it here keeps this test independent
    of that upstream fix."""
    adata = _make_synth(n_cells=600, n_genes=40, n_guides=8, sparse=True, seed=1)
    cell = _write_cell(adata, tmp_path / "t1.csad",
                       group_by="comparison", reference="ntc")
    shard = _write_shard(adata, tmp_path / "t1.shad",
                         group_by="comparison", reference=["ntc"])
    return cell, shard, adata


@pytest.fixture
def twin_mode2(tmp_path):
    full = _make_synth(n_cells=600, n_genes=40, n_guides=8, sparse=True, seed=2)
    is_ntc = full.obs["comparison"].to_numpy().astype(str) == "ntc"
    adata_g = full[~is_ntc].copy()
    adata_ntc = full[is_ntc].copy()
    cell = _write_cell(adata_g, tmp_path / "t2.csad",
                       group_by="comparison", reference=None)
    shard = _write_shard(adata_g, tmp_path / "t2.shad",
                         group_by="comparison", reference=None)
    return cell, shard, adata_g, adata_ntc


def _srt(df):
    return df.sort(["target", "feature"])


def _assert_exact(a, b):
    from polars.testing import assert_frame_equal
    assert_frame_equal(_srt(a), _srt(b), check_exact=True)


@needs_cuda
@pytest.mark.parametrize("kw", [
    {},
    {"cpm_normalize": True},
    {"normalize_target_sum": "median"},
], ids=["plain", "cpm", "median"])
def test_cross_layout_identical_mode1(twin_mode1, kw):
    """MERGE GATE: cell layout and shard layout produce byte-identical de()
    output from the same data (default ordering). Both writers preserve input
    row order within a group, so even the float64 group means are reduced over
    identical row orders -- exact, not `close`."""
    cell, shard, _ = twin_mode1
    _assert_exact(gpudge.de(archive=cell, **kw), gpudge.de(archive=shard, **kw))


@needs_cuda
def test_cross_layout_identical_mode2(twin_mode2):
    cell, shard, _, adata_ntc = twin_mode2
    _assert_exact(gpudge.de(archive=cell, reference=adata_ntc),
                  gpudge.de(archive=shard, reference=adata_ntc))


@needs_cuda
def test_cell_stream_matches_in_memory_mode1(cell_mode1):
    from test_shard_stream import _assert_equiv
    p, adata = cell_mode1
    _assert_equiv(gpudge.de(archive=p),
                  gpudge.de(adata, groupby="comparison", reference="ntc"))


@needs_cuda
def test_cell_stream_matches_in_memory_mode2(cell_mode2):
    from test_shard_stream import _assert_equiv
    p, adata_g, adata_ntc, _full = cell_mode2
    _assert_equiv(gpudge.de(archive=p, reference=adata_ntc),
                  gpudge.de(adata_g, groupby="comparison", reference=adata_ntc))


@needs_cuda
def test_cell_stream_chunk_size_invariance(cell_mode1):
    p, _ = cell_mode1
    _assert_exact(gpudge.de(archive=p, gpu_gene_chunk_size=7),
                  gpudge.de(archive=p, gpu_gene_chunk_size=40))


@needs_cuda
def test_cell_stream_batch_size_invariance(cell_mode1, monkeypatch):
    """One batch per group vs one batch for everything: identical output.

    Asserts the two settings really produce different batch counts first --
    otherwise this compares two identical runs and passes vacuously."""
    from gpudge import _cell_stream as cs
    from gpudge._stream_backend import open_backend
    p, _ = cell_mode1

    monkeypatch.setattr(cs, "_BATCH_TARGET_BYTES", 1)          # one group per batch
    b = open_backend(p, n_workers=2, prefetch=0)
    n_tiny = len(b._batches)
    b.close()
    tiny = gpudge.de(archive=p)

    monkeypatch.setattr(cs, "_BATCH_TARGET_BYTES", 1 << 40)    # everything in one
    b = open_backend(p, n_workers=2, prefetch=0)
    n_huge = len(b._batches)
    b.close()
    huge = gpudge.de(archive=p)

    assert n_tiny > n_huge == 1, f"batch plan did not change: {n_tiny} vs {n_huge}"
    _assert_exact(tiny, huge)


# --- 2026-08 ultrareview (lows): multi-label reference pool ------------------

@pytest.fixture
def cell_multiref(tmp_path):
    """Cell archive whose reference pool spans TWO labels (ntc + safe)."""
    adata = _make_synth(n_cells=600, n_genes=40, n_guides=8, sparse=True, seed=1)
    comp = adata.obs["comparison"].to_numpy().astype(str)
    # Re-label one target group as a second control so the pool has 2 labels.
    # Via a python list, NOT in place: the array is fixed-width ('<U3' here),
    # so `comp[mask] = "safe"` silently truncates to 'saf'.
    second = sorted(set(comp) - {"ntc"})[0]
    adata.obs["comparison"] = np.array(
        ["safe" if c == second else c for c in comp], dtype=object).astype(str)
    p = _write_cell(adata, tmp_path / "multiref.csad",
                    group_by="comparison", reference=["ntc", "safe"])
    return p, adata


def test_cell_backend_accepts_a_list_naming_the_whole_pool(cell_multiref):
    """`reference=['ntc','safe']` used to produce the self-contradictory
    `... is not among the archive's reference labels ['ntc','safe']`."""
    from gpudge._stream_backend import open_backend
    p, adata = cell_multiref
    b = open_backend(p, n_workers=2, prefetch=0)
    try:
        ref_X, msg = b.resolve_archive_reference("comparison", ["ntc", "safe"])
        comp = adata.obs["comparison"].to_numpy().astype(str)
        n_ref = int(((comp == "ntc") | (comp == "safe")).sum())
        assert ref_X.shape == (n_ref, adata.n_vars)
        assert msg == "ntc|safe"
        # ... and it must be the same pool reference=None resolves to.
        ref_none, msg_none = b.resolve_archive_reference("comparison", None)
        assert msg_none == msg
        np.testing.assert_array_equal(ref_X.toarray(), ref_none.toarray())
    finally:
        b.close()


def test_cell_backend_accepts_one_label_of_a_multi_label_pool(cell_multiref):
    """Specified: membership-only validation, pool used whole. Pinned so the
    contract is visible at the backend level, not just in the pure helper."""
    from gpudge._stream_backend import open_backend
    p, adata = cell_multiref
    b = open_backend(p, n_workers=2, prefetch=0)
    try:
        ref_none, _ = b.resolve_archive_reference("comparison", None)
        for partial, want in (("ntc", "ntc"), (["safe"], "safe")):
            ref_X, msg = b.resolve_archive_reference("comparison", partial)
            # ... and it really is the WHOLE pool, not the named label's rows.
            np.testing.assert_array_equal(ref_X.toarray(), ref_none.toarray())
            assert msg == want
    finally:
        b.close()
