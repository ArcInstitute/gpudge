# tests/test_shard_stream.py
from __future__ import annotations
import numpy as np
import pytest
import torch
import anndata as ad
import scipy.sparse as sp
import gpudge
from gpudge import ALL_OTHERS, _shard_stream as ss
from gpudge._mwu import mwu_one_group, _tie_term_per_gene
from conftest import LFC_TAUS, _make_synth, needs_cuda   # synth generator + cuda marker

# NOTE: cellstream-free tests (the pure _iter_kwargs helper, de()-level
# stream_n_workers/stream_prefetch guards) live in test_api.py — the module-level
# pytest.importorskip("cellstream")
# below skips this WHOLE module (incl. anything above it) when the streaming extra
# is absent, so those would get no cellstream-less CPU-CI coverage here.


def test_neither_adata_nor_archive_raises():
    with pytest.raises(ValueError, match="exactly one of adata= or archive="):
        gpudge.de()


def test_both_adata_and_archive_raises(synth_small):
    with pytest.raises(ValueError, match="exactly one of adata= or archive="):
        gpudge.de(synth_small, archive="/nonexistent", groupby="target_guide",
                  reference="non-targeting-0")


def test_densify_input_with_archive_raises(synth_small):
    with pytest.raises(ValueError, match="densify_input"):
        gpudge.de(archive="/nonexistent", densify_input=True)


def test_all_others_with_archive_raises():
    with pytest.raises(NotImplementedError, match="ALL_OTHERS"):
        gpudge.de(archive="/nonexistent", reference=ALL_OTHERS)


def test_anndata_reference_in_memory_now_supported(synth_small):
    # In-memory AnnData reference is now supported (feat/inmem-external-reference);
    # it must validate the gene axis rather than blanket-rejecting. A same-object
    # reference has matching var_names, so a var-MISMATCH proves the new path is
    # reached (a var-slice of synth_small has the wrong n_vars).
    bad = synth_small[:, : synth_small.n_vars - 1].copy()
    with pytest.raises(ValueError, match="gene"):
        gpudge.de(synth_small, groupby="target_guide", reference=bad)


def test_min_feature_filter_with_archive_raises():
    with pytest.raises(ValueError, match="min_feature_filter was removed"):
        gpudge.de(shard_archive="/nonexistent", min_feature_filter=0.5)


def test_legacy_all_others_with_archive_warns_and_raises():
    """The pre-v0.1 reference='all_others' spelling must be remapped (with a
    DeprecationWarning) BEFORE the streaming dispatch, so the SAME
    NotImplementedError fires as for the ALL_OTHERS sentinel. Regression for the
    ultrareview finding: previously the remap ran only on the in-memory path, so
    streaming skipped the warning + NotImplementedError and fell through to a
    misleading 'not among the archive's reference labels' error (and, worst
    case, a real group literally named 'all_others' would be used as a literal
    reference instead of triggering 1-vs-rest)."""
    with pytest.warns(DeprecationWarning, match=r"reference='all_others' is deprecated"):
        with pytest.raises(NotImplementedError, match="ALL_OTHERS"):
            gpudge.de(archive="/nonexistent", reference="all_others")


def test_duplicate_output_columns_with_archive_raises():
    """The output_columns duplicate-destination check must apply to the
    streaming path too (hoisted into de() above the dispatch). Previously
    streaming validated only unknown keys, so a duplicate destination surfaced
    as an opaque polars DuplicateError after compute. Regression for the
    ultrareview finding."""
    with pytest.raises(ValueError, match="same name"):
        gpudge.de(shard_archive="/nonexistent",
                  output_columns={"target": "x", "feature": "x"})


def test_unknown_output_columns_with_archive_raises():
    """Unknown output_columns keys are rejected on the streaming path too."""
    with pytest.raises(KeyError, match="output_columns"):
        gpudge.de(shard_archive="/nonexistent", output_columns={"p_values": "p"})


# ---------------------------------------------------------------------------
# Task-2: archive fixtures + _resolve_streaming tests
# ---------------------------------------------------------------------------
# cellstream is the optional [streaming] extra; skip the whole streaming-test
# module cleanly when it isn't installed instead of erroring at collection.
cellstream = pytest.importorskip("cellstream", reason="requires gpudge[streaming]")


def _write_archive(adata, path, *, group_by, reference, target_shard_bytes):
    cellstream.write_sharded(
        adata, str(path), format="v2",
        group_by=group_by, reference=reference,
        target_shard_bytes=target_shard_bytes,
    )
    return str(path)


@pytest.fixture
def archive_mode1(tmp_path):
    """Small sparse archive WITH a reference shard (comparison='ntc').
    target_shard_bytes tiny → several guide shards."""
    adata = _make_synth(n_cells=600, n_genes=40, n_guides=8, sparse=True, seed=1)
    d = _write_archive(adata, tmp_path / "m1", group_by="comparison",
                       reference=["ntc"], target_shard_bytes=4096)
    return d, adata


@pytest.fixture
def archive_mode2(tmp_path):
    """Guide-only archive WITHOUT a reference + a separate NTC AnnData."""
    full = _make_synth(n_cells=600, n_genes=40, n_guides=8, sparse=True, seed=2)
    is_ntc = np.char.startswith(full.obs["comparison"].to_numpy().astype(str), "ntc")
    adata_g = full[~is_ntc].copy()
    adata_ntc = full[is_ntc].copy()
    d = _write_archive(adata_g, tmp_path / "m2", group_by="comparison",
                       reference=None, target_shard_bytes=4096)
    return d, adata_g, adata_ntc, full


def test_resolve_mode1_basic(archive_mode1):
    from gpudge._stream_backend import open_backend
    d, adata = archive_mode1
    b = open_backend(d, n_workers=2, prefetch=0)
    groupby, mode, ref_X, _ = ss._resolve_streaming(b, None, None)
    assert groupby == "comparison"
    assert mode == "archive_ref"
    n_ntc = int(np.char.startswith(
        adata.obs["comparison"].to_numpy().astype(str), "ntc").sum())
    assert ref_X.shape[0] == n_ntc


def test_should_device_decode_true_on_real_packed_archive(archive_mode1, monkeypatch):
    """Regression for the schema_version-3 selector bug: a real v0.5.x archive
    written by cellstream.write_sharded is a single-file **packed** container with
    ``schema_version == 3``. The selector previously checked ``== 2``, so device
    decode silently fell back to host in production (the bit-parity gate missed it
    because it monkeypatches ``_should_device_decode``). This exercises the
    NATURAL selector on a real packed archive; cupy/x_cupy are monkeypatched so it
    runs wherever cellstream is installed, no GPU needed."""
    d, _ = archive_mode1
    arch = cellstream.ShardedArchive(d)
    assert arch.schema_version == 3                        # v0.5.x packed container
    monkeypatch.setattr(ss, "_cupy_available", lambda: True)
    monkeypatch.setattr(ss, "_x_cupy_available", lambda: True)
    assert ss._should_device_decode(arch) is True


def test_resolve_mode1_groupby_mismatch_raises(archive_mode1):
    from gpudge._stream_backend import open_backend
    d, _ = archive_mode1
    b = open_backend(d, n_workers=2, prefetch=0)
    with pytest.raises(ValueError, match="groupby"):
        ss._resolve_streaming(b, "WRONG", None)


def test_resolve_mode1_reference_mismatch_raises(archive_mode1):
    from gpudge._stream_backend import open_backend
    d, _ = archive_mode1
    b = open_backend(d, n_workers=2, prefetch=0)
    with pytest.raises(ValueError, match="reference"):
        ss._resolve_streaming(b, None, "not-a-ref-label")


def test_resolve_mode2_external_pool(archive_mode2):
    from gpudge._stream_backend import open_backend
    d, adata_g, adata_ntc, _ = archive_mode2
    b = open_backend(d, n_workers=2, prefetch=0)
    groupby, mode, ref_X, _ = ss._resolve_streaming(b, None, adata_ntc)
    assert mode == "external_ref"
    assert ref_X.shape[0] == adata_ntc.n_obs


def test_resolve_mode2_external_csc_coerced_to_csr(archive_mode2):
    """A CSC external reference.X is coerced to canonical CSR (one warning),
    matching the in-mem path's ensure_csr; closes the perf gap + the parity
    question by construction. #79c"""
    from gpudge._stream_backend import open_backend
    d, adata_g, adata_ntc, _ = archive_mode2
    b = open_backend(d, n_workers=2, prefetch=0)
    ref_csc = adata_ntc.copy()
    ref_csc.X = ref_csc.X.tocsc()
    with pytest.warns(UserWarning, match="converting to CSR"):
        groupby, mode, ref_X, _ = ss._resolve_streaming(b, None, ref_csc)
    assert mode == "external_ref"
    assert sp.issparse(ref_X) and ref_X.format == "csr"
    assert ref_X.shape[0] == adata_ntc.n_obs


def test_resolve_mode2_on_archive_with_reference_warns(archive_mode1):
    # An AnnData reference on a reference-bearing archive now WARNS and uses the
    # external pool (Semantics A); the archive's own reference shard is ignored.
    from gpudge._stream_backend import open_backend
    d, adata = archive_mode1
    b = open_backend(d, n_workers=2, prefetch=0)
    with pytest.warns(UserWarning, match="ignored in favor of the external"):
        groupby, mode, ref_X, _ = ss._resolve_streaming(b, None, adata)
    assert mode == "external_ref"
    assert ref_X.shape[0] == adata.n_obs            # external pool, not the archive ref shard


def test_resolve_mode2_gene_axis_mismatch_raises(archive_mode2):
    from gpudge._stream_backend import open_backend
    d, adata_g, adata_ntc, _ = archive_mode2
    b = open_backend(d, n_workers=2, prefetch=0)
    bad = adata_ntc[:, :adata_ntc.n_vars - 1].copy()    # wrong n_vars
    with pytest.raises(ValueError, match="gene"):
        ss._resolve_streaming(b, None, bad)


@pytest.fixture
def archive_ref_and_noref(tmp_path):
    """One synth → a reference-bearing archive AND a no-reference archive over the
    SAME guide cells, plus the external NTC AnnData. Proves the archive's reference
    shard is ignored under a Mode-2 external pool."""
    full = _make_synth(n_cells=600, n_genes=40, n_guides=8, sparse=True, seed=7)
    is_ntc = np.char.startswith(full.obs["comparison"].to_numpy().astype(str), "ntc")
    guides = full[~is_ntc].copy()
    ext_ntc = full[is_ntc].copy()
    d_ref = _write_archive(full, tmp_path / "withref", group_by="comparison",
                           reference=["ntc"], target_shard_bytes=4096)
    d_noref = _write_archive(guides, tmp_path / "noref", group_by="comparison",
                             reference=None, target_shard_bytes=4096)
    return d_ref, d_noref, ext_ntc


@needs_cuda
def test_external_pool_ignores_archive_reference_shard(archive_ref_and_noref):
    d_ref, d_noref, ext_ntc = archive_ref_and_noref
    with pytest.warns(UserWarning, match="ignored in favor of the external"):
        got = gpudge.de(shard_archive=d_ref, reference=ext_ntc)
    exp = gpudge.de(shard_archive=d_noref, reference=ext_ntc)
    # the archive's own reference label must NOT appear as a target
    assert "ntc" not in set(got["target"].to_list())
    # identical to running the external pool against a no-reference archive
    keys = ["target", "feature"]
    got_s = got.sort(keys)
    exp_s = exp.sort(keys)
    from polars.testing import assert_frame_equal
    # check_exact=True is load-bearing: the docstring claims identity, and
    # polars defaults to a 1e-5 relative tolerance. (ultrareview 2026-08)
    assert_frame_equal(got_s, exp_s, check_exact=True)


def test_resolve_non_grouped_archive_raises(tmp_path):
    from gpudge._stream_backend import open_backend
    adata = _make_synth(n_cells=200, n_genes=20, n_guides=4, sparse=True)
    cellstream.write_sharded(adata, str(tmp_path / "ng"), format="v2")  # no group_by
    b = open_backend(str(tmp_path / "ng"), n_workers=2, prefetch=0)
    with pytest.raises(ValueError, match="group_by"):
        ss._resolve_streaming(b, None, None)


def test_enumerate_targets(archive_mode1):
    from gpudge._stream_backend import open_backend
    d, adata = archive_mode1
    b = open_backend(d, n_workers=2, prefetch=0)
    targets, max_group_rows = b.targets()
    tgt_index = {t: i for i, t in enumerate(targets)}
    # All non-ntc comparison labels should appear exactly once; ntc excluded.
    comp = adata.obs["comparison"].to_numpy().astype(str)
    expected = {c for c in set(comp) if not c.startswith("ntc")}
    assert set(targets) == expected
    assert len(targets) == len(set(targets))           # disjoint, no dups
    assert all(tgt_index[t] == i for i, t in enumerate(targets))
    # max_group_rows feeds the streaming sizer's directional target-peak term;
    # a constant 0 would silently disable that budget, so pin the real value.
    assert max_group_rows == max(int((comp == t).sum()) for t in targets)

def test_enumerate_targets_mode2_includes_all(archive_mode2):
    from gpudge._stream_backend import open_backend
    d, adata_g, _, _ = archive_mode2
    b = open_backend(d, n_workers=2, prefetch=0)
    targets, max_group_rows = b.targets()
    comp = adata_g.obs["comparison"].to_numpy().astype(str)
    assert set(targets) == set(comp)                   # no reference shard → all groups
    assert max_group_rows == max(int((comp == t).sum()) for t in targets)


def test_open_backend_shard(archive_mode1):
    """open_backend returns a shard backend whose metadata matches the archive."""
    from gpudge._stream_backend import open_backend
    d, adata = archive_mode1
    b = open_backend(d, n_workers=2, prefetch=0)
    try:
        arch = cellstream.ShardedArchive(d)
        assert b.n_vars == arch.n_vars
        assert np.array_equal(b.var_names, np.asarray(arch.var.index))
        assert b.group_by == "comparison"
        assert b.has_archive_reference is True
        assert isinstance(b.supports_device_decode, bool)
    finally:
        b.close()


def test_shard_backend_targets_iterates_shards_once(archive_mode1, monkeypatch):
    """targets() is cached: two calls, ONE metadata pass over the shards.

    Comparing the two return values would pass even with no caching at all;
    counting the underlying iteration is what actually pins it.
    """
    from gpudge import _shard_stream as ss
    from gpudge._stream_backend import open_backend
    d, _ = archive_mode1
    b = open_backend(d, n_workers=2, prefetch=0)
    calls = []
    real = ss._enumerate_targets
    monkeypatch.setattr(ss, "_enumerate_targets",
                        lambda a: (calls.append(1), real(a))[1])
    try:
        t1, m1 = b.targets()
        t2, m2 = b.targets()
        assert (t1, m1) == (t2, m2)
        assert len(calls) == 1
    finally:
        b.close()


@pytest.mark.parametrize("median", [False, True], ids=["plain", "median"])
def test_stream_de_makes_the_expected_archive_passes(archive_mode1, median,
                                                     monkeypatch, host_decode):
    """Count archive I/O at the STREAM_DE level, not the backend level.

    The backend-level spies below drive targets()/target_row_sums()/
    target_source() by hand, so they cannot see stream_de calling one of them
    twice -- which is exactly the regression the "shard path is unchanged"
    claim needs guarded. Stub refpool_de_core (so no GPU is needed), exhaust
    the target_source it is handed, and count what the ARCHIVE saw:
    2 passes over iter_group_shards normally, 3 under median, and exactly one
    read_reference()."""
    from gpudge import _refpool, _shard_stream as ss
    from gpudge._stream_backend import open_backend as _real_open

    counts = {"iter": 0, "ref": 0}

    def counting_open(archive, **kw):
        b = _real_open(archive, **kw)
        real_iter, real_ref = b._arch.iter_group_shards, b._arch.read_reference
        b._arch.iter_group_shards = lambda *a, **k: (
            counts.__setitem__("iter", counts["iter"] + 1), real_iter(*a, **k))[1]
        b._arch.read_reference = lambda *a, **k: (
            counts.__setitem__("ref", counts["ref"] + 1), real_ref(*a, **k))[1]
        return b

    def fake_core(*, target_source, **kw):
        for _ in target_source(False):        # drive the generator to completion
            pass
        return "sentinel"

    # stream_de does `from ._stream_backend import open_backend` INSIDE the
    # function body, so the module attribute is what it resolves at call time.
    monkeypatch.setattr("gpudge._stream_backend.open_backend", counting_open)
    monkeypatch.setattr(_refpool, "refpool_de_core", fake_core)

    d, _ = archive_mode1
    out = ss.stream_de(
        d, groupby=None, reference=None, mean_calc="arithmetic", epsilon=1e-9,
        gpu_gene_chunk_size=None, oom_recovery=True, cpm_normalize=False,
        normalize_target_sum=("median" if median else None), output_columns=None,
        filter_gene_min_mean_value=None, filter_gene_min_total_value=None,
        filter_gene_min_cpm_cell=None, filter_gene_min_cpm_bulk=None,
        keep_genes=None, stream_n_workers=2, stream_prefetch=0, device=None)
    assert out == "sentinel"
    assert counts["ref"] == 1, counts
    assert counts["iter"] == (3 if median else 2), counts


def test_shard_backend_reads_reference_exactly_once(archive_mode1):
    """The reference shard is expensive; the refactor must not read it twice."""
    from gpudge._stream_backend import open_backend
    d, _ = archive_mode1
    b = open_backend(d, n_workers=2, prefetch=0)
    calls = []
    real = b._arch.read_reference
    b._arch.read_reference = lambda *a, **k: (calls.append(1), real(*a, **k))[1]
    try:
        ref_X, msg = b.resolve_archive_reference("comparison", None)
        assert len(calls) == 1
        assert ref_X.shape[0] > 0 and msg
    finally:
        b.close()


@pytest.fixture
def host_decode(monkeypatch):
    """Force the HOST CSR decode path for a backend test that reads shard data.

    _should_device_decode is True whenever cupy merely *imports* -- and cupy is
    importable on CPU boxes that have no usable CUDA driver, where gs.x_cupy()
    then dies inside cellstream's nvcomp codec. The tests below are about the host
    path's I/O structure, so they pin it explicitly instead of depending on
    whether the machine happens to have cupy installed. (The device path has its
    own needs_cuda-gated parity tests further down.)"""
    monkeypatch.setattr(ss, "_cupy_available", lambda: False)


def test_shard_backend_target_row_sums_matches_manual(archive_mode1, host_decode):
    """target_row_sums() equals a manual sweep of the shards, in the same order."""
    from gpudge._csr_dense import csr_row_sums
    from gpudge._stream_backend import open_backend
    d, _ = archive_mode1
    b = open_backend(d, n_workers=2, prefetch=0)
    try:
        arch = cellstream.ShardedArchive(d)
        manual = np.concatenate(
            [csr_row_sums(gs.x()) for gs in arch.iter_group_shards()])
        np.testing.assert_array_equal(b.target_row_sums(), manual)
    finally:
        b.close()


@pytest.mark.parametrize("median", [False, True], ids=["plain", "median"])
def test_shard_backend_iterates_shards_the_expected_number_of_times(
        archive_mode1, median, host_decode):
    """Shard I/O count is part of "byte-identical": the pre-#110 driver made
    exactly TWO passes over iter_group_shards (target enumeration + the target
    source), or THREE when normalize_target_sum='median' added the row-sum
    pre-pass. The refactor must not add a pass."""
    from gpudge._stream_backend import open_backend
    d, _ = archive_mode1
    b = open_backend(d, n_workers=2, prefetch=0)
    calls = []
    real = b._arch.iter_group_shards
    b._arch.iter_group_shards = lambda *a, **k: (calls.append(1), real(*a, **k))[1]
    try:
        b.targets()                                  # pass 1: metadata
        if median:
            b.target_row_sums()                      # pass 2: median pre-pass
        for _ in b.target_source(False):             # final pass: the data
            pass
        assert len(calls) == (3 if median else 2), calls
    finally:
        b.close()


@pytest.mark.parametrize("iter_kw", [{}, {"prefetch": 2, "n_workers": 2}],
                         ids=["lazy", "prefetch"])
def test_group_shard_x_byte_identical_to_to_anndata(archive_mode1, iter_kw):
    """The streaming hot loop reads shard matrices via gs.x() (raw CSR, no
    AnnData wrapper). cellstream guarantees gs.x() is byte-identical to
    gs.to_anndata().X; guard it here on the real reader (no GPU needed) so a
    regression in the swap surfaces on CPU. Covers both the lazy (prefetch=0)
    and decode-ahead (prefetch>=1) paths, since x() resolves differently in
    each."""
    d, _ = archive_mode1
    arch = cellstream.ShardedArchive(d)
    n_shards = 0
    for gs in arch.iter_group_shards(**iter_kw):
        X = gs.x()                               # what stream_de() consumes
        ref = gs.to_anndata().X                  # the wrapper path it replaced
        assert X.shape == ref.shape
        assert X.dtype == ref.dtype              # dtype-preserving (e.g. no uint16→int32 widen)
        assert isinstance(X, sp.csr_matrix) and isinstance(ref, sp.csr_matrix)
        np.testing.assert_array_equal(X.data, ref.data)
        np.testing.assert_array_equal(X.indices, ref.indices)
        np.testing.assert_array_equal(X.indptr, ref.indptr)
        # group row slices the loop also relies on are unaffected by x()
        for _label, sl in gs.groups.items():
            np.testing.assert_array_equal(X[sl].toarray(), ref[sl].toarray())
        n_shards += 1
    assert n_shards > 0                          # the fixture really produced shards


# ---------------------------------------------------------------------------
# Task-4: _reference_prepass + oversized guard
# ---------------------------------------------------------------------------


def test_reference_prepass_oversized_guard(monkeypatch):
    # Force a tiny "free GPU bytes" so the projected sorted-ref allocation fails.
    if torch.cuda.is_available():
        monkeypatch.setattr(torch.cuda, "mem_get_info",
                            lambda *a, **k: (1024, 1024))
    else:
        # On CPU nodes, _reference_prepass must still guard before touching CUDA.
        monkeypatch.setattr(ss, "_free_gpu_bytes", lambda dev: 1024)
    X = sp.random(5000, 4000, density=0.1, format="csr", dtype=np.float32)
    with pytest.raises(RuntimeError, match="reference.*too large|does not fit") as ei:
        ss._reference_prepass(
            X, n_genes=4000, device=torch.device("cuda"), chunk=512,
            mean_calc="arithmetic", scale_main=False, scale_num=1.0e6,
            need_other_unit=False,
            need_row_sums=False, need_row_scales=False, oom_recovery=True)
    # #79d: message must be path-agnostic — the prepass is shared by both callers,
    # so it must not advise "use the in-memory de(adata=...) path" (circular when
    # the caller already IS the in-mem path).
    assert "in-memory" not in str(ei.value).lower()
    assert "de(adata=" not in str(ei.value)


@needs_cuda
def test_reference_prepass_sorted_and_means():
    rng = np.random.default_rng(0)
    X = sp.csr_matrix(rng.negative_binomial(2, 0.3, size=(120, 30)).astype(np.float32))
    dev = torch.device("cuda")
    out = ss._reference_prepass(
        X, n_genes=30, device=dev, chunk=8, mean_calc="arithmetic",
        scale_main=False, scale_num=1.0e6, need_other_unit=False, need_row_sums=False,
        need_row_scales=False, oom_recovery=True)
    # sorted_ref_full[g] == sorted column g of the dense reference.
    dense = torch.from_numpy(X.toarray().astype(np.float32)).to(dev)
    expected_sorted = torch.sort(dense.T, dim=1).values         # (genes, n_ref)
    assert torch.allclose(out["sorted_ref_full"], expected_sorted)
    assert out["n_ref"] == 120
    expected_mean = dense.to(torch.float64).mean(dim=0).cpu().numpy()
    assert np.allclose(out["arith_ref"], expected_mean)
    assert np.allclose(out["ref_mean"], expected_mean)          # arithmetic ⇒ same


# ---------------------------------------------------------------------------
# Task-5: group_chunk_stats shared per-group-chunk compute
# ---------------------------------------------------------------------------


@needs_cuda
def test_group_chunk_stats_matches_kernels():
    rng = np.random.default_rng(3)
    dev = torch.device("cuda")
    n_ref, m, w = 50, 13, 9
    ref = torch.from_numpy(rng.negative_binomial(2, 0.3, (n_ref, w)).astype(np.float32)).to(dev)
    grp = torch.from_numpy(rng.negative_binomial(2, 0.3, (m, w)).astype(np.float32)).to(dev)
    sorted_ref = torch.sort(ref.T.contiguous(), dim=1).values     # (w, n_ref)
    ref_tie = _tie_term_per_gene(sorted_ref)
    arith, reported, other, u1, p, _du, _dp, _ts = ss.group_chunk_stats(
        grp, sorted_ref, ref_tie, n_ref, mean_calc="arithmetic", scale_main=False)
    # returns are GPU tensors → move to host to compare
    exp_mean = grp.to(torch.float64).mean(dim=0).cpu().numpy()
    assert np.allclose(arith.cpu().numpy(), exp_mean)
    assert np.allclose(reported.cpu().numpy(), exp_mean)
    assert other is None
    # MWU vs the kernel directly
    ku, kp = mwu_one_group(sorted_ref, ref_tie, grp.T.contiguous(), n_ref=n_ref)
    assert np.allclose(u1.cpu().numpy(), ku.cpu().numpy())
    assert np.allclose(p.cpu().numpy(), kp.cpu().numpy())


# ---------------------------------------------------------------------------
# Task-6: stream_de equivalence + chunk-size invariance
# ---------------------------------------------------------------------------
def _assert_equiv(df_stream, df_mem):
    # 100% row coverage: same (target, feature) set and height.
    assert df_stream.height == df_mem.height
    key = ["target", "feature"]
    a = df_stream.sort(key)
    b = df_mem.sort(key)
    assert a.select(key).equals(b.select(key))
    for col in ("log2_fold_change", "p_value", "p_adj"):
        x = a[col].to_numpy()
        y = b[col].to_numpy()
        # Exact-ish equivalence at the in-memory chunk-invariance tolerance,
        # with equal_nan so NaN positions must align too. Pearson r alone is
        # invariant under any affine y=a*x+b, so it is blind to a systematic
        # scale/offset drift (e.g. a normalization constant applied once vs
        # twice) — exactly the failure mode most plausible in streaming (M2).
        ok = np.isfinite(x) & np.isfinite(y)
        # Derive the diagnostic max-abs-diff from finite-in-both pairs so the
        # message can't raise on an all-NaN slice and mask the real assertion.
        max_abs = float(np.abs(x - y)[ok].max()) if ok.any() else float("nan")
        assert np.allclose(x, y, rtol=1e-5, atol=1e-7, equal_nan=True), (
            f"{col}: streaming vs in-memory differ beyond tol (max abs diff {max_abs})"
        )
        assert ok.any(), f"{col}: no finite pairs to compare (all NaN/inf)"
        r = np.corrcoef(x[ok], y[ok])[0, 1]
        assert r > 0.9999999, f"{col} pearson {r}"  # coarse secondary guard
    assert np.array_equal(a["target_ncells"].to_numpy(), b["target_ncells"].to_numpy())
    assert np.array_equal(a["ref_ncells"].to_numpy(), b["ref_ncells"].to_numpy())


@needs_cuda
def test_stream_mode1_equivalence(archive_mode1):
    d, adata = archive_mode1
    df_stream = gpudge.de(shard_archive=d)
    df_mem = gpudge.de(adata, groupby="comparison", reference="ntc")
    _assert_equiv(df_stream, df_mem)


@needs_cuda
def test_stream_prefetch_matches_serial(archive_mode1):
    """Prefetch (stream_prefetch>0) is byte-identical to serial (stream_prefetch=0)."""
    from polars.testing import assert_frame_equal
    d, _ = archive_mode1
    serial = gpudge.de(shard_archive=d, stream_prefetch=0).sort(["target", "feature"])
    prefetch = gpudge.de(shard_archive=d, stream_n_workers=4,
                         stream_prefetch=2).sort(["target", "feature"])
    # check_exact=True is load-bearing: polars defaults to check_exact=False with a
    # 1e-5 relative tolerance, which would let this 'byte-identical' claim pass on a
    # frame that merely agrees to 5 digits. (#110 codex checkpoint 2)
    assert_frame_equal(serial, prefetch, check_exact=True)   # prefetch is byte-identical upstream


@needs_cuda
def test_stream_prefetch_matches_serial_median(archive_mode1):
    """The prefetched median pre-pass is byte-identical to the serial pre-pass."""
    from polars.testing import assert_frame_equal
    d, _ = archive_mode1
    serial = gpudge.de(shard_archive=d, normalize_target_sum="median",
                       stream_prefetch=0).sort(["target", "feature"])
    prefetch = gpudge.de(shard_archive=d, normalize_target_sum="median",
                         stream_n_workers=4, stream_prefetch=2).sort(["target", "feature"])
    assert_frame_equal(serial, prefetch, check_exact=True)


@needs_cuda
def test_stream_mode2_equivalence(archive_mode2):
    d, adata_g, adata_ntc, full = archive_mode2
    df_stream = gpudge.de(shard_archive=d, reference=adata_ntc)
    # In-memory equivalent: concat guide + ntc, reference label = "ntc".
    merged = ad.concat([adata_g, adata_ntc], join="outer")
    df_mem = gpudge.de(merged, groupby="comparison", reference="ntc")
    _assert_equiv(df_stream, df_mem)


@needs_cuda
def test_stream_chunk_size_invariance(archive_mode1):
    d, _ = archive_mode1
    a = gpudge.de(shard_archive=d, gpu_gene_chunk_size=4)
    b = gpudge.de(shard_archive=d, gpu_gene_chunk_size=4096)
    _assert_equiv(a, b)


# ---------------------------------------------------------------------------
# Task-7: CPM / filter / keep_genes / geometric parity
# ---------------------------------------------------------------------------
@needs_cuda
@pytest.mark.parametrize("kw", [
    dict(cpm_normalize=True),
    dict(filter_gene_min_mean_value=0.5),
    dict(filter_gene_min_total_value=5.0),
    dict(filter_gene_min_cpm_cell=1.0),
    dict(filter_gene_min_cpm_bulk=1.0),
])
def test_stream_filter_cpm_parity(archive_mode1, kw):
    d, adata = archive_mode1
    df_stream = gpudge.de(shard_archive=d, **kw)
    df_mem = gpudge.de(adata, groupby="comparison", reference="ntc", **kw)
    _assert_equiv(df_stream, df_mem)


@needs_cuda
def test_streaming_numeric_target_matches_inmemory(archive_mode1):
    """Numeric normalize_target_sum: streaming == in-memory on the same data."""
    import gpudge
    d, adata = archive_mode1
    out_mem = gpudge.de(adata, groupby="comparison", reference="ntc",
                        normalize_target_sum=5e5)
    out_str = gpudge.de(shard_archive=d, normalize_target_sum=5e5)
    j = out_mem.join(out_str, on=["target", "feature"], suffix="_s")
    assert np.allclose(j["log2_fold_change"].to_numpy(),
                       j["log2_fold_change_s"].to_numpy(), rtol=1e-6, equal_nan=True)
    assert np.allclose(j["p_value"].to_numpy(),
                       j["p_value_s"].to_numpy(), equal_nan=True)
    assert out_mem.height == out_str.height


@needs_cuda
def test_streaming_median_matches_inmemory_median(archive_mode1):
    """'median' in streaming (pre-pass) == 'median' in-memory: the global median
    over all shards equals the in-memory median over the same cells."""
    import gpudge
    d, adata = archive_mode1
    out_mem = gpudge.de(adata, groupby="comparison", reference="ntc",
                        normalize_target_sum="median")
    out_str = gpudge.de(shard_archive=d, normalize_target_sum="median")
    j = out_mem.join(out_str, on=["target", "feature"], suffix="_s")
    assert np.allclose(j["p_value"].to_numpy(),
                       j["p_value_s"].to_numpy(), equal_nan=True)
    assert np.allclose(j["log2_fold_change"].to_numpy(),
                       j["log2_fold_change_s"].to_numpy(), rtol=1e-6, equal_nan=True)


@needs_cuda
def test_stream_keep_genes_parity(archive_mode1):
    d, adata = archive_mode1
    keep = np.zeros(adata.n_vars, dtype=bool)
    keep[::2] = True
    df_stream = gpudge.de(shard_archive=d, keep_genes=keep)
    df_mem = gpudge.de(adata, groupby="comparison", reference="ntc", keep_genes=keep)
    _assert_equiv(df_stream, df_mem)


@needs_cuda
def test_stream_geometric_parity(archive_mode1):
    d, adata = archive_mode1
    df_stream = gpudge.de(shard_archive=d, mean_calc="geometric")
    df_mem = gpudge.de(adata, groupby="comparison", reference="ntc", mean_calc="geometric")
    _assert_equiv(df_stream, df_mem)


@needs_cuda
def test_stream_empty_targets(tmp_path):
    # An archive with NO target groups → the n_targets==0 empty_output_frame
    # early return. ntc_frac must be 1.0, not 0.99: at 0.99 _make_synth still
    # emits a target cell (n_target = 120 - int(120*0.99) = 2), so this reached
    # a 2-row populated result and never exercised the early return it claims to.
    adata = _make_synth(n_cells=120, n_genes=20, n_guides=2, sparse=True, ntc_frac=1.0)
    d = _write_archive(adata, tmp_path / "few", group_by="comparison",
                       reference=["ntc"], target_shard_bytes=4096)
    df = gpudge.de(shard_archive=d)
    from gpudge._output import DEFAULT_OUTPUT_COLUMNS
    assert df.height == 0
    assert list(df.columns) == list(DEFAULT_OUTPUT_COLUMNS)


@needs_cuda
def test_stream_empty_targets_respects_output_columns(tmp_path):
    """The empty-archive early return must honour output_columns (route through
    the same select/rename as the non-empty path), so the schema is identical
    whether or not the archive has targets. Regression for the ultrareview
    finding: the early return emitted DEFAULT_OUTPUT_COLUMNS verbatim, so a
    caller passing output_columns got the wrong column names on empty results.

    ntc_frac=1.0 (not 0.99) so the archive is genuinely target-less and hits the
    empty_output_frame(output_columns) early return -- at 0.99 _make_synth still
    emits a target cell, so this select/rename ran on a populated frame and the
    early return went untested."""
    adata = _make_synth(n_cells=120, n_genes=20, n_guides=2, sparse=True, ntc_frac=1.0)
    d = _write_archive(adata, tmp_path / "few", group_by="comparison",
                       reference=["ntc"], target_shard_bytes=4096)
    df = gpudge.de(shard_archive=d, output_columns={"target": "guide", "p_value": "p"})
    assert df.height == 0
    assert list(df.columns) == ["guide", "p"]
    # Schema must be typed (not Null) so the empty result aligns with non-empty
    # results downstream (concat / schema-aware access). See Gemini review.
    import polars as pl
    assert df.schema["guide"] == pl.String
    assert df.schema["p"] == pl.Float64



def test_resolve_mode1_reference_read_none_raises(archive_mode1, monkeypatch):
    # Manifest declares a reference shard but read_reference() returns None
    # (inconsistent/corrupted archive) → clear ValueError, not AttributeError.
    from gpudge._stream_backend import open_backend
    d, _ = archive_mode1
    b = open_backend(d, n_workers=2, prefetch=0)
    monkeypatch.setattr(b._arch, "read_reference", lambda *a, **k: None)
    with pytest.raises(ValueError, match="returned None|inconsistent"):
        ss._resolve_streaming(b, None, None)


# ---------------------------------------------------------------------------
# #69 MERGE GATE: device (x_cupy) decode == host CSR decode, byte-for-byte.
# Forces each path via _should_device_decode; needs cellstream>=0.5.5 (x_cupy) in
# the env for the device leg (see the plan's Phase-6 GPU prerequisite).
# ---------------------------------------------------------------------------
@needs_cuda
@pytest.mark.parametrize("cpm", [False, True], ids=["raw", "cpm"])
def test_device_vs_host_decode_bit_identical_mode1(archive_mode1, cpm, monkeypatch):
    from polars.testing import assert_frame_equal
    d, _ = archive_mode1
    keys = ["target", "feature"]

    monkeypatch.setattr(ss, "_should_device_decode", lambda arch: False)
    host = gpudge.de(shard_archive=d, cpm_normalize=cpm)
    monkeypatch.setattr(ss, "_should_device_decode", lambda arch: True)
    dev = gpudge.de(shard_archive=d, cpm_normalize=cpm)

    assert_frame_equal(dev.sort(keys), host.sort(keys), check_exact=True)


@needs_cuda
@pytest.mark.parametrize("cpm", [False, True], ids=["raw", "cpm"])
def test_device_vs_host_decode_bit_identical_mode2(archive_mode2, cpm, monkeypatch):
    from polars.testing import assert_frame_equal
    d, _adata_g, adata_ntc, _full = archive_mode2
    keys = ["target", "feature"]

    monkeypatch.setattr(ss, "_should_device_decode", lambda arch: False)
    host = gpudge.de(shard_archive=d, reference=adata_ntc, cpm_normalize=cpm)
    monkeypatch.setattr(ss, "_should_device_decode", lambda arch: True)
    dev = gpudge.de(shard_archive=d, reference=adata_ntc, cpm_normalize=cpm)

    assert_frame_equal(dev.sort(keys), host.sort(keys), check_exact=True)


@needs_cuda
def test_group_chunk_stats_directional_matches_kernel():
    """group_chunk_stats must pass lfc_combos straight through to the kernel."""
    from gpudge._mwu import _tie_term_per_gene, mwu_one_group_lfc

    rng = np.random.default_rng(5)
    dev = torch.device("cuda")
    n_ref, m, w = 30, 11, 6
    ref = torch.from_numpy(
        rng.negative_binomial(2, 0.4, (n_ref, w)).astype(np.float32)).to(dev)
    grp = torch.from_numpy(
        rng.negative_binomial(2, 0.4, (m, w)).astype(np.float32)).to(dev)
    sorted_ref = torch.sort(ref.T.contiguous(), dim=1).values
    ref_tie = _tie_term_per_gene(sorted_ref)
    combos = ((0.25, "up"), (0.25, "down"))

    out = ss.group_chunk_stats(grp, sorted_ref, ref_tie, n_ref,
                               mean_calc="arithmetic", scale_main=False,
                               lfc_combos=combos)
    assert len(out) == 8
    _, _, _, u1, p, du, dp, ts = out

    u1_e, p_e, du_e, dp_e = mwu_one_group_lfc(
        sorted_ref, ref_tie, grp.T.contiguous(), n_ref=n_ref,
        lfc_combos=combos)
    assert torch.equal(u1, u1_e) and torch.equal(p, p_e)
    assert torch.equal(du, du_e) and torch.equal(dp, dp_e)
    assert ts is None


@needs_cuda
def test_group_chunk_stats_none_combos_returns_none_dirs():
    from gpudge._mwu import _tie_term_per_gene

    rng = np.random.default_rng(6)
    dev = torch.device("cuda")
    n_ref, m, w = 20, 7, 4
    ref = torch.from_numpy(
        rng.negative_binomial(2, 0.4, (n_ref, w)).astype(np.float32)).to(dev)
    grp = torch.from_numpy(
        rng.negative_binomial(2, 0.4, (m, w)).astype(np.float32)).to(dev)
    sorted_ref = torch.sort(ref.T.contiguous(), dim=1).values
    ref_tie = _tie_term_per_gene(sorted_ref)
    out = ss.group_chunk_stats(grp, sorted_ref, ref_tie, n_ref,
                               mean_calc="arithmetic", scale_main=False)
    assert len(out) == 8
    assert out[5] is None and out[6] is None and out[7] is None


def _assert_equiv_lfc(df_stream, df_mem):
    """Apply the existing base comparison plus its numeric guards to every
    directional column."""
    _assert_equiv(df_stream, df_mem)
    key = ["target", "feature"]
    a = df_stream.sort(key)
    b = df_mem.sort(key)
    directional = [c for c in a.columns if c.startswith("tau=")]
    assert directional == [c for c in b.columns if c.startswith("tau=")]
    for col in directional:
        x = a[col].to_numpy()
        y = b[col].to_numpy()
        ok = np.isfinite(x) & np.isfinite(y)
        max_abs = float(np.abs(x - y)[ok].max()) if ok.any() else float("nan")
        assert np.allclose(x, y, rtol=1e-5, atol=1e-7, equal_nan=True), (
            f"{col}: streaming vs in-memory differ beyond tol "
            f"(max abs diff {max_abs})"
        )
        assert ok.any(), f"{col}: no finite pairs to compare (all NaN/inf)"
        r = np.corrcoef(x[ok], y[ok])[0, 1]
        assert r > 0.9999999, f"{col} pearson {r}"


def _assert_lfc_base_unchanged(with_t, base):
    from polars.testing import assert_frame_equal

    assert_frame_equal(with_t.select(base.columns), base,
                       check_dtypes=True, check_exact=True)


@needs_cuda
def test_stream_mode1_equivalence_with_lfc_grid(archive_mode1):
    d, adata = archive_mode1
    stream_base = gpudge.de(shard_archive=d)
    stream_lfc = gpudge.de(shard_archive=d, lfc_threshold=LFC_TAUS)
    _assert_lfc_base_unchanged(stream_lfc, stream_base)

    mem_base = gpudge.de(adata, groupby="comparison", reference="ntc")
    mem_lfc = gpudge.de(
        adata, groupby="comparison", reference="ntc",
        lfc_threshold=LFC_TAUS)
    _assert_lfc_base_unchanged(mem_lfc, mem_base)
    _assert_equiv_lfc(stream_lfc, mem_lfc)


@needs_cuda
def test_stream_mode2_equivalence_with_lfc_grid(archive_mode2):
    d, adata_g, adata_ntc, _full = archive_mode2
    stream_base = gpudge.de(shard_archive=d, reference=adata_ntc)
    stream_lfc = gpudge.de(
        shard_archive=d, reference=adata_ntc, lfc_threshold=LFC_TAUS)
    _assert_lfc_base_unchanged(stream_lfc, stream_base)

    merged = ad.concat([adata_g, adata_ntc], join="outer")
    mem_base = gpudge.de(merged, groupby="comparison", reference="ntc")
    mem_lfc = gpudge.de(
        merged, groupby="comparison", reference="ntc",
        lfc_threshold=LFC_TAUS)
    _assert_lfc_base_unchanged(mem_lfc, mem_base)
    _assert_equiv_lfc(stream_lfc, mem_lfc)


@needs_cuda
def test_stream_prefetch_matches_serial_with_lfc_grid(archive_mode1):
    from polars.testing import assert_frame_equal

    d, _ = archive_mode1
    keys = ["target", "feature"]
    serial_base = gpudge.de(
        shard_archive=d, stream_prefetch=0).sort(keys)
    serial_lfc = gpudge.de(
        shard_archive=d, stream_prefetch=0,
        lfc_threshold=LFC_TAUS).sort(keys)
    _assert_lfc_base_unchanged(serial_lfc, serial_base)

    prefetch_base = gpudge.de(shard_archive=d).sort(keys)
    prefetch_lfc = gpudge.de(
        shard_archive=d, lfc_threshold=LFC_TAUS).sort(keys)
    _assert_lfc_base_unchanged(prefetch_lfc, prefetch_base)
    assert_frame_equal(serial_lfc, prefetch_lfc,
                       check_dtypes=True, check_exact=True)


@needs_cuda
def test_device_vs_host_decode_bit_identical_with_lfc_grid(
    archive_mode1, monkeypatch
):
    from polars.testing import assert_frame_equal

    d, _ = archive_mode1
    keys = ["target", "feature"]

    monkeypatch.setattr(ss, "_should_device_decode", lambda arch: False)
    host_base = gpudge.de(shard_archive=d).sort(keys)
    host_lfc = gpudge.de(
        shard_archive=d, lfc_threshold=LFC_TAUS).sort(keys)
    _assert_lfc_base_unchanged(host_lfc, host_base)

    monkeypatch.setattr(ss, "_should_device_decode", lambda arch: True)
    device_base = gpudge.de(shard_archive=d).sort(keys)
    device_lfc = gpudge.de(
        shard_archive=d, lfc_threshold=LFC_TAUS).sort(keys)
    _assert_lfc_base_unchanged(device_lfc, device_base)
    assert_frame_equal(device_lfc, host_lfc,
                       check_dtypes=True, check_exact=True)


@needs_cuda
def test_stream_chunk_size_invariance_with_lfc_grid(archive_mode1):
    from polars.testing import assert_frame_equal

    d, _ = archive_mode1
    small_base = gpudge.de(
        shard_archive=d, gpu_gene_chunk_size=4, oom_recovery=False)
    small_lfc = gpudge.de(
        shard_archive=d, gpu_gene_chunk_size=4, oom_recovery=False,
        lfc_threshold=LFC_TAUS)
    _assert_lfc_base_unchanged(small_lfc, small_base)

    large_base = gpudge.de(shard_archive=d, gpu_gene_chunk_size=4096)
    large_lfc = gpudge.de(
        shard_archive=d, gpu_gene_chunk_size=4096,
        lfc_threshold=LFC_TAUS)
    _assert_lfc_base_unchanged(large_lfc, large_base)
    _assert_equiv_lfc(small_lfc, large_lfc)
    # Both legs are the SAME streaming driver at two chunk widths, so this is
    # bit-exact, not merely close: every per-gene reduction runs along the cell
    # axis and never spans a gene-chunk boundary. (The mode1/mode2 siblings
    # above compare two DIFFERENT drivers, which is why those stay
    # tolerance-based, mirroring the existing base-column gate.) Codex review P1.
    keys = ["target", "feature"]
    assert_frame_equal(small_lfc.sort(keys), large_lfc.sort(keys),
                       check_dtypes=True, check_exact=True)


@needs_cuda
def test_stream_empty_targets_with_lfc_grid(archive_mode1, tmp_path):
    from polars.testing import assert_frame_equal

    populated_path, _ = archive_mode1
    populated_base = gpudge.de(shard_archive=populated_path)
    populated_lfc = gpudge.de(
        shard_archive=populated_path, lfc_threshold=LFC_TAUS)
    _assert_lfc_base_unchanged(populated_lfc, populated_base)

    # ntc_frac=1.0, NOT the 0.99 the older empty-archive tests use: at 0.99 the
    # archive still contains a target group (120 cells, 1 non-ntc guide), so
    # refpool_de_core's n_targets == 0 early return -- the whole point here --
    # is never reached. Verified: 0.99 yields a 20-row result, 1.0 yields 0.
    adata = _make_synth(
        n_cells=120, n_genes=20, n_guides=2, sparse=True, ntc_frac=1.0)
    empty_path = _write_archive(
        adata, tmp_path / "few_lfc", group_by="comparison",
        reference=["ntc"], target_shard_bytes=4096)
    empty_base = gpudge.de(shard_archive=empty_path)
    empty_lfc = gpudge.de(
        shard_archive=empty_path, lfc_threshold=LFC_TAUS)
    _assert_lfc_base_unchanged(empty_lfc, empty_base)
    assert empty_lfc.height == 0
    assert empty_lfc.columns == populated_lfc.columns
    assert empty_lfc.schema == populated_lfc.schema

    output_columns = {
        "target": "t",
        "feature": "f",
        "tau=+0_p": "p_up",
        "tau=+0_Ueffect": "Ueffect_up",
        "tau=+0_padj": "q_up",
    }
    selected_empty = gpudge.de(
        shard_archive=empty_path, lfc_threshold=LFC_TAUS,
        output_columns=output_columns)
    selected_populated = populated_lfc.select(
        list(output_columns)).rename(output_columns).head(0)
    assert_frame_equal(selected_empty, selected_populated,
                       check_dtypes=True, check_exact=True)


# --- tau_star over the streaming paths -----------------------------------
# CI is CPU-only and does not even COLLECT this module (it needs cellstream), so
# these run only on a GPU host with cellstream installed. That is precisely why they
# exist: the tau* accumulators are threaded differently on each path
# (`__init__.py` carries a reference row and drops it via target_indices;
# `_refpool.py` does not have one at all), and a shape or indexing slip there
# corrupts or crashes a whole production path with nothing else to catch it.

TAUSTAR_LEVELS = (0.5, 0.05)


@needs_cuda
def test_core_sizer_receives_the_row_count_under_streaming(archive_mode1,
                                                           monkeypatch):
    """The ONLY route to _refpool._auto_gene_chunk_size: inmem_external_ref_de
    computes an explicit chunk and passes it down, bypassing the core's sizer,
    and Mode-1 uses the `gpudge` binding. A spy-all-three test would pass
    without ever exercising this binding."""
    from gpudge import _refpool as rp

    archive, _adata = archive_mode1
    real = rp._auto_gene_chunk_size
    seen = []

    def spy(*args, **kwargs):
        seen.append(kwargs.get("n_levels"))
        return real(*args, **kwargs)

    monkeypatch.setattr(rp, "_auto_gene_chunk_size", spy)
    gpudge.de(shard_archive=archive, tau_star=[0.05], tau_star_se=True,
              gpu_gene_chunk_size=None)
    assert seen == [5], seen


def _assert_equiv_taustar(df_stream, df_mem):
    """The base comparison plus its numeric guards on every tau* column."""
    _assert_equiv(df_stream, df_mem)
    key = ["target", "feature"]
    a = df_stream.sort(key)
    b = df_mem.sort(key)
    cols = [c for c in a.columns if c.startswith("tau*_")]
    assert cols == [c for c in b.columns if c.startswith("tau*_")]
    assert cols, "fixture produced no tau* columns to compare"
    for col in cols:
        x = a[col].to_numpy()
        y = b[col].to_numpy()
        # EXACT, not allclose. Spec 5f requires bit-identity here, and the two
        # paths run the IDENTICAL kernel on the same sorted reference -- so any
        # difference means the preprocessing or accumulator plumbing fed them
        # different inputs, which is precisely the bug class this gate exists
        # for. The lfc twin above uses allclose because its columns also carry
        # the mean accumulations, which legitimately differ in float ordering;
        # tau* has no such term. assert_array_equal treats NaN as equal to NaN
        # and matches signed infinities, so the undefined and unbounded genes
        # compare correctly while a +inf/-inf swap still fails.
        np.testing.assert_array_equal(x, y, err_msg=col)
        # Small groups commonly have an unbounded endpoint, making
        # tau*_se entirely +inf; exact parity above remains the release gate.
        if col == "tau*_se":
            continue
        ok = np.isfinite(x)
        assert ok.any(), f"{col}: no finite pairs to compare (all NaN/inf)"


@needs_cuda
def test_stream_mode1_equivalence_with_tau_star(archive_mode1):
    d, adata = archive_mode1
    stream_base = gpudge.de(shard_archive=d)
    stream_ts = gpudge.de(shard_archive=d, tau_star=TAUSTAR_LEVELS,
                          tau_star_se=True)
    _assert_lfc_base_unchanged(stream_ts, stream_base)

    mem_base = gpudge.de(adata, groupby="comparison", reference="ntc")
    mem_ts = gpudge.de(adata, groupby="comparison", reference="ntc",
                       tau_star=TAUSTAR_LEVELS, tau_star_se=True)
    _assert_lfc_base_unchanged(mem_ts, mem_base)
    _assert_equiv_taustar(stream_ts, mem_ts)


@needs_cuda
def test_stream_mode2_equivalence_with_tau_star(archive_mode2):
    d, adata_g, adata_ntc, _full = archive_mode2
    stream_base = gpudge.de(shard_archive=d, reference=adata_ntc)
    stream_ts = gpudge.de(shard_archive=d, reference=adata_ntc,
                          tau_star=TAUSTAR_LEVELS, tau_star_se=True)
    _assert_lfc_base_unchanged(stream_ts, stream_base)

    merged = ad.concat([adata_g, adata_ntc], join="outer")
    mem_base = gpudge.de(merged, groupby="comparison", reference="ntc")
    mem_ts = gpudge.de(merged, groupby="comparison", reference="ntc",
                       tau_star=TAUSTAR_LEVELS, tau_star_se=True)
    _assert_lfc_base_unchanged(mem_ts, mem_base)
    _assert_equiv_taustar(stream_ts, mem_ts)


@needs_cuda
def test_stream_tau_star_composes_with_lfc_grid(archive_mode1):
    """Both features active on the streaming path: group_chunk_stats then runs
    mwu_one_group_lfc AND mwu_one_group_taustar, and the lfc columns must be
    the ones the caller sees, unchanged."""
    from polars.testing import assert_frame_equal

    d, _adata = archive_mode1
    lfc_only = gpudge.de(shard_archive=d, lfc_threshold=LFC_TAUS)
    both = gpudge.de(shard_archive=d, lfc_threshold=LFC_TAUS,
                     tau_star=TAUSTAR_LEVELS)
    n = len(TAUSTAR_LEVELS)
    assert both.columns[:-n] == lfc_only.columns
    assert both.columns[-n:] == ["tau*_p0.05", "tau*_p0.5"]
    assert_frame_equal(both.select(lfc_only.columns), lfc_only,
                       check_dtypes=True, check_exact=True)

    # Same branch with the SE rows on: the streaming accumulator is now sized
    # len(levels) + 3 while the lfc accumulators are unchanged, so this pins
    # that the two row counts stay independent through group_chunk_stats.
    with_se = gpudge.de(shard_archive=d, lfc_threshold=LFC_TAUS,
                        tau_star=TAUSTAR_LEVELS, tau_star_se=True)
    assert with_se.columns[-(n + 3):] == [
        "tau*_p0.05", "tau*_p0.5",
        "tau*_lo_p0.025", "tau*_hi_p0.025", "tau*_se"]
    assert with_se.columns[:-(n + 3)] == lfc_only.columns
    assert_frame_equal(with_se.select(both.columns), both,
                       check_dtypes=True, check_exact=True)


@needs_cuda
def test_stream_empty_targets_with_tau_star(archive_mode1, tmp_path):
    """n_targets == 0 takes refpool_de_core's empty_output_frame early return,
    a DIFFERENT code path from the populated projection -- so the tau* columns
    must be added to `extra_names` there as well, typed and in order."""
    from polars.testing import assert_frame_equal

    populated_path, _ = archive_mode1
    populated_ts = gpudge.de(shard_archive=populated_path,
                             tau_star=TAUSTAR_LEVELS, tau_star_se=True)

    # ntc_frac=1.0 so no target group survives; see the lfc twin above.
    adata = _make_synth(
        n_cells=120, n_genes=20, n_guides=2, sparse=True, ntc_frac=1.0)
    empty_path = _write_archive(
        adata, tmp_path / "few_ts", group_by="comparison",
        reference=["ntc"], target_shard_bytes=4096)
    empty_ts = gpudge.de(shard_archive=empty_path, tau_star=TAUSTAR_LEVELS,
                         tau_star_se=True)
    assert empty_ts.height == 0
    assert empty_ts.columns == populated_ts.columns
    assert empty_ts.schema == populated_ts.schema

    output_columns = {"target": "t", "feature": "f", "tau*_p0.5": "hl"}
    selected_empty = gpudge.de(
        shard_archive=empty_path, tau_star=TAUSTAR_LEVELS,
        output_columns=output_columns)
    selected_populated = populated_ts.select(
        list(output_columns)).rename(output_columns).head(0)
    assert_frame_equal(selected_empty, selected_populated,
                       check_dtypes=True, check_exact=True)


@needs_cuda
def test_reference_residency_guard_trims_allocators_before_reading_free_vram(
        monkeypatch):
    """The residency guard is a HARD guard -- a false negative refuses the run
    with "use a larger-memory GPU" for a reference that fits. It must therefore
    read free VRAM only AFTER the caching allocators are trimmed, or a previous
    de() in the same process can make it lie.

    Discriminating: asserts the ORDER of the two calls, so moving the trim after
    the read (or dropping it) fails. Deliberately does not assert on the two soft
    SIZER read sites -- deferring the reclaim there is a documented decision in
    the #76 design. (ultrareview 2026-08.)
    """
    import numpy as np
    import scipy.sparse as sp
    from gpudge import _shard_stream as _ss

    order = []
    real_release = _ss._release_gpu_memory
    real_free = _ss._free_gpu_bytes

    def release_spy(*a, **k):
        order.append("trim")
        return real_release(*a, **k)

    def free_spy(*a, **k):
        order.append("read_free")
        return real_free(*a, **k)

    monkeypatch.setattr(_ss, "_release_gpu_memory", release_spy)
    monkeypatch.setattr(_ss, "_free_gpu_bytes", free_spy)

    ref_X = sp.csr_matrix(np.arange(40, dtype=np.float32).reshape(8, 5))
    _ss._reference_prepass(
        ref_X, 5, torch.device("cuda"), 5, mean_calc="arithmetic",
        scale_main=False, scale_num=1.0e6, need_other_unit=False,
        need_row_sums=False, need_row_scales=False, oom_recovery=False)

    assert "trim" in order, "the allocators were never trimmed"
    assert "read_free" in order, "free VRAM was never read -- spy not wired up"
    assert order.index("trim") < order.index("read_free"), (
        f"free VRAM was read before the trim: {order}")
