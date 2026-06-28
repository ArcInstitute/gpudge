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
from conftest import _make_synth, needs_cuda   # synth generator + cuda marker


def test_neither_adata_nor_archive_raises():
    with pytest.raises(ValueError, match="exactly one of adata"):
        gpudge.de()


def test_both_adata_and_archive_raises(synth_small):
    with pytest.raises(ValueError, match="exactly one of adata"):
        gpudge.de(synth_small, shard_archive="/nonexistent", groupby="target_guide",
                  reference="non-targeting-0")


def test_densify_input_with_archive_raises(synth_small):
    with pytest.raises(ValueError, match="densify_input"):
        gpudge.de(shard_archive="/nonexistent", densify_input=True)


def test_all_others_with_archive_raises():
    with pytest.raises(NotImplementedError, match="ALL_OTHERS"):
        gpudge.de(shard_archive="/nonexistent", reference=ALL_OTHERS)


def test_anndata_reference_without_archive_raises(synth_small):
    # AnnData reference is streaming-only.
    with pytest.raises(ValueError, match="AnnData reference"):
        gpudge.de(synth_small, groupby="target_guide", reference=synth_small)


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
    with pytest.warns(DeprecationWarning, match="deprecated"):
        with pytest.raises(NotImplementedError, match="ALL_OTHERS"):
            gpudge.de(shard_archive="/nonexistent", reference="all_others")


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


def test_nonfinite_epsilon_with_archive_raises():
    """Streaming epsilon guard mirrors in-memory de(): reject NaN/inf, not just
    < 0 (the parity gap the ultrareview's epsilon validation left in stream_de).
    Fires before the archive is opened, so /nonexistent is fine."""
    for bad in (float("inf"), float("nan")):
        with pytest.raises(ValueError, match="epsilon must be a finite"):
            gpudge.de(shard_archive="/nonexistent", epsilon=bad)


# ---------------------------------------------------------------------------
# Task-2: archive fixtures + _resolve_streaming tests
# ---------------------------------------------------------------------------
# shardad is the optional [streaming] extra; skip the whole streaming-test
# module cleanly when it isn't installed instead of erroring at collection.
shardad = pytest.importorskip("shardad", reason="requires gpudge[streaming]")


def _write_archive(adata, path, *, group_by, reference, target_shard_bytes):
    shardad.write_sharded(
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
    d, adata = archive_mode1
    arch = shardad.ShardedArchive(d)
    groupby, mode, ref_X, _ = ss._resolve_streaming(arch, None, None)
    assert groupby == "comparison"
    assert mode == "archive_ref"
    n_ntc = int(np.char.startswith(
        adata.obs["comparison"].to_numpy().astype(str), "ntc").sum())
    assert ref_X.shape[0] == n_ntc


def test_resolve_mode1_groupby_mismatch_raises(archive_mode1):
    d, _ = archive_mode1
    arch = shardad.ShardedArchive(d)
    with pytest.raises(ValueError, match="groupby"):
        ss._resolve_streaming(arch, "WRONG", None)


def test_resolve_mode1_reference_mismatch_raises(archive_mode1):
    d, _ = archive_mode1
    arch = shardad.ShardedArchive(d)
    with pytest.raises(ValueError, match="reference"):
        ss._resolve_streaming(arch, None, "not-a-ref-label")


def test_resolve_mode2_external_pool(archive_mode2):
    d, adata_g, adata_ntc, _ = archive_mode2
    arch = shardad.ShardedArchive(d)
    groupby, mode, ref_X, _ = ss._resolve_streaming(arch, None, adata_ntc)
    assert mode == "external_ref"
    assert ref_X.shape[0] == adata_ntc.n_obs


def test_resolve_mode2_on_archive_with_reference_warns(archive_mode1):
    # An AnnData reference on a reference-bearing archive now WARNS and uses the
    # external pool (Semantics A); the archive's own reference shard is ignored.
    d, adata = archive_mode1
    arch = shardad.ShardedArchive(d)
    with pytest.warns(UserWarning, match="ignored in favor of the external"):
        groupby, mode, ref_X, _ = ss._resolve_streaming(arch, None, adata)
    assert mode == "external_ref"
    assert ref_X.shape[0] == adata.n_obs            # external pool, not the archive ref shard


def test_resolve_mode2_gene_axis_mismatch_raises(archive_mode2):
    d, adata_g, adata_ntc, _ = archive_mode2
    arch = shardad.ShardedArchive(d)
    bad = adata_ntc[:, :adata_ntc.n_vars - 1].copy()    # wrong n_vars
    with pytest.raises(ValueError, match="gene"):
        ss._resolve_streaming(arch, None, bad)


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
    assert_frame_equal(got_s, exp_s)


def test_resolve_non_grouped_archive_raises(tmp_path):
    adata = _make_synth(n_cells=200, n_genes=20, n_guides=4, sparse=True)
    shardad.write_sharded(adata, str(tmp_path / "ng"), format="v2")  # no group_by
    arch = shardad.ShardedArchive(str(tmp_path / "ng"))
    with pytest.raises(ValueError, match="group_by"):
        ss._resolve_streaming(arch, None, None)


def test_enumerate_targets(archive_mode1):
    d, adata = archive_mode1
    arch = shardad.ShardedArchive(d)
    targets, tgt_index = ss._enumerate_targets(arch)
    # All non-ntc comparison labels should appear exactly once; ntc excluded.
    comp = adata.obs["comparison"].to_numpy().astype(str)
    expected = {c for c in set(comp) if not c.startswith("ntc")}
    assert set(targets) == expected
    assert len(targets) == len(set(targets))           # disjoint, no dups
    assert all(tgt_index[t] == i for i, t in enumerate(targets))

def test_enumerate_targets_mode2_includes_all(archive_mode2):
    d, adata_g, _, _ = archive_mode2
    arch = shardad.ShardedArchive(d)
    targets, _ = ss._enumerate_targets(arch)
    comp = adata_g.obs["comparison"].to_numpy().astype(str)
    assert set(targets) == set(comp)                   # no reference shard → all groups


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
    with pytest.raises(RuntimeError, match="reference.*too large|does not fit"):
        ss._reference_prepass(
            X, n_genes=4000, device=torch.device("cuda"), chunk=512,
            mean_calc="arithmetic", scale_main=False, scale_num=1.0e6,
            need_other_unit=False,
            need_row_sums=False, need_row_scales=False, oom_recovery=True)


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
    arith, reported, other, u1, p = ss.group_chunk_stats(
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
    # An archive whose only non-reference content is empty → empty result, right schema.
    adata = _make_synth(n_cells=120, n_genes=20, n_guides=2, sparse=True, ntc_frac=0.99)
    d = _write_archive(adata, tmp_path / "few", group_by="comparison",
                       reference=["ntc"], target_shard_bytes=4096)
    df = gpudge.de(shard_archive=d)
    from gpudge._output import DEFAULT_OUTPUT_COLUMNS
    assert list(df.columns) == list(DEFAULT_OUTPUT_COLUMNS)


@needs_cuda
def test_stream_empty_targets_respects_output_columns(tmp_path):
    """The empty-archive early return must honour output_columns (route through
    the same select/rename as the non-empty path), so the schema is identical
    whether or not the archive has targets. Regression for the ultrareview
    finding: the early return emitted DEFAULT_OUTPUT_COLUMNS verbatim, so a
    caller passing output_columns got the wrong column names on empty results."""
    adata = _make_synth(n_cells=120, n_genes=20, n_guides=2, sparse=True, ntc_frac=0.99)
    d = _write_archive(adata, tmp_path / "few", group_by="comparison",
                       reference=["ntc"], target_shard_bytes=4096)
    df = gpudge.de(shard_archive=d, output_columns={"target": "guide", "p_value": "p"})
    assert list(df.columns) == ["guide", "p"]
    # Schema must be typed (not Null) so the empty result aligns with non-empty
    # results downstream (concat / schema-aware access). See Gemini review.
    import polars as pl
    assert df.schema["guide"] == pl.String
    assert df.schema["p"] == pl.Float64


def test_missing_shardad_import_message(monkeypatch):
    import builtins
    real_import = builtins.__import__
    def fake(name, *a, **k):
        if name == "shardad":
            raise ImportError("no shardad")
        return real_import(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", fake)
    with pytest.raises(ImportError, match="streaming"):
        ss._import_shardad()


def test_resolve_mode1_reference_read_none_raises(archive_mode1, monkeypatch):
    # Manifest declares a reference shard but read_reference() returns None
    # (inconsistent/corrupted archive) → clear ValueError, not AttributeError.
    d, _ = archive_mode1
    arch = shardad.ShardedArchive(d)
    monkeypatch.setattr(arch, "read_reference", lambda *a, **k: None)
    with pytest.raises(ValueError, match="returned None|inconsistent"):
        ss._resolve_streaming(arch, None, None)
