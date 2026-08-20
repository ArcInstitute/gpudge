# tests/test_api.py
import numpy as np
import pytest
import polars as pl
import torch
import gpudge
from scipy.stats import mannwhitneyu, false_discovery_control
from gpudge import de
from conftest import needs_cuda


@needs_cuda
def test_de_end_to_end_matches_scipy_ground_truth(synth_small):
    """Small synthetic: gpudge p_value/p_adj agree with scipy."""
    result = de(
        synth_small,
        groupby="comparison",
        reference="ntc",
    )
    assert isinstance(result, pl.DataFrame)
    # All non-ref guides in output
    target_groups = sorted(g for g in synth_small.obs["comparison"].unique()
                           if g != "ntc")
    assert sorted(result["target"].unique().to_list()) == target_groups

    # Ground truth: scipy MWU per (guide, gene)
    X = np.asarray(synth_small.X)
    labels = synth_small.obs["comparison"].to_numpy()
    ref_X = X[labels == "ntc"]
    # The full Cartesian key set, asserted: without this a missing or duplicated
    # gene passes, because the loop only ever checks the rows that ARE there.
    expected_keys = {(t, f) for t in target_groups
                     for f in synth_small.var_names}
    got_keys = set(zip(result["target"].to_list(), result["feature"].to_list()))
    assert got_keys == expected_keys
    assert len(got_keys) == result.height          # no duplicate rows
    # EVERY (guide, gene) pair, not 5 sampled ones: scipy on a 250-row fixture
    # costs milliseconds, and sampling left 98% of the oracle unused.
    got, want = [], []
    for r in result.to_dicts():
        gX = X[labels == r["target"]]
        j = int(synth_small.var_names.get_loc(r["feature"]))
        sp_res = mannwhitneyu(gX[:, j], ref_X[:, j],
                              alternative="two-sided", method="asymptotic")
        got.append(r["p_value"])
        want.append(sp_res.pvalue)
    # A 1e-3 ABSOLUTE bound was ~4 orders looser than the residual the 2026-08
    # review measured here (6e-8 abs / 4e-7 rel) and would have tolerated a
    # z-shift of order 1e-3 at p~0.5. Relative now, with atol kept in the
    # picture. NOT rtol=1e-6: that is only ~2.5x the measured relative residual,
    # close enough to the float32 staging floor (gpudge_arc#115) to be a
    # flakiness risk on another GPU generation. 1e-5 leaves ~25x.
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-7)


@needs_cuda
def test_filter_gene_min_mean_value_drops_low_expression(synth_small):
    """Inject a clearly sub-threshold gene and confirm it is dropped."""
    import numpy as np
    a = synth_small.copy()
    X = np.asarray(a.X)
    X[:, 0] = 0.0
    a.X = X      # g0 mean 0
    full = de(a, groupby="comparison", reference="ntc")
    filt = de(a, groupby="comparison", reference="ntc",
              filter_gene_min_mean_value=1.0)
    assert filt.height < full.height
    assert "g0" not in set(filt["feature"].to_list())


@needs_cuda
def test_de_output_columns_rename_select(synth_small):
    df = de(synth_small, groupby="comparison", reference="ntc",
            output_columns={"target": "guide", "p_value": "p"})
    assert df.columns == ["guide", "p"]
    assert ((df["p"] >= 0.0) & (df["p"] <= 1.0)).all()  # 'p' holds p_value, not target


@needs_cuda
def test_de_all_others_reference(synth_small):
    from gpudge import ALL_OTHERS
    df = de(synth_small, groupby="comparison", reference=ALL_OTHERS)
    # All groups appear as targets, including ntc
    assert "ntc" in df["target"].unique().to_list()
    assert ((df["p_value"] >= 0.0) & (df["p_value"] <= 1.0)).all()
    assert df["p_adj"].is_finite().all()


@needs_cuda
def test_de_csc_input_matches_csr_literal_ref(synth_small_sparse):
    """A CSC adata.X is coerced to CSR (exactly one warning) and yields
    bit-identical output to the CSR-input run. A MATERIALIZED AnnData is coerced
    IN PLACE — deliberate, and load-bearing for memory: it drops the caller's CSC
    refcount so only one sparse encoding is live (#66 design). Views cannot be
    rebound and keep a local instead; see
    test_de_on_a_csc_view_gets_the_CSR_fast_path. #66, ultrareview 2026-08"""
    import warnings as _w
    from polars.testing import assert_frame_equal
    csr_adata = synth_small_sparse                 # CSR from the fixture
    csc_adata = csr_adata.copy()
    csc_adata.X = csc_adata.X.tocsc()
    df_csr = de(csr_adata, groupby="comparison", reference="ntc")
    with _w.catch_warnings(record=True) as rec:
        _w.simplefilter("always")
        df_csc = de(csc_adata, groupby="comparison", reference="ntc")
    coerce = [w for w in rec if "converting to CSR" in str(w.message)]
    assert len(coerce) == 1
    keys = ["target", "feature"]
    assert_frame_equal(df_csr.sort(keys), df_csc.sort(keys),
                       check_exact=True)
    assert csc_adata.X.format == "csr"             # materialized input: coerced
    #                                                in place, per the #66 memory
    #                                                contract (a local copy would
    #                                                keep both encodings live).


@needs_cuda
def test_de_csc_all_others_not_coerced(synth_small_sparse):
    """reference=ALL_OTHERS must NOT coerce adata.X: the one-vs-rest densify
    never uses the numba CSR kernel, so coercing a CSC input would only add a
    per-chunk CSR->CSC round-trip. adata.X stays CSC, no warning, result
    bit-identical to the CSR run. #66"""
    import warnings as _w
    from polars.testing import assert_frame_equal
    from gpudge import ALL_OTHERS
    csr_adata = synth_small_sparse                 # CSR from the fixture
    csc_adata = csr_adata.copy()
    csc_adata.X = csc_adata.X.tocsc()
    df_csr = de(csr_adata, groupby="comparison", reference=ALL_OTHERS)
    with _w.catch_warnings(record=True) as rec:
        _w.simplefilter("always")
        df_csc = de(csc_adata, groupby="comparison", reference=ALL_OTHERS)
    coerce = [w for w in rec if "converting to CSR" in str(w.message)]
    assert len(coerce) == 0                         # ALL_OTHERS is not coerced
    assert csc_adata.X.format == "csc"              # left untouched
    keys = ["target", "feature"]
    assert_frame_equal(df_csr.sort(keys), df_csc.sort(keys),
                       check_exact=True)


@needs_cuda
def test_de_legacy_all_others_string_is_deprecated_but_works(synth_small):
    """Pre-v0.1 reference='all_others' still works with a DeprecationWarning."""
    with pytest.warns(DeprecationWarning, match="deprecated"):
        df = de(synth_small, groupby="comparison", reference="all_others")
    assert "ntc" in df["target"].unique().to_list()


@needs_cuda
def test_de_geometric_mean_option(synth_small):
    df_a = de(synth_small, groupby="comparison", reference="ntc",
              mean_calc="arithmetic")
    df_g = de(synth_small, groupby="comparison", reference="ntc",
              mean_calc="geometric")
    # p_values are identical (MWU doesn't depend on mean type)
    # log2_fold_change differs because it's computed from the means
    assert df_a.height == df_g.height
    j = df_a.join(df_g, on=["target","feature"], suffix="_g")
    assert j.height == df_a.height
    assert (j["p_value"] - j["p_value_g"]).abs().mean() < 1e-9
    assert (j["log2_fold_change"] - j["log2_fold_change_g"]).abs().mean() > 1e-3


def test_de_requires_cuda():
    """If CUDA is unavailable, de() hard-fails with a clear message."""
    import torch
    if torch.cuda.is_available():
        pytest.skip("this test exercises the no-CUDA branch")
    import anndata as ad
    import numpy as np
    a = ad.AnnData(X=np.zeros((4, 2), dtype=np.float32),
                   obs={"comparison": ["ntc","ntc","g","g"]})
    with pytest.raises(RuntimeError, match="CUDA"):
        de(a, groupby="comparison", reference="ntc")


@needs_cuda
def test_de_all_others_geometric_not_implemented():
    """The all-others × geometric combination is not supported."""
    import anndata as ad
    import numpy as np
    from gpudge import ALL_OTHERS
    a = ad.AnnData(X=np.zeros((6, 3), dtype=np.float32),
                   obs={"comparison": ["ntc","ntc","g","g","g","g"]})
    with pytest.raises(NotImplementedError, match=r"__all_others__"):
        de(a, groupby="comparison", reference=ALL_OTHERS,
           mean_calc="geometric")


# --- #22: gpu_gene_chunk_size invariance + OOM auto-recovery ---

@needs_cuda
def test_de_chunk_size_invariance(synth_medium):
    """de() results are identical across chunk sizes (chunk is only a
    memory-tiling detail; per-gene math is chunk-invariant)."""
    small = de(synth_medium, groupby="comparison", reference="ntc",
               gpu_gene_chunk_size=16, oom_recovery=False).sort(
                   ["target", "feature"])
    whole = de(synth_medium, groupby="comparison", reference="ntc",
               gpu_gene_chunk_size=synth_medium.n_vars,
               oom_recovery=False).sort(["target", "feature"])
    assert small.height == whole.height           # same row set across chunks
    j = small.join(whole, on=["target", "feature"], suffix="_w")
    assert j.height == whole.height               # exact (target, feature) match
    # EXACT: chunk-invariance is a bit-identity contract (de()'s docstring says
    # results are identical regardless of chunk size), and a 1e-6 tolerance
    # masked precisely the 1-2 ULP tie-term drift the 2026-08 ultrareview found
    # end-to-end. NaN positions must align too.
    for col in ("log2_fold_change", "p_value", "p_adj"):
        x, y = j[col].to_numpy(), j[f"{col}_w"].to_numpy()
        assert np.array_equal(x, y, equal_nan=True), (
            f"{col}: not chunk-invariant (max abs diff "
            f"{np.nanmax(np.abs(x - y))})")


@needs_cuda
def test_de_oom_recovery_downshift_matches_small_chunk(synth_medium, monkeypatch):
    """A forced OOM at the initial chunk downshifts (128 -> floor 64) and
    retries; the result equals an explicit chunk=64 run (idempotent retry)."""
    want = de(synth_medium, groupby="comparison", reference="ntc",
              gpu_gene_chunk_size=64, oom_recovery=False).sort(
                  ["target", "feature"])
    real = gpudge._stream.run_gene_chunks_with_recovery

    def flaky(n_genes, initial_chunk, process, **kw):
        tripped = {"done": False}

        def wrapped(a, b):
            if not tripped["done"] and (b - a) == initial_chunk:
                tripped["done"] = True
                raise torch.cuda.OutOfMemoryError("simulated")
            return process(a, b)
        return real(n_genes, initial_chunk, wrapped, **kw)

    monkeypatch.setattr(gpudge, "run_gene_chunks_with_recovery", flaky)
    got = de(synth_medium, groupby="comparison", reference="ntc",
             gpu_gene_chunk_size=128, oom_recovery=True).sort(
                 ["target", "feature"])
    assert got.height == want.height
    j = got.join(want, on=["target", "feature"], suffix="_w")
    assert j.height == want.height
    # EXACT: chunk-invariance is a bit-identity contract (de()'s docstring says
    # results are identical regardless of chunk size), and a 1e-6 tolerance
    # masked precisely the 1-2 ULP tie-term drift the 2026-08 ultrareview found
    # end-to-end. NaN positions must align too.
    for col in ("log2_fold_change", "p_value", "p_adj"):
        x, y = j[col].to_numpy(), j[f"{col}_w"].to_numpy()
        assert np.array_equal(x, y, equal_nan=True), (
            f"{col}: not chunk-invariant (max abs diff "
            f"{np.nanmax(np.abs(x - y))})")


@needs_cuda
def test_de_oom_recovery_false_raises(synth_medium, monkeypatch):
    """With oom_recovery=False, the first OOM raises instead of downshifting."""
    real = gpudge._stream.run_gene_chunks_with_recovery

    def always_oom(n_genes, initial_chunk, process, **kw):
        def wrapped(a, b):
            raise torch.cuda.OutOfMemoryError("simulated")
        return real(n_genes, initial_chunk, wrapped, **kw)

    monkeypatch.setattr(gpudge, "run_gene_chunks_with_recovery", always_oom)
    with pytest.raises(RuntimeError, match=r"oom_recovery=False"):
        de(synth_medium, groupby="comparison", reference="ntc",
           gpu_gene_chunk_size=128, oom_recovery=False)


# --- #27: ALL_OTHERS path also chunk-invariant + OOM-recoverable ---

@needs_cuda
def test_de_all_others_chunk_size_invariance(synth_medium):
    """The ALL_OTHERS (one-vs-rest) path is chunk-invariant too: gpudge#27
    wrapped its loop in the recovery driver, which re-slices [start:stop] per
    chunk, so a small chunk must yield the same stats as one whole-matrix pass."""
    from gpudge import ALL_OTHERS
    small = de(synth_medium, groupby="comparison", reference=ALL_OTHERS,
               gpu_gene_chunk_size=16, oom_recovery=False).sort(
                   ["target", "feature"])
    whole = de(synth_medium, groupby="comparison", reference=ALL_OTHERS,
               gpu_gene_chunk_size=synth_medium.n_vars,
               oom_recovery=False).sort(["target", "feature"])
    assert small.height == whole.height           # same row set across chunks
    j = small.join(whole, on=["target", "feature"], suffix="_w")
    assert j.height == whole.height               # exact (target, feature) match
    # EXACT: chunk-invariance is a bit-identity contract (de()'s docstring says
    # results are identical regardless of chunk size), and a 1e-6 tolerance
    # masked precisely the 1-2 ULP tie-term drift the 2026-08 ultrareview found
    # end-to-end. NaN positions must align too.
    for col in ("log2_fold_change", "p_value", "p_adj"):
        x, y = j[col].to_numpy(), j[f"{col}_w"].to_numpy()
        assert np.array_equal(x, y, equal_nan=True), (
            f"{col}: not chunk-invariant (max abs diff "
            f"{np.nanmax(np.abs(x - y))})")


@needs_cuda
def test_de_all_others_oom_recovery_downshift_matches_small_chunk(synth_medium,
                                                                  monkeypatch):
    """A CUDA OOM raised mid-body (inside _rank_with_ties, after the slice +
    upload + group_means) on the ALL_OTHERS path is caught by the recovery
    driver, which downshifts (128 -> 64) and re-slices; the result equals an
    explicit chunk=64 run (idempotent retry after a partial GPU allocation)."""
    from gpudge import ALL_OTHERS
    want = de(synth_medium, groupby="comparison", reference=ALL_OTHERS,
              gpu_gene_chunk_size=64, oom_recovery=False).sort(
                  ["target", "feature"])
    real_rank = gpudge._rank_with_ties
    calls = {"n": 0}

    def flaky_rank(X_chunk):
        # OOM on the first (128-wide) chunk, after the slice/upload/group_means
        # — forcing the driver to free the partial allocation and re-slice at 64.
        calls["n"] += 1
        if calls["n"] == 1:
            raise torch.cuda.OutOfMemoryError("simulated")
        return real_rank(X_chunk)

    monkeypatch.setattr(gpudge, "_rank_with_ties", flaky_rank)
    got = de(synth_medium, groupby="comparison", reference=ALL_OTHERS,
             gpu_gene_chunk_size=128, oom_recovery=True).sort(
                 ["target", "feature"])
    assert got.height == want.height              # no rows lost on the retry
    j = got.join(want, on=["target", "feature"], suffix="_w")
    assert j.height == want.height                # exact (target, feature) match
    # EXACT: chunk-invariance is a bit-identity contract (de()'s docstring says
    # results are identical regardless of chunk size), and a 1e-6 tolerance
    # masked precisely the 1-2 ULP tie-term drift the 2026-08 ultrareview found
    # end-to-end. NaN positions must align too.
    for col in ("log2_fold_change", "p_value", "p_adj"):
        x, y = j[col].to_numpy(), j[f"{col}_w"].to_numpy()
        assert np.array_equal(x, y, equal_nan=True), (
            f"{col}: not chunk-invariant (max abs diff "
            f"{np.nanmax(np.abs(x - y))})")


# --- ultrareview: fail-fast entry-point validation (GPU-free; the guards fire
# before the CUDA check, so they raise the same error with or without a GPU) ---

def _tiny_adata():
    import anndata as ad
    return ad.AnnData(X=np.zeros((4, 2), dtype=np.float32),
                      obs={"comparison": ["ntc", "ntc", "g", "g"]})


def test_de_rejects_invalid_mean_calc():
    with pytest.raises(ValueError, match="mean_calc"):
        de(_tiny_adata(), groupby="comparison", reference="ntc",
           mean_calc="bogus")


def test_de_rejects_negative_epsilon():
    with pytest.raises(ValueError, match="epsilon"):
        de(_tiny_adata(), groupby="comparison", reference="ntc", epsilon=-1.0)


def test_cpm_normalize_and_target_sum_mutually_exclusive():
    with pytest.raises(ValueError, match="only one"):
        gpudge.de(_tiny_adata(), groupby="comparison", reference="ntc",
                  cpm_normalize=True, normalize_target_sum=1e6)


@needs_cuda
def test_normalize_target_sum_bad_string(synth_small):
    with pytest.raises(ValueError, match="median"):
        gpudge.de(synth_small, groupby="comparison", reference="ntc",
                  normalize_target_sum="mean")


@needs_cuda
def test_normalize_target_sum_nonpositive(synth_small):
    with pytest.raises(ValueError, match="positive"):
        gpudge.de(synth_small, groupby="comparison", reference="ntc",
                  normalize_target_sum=0)


def test_de_rejects_unknown_output_columns():
    with pytest.raises(KeyError, match="output_columns"):
        de(_tiny_adata(), groupby="comparison", reference="ntc",
           output_columns={"p_values": "p"})


def test_de_rejects_duplicate_output_columns():
    with pytest.raises(ValueError, match="same name"):
        de(_tiny_adata(), groupby="comparison", reference="ntc",
           output_columns={"target": "x", "feature": "x"})


def test_de_rejects_nonfinite_epsilon():
    """L3: epsilon=nan/inf passed `epsilon < 0` and produced all-NaN log2fc."""
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite"):
            de(_tiny_adata(), groupby="comparison", reference="ntc", epsilon=bad)


def test_de_rejects_nonfinite_epsilon_streaming_dispatch():
    """Parity with the in-memory check: the streaming dispatch (shard_archive=)
    rejects non-finite epsilon at the same de()-level guard, which fires before
    any cellstream import — so this runs without the optional streaming extra."""
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite"):
            de(shard_archive="/nonexistent", epsilon=bad)


def test_de_rejects_bad_stream_n_workers():
    """stream_n_workers must be an int >= 1, rejected at the de()-level guard
    (fires before any cellstream import or archive open — runs on cellstream-less CPU
    CI). Lives here, not in test_shard_stream.py, whose module-level
    importorskip('cellstream') would skip it without the streaming extra."""
    with pytest.raises(ValueError, match="stream_n_workers"):
        de(shard_archive="/nonexistent", stream_n_workers=0)
    for bad in (1.5, True):
        with pytest.raises(TypeError, match="stream_n_workers"):
            de(shard_archive="/nonexistent", stream_n_workers=bad)


def test_de_rejects_bad_stream_prefetch():
    """stream_prefetch must be an int >= 0 (0 disables prefetch)."""
    with pytest.raises(ValueError, match="stream_prefetch"):
        de(shard_archive="/nonexistent", stream_prefetch=-1)
    for bad in (1.5, True):
        with pytest.raises(TypeError, match="stream_prefetch"):
            de(shard_archive="/nonexistent", stream_prefetch=bad)


def test_iter_kwargs_serial_and_prefetch():
    """_iter_kwargs(n_workers, prefetch) is a pure helper (no cellstream) — keep it
    covered on CPU CI. prefetch<=0 -> serial (bare iter_group_shards());
    prefetch>=1 -> {prefetch, n_workers} for that-many-way decode-ahead."""
    from gpudge import _shard_stream as ss
    assert ss._iter_kwargs(16, 0) == {}
    assert ss._iter_kwargs(16, -1) == {}
    assert ss._iter_kwargs(16, 2) == {"prefetch": 2, "n_workers": 16}
    assert ss._iter_kwargs(8, 4) == {"prefetch": 4, "n_workers": 8}


def test_missing_cellstream_import_message(monkeypatch):
    """The missing-streaming-extra error must name BOTH installs that work.

    Lives here, not in test_shard_stream.py, whose module-level
    importorskip('cellstream') skips it precisely when cellstream is absent --
    the one situation this message exists for. The test could therefore never
    run in the environment it describes.

    Asserts the commands in full. The previous `match="streaming"` is satisfied
    by the extra's name alone, so it stayed green while every install command in
    the message was rewritten. (codex, checkpoint 2.)

    `sys.modules[name] = None` is the import machinery's own "this module is
    unavailable" marker, so it raises ImportError for this one name and leaves
    every other import alone -- unlike replacing builtins.__import__, which was
    the previous technique and hooks every import in the interpreter for the
    duration. Verified against a venv that HAS cellstream installed; without one
    the guard is unfalsifiable, because the bare import already fails.
    (Gemini review, PR #144.)
    """
    import sys
    from gpudge._stream_backend import _import_cellstream

    monkeypatch.setitem(sys.modules, "cellstream", None)
    with pytest.raises(ImportError) as excinfo:
        _import_cellstream()
    msg = str(excinfo.value)
    assert "pip install 'gpudge[streaming]'" in msg      # from PyPI
    assert "pip install '.[streaming]'" in msg           # from a checkout


def test_should_device_decode_matrix(monkeypatch):
    """_should_device_decode truth table (pure helper, no cellstream/cupy needed —
    monkeypatched). Lives here, not in test_shard_stream.py, whose module-level
    importorskip('cellstream') would skip it on cellstream-less CPU CI. Device decode
    requires an x_cupy-capable archive (packed schema_version 3 — the v0.5.x
    default — OR legacy v2-directory 2) AND cupy AND cellstream.x_cupy."""
    from gpudge import _shard_stream as ss

    class _Arch:
        schema_version = 3                     # v0.5.x packed (the real format)

    a = _Arch()
    monkeypatch.setattr(ss, "_cupy_available", lambda: True)
    monkeypatch.setattr(ss, "_x_cupy_available", lambda: True)
    assert ss._should_device_decode(a) is True     # packed -> device

    a.schema_version = 2                       # legacy v2-directory -> device
    assert ss._should_device_decode(a) is True

    a.schema_version = 1                       # v1 archive -> host
    assert ss._should_device_decode(a) is False

    a.schema_version = 3
    monkeypatch.setattr(ss, "_cupy_available", lambda: False)   # no cupy -> host
    assert ss._should_device_decode(a) is False

    monkeypatch.setattr(ss, "_cupy_available", lambda: True)
    monkeypatch.setattr(ss, "_x_cupy_available", lambda: False)  # old cellstream -> host
    assert ss._should_device_decode(a) is False


def test_de_rejects_empty_output_columns():
    """L4: output_columns={} passed validation and yielded a 0-column frame."""
    with pytest.raises(ValueError, match="non-empty"):
        de(_tiny_adata(), groupby="comparison", reference="ntc",
           output_columns={})


def test_de_rejects_non_string_reference():
    """L5: a list/array reference gave an opaque 'truth value of an array is
    ambiguous' error instead of a clear message."""
    import numpy as np
    with pytest.raises(ValueError, match="reference"):
        de(_tiny_adata(), groupby="comparison", reference=["ntc"])
    with pytest.raises(ValueError, match="reference"):
        de(_tiny_adata(), groupby="comparison", reference=np.array(["ntc", "x"]))


@needs_cuda
def test_de_all_others_single_cell_p_is_finite():
    """L1: a 1-cell all_others input made tie_corr divide by N(N-1)=0, giving
    a NaN p; the denominator clamp must yield the graceful sentinel instead."""
    import anndata as ad
    import numpy as np
    X = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)   # 1 cell, 3 genes
    adata = ad.AnnData(X=X, obs={"comparison": ["G0"]},
                       var={"gene_id": ["g0", "g1", "g2"]})
    adata.var_names = ["g0", "g1", "g2"]
    df = de(adata, groupby="comparison", reference=gpudge.ALL_OTHERS)
    assert np.isfinite(df["p_value"].to_numpy()).all()


@needs_cuda
def test_log2fc_matches_closed_form(synth_small):
    """L9: log2_fold_change must equal log2((target_mean+eps)/(ref_mean+eps)).
    log2fc is the one output column scipy can't validate, and the README pins
    it as bit-perfect vs pdex -- so check it against its closed form."""
    eps = 0.5
    df = de(synth_small, groupby="comparison", reference="ntc", epsilon=eps)
    tm = df["target_mean"].to_numpy()
    rm = df["ref_mean"].to_numpy()
    expected = np.log2((tm + eps) / (rm + eps))
    np.testing.assert_allclose(
        df["log2_fold_change"].to_numpy(), expected, rtol=1e-5, atol=1e-6)


# --- ultrareview: integration-level correctness cross-checks vs scipy ---

@needs_cuda
def test_de_p_adj_matches_scipy_bh_per_target(synth_small):
    """de()'s p_adj equals scipy Benjamini-Hochberg applied independently per
    target — verifies the de()-level FDR wiring (the per-target row→group
    mapping), not just the bh_per_group unit."""
    df = de(synth_small, groupby="comparison", reference="ntc")
    for tgt in df["target"].unique().to_list():
        sub = df.filter(pl.col("target") == tgt)
        want = false_discovery_control(sub["p_value"].to_numpy(), method="bh")
        np.testing.assert_allclose(sub["p_adj"].to_numpy(), want,
                                   rtol=1e-9, atol=1e-12)


@needs_cuda
def test_de_all_others_p_value_matches_scipy(synth_small):
    """ALL_OTHERS p_values match scipy one-vs-rest MWU (verifies _rank_with_ties
    + the 1-vs-rest assembly, not only chunk-invariance)."""
    from gpudge import ALL_OTHERS
    df = de(synth_small, groupby="comparison", reference=ALL_OTHERS)
    X = np.asarray(synth_small.X)
    labels = synth_small.obs["comparison"].to_numpy()
    for r in df.sample(n=5, seed=0).to_dicts():
        j = int(synth_small.var_names.get_loc(r["feature"]))
        in_grp = labels == r["target"]
        sp_res = mannwhitneyu(X[in_grp, j], X[~in_grp, j],
                              alternative="two-sided", method="asymptotic")
        assert abs(r["p_value"] - sp_res.pvalue) < 1e-3


@needs_cuda
def test_de_output_columns_dest_shadows_default(synth_small):
    """A destination name that shadows an unselected default column must not
    collide — select-then-rename, not rename-then-select. (Codex review.)"""
    df = de(synth_small, groupby="comparison", reference="ntc",
            output_columns={"target": "feature"})
    assert df.columns == ["feature"]
    # holds the target (guide) names, not the original gene/feature names
    assert set(df["feature"].unique().to_list()) <= set(
        synth_small.obs["comparison"].unique().tolist())


@needs_cuda
def test_filter_gene_min_mean_value_reads_as_supplied_X(synth_small):
    import numpy as np
    from gpudge import de
    a = synth_small.copy()
    X = np.asarray(a.X)
    X[:, 0] = 0.0
    a.X = X      # gene g0 all-zero
    out = de(a, groupby="comparison", reference="ntc",
             filter_gene_min_mean_value=0.5)
    assert "g0" not in set(out["feature"].to_list())
    assert out.height > 0


@needs_cuda
def test_filter_gene_min_total_value_threshold(synth_small):
    from gpudge import de
    a = synth_small.copy()
    full = de(a, groupby="comparison", reference="ntc")
    hi = de(a, groupby="comparison", reference="ntc",
            filter_gene_min_total_value=1e12)
    assert hi.height == 0
    lo = de(a, groupby="comparison", reference="ntc",
            filter_gene_min_total_value=-1.0)
    assert lo.height == full.height


@needs_cuda
def test_keep_genes_restricts_features(synth_small):
    import numpy as np
    from gpudge import de
    a = synth_small.copy()
    mask = np.zeros(a.n_vars, dtype=bool)
    mask[[1, 3, 5]] = True
    out = de(a, groupby="comparison", reference="ntc",
             keep_genes=mask)
    assert set(out["feature"].to_list()) == {"g1", "g3", "g5"}


@needs_cuda
def test_keep_genes_bad_dtype_raises(synth_small):
    import numpy as np
    from gpudge import de
    with pytest.raises(ValueError, match="boolean"):
        de(synth_small, groupby="comparison", reference="ntc",
           keep_genes=np.ones(synth_small.n_vars, dtype=int))


@needs_cuda
def test_filter_gene_all_others_mean_value(synth_small):
    import numpy as np
    from gpudge import de, ALL_OTHERS
    a = synth_small.copy()
    X = np.asarray(a.X)
    X[:, 0] = 0.0
    a.X = X
    out = de(a, groupby="comparison", reference=ALL_OTHERS,
             filter_gene_min_mean_value=0.5)
    assert "g0" not in set(out["feature"].to_list())
    assert out.height > 0


@needs_cuda
def test_min_feature_filter_removed_raises(synth_small):
    with pytest.raises(ValueError, match="filter_gene_min_mean_value"):
        de(synth_small, groupby="comparison", reference="ntc",
           min_feature_filter=1.0)


def test_csr_row_sums_accumulates_in_float64_without_numba(monkeypatch):
    """The non-numba fallback must reduce in float64, not in X.data's dtype.

    scipy's X.sum(axis=1) reduces in the input dtype; a float32 CSR whose row
    total passes 2**24 then rounds, while the same counts as an integer dtype
    reduce exactly. The streaming cell path hands gpudge float32 CSR, so this is
    the difference between two layouts agreeing and disagreeing.
    """
    import scipy.sparse as sp
    from gpudge import _csr_dense as cd

    # One row whose exact total is 2**24 + 1 -- not representable in float32.
    vals = np.array([2.0 ** 24, 1.0], dtype=np.float32)
    X = sp.csr_matrix((vals, np.array([0, 1]), np.array([0, 2])), shape=(1, 2))
    monkeypatch.setattr(cd, "HAS_NUMBA", False)
    got = cd.csr_row_sums(X)
    assert got.dtype == np.float64
    assert got[0] == 2.0 ** 24 + 1.0, f"reduced in float32: {got[0]!r}"


def test_archive_and_shard_archive_together_raises():
    with pytest.raises(ValueError, match="only archive="):
        gpudge.de(archive="a.csad", shard_archive="b.shad")


def test_shard_archive_is_deprecated_alias():
    """shard_archive= still works but warns. stream_n_workers=0 makes the call
    fail on archive-free validation, so the test cannot depend on GPU presence
    or on which error a missing file produces."""
    with pytest.warns(DeprecationWarning, match=r"use de\(archive="):
        with pytest.raises(ValueError, match="stream_n_workers"):
            gpudge.de(shard_archive="/nonexistent/x.shad", stream_n_workers=0)


def test_neither_adata_nor_archive_raises_mentions_archive():
    with pytest.raises(ValueError, match="exactly one of adata= or archive="):
        gpudge.de()


def test_both_adata_and_archive_raises(synth_small):
    with pytest.raises(ValueError, match="exactly one of adata= or archive="):
        gpudge.de(synth_small, archive="x.csad")


def test_tau_star_se_is_documented_with_its_domain_caveat():
    """#112: on raw counts the delta=0 tie atom dominates hi-lo, so tau*_se is
    not a sampling SE there. The caveat must be ON the parameter itself."""
    doc = de.__doc__
    assert "tau_star_se : bool" in doc
    assert "tau*_lo_p0.025" in doc and "tau*_hi_p0.025" in doc
    assert "tau*_se" in doc
    # Slice to the END of the parameter block, not a magic character count.
    # The plan prescribed [:3000], but the docstring it also prescribed puts
    # `normalize_target_sum` ~3.6k characters in, so that window could never
    # see it. Widening to a bigger number would instead spill into the NEXT
    # parameter and stop proving the caveat is on THIS one; the block boundary
    # proves exactly the intended claim and survives the block growing.
    block = doc.split("tau_star_se : bool")[1].split("gpu_gene_chunk_size :")[0]
    assert "normalize_target_sum" in block


def test_densify_input_on_a_view_raises_before_any_gpu_work(synth_small_sparse):
    """densify_input=True cannot be honoured on an AnnData VIEW: assigning to a
    view's .X writes THROUGH to the parent instead of rebinding, so the dense
    array was built, scattered back into the parent and discarded -- the caller
    paid the full dense allocation (~310 GB peak at CCL_2 scale per the docstring)
    for no speedup, after a warning saying the mutation had happened.

    Deliberately NOT @needs_cuda: the guard is a preflight that runs before the
    torch.cuda.is_available() probe, so this pins the ordering too -- a
    regression that moved the check later would fail here on a CPU-only host.
    (codex review of the ultrareview 2026-08 fix.)
    """
    view = synth_small_sparse[synth_small_sparse.obs["comparison"] != "__none__"]
    assert view.is_view                                # sanity: really a view
    with pytest.raises(ValueError, match=r"adata\.copy\(\)"):
        de(view, groupby="comparison", reference="ntc", densify_input=True)


@needs_cuda
def test_de_on_a_csc_view_gets_the_CSR_fast_path(synth_small_sparse, monkeypatch):
    """A CSC-backed VIEW must reach the numba CSR gather, and the caller's matrix
    must be untouched.

    Spying on the gather is the only assertion that DISCRIMINATES. Under the
    pre-fix `adata.X = ensure_csr(...)`, anndata wrote the CSR back through into
    the parent and re-read the parent slice, so `adata.X` stayed a
    SparseCSCMatrixView -- but the parent stayed CSC and the OUTPUT stayed
    correct, only slower. So checking output + `parent.X.format` (as the first
    version of this test did) passes on the buggy code too; what actually broke
    was that every gather tile fell back to scipy slicing, the regression #66
    added the coercion to prevent. (codex review round 2.)
    """
    import scipy.sparse as _sp
    from polars.testing import assert_frame_equal
    parent = synth_small_sparse.copy()
    parent.X = parent.X.tocsc()
    view = parent[parent.obs["comparison"] != "__none__"]
    assert view.is_view

    seen = []
    real = gpudge.csr_rows_col_range_to_dense

    def spy(X, *args, **kwargs):
        seen.append(X.format if _sp.issparse(X) else f"dense:{type(X).__name__}")
        return real(X, *args, **kwargs)

    monkeypatch.setattr(gpudge, "csr_rows_col_range_to_dense", spy)
    keys = ["target", "feature"]
    got = de(view, groupby="comparison", reference="ntc").sort(keys)

    assert seen, "the gather was never called -- the spy is not wired up"
    assert set(seen) == {"csr"}, f"gather saw non-CSR input: {sorted(set(seen))}"
    assert parent.X.format == "csc"          # caller's matrix untouched

    monkeypatch.setattr(gpudge, "csr_rows_col_range_to_dense", real)
    materialised = view.to_memory() if hasattr(view, "to_memory") else view.copy()
    materialised.X = materialised.X.tocsr()
    exp = de(materialised, groupby="comparison", reference="ntc").sort(keys)
    assert_frame_equal(got, exp, check_exact=True)


@needs_cuda
def test_densify_input_releases_the_sparse_matrix_before_the_chunk_loop(
        synth_small_sparse, monkeypatch):
    """densify_input=True must release the sparse encoding BEFORE the gene-chunk
    loop, not merely by the time de() returns.

    Observing a weakref after de() returns proves nothing: CPython destroys the
    function frame on return, so every internal local dies regardless and the
    test passes with or without the fix (I wrote that version first and it did
    exactly that). The contract is about PEAK memory *during* the call — holding
    the sparse matrix alongside the dense one is the ~310 GB at CCL_2 scale that
    densify_input exists to avoid — so the observation has to happen mid-call.
    This hooks the gather, which runs after densification, and checks the sparse
    matrix is already unreachable at that point. (codex review round 4.)
    """
    import gc
    import weakref
    a = synth_small_sparse.copy()
    sparse_ref = weakref.ref(a.X)
    observed = []
    real = gpudge.csr_rows_col_range_to_dense

    def spy(X, *args, **kwargs):
        if not observed:                     # first gather call only
            gc.collect()
            observed.append(sparse_ref() is None)
        return real(X, *args, **kwargs)

    monkeypatch.setattr(gpudge, "csr_rows_col_range_to_dense", spy)
    with pytest.warns(UserWarning, match="densify_input=True"):
        de(a, groupby="comparison", reference="ntc", densify_input=True)
    assert observed, "the gather never ran -- the spy is not wired up"
    assert observed[0], (
        "the sparse matrix was still reachable when the chunk loop started: "
        "densify_input freed nothing, so sparse and dense were both resident")


@needs_cuda
def test_densify_input_releases_the_COERCED_csr_before_the_chunk_loop(
        synth_small_sparse, monkeypatch):
    """Same contract on the CSC path: the CSR that ensure_csr produced must also
    be gone by the time the chunk loop runs. (codex review round 4.)"""
    import gc
    import weakref
    captured = {}
    real_ensure = gpudge.ensure_csr
    real_gather = gpudge.csr_rows_col_range_to_dense
    observed = []

    def ensure_spy(X, **kwargs):
        out = real_ensure(X, **kwargs)
        if out is not X:
            captured["ref"] = weakref.ref(out)
        return out

    def gather_spy(X, *args, **kwargs):
        if not observed and "ref" in captured:
            gc.collect()
            observed.append(captured["ref"]() is None)
        return real_gather(X, *args, **kwargs)

    monkeypatch.setattr(gpudge, "ensure_csr", ensure_spy)
    monkeypatch.setattr(gpudge, "csr_rows_col_range_to_dense", gather_spy)
    a = synth_small_sparse.copy()
    a.X = a.X.tocsc()
    with pytest.warns(UserWarning):          # coercion + densify warnings
        de(a, groupby="comparison", reference="ntc", densify_input=True)
    assert "ref" in captured, "ensure_csr never coerced -- the spy is not wired up"
    assert observed, "the gather never ran -- the spy is not wired up"
    assert observed[0], (
        "the coerced CSR was still reachable when the chunk loop started")





@needs_cuda
def test_de_is_silent_on_the_tie_fallback(synth_small, monkeypatch):
    """A real de() run that hits the tie-correction fallback emits NO warning.

    Pins the documented contract end-to-end (see de()'s gpu_gene_chunk_size
    docstring): the limitation is documented, not warned about. Discriminating —
    re-wiring a warning into any entry point fails this. (PR #124.)
    """
    import warnings as _w
    import gpudge._mwu as _mwu
    monkeypatch.setattr(_mwu, "_TIE_INT64_MAX_K", 8)   # force the fallback
    with _w.catch_warnings(record=True) as rec:
        _w.simplefilter("always")
        out = de(synth_small, groupby="comparison", reference="ntc")
    assert out.height > 0
    noisy = [r for r in rec if "gpu_gene_chunk_size" in str(r.message)
             or "tie-correction" in str(r.message)]
    assert not noisy, f"de() warned about the fallback: {[str(r.message) for r in noisy]}"


# --- 2026-08 ultrareview (lows) ---------------------------------------------

@pytest.mark.parametrize("bad", ["False", "false", "no", "0", 1, 0.5, None, 0])
def test_de_rejects_non_bool_cpm_normalize(bad):
    """`cpm_normalize='false'` was byte-identical to `cpm_normalize=True`.

    CPU-runnable on purpose: the guard sits in de()'s fail-fast block, above
    the CUDA probe, so a typo costs nothing.
    """
    with pytest.raises(ValueError, match="cpm_normalize must be True or False"):
        de(_tiny_adata(), groupby="comparison", reference="ntc",
           cpm_normalize=bad)


@pytest.mark.parametrize("good", [True, False])
def test_de_accepts_real_bool_cpm_normalize_past_the_guard(good, monkeypatch):
    """The guard must not reject the two values that ARE legal. Asserts the
    EXACT CUDA-unavailable error rather than suppressing RuntimeError, which
    would have swallowed a guard regression and a GPU crash alike."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    for value in (good, np.bool_(good)):
        with pytest.raises(RuntimeError, match="gpudge requires a CUDA GPU"):
            de(_tiny_adata(), groupby="comparison", reference="ntc",
               cpm_normalize=value)


@needs_cuda
def test_de_reference_only_input_returns_the_typed_empty_frame():
    """A reference-only object (the only group IS the reference) has nothing to
    test. It used to run the WHOLE GPU pass and then die on
    `IndexError: arrays used as indices must be of integer (or boolean) type`
    from an untyped, and therefore float64, `np.array([])`."""
    import anndata as ad
    import scipy.sparse as sp
    from gpudge._output import empty_output_frame
    rng = np.random.default_rng(0)
    X = rng.integers(0, 5, size=(40, 12)).astype(np.float32)
    a = ad.AnnData(X=sp.csr_matrix(X), obs={"comparison": ["ntc"] * 40},
                   var={"gene_id": [f"g{i}" for i in range(12)]})
    out = de(a, groupby="comparison", reference="ntc")
    assert out.height == 0
    # Same schema as refpool_de_core's n_targets == 0 return, not merely "empty".
    assert out.schema == empty_output_frame().schema


@needs_cuda
def test_de_reference_only_input_keeps_the_schema_with_extras():
    """The zero-target return must carry the lfc/tau* columns too, or a caller
    concatenating per-chunk frames gets a schema mismatch."""
    import anndata as ad
    import scipy.sparse as sp
    rng = np.random.default_rng(0)
    X = rng.integers(0, 5, size=(40, 12)).astype(np.float32)
    a = ad.AnnData(X=sp.csr_matrix(X), obs={"comparison": ["ntc"] * 40},
                   var={"gene_id": [f"g{i}" for i in range(12)]})
    out = de(a, groupby="comparison", reference="ntc",
             lfc_threshold=1.0, tau_star=0.5)
    assert out.height == 0
    populated = de(_synth_two_group(), groupby="comparison", reference="ntc",
                   lfc_threshold=1.0, tau_star=0.5)
    assert out.schema == populated.schema


def _synth_two_group():
    import anndata as ad
    import scipy.sparse as sp
    rng = np.random.default_rng(1)
    X = rng.integers(0, 5, size=(40, 12)).astype(np.float32)
    return ad.AnnData(X=sp.csr_matrix(X),
                      obs={"comparison": ["ntc"] * 20 + ["g1"] * 20},
                      var={"gene_id": [f"g{i}" for i in range(12)]})


@needs_cuda
def test_de_zero_gene_input_returns_the_typed_empty_frame():
    """0-var AnnData: the auto sizer returned a 0 chunk and the driver raised
    `initial_chunk must be > 0, got 0` — a parameter name the caller never
    passed — while the SAME input with a pinned gpu_gene_chunk_size already
    returned a correct 0-row frame."""
    import anndata as ad
    import scipy.sparse as sp
    from gpudge._output import empty_output_frame
    a = ad.AnnData(X=sp.csr_matrix((9, 0), dtype=np.float32),
                   obs={"comparison": ["ntc"] * 3 + ["g1"] * 3 + ["g2"] * 3})
    out = de(a, groupby="comparison", reference="ntc")
    assert out.height == 0
    assert out.schema == empty_output_frame().schema
    # The auto path must now agree with the pinned one it used to contradict.
    pinned = de(a, groupby="comparison", reference="ntc",
                gpu_gene_chunk_size=256)
    assert out.schema == pinned.schema


@needs_cuda
def test_de_epsilon_zero_emits_no_runtime_warning():
    """epsilon=0 is documented to yield NaN / ±inf. Producing a DOCUMENTED
    value must not also emit an unsuppressed numpy RuntimeWarning — which
    under `-W error::RuntimeWarning` turned the documented path into a crash."""
    import warnings
    a = _epsilon_fixture()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        de(a, groupby="comparison", reference="ntc", epsilon=0.0)
    runtime = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert not runtime, [str(w.message) for w in runtime]


@needs_cuda
def test_de_epsilon_zero_raises_nothing_under_a_strict_filter():
    """The consequence, pinned separately from the warning itself."""
    import warnings
    a = _epsilon_fixture()
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        de(a, groupby="comparison", reference="ntc", epsilon=0.0)


def _epsilon_fixture():
    """g0 zero in both groups, g1 target-only, g2 reference-only, g3 normal."""
    import anndata as ad
    import scipy.sparse as sp
    X = np.zeros((40, 4), dtype=np.float32)
    X[:20, 1] = 3.0                      # target-only  -> ref_mean 0  -> +inf
    X[20:, 2] = 3.0                      # ref-only     -> tgt_mean 0  -> -inf
    X[:, 3] = np.arange(40, dtype=np.float32) % 5 + 1
    a = ad.AnnData(X=sp.csr_matrix(X),
                   obs={"comparison": ["g1"] * 20 + ["ntc"] * 20})
    # var_names, NOT a `gene_id` COLUMN: `var={"gene_id": [...]}` leaves the
    # index a RangeIndex, so `feature` comes back as "0".."3".
    a.var_names = [f"g{i}" for i in range(4)]
    return a


@needs_cuda
def test_de_epsilon_zero_pins_the_documented_degenerate_log2fc():
    """README and the de() docstring both promise NaN for a both-zero gene and
    ±inf for a one-sided one, "matching pdex". Nothing asserted it: replacing
    the whole ±inf/NaN contract with 0/±30 left the ENTIRE suite green when the
    2026-08 review measured it (737 cases at the time)."""
    df = de(_epsilon_fixture(), groupby="comparison", reference="ntc",
            epsilon=0.0).sort("feature")
    lfc = dict(zip(df["feature"].to_list(),
                   df["log2_fold_change"].to_numpy()))
    assert np.isnan(lfc["g0"]), lfc          # zero in BOTH groups  -> NaN
    assert lfc["g1"] == np.inf, lfc          # target-only          -> +inf
    assert lfc["g2"] == -np.inf, lfc         # reference-only       -> -inf
    assert np.isfinite(lfc["g3"]), lfc       # control


@pytest.mark.parametrize("bad", [0, -1, 1.5, "256", True, False])
def test_de_rejects_a_bad_gpu_gene_chunk_size(bad):
    """It was unvalidated: a 0 reached `run_gene_chunks_with_recovery` and
    raised `initial_chunk must be > 0, got 0` — an internal parameter name the
    caller never passed. `True` is rejected explicitly (bool is an Integral)."""
    with pytest.raises(ValueError, match="gpu_gene_chunk_size must be"):
        de(_tiny_adata(), groupby="comparison", reference="ntc",
           gpu_gene_chunk_size=bad)
