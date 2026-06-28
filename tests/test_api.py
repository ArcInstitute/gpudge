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
    # Spot-check 5 random (guide, gene) pairs
    rows = result.sample(n=5, seed=0).to_dicts()
    for r in rows:
        gX = X[labels == r["target"]]
        j = int(synth_small.var_names.get_loc(r["feature"]))
        sp_res = mannwhitneyu(gX[:, j], ref_X[:, j],
                              alternative="two-sided", method="asymptotic")
        assert abs(r["p_value"] - sp_res.pvalue) < 1e-3


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
    for col in ("log2_fold_change", "p_value", "p_adj"):
        assert (j[col] - j[f"{col}_w"]).abs().max() < 1e-6


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
    for col in ("log2_fold_change", "p_value", "p_adj"):
        assert (j[col] - j[f"{col}_w"]).abs().max() < 1e-6


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
    for col in ("log2_fold_change", "p_value", "p_adj"):
        assert (j[col] - j[f"{col}_w"]).abs().max() < 1e-6


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
    for col in ("log2_fold_change", "p_value", "p_adj"):
        assert (j[col] - j[f"{col}_w"]).abs().max() < 1e-6


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
    any shardad import — so this runs without the optional streaming extra."""
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite"):
            de(shard_archive="/nonexistent", epsilon=bad)


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
