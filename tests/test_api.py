# tests/test_api.py
import numpy as np
import pytest
import polars as pl
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
        min_feature_filter=0.0,   # no filter so every (g, gene) is present
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
    rng = np.random.default_rng(0)
    rows = result.sample(n=5, seed=0).to_dicts()
    for r in rows:
        gX = X[labels == r["target"]]
        j = int(synth_small.var_names.get_loc(r["feature"]))
        sp_res = mannwhitneyu(gX[:, j], ref_X[:, j],
                              alternative="two-sided", method="asymptotic")
        assert abs(r["p_value"] - sp_res.pvalue) < 1e-3


@needs_cuda
def test_de_min_feature_filter_drops_low_expression(synth_small):
    """High threshold → fewer rows than full grid."""
    full = de(synth_small, groupby="comparison", reference="ntc",
              min_feature_filter=0.0)
    filt = de(synth_small, groupby="comparison", reference="ntc",
              min_feature_filter=1.0)
    assert filt.height < full.height


@needs_cuda
def test_de_output_columns_rename_select(synth_small):
    df = de(synth_small, groupby="comparison", reference="ntc",
            output_columns={"target": "guide", "p_value": "p"})
    assert df.columns == ["guide", "p"]


@needs_cuda
def test_de_all_others_reference(synth_small):
    from gpudge import ALL_OTHERS
    df = de(synth_small, groupby="comparison", reference=ALL_OTHERS,
            min_feature_filter=0.0)
    # All groups appear as targets, including ntc
    assert "ntc" in df["target"].unique().to_list()


@needs_cuda
def test_de_legacy_all_others_string_is_deprecated_but_works(synth_small):
    """Pre-v0.1 reference='all_others' still works with a DeprecationWarning."""
    with pytest.warns(DeprecationWarning, match="deprecated"):
        df = de(synth_small, groupby="comparison", reference="all_others",
                min_feature_filter=0.0)
    assert "ntc" in df["target"].unique().to_list()


@needs_cuda
def test_de_geometric_mean_option(synth_small):
    df_a = de(synth_small, groupby="comparison", reference="ntc",
              mean_calc="arithmetic", min_feature_filter=0.0)
    df_g = de(synth_small, groupby="comparison", reference="ntc",
              mean_calc="geometric", min_feature_filter=0.0)
    # p_values are identical (MWU doesn't depend on mean type)
    # log2_fold_change differs because it's computed from the means
    j = df_a.join(df_g, on=["target","feature"], suffix="_g")
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
