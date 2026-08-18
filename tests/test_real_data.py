# tests/test_real_data.py
from pathlib import Path
import pytest
import numpy as np
import polars as pl
import scanpy as sc
from gpudge import de
from conftest import needs_cuda

import os
REAL = Path(os.environ.get("GPUDGE_REAL_DATA_DIR", "/nonexistent-real-data"))
needs_data = pytest.mark.skipif(
    not REAL.exists(),
    reason="real validation data not mounted (set GPUDGE_REAL_DATA_DIR)")


@needs_cuda
@needs_data
def test_real_chunk_matches_cpu_pdex_baseline():
    chunk = sc.read_h5ad(REAL / "ad_gene_ds" / "chunk_0000.h5ad")
    ntc = sc.read_h5ad(REAL / "ad_gene_ds" / "ntc.h5ad")
    import anndata as ad
    sc.pp.normalize_total(chunk, target_sum=1e6)
    sc.pp.normalize_total(ntc, target_sum=1e6)
    combined = ad.concat([ntc, chunk], join="inner")
    combined.obs["comparison"] = np.where(
        combined.obs["target_guide"].astype(str).str.startswith("non-targeting"),
        "ntc",
        combined.obs["target_guide"].astype(str),
    )
    got = de(combined, groupby="comparison", reference="ntc",
             mean_calc="geometric", epsilon=0.0,
             filter_gene_min_mean_value=1.0)

    # Compare against CPU baseline restricted to chunk_0000's guides
    cpu = pl.read_parquet(REAL / "target_de.parquet")
    chunk_guides = set(combined.obs.loc[combined.obs["comparison"] != "ntc",
                                         "comparison"].unique())
    cpu = cpu.filter(pl.col("target").is_in(list(chunk_guides)))
    j = cpu.join(got, on=["target", "feature"], how="inner", suffix="_gpu")
    # COVERAGE first. The join had no height assertion, so losing 99.9% of the
    # rows still "passed" -- and coverage is exactly 1.0 on this data, so the
    # assertion is free. It also has to hold in BOTH directions: an inner join
    # is silent about rows either side failed to contribute.
    assert j.height == cpu.height == got.height, (
        f"join coverage: cpu={cpu.height} got={got.height} joined={j.height}")
    # ... and equal heights alone do not prove coverage: duplicate keys on one
    # side can make up for keys missing on the other. (codex review)
    keys = ["target", "feature"]
    assert cpu.select(keys).n_unique() == cpu.height
    assert got.select(keys).n_unique() == got.height
    assert cpu.join(got, on=keys, how="anti").height == 0
    assert got.join(cpu, on=keys, how="anti").height == 0

    # Schema must include the join keys + at least p_value + log2_fold_change
    from scipy.stats import pearsonr
    # Pearson is EXACTLY invariant to any affine transform, so on its own it
    # cannot see an added offset or a rescale at any threshold -- a log2FC
    # max-abs difference of 1.351 passed the old r > 0.999. Keep it as the
    # shape check, but tightened to the achieved value, and pin the actual
    # VALUES with absolute bounds (measured 2026-08-18 on chunk_0000 of CCL_1,
    # 22,049 rows, one sm_90 H100).
    for col, atol, r_min in (("log2_fold_change", 1e-3, 0.9999999),
                             ("p_value", 1e-9, 0.9999999)):
        a, b = j[col].to_numpy(), j[f"{col}_gpu"].to_numpy()
        r = pearsonr(a, b).statistic
        assert r > r_min, f"{col}: r={r:.12f} below threshold"
        # rtol=0: log2FC crosses zero, where any relative bound is meaningless
        # (measured max RELATIVE difference 0.71 against a max ABSOLUTE one of
        # 5.2e-5). Measured maxabs: log2FC 5.2e-5, p_value 1.6e-15 -- so both
        # bounds carry >= 10x headroom while killing an offset or a rescale.
        np.testing.assert_allclose(b, a, rtol=0, atol=atol,
                                   err_msg=f"{col} vs the CPU pdex baseline")
