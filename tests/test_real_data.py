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
             mean_calc="geometric",       # to match pdex's default
             min_feature_filter=1.0,
             epsilon=0.0)

    # Compare against CPU baseline restricted to chunk_0000's guides
    cpu = pl.read_parquet(REAL / "target_de.parquet")
    chunk_guides = set(combined.obs.loc[combined.obs["comparison"] != "ntc",
                                         "comparison"].unique())
    cpu = cpu.filter(pl.col("target").is_in(list(chunk_guides)))
    j = cpu.join(got, on=["target", "feature"], how="inner", suffix="_gpu")
    # Schema must include the join keys + at least p_value + log2_fold_change
    from scipy.stats import pearsonr
    for col in ("log2_fold_change", "p_value"):
        r = pearsonr(j[col].to_numpy(), j[f"{col}_gpu"].to_numpy()).statistic
        assert r > 0.999, f"{col}: r={r:.6f} below threshold"
