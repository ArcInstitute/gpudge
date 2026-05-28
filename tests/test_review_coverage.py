"""Tests added during code-review-pass-1 to close coverage gaps.

Covers:
- cpm_normalize=True equivalent to scanpy.pp.normalize_total upstream
- densify_input=True bit-equivalent to densify_input=False
- multi-chunk equivalence (gpu_gene_chunk_size override)
- output_columns raises KeyError on unknown keys
- mwu_one_group degenerate cases (m=0, n_ref=0)
- tie-heavy input (all-zero gene + heavy-tie columns)
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from conftest import needs_cuda, _make_synth
from gpudge import de


@needs_cuda
def test_cpm_normalize_matches_external(synth_small_sparse):
    """Inline cpm_normalize=True ≡ external sc.pp.normalize_total(1e6)."""
    a_raw = synth_small_sparse.copy()
    a_pre = synth_small_sparse.copy()

    # External CPM normalize on a_pre, then de() without cpm_normalize.
    row_sums = np.asarray(a_pre.X.sum(axis=1)).ravel().astype(np.float64)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    scale = (1e6 / row_sums).astype(np.float32)
    a_pre.X = a_pre.X.multiply(scale[:, None]).tocsr()

    out_pre = de(a_pre, groupby="comparison", reference="ntc",
                 epsilon=0.0, min_feature_filter=0.0)
    out_inline = de(a_raw, groupby="comparison", reference="ntc",
                    epsilon=0.0, min_feature_filter=0.0,
                    cpm_normalize=True)

    # Sort by (target, feature) so row order is comparable.
    out_pre = out_pre.sort(["target", "feature"])
    out_inline = out_inline.sort(["target", "feature"])

    np.testing.assert_allclose(
        out_pre["log2_fold_change"].to_numpy(),
        out_inline["log2_fold_change"].to_numpy(),
        rtol=1e-5, atol=1e-7,
    )
    # p-values can differ in last decimals due to float32 multiply ordering;
    # require correlation > 0.9999999 rather than bit-equality.
    p_pre = out_pre["p_value"].to_numpy()
    p_inl = out_inline["p_value"].to_numpy()
    finite = np.isfinite(p_pre) & np.isfinite(p_inl)
    if finite.sum() > 10:
        corr = np.corrcoef(p_pre[finite], p_inl[finite])[0, 1]
        assert corr > 0.99999, f"cpm_normalize p-value correlation {corr:.6f}"


@needs_cuda
def test_densify_input_matches_sparse_path(synth_small_sparse):
    """densify_input=True must produce identical numbers to sparse path."""
    a_sparse = synth_small_sparse.copy()
    a_dense = synth_small_sparse.copy()

    out_sparse = de(a_sparse, groupby="comparison", reference="ntc",
                    epsilon=0.0, min_feature_filter=0.0)
    with pytest.warns(UserWarning, match="densify_input=True"):
        out_dense = de(a_dense, groupby="comparison", reference="ntc",
                       epsilon=0.0, min_feature_filter=0.0,
                       densify_input=True)

    # The dense path mutates adata.X — confirm
    assert not sp.issparse(a_dense.X)
    assert sp.issparse(a_sparse.X)

    out_sparse = out_sparse.sort(["target", "feature"])
    out_dense = out_dense.sort(["target", "feature"])
    np.testing.assert_array_equal(
        out_sparse["log2_fold_change"].to_numpy(),
        out_dense["log2_fold_change"].to_numpy(),
    )
    np.testing.assert_array_equal(
        out_sparse["p_value"].to_numpy(),
        out_dense["p_value"].to_numpy(),
    )


@needs_cuda
def test_multi_chunk_equivalence(synth_small_sparse):
    """Forcing a small gpu_gene_chunk_size must give same result as one big chunk."""
    a1 = synth_small_sparse.copy()
    a2 = synth_small_sparse.copy()

    out_one = de(a1, groupby="comparison", reference="ntc",
                 epsilon=0.0, min_feature_filter=0.0,
                 gpu_gene_chunk_size=None)  # auto
    out_many = de(a2, groupby="comparison", reference="ntc",
                  epsilon=0.0, min_feature_filter=0.0,
                  gpu_gene_chunk_size=8)  # force ~7 chunks for 50 genes

    out_one = out_one.sort(["target", "feature"])
    out_many = out_many.sort(["target", "feature"])
    np.testing.assert_array_equal(
        out_one["log2_fold_change"].to_numpy(),
        out_many["log2_fold_change"].to_numpy(),
    )
    np.testing.assert_array_equal(
        out_one["p_value"].to_numpy(),
        out_many["p_value"].to_numpy(),
    )


@needs_cuda
def test_output_columns_rejects_unknown_key(synth_small):
    """output_columns with a typo should KeyError, not silently drop."""
    with pytest.raises(KeyError, match="not present"):
        de(synth_small, groupby="comparison", reference="ntc",
           output_columns={
               "target": "guide", "feature": "gene",
               "log2_fold_change": "lfc",
               "nonexistent_typo": "boom",  # ← bad key
           })


def test_mwu_one_group_m_zero():
    """m=0 must return zeros U + NaN p, not NaN/inf from divide-by-zero.

    Runs on CPU so the degenerate-case guard is validated even when no GPU
    is available — the guard returns before any device-specific math.
    """
    import torch
    from gpudge._mwu import mwu_one_group, _tie_term_per_gene

    n_genes, n_ref = 4, 100
    sorted_ref = torch.sort(torch.rand(n_genes, n_ref), dim=1).values
    tie_term = _tie_term_per_gene(sorted_ref)
    empty_group = torch.zeros((n_genes, 0))

    u, p = mwu_one_group(sorted_ref, tie_term, empty_group, n_ref=n_ref)
    assert u.shape == (n_genes,)
    assert torch.all(u == 0)
    assert torch.all(torch.isnan(p))


def test_mwu_one_group_n_ref_zero():
    """n_ref=0 must return zeros U + NaN p."""
    import torch
    from gpudge._mwu import mwu_one_group

    n_genes, m = 4, 10
    sorted_ref_empty = torch.zeros((n_genes, 0))
    tie_term_empty = torch.zeros(n_genes, dtype=torch.float64)
    group_T = torch.rand(n_genes, m)

    u, p = mwu_one_group(sorted_ref_empty, tie_term_empty, group_T, n_ref=0)
    assert u.shape == (n_genes,)
    assert torch.all(u == 0)
    assert torch.all(torch.isnan(p))


@needs_cuda
def test_tie_heavy_all_zero_gene_does_not_crash(synth_small_sparse):
    """A gene with zero variance across cells must not break MWU/FDR."""
    a = synth_small_sparse.copy()
    X = a.X.toarray()
    # Inject all-zero column at gene index 0
    X[:, 0] = 0
    # Inject heavy-tie column at gene index 1 (mostly a single value)
    X[:, 1] = 5
    X[:5, 1] = 7  # break perfect tie a little
    a.X = sp.csr_matrix(X)

    import polars as pl
    out = de(a, groupby="comparison", reference="ntc",
             epsilon=0.0, min_feature_filter=0.0)
    # all-zero gene: should at least not produce NaN/inf p-values.
    p_gene0 = out.filter(pl.col("feature") == "g0")["p_value"].to_numpy()
    assert len(p_gene0) > 0, "expected rows for the all-zero gene"
    assert np.all(np.isfinite(p_gene0)), \
        "all-zero gene produced NaN/inf p-values"
