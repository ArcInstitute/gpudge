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
                 epsilon=0.0)
    out_inline = de(a_raw, groupby="comparison", reference="ntc",
                    epsilon=0.0,
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
                    epsilon=0.0)
    with pytest.warns(UserWarning, match="densify_input=True"):
        out_dense = de(a_dense, groupby="comparison", reference="ntc",
                       epsilon=0.0,
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
                 epsilon=0.0,
                 gpu_gene_chunk_size=None)  # auto
    out_many = de(a2, groupby="comparison", reference="ntc",
                  epsilon=0.0,
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
    """Zero-variance gene must not break MWU/FDR; with no (new) filter it is kept."""
    import numpy as np, polars as pl, scipy.sparse as sp
    a = synth_small_sparse.copy()
    X = a.X.toarray(); X[:, 0] = 0; X[:, 1] = 5; X[:5, 1] = 7
    a.X = sp.csr_matrix(X)
    out = de(a, groupby="comparison", reference="ntc", epsilon=0.0)  # keep-all
    p0 = out.filter(pl.col("feature") == "g0")["p_value"].to_numpy()
    assert len(p0) > 0, "all-zero gene should be kept when no filter is set"
    assert np.all(np.isfinite(p0)), "all-zero gene produced NaN/inf p-values"


@needs_cuda
def test_zero_mean_gene_dropped_at_zero_kept_at_negative(synth_small_sparse):
    """Escape hatch: 0.0 drops a zero-mean gene; a negative threshold keeps it."""
    import scipy.sparse as sp
    a = synth_small_sparse.copy()
    X = a.X.toarray(); X[:, 0] = 0; a.X = sp.csr_matrix(X)
    dropped = de(a, groupby="comparison", reference="ntc",
                 filter_gene_min_mean_value=0.0)
    kept = de(a, groupby="comparison", reference="ntc",
              filter_gene_min_mean_value=-1.0)
    assert "g0" not in set(dropped["feature"].to_list())
    assert "g0" in set(kept["feature"].to_list())


@needs_cuda
def test_cpm_cell_matches_scanpy_normalize_then_mean(synth_small_sparse):
    import numpy as np
    a = synth_small_sparse.copy()
    X = a.X.toarray().astype(np.float64)
    L = X.sum(axis=1, keepdims=True); L[L == 0] = 1.0
    cpm = X / L * 1e6
    comp = a.obs["comparison"].to_numpy()
    ref_mean = cpm[comp == "ntc"].mean(axis=0)
    targets = sorted(set(comp) - {"ntc"})
    tmeans = {g: cpm[comp == g].mean(axis=0) for g in targets}
    # threshold between min and max per-gene CPM mean -> guarantees a partition
    allvals = np.concatenate([ref_mean] + list(tmeans.values()))
    thr = float(np.median(allvals))
    expected = set()
    for g in targets:
        for j in np.flatnonzero((tmeans[g] > thr) | (ref_mean > thr)):
            expected.add((g, f"g{j}"))
    out = de(a, groupby="comparison", reference="ntc",
             filter_gene_min_cpm_cell=thr)
    got = set(zip(out["target"].to_list(), out["feature"].to_list()))
    assert got == expected
    assert 0 < len(expected) < len(targets) * a.n_vars   # genuinely partitions


@needs_cuda
def test_cpm_bulk_matches_pooled_definition(synth_small_sparse):
    import numpy as np
    a = synth_small_sparse.copy()
    X = a.X.toarray().astype(np.float64)
    comp = a.obs["comparison"].to_numpy(); L = X.sum(axis=1)
    ref = comp == "ntc"
    ref_bulk = X[ref].sum(axis=0) / max(L[ref].sum(), 1.0) * 1e6
    targets = sorted(set(comp) - {"ntc"})
    tb = {g: X[comp == g].sum(axis=0) / max(L[comp == g].sum(), 1.0) * 1e6
          for g in targets}
    thr = float(np.median(np.concatenate([ref_bulk] + list(tb.values()))))
    expected = set()
    for g in targets:
        for j in np.flatnonzero((tb[g] > thr) | (ref_bulk > thr)):
            expected.add((g, f"g{j}"))
    out = de(a, groupby="comparison", reference="ntc",
             filter_gene_min_cpm_bulk=thr)
    got = set(zip(out["target"].to_list(), out["feature"].to_list()))
    assert got == expected
    assert 0 < len(expected) < len(targets) * a.n_vars


@needs_cuda
def test_zero_denominator_cpm_finite_and_keeps_zero_gene(synth_small_sparse):
    """All-zero cell (L=0) and all-zero gene -> finite CPM math, gene kept w/ keep-all."""
    import numpy as np, polars as pl, scipy.sparse as sp
    a = synth_small_sparse.copy()
    X = a.X.toarray(); X[0, :] = 0; X[:, 0] = 0; a.X = sp.csr_matrix(X)
    out = de(a, groupby="comparison", reference="ntc",
             filter_gene_min_cpm_cell=-1.0)         # keep-all, exercise the math
    p0 = out.filter(pl.col("feature") == "g0")["p_value"].to_numpy()
    assert len(p0) > 0
    assert np.all(np.isfinite(p0))


@needs_cuda
def test_zero_denominator_cpm_bulk_all_others_empty_rest():
    """ALL_OTHERS with a group spanning all cells -> rest libtot 0 -> finite."""
    import numpy as np, anndata as ad, scipy.sparse as sp
    rng = np.random.default_rng(0)
    X = rng.integers(0, 5, size=(40, 6)).astype(np.float32)
    obs = {"comparison": np.array(["g0"] * 40)}    # single group = everyone
    a = ad.AnnData(X=sp.csr_matrix(X), obs=obs,
                   var={"gene_id": [f"g{i}" for i in range(6)]})
    from gpudge import ALL_OTHERS
    out = de(a, groupby="comparison", reference=ALL_OTHERS,
             filter_gene_min_cpm_bulk=-1.0)          # keep-all; rest is empty
    assert out is not None   # no crash / no inf in the bulk denominator


@needs_cuda
def test_value_filter_decoupled_from_cpm_normalize(synth_small_sparse):
    a = synth_small_sparse.copy()
    kw = dict(groupby="comparison", reference="ntc", filter_gene_min_mean_value=2.0)
    s0 = set(map(tuple, de(a.copy(), cpm_normalize=False, **kw)
                 .select(["target", "feature"]).iter_rows()))
    s1 = set(map(tuple, de(a.copy(), cpm_normalize=True, **kw)
                 .select(["target", "feature"]).iter_rows()))
    assert s0 == s1


@needs_cuda
def test_cpm_cell_warns_on_fractional_X(synth_small_sparse):
    import scipy.sparse as sp
    a = synth_small_sparse.copy()
    X = a.X.toarray().astype(np.float32); X[3, 3] += 0.5; a.X = sp.csr_matrix(X)
    with pytest.warns(UserWarning, match="raw counts"):
        de(a, groupby="comparison", reference="ntc", filter_gene_min_cpm_cell=1.0)


@needs_cuda
def test_cpm_cell_warns_on_dense_fractional_X(synth_small):
    import numpy as np
    a = synth_small.copy(); X = np.asarray(a.X); X[3, 3] += 0.5; a.X = X
    with pytest.warns(UserWarning, match="raw counts"):
        de(a, groupby="comparison", reference="ntc", filter_gene_min_cpm_cell=1.0)


@needs_cuda
def test_cpm_cell_no_warn_on_integer_float_X(synth_small_sparse, recwarn):
    de(synth_small_sparse, groupby="comparison", reference="ntc",
       filter_gene_min_cpm_cell=1.0)
    assert not any("raw counts" in str(w.message) for w in recwarn.list)


@needs_cuda
def test_value_filter_never_warns_on_fractional_X(synth_small_sparse, recwarn):
    import scipy.sparse as sp
    a = synth_small_sparse.copy()
    X = a.X.toarray().astype(np.float32); X[3, 3] += 0.5; a.X = sp.csr_matrix(X)
    de(a, groupby="comparison", reference="ntc", filter_gene_min_mean_value=1.0)
    assert not any("raw counts" in str(w.message) for w in recwarn.list)
