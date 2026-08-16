# tests/test_lfc_integration.py
"""End-to-end de() with lfc_threshold, on CPU-sized fixtures."""
import numpy as np
import polars as pl
import pytest

from conftest import needs_cuda
from gpudge import ALL_OTHERS, de
from gpudge._lfc import lfc_column_names, normalize_lfc_spec


def _adata(seed=0, n_cells=240, n_genes=12, n_guides=3):
    import anndata as ad
    rng = np.random.default_rng(seed)
    X = rng.negative_binomial(2, 0.3, (n_cells, n_genes)).astype(np.float32)
    lab = np.array(["ntc"] * (n_cells // 2) +
                   [f"g{i % n_guides}" for i in range(n_cells - n_cells // 2)])
    rng.shuffle(lab)
    a = ad.AnnData(X=X, obs={"grp": lab})
    a.var_names = [f"f{i}" for i in range(n_genes)]
    return a


@needs_cuda
def test_base_columns_are_bit_identical_with_and_without_threshold():
    """THE RELEASE GATE.

    assert_frame_equal, NOT .tolist(): list comparison silently fails on NaN
    (nan != nan, and NaN p is a real output for degenerate groups) and pins
    neither dtype nor schema.

    check_exact=True is MANDATORY: polars defaults to check_exact=False with
    rel_tol=1e-5, abs_tol=1e-8 (verified, polars 1.41), so without it this
    "byte-identity gate" would accept changed base values.
    """
    from polars.testing import assert_frame_equal
    a = _adata()
    base = de(a, groupby="grp", reference="ntc")
    with_t = de(a, groupby="grp", reference="ntc", lfc_threshold=[0.0, 0.5])
    assert_frame_equal(with_t.select(base.columns), base,
                       check_dtypes=True, check_exact=True)


@needs_cuda
def test_directional_columns_present_ordered_and_grouped_per_combo():
    """Column layout is fully pinned: base columns first (unchanged order),
    then tau ASCENDING regardless of input order, down before up, and
    p/Ueffect/q grouped PER COMBO."""
    from gpudge._output import DEFAULT_OUTPUT_COLUMNS
    a = _adata()
    # deliberately supplied descending, and with `down` first
    df = de(a, groupby="grp", reference="ntc", lfc_threshold=[0.5, 0.25],
            lfc_threshold_alt=("down", "up"))
    combos = normalize_lfc_spec([0.5, 0.25], ("up", "down"))
    assert df.columns == list(DEFAULT_OUTPUT_COLUMNS) + lfc_column_names(combos)


@needs_cuda
def test_single_direction_computes_only_that_direction():
    a = _adata()
    df = de(a, groupby="grp", reference="ntc", lfc_threshold=0.5,
            lfc_threshold_alt=("up",))
    assert "tau=+0.5_p" in df.columns
    assert "tau=-0.5_p" not in df.columns


@needs_cuda
def test_sweep_equals_separate_calls():
    """One call with a grid == one call per tau, bit-for-bit."""
    from polars.testing import assert_frame_equal
    a = _adata()
    both = de(a, groupby="grp", reference="ntc", lfc_threshold=[0.25, 0.75])
    for t in (0.25, 0.75):
        one = de(a, groupby="grp", reference="ntc", lfc_threshold=[t])
        cols = [c for c in one.columns if c.startswith("tau=")]
        assert_frame_equal(both.select(cols), one.select(cols),
                           check_dtypes=True, check_exact=True)


@needs_cuda
def test_zero_row_result_keeps_full_directional_schema():
    """A fully-filtered result must have the SAME ordered schema as a
    populated one."""
    a = _adata()
    full = de(a, groupby="grp", reference="ntc", lfc_threshold=[0.5])
    none = de(a, groupby="grp", reference="ntc", lfc_threshold=[0.5],
              keep_genes=np.zeros(a.n_vars, dtype=bool))
    assert none.height == 0
    assert none.columns == full.columns
    assert none.schema == full.schema


@needs_cuda
def test_output_columns_can_select_directional_columns():
    a = _adata()
    # all THREE directional column kinds: p-value and Ueffect arrive via
    # the extra_columns channel, p_adj via the post-BH with_columns -- three
    # different routes, so selecting only two would leave one untested.
    df = de(a, groupby="grp", reference="ntc", lfc_threshold=[0.5],
            output_columns={"target": "t", "feature": "f",
                            "tau=+0.5_p": "p_up",
                            "tau=+0.5_Ueffect": "Ueffect_up",
                            "tau=+0.5_padj": "q_up"})
    assert df.columns == ["t", "f", "p_up", "Ueffect_up", "q_up"]
    with pytest.raises(KeyError):
        de(a, groupby="grp", reference="ntc", lfc_threshold=[0.5],
           output_columns={"tau=+9_nope": "x"})


@needs_cuda
def test_padj_matches_an_independent_bh_oracle():
    """A real per-group BH oracle, not just q >= p (which many wrong impls
    satisfy). Also asserts the base p_adj is unperturbed."""
    a = _adata()
    base = de(a, groupby="grp", reference="ntc")
    df = de(a, groupby="grp", reference="ntc", lfc_threshold=[0.5])
    np.testing.assert_array_equal(df["p_adj"].to_numpy(),
                                  base["p_adj"].to_numpy())
    col, qcol = "tau=+0.5_p", "tau=+0.5_padj"
    for tgt in df["target"].unique():
        sub = df.filter(pl.col("target") == tgt)
        p = sub[col].to_numpy()
        m = p.size
        order = np.argsort(p, kind="stable")
        unadj = m * p[order] / np.arange(1, m + 1)
        expected = np.empty(m)
        expected[order] = np.clip(
            np.minimum.accumulate(unadj[::-1])[::-1], 0.0, 1.0)
        np.testing.assert_allclose(sub[qcol].to_numpy(), expected, rtol=1e-12)


@needs_cuda
def test_rank_direction_can_oppose_log2fc_sign():
    """Regression fixture for spec 3.5 — pins the contradiction as intended
    behaviour, not a bug to be 'fixed' later."""
    import anndata as ad
    # target nine 0s + one 1000 vs ref ten 1s -> mean UP, rank DOWN
    tgt = np.array([[0.]] * 9 + [[1000.]], dtype=np.float32)
    ref = np.ones((10, 1), dtype=np.float32)
    X = np.vstack([tgt, ref])
    lab = np.array(["t"] * 10 + ["r"] * 10)
    a = ad.AnnData(X=X, obs={"grp": lab})
    a.var_names = ["gene0"]
    df = de(a, groupby="grp", reference="r", lfc_threshold=[0.0])
    row = df.filter(pl.col("target") == "t").row(0, named=True)
    assert row["log2_fold_change"] > 0            # mean says UP
    assert row["tau=-0_p"] < 0.01   # ranks say DOWN
    assert row["tau=+0_p"] > 0.99


@needs_cuda
def test_null_boundary_crossing():
    """Spec 5.5: two groups separated by exactly delta log2 units -- tau < delta
    is significant, tau > delta is not, with the crossing near tau = delta.

    Uses a clean location/scale shift (target = reference * 2**delta) so the
    rank test and the mean-ratio agree; spec 3.5 warns they need not in general.
    """
    import anndata as ad
    delta = 1.0
    rng = np.random.default_rng(4)
    # n = 1000, not 300: at n = 300 the seeded fixture's float64 scipy oracle
    # gives p = [0.048, 0.052, 0.039, 0.042, 0.050, 0.067] at tau = 0.9, so
    # `all(p < 0.05)` fails for two genes. At n = 1000 the max is ~0.0018.
    n, g = 1000, 6
    ref = rng.gamma(4.0, 25.0, (n, g)).astype(np.float32)
    tgt = (ref * np.float32(2.0 ** delta))[rng.permutation(n)]
    X = np.vstack([tgt, ref])
    lab = np.array(["t"] * n + ["r"] * n)
    a = ad.AnnData(X=X, obs={"grp": lab})
    a.var_names = [f"f{i}" for i in range(g)]
    df = de(a, groupby="grp", reference="r",
            lfc_threshold=[0.5, delta - 0.1, delta + 0.1, 2.0],
            lfc_threshold_alt=("up",))
    p_below = df["tau=+0.9_p"].to_numpy()
    p_above = df["tau=+1.1_p"].to_numpy()
    assert np.all(p_below < 0.05), p_below       # tau < delta -> significant
    assert np.all(p_above > 0.05), p_above       # tau > delta -> not
    # and monotone across the whole grid for this tie-free fixture
    grid = np.column_stack([df[f"tau=+{t:g}_p"].to_numpy()
                            for t in (0.5, 0.9, 1.1, 2.0)])
    assert np.all(np.diff(grid, axis=1) >= -1e-12)


def test_all_others_rejects_threshold():
    a = _adata()
    with pytest.raises(NotImplementedError, match="ALL_OTHERS"):
        de(a, groupby="grp", reference=ALL_OTHERS, lfc_threshold=0.5)


def test_invalid_threshold_rejected_before_gpu_work():
    a = _adata()
    with pytest.raises(ValueError, match="lfc_threshold"):
        de(a, groupby="grp", reference="ntc", lfc_threshold=-1.0)
