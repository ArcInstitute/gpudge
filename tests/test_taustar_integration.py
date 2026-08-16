"""End-to-end de() with tau_star, on CPU-sized fixtures (GPU-gated)."""

import numpy as np
import polars as pl
import pytest

from conftest import needs_cuda
from gpudge import ALL_OTHERS, de
from gpudge._taustar import taustar_column_names


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


def _assert_columns_bit_identical(got, want):
    """Every column of ``want`` reproduced EXACTLY in ``got``, NaN included.

    Not ``.to_list() == .to_list()``: two separate to_list() calls build
    distinct float objects, so a NaN compares unequal to itself and the
    assertion fails on a column that is in fact identical. np.testing's exact
    comparison treats NaN as equal to NaN, which is what bit-identity means
    here. Also checks the dtype -- a Float32/Float64 slip would otherwise
    survive an element-wise value comparison.
    """
    for col in want.columns:
        assert col in got.columns, col
        assert got.schema[col] == want.schema[col], col
        a, b = got[col], want[col]
        if b.dtype.is_numeric():
            np.testing.assert_array_equal(a.to_numpy(), b.to_numpy(),
                                          err_msg=col)
        else:
            assert a.to_list() == b.to_list(), col


@needs_cuda
def test_base_columns_are_bit_identical_with_and_without_tau_star():
    a = _adata()
    base = de(adata=a, groupby="grp", reference="ntc")
    with_ts = de(adata=a, groupby="grp", reference="ntc", tau_star=(0.5, 0.05))
    _assert_columns_bit_identical(with_ts, base)


@needs_cuda
def test_columns_are_named_and_ordered_canonically():
    a = _adata()
    out = de(adata=a, groupby="grp", reference="ntc", tau_star=[0.5, 0.05])
    names = taustar_column_names((0.05, 0.5))
    assert names == ["tau*_p0.05", "tau*_p0.5"]
    assert out.columns[-2:] == names
    for n in names:
        assert out.schema[n] == pl.Float64


@needs_cuda
def test_tau_star_composes_with_lfc_threshold():
    """The release gate for composition: turning tau_star ON must leave BOTH
    the base columns AND every existing tau=<±τ>_{p,Ueffect,padj} column
    bit-identical, with the tau* columns appended after them.

    Column presence and order alone would not catch it. When both features are
    active, group_chunk_stats runs mwu_one_group_lfc for the base + directional
    results and mwu_one_group_taustar separately -- so the base test is
    genuinely computed twice, and this pins that the pair the caller SEES comes
    from the lfc kernel unchanged.
    """
    a = _adata()
    lfc_only = de(adata=a, groupby="grp", reference="ntc",
                  lfc_threshold=[0.25])
    both = de(adata=a, groupby="grp", reference="ntc",
              lfc_threshold=[0.25], tau_star=(0.5,))

    assert "tau=+0.25_p" in both.columns
    assert both.columns[-1] == "tau*_p0.5"
    # tau* is APPENDED: the lfc layout is a strict prefix, unperturbed.
    assert both.columns[:-1] == lfc_only.columns
    _assert_columns_bit_identical(both, lfc_only)


@needs_cuda
def test_se_composes_with_lfc_threshold():
    """The SE flag on the OTHER kernel branch.

    When lfc_threshold and tau_star are both set, group_chunk_stats takes its
    first branch -- mwu_one_group_lfc for the base + directional results and
    mwu_one_group_taustar for tau* -- a different call site from the tau*-only
    branch every other test here exercises.

    Comparing the tau* VALUES against a run without lfc_threshold is what pins
    it, and names alone are not enough. With a single level, dropping
    `taustar_se=taustar_se` from that call returns ONE row into a four-row
    accumulator, and `taustar_chunk[:, g] = taustar_t` BROADCASTS a 1-row
    tensor silently (measured: torch and numpy both broadcast 1 -> 4 and both
    raise on 2 -> 5). The schema would be intact and every SE column would
    simply hold a copy of tau*_p0.5.

    Chunk width is pinned on both runs so the only difference is the presence
    of the lfc grid; results are chunk-invariant by contract, but leaving the
    auto sizer to pick two different widths would confound this comparison.
    """
    a = _adata()
    ts_kw = dict(groupby="grp", reference="ntc", tau_star=(0.5,),
                 tau_star_se=True, gpu_gene_chunk_size=4096,
                 oom_recovery=False)
    lfc_only = de(adata=a, groupby="grp", reference="ntc",
                  lfc_threshold=[0.25], gpu_gene_chunk_size=4096,
                  oom_recovery=False)
    se_only = de(adata=a, **ts_kw)
    both = de(adata=a, **ts_kw, lfc_threshold=[0.25])

    tail = ["tau*_p0.5", "tau*_lo_p0.025", "tau*_hi_p0.025", "tau*_se"]
    assert both.columns[-4:] == tail
    # The lfc layout stays a strict, bit-identical prefix.
    assert both.columns[:-4] == lfc_only.columns
    _assert_columns_bit_identical(both.select(lfc_only.columns), lfc_only)
    # ...and the tau* block is bit-identical to the lfc-free run. This is the
    # assertion a broadcast fails: under it lo == hi == se == tau*_p0.5.
    _assert_columns_bit_identical(both.select(tail), se_only.select(tail))


@needs_cuda
def test_sign_agrees_with_ueffect():
    """tau* cannot contradict the rank test -- that is the whole point.

    Asserted wherever tau* is not NUMERICALLY ZERO, and separately bounded
    where it is. Spec 3.6: at q = 0.5 the continuity correction puts the up and
    down levels at mu + 0.5 and mu - 0.5, so signed tau* has a HALF-PAIR
    DISCONTINUITY AT ZERO. A gene whose crossing lands on the delta = 0
    coincidence plateau (spec 3.4) therefore has |tau*| at the bisection's
    resolution floor, and which side of zero the final midpoint falls on is set
    by the bracket geometry, not by the effect direction -- ``Ueffect`` is
    measured AT delta = 0, ON the plateau, while tau* inverts the
    coincidence-free ``p~`` that is defined off it.

    The true crossing for such a gene is AT zero -- provably so. For an up gene
    with Ueffect > 0, U1(0) = A + E/2 > mu, and both sit on a half-integer
    lattice, so U1(0) >= mu + 0.5 = L_up; immediately left of zero
    U1(0-) = A + E >= U1(0) >= L_up, so the significant region reaches zero from
    the left. A genuinely negative crossing would need A + E < mu + 0.5 <=
    A + E/2, i.e. E < 0. The down case mirrors it: a genuinely positive
    crossing would need A > mu - 0.5 >= A + E/2. So a LARGE sign contradiction
    is arithmetically impossible in either direction, and every disagreement
    the suite can see is the returned value sitting on the wrong side of a
    crossing that is exactly zero.

    THE TOLERANCE IS THE BISECTION RESIDUAL, NOT THE PLATEAU WIDTH. The float64
    equality plateau around an exact tie is O(1e-16) in delta; the observed
    magnitude is dominated by the finite bisection, which leaves
    (hi0 - lo0) / 2**(iters + 1). A fixed 1e-4 threshold at the default 20
    iterations is therefore NOT universal -- a valid, finite, non-negative
    fixture with a wide bracket (target [1.0], reference spanning the float32
    range) returns -1.32e-4 for a crossing that is exactly zero. Hence
    tau_star_iters=40 here: over the whole positive-finite float32 domain each
    array spans < 277 log2 units, so the initial bracket is < 554, bounding the
    40-step midpoint error at 554 / 2**41 ~ 2.5e-10. That makes the 1e-8
    threshold universal for the supported domain with ~40x of margin, while
    still sitting ~1e5 below any meaningful shift. Concluding the true crossing
    is EXACTLY zero (not merely non-contradictory) additionally uses float32
    breakpoint discreteness: the smallest nonzero log2 ratio near 1 is ~8.6e-8,
    far above the residual, so there is no nearby breakpoint to hide behind.
    """
    a = _adata(seed=4)
    out = de(adata=a, groupby="grp", reference="ntc", tau_star=(0.5,),
             tau_star_iters=40)
    ts = out["tau*_p0.5"].to_numpy()
    ue = out["Ueffect"].to_numpy()
    finite = np.isfinite(ts) & (np.abs(ue) > 1e-12)

    # Any sign disagreement must be zero to within the bisection residual. A
    # larger one contradicts the rank test and the whole premise of the feature.
    disagree = finite & (np.sign(ts) != np.sign(ue))
    assert np.all(np.abs(ts[disagree]) <= 1e-8), ts[disagree]

    # And there must be meaningful agreement left to speak of, or the bound
    # above would pass vacuously on an all-zero column.
    meaningful = finite & (np.abs(ts) > 1e-8)
    assert meaningful.sum() > 0
    assert np.all(np.sign(ts[meaningful]) == np.sign(ue[meaningful]))


@needs_cuda
def test_the_bound_sits_between_zero_and_the_point_estimate():
    """DIRECTION-AWARE: for an up gene the 0.05 bound is BELOW the 0.5 point
    estimate, for a down gene ABOVE it. Comparing magnitudes instead is wrong
    whenever the interval straddles zero -- estimate +0.1 with a bound of -0.3
    is a legal result whose magnitudes are ordered the other way."""
    a = _adata(seed=6)
    out = de(adata=a, groupby="grp", reference="ntc", tau_star=(0.5, 0.05))
    lo = out["tau*_p0.05"].to_numpy()
    hi = out["tau*_p0.5"].to_numpy()
    ue = out["Ueffect"].to_numpy()
    ok = np.isfinite(lo) & np.isfinite(hi)
    up = ok & (ue >= 0)
    down = ok & (ue < 0)
    assert np.all(lo[up] <= hi[up] + 1e-9)
    assert np.all(lo[down] >= hi[down] - 1e-9)


@needs_cuda
def test_all_others_is_rejected():
    # NotImplementedError, matching the lfc_threshold guard -- asserting
    # ValueError here would pin the WRONG contract.
    a = _adata()
    with pytest.raises(NotImplementedError, match="tau_star"):
        de(adata=a, groupby="grp", reference=ALL_OTHERS, tau_star=(0.5,))


@needs_cuda
def test_zero_row_result_keeps_the_tau_star_schema():
    a = _adata()
    out = de(adata=a, groupby="grp", reference="ntc", tau_star=(0.5,),
             filter_gene_min_mean_value=1e12)   # filters everything out
    assert out.height == 0
    assert "tau*_p0.5" in out.columns
    assert out.schema["tau*_p0.5"] == pl.Float64


@needs_cuda
def test_output_columns_projection_can_select_tau_star():
    a = _adata()
    out = de(adata=a, groupby="grp", reference="ntc", tau_star=(0.5,),
             output_columns={
                 "target": "target",
                 "feature": "feature",
                 "tau*_p0.5": "tau*_p0.5",
             })
    assert out.columns == ["target", "feature", "tau*_p0.5"]


def _control_pair(seed=0, n_genes=12, n_target=150, n_ctrl=200, n_guides=3):
    import anndata as ad
    rng = np.random.default_rng(seed)
    xt = rng.negative_binomial(2, 0.3, (n_target, n_genes)).astype(np.float32)
    xc = rng.negative_binomial(2, 0.3, (n_ctrl, n_genes)).astype(np.float32)
    lab = np.array([f"g{i % n_guides}" for i in range(n_target)])
    t = ad.AnnData(X=xt, obs={"grp": lab})
    c = ad.AnnData(X=xc, obs={"grp": np.array(["ctrl"] * n_ctrl)})
    names = [f"f{i}" for i in range(n_genes)]
    t.var_names = names
    c.var_names = names
    return t, c


@needs_cuda
def test_external_reference_path_emits_tau_star():
    t, c = _control_pair()
    out = de(adata=t, groupby="grp", reference=c, tau_star=(0.5, 0.05))
    assert out.columns[-2:] == ["tau*_p0.05", "tau*_p0.5"]
    assert np.isfinite(out["tau*_p0.5"].to_numpy()).any()


@needs_cuda
def test_external_reference_base_columns_are_bit_identical():
    t, c = _control_pair(seed=2)
    base = de(adata=t, groupby="grp", reference=c)
    with_ts = de(adata=t, groupby="grp", reference=c, tau_star=(0.5,))
    _assert_columns_bit_identical(with_ts, base)


@needs_cuda
def test_external_reference_zero_row_result_keeps_the_schema():
    t, c = _control_pair(seed=3)
    out = de(adata=t, groupby="grp", reference=c, tau_star=(0.5,),
             filter_gene_min_mean_value=1e12)
    assert out.height == 0
    assert out.schema["tau*_p0.5"] == pl.Float64


@needs_cuda
def test_external_reference_matches_the_literal_reference_path():
    """Cross-path parity (spec 5f): the same comparison computed through the
    literal-reference path and the external-reference path must agree."""
    import anndata as ad
    rng = np.random.default_rng(9)
    n_genes = 10
    xt = rng.negative_binomial(2, 0.3, (120, n_genes)).astype(np.float32)
    xc = rng.negative_binomial(2, 0.3, (180, n_genes)).astype(np.float32)
    names = [f"f{i}" for i in range(n_genes)]
    lab = np.array([f"g{i % 2}" for i in range(120)])

    t = ad.AnnData(X=xt, obs={"grp": lab})
    t.var_names = names
    c = ad.AnnData(X=xc, obs={"grp": np.array(["ctrl"] * 180)})
    c.var_names = names
    ext = de(adata=t, groupby="grp", reference=c, tau_star=(0.5, 0.05))

    merged = ad.concat([t, c])
    merged.var_names = names
    lit = de(adata=merged, groupby="grp", reference="ctrl", tau_star=(0.5, 0.05))

    key = ["target", "feature"]
    a = ext.sort(key)
    b = lit.filter(pl.col("target") != "ctrl").sort(key)
    for col in ("tau*_p0.05", "tau*_p0.5"):
        np.testing.assert_array_equal(a[col].to_numpy(), b[col].to_numpy())


# --- tau_star_se ----------------------------------------------------------

@needs_cuda
def test_se_emits_three_columns_after_the_level_columns():
    a = _adata()
    df = de(a, groupby="grp", reference="ntc", tau_star=[0.05],
            tau_star_se=True)
    assert df.columns[-5:] == ["tau*_p0.05", "tau*_p0.5",
                               "tau*_lo_p0.025", "tau*_hi_p0.025", "tau*_se"]


@needs_cuda
def test_se_forces_the_point_estimate_column_even_when_not_requested():
    a = _adata()
    df = de(a, groupby="grp", reference="ntc", tau_star=0.05,
            tau_star_se=True)
    assert "tau*_p0.5" in df.columns


@needs_cuda
@pytest.mark.parametrize("levels", [
    [0.05, 0.5],     # 0.5 already present -- no insertion
    [0.05],          # insertion AFTER an existing level
    [0.9],           # insertion BEFORE an existing level
    0.5,             # scalar spelling
])
def test_se_leaves_every_pre_existing_column_bit_identical(levels):
    """The release gate. Stated precisely: every column present in the se=False
    result appears in the se=True result, bit-identical. The schemas are
    INTENTIONALLY different -- se=True also carries the three SE columns and,
    when 0.5 was not requested, tau*_p0.5.

    Parameterized over the insertion path: a [0.05, 0.5] grid alone never
    exercises automatic 0.5 insertion at all.
    """
    a = _adata()
    kw = dict(groupby="grp", reference="ntc", tau_star=levels)
    off = de(a, **kw)
    on = de(a, **kw, tau_star_se=True)
    assert set(off.columns) <= set(on.columns)
    _assert_columns_bit_identical(on.select(off.columns), off)


@needs_cuda
@pytest.mark.parametrize("chunk", [4, 7, 4096])
def test_se_results_are_invariant_to_the_gene_chunk_width(chunk):
    """The wider accumulators can change what the auto sizer picks, so a
    12-gene fixture where both runs happen to land on the same chunk proves
    nothing about structural identity. Pin explicit, DIFFERENT chunk widths and
    require identical output -- results are chunk-invariant by contract, so
    this doubles as a direct test of that contract."""
    a = _adata()
    kw = dict(groupby="grp", reference="ntc", tau_star=[0.05, 0.5],
              tau_star_se=True, oom_recovery=False)
    ref = de(a, **kw, gpu_gene_chunk_size=4096)
    got = de(a, **kw, gpu_gene_chunk_size=chunk)
    _assert_columns_bit_identical(got, ref)


@needs_cuda
def test_se_interval_brackets_the_point_estimate_end_to_end():
    a = _adata()
    df = de(a, groupby="grp", reference="ntc", tau_star=0.5,
            tau_star_se=True)
    lo = df["tau*_lo_p0.025"].to_numpy()
    hi = df["tau*_hi_p0.025"].to_numpy()
    est = df["tau*_p0.5"].to_numpy()
    good = np.isfinite(lo) & np.isfinite(hi) & np.isfinite(est)
    assert good.any()
    assert np.all(lo[good] <= est[good]) and np.all(est[good] <= hi[good])
    se = df["tau*_se"].to_numpy()
    assert np.all(se[~np.isnan(se)] >= 0.0)


@needs_cuda
def test_se_columns_are_valid_output_columns_keys():
    a = _adata()
    df = de(a, groupby="grp", reference="ntc", tau_star=0.5, tau_star_se=True,
            output_columns={"target": "t", "feature": "f",
                            "tau*_se": "shift_se",
                            "tau*_lo_p0.025": "shift_lo"})
    assert df.columns == ["t", "f", "shift_se", "shift_lo"]


def test_se_without_tau_star_raises_before_any_gpu_work():
    """No needs_cuda: validation must fire at entry, on any machine."""
    a = _adata()
    with pytest.raises(ValueError, match="tau_star_se=True requires tau_star"):
        de(a, groupby="grp", reference="ntc", tau_star_se=True)


def test_se_rejects_a_non_bool_before_any_gpu_work():
    a = _adata()
    with pytest.raises(ValueError, match="tau_star_se must be True or False"):
        de(a, groupby="grp", reference="ntc", tau_star=0.5, tau_star_se=1)


def test_se_inherits_the_all_others_guard():
    a = _adata()
    with pytest.raises(NotImplementedError, match="tau_star is not supported"):
        de(a, groupby="grp", reference=ALL_OTHERS, tau_star=0.5,
           tau_star_se=True)


@needs_cuda
def test_se_matches_across_the_in_memory_and_external_reference_paths():
    """Mode-1 vs inmem_external_ref_de: bit-identical on the three new columns,
    same gate as the existing tau* parity tests (#79 drift).

    Not optional coverage -- inmem_external_ref_de is the layer whose own sizer
    call this plan originally missed, and it is the path dge_robust uses."""
    import anndata as ad
    a = _adata()
    is_ntc = (a.obs["grp"] == "ntc").to_numpy()
    tgt = a[~is_ntc].copy()
    ref = ad.AnnData(X=a[is_ntc].X.copy(),
                     obs={"grp": ["ntc"] * int(is_ntc.sum())})
    ref.var_names = a.var_names
    kw = dict(tau_star=0.5, tau_star_se=True)
    mode1 = de(a, groupby="grp", reference="ntc", **kw)
    mode2 = de(tgt, groupby="grp", reference=ref, **kw)
    j = mode1.join(mode2, on=["target", "feature"], how="inner", suffix="_m2")
    assert j.height > 0
    for c in ("tau*_lo_p0.025", "tau*_hi_p0.025", "tau*_se"):
        np.testing.assert_array_equal(j[c].to_numpy(), j[f"{c}_m2"].to_numpy())


def _spy_sizer(monkeypatch, modname, attr):
    """Record the n_levels ONE named sizer binding receives, then delegate.

    One binding per test, and the caller names it. The three bindings are
    DISTINCT module attributes -- `gpudge` and `gpudge._refpool` each did
    `from ._stream import _auto_gene_chunk_size`, so patching one does not
    affect the other -- and recording only `attr` would make the Mode-1 and
    core bindings indistinguishable in the log.
    """
    import importlib
    mod = importlib.import_module(modname)
    real = getattr(mod, attr)
    seen = []

    def spy(*args, **kwargs):
        seen.append(kwargs.get("n_levels"))
        return real(*args, **kwargs)

    monkeypatch.setattr(mod, attr, spy)
    return seen


@needs_cuda
def test_mode1_sizer_receives_the_row_count(monkeypatch):
    """tau_star=[0.05] + se -> levels {0.05, 0.5} + 3 SE rows = 5.

    Pins WHAT the driver passes. The monotonic-arithmetic test in
    test_stream.py cannot catch a missed CALL SITE, and one was in fact missed
    in the first draft of this plan.
    """
    seen = _spy_sizer(monkeypatch, "gpudge", "_auto_gene_chunk_size")
    de(_adata(), groupby="grp", reference="ntc", tau_star=[0.05],
       tau_star_se=True, gpu_gene_chunk_size=None)
    assert seen == [5], seen


@needs_cuda
def test_external_reference_sizer_receives_the_row_count(monkeypatch):
    """The path that was missed: inmem_external_ref_de has its OWN
    _auto_gene_chunk_size_inmem call. Without this the external-reference path
    would size from len(levels)=2 while the core writes 5 rows."""
    import anndata as ad
    a = _adata()
    is_ntc = (a.obs["grp"] == "ntc").to_numpy()
    tgt = a[~is_ntc].copy()
    ref = ad.AnnData(X=a[is_ntc].X.copy(),
                     obs={"grp": ["ntc"] * int(is_ntc.sum())})
    ref.var_names = a.var_names
    seen = _spy_sizer(monkeypatch, "gpudge._refpool",
                      "_auto_gene_chunk_size_inmem")
    de(tgt, groupby="grp", reference=ref, tau_star=[0.05], tau_star_se=True,
       gpu_gene_chunk_size=None)
    assert seen == [5], seen
