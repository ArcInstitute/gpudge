"""Gates on the quickstart: the committed dataset, the runnable example, and every number
``docs/tutorial.md`` shows a reader.

The dataset is small enough (600 x 1000) to recompute the whole result with SciPy on the
CPU in about a second. So the tutorial's transcript is checked against an **independent
oracle** rather than against gpudge's own output -- which means these gates run in CI,
without a GPU, and would catch both a stale transcript and a regression in `de()`.

That matters more than it sounds: this project's CI-coverage figures went stale three
times in three days because nothing asserted them. A tutorial is worse, because a reader
takes its numbers on trust.
"""
from __future__ import annotations

import ast
import hashlib
import re
import runpy
import sys
from types import SimpleNamespace
from pathlib import Path

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp
from scipy.stats import mannwhitneyu

from conftest import needs_cuda

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "docs" / "data" / "H1-VCC-2025-training.h5ad"
TUTORIAL = REPO / "docs" / "tutorial.md"
QUICKSTART = REPO / "examples" / "quickstart.py"

# The recipe in docs/make_tutorial_data.py. Changing the subset must break these.
N_CELLS, N_GENES = 600, 1000
GROUPS = {"MED12", "SRC", "STAT1", "TET1", "TMSB4X", "non-targeting"}
CELLS_PER_GROUP = 100
CONTROL = "non-targeting"
ON_TARGET = "TMSB4X"
EPSILON = 1e-9          # de()'s default


@pytest.fixture(scope="module")
def adata() -> ad.AnnData:
    return ad.read_h5ad(DATA)


@pytest.fixture(scope="module")
def quickstart_module() -> dict:
    """The example's globals, so tests compare against its ACTUAL constants."""
    return runpy.run_path(str(QUICKSTART), run_name="quickstart_under_test")


def _oracle_rows(X: np.ndarray, labels: np.ndarray, genes: list[str]) -> list[tuple]:
    """The oracle proper, taking already-normalized CPM values so its NaN-p-value branch
    is reachable from synthetic input (see `test_oracle_emits_every_gene_when_a_p_value_
    is_nan` and `test_oracle_sorts_nan_p_adj_last`).
    """
    ref = X[labels == CONTROL]
    rows: list[tuple] = []
    for group in sorted(set(labels) - {CONTROL}):
        target = X[labels == group]
        p = np.asarray(mannwhitneyu(target, ref, alternative="two-sided",
                                    method="asymptotic", use_continuity=True,
                                    axis=0).pvalue)
        lfc = np.log2((target.mean(0) + EPSILON) / (ref.mean(0) + EPSILON))
        # BH over the non-NaN p-values only. np.argsort puts NaN last, so reversing for
        # the running minimum would put it first and propagate NaN across every gene --
        # a silent all-NaN oracle rather than a loud failure. MEASURED on the committed
        # subset: 0 NaN p-values and 0 zero-variance genes across all five groups. That
        # is a measurement, not a consequence of the top-1000 cut -- ranking genes by
        # total count does not preclude a constant one -- so the branch is insurance for
        # a future re-cut, and the tests below drive it. (Gemini + codex, PR #129.)
        adj = np.full(p.size, np.nan)
        finite = ~np.isnan(p)
        if finite.any():
            pf = p[finite]
            m = pf.size            # gpudge's denominator: _fdr.py drops NaN rows, then
                                   # uses `m = e - s`, the group's NON-NaN count. Using
                                   # p.size here would silently diverge from de() the
                                   # moment any gene produced NaN -- the exact case this
                                   # branch exists for. (Gemini review, PR #129.)
            order = np.argsort(pf, kind="stable")
            sub = np.empty(m)
            sub[order] = np.minimum.accumulate(
                (pf[order] * m / np.arange(1, m + 1))[::-1])[::-1].clip(max=1.0)
            adj[finite] = sub
        # range(p.size), NOT range(m): `m` is the BH denominator defined only inside the
        # branch above, so with any NaN p-value this would drop the trailing genes, and
        # with an all-NaN group it would either raise UnboundLocalError or silently reuse
        # the PREVIOUS group's count. Every gene gets a row; a degenerate one carries a
        # NaN p_adj. (Gemini + Copilot review, PR #129.)
        rows.extend((group, genes[j], lfc[j], adj[j]) for j in range(p.size))

    # NaN sorts last, explicitly: NaN compares False against everything, so a bare r[3]
    # key would leave a degenerate gene wherever the sort happened to drop it -- possibly
    # ahead of the real hits the tests compare. (Copilot review, PR #129.)
    rows.sort(key=lambda r: (np.isnan(r[3]), 0.0 if np.isnan(r[3]) else r[3]))
    return rows


@pytest.fixture(scope="module")
def oracle(adata) -> list[tuple]:
    """Recompute the quickstart's result with SciPy: CPM-normalize, two-sided asymptotic
    Mann-Whitney with continuity correction, BH within each perturbation. Returns rows of
    (target, feature, log2fc, p_adj) sorted by p_adj.

    Deliberately NOT written with gpudge -- an oracle that shares the implementation
    cannot catch the implementation drifting.
    """
    X = adata.X.toarray().astype(np.float64)
    X = X * (1e6 / X.sum(axis=1, keepdims=True))
    return _oracle_rows(X, adata.obs["target_gene"].to_numpy().astype(str),
                        [str(v) for v in adata.var_names])


def _de_kwargs(source: str) -> dict:
    """The keyword arguments of the single `de(...)` call in `source`, as real values."""
    calls = [n for n in ast.walk(ast.parse(source))
             if isinstance(n, ast.Call)
             and (getattr(n.func, "attr", None) == "de" or getattr(n.func, "id", None) == "de")]
    assert len(calls) == 1, f"expected exactly one de() call, found {len(calls)}"
    out = {}
    for kw in calls[0].keywords:
        assert kw.arg is not None, (
            "de() is called with **kwargs; this gate compares the tutorial's literal "
            "arguments and cannot see through unpacking")
        try:
            out[kw.arg] = ast.literal_eval(kw.value)
        except ValueError:
            out[kw.arg] = ast.unparse(kw.value)   # a name, e.g. GROUPBY
    return out


def _tutorial_de_snippet() -> str:
    """The ```python block in the tutorial that calls de()."""
    for block in re.findall(r"```python\n(.*?)```", TUTORIAL.read_text(), re.S):
        if re.search(r"\bde\(", block):
            return block
    raise AssertionError("the tutorial no longer shows a de() call")


def _tutorial_top_rows() -> list[tuple[str, str, str, str]]:
    """Parse the top-hits transcript out of docs/tutorial.md."""
    doc = TUTORIAL.read_text()
    block = re.search(
        r"```\n\s*target\s+gene\s+log2FC\s+p_adj\n(.*?)```", doc, re.S)
    assert block, "the tutorial no longer shows a top-hits table"
    rows = []
    for line in block.group(1).strip().splitlines():
        m = re.match(r"\s*(\S+)\s+(\S+)\s+(-?[\d.]+)\s+([\d.]+(?:e[+-]\d+)?)\s*$", line)
        assert m, f"unparseable transcript row: {line!r}"
        rows.append((m[1], m[2], m[3], m[4]))     # keep as rendered
    return rows


# --------------------------------------------------------------------- the data

def test_the_tutorial_data_ships_with_the_repo():
    """A reader must not have to download anything, and the file must stay small enough
    to belong in git."""
    assert DATA.is_file(), f"{DATA} is committed and the tutorial depends on it"
    assert DATA.stat().st_size < 8_000_000, "re-cut the subset rather than let it grow"


def test_the_committed_data_matches_its_documented_provenance():
    """docs/data/README.md records where this file came from, its CC0 licence and its
    checksum. Swapping the file without updating that record would leave a licence and
    provenance statement describing something else -- so the checksum is a gate, not a
    note."""
    notice = (DATA.parent / "README.md").read_text()
    digest = hashlib.sha256(DATA.read_bytes()).hexdigest()
    assert digest in notice, (
        f"docs/data/README.md does not record this file's sha256 ({digest}); "
        "update the notice if the subset was deliberately re-cut")
    assert str(DATA.stat().st_size) in notice.replace(",", ""), "size not recorded"
    assert "CC0 1.0" in notice, "the data licence must stay stated"


def test_tutorial_data_has_the_documented_shape(adata):
    assert adata.shape == (N_CELLS, N_GENES)
    counts = adata.obs["target_gene"].value_counts()
    assert set(counts.index) == GROUPS
    assert set(counts.to_numpy().tolist()) == {CELLS_PER_GROUP}


def test_tutorial_data_is_still_raw_counts(adata):
    """`cpm_normalize=True` assumes raw counts. A normalized replacement would make the
    example double-normalize and teach the exact mistake the tutorial warns about.

    Integrality alone is too weak -- rounded CPM is integral too -- so this also pins the
    magnitude: CPM rows sum to 1e6 by construction, raw ones do not."""
    X = adata.X
    assert sp.issparse(X) and X.format == "csr", "the example advertises sparse CSR"
    data = X.data
    assert np.all(np.abs(data - np.rint(data)) < 1e-6), "values are not integral"
    row_sums = np.asarray(X.sum(axis=1)).ravel()
    assert row_sums.min() > 1_000, "library sizes look too small to be raw counts"
    assert not np.allclose(row_sums, 1e6), "rows sum to 1e6 -- this is CPM, not counts"
    assert row_sums.std() > 1.0, "identical library sizes mean this was normalized"


def test_on_target_knockdown_is_present_in_the_data(adata):
    """Assert the on-target effect is present in the raw counts, so a failure in the
    result tests points at the analysis rather than at the subset having drifted. Raw
    means, not the CPM the tutorial analyses -- a coarser check, deliberately: its job is
    to characterise the DATA, and the oracle tests cover the analysed values."""
    assert ON_TARGET in set(adata.var_names)
    j = list(adata.var_names).index(ON_TARGET)
    labels = adata.obs["target_gene"].to_numpy()
    col = adata.X[:, j].toarray().ravel()
    knocked = col[labels == ON_TARGET].mean()
    control = col[labels == CONTROL].mean()
    assert knocked < 0.2 * control, f"not knocked down: {knocked:.2f} vs {control:.2f}"


# ------------------------------------------------- the oracle's own NaN-p-value path

# The committed subset yields no NaN p-value, so the oracle's NaN branch never runs
# against real data -- which is how the `range(m)` truncation it now guards against got
# written in the first place. These tests drive that branch directly.
#
# They inject the p-values rather than provoking them, deliberately: whether a
# zero-variance gene comes back as NaN or as 1.0 is SciPy's business and has been
# reported both ways, so a test that provoked one would be pinning SciPy's tie policy
# instead of the oracle's NaN handling. (codex review, PR #129.)

def _fixed_pvalues(*per_call):
    """A stand-in for `mannwhitneyu` returning the given p-vector on each successive
    call -- one call per target group, in `sorted(set(labels) - {CONTROL})` order."""
    it = iter(per_call)
    def fake(*_args, **_kwargs):
        return SimpleNamespace(pvalue=np.asarray(next(it), dtype=float))
    return fake


def _nan_scenario(monkeypatch):
    """Two target groups: `g1` has one NaN among two finite p-values, `g2` has nothing but
    NaN, so the BH branch runs for g1 and is skipped entirely for g2. Genes are built to
    have distinct log2FC signs in g1 (up / unchanged / down) so a column shuffle shows up.
    """
    genes = ["up", "same", "down"]
    labels = np.array(["g1"] * 3 + ["g2"] * 3 + [CONTROL] * 4)
    X = np.empty((10, 3))
    X[:, 0] = [100.0, 110.0, 120.0,  50.0, 50.0, 50.0,  10.0, 12.0, 11.0, 9.0]
    X[:, 1] = 20.0
    X[:, 2] = [1.0, 1.0, 1.0,  30.0, 30.0, 30.0,  100.0, 90.0, 110.0, 100.0]
    monkeypatch.setattr(sys.modules[__name__], "mannwhitneyu",
                        _fixed_pvalues([0.01, np.nan, 0.4], [np.nan, np.nan, np.nan]))
    return genes, _oracle_rows(X, labels, genes)


def test_oracle_emits_every_gene_when_a_p_value_is_nan(monkeypatch):
    """One row per gene per group -- not one per non-NaN gene. With `range(m)` this drops
    g1's third gene, then reuses g1's stale `m` for the all-NaN g2 (or raises
    UnboundLocalError if such a group comes first). BH must also keep using the group's
    NON-NaN count as its denominator, which is what gpudge's `bh_per_group` does."""
    genes, rows = _nan_scenario(monkeypatch)

    assert len(rows) == 6, f"expected 6 rows, got {len(rows)}"
    assert {(r[0], r[1]) for r in rows} == {(g, gene) for g in ("g1", "g2") for gene in genes}

    padj = {(r[0], r[1]): r[3] for r in rows}
    # m = 2, the non-NaN count: 0.01*2/1 = 0.02, 0.4*2/2 = 0.4, running minimum from the
    # right. With m = 3 these would be 0.03 and 0.6 instead.
    assert padj[("g1", "up")] == pytest.approx(0.02)
    assert padj[("g1", "down")] == pytest.approx(0.4)
    assert np.isnan(padj[("g1", "same")])
    assert all(np.isnan(padj[("g2", g)]) for g in genes)

    lfc = {(r[0], r[1]): r[2] for r in rows}
    assert lfc[("g1", "up")] > 0 > lfc[("g1", "down")], "gene columns are misaligned"
    assert lfc[("g1", "same")] == pytest.approx(0.0, abs=1e-12)


def test_oracle_sorts_nan_p_adj_last(monkeypatch):
    """NaN compares False against everything, so a bare `r[3]` sort key leaves a NaN row
    wherever the sort happened to drop it -- possibly ahead of the real hits the transcript
    tests read off the top."""
    _, rows = _nan_scenario(monkeypatch)

    finite = [r for r in rows if not np.isnan(r[3])]
    assert len(finite) == 2
    assert rows[:2] == finite, "the finite rows must come first, in ascending p_adj order"
    assert rows[0][3] < rows[1][3]


# ------------------------------------------------- the tutorial's actual numbers

def test_tutorial_transcript_matches_an_independent_scipy_oracle(oracle):
    """Every row of the tutorial's top-hits table, recomputed from scratch. This is the
    gate that makes the transcript trustworthy: it fails if the tutorial is edited to say
    something untrue, AND if de() ever stops producing these numbers."""
    shown = _tutorial_top_rows()
    assert len(shown) == 5, f"the tutorial shows {len(shown)} rows, expected the top 5"
    for i, (target, gene, lfc_s, padj_s) in enumerate(shown):
        o_target, o_gene, o_lfc, o_padj = oracle[i]
        assert (target, gene) == (o_target, o_gene), (
            f"row {i}: tutorial says {target}/{gene}, oracle says {o_target}/{o_gene}")
        # Compare the RENDERED strings, exactly as examples/quickstart.py prints them.
        # A numeric tolerance looser than the display (rel=0.01 vs `.2e`) lets a wrong
        # printed digit through, which is the only thing a reader ever sees.
        assert lfc_s == f"{o_lfc:.2f}", f"row {i} log2FC: shown {lfc_s}, oracle {o_lfc:.2f}"
        assert padj_s == f"{o_padj:.2e}", (
            f"row {i} p_adj: shown {padj_s}, oracle {o_padj:.2e}")


def test_tutorial_on_target_block_matches_the_oracle(oracle, adata):
    """The three figures in the on-target transcript: log2FC, p_adj, and the rank claim."""
    doc = TUTORIAL.read_text()
    # Scope to the on-target transcript block; searching the whole document would let a
    # figure elsewhere satisfy a regex about this one.
    block = re.search(r"```\non-target check .*?```", doc, re.S)
    assert block, "the tutorial no longer shows the on-target block"
    block = block.group(0)

    lfc_s = re.search(r"log2FC = (-?[\d.]+)", block)[1]
    fold_s = re.search(r"\(([\d.]+)x control\)", block)[1]
    padj_s = re.search(r"p_adj\s+= ([\d.]+e[+-]\d+)", block)[1]
    rank_shown, n_shown = re.search(r"rank\s+= (\d+) of (\d+) genes", block).groups()

    hit = next((r for r in oracle if r[0] == ON_TARGET and r[1] == ON_TARGET), None)
    assert hit is not None, f"the oracle has no {ON_TARGET}-in-{ON_TARGET} row"
    assert lfc_s == f"{hit[2]:.2f}"
    assert fold_s == f"{2 ** hit[2]:.3f}"
    assert padj_s == f"{hit[3]:.2e}"

    own = sorted((r for r in oracle if r[0] == ON_TARGET), key=lambda r: r[3])
    assert int(rank_shown) == [r[1] for r in own].index(ON_TARGET) + 1
    assert int(n_shown) == adata.n_vars

    # The prose around the block makes two quantitative claims of its own.
    fold_down = round(1 / 2 ** hit[2])
    assert f"{fold_down}-fold knockdown" in doc, (
        f"the prose no longer says {fold_down}-fold")
    assert f"{(len(GROUPS) - 1) * N_GENES} rows" in doc


def test_tutorial_and_quickstart_share_the_same_constants(quickstart_module):
    """Compare the example's ACTUAL values, not the presence of tokens in prose -- a
    token test passes on a comment mentioning the name it is supposed to be pinning."""
    doc = TUTORIAL.read_text()
    assert quickstart_module["GROUPBY"] == "target_gene"
    assert quickstart_module["CONTROL"] == CONTROL
    assert quickstart_module["ON_TARGET"] == ON_TARGET
    assert Path(quickstart_module["DATA"]).resolve() == DATA.resolve()
    for value in (quickstart_module["GROUPBY"], quickstart_module["CONTROL"],
                  quickstart_module["ON_TARGET"]):
        assert value in doc, f"the tutorial never mentions {value!r}"
    # Compare the de() calls themselves. Token presence is not enough: the quickstart's
    # explanatory comment contains the string "cpm_normalize=True", so a token test stays
    # green even if the actual call is flipped to False.
    # The example passes its constants by name where the tutorial inlines the literals,
    # so resolve names through the module before comparing VALUES.
    example = {k: quickstart_module.get(v, v) if isinstance(v, str) else v
               for k, v in _de_kwargs(QUICKSTART.read_text()).items()}
    assert example == _de_kwargs(_tutorial_de_snippet()), (
        f"the tutorial's de() call and the example's have diverged: "
        f"{_de_kwargs(_tutorial_de_snippet())} vs {example}")
    assert example["cpm_normalize"] is True


def test_tutorial_states_there_is_no_cpu_path():
    """gpudge has no CPU fallback. A quickstart that fails to say so sends readers into a
    RuntimeError with no explanation."""
    doc = TUTORIAL.read_text().lower()
    assert re.search(r"no cpu (fallback|path)", doc)
    assert "cpu fallback" not in doc.replace("no cpu fallback", ""), (
        "a second, possibly contradictory CPU claim appeared")


# ------------------------------------------------------------------- on a GPU

@needs_cuda
def test_quickstart_runs_end_to_end(tmp_path):
    """Execute the example as a reader would. Its own asserts carry the correctness
    claims; this test exists so they actually run somewhere."""
    out = tmp_path / "de.parquet"
    old = sys.argv[:]
    sys.argv = [str(QUICKSTART), "--out", str(out)]
    try:
        with pytest.raises(SystemExit) as exc:
            runpy.run_path(str(QUICKSTART), run_name="__main__")
        assert exc.value.code == 0
    finally:
        sys.argv = old
    assert out.is_file(), "--out should have written a parquet"


@needs_cuda
def test_de_agrees_with_the_oracle_the_tutorial_is_gated_on(oracle, quickstart_module):
    """Close the loop: the CPU oracle gates the transcript, and this gates gpudge against
    the oracle. Without it the two could drift together undetected."""
    res = quickstart_module["run"](DATA)
    assert res.shape == ((len(GROUPS) - 1) * N_GENES, 10)

    assert res.columns == ["target", "feature", "target_mean", "ref_mean",
                           "target_ncells", "ref_ncells", "log2_fold_change",
                           "p_value", "Ueffect", "p_adj"]

    got = {(r["target"], r["feature"]): (r["log2_fold_change"], r["p_adj"])
           for r in res.iter_rows(named=True)}
    assert len(got) == len(oracle), "duplicate or missing (target, feature) pairs"
    assert set(got) == {(t, g) for t, g, _, _ in oracle}

    # ALL 5000 rows, not a sample -- five rows covering two perturbations could not
    # detect three broken ones.
    #
    # Tolerances are set from measurement, not guessed. gpudge stages CPM in float32
    # where this oracle works in float64, which moves values on the Mann-Whitney tie
    # boundary. Measured across all 5000 rows on an H100: log2FC differs by at most
    # 2.35e-08 (the documented float32 staging floor), and p_adj by at most 5.2e-03
    # relative, on 36 rows, with none above 1e-02 and a p99 of 1.1e-06. The bounds below
    # carry ~40x and ~4x headroom over that -- still orders of magnitude tighter than
    # anything an actual regression would slip through.
    loose = 0
    for target, gene, lfc, p_adj in oracle:
        g_lfc, g_padj = got[(target, gene)]
        assert g_lfc == pytest.approx(lfc, abs=1e-6), f"{target}/{gene} log2FC"
        assert g_padj == pytest.approx(p_adj, rel=2e-2, abs=0), f"{target}/{gene} p_adj"
        if p_adj > 0 and abs(g_padj - p_adj) / p_adj > 1e-3:
            loose += 1
    # Pin the CHARACTER of the disagreement too, not just its bound: a float32 boundary
    # effect touches a handful of rows. If that fraction grows, the staging changed.
    assert loose < 0.01 * len(oracle), (
        f"{loose}/{len(oracle)} rows differ from the oracle by >1e-3 relative; "
        "measured 36 -- the float32 staging behaviour has changed")

    top = res.sort("p_adj").row(0, named=True)
    assert (top["target"], top["feature"]) == (oracle[0][0], oracle[0][1])
