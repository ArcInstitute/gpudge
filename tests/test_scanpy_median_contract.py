"""CPU contract tests for ``normalize_target_sum="median"`` against scanpy.

gpudge documents this knob as scanpy's ``normalize_total(target_sum=None)``
behaviour, but every test of that claim was ``needs_cuda`` — so neither CI cell
ever executed it and the claim went unverified on routine runs. These tests are
GPU-free: they compare :func:`gpudge._normalize.resolve_target_sum` with what
scanpy actually does, so a scanpy change is caught on both matrix cells.

Writing them surfaced a scanpy inconsistency, present in 1.11.5 and 1.12.1 —
whichever version either CI cell resolves. In
``preprocessing/_normalization.py``, ``normalize_total(target_sum=None)`` picks
its target on two different branches:

* CSR input   -> ``np.median(counts_per_cell)``          — over ALL cells
* dense/Dask  -> ``_compute_nnz_median(counts_per_cell)`` — over positive cells

so the two *can* disagree when some cell has zero total counts. (They need not:
row sums ``[0, 10, 10, 20]`` give 10 either way. Ties between the middle order
statistics mask it, which is one reason this went unnoticed.) CSC input is
converted to CSR first, so it follows the CSR branch.

gpudge implements the positive-cell median — matching scanpy's dense/Dask
branch and its internal ``_compute_nnz_median`` helper. Note scanpy's *public*
docstring says only "median of total counts for observations", without
specifying positive ones, so neither branch straightforwardly contradicts it.
gpudge's choice is the safer one regardless: on row sums ``[0, 0, 10]`` the
all-cell median is 0, and normalizing to a target of 0 zeroes out the one
non-empty cell.

Reported upstream as scverse/scanpy#4251 and fixed by scverse/scanpy#4256
(merged to scanpy ``main`` 2026-07-27, backported to the 1.12.x branch as
scverse/scanpy#4259, targeted at 1.12.4): the CSR branch now calls
``_compute_nnz_median`` too, converging on gpudge's rule. No scanpy release
carries the fix as of 1.12.3.

Affected releases run from 1.11.2 — where scanpy PR #3571 introduced the split
— through 1.12.3, plus the 1.13.0a1 prerelease, cut three days before the fix
merged. 1.11.0 and 1.11.1 predate the split: their ``_normalize_data`` takes
``np.median(counts[counts > 0])`` on every non-Dask path, so CSR and dense
already agreed there. Both CI cells currently resolve an affected version, and
the py3.11 cell will keep doing so unless the fix is also backported to 1.11.x
— it caps at 1.11.5, since scanpy >= 1.12 requires py >= 3.12.

Because gpudge's sparse paths use CSR, a user comparing gpudge against scanpy on
the *same* CSR object can therefore get a different normalization target. That
is not harmless: see ``test_target_choice_can_change_mwu_ties`` — the target is
a common scale only in exact arithmetic, and gpudge's float32 row scales mean a
different target can create or destroy ties and move U, p and p_adj. log2FC can
be target-dependent too — negligibly when both means greatly exceed ``epsilon``,
materially for zero or near-zero means, and regardless of ``epsilon`` under
``mean_calc="geometric"`` (see the ``de()`` docstring).
"""
# Deliberately importlib.metadata rather than ``sc.__version__``: scanpy 1.12.1
# raises `FutureWarning: __version__ is deprecated, use
# importlib.metadata.version('scanpy') instead`, so reading the attribute would
# trip this module's own warning filters and break when scanpy removes it.
from importlib.metadata import version

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from gpudge._mwu import _tie_term_per_gene, mwu_one_group
from gpudge._normalize import resolve_target_sum

anndata = pytest.importorskip("anndata")
sc = pytest.importorskip("scanpy")

# scanpy warns "Some cells have zero counts" on the empty-cell fixture. That
# condition is the whole point of these tests, so the warning is expected rather
# than a problem — filtered by exact message so an unrelated warning still shows.
pytestmark = pytest.mark.filterwarnings(
    "ignore:Some cells have zero counts:UserWarning"
)


def _row_sums(X):
    return np.asarray(X.sum(axis=1)).ravel()


def _scanpy_target(dense, *, as_csr):
    """The target scanpy actually normalized to, recovered empirically.

    ``normalize_total`` rescales every positive-total cell so its new total *is*
    the target, so reading those row sums back recovers it without depending on
    scanpy internals.
    """
    X = sp.csr_matrix(dense) if as_csr else dense.copy()
    a = anndata.AnnData(X=X)
    row_sums = _row_sums(a.X)
    sc.pp.normalize_total(a, target_sum=None)
    positive = row_sums > 0
    assert positive.any(), "fixture must have at least one non-empty cell"
    new_sums = _row_sums(a.X)[positive]
    np.testing.assert_allclose(
        new_sums, new_sums[0], rtol=1e-5,
        err_msg="scanpy normalized positive cells to differing totals",
    )
    return float(new_sums[0])


def _gpudge_target(dense):
    return resolve_target_sum(
        cpm_normalize=False,
        normalize_target_sum="median",
        row_sums=_row_sums(sp.csr_matrix(dense)),
    )


def _single_feature_matrix(row_sums):
    """Dense matrix whose row totals are exactly ``row_sums``.

    One populated feature per row, so a row total is a single stored value and
    cannot pick up accumulation error.
    """
    dense = np.zeros((len(row_sums), 3), dtype=np.float32)
    dense[:, 0] = np.asarray(row_sums, dtype=np.float32)
    return dense


@pytest.fixture
def counts_with_an_empty_cell():
    """Row sums 0, 10, 20, 30 — the two candidate medians differ (20 vs 15).

    Chosen so ties cannot mask the difference: with an even number of cells the
    all-cell median averages the two middle order statistics, while the
    positive-only median picks a single one.
    """
    return _single_feature_matrix([0.0, 10.0, 20.0, 30.0])


@pytest.fixture
def counts_all_cells_populated():
    """Row sums 10, 20, 30 — every cell positive, so both rules give 20."""
    return _single_feature_matrix([10.0, 20.0, 30.0])


# --- gpudge's own contract (strict) ---------------------------------------

def test_median_is_over_positive_cells_only(counts_with_an_empty_cell):
    """gpudge ignores empty cells when taking the median — the documented rule."""
    assert _gpudge_target(counts_with_an_empty_cell) == 20.0


def test_matches_scanpy_dense_path_with_an_empty_cell(counts_with_an_empty_cell):
    """The dense branch uses the positive-cell median, which is what gpudge does."""
    assert _gpudge_target(counts_with_an_empty_cell) == pytest.approx(
        _scanpy_target(counts_with_an_empty_cell, as_csr=False)
    )


def test_matches_both_scanpy_paths_when_no_cell_is_empty(counts_all_cells_populated):
    """With no empty cell the branches coincide, and gpudge agrees with both.

    This is the case real data almost always falls in — empty cells are usually
    filtered upstream — which is why the divergence stayed hidden.
    """
    gpudge_target = _gpudge_target(counts_all_cells_populated)
    assert gpudge_target == pytest.approx(
        _scanpy_target(counts_all_cells_populated, as_csr=True)
    )
    assert gpudge_target == pytest.approx(
        _scanpy_target(counts_all_cells_populated, as_csr=False)
    )


def test_all_cell_median_can_zero_out_the_only_populated_cell():
    """Why gpudge's rule is the safer one, independent of what scanpy does.

    On row sums [0, 0, 10] the all-cell median is 0; normalizing to a target of
    0 destroys the one cell carrying counts. The positive-cell median is 10.
    """
    row_sums = np.array([0.0, 0.0, 10.0])
    assert float(np.median(row_sums)) == 0.0
    assert _gpudge_target(_single_feature_matrix(row_sums)) == 10.0


# --- upstream divergence (canary, not a gpudge assertion) ------------------

def test_scanpy_csr_and_dense_disagree_on_empty_cells(counts_with_an_empty_cell):
    """Record the scanpy inconsistency, without turning a scanpy fix into red CI.

    On a scanpy whose two branches agree *on the positive-cell median* the test
    SKIPS rather than failing — the same reasoning as the deliberate ruff upper
    bound in pyproject.toml: an upstream release should not turn `main` red on
    code that did not change.

    It checks *which way* they agree before skipping, which is the one
    gpudge-facing claim this test makes. scverse/scanpy#4256 agrees on the
    positive-cell median, gpudge's rule; agreeing on the all-cell median
    instead would break gpudge's parity claim on *both* paths rather than one,
    so that case deliberately FAILS rather than skipping.

    The target check carries an explicit tolerance rather than `pytest.approx`'s
    default `rel=1e-6`, which is tighter than the `np.isclose` gate that admits
    into this branch.
    """
    csr_target = _scanpy_target(counts_with_an_empty_cell, as_csr=True)
    dense_target = _scanpy_target(counts_with_an_empty_cell, as_csr=False)
    if np.isclose(csr_target, dense_target):
        # Agreed — but pin WHICH way, before skipping. Explicit tolerance on
        # the same scale as the np.isclose gate above, rather than
        # pytest.approx's tighter rel=1e-6 default. Only the scale matches:
        # the two combine rel/abs differently (numpy adds, approx takes the
        # max), and they compare different pairs anyway — the gate is CSR vs
        # dense, this is CSR vs 20.0.
        assert csr_target == pytest.approx(20.0, rel=1e-5, abs=1e-8), (
            f"scanpy {version('scanpy')} agrees across CSR/dense on "
            f"{csr_target}, not the positive-cell median (20.0) that gpudge "
            "implements — the normalize_target_sum parity claim no longer "
            "holds on either path"
        )
        # Behaviour-based, not version-inferred: `scanpy>=1.11` admits both
        # builds carrying scverse/scanpy#4256 and builds predating the 1.11.2
        # split, and both land here. Phrased non-exhaustively — a patched or
        # forked scanpy would land here too.
        pytest.skip(
            f"scanpy {version('scanpy')}: CSR and dense agree on the "
            "positive-cell median, matching gpudge — expected on builds "
            "predating the 1.11.2 split and on those carrying "
            "scverse/scanpy#4256. Nothing to do."
        )
    # approx, not exact: both are recovered from float32 rescaling, so a
    # different scipy/BLAS build could shift the last bits without changing the
    # branch behaviour under test.
    assert csr_target == pytest.approx(15.0)    # np.median over ALL row sums
    assert dense_target == pytest.approx(20.0)  # median over positive rows only


def test_scanpy_leaves_empty_cells_at_zero(counts_with_an_empty_cell):
    """scanpy does not invent counts for an empty cell (no divide-by-zero blowup)."""
    a = anndata.AnnData(X=sp.csr_matrix(counts_with_an_empty_cell))
    sc.pp.normalize_total(a, target_sum=None)
    assert _row_sums(a.X)[0] == 0.0


# --- why the divergence is not cosmetic ------------------------------------

def _mwu_at_target(row_sums, target, *, n_target):
    """Run gpudge's own MWU kernel on CPU over one gene, normalized to ``target``.

    Mirrors what the driver does: row scale ``target / row_sum`` cast to float32,
    applied in float32. Returns ``(U1, p)``.
    """
    positive = row_sums > 0
    counts = row_sums.astype(np.float32)
    safe = np.where(positive, row_sums, 1.0)
    scales = (target / safe).astype(np.float32)
    values = (counts * scales).astype(np.float32)[positive]

    target_vals = torch.tensor(values[:n_target], dtype=torch.float32).reshape(1, -1)
    ref = torch.tensor(values[n_target:], dtype=torch.float32).reshape(1, -1)
    sorted_ref, _ = torch.sort(ref, dim=1)
    u1, p = mwu_one_group(sorted_ref, _tie_term_per_gene(sorted_ref),
                          target_vals, n_ref=ref.shape[1])
    return float(u1[0]), float(p[0]), len(np.unique(values))


def test_target_choice_can_change_mwu_ties():
    """The target is a common scale only in EXACT arithmetic.

    gpudge builds row scales as ``(target_sum / row_sums).astype(np.float32)``
    and the rank kernel treats equal float32 values as ties, so two targets
    differing by a constant factor can still produce different tie structure —
    and therefore different U and p. Constructed rather than sampled, to make
    the mechanism unmissable: at 2.0 every positive cell collapses to one
    float32 value, so all pairs tie and U sits at the midpoint m·n/2 with p
    driven to 1.0 by gpudge's zero-variance clamp; at 1.5 two strata survive
    and the same data is strongly significant.

    Asserted against gpudge's own kernel rather than scipy, since the claim
    being documented is about gpudge's outputs.
    """
    row_sums = np.array([0, 0] + [1] * 10 + [2] * 2 + [21] * 10, dtype=np.float64)
    positive = row_sums > 0
    target_all = float(np.median(row_sums))                 # scanpy CSR branch
    target_positive = float(np.median(row_sums[positive]))  # gpudge / dense branch
    assert (target_all, target_positive) == (1.5, 2.0)

    u_all, p_all, distinct_all = _mwu_at_target(row_sums, target_all, n_target=12)
    u_pos, p_pos, distinct_pos = _mwu_at_target(row_sums, target_positive, n_target=12)

    assert (distinct_all, distinct_pos) == (2, 1), "expected collapse only at 2.0"
    # These two are structurally exact, not numerics-dependent: with every value
    # tied, U is the midpoint m·n/2 = 60 and gpudge's zero-variance clamp sends p
    # to 1.0; with complete separation U1 is 0.
    assert (u_pos, p_pos) == (60.0, 1.0)
    assert u_all == 0.0
    # The p-value itself is only asserted qualitatively — its exact value runs
    # through torch's erfc and should not pin the test to a torch build.
    assert p_all < 1e-5 < p_pos, "the two targets must not agree on significance"
