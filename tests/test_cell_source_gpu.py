"""GPU parity gate for de(cell_source=...).

de(adata=, reference=<AnnData>) and de(cell_source=) run the IDENTICAL
refpool_de_core over the identical cells, so their output must be
byte-identical. That is the merge gate for #86.

check_exact=True is NOT optional: polars' assert_frame_equal defaults to
check_exact=False, which silently turned a byte-identity gate into a tolerance
check once already (#110).
"""
import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp
from polars.testing import assert_frame_equal

import gpudge
from gpudge import CellGroup

from conftest import needs_cuda

N_GENES = 40

EXACT = dict(check_exact=True, check_column_order=True,
             check_row_order=True, check_dtypes=True)


def _fixture(seed=0, n_per_group=25, n_ref=60, n_groups=4):
    rng = np.random.default_rng(seed)
    labels = np.repeat([f"G{i}" for i in range(n_groups)], n_per_group)
    X = sp.csr_matrix(
        rng.poisson(2.0, size=(labels.size, N_GENES)).astype(np.float32))
    var_names = np.asarray([f"g{j}" for j in range(N_GENES)], dtype=str)
    adata = ad.AnnData(X=X, obs={"pert": labels})
    adata.var_names = var_names
    reference = ad.AnnData(X=sp.csr_matrix(
        rng.poisson(2.0, size=(n_ref, N_GENES)).astype(np.float32)))
    reference.var_names = var_names
    return adata, reference, labels, var_names


def _source_from(adata, labels, targets):
    """Yield each group as its own CSR, in a DELIBERATELY reversed order."""
    def factory():
        for label in list(targets)[::-1]:
            yield CellGroup(label, adata.X[np.flatnonzero(labels == label)])
    return factory


@needs_cuda
@pytest.mark.parametrize("extra", [
    {},
    {"cpm_normalize": True},
    {"cpm_normalize": True, "filter_gene_min_cpm_cell": 1.0},
    # cpm_bulk is a DISTINCT branch -- the only consumer of Ls_for_rows.sum()
    # (_refpool.py:196), so the cpm_cell case above does not cover it.
    {"cpm_normalize": True, "filter_gene_min_cpm_bulk": 1.0},
    {"lfc_threshold": 0.5, "tau_star": (0.5,)},
])
def test_cell_source_is_byte_identical_to_inmem_external_ref(extra):
    adata, reference, labels, var_names = _fixture()
    targets = np.unique(labels)          # ingest uses np.unique -> sorted

    expected = gpudge.de(adata=adata, groupby="pert", reference=reference,
                         **extra)
    got = gpudge.de(cell_source=_source_from(adata, labels, targets),
                    targets=targets, var_names=var_names,
                    reference=reference, **extra)
    assert_frame_equal(got, expected, **EXACT)


@needs_cuda
def test_rows_subset_of_a_shared_matrix_matches_pregathered_groups():
    """Also pins the slicing decision in spec Section 4: the two row-sum routes
    must agree numerically, not just in cost."""
    adata, reference, labels, var_names = _fixture(seed=2)
    targets = np.unique(labels)

    def shared():
        for label in targets:
            yield CellGroup(label, adata.X, np.flatnonzero(labels == label))

    kw = dict(targets=targets, var_names=var_names, reference=reference,
              cpm_normalize=True, filter_gene_min_cpm_bulk=1.0)
    pregathered = gpudge.de(
        cell_source=_source_from(adata, labels, targets), **kw)
    subset = gpudge.de(cell_source=shared, **kw)
    assert_frame_equal(subset, pregathered, **EXACT)


@needs_cuda
def test_a_bare_matrix_reference_matches_an_anndata_reference():
    adata, reference, labels, var_names = _fixture(seed=4)
    targets = np.unique(labels)
    kw = dict(cell_source=_source_from(adata, labels, targets),
              targets=targets, var_names=var_names, cpm_normalize=True)
    assert_frame_equal(gpudge.de(reference=reference.X, **kw),
                       gpudge.de(reference=reference, **kw), **EXACT)
