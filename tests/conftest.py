# tests/conftest.py
"""Pytest fixtures for gpudge tests."""
from __future__ import annotations

import numpy as np
import pytest
import anndata as ad
import scipy.sparse as sp

# Skip marker — used by tests that require a CUDA device.
import torch
needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="requires CUDA GPU",
)

# Shared tau grid for the lfc_threshold GPU parity gates. 0.4 is deliberately
# NON-integer, so the float64 scale factor is irrational-ish and the _bounds
# boundary correction (spec 3.2b) is exercised on nearly every value; 1.0 is an
# exact power of two, where the scaling is exact and cross ties are genuine;
# 0.0 keeps the recombination identity in play.
LFC_TAUS = [0.0, 0.4, 1.0]


def _make_synth(n_cells: int, n_genes: int, n_guides: int,
                ntc_frac: float = 0.3, seed: int = 0,
                sparse: bool = False) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    # Most genes ~0; rare genes spike. Realistic-ish sparse pattern.
    X = rng.negative_binomial(2, 0.2, size=(n_cells, n_genes)).astype(np.float32)
    if sparse:
        X = sp.csr_matrix(X)

    n_ntc = int(n_cells * ntc_frac)
    n_target = n_cells - n_ntc
    guide_labels = np.concatenate([
        np.repeat([f"G{i}" for i in range(n_guides)],
                  n_target // n_guides + 1)[:n_target],
        np.array([f"non-targeting-{i}" for i in range(n_ntc)]),
    ])
    rng.shuffle(guide_labels)
    obs = {"target_guide": guide_labels,
           "comparison": np.where(np.char.startswith(guide_labels, "non-targeting"),
                                  "ntc", guide_labels)}
    var = {"gene_id": [f"g{i}" for i in range(n_genes)]}
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.var_names = [f"g{i}" for i in range(n_genes)]
    return adata


@pytest.fixture
def synth_small():
    """Tiny dense AnnData: 200 cells × 50 genes × 5 guides + NTC."""
    return _make_synth(n_cells=200, n_genes=50, n_guides=5, sparse=False)


@pytest.fixture
def synth_small_sparse():
    """Same as synth_small but sparse CSR."""
    return _make_synth(n_cells=200, n_genes=50, n_guides=5, sparse=True)


@pytest.fixture
def synth_medium():
    """Medium: 2000 cells × 200 genes × 20 guides."""
    return _make_synth(n_cells=2000, n_genes=200, n_guides=20, sparse=False)
