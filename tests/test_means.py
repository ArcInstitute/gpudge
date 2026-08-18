# tests/test_means.py
import numpy as np
import pytest
import torch
from gpudge._means import group_means
from conftest import needs_cuda


# `group_means` is device-agnostic, but until the 2026-08 ultrareview its only
# oracle tests were @needs_cuda with a hard-coded .cuda(), so CPU CI validated
# NONE of target_mean / ref_mean / log2_fold_change: destroying the segment
# reduction, the mean division or the empty-group guard left the CPU suite
# green. Parametrized over both devices instead. The oracles also reduce in
# FLOAT64 now -- reducing in float32, as they did, made a float32-accumulation
# regression invisible on every device (it moved the oracle the same way).
DEVICES = ["cpu", pytest.param("cuda", marks=needs_cuda)]


def _gold_arith(X, labels, n_groups):
    out = np.zeros((n_groups, X.shape[1]), dtype=np.float64)
    for g in range(n_groups):
        m = labels == g
        if m.sum() == 0:
            continue
        out[g] = X[m].astype(np.float64).mean(axis=0)
    return out


def _gold_geom(X, labels, n_groups):
    out = np.zeros((n_groups, X.shape[1]), dtype=np.float64)
    for g in range(n_groups):
        m = labels == g
        if m.sum() == 0:
            continue
        out[g] = np.expm1(np.log1p(X[m].astype(np.float64)).mean(axis=0))
    return out


def _fixture(n_cells=200, n_genes=50, n_groups=5, empty_group=False):
    rng = np.random.default_rng(0)
    X = rng.exponential(1.0, (n_cells, n_genes)).astype(np.float32)
    labels = rng.integers(0, n_groups, n_cells).astype(np.int32)
    if empty_group:
        # Leave the LAST group with no cells: the empty-group guard has no
        # oracle otherwise, since every group is populated at these sizes.
        labels[labels == n_groups - 1] = 0
        assert (labels == n_groups - 1).sum() == 0
    return X, labels, n_groups


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("empty_group", [False, True])
def test_arith_means_match_numpy_dense(device, empty_group):
    X, labels, n_groups = _fixture(empty_group=empty_group)
    got = group_means(torch.from_numpy(X).to(device),
                      torch.from_numpy(labels).to(device),
                      n_groups, kind="arithmetic").cpu().numpy()
    # rtol=1e-11 against a float64 oracle, not 1e-5 against a float32 one: both
    # sides now reduce in float64, so the only residual is summation ORDER
    # (~n_cells * eps ~ 4e-14). Bit-exact on CPU as measured.
    np.testing.assert_allclose(got, _gold_arith(X, labels, n_groups),
                               rtol=1e-11, atol=1e-13)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("empty_group", [False, True])
def test_geom_means_match_numpy_dense(device, empty_group):
    X, labels, n_groups = _fixture(empty_group=empty_group)
    got = group_means(torch.from_numpy(X).to(device),
                      torch.from_numpy(labels).to(device),
                      n_groups, kind="geometric").cpu().numpy()
    # Looser than the arithmetic pair on purpose: group_means takes log1p in
    # FLOAT32 (the oracle takes it in float64), which is worth ~2e-8 relative.
    np.testing.assert_allclose(got, _gold_geom(X, labels, n_groups),
                               rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("device", DEVICES)
def test_unknown_kind_raises(device):
    X = torch.zeros((10, 5), device=device)
    labels = torch.zeros(10, dtype=torch.int32, device=device)
    with pytest.raises(ValueError, match="kind"):
        group_means(X, labels, 1, kind="harmonic")


def test_geometric_mean_out_of_domain_propagates_deterministically():
    """Geometric mean uses log1p, defined only for X > -1. Input domain is the
    caller's responsibility (gpudge does not transform X); out-of-domain inputs
    propagate DETERMINISTICALLY rather than raising or clamping:
      * X < -1  -> NaN          (log1p(X) is NaN)
      * X == -1 -> -1.0         (log1p(-1) = -inf -> expm1(-inf) = -1.0)
    Pin BOTH boundaries so they can't drift. (ultrareview #47; CPU tensors —
    group_means is device-agnostic.)"""
    # 3 genes: in-domain (>-1), the exact boundary (==-1), out-of-domain (<-1).
    X = torch.tensor([[2.0, -1.0, -2.0], [4.0, -1.0, -3.0]], dtype=torch.float32)
    labels = torch.tensor([0, 0], dtype=torch.int32)
    geo = group_means(X, labels, 1, kind="geometric")
    assert torch.isfinite(geo[0, 0])           # in-domain (>-1) -> finite
    assert geo[0, 1].item() == -1.0            # X == -1 -> -1.0 (finite, not NaN)
    assert torch.isnan(geo[0, 2])              # X < -1 -> NaN (not an exception)
    # arithmetic mode has no domain restriction (no log1p) — stays finite.
    assert torch.isfinite(group_means(X, labels, 1, kind="arithmetic")).all()
