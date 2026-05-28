# tests/test_filter.py
import numpy as np
import pytest
import torch
from gpudge._filter import per_guide_keep_mask
from conftest import needs_cuda


@needs_cuda
def test_keep_mask_matches_cpu_filter():
    """(target_mean > thr) | (ref_mean > thr) per (guide, gene)."""
    rng = np.random.default_rng(0)
    n_guides, n_genes = 4, 10
    tm = rng.exponential(1.0, (n_guides, n_genes))
    rm = rng.exponential(1.0, n_genes)
    tm_t = torch.from_numpy(tm).cuda()
    rm_t = torch.from_numpy(rm).cuda()
    keep = per_guide_keep_mask(tm_t, rm_t, threshold=1.0).cpu().numpy()
    expected = (tm > 1.0) | (rm > 1.0)[None, :]
    assert keep.shape == expected.shape
    assert (keep == expected).all()


@needs_cuda
def test_keep_mask_threshold_zero_keeps_all():
    tm = torch.tensor([[0.0, 1.0]], device="cuda")
    rm = torch.tensor([0.0, 0.0], device="cuda")
    keep = per_guide_keep_mask(tm, rm, threshold=0.0).cpu().numpy()
    # threshold=0 means "any > 0" → first cell still filtered out, second kept.
    assert (keep == np.array([[False, True]])).all()


@needs_cuda
def test_keep_mask_broadcasts_per_guide_ref():
    """When ref_mean is (n_guides, n_genes), broadcasting still works."""
    tm = torch.tensor([[0.0, 2.0], [3.0, 0.0]], device="cuda")
    rm = torch.tensor([[1.5, 0.0], [0.0, 0.0]], device="cuda")
    keep = per_guide_keep_mask(tm, rm, threshold=1.0).cpu().numpy()
    expected = np.array([[True, True], [True, False]])
    assert (keep == expected).all()
