# tests/test_stream.py
import numpy as np
import pytest
import scipy.sparse as sp
import torch
from gpudge._stream import (
    iter_gene_chunks,
    _auto_gene_chunk_size,
    run_gene_chunks_with_recovery,
)
from conftest import needs_cuda


@needs_cuda
def test_dense_chunks_reassemble_full_matrix():
    X = np.arange(60, dtype=np.float32).reshape(5, 12)
    chunks = list(iter_gene_chunks(X, chunk_size=4, device="cuda"))
    got = torch.cat([c for _, c in chunks], dim=1).cpu().numpy()
    np.testing.assert_array_equal(got, X)
    starts = [s for s, _ in chunks]
    assert starts == [0, 4, 8]


@needs_cuda
def test_sparse_chunks_reassemble_full_matrix():
    X_dense = np.arange(60, dtype=np.float32).reshape(5, 12)
    X = sp.csr_matrix(X_dense)
    chunks = list(iter_gene_chunks(X, chunk_size=5, device="cuda"))
    got = torch.cat([c for _, c in chunks], dim=1).cpu().numpy()
    np.testing.assert_array_equal(got, X_dense)


@needs_cuda
def test_chunk_size_exceeds_n_genes_yields_one_chunk():
    X = np.zeros((3, 5), dtype=np.float32)
    chunks = list(iter_gene_chunks(X, chunk_size=100, device="cuda"))
    assert len(chunks) == 1


# --- _auto_gene_chunk_size (pure heuristic; GPU-free) ---

def test_auto_chunk_ref_mode_precise():
    # c50 regime with free=39 GiB passed explicitly: NTC ref 73,230 cells,
    # 4673 groups, arithmetic. 0.20 budget -> 4416 (deterministic formula
    # output; on the bench 4096-8192 throughput plateau).
    chunk = _auto_gene_chunk_size(
        free_bytes=39 * 1024**3, budget_n=73_230, n_groups=4673,
        mean_calc="arithmetic", n_genes=18_533, ref_mode=True)
    assert chunk == 4416
    assert chunk % 64 == 0


def test_auto_chunk_all_others_uses_020_fraction():
    # all_others (ref_mode=False) is now wrapped by OOM recovery (gpudge#27),
    # so it uses the same 0.20 budget as ref-mode (0.18 -> 3136 previously;
    # 0.20 -> 3456 here).
    chunk = _auto_gene_chunk_size(
        free_bytes=39 * 1024**3, budget_n=100_000, n_groups=20,
        mean_calc="arithmetic", n_genes=18_533, ref_mode=False)
    assert chunk == 3456


def test_auto_chunk_scales_inversely_with_reference():
    small_ref = _auto_gene_chunk_size(
        free_bytes=39 * 1024**3, budget_n=73_230, n_groups=4673,
        mean_calc="arithmetic", n_genes=18_533, ref_mode=True)
    big_ref = _auto_gene_chunk_size(
        free_bytes=39 * 1024**3, budget_n=200_000, n_groups=4673,
        mean_calc="arithmetic", n_genes=18_533, ref_mode=True)
    assert big_ref < small_ref            # bigger reference -> smaller chunk


def test_auto_chunk_caps_at_n_genes_and_floors_at_16():
    assert _auto_gene_chunk_size(
        free_bytes=39 * 1024**3, budget_n=10, n_groups=2,
        mean_calc="arithmetic", n_genes=50, ref_mode=True) == 50
    assert _auto_gene_chunk_size(
        free_bytes=1 * 1024**3, budget_n=50_000_000, n_groups=4673,
        mean_calc="arithmetic", n_genes=18_533, ref_mode=True) == 16


# --- run_gene_chunks_with_recovery (OOM driver; GPU-free) ---

def test_driver_covers_all_genes_no_oom():
    calls = []
    run_gene_chunks_with_recovery(
        50, 20, lambda a, b: calls.append((a, b)), oom_recovery=True)
    assert calls == [(0, 20), (20, 40), (40, 50)]


def test_driver_halves_and_retries_on_oom():
    seen = []
    state = {"failed": False}

    def process(a, b):
        seen.append((a, b))
        if not state["failed"] and (b - a) > 10:   # OOM once at the big chunk
            state["failed"] = True
            raise torch.cuda.OutOfMemoryError("simulated")

    run_gene_chunks_with_recovery(50, 20, process, oom_recovery=True, floor=1)
    assert seen[0] == (0, 20)            # big chunk attempted first
    successful = seen[1:]
    assert successful[0] == (0, 10)      # retried from the same start, halved
    covered = 0
    for a, b in successful:
        assert a == covered and (b - a) <= 10
        covered = b
    assert covered == 50


def test_driver_subfloor_initial_downshifts_before_raising():
    # initial_chunk (40) < default floor (64): recovery should still halve once
    # (to ~20) instead of raising on the first OOM. (Gemini review, PR #25.)
    seen = []
    n = {"calls": 0}

    def process(a, b):
        seen.append((a, b))
        n["calls"] += 1
        if n["calls"] == 1:        # OOM the first (40-wide) attempt
            raise torch.cuda.OutOfMemoryError("simulated")

    run_gene_chunks_with_recovery(40, 40, process, oom_recovery=True)  # floor=64
    assert seen[0] == (0, 40)                 # tried 40 first
    assert (seen[1][1] - seen[1][0]) <= 20    # downshifted, not raised
    assert seen[-1][1] == 40                  # finished covering all genes


def test_driver_below_16_initial_downshifts_once():
    # explicit sub-16 initial chunk should still downshift once (~initial//2)
    # before raising, not raise on the first OOM. (Gemini review, PR #25.)
    seen = []
    n = {"c": 0}

    def process(a, b):
        seen.append((a, b))
        n["c"] += 1
        if n["c"] == 1:
            raise torch.cuda.OutOfMemoryError("simulated")

    run_gene_chunks_with_recovery(8, 8, process, oom_recovery=True)  # floor=64
    assert seen[0] == (0, 8)
    assert (seen[1][1] - seen[1][0]) <= 4     # downshifted to <=4, not raised
    assert seen[-1][1] == 8


def test_driver_oom_recovery_false_raises():
    def process(a, b):
        raise torch.cuda.OutOfMemoryError("simulated")
    with pytest.raises(RuntimeError, match=r"oom_recovery=False"):
        run_gene_chunks_with_recovery(50, 20, process, oom_recovery=False)


def test_driver_floor_exhaustion_raises():
    def process(a, b):
        raise torch.cuda.OutOfMemoryError("simulated")
    with pytest.raises(RuntimeError, match=r"CUDA OOM at gpu_gene_chunk_size=64"):
        run_gene_chunks_with_recovery(1000, 64, process, oom_recovery=True, floor=64)
