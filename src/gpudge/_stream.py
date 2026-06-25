# src/gpudge/_stream.py
"""Iterate (start_col, dense_torch_tensor_on_device) chunks across genes.

Accepts dense numpy ndarray or scipy.sparse CSR. For sparse, each column-block
is densified to float32 on the CPU host per chunk and then moved to device, so
the full dense matrix is never in host memory at once.
"""
from __future__ import annotations

from typing import Callable, Iterator
import gc
import logging

import numpy as np
import scipy.sparse as sp
import torch

log = logging.getLogger("gpudge")


def iter_gene_chunks(
    X,                          # ndarray (cells, genes) or CSR
    *,
    chunk_size: int,
    device: str | torch.device = "cuda",
) -> Iterator[tuple[int, torch.Tensor]]:
    """Yield (start_col, chunk_tensor) for col-blocks of size `chunk_size`."""
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
    n_cells, n_genes = X.shape
    for start in range(0, n_genes, chunk_size):
        stop = min(start + chunk_size, n_genes)
        if sp.issparse(X):
            # Slice CSR by columns → CSC for efficiency, then to dense
            block_csc = X[:, start:stop].tocsc()
            t = torch.from_numpy(block_csc.toarray().astype(np.float32, copy=False))
        else:
            t = torch.from_numpy(np.ascontiguousarray(X[:, start:stop],
                                                     dtype=np.float32))
        yield start, t.to(device, non_blocking=True)


def _auto_gene_chunk_size(
    *,
    free_bytes: int,
    budget_n: int,
    n_groups: int,
    mean_calc: str,
    n_genes: int,
    ref_mode: bool,
) -> int:
    """Pick a gene-chunk size from free GPU memory (pure; GPU-free).

    Working memory is dominated by reference-cell ranking buffers (~24 bytes
    per reference cell per gene: fp32 value + int64 sort index + fp64 rank +
    workspace). In ref-mode we also hold per-chunk ``(n_groups, chunk)`` fp64
    accumulators (3 for arithmetic, 4 for geometric). ``budget_n`` is the
    reference cell count (ref-mode) or n_cells (all-others). Budget is 20% of
    free memory (both paths are backstopped by the OOM-recovery driver), capped
    at 16 GB; floored at 16 and (when >= 64) rounded down to a multiple of 64.
    """
    accumulator_bytes_per_gene = (
        8 * (4 if mean_calc == "geometric" else 3) * n_groups if ref_mode else 0
    )
    bytes_per_gene = max(budget_n * 24 + accumulator_bytes_per_gene, 1)
    # 0.20 of free memory for both paths: each is now backstopped by the
    # OOM-recovery driver (gpudge#27 wrapped all_others too), so an over-estimate
    # downshifts instead of crashing.
    budget = min(int(free_bytes * 0.20), 16 * 1024**3)
    chunk = max(16, budget // bytes_per_gene)
    chunk = min(int(chunk), n_genes)
    if chunk >= 64:
        chunk = (chunk // 64) * 64
    return chunk


def run_gene_chunks_with_recovery(
    n_genes: int,
    initial_chunk: int,
    process_chunk: Callable[[int, int], None],
    *,
    oom_recovery: bool = True,
    floor: int = 64,
) -> int:
    """Drive ``process_chunk(start, stop)`` over gene-chunks of width ``chunk``.

    On ``torch.cuda.OutOfMemoryError`` it empties the CUDA cache and, if
    ``oom_recovery`` is True and ``chunk > floor``, halves the chunk (>= floor)
    and retries from the SAME start. (If ``initial_chunk`` is already below
    ``floor``, the effective floor drops to ``initial_chunk // 2`` so a sub-floor
    chunk can still downshift once.) ``process_chunk`` must be idempotent over a
    re-processed gene range (gpudge writes per-gene accumulators by absolute
    index, so re-covering ``[start, stop)`` overwrites identically). Raises
    ``RuntimeError`` if ``oom_recovery`` is False or the floor still OOMs.

    Returns the final chunk width — the initial width if no OOM occurred, or the
    downshifted width it settled on. Callers that drive multiple passes (the
    shard-streaming driver, one pass per target group) can feed this back as the
    next pass's ``initial_chunk`` so a downshift persists instead of being
    rediscovered (and re-paying the OOM + ``empty_cache``) every pass.
    """
    if initial_chunk <= 0:
        raise ValueError(f"initial_chunk must be > 0, got {initial_chunk}")
    start, chunk = 0, initial_chunk
    # If the initial chunk is already below `floor`, lower the effective floor
    # (to >= 1) so any sub-floor chunk can still downshift once instead of
    # raising on the first OOM. (Gemini review, PR #25.)
    actual_floor = floor if initial_chunk >= floor else max(1, initial_chunk // 2)
    while start < n_genes:
        stop = min(start + chunk, n_genes)
        # Catch OOM with a flag and handle it AFTER the except block exits. An
        # active `except` keeps the exception's traceback alive, which holds
        # process_chunk's frame and the GPU tensors it allocated — so
        # empty_cache() inside the except can't reclaim them and the smaller
        # retry gains no memory. Clearing the traceback first lets those tensors
        # be freed before we retry. (Gemini review, PR #25.)
        oom = False
        try:
            process_chunk(start, stop)
        except torch.cuda.OutOfMemoryError:
            oom = True
        if not oom:
            start = stop                       # advance only on success
            continue
        if torch.cuda.is_available():
            # gc.collect() first so dead tensors held only by reference cycles
            # or delayed refcounts are released before we hand cached blocks
            # back to the driver — the canonical PyTorch OOM-recovery idiom.
            # (Gemini review, PR #25.)
            gc.collect()
            torch.cuda.empty_cache()
        if not oom_recovery or chunk <= actual_floor:
            raise RuntimeError(
                f"de(): CUDA OOM at gpu_gene_chunk_size={chunk}"
                + ("" if oom_recovery else " (oom_recovery=False)")
                + "; pass a smaller gpu_gene_chunk_size or reduce the "
                "reference group."
            )
        new_chunk = max(actual_floor, chunk // 2)
        log.warning(
            "de(): gene-chunk OOM at gpu_gene_chunk_size=%d -> retrying at %d",
            chunk, new_chunk,
        )
        chunk = new_chunk
    return chunk
