# src/gpudge/_stream.py
"""Iterate (start_col, dense_torch_tensor_on_device) chunks across genes.

Accepts dense numpy ndarray or scipy.sparse CSR. For sparse, each column-block
is densified to float32 on the CPU host per chunk and then moved to device, so
the full dense matrix is never in host memory at once.
"""
from __future__ import annotations

from typing import Callable, Iterator
import logging

import numpy as np
import scipy.sparse as sp
import torch

from ._gpu_mem import _release_gpu_memory, oom_error_types

log = logging.getLogger("gpudge")

_TARGET_TILE_BYTES = 96   # per target-cell/gene: f32 tile + the ~10 (m, chunk)
                          # f64/int64 MWU working arrays. MUST equal
                          # _refpool._INMEM_TILE_BYTES -- same physical thing,
                          # two sizers. Pinned by a test.
_LFC_TILE_BYTES = 80      # added when a tau grid is active: the two hoisted
                          # (m, chunk) f64 target copies (gT64/gs64) plus one
                          # extra MWU working set. CONSTANT in n_combos.
_TAUSTAR_TILE_BYTES = 80  # added when tau_star is active. CONSTANT in n_levels.
                          # Named tensors total ~24 B/target-cell/gene: the
                          # hoisted (m, chunk) f64 gT64, the per-probe scaled
                          # copy, and _cross_tie_generic's (chunk, m) f64 rc.
                          # Set to 80 -- the _LFC_TILE_BYTES value -- because
                          # tau* runs the SAME mixed-dtype _bounds probe as the
                          # directional path, whose 80 already covers that
                          # probe's unnamed temporaries, and because the failure
                          # directions are not symmetric: over-budgeting costs a
                          # smaller chunk, under-budgeting costs an OOM
                          # downshift to chunk 64 (10-20x). Upper bound by
                          # construction, not a measurement -- if it is ever
                          # tightened, pin it with torch.cuda.max_memory_allocated
                          # around a real gene chunk rather than by inspection.


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
    n_combos: int = 0,
    n_levels: int = 0,
    max_group_rows: int = 0,
) -> int:
    """Pick a gene-chunk size from free GPU memory (pure; GPU-free).

    Working memory is dominated by reference-cell ranking buffers (~24 bytes
    per reference cell per gene: fp32 value + int64 sort index + fp64 rank +
    workspace). In ref-mode we also hold per-chunk ``(n_groups, chunk)`` fp64
    accumulators (3 for arithmetic, 4 for geometric). ``budget_n`` is the
    reference cell count (ref-mode) or n_cells (all-others). Budget is 20% of
    free memory (both paths are backstopped by the OOM-recovery driver), capped
    at 16 GB; floored at 16 and (when >= 64) rounded down to a multiple of 64.

    ``n_levels`` is the tau* accumulator ROW count, not the number of requested
    levels: it multiplies accumulator bytes, so callers pass
    ``len(levels) + 3*taustar_se`` (the SE block appends lo/hi/se rows). The
    parameter keeps its historical name because tests/test_stream.py passes it
    by keyword in 15 places.
    """
    if ref_mode:
        # ref-mode budgets the small pre-sorted reference pool (~24 B/cell/gene:
        # fp32 value + int64 sort index + fp64 rank + workspace) plus per-chunk
        # (n_groups, chunk) fp64 accumulators (3 arithmetic / 4 geometric).
        per_cell_bytes = 24
        accumulator_bytes_per_gene = (
            8 * (4 if mean_calc == "geometric" else 3) * n_groups)
        # Directional (lfc_threshold) buffers, f64, two of them (U and p):
        # the (n_combos, n_groups, chunk) accumulators, PLUS the kernel's own
        # (n_combos, chunk) outputs, which stay live across the whole combo
        # loop -- hence n_groups + 1, not n_groups. per_cell_bytes is
        # deliberately NOT touched: the shift is applied to the TARGET,
        # transiently, so nothing extra is held per REFERENCE cell (spec 4.5).
        accumulator_bytes_per_gene += 8 * 2 * n_combos * (n_groups + 1)
        # ONE accumulator (tau*), not the lfc pair (U and p) -- hence the
        # factor 1. Same n_groups + 1 reasoning: the kernel's own
        # (n_levels, chunk) output stays live across the level loop.
        accumulator_bytes_per_gene += 8 * 1 * n_levels * (n_groups + 1)
    else:
        # all_others ranks ALL cells: _rank_with_ties holds ~6 simultaneous full
        # (n_cells, chunk) f64/int64 arrays + the f32 X_chunk (~64 B/cell/gene,
        # not 24), and the (n_groups, chunk) f64 rank_sums/U/p accumulators were
        # previously unbudgeted -- together a ~2.5x under-estimate that risked a
        # first-chunk OOM + recovery tax on large runs. (L6)
        per_cell_bytes = 64
        accumulator_bytes_per_gene = 8 * 3 * n_groups
    bytes_per_gene = max(budget_n * per_cell_bytes + accumulator_bytes_per_gene, 1)
    # Phase-1 (target) peak. The
    # reference-cell term above models the Phase-0 reference sort; Phase 1 holds
    # the RESIDENT sorted reference (budget_n f32 per gene) plus the target tile
    # and its MWU working arrays plus the device accumulators -- so it is a
    # complete peak, not just the LFC delta. CONSTANT in n_combos beyond
    # `accumulator_bytes_per_gene` (the per-combo transients are freed each
    # iteration). Kept as a separate addend rather than folded into the
    # `bytes_per_gene` expression above.
    # UN-GATED as of the 2026-08 ultrareview: modelled whenever the caller knows
    # the tile height, not only when a tau grid or tau* is active. The old
    # `if n_combos or n_levels` gate was a scope decision that kept n_combos=0
    # byte-identical while lfc_threshold landed, and it left the BASE streaming
    # path budgeting only the Phase-0 reference sort — a target-dominated
    # workload then got a chunk too large for Phase 1 and paid an OOM downshift
    # (or, with oom_recovery=False, an outright OOM). `_auto_gene_chunk_size_inmem`
    # already modelled it unconditionally, so this also makes the two sizers
    # agree. The feature tiles stay conditional addends; `_TARGET_TILE_BYTES`
    # alone is the no-feature cost. Callers that cannot know the tile pass
    # max_group_rows=0 (cell_source_de) and are unaffected.
    if max_group_rows > 0 or n_combos or n_levels:
        target_peak = (budget_n * 4
                       + max_group_rows * (
                           _TARGET_TILE_BYTES
                           + (_LFC_TILE_BYTES if n_combos else 0)
                           + (_TAUSTAR_TILE_BYTES if n_levels else 0))
                       + accumulator_bytes_per_gene)
        bytes_per_gene = max(bytes_per_gene, target_peak)
    # 0.20 of free memory for both paths: each is now backstopped by the
    # OOM-recovery driver (gpudge#27 wrapped all_others too), so an over-estimate
    # downshifts instead of crashing.
    budget = min(int(free_bytes * 0.20), 16 * 1024**3)
    chunk = max(16, budget // bytes_per_gene)
    chunk = min(int(chunk), n_genes)
    if chunk >= 64:
        chunk = (chunk // 64) * 64
    return chunk


def _pinned_buf_width(chunk: int, n_genes: int) -> int:
    """Column width for the pinned target-tile host buffers.

    The gene-chunk loop never emits a tile wider than ``n_genes``, so a
    user-pinned ``gpu_gene_chunk_size`` above ``n_genes`` would over-allocate
    page-locked host memory (both the Mode-1 ``group_host_bufs`` in ``de()`` and
    the Mode-2 ``_PinnedTileUploader``). Clamp it. No-op on the auto-sized path
    (the sizers above already cap at ``n_genes``). Value-preserving: the pack
    view ``buf.reshape(-1)[:m*ch]`` still fits since ``ch <= min(chunk,
    n_genes)``. #80b
    """
    return min(int(chunk), int(n_genes))


def run_gene_chunks_with_recovery(
    n_genes: int,
    initial_chunk: int,
    process_chunk: Callable[[int, int], None],
    *,
    oom_recovery: bool = True,
    floor: int = 64,
) -> int:
    """Drive ``process_chunk(start, stop)`` over gene-chunks of width ``chunk``.

    On a CUDA out-of-memory error (torch's, or cupy's on the device-decode
    path — see ``oom_error_types``) it trims torch's and cupy's caches and, if
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
    rediscovered (and re-paying the OOM + GPU-cache reclaim) every pass.
    """
    if initial_chunk <= 0:
        raise ValueError(f"initial_chunk must be > 0, got {initial_chunk}")
    # Catch torch's OOM AND (on the device-decode path) cupy's — a disjoint
    # MemoryError-based type a torch-only except would miss, crashing de()
    # instead of downshifting. Resolved once per pass; cheap (sys.modules-cached).
    oom_types = oom_error_types()
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
        except oom_types:
            oom = True
        if not oom:
            start = stop                       # advance only on success
            continue
        # Return BOTH torch's and cupy's cached blocks to the driver so the
        # halved retry regains device memory — torch.cuda.empty_cache() alone
        # does not release cupy's pool, where the device-decode densify
        # allocates. run_gc=True collects dead tensors held by reference cycles
        # or delayed refcounts first (the canonical PyTorch OOM-recovery idiom).
        # Runs here, AFTER the except block exits, so the traceback no longer
        # pins the failed chunk's tensors and they can free before the retry.
        # (#78; Gemini review, PR #25.)
        _release_gpu_memory(run_gc=True)
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
