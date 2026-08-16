# src/gpudge/_gpu_mem.py
"""Return reclaimable GPU memory to the CUDA driver.

de() sizes its work to fill the card, and its caching allocators (torch, and
cupy on the streaming device-decode path) retain freed blocks in-pool rather
than returning them to the driver. A same-process caller that does its own GPU
work after de() then finds the driver starved (cudaMalloc / cuBLAS OOM even
though the bytes are "free" inside a pool), and de()'s auto-sizer, reading
cudaMemGetInfo before it sizes, sees a starved "free" if a prior caller phase
left a pool populated. This helper trims those pools. gpudge_arc#76.
"""
from __future__ import annotations

import gc

import torch


def _release_gpu_memory(run_gc: bool = False) -> None:
    """Return gpudge's reclaimable GPU caches to the CUDA driver.

    Empties torch's caching allocator and, when cupy is importable, frees its
    default device and pinned memory pools' UNUSED blocks — so driver-level
    allocators (cuBLAS, cuSPARSE, RMM, …) and other frameworks can reuse the
    bytes. ``free_all_blocks()`` frees only cached/unused blocks; live arrays
    are never touched. When ``run_gc`` is True a ``gc.collect()`` runs first, so
    a caller's dropped-but-cyclic device arrays are collected before the pools
    are trimmed. No-op when CUDA is unavailable. Never raises — a reclaim
    failure must not mask a successful de(). gpudge_arc#76.
    """
    try:
        if not torch.cuda.is_available():
            return
        if run_gc:
            gc.collect()
        torch.cuda.empty_cache()
        try:
            import cupy  # optional dep (gpudge[streaming-gpu]); absent -> skip
        except Exception:
            return
        try:
            cupy.get_default_memory_pool().free_all_blocks()
            cupy.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass
    except Exception:
        pass


def oom_error_types() -> tuple[type[BaseException], ...]:
    """CUDA out-of-memory exception types to catch in the gene-chunk recovery loop.

    Always includes ``torch.cuda.OutOfMemoryError``. On the streaming
    device-decode path each target tile is densified through cupy, which raises
    ``cupy.cuda.memory.OutOfMemoryError`` — a DISJOINT hierarchy (MemoryError-based,
    not RuntimeError-based: ``issubclass`` is False), so a torch-only ``except``
    would let it crash de() instead of downshifting the gene chunk. When cupy is
    importable its OOM type is appended so both allocators' OOMs trigger the same
    chunk-halving recovery. The SPECIFIC cupy type is used (not a broad
    ``MemoryError``) so genuine host ``MemoryError``s still surface. gpudge_arc#78.
    """
    types: tuple[type[BaseException], ...] = (torch.cuda.OutOfMemoryError,)
    try:
        import cupy  # optional dep (gpudge[streaming-gpu]); absent -> torch-only
        # Resolve the type INSIDE the try: a partial/broken/stubbed cupy can
        # import but leave cupy.cuda.memory unreachable, and this helper runs on
        # every recovery pass (incl. CPU/host) — it must never raise. gpudge_arc#78.
        cupy_oom = cupy.cuda.memory.OutOfMemoryError
    except Exception:
        return types
    return types + (cupy_oom,)
