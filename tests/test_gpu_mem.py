# tests/test_gpu_mem.py
"""Unit tests for _release_gpu_memory (CPU; torch.cuda + cupy mocked). #76"""
from __future__ import annotations

import gc
import sys
import types

import gpudge._gpu_mem as gm


def test_release_noop_when_cuda_unavailable(monkeypatch):
    import torch
    called = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "empty_cache",
                        lambda: called.append("empty_cache"))
    gm._release_gpu_memory()
    assert called == []            # short-circuits before empty_cache


def test_release_calls_torch_and_cupy(monkeypatch):
    import torch
    called = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache",
                        lambda: called.append("empty_cache"))

    class _Pool:
        def free_all_blocks(self):
            called.append("free_all_blocks")

    fake_cupy = types.SimpleNamespace(
        get_default_memory_pool=lambda: _Pool(),
        get_default_pinned_memory_pool=lambda: _Pool())
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)

    gm._release_gpu_memory(run_gc=True)
    assert called == ["empty_cache", "free_all_blocks", "free_all_blocks"]


def test_release_survives_missing_cupy(monkeypatch):
    import torch
    called = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache",
                        lambda: called.append("empty_cache"))
    # A None entry in sys.modules is CPython's "halt" sentinel: `import cupy`
    # raises ImportError ("import of cupy halted; None in sys.modules"),
    # simulating cupy absent even when it is installed. (Verified empirically.)
    monkeypatch.setitem(sys.modules, "cupy", None)
    gm._release_gpu_memory()               # must not raise
    assert called == ["empty_cache"]


def test_release_runs_gc_only_when_requested(monkeypatch):
    import torch
    flags = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(gc, "collect", lambda: flags.append("gc"))
    monkeypatch.setitem(sys.modules, "cupy", None)
    gm._release_gpu_memory(run_gc=True)
    assert flags == ["gc"]
    flags.clear()
    gm._release_gpu_memory(run_gc=False)
    assert flags == []


# --- oom_error_types (which OOM exceptions the recovery loop catches) #78 ---

def test_oom_error_types_includes_torch():
    import torch
    assert torch.cuda.OutOfMemoryError in gm.oom_error_types()


def test_oom_error_types_includes_cupy_iff_available():
    # Environment-adaptive: cupy's OOM type is present exactly when cupy imports.
    types_ = gm.oom_error_types()
    try:
        from cupy.cuda.memory import OutOfMemoryError as CupyOOM
    except Exception:
        assert len(types_) == 1                 # cupy absent -> torch-only
    else:
        assert CupyOOM in types_


def test_oom_error_types_torch_only_when_cupy_absent(monkeypatch):
    # sys.modules["cupy"] = None is CPython's halt sentinel: `import cupy` raises
    # even when cupy is installed, so the fallback branch runs deterministically.
    import torch
    monkeypatch.setitem(sys.modules, "cupy", None)
    assert gm.oom_error_types() == (torch.cuda.OutOfMemoryError,)


def test_oom_error_types_survives_broken_cupy(monkeypatch):
    # `import cupy` succeeds but the OOM type is unreachable (partial/broken
    # install, or a stub in sys.modules): must fall back to torch-only and NEVER
    # raise — the recovery loop calls this on every pass, incl. CPU/host runs.
    import torch
    monkeypatch.setitem(sys.modules, "cupy", types.SimpleNamespace())  # no .cuda
    assert gm.oom_error_types() == (torch.cuda.OutOfMemoryError,)
