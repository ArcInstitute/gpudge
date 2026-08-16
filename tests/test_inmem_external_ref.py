# tests/test_inmem_external_ref.py
"""CPU-only validation + dispatch-routing tests for the in-memory external
reference path de(adata=..., reference=<AnnData>). The numerical correctness
gate is the GPU bit-identity parity test in test_inmem_external_ref_gpu.py."""
from __future__ import annotations

import numpy as np
import pytest
import anndata as ad  # noqa: F401  (kept for parity with sibling test modules)

import gpudge
from conftest import _make_synth   # bare import (no tests. prefix) — CI gotcha


@pytest.fixture
def targets_and_ref():
    """A targets AnnData (guides only) + a separate reference AnnData (ntc),
    sharing the same gene axis."""
    full = _make_synth(n_cells=400, n_genes=30, n_guides=6, sparse=True, seed=11)
    is_ntc = np.char.startswith(
        full.obs["comparison"].to_numpy().astype(str), "ntc")
    targets = full[~is_ntc].copy()
    ref = full[is_ntc].copy()
    return targets, ref


def test_var_axis_count_mismatch_raises(targets_and_ref):
    targets, ref = targets_and_ref
    bad = ref[:, : ref.n_vars - 1].copy()          # wrong n_vars
    with pytest.raises(ValueError, match="gene"):
        gpudge.de(targets, groupby="comparison", reference=bad)


def test_var_axis_order_mismatch_raises(targets_and_ref):
    targets, ref = targets_and_ref
    shuffled = ref[:, ::-1].copy()                  # same genes, reversed order
    with pytest.raises(ValueError, match="var_names|gene axis"):
        gpudge.de(targets, groupby="comparison", reference=shuffled)


def test_empty_reference_raises(targets_and_ref):
    targets, ref = targets_and_ref
    empty = ref[:0].copy()                          # 0 cells
    with pytest.raises(ValueError, match="0 cells|non-empty"):
        gpudge.de(targets, groupby="comparison", reference=empty)


def test_densify_input_with_external_ref_raises(targets_and_ref):
    targets, ref = targets_and_ref
    with pytest.raises(ValueError, match="densify_input"):
        gpudge.de(targets, groupby="comparison", reference=ref, densify_input=True)


def test_groupby_required_with_external_ref(targets_and_ref):
    targets, ref = targets_and_ref
    with pytest.raises(ValueError, match="groupby"):
        gpudge.de(targets, reference=ref)           # groupby=None


def test_non_str_non_anndata_reference_still_raises(synth_small):
    # A list reference is neither a group label nor an AnnData -> clear error.
    with pytest.raises(ValueError, match="group-label string|AnnData"):
        gpudge.de(synth_small, groupby="target_guide", reference=[1, 2, 3])


def test_anndata_reference_routes_to_core(monkeypatch, targets_and_ref):
    """A valid AnnData reference in-memory dispatches to inmem_external_ref_de
    (NOT the label-ref loop). Proven on CPU by faking CUDA-available and stubbing
    the core wrapper."""
    targets, ref = targets_and_ref
    import gpudge._refpool as rp
    import torch

    called = {}

    def _stub(adata, **kwargs):
        called["adata_is_targets"] = adata is targets
        called["reference_is_ref"] = kwargs.get("reference") is ref
        called["groupby"] = kwargs.get("groupby")
        return "ROUTED"

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(rp, "inmem_external_ref_de", _stub)
    out = gpudge.de(targets, groupby="comparison", reference=ref)
    assert out == "ROUTED"
    assert called == {"adata_is_targets": True, "reference_is_ref": True,
                      "groupby": "comparison"}


def test_inmem_sizer_reserves_resident_and_rounds():
    from gpudge._refpool import _auto_gene_chunk_size_inmem

    # Tiny free vs a large resident reference -> floors at 64.
    # resident = 77234*18533*4 + 18533*8 ~= 5.73 GB; available ~= 0.27 GB.
    resident_heavy = _auto_gene_chunk_size_inmem(
        free_bytes=6_000_000_000, n_ref=77_234, n_genes=18_533,
        max_group_rows=20_000)
    assert resident_heavy == 64

    # Capped at n_genes when free is huge, then rounded DOWN to a multiple of 64
    # (2000 -> 1984).
    assert _auto_gene_chunk_size_inmem(
        free_bytes=400_000_000_000, n_ref=100, n_genes=2_000,
        max_group_rows=100) == 1_984

    # Real #72 shape on a dedicated ~80 GiB H100 (repro logged free ~= 84.5e9 B):
    # lands near the empirically-safe 4544 (was 2304 under the old 16 GiB cap).
    real72 = _auto_gene_chunk_size_inmem(
        free_bytes=84_500_000_000, n_ref=77_234, n_genes=18_533,
        max_group_rows=77_234)
    assert real72 % 64 == 0
    assert 4_096 <= real72 <= 5_120       # ~4608; comfortably > the old 2304

    # Budget CAP plateaus on big GPUs: once available*fraction exceeds the cap,
    # more free memory does not grow the chunk.
    huge = _auto_gene_chunk_size_inmem(
        free_bytes=200_000_000_000, n_ref=77_234, n_genes=18_533,
        max_group_rows=77_234)
    assert huge == real72

    # Fraction-of-available protects smaller GPUs: a 40 GB card is
    # fraction-limited (budget < cap) -> a strictly smaller chunk than the 80 GB
    # pick, never an OOM-inducing over-provision.
    small_gpu = _auto_gene_chunk_size_inmem(
        free_bytes=40_000_000_000, n_ref=77_234, n_genes=18_533,
        max_group_rows=77_234)
    assert small_gpu % 64 == 0
    assert small_gpu < real72

    # Monotonic in free (until the cap plateau).
    assert small_gpu <= _auto_gene_chunk_size_inmem(
        free_bytes=60_000_000_000, n_ref=77_234, n_genes=18_533,
        max_group_rows=77_234) <= real72


def test_inmem_sizer_reserves_headroom():
    """The sizer leaves a headroom floor for external handles: subtracting
    _INMEM_HEADROOM_BYTES from free before budgeting yields a strictly smaller
    chunk than a hypothetical no-reserve sizer with that much extra free, and
    never underflows below the 64 floor. #76"""
    from gpudge._refpool import (
        _auto_gene_chunk_size_inmem, _INMEM_HEADROOM_BYTES)

    kw = dict(n_ref=77_234, n_genes=18_533, max_group_rows=77_234)
    # Fraction-limited regime (budget < cap) so the reserve actually bites.
    with_headroom = _auto_gene_chunk_size_inmem(free_bytes=40_000_000_000, **kw)
    # A no-reserve sizer would see exactly HEADROOM more free -> the old pick.
    as_if_no_headroom = _auto_gene_chunk_size_inmem(
        free_bytes=40_000_000_000 + _INMEM_HEADROOM_BYTES, **kw)
    assert with_headroom < as_if_no_headroom
    assert with_headroom % 64 == 0

    # Degenerate: free barely covering resident + headroom -> floor 64 (never < 0).
    resident = 77_234 * 18_533 * 4 + 18_533 * 8
    assert _auto_gene_chunk_size_inmem(
        free_bytes=resident + _INMEM_HEADROOM_BYTES + 1, **kw) == 64


def test_inmem_wires_explicit_chunk_and_uploader(monkeypatch, targets_and_ref):
    """inmem_external_ref_de computes the chunk via _auto_gene_chunk_size_inmem,
    builds a _PinnedTileUploader(max_rows, chunk, device), and passes BOTH into
    refpool_de_core. Proven on CPU by faking CUDA + stubbing the CUDA-only leaves.
    #72"""
    targets, ref = targets_and_ref
    import gpudge._refpool as rp
    import torch

    seen = {}

    class _FakeUploader:
        def __init__(self, max_rows, chunk, device):
            seen["uploader_args"] = (max_rows, chunk)

    def _core_stub(**kwargs):
        seen["chunk"] = kwargs.get("gpu_gene_chunk_size")
        seen["uploader"] = kwargs.get("uploader")
        return "ROUTED"

    FREE = 40_000_000_000
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda dev: (FREE, FREE))
    monkeypatch.setattr(rp, "_PinnedTileUploader", _FakeUploader)
    monkeypatch.setattr(rp, "refpool_de_core", _core_stub)
    monkeypatch.setattr(rp, "HAS_NUMBA", True)

    out = gpudge.de(targets, groupby="comparison", reference=ref)
    assert out == "ROUTED"

    n_ref = ref.n_obs
    n_genes = ref.n_vars
    _, counts = np.unique(
        targets.obs["comparison"].to_numpy().astype(str), return_counts=True)
    max_rows = int(counts.max())
    expected_chunk = rp._auto_gene_chunk_size_inmem(
        free_bytes=FREE, n_ref=n_ref, n_genes=n_genes, max_group_rows=max_rows)

    assert seen["chunk"] == expected_chunk
    assert seen["uploader"] is not None
    assert seen["uploader_args"] == (max_rows, expected_chunk)


def test_inmem_explicit_chunk_bypasses_sizer(monkeypatch, targets_and_ref):
    """A user-supplied gpu_gene_chunk_size is passed through unchanged (the
    resident-aware sizer is not consulted). #72"""
    targets, ref = targets_and_ref
    import gpudge._refpool as rp
    import torch

    seen = {}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(rp, "_PinnedTileUploader", lambda *a, **k: object())
    monkeypatch.setattr(rp, "HAS_NUMBA", True)

    def _boom(**kwargs):
        raise AssertionError("sizer must not run when a chunk is supplied")
    monkeypatch.setattr(rp, "_auto_gene_chunk_size_inmem", _boom)

    def _core_stub(**kwargs):
        seen["chunk"] = kwargs.get("gpu_gene_chunk_size")
        return "OK"
    monkeypatch.setattr(rp, "refpool_de_core", _core_stub)

    out = gpudge.de(targets, groupby="comparison", reference=ref,
                    gpu_gene_chunk_size=128)
    assert out == "OK"
    assert seen["chunk"] == 128


def test_inmem_uploader_buffer_width_capped_at_n_genes(monkeypatch, targets_and_ref):
    """A user-pinned gpu_gene_chunk_size far above n_genes must NOT size the
    pinned uploader buffers at the raw chunk (over-allocation of page-locked
    host memory). The buffer width is clamped to n_genes; the chunk passed to
    the core is left unchanged (the gene-chunk loop caps it). #80b"""
    targets, ref = targets_and_ref
    import gpudge._refpool as rp
    import torch

    seen = {}

    class _FakeUploader:
        def __init__(self, max_rows, chunk, device):
            seen["uploader_args"] = (max_rows, chunk)

    def _core_stub(**kwargs):
        seen["core_chunk"] = kwargs.get("gpu_gene_chunk_size")
        return "OK"

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(rp, "_PinnedTileUploader", _FakeUploader)
    monkeypatch.setattr(rp, "refpool_de_core", _core_stub)
    monkeypatch.setattr(rp, "HAS_NUMBA", True)

    n_genes = ref.n_vars                       # 30
    _, counts = np.unique(
        targets.obs["comparison"].to_numpy().astype(str), return_counts=True)
    max_rows = int(counts.max())

    out = gpudge.de(targets, groupby="comparison", reference=ref,
                    gpu_gene_chunk_size=100_000)
    assert out == "OK"
    # buffer width clamped to n_genes; core still receives the raw pinned chunk
    assert seen["uploader_args"] == (max_rows, n_genes)
    assert seen["core_chunk"] == 100_000


def test_inmem_uploader_buffer_width_unchanged_below_n_genes(monkeypatch, targets_and_ref):
    """A pinned chunk below n_genes is used verbatim as the buffer width. #80b"""
    targets, ref = targets_and_ref
    import gpudge._refpool as rp
    import torch

    seen = {}

    class _FakeUploader:
        def __init__(self, max_rows, chunk, device):
            seen["chunk"] = chunk

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(rp, "_PinnedTileUploader", _FakeUploader)
    monkeypatch.setattr(rp, "refpool_de_core", lambda **k: "OK")
    monkeypatch.setattr(rp, "HAS_NUMBA", True)

    gpudge.de(targets, groupby="comparison", reference=ref,
              gpu_gene_chunk_size=8)   # 8 < n_genes(30)
    assert seen["chunk"] == 8


def test_inmem_dense_adata_x_no_uploader_no_crash(monkeypatch, targets_and_ref):
    """A dense adata.X must not crash the uploader guard: ensure_csr passes dense
    through, and the uploader's out= fast path is CSR-only, so the uploader stays
    None and the legacy path handles dense. #72 regression guard."""
    targets, ref = targets_and_ref
    targets = targets.copy()
    targets.X = np.asarray(targets.X.toarray())      # dense adata.X
    import gpudge._refpool as rp
    import torch

    seen = {}

    def _core_stub(**kwargs):
        seen["uploader"] = kwargs.get("uploader")
        seen["chunk"] = kwargs.get("gpu_gene_chunk_size")
        return "OK"

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info",
                        lambda dev: (40_000_000_000, 40_000_000_000))
    monkeypatch.setattr(rp, "refpool_de_core", _core_stub)
    monkeypatch.setattr(rp, "HAS_NUMBA", True)

    out = gpudge.de(targets, groupby="comparison", reference=ref)
    assert out == "OK"
    assert seen["uploader"] is None        # dense -> no pinned uploader
    assert seen["chunk"] is not None       # resident-aware sizer still runs


def test_inmem_reclaims_before_auto_sizing(monkeypatch, targets_and_ref):
    """When auto-sizing, de() reclaims pooled GPU memory (with gc) BEFORE it
    reads free VRAM, so a caller's stale pool can't starve the chunk. #76"""
    targets, ref = targets_and_ref
    import gpudge._refpool as rp
    import torch

    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(rp, "_release_gpu_memory",
                        lambda run_gc=False: calls.append(("reclaim", run_gc)))

    def _meminfo(dev):
        calls.append(("mem_get_info", None))
        return (40_000_000_000, 40_000_000_000)
    monkeypatch.setattr(torch.cuda, "mem_get_info", _meminfo)
    monkeypatch.setattr(rp, "_PinnedTileUploader", lambda *a, **k: object())
    monkeypatch.setattr(rp, "HAS_NUMBA", True)
    monkeypatch.setattr(rp, "refpool_de_core", lambda **k: "OK")

    out = gpudge.de(targets, groupby="comparison", reference=ref)
    assert out == "OK"
    assert ("reclaim", True) in calls
    assert calls.index(("reclaim", True)) < calls.index(("mem_get_info", None))


def test_inmem_explicit_chunk_skips_reclaim(monkeypatch, targets_and_ref):
    """An explicit gpu_gene_chunk_size skips sizing -> skips the reclaim, so a
    caller who wants their pool untouched has an escape hatch. #76"""
    targets, ref = targets_and_ref
    import gpudge._refpool as rp
    import torch

    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(rp, "_release_gpu_memory",
                        lambda run_gc=False: calls.append("reclaim"))
    monkeypatch.setattr(rp, "_PinnedTileUploader", lambda *a, **k: object())
    monkeypatch.setattr(rp, "HAS_NUMBA", True)
    monkeypatch.setattr(rp, "refpool_de_core", lambda **k: "OK")

    out = gpudge.de(targets, groupby="comparison", reference=ref,
                    gpu_gene_chunk_size=128)
    assert out == "OK"
    assert calls == []


def test_de_signature_has_release_gpu_memory_default_true():
    import inspect
    sig = inspect.signature(gpudge.de)
    assert sig.parameters["release_gpu_memory"].default is True


def test_de_releases_gpu_memory_on_exit(monkeypatch, targets_and_ref):
    """de() returns its GPU caches to the driver on a successful return by
    default, and skips it when release_gpu_memory=False. #76"""
    targets, ref = targets_and_ref
    import gpudge._refpool as rp
    import torch

    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(gpudge, "_release_gpu_memory",
                        lambda run_gc=False: calls.append(run_gc))
    monkeypatch.setattr(rp, "inmem_external_ref_de", lambda *a, **k: "DF")

    out = gpudge.de(targets, groupby="comparison", reference=ref)
    assert out == "DF"
    assert calls == [False]           # exit-reclaim once, no gc

    calls.clear()
    out2 = gpudge.de(targets, groupby="comparison", reference=ref,
                     release_gpu_memory=False)
    assert out2 == "DF"
    assert calls == []


# ---------------------------------------------------------------------------
# #79a: in-mem external-ref samples the TARGET adata.X for the raw-counts
# warning (the shared core only samples ref_X). #79b: shared-core empty-ref guard.
# ---------------------------------------------------------------------------
def _stub_cuda_and_core(monkeypatch):
    """Fake CUDA + stub the CUDA-only leaves so inmem_external_ref_de runs its
    CPU-side validation/warning code and returns without a GPU."""
    import gpudge._refpool as rp
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info",
                        lambda dev: (40_000_000_000, 40_000_000_000))
    monkeypatch.setattr(rp, "_PinnedTileUploader", lambda *a, **k: object())
    monkeypatch.setattr(rp, "HAS_NUMBA", True)
    monkeypatch.setattr(rp, "refpool_de_core", lambda **k: "OK")


def test_inmem_external_ref_target_noncount_warns(monkeypatch, targets_and_ref):
    """A fractional target adata.X + an active cpm filter warns about adata.X
    (mirrors the literal-ref in-mem path; #79a)."""
    targets, ref = targets_and_ref
    targets = targets.copy()
    targets.X = (targets.X * 0.5).tocsr()          # fractional -> non-count
    _stub_cuda_and_core(monkeypatch)
    with pytest.warns(UserWarning, match="adata.X does not look like raw counts"):
        gpudge.de(targets, groupby="comparison", reference=ref,
                  filter_gene_min_cpm_cell=1.0)


def test_inmem_external_ref_counts_target_no_noncount_warn(monkeypatch, targets_and_ref):
    """Raw-count target + an active cpm filter must NOT emit the target
    raw-counts warning (proves the gate is non-count-ness, not filter presence)."""
    import warnings as _w
    targets, ref = targets_and_ref                 # _make_synth X is integer counts
    _stub_cuda_and_core(monkeypatch)
    with _w.catch_warnings(record=True) as rec:
        _w.simplefilter("always")
        gpudge.de(targets, groupby="comparison", reference=ref,
                  filter_gene_min_cpm_cell=1.0)
    assert not any("adata.X does not look like raw counts" in str(w.message)
                   for w in rec)


def test_inmem_external_ref_noncount_target_no_filter_silent(monkeypatch, targets_and_ref):
    """A fractional target with NO cpm filter is silent (the warning is scoped
    to the cpm filters' raw-count assumption)."""
    import warnings as _w
    targets, ref = targets_and_ref
    targets = targets.copy()
    targets.X = (targets.X * 0.5).tocsr()
    _stub_cuda_and_core(monkeypatch)
    with _w.catch_warnings(record=True) as rec:
        _w.simplefilter("always")
        gpudge.de(targets, groupby="comparison", reference=ref)   # no cpm filter
    assert not any("adata.X does not look like raw counts" in str(w.message)
                   for w in rec)


def test_refpool_core_empty_reference_raises():
    """The shared core rejects a 0-cell reference (protects BOTH the streaming
    and in-mem external-ref callers; streaming previously returned all-NaN). #79b
    Raises before any CUDA op (explicit chunk -> no mem_get_info)."""
    import scipy.sparse as sp
    import torch
    from gpudge._refpool import refpool_de_core

    ref_X = sp.csr_matrix((0, 5), dtype=np.float32)

    def _src(need_row_sums):
        yield (0, sp.csr_matrix((3, 5), dtype=np.float32),
               np.array([0, 1, 2], dtype=np.int64), None)

    with pytest.raises(ValueError, match="empty|0 cells|non-empty"):
        refpool_de_core(
            ref_X=ref_X, target_source=_src, targets=np.array(["g1"]),
            n_genes=5, var_names=np.array([f"g{i}" for i in range(5)]),
            device=torch.device("cpu"), mean_calc="arithmetic", epsilon=1.0,
            gpu_gene_chunk_size=64, oom_recovery=False, target_sum=None,
            output_columns=None,
            filter_gene_min_mean_value=None, filter_gene_min_total_value=None,
            filter_gene_min_cpm_cell=None, filter_gene_min_cpm_bulk=None,
            keep_genes_arr=None)


def test_inmem_external_ref_xhost_row_sums_scanned_once(monkeypatch, targets_and_ref):
    """csr_row_sums(X_host) is memoized: the target raw-counts warning backstop
    and the per-group row sums share ONE scan of X_host, so a cpm filter on a
    clean count matrix does not force an extra full O(nnz) pass. #79a
    (Copilot review)."""
    import gpudge._refpool as rp
    import torch
    targets, ref = targets_and_ref
    targets = targets.copy()
    targets.X = targets.X.tocsr()             # already CSR -> ensure_csr passthrough
    x_host_id = id(targets.X)

    real = rp.csr_row_sums
    n_xhost = {"n": 0}

    def _counting(X):
        if id(X) == x_host_id:
            n_xhost["n"] += 1
        return real(X)

    captured = {}

    def _core_stub(**kwargs):
        captured["src"] = kwargs["target_source"]
        return "OK"

    monkeypatch.setattr(rp, "csr_row_sums", _counting)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info",
                        lambda dev: (40_000_000_000, 40_000_000_000))
    monkeypatch.setattr(rp, "_PinnedTileUploader", lambda *a, **k: object())
    monkeypatch.setattr(rp, "HAS_NUMBA", True)
    monkeypatch.setattr(rp, "refpool_de_core", _core_stub)

    gpudge.de(targets, groupby="comparison", reference=ref,
              filter_gene_min_cpm_cell=1.0)
    # Drive the target source exactly as refpool_de_core would (a cpm filter is
    # active -> need_row_sums=True).
    list(captured["src"](True))
    assert n_xhost["n"] == 1                   # warning backstop + source share one pass
