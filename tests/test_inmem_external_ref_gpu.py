# tests/test_inmem_external_ref_gpu.py
"""GPU bit-identity gate: in-memory de(reference=<AnnData>) == streaming Mode-2
de(shard_archive=..., reference=<AnnData>) on the SAME cells + same reference.

Bit-identity is the hard merge gate: both paths run the identical
refpool_de_core, so the only way they can differ is target cell VALUES or
within-group cell ORDER. We eliminate both by reconstructing the in-memory
targets AnnData directly from the archive (ad.concat over the group shards), so
the in-memory group cell order equals the archive's group cell order and every
float64 reduction is bit-for-bit identical.
"""
from __future__ import annotations

import numpy as np
import pytest
import anndata as ad

import gpudge
from conftest import _make_synth, needs_cuda   # bare import — CI gotcha

shardad = pytest.importorskip("shardad", reason="requires gpudge[streaming]")
from polars.testing import assert_frame_equal   # noqa: E402

# check_exact=True is NOT optional in this file. polars' assert_frame_equal
# defaults to check_exact=False with rel_tol=1e-5 / abs_tol=1e-8, which silently
# turned this module's "bit-identity merge gate" (see the docstring above) into a
# tolerance check — a 1-float32-ULP relative drift passed green at every
# magnitude. Third occurrence of this trap in the repo (#110 was the first).
# check_dtypes / check_row_order / check_column_order already default True; they
# are named here so the intent survives a future polars default change.
# (ultrareview 2026-08)
EXACT = dict(check_exact=True, check_column_order=True,
             check_row_order=True, check_dtypes=True)


def _build(tmp_path, *, seed):
    """A guides-only archive (targets) + a separate NTC reference AnnData +
    the in-memory targets AnnData RECONSTRUCTED from the archive (guarantees
    identical within-group cell order)."""
    full = _make_synth(n_cells=600, n_genes=40, n_guides=8, sparse=True, seed=seed)
    is_ntc = np.char.startswith(
        full.obs["comparison"].to_numpy().astype(str), "ntc")
    guides = full[~is_ntc].copy()
    ref = full[is_ntc].copy()
    d = str(tmp_path / "m2")
    shardad.write_sharded(guides, d, format="v2", group_by="comparison",
                          reference=None, target_shard_bytes=4096)
    arch = shardad.ShardedArchive(d)
    inmem = ad.concat([gs.to_anndata() for gs in arch.iter_group_shards()],
                      axis=0)
    # var axis must round-trip identically (gene-axis check + parity rely on it)
    assert list(inmem.var_names) == list(ref.var_names)
    return d, inmem, ref


_KWSWEEP = [
    dict(),
    dict(cpm_normalize=True),
    dict(normalize_target_sum=5e5),
    dict(normalize_target_sum="median"),
    dict(filter_gene_min_mean_value=0.5),
    dict(filter_gene_min_total_value=5.0),
    dict(filter_gene_min_cpm_cell=1.0),
    dict(filter_gene_min_cpm_bulk=1.0),
    dict(mean_calc="geometric"),
]


@needs_cuda
@pytest.mark.parametrize(
    "kw", _KWSWEEP,
    ids=[",".join(f"{k}={v}" for k, v in d.items()) or "default" for d in _KWSWEEP])
def test_inmem_external_ref_bit_identical_to_streaming(tmp_path, kw):
    d, inmem, ref = _build(tmp_path, seed=21)
    df_stream = gpudge.de(shard_archive=d, reference=ref, **kw)
    df_inmem = gpudge.de(inmem, groupby="comparison", reference=ref, **kw)
    keys = ["target", "feature"]
    assert_frame_equal(df_stream.sort(keys), df_inmem.sort(keys), **EXACT)


@needs_cuda
def test_inmem_external_ref_keep_genes_bit_identical(tmp_path):
    d, inmem, ref = _build(tmp_path, seed=22)
    keep = np.zeros(ref.n_vars, dtype=bool)
    keep[::2] = True
    keys = ["target", "feature"]
    df_stream = gpudge.de(shard_archive=d, reference=ref, keep_genes=keep)
    df_inmem = gpudge.de(inmem, groupby="comparison", reference=ref,
                         keep_genes=keep)
    assert_frame_equal(df_stream.sort(keys), df_inmem.sort(keys), **EXACT)


@needs_cuda
@pytest.mark.parametrize(
    "kw", _KWSWEEP,
    ids=[",".join(f"{k}={v}" for k, v in d.items()) or "default" for d in _KWSWEEP])
def test_inmem_external_ref_bit_identical_small_chunk(tmp_path, kw):
    """Force a multi-chunk run (tiny gpu_gene_chunk_size, 40 genes -> 12,12,12,4)
    so the in-mem uploader's double-buffer alternation + trailing short chunk are
    exercised, and assert it stays byte-identical to streaming (legacy path)
    Mode-2. #72"""
    d, inmem, ref = _build(tmp_path, seed=25)
    keys = ["target", "feature"]
    df_stream = gpudge.de(shard_archive=d, reference=ref,
                          gpu_gene_chunk_size=12, **kw)
    df_inmem = gpudge.de(inmem, groupby="comparison", reference=ref,
                         gpu_gene_chunk_size=12, **kw)
    assert_frame_equal(df_stream.sort(keys), df_inmem.sort(keys), **EXACT)


@needs_cuda
def test_inmem_external_ref_output_columns_bit_identical(tmp_path):
    d, inmem, ref = _build(tmp_path, seed=23)
    oc = {"target": "guide", "feature": "gene", "p_value": "p", "p_adj": "padj"}
    keys = ["guide", "gene"]
    df_stream = gpudge.de(shard_archive=d, reference=ref, output_columns=oc)
    df_inmem = gpudge.de(inmem, groupby="comparison", reference=ref,
                         output_columns=oc)
    assert_frame_equal(df_stream.sort(keys), df_inmem.sort(keys), **EXACT)


@needs_cuda
def test_inmem_external_ref_csc_ref_bit_identical_to_streaming(tmp_path):
    """A CSC external reference coerces to CSR on BOTH paths -> still
    bit-identical streaming-vs-in-mem. The CSR-only _KWSWEEP never exercised a
    CSC external ref; this is the #79c parity merge gate."""
    d, inmem, ref = _build(tmp_path, seed=26)
    ref_csc = ref.copy()
    ref_csc.X = ref_csc.X.tocsc()
    keys = ["target", "feature"]
    df_stream = gpudge.de(shard_archive=d, reference=ref_csc)
    df_inmem = gpudge.de(inmem, groupby="comparison", reference=ref_csc)
    assert_frame_equal(df_stream.sort(keys), df_inmem.sort(keys), **EXACT)


@needs_cuda
def test_inmem_external_ref_csc_matches_csr(tmp_path):
    """CSC adata.X AND CSC reference.X coerce to CSR (one warning each) and give
    bit-identical output to the CSR-input run. External-ref path is
    NON-mutating: the caller's AnnData formats are left unchanged. #66"""
    import warnings as _w
    _, inmem, ref = _build(tmp_path, seed=24)   # archive dir unused (in-memory only)
    df_csr = gpudge.de(inmem, groupby="comparison", reference=ref)
    inmem_csc = inmem.copy()
    inmem_csc.X = inmem_csc.X.tocsc()
    ref_csc = ref.copy()
    ref_csc.X = ref_csc.X.tocsc()
    with _w.catch_warnings(record=True) as rec:
        _w.simplefilter("always")
        df_csc = gpudge.de(inmem_csc, groupby="comparison", reference=ref_csc)
    coerce = [w for w in rec if "converting to CSR" in str(w.message)]
    assert len(coerce) == 2                        # adata.X + reference.X
    keys = ["target", "feature"]
    assert_frame_equal(df_csr.sort(keys), df_csc.sort(keys), **EXACT)
    assert inmem_csc.X.format == "csc"             # NON-mutating
    assert ref_csc.X.format == "csc"


@needs_cuda
def test_pinned_tile_uploader_matches_plain_densify():
    """The uploader returns tiles bit-identical to a plain densify+H2D, across
    >2 tiles (buffer alternation) and a trailing short chunk. #72"""
    import torch
    import scipy.sparse as sp
    from gpudge._refpool import _PinnedTileUploader
    from gpudge._csr_dense import csr_rows_col_range_to_dense

    full = _make_synth(n_cells=200, n_genes=50, n_guides=4, sparse=True, seed=7)
    X = sp.csr_matrix(full.X)                       # canonical CSR
    rows = np.arange(30, dtype=np.int64)            # a target-group-like row subset
    device = torch.device("cuda")
    chunk = 16                                      # 50 genes -> 4 tiles (16,16,16,2)
    up = _PinnedTileUploader(max_rows=int(rows.size), chunk=chunk, device=device)

    for start in range(0, X.shape[1], chunk):
        stop = min(start + chunk, X.shape[1])
        got = up.upload(X, rows, start, stop)
        want = torch.from_numpy(
            csr_rows_col_range_to_dense(X, rows, start, stop)).to(device)
        torch.cuda.synchronize()
        assert got.shape == want.shape
        assert torch.equal(got, want)               # bit-identical


@needs_cuda
def test_release_gpu_memory_toggle(tmp_path):
    """de(release_gpu_memory=True) returns torch's caching-allocator reserve to
    the driver; False keeps it resident. Uses torch.cuda.memory_reserved() (the
    torch cache high-water) so the assertion is size-independent. #76"""
    import torch
    _, inmem, ref = _build(tmp_path, seed=41)   # shard archive path unused here

    torch.cuda.empty_cache()
    gpudge.de(inmem, groupby="comparison", reference=ref,
              release_gpu_memory=False)
    reserved_hold = torch.cuda.memory_reserved()

    gpudge.de(inmem, groupby="comparison", reference=ref,
              release_gpu_memory=True)
    reserved_release = torch.cuda.memory_reserved()

    assert reserved_hold > 0                    # cache retained when not releasing
    assert reserved_release < reserved_hold     # empty_cache returned it to driver


@needs_cuda
def test_inmem_external_ref_large_pinned_chunk_bit_identical(tmp_path):
    """Mode-2 (in-mem external ref): a pinned gpu_gene_chunk_size >> n_genes
    (uploader buffer clamped to n_genes) yields the SAME frame as the auto-sized
    run. Proves the clamped uploader still packs/uploads tiles correctly. #80b"""
    d, inmem, ref = _build(tmp_path, seed=27)   # n_genes == 40
    keys = ["target", "feature"]
    df_auto = gpudge.de(inmem, groupby="comparison", reference=ref)
    df_big = gpudge.de(inmem, groupby="comparison", reference=ref,
                       gpu_gene_chunk_size=100_000)
    assert_frame_equal(df_auto.sort(keys), df_big.sort(keys), **EXACT)


@needs_cuda
def test_mode1_literal_ref_large_pinned_chunk_bit_identical():
    """Mode-1 (literal reference, inline group_host_bufs): a pinned
    gpu_gene_chunk_size >> n_genes (buffer clamped to n_genes) yields the SAME
    frame as the auto-sized run. Exercises the __init__.py pinned-buffer path.
    #80b"""
    adata = _make_synth(n_cells=600, n_genes=40, n_guides=8, sparse=True, seed=28)
    keys = ["target", "feature"]
    df_auto = gpudge.de(adata, groupby="comparison", reference="ntc")
    df_big = gpudge.de(adata, groupby="comparison", reference="ntc",
                       gpu_gene_chunk_size=100_000)
    assert_frame_equal(df_auto.sort(keys), df_big.sort(keys), **EXACT)
