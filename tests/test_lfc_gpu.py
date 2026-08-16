# tests/test_lfc_gpu.py
"""GPU gate: directional results must be bit-identical across the in-memory
explicit-reference paths (literal ref-mode vs external reference AnnData),
exactly like the existing base-column parity gate. Streaming parity lives in
tests/test_shard_stream.py, next to the archive fixtures.

Every assert_frame_equal here passes check_exact=True: polars defaults to
check_exact=False with rel_tol=1e-5 (verified, polars 1.41), which would turn
these bit-parity gates into approximate ones.
"""
import numpy as np
import pytest
from polars.testing import assert_frame_equal

from conftest import LFC_TAUS, needs_cuda
from gpudge import de


def _split(seed=1, n_cells=400, n_genes=16, n_guides=4, sparse=False):
    import anndata as ad
    import scipy.sparse as sp
    rng = np.random.default_rng(seed)
    X = rng.negative_binomial(2, 0.3, (n_cells, n_genes)).astype(np.float32)
    if sparse:
        X = sp.csr_matrix(X)
    lab = np.array(["ntc"] * (n_cells // 2) +
                   [f"g{i % n_guides}" for i in range(n_cells - n_cells // 2)])
    rng.shuffle(lab)
    a = ad.AnnData(X=X, obs={"grp": lab})
    a.var_names = [f"f{i}" for i in range(n_genes)]
    tgt = a[a.obs["grp"] != "ntc"].copy()
    ref = a[a.obs["grp"] == "ntc"].copy()
    return a, tgt, ref


@needs_cuda
def test_inmem_refmode_matches_external_reference():
    a, tgt, ref = _split()
    x = de(a, groupby="grp", reference="ntc",
           lfc_threshold=LFC_TAUS).sort(["target", "feature"])
    y = de(tgt, groupby="grp", reference=ref,
           lfc_threshold=LFC_TAUS).sort(["target", "feature"])
    assert_frame_equal(x, y, check_dtypes=True, check_exact=True)


@needs_cuda
def test_grid_result_equals_per_tau_results():
    a, _, _ = _split()
    grid = de(a, groupby="grp", reference="ntc", lfc_threshold=LFC_TAUS)
    for t in LFC_TAUS:
        one = de(a, groupby="grp", reference="ntc", lfc_threshold=[t])
        cols = [c for c in one.columns if c.startswith("tau=")]
        assert_frame_equal(grid.select(cols), one.select(cols),
                           check_dtypes=True, check_exact=True)


@needs_cuda
def test_multi_chunk_matches_single_chunk():
    """Forces >1 gene chunk so the per-chunk [start:stop] accumulator writes
    are exercised. NOTE: both legs pin gpu_gene_chunk_size, so this does NOT
    exercise the auto-sizer -- that is the next test's job."""
    a, _, _ = _split(n_genes=64)
    one = de(a, groupby="grp", reference="ntc", lfc_threshold=LFC_TAUS,
             gpu_gene_chunk_size=64)
    many = de(a, groupby="grp", reference="ntc", lfc_threshold=LFC_TAUS,
              gpu_gene_chunk_size=8, oom_recovery=False)
    assert_frame_equal(one, many, check_dtypes=True, check_exact=True)


@needs_cuda
def test_auto_sizer_receives_max_group_rows_and_shrinks(monkeypatch):
    """The sizer test that actually tests the sizer.

    de() reaches _auto_gene_chunk_size only after its CUDA guard, so this is a
    needs_cuda test -- but the pure sizer ARITHMETIC is already covered on CPU
    in tests/test_stream.py; here we prove the THREADING: that de() actually
    hands the directional path a nonzero max_group_rows and n_combos, and that
    those change the chosen chunk.

    A plain end-to-end run cannot catch an under-budgeted sizer on a CI-sized
    fixture (n_genes caps the chunk long before the budget bites), so force a
    small deterministic `free`, capture de()'s call, and compare the budgeted
    result against the unbudgeted one.
    """
    import gpudge
    import torch
    seen = {}
    real = gpudge._auto_gene_chunk_size

    def _spy(**kw):
        seen.update(kw)
        return real(**kw)

    monkeypatch.setattr(gpudge, "_auto_gene_chunk_size", _spy)
    monkeypatch.setattr(torch.cuda, "mem_get_info",
                        lambda *a, **k: (512 * 1024**2, 512 * 1024**2))

    # small ref, big groups -> target-dominant. n_genes=400 keeps the fixture
    # cheap while staying discriminating (budgeted 320 vs unbudgeted 384).
    a, _, _ = _split(n_cells=6200, n_genes=400, n_guides=2)
    # ntc is half the cells; the two guide groups are ~1550 each.
    de(a, groupby="grp", reference="ntc", lfc_threshold=LFC_TAUS)
    assert seen["n_combos"] == len(LFC_TAUS) * 2
    assert seen["max_group_rows"] == 1550
    budgeted = real(**seen)
    unbudgeted = real(**{**seen, "n_combos": 0, "max_group_rows": 0})
    assert (budgeted, unbudgeted) == (320, 384)


@needs_cuda
def test_auto_sized_grid_completes_without_oom_recovery():
    """Companion end-to-end check: an auto-sized directional run with
    oom_recovery=False must not raise, and must match a pinned-chunk run."""
    import anndata as ad
    rng = np.random.default_rng(9)
    n_ref, n_tgt, g = 200, 6000, 400
    X = rng.negative_binomial(2, 0.3, (n_ref + n_tgt, g)).astype(np.float32)
    lab = np.array(["ntc"] * n_ref + ["t0"] * (n_tgt // 2) + ["t1"] * (n_tgt - n_tgt // 2))
    a = ad.AnnData(X=X, obs={"grp": lab})
    a.var_names = [f"f{i}" for i in range(g)]
    got = de(a, groupby="grp", reference="ntc", lfc_threshold=LFC_TAUS,
             oom_recovery=False)                     # no gpu_gene_chunk_size
    ref = de(a, groupby="grp", reference="ntc", lfc_threshold=LFC_TAUS,
             gpu_gene_chunk_size=64)
    assert_frame_equal(got.sort(["target", "feature"]),
                       ref.sort(["target", "feature"]),
                       check_dtypes=True, check_exact=True)


@needs_cuda
def test_no_mixed_dtype_searchsorted_is_ever_issued(monkeypatch):
    """Spec 3.2b/5.17b. torch.searchsorted with a float32 boundary and float64
    values UPCASTS AND COPIES the whole boundary (ATen BucketizationUtils:
    raw_boundaries.to(common_stype)) -- 0.05 ms -> 65.26 ms against a 160 MB
    reference. The _bounds helper exists to avoid that; this test proves the
    directional path actually uses it.

    A memory-delta test is NOT a substitute: on any CI-sized fixture the
    legitimate directional buffers already exceed the would-be float64 reference
    copy, so it could not fail.
    """
    import torch
    real = torch.searchsorted

    def _guard(sorted_sequence, input, **kw):
        # RAISE, do not record-and-assert: the native mixed-dtype path is the
        # expensive thing we are forbidding, so do not execute it.
        if sorted_sequence.dtype != input.dtype:
            raise AssertionError(
                f"mixed-dtype searchsorted ({sorted_sequence.dtype} boundary, "
                f"{input.dtype} values): torch upcasts and COPIES the whole "
                "float32 reference there (ATen BucketizationUtils). Use _bounds.")
        return real(sorted_sequence, input, **kw)

    monkeypatch.setattr(torch, "searchsorted", _guard)
    a, _, _ = _split()
    de(a, groupby="grp", reference="ntc", lfc_threshold=LFC_TAUS)


@needs_cuda
def test_external_reference_base_columns_unchanged():
    """Release gate on the external-reference driver (the literal-reference
    driver is covered in tests/test_lfc_integration.py)."""
    _, tgt, ref = _split()
    base = de(tgt, groupby="grp", reference=ref)
    with_t = de(tgt, groupby="grp", reference=ref, lfc_threshold=LFC_TAUS)
    assert_frame_equal(with_t.select(base.columns), base,
                       check_dtypes=True, check_exact=True)


@needs_cuda
def test_pinned_uploader_directional_matches_legacy_branch(monkeypatch):
    """Cover the SECOND Phase-1 branch of `_accumulate_target_group`.

    That wrapper has two branches and the pinned-uploader one carries its OWN
    directional accumulator code — device-to-device ``dir_u_dev[:, start:stop]``
    writes plus one batched D2H per group — distinct from the legacy branch's
    per-chunk ``dir_U_acc[:, g, start:stop] = …cpu()``. It engages only when
    numba is available AND the target X is host CSR, so every other test here
    (dense fixtures) takes the legacy branch and would ship the pinned
    directional code untested. Force both branches over the SAME sparse input:
    the result must be bit-identical.

    `_refpool.HAS_NUMBA = False` disables only the uploader construction; the
    densify still resolves numba from `_csr_dense`'s own namespace, so the two
    legs differ in the accumulator path and nothing else.
    """
    from gpudge import _refpool
    if not _refpool.HAS_NUMBA:
        pytest.skip("pinned uploader requires the [fast] numba extra")

    _, tgt, ref = _split(seed=13, sparse=True)
    assert tgt.X.format == "csr" and ref.X.format == "csr"
    keys = ["target", "feature"]
    pinned = de(tgt, groupby="grp", reference=ref,
                lfc_threshold=LFC_TAUS).sort(keys)

    monkeypatch.setattr(_refpool, "HAS_NUMBA", False)
    legacy = de(tgt, groupby="grp", reference=ref,
                lfc_threshold=LFC_TAUS).sort(keys)

    assert_frame_equal(pinned, legacy, check_dtypes=True, check_exact=True)
