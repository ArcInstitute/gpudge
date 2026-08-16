# src/gpudge/_shard_stream.py
"""Native shard-streaming driver for de(shard_archive=...)."""
from __future__ import annotations

import logging
import time
import warnings

import anndata as ad
import numpy as np
import torch

from ._csr_dense import csr_row_sums, csr_rows_col_range_to_dense, ensure_csr
from ._filter import validate_keep_genes
from ._lfc import lfc_column_names
from ._mwu import (
    _tie_term_per_gene, mwu_one_group, mwu_one_group_lfc,
    mwu_one_group_taustar,
)
from ._output import DEFAULT_OUTPUT_COLUMNS
from ._stream import run_gene_chunks_with_recovery
from ._taustar import taustar_column_names

logger = logging.getLogger(__name__)


def _cupy_available() -> bool:
    try:
        import cupy  # noqa: F401
    except Exception:
        return False
    return True


def _x_cupy_available() -> bool:
    """True iff the installed shardad exposes GroupShard.x_cupy (>= 0.5.5)."""
    try:
        from shardad.read import GroupShard
    except Exception:
        return False
    return hasattr(GroupShard, "x_cupy")


def _should_device_decode(arch) -> bool:
    """Device GPU decode (x_cupy) is available iff the archive is x_cupy-capable
    (not v1), cupy is importable, and the installed shardad exposes
    GroupShard.x_cupy. Otherwise the caller falls back to the CPU-parallel
    prefetch path (iter_group_shards + gs.x()).

    x_cupy raises only for v1 archives; it works for both the **v0.5.x single-file
    packed** container (``schema_version == 3`` — the current/default shardad
    format) and legacy v2-directory archives (``schema_version == 2``). Checking
    only ``== 2`` was a bug: real packed archives (3) silently fell back to host,
    so device decode never engaged in production. v1 (``schema_version == 1``) is
    CPU-only.
    """
    return (getattr(arch, "schema_version", None) in (2, 3)
            and _cupy_available() and _x_cupy_available())


def _resolve_streaming(backend, groupby, reference):
    """Resolve groupby + reference mode from the backend.

    Returns (groupby, mode, ref_X, ref_msg_label) where mode is
    "archive_ref" (the archive's own reference pool) or "external_ref"
    (AnnData pool). Layout-agnostic: everything layout-specific lives behind
    ``backend.has_archive_reference`` / ``backend.resolve_archive_reference``.
    """
    arch_groupby = backend.group_by
    if arch_groupby is None:
        raise ValueError(
            "archive was not written with a group_by key (not target-aware). "
            "Re-write with shardad.write_sharded(..., group_by=<obs column>)."
        )
    if groupby is not None and groupby != arch_groupby:
        raise ValueError(
            f"groupby={groupby!r} does not match the archive's group_by="
            f"{arch_groupby!r}."
        )
    groupby = arch_groupby

    if isinstance(reference, ad.AnnData):
        # Mode 2: external reference pool; all archive groups are targets.
        # If the archive ALSO designates its own reference, the external pool
        # wins and the archive's reference is ignored (Semantics A).
        if backend.has_archive_reference:
            warnings.warn(
                "de(archive=..., reference=<AnnData>): the archive designates "
                "its own reference pool, which will be ignored in favor of the "
                "external reference AnnData pool. Omit reference= (or pass "
                "reference=<label>) to use the archive's own reference instead.",
                UserWarning,
                # de() -> stream_de() -> _resolve_streaming() -> warn: 4 points at de().
                stacklevel=4,
            )
        if reference.n_vars != backend.n_vars:
            raise ValueError(
                f"reference AnnData has {reference.n_vars} genes but the archive "
                f"has {backend.n_vars}; the reference and targets must share the "
                "gene axis."
            )
        if not np.array_equal(np.asarray(reference.var_names), backend.var_names):
            raise ValueError(
                "reference AnnData var_names do not match the archive's gene "
                "axis order; align the reference to the archive var_names."
            )
        # Coerce the external pool to canonical CSR, matching the in-mem path.
        # stacklevel=5: de -> stream_de -> _resolve_streaming -> ensure_csr ->
        # warn points at the user's de() call. #79c
        ref_X = ensure_csr(reference.X, name="reference.X", stacklevel=5)
        return groupby, "external_ref", ref_X, "<external AnnData pool>"

    # Mode 1: the archive's own reference pool.
    if not backend.has_archive_reference:
        raise ValueError(
            "archive has no reference. Either supply an external pool via "
            "reference=<AnnData>, or re-write the archive with "
            "shardad.write_sharded(..., reference=<label(s)>)."
        )
    ref_X, msg_label = backend.resolve_archive_reference(groupby, reference)
    return groupby, "archive_ref", ref_X, msg_label


def _iter_kwargs(n_workers, prefetch):
    """Kwargs for ``iter_group_shards()`` on the *decode* passes.

    Only used on the **host** decode path. On the GPU device-decode path
    (``_should_device_decode(arch)`` True) these kwargs are unused: ``x_cupy()``
    requires ``prefetch=0`` and GPU decode replaces CPU-parallel decode-ahead
    entirely, so ``stream_n_workers`` / ``stream_prefetch`` drive only the host
    fallback.

    ``prefetch <= 0`` → ``{}`` (serial; the bare call is byte-identical to
    pre-prefetch shardad — the low-host-RAM fallback). ``prefetch >= 1`` →
    decode ``prefetch`` shards ahead with ``n_workers``-way parallel decode,
    overlapping shard decode with GPU compute.

    Sizing (measured on the full CCL_2 archive, 2026-06-28): peak host RAM is set
    by the **decode batch** — roughly one decoded shard per worker, ~14 GB ×
    ``n_workers`` here (n_workers 4/8/16 → ~75/126/223 GB) — while **prefetch
    depth barely affects throughput** (compute ≈ 1.1 s/shard ≪ ~17 s/shard
    decode, so the bounded queue drains fast). So keep ``prefetch`` shallow
    (default 2) and use ``n_workers`` to trade speed vs host RAM (n_workers
    4/8/16 → ~2.8/3.8/4.7× over serial). ``n_workers`` past the decode-core count
    (≈16 here) stops helping and only adds RAM.

    ``cast_back`` is intentionally left at its ``True`` default. gpudge's
    ``csr_rows_col_range_to_dense`` has no SIMD path for narrow int dtypes
    (~50× slowdown for uint16 vs float32 on CCL_2), and the densifier emits
    float32 regardless, so ``cast_back=False`` would only crush the compute side.
    """
    if prefetch <= 0:
        return {}
    return {"prefetch": prefetch, "n_workers": n_workers}


def _enumerate_targets(arch):
    """Cheap label-only pass over iter_group_shards (no shard data I/O):
    returns the ordered target labels + a label→index map + maximum target
    group size. Labels are disjoint across shards (the planner never splits a
    group), so order is stable."""
    targets: list[str] = []
    max_group_rows = 0
    for gs in arch.iter_group_shards():
        targets.extend(str(lab) for lab in gs.labels)
        for sl in gs.groups.values():
            max_group_rows = max(max_group_rows, sl.stop - sl.start)
    tgt_index = {lab: i for i, lab in enumerate(targets)}
    if len(tgt_index) != len(targets):                 # defensive: should never happen
        raise ValueError("shard_archive has duplicate group labels across shards.")
    return targets, tgt_index, max_group_rows


class _ShardBackend:
    """``layout='shard'`` backend -- the pre-#110 streaming driver, unchanged.

    LEGACY. Shard layout is slated for deprecation; removing it is deleting
    this class and this module's shardad dependency.

    Every decode/reference expression here is lifted from the pre-#110
    ``stream_de``/``_resolve_streaming`` unchanged, so the shard path's OUTPUT
    is byte-identical (proven by the gate, and by
    ``test_stream_de_makes_the_expected_archive_passes``). It is not a literal
    verbatim move: ``_should_device_decode`` and ``_iter_kwargs`` now run at
    construction rather than after reference resolution, so the log line and
    the device-selection point moved earlier. Neither can raise, and the median
    value sequence, the 2-vs-3 shard-pass counts, the single ``read_reference()``
    and the ``del Xs, Ls`` hygiene are all preserved -- but claim
    output-identity, not expression-identity.
    """

    def __init__(self, arch, *, n_workers, prefetch):
        self._arch = arch
        self._iter_kwargs = _iter_kwargs(n_workers, prefetch)
        self.n_vars = int(arch.n_vars)
        self.var_names = np.asarray(arch.var.index)
        self.group_by = arch.manifest.get("group_by")
        self.has_archive_reference = arch.manifest.get("reference_shard") is not None
        self.supports_device_decode = _should_device_decode(arch)
        self._targets = None
        self._tgt_index = None
        self._max_group_rows = None
        if self.supports_device_decode:
            logger.info("shard-streaming: GPU device decode (x_cupy, prefetch disabled)")
        else:
            logger.info("shard-streaming: host CSR decode (n_workers=%d, prefetch=%d)",
                        n_workers, prefetch)

    def close(self):
        # ShardedArchive exposes no close(); dropping the reference is all there
        # is to do. Present so both backends satisfy one contract.
        self._arch = None

    def _iter(self):
        # Device decode requires prefetch=0 (x_cupy raises on a pre-decoded
        # shard), so the kwargs drive only the host fallback -- verbatim from
        # the pre-#110 driver.
        return (self._arch.iter_group_shards() if self.supports_device_decode
                else self._arch.iter_group_shards(**self._iter_kwargs))

    def resolve_archive_reference(self, groupby, reference):
        ref_adata = self._arch.read_reference()        # read once; reused by Phase 0
        if ref_adata is None:
            raise ValueError(
                "archive designates a reference shard but reading it returned "
                "None; the archive's manifest and reference shard are inconsistent "
                "(possibly corrupted)."
            )
        if groupby not in ref_adata.obs:
            raise ValueError(
                f"groupby column {groupby!r} not found in the reference shard's obs "
                f"(available: {list(ref_adata.obs.columns)})."
            )
        ref_col = ref_adata.obs[groupby]
        n_missing = int(ref_col.isna().sum())
        if n_missing:
            raise ValueError(
                f"the reference shard's obs[{groupby!r}] has {n_missing} cell(s) "
                f"with a missing (NaN/None) label; .astype(str) would turn them "
                f"into a bogus 'nan'/'None' reference label. Re-write the archive "
                f"without unassigned reference cells. (mirrors the _ingest guard)"
            )
        ref_label_set = set(np.asarray(ref_col).astype(str).tolist())
        if reference is not None and str(reference) not in ref_label_set:
            raise ValueError(
                f"reference={reference!r} is not among the archive's reference "
                f"labels {sorted(ref_label_set)}."
            )
        msg_label = (str(reference) if reference is not None
                     else "|".join(sorted(ref_label_set)))
        return ref_adata.X, msg_label

    def targets(self):
        if self._tgt_index is None:
            (self._targets, self._tgt_index,
             self._max_group_rows) = _enumerate_targets(self._arch)
        return list(self._targets), self._max_group_rows

    def target_row_sums(self):
        sums = []
        for gs in self._iter():
            if self.supports_device_decode:
                from ._csr_dense_gpu import cupy_csr_row_sums
                Xs = gs.x_cupy()                       # device cupy CSR
                sums.append(cupy_csr_row_sums(Xs))     # host f64, bit-identical
            else:
                Xs = gs.x()                            # host CSR (pre-decoded when prefetch>0)
                sums.append(csr_row_sums(Xs))
            del Xs
        logger.info("median pre-pass: %d shards", len(sums))
        return (np.concatenate(sums) if sums
                else np.zeros(0, dtype=np.float64))

    def target_source(self, need_row_sums):
        # Outer: guide-shards. Device path decodes each shard on the GPU (x_cupy,
        # prefetch=0); host path uses CPU-parallel decode-ahead + gs.x(). Yields
        # absolute target index g so the core writes accumulators by index.
        # `del Xs, Ls` per shard releases the shard before the next decodes.
        self.targets()                                 # ensure _tgt_index
        for gs in self._iter():
            if self.supports_device_decode:
                from ._csr_dense_gpu import cupy_csr_row_sums
                Xs = gs.x_cupy()                 # device cupy CSR
                Ls = cupy_csr_row_sums(Xs) if need_row_sums else None
            else:
                Xs = gs.x()                      # host CSR; pre-decoded when prefetch>0
                Ls = csr_row_sums(Xs) if need_row_sums else None
            for label, sl in gs.groups.items():
                rows = np.arange(sl.start, sl.stop, dtype=np.int64)
                yield (self._tgt_index[str(label)], Xs, rows,
                       (Ls[rows] if Ls is not None else None))
            del Xs, Ls


def _free_gpu_bytes(device) -> int:
    free, _ = torch.cuda.mem_get_info(device)
    return int(free)


def _reference_prepass(ref_X, n_genes, device, chunk, *, mean_calc,
                       scale_main, scale_num, need_other_unit, need_row_sums,
                       need_row_scales, oom_recovery):
    """Read the reference once, sort per gene-chunk, keep the sorted form +
    tie terms resident on the GPU; compute reference means."""
    n_ref = int(ref_X.shape[0])

    # Oversized guard: the resident sorted reference is dense n_ref×n_genes f32.
    projected = n_ref * n_genes * 4
    free = _free_gpu_bytes(device)
    if projected >= free:
        raise RuntimeError(
            f"reference pool is too large to keep resident on the GPU: "
            f"n_ref={n_ref} × n_genes={n_genes} × 4 B = {projected/1e9:.1f} GB "
            f"vs ~{free/1e9:.1f} GB free. Reduce the reference pool size or use "
            f"a larger-memory GPU."
        )

    # Per-cell library sizes / scales for CPM (computed once on the reference).
    ref_row_sums = csr_row_sums(ref_X) if need_row_sums else None
    if need_row_scales:
        safe = np.where(ref_row_sums == 0, 1.0, ref_row_sums)
        ref_scales = torch.from_numpy((scale_num / safe).astype(np.float32)).to(device)
    else:
        ref_scales = None
    ref_libtot = float(ref_row_sums.sum()) if ref_row_sums is not None else None

    sorted_ref_full = torch.empty((n_genes, n_ref), dtype=torch.float32, device=device)
    ref_tie_term_full = torch.empty(n_genes, dtype=torch.float64, device=device)
    arith_ref = np.zeros(n_genes, dtype=np.float64)
    ref_mean = np.zeros(n_genes, dtype=np.float64)
    other_ref = np.zeros(n_genes, dtype=np.float64) if need_other_unit else None
    all_rows = np.arange(n_ref, dtype=np.int64)

    def _chunk(start, stop):
        dense = csr_rows_col_range_to_dense(ref_X, all_rows, start, stop)  # (n_ref, w) f32
        t = torch.from_numpy(dense).to(device)                            # UNSCALED
        # Other-unit ref mean (no scaling) only when a cpm-decoupled filter needs it.
        if other_ref is not None:
            if scale_main:
                other_ref[start:stop] = t.to(torch.float64).mean(dim=0).cpu().numpy()
            else:
                sc = ref_scales.unsqueeze(1).to(torch.float64)
                other_ref[start:stop] = (t.to(torch.float64) * sc).mean(dim=0).cpu().numpy()
        if scale_main:
            t = t.mul_(ref_scales.unsqueeze(1))
        tf64 = t.to(torch.float64)
        a = tf64.mean(dim=0).cpu().numpy()
        arith_ref[start:stop] = a
        if mean_calc == "geometric":
            ref_mean[start:stop] = torch.expm1(
                torch.log1p(tf64).mean(dim=0)).cpu().numpy()
        else:
            ref_mean[start:stop] = a
        st = torch.sort(t.T.contiguous(), dim=1).values                   # (w, n_ref)
        sorted_ref_full[start:stop] = st
        ref_tie_term_full[start:stop] = _tie_term_per_gene(st)

    final_chunk = run_gene_chunks_with_recovery(
        n_genes, chunk, _chunk, oom_recovery=oom_recovery)
    return dict(sorted_ref_full=sorted_ref_full, ref_tie_term_full=ref_tie_term_full,
                ref_mean=ref_mean, arith_ref=arith_ref, other_ref=other_ref,
                n_ref=n_ref, ref_libtot=ref_libtot, ref_row_sums=ref_row_sums,
                final_chunk=final_chunk)


def group_chunk_stats(group_t, sorted_ref_chunk, ref_tie_chunk, n_ref, *,
                      mean_calc, scale_main, group_scales=None,
                      want_other=False, lfc_combos=None,
                      taustar_levels=None, taustar_iters=None,
                      taustar_se=False):
    """One target group's gene-chunk stats. ``group_t`` (m, chunk) f32 is the
    group's dense slice, UNSCALED; it is scaled **in place** iff
    ``scale_main``. ``group_scales`` (m,) = numerator/L is required when
    scale_main, or for the scaled OTHER-unit mean when not scale_main.
    Returns (arith, reported, other|None, u1, p, dir_u1|None, dir_p|None,
    taustar|None). ``arith``/``reported``/``other``/``u1``/``p`` are (chunk,) f64
    torch tensors ON DEVICE; callers move them to host. ``other`` is None unless
    ``want_other``; ``dir_u1``/``dir_p`` are (n_combos, chunk) f64 on device and
    are None unless ``lfc_combos`` is given; ``taustar`` is
    (len(taustar_column_names(taustar_levels, taustar_se)), chunk) f64 on device
    -- i.e. ``len(taustar_levels) + 3`` when ``taustar_se``, since the SE block
    appends lo/hi/se ROWS -- and is None unless ``taustar_levels`` is given.
    Mirrors the
    in-memory ref-mode body exactly (unscaled other-unit, scale-in-place, f64
    mean + f32 MWU)."""
    other = None
    if want_other:
        un64 = group_t.to(torch.float64)                   # from the UNSCALED tensor
        if scale_main:
            other = un64.mean(dim=0)                        # other unit = unscaled
        else:
            sc = group_scales.unsqueeze(1).to(torch.float64)
            other = (un64 * sc).mean(dim=0)                 # other unit = scaled
        del un64
    if scale_main:
        group_t = group_t.mul_(group_scales.unsqueeze(1))   # scale in place
    xf64 = group_t.to(torch.float64)                        # X-units (scaled iff main)
    arith = xf64.mean(dim=0)
    if mean_calc == "geometric":
        reported = torch.expm1(torch.log1p(xf64).mean(dim=0))
    else:
        reported = arith
    group_tt = group_t.T.contiguous()
    dir_u1 = dir_p = taustar = None
    if lfc_combos is not None:
        u1, p, dir_u1, dir_p = mwu_one_group_lfc(
            sorted_ref_chunk, ref_tie_chunk, group_tt,
            n_ref=n_ref, lfc_combos=lfc_combos)
        if taustar_levels is not None:
            # Both features on: the base test IS computed twice. Unavoidable
            # without merging the two kernels, and it is the rarer
            # configuration -- tau* answers the question the grid was being
            # used to approximate, so callers usually want one or the other.
            _u1_ts, _p_ts, taustar = mwu_one_group_taustar(
                sorted_ref_chunk, ref_tie_chunk, group_tt,
                n_ref=n_ref, taustar_levels=taustar_levels,
                taustar_iters=taustar_iters, taustar_se=taustar_se)
    elif taustar_levels is not None:
        # tau* ONLY -- the common configuration, and the one dge_robust runs.
        # Use the tau* kernel as the BASE provider instead of paying for a
        # separate mwu_one_group pass: its u1/p are bit-identical by
        # construction (same helpers, same layout, same summation order),
        # pinned by
        # test_mwu_taustar.py::test_base_columns_are_bit_identical_to_mwu_one_group.
        u1, p, taustar = mwu_one_group_taustar(
            sorted_ref_chunk, ref_tie_chunk, group_tt,
            n_ref=n_ref, taustar_levels=taustar_levels,
            taustar_iters=taustar_iters, taustar_se=taustar_se)
    else:
        u1, p = mwu_one_group(sorted_ref_chunk, ref_tie_chunk, group_tt,
                              n_ref=n_ref)
    return arith, reported, other, u1, p, dir_u1, dir_p, taustar


def stream_de(archive, *, groupby, reference, mean_calc, epsilon,
              gpu_gene_chunk_size, oom_recovery, cpm_normalize,
              normalize_target_sum, output_columns, lfc_combos=None,
              taustar_levels=None, taustar_iters=None, taustar_se=False,
              filter_gene_min_mean_value, filter_gene_min_total_value,
              filter_gene_min_cpm_cell, filter_gene_min_cpm_bulk,
              keep_genes, stream_n_workers, stream_prefetch, device):
    # Archive-free parameter checks first, so a bad call fails fast without
    # opening the archive (matches de()'s validate-before-dispatch intent and
    # keeps stream_de() safe to call directly).
    if output_columns is not None:
        if not output_columns:
            raise ValueError(
                "output_columns must be a non-empty dict mapping default column "
                "names to output names, or None (got an empty dict).")
        allowed = tuple(DEFAULT_OUTPUT_COLUMNS) + tuple(
            (lfc_column_names(lfc_combos) if lfc_combos else [])
            + (taustar_column_names(taustar_levels, taustar_se)
               if taustar_levels else []))
        unknown = [k for k in output_columns if k not in allowed]
        if unknown:
            raise KeyError(
                f"output_columns keys not in de() output schema: {unknown}. "
                f"Valid keys: {list(allowed)}")
        dests = list(output_columns.values())
        if len(set(dests)) != len(dests):
            raise ValueError(
                f"output_columns maps multiple keys to the same name: {dests}.")
    if mean_calc not in ("arithmetic", "geometric"):
        raise ValueError(f"mean_calc must be 'arithmetic' or 'geometric', got {mean_calc!r}.")
    if not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError(f"epsilon must be a finite value >= 0, got {epsilon!r}.")
    # Mirror de()'s guard so a direct stream_de() call also fails fast (and never
    # feeds a bad value to _iter_kwargs / shardad).
    for _name, _val in (("stream_n_workers", stream_n_workers),
                        ("stream_prefetch", stream_prefetch)):
        if not isinstance(_val, (int, np.integer)) or isinstance(_val, bool):
            raise TypeError(f"{_name} must be an int, got {type(_val).__name__}.")
    if stream_n_workers < 1:
        raise ValueError(f"stream_n_workers must be >= 1, got {stream_n_workers}.")
    if stream_prefetch < 0:
        raise ValueError(
            f"stream_prefetch must be >= 0 (0 disables prefetch), got {stream_prefetch}.")

    from ._stream_backend import open_backend
    backend = open_backend(archive, n_workers=stream_n_workers,
                           prefetch=stream_prefetch)
    try:
        n_genes = backend.n_vars
        var_names = backend.var_names

        groupby, mode, ref_X, _ = _resolve_streaming(backend, groupby, reference)

        keep_genes_arr = (validate_keep_genes(keep_genes, n_genes)
                          if keep_genes is not None else None)

        from ._normalize import resolve_target_sum
        _median_requested = (isinstance(normalize_target_sum, str)
                             and normalize_target_sum == "median")
        if _median_requested:
            # Up-front pass over the reference + every target group to gather
            # per-cell library sizes, so the global median target can be computed
            # before any scaling. ~2x archive I/O, only when 'median' is requested.
            _t0 = time.perf_counter()
            _row_sums_all = np.concatenate(
                [csr_row_sums(ref_X), backend.target_row_sums()])  # reference stays host
            logger.info("median pre-pass: %.1fs", time.perf_counter() - _t0)
        else:
            _row_sums_all = None
        target_sum = resolve_target_sum(
            cpm_normalize=cpm_normalize,
            normalize_target_sum=normalize_target_sum, row_sums=_row_sums_all)
        targets, max_group_rows = backend.targets()

        from ._refpool import refpool_de_core
        return refpool_de_core(
            ref_X=ref_X, target_source=backend.target_source,
            targets=np.asarray(targets), n_genes=n_genes, var_names=var_names,
            device=device, mean_calc=mean_calc, epsilon=epsilon,
            gpu_gene_chunk_size=gpu_gene_chunk_size, oom_recovery=oom_recovery,
            target_sum=target_sum, output_columns=output_columns,
            lfc_combos=lfc_combos, max_group_rows=max_group_rows,
            taustar_levels=taustar_levels, taustar_iters=taustar_iters,
            taustar_se=taustar_se,
            filter_gene_min_mean_value=filter_gene_min_mean_value,
            filter_gene_min_total_value=filter_gene_min_total_value,
            filter_gene_min_cpm_cell=filter_gene_min_cpm_cell,
            filter_gene_min_cpm_bulk=filter_gene_min_cpm_bulk,
            keep_genes_arr=keep_genes_arr, warn_noncount=True)
    finally:
        # Never let a close failure MASK the real error. CellStore.close()
        # raises BufferError while an mmap view is still exported, so an
        # unguarded close() in finally would replace a genuine DE traceback
        # with a confusing cleanup one. A failed close costs a leaked mapping
        # for the life of the process -- worth a warning, never worth losing
        # the exception that actually explains the failure.
        try:
            backend.close()
        except Exception:                       # pragma: no cover - defensive
            logger.warning("failed to close the streaming backend", exc_info=True)


def _streaming_keep_mask(n_targets, n_genes, chunk, arith_target_acc, arith_ref,
                         other_target_acc, other_ref, target_counts, n_ref,
                         group_libtot, ref_libtot, target_sum,
                         min_mean, min_total, min_cpm_cell, min_cpm_bulk,
                         keep_genes_arr):
    """Final chunked filter pass. Builds a per-target group_libtot-style layout
    that matches the in-memory ref-mode helper, where the reference is appended
    as the last row so _refmode_chunk_keep's ref_label_idx points at it.

    The reference's own appended row is necessarily self-referential under any
    filter (it is compared against itself — e.g. for min_cpm_bulk its bulk-CPM
    equals the reference scalar it is tested against), so its keep decision is
    meaningless; it is discarded by the final ``out[:n_targets]`` slice and
    never affects a real target row (the filter is independent per row)."""
    from . import _refmode_chunk_keep
    if (min_mean is None and min_total is None and min_cpm_cell is None
            and min_cpm_bulk is None and keep_genes_arr is None):
        return np.ones((n_targets, n_genes), dtype=bool)
    # Stack reference as the final row (index n_targets) so the existing helper —
    # which expects (n_groups, n_genes) target accumulators + a reference row —
    # can be reused verbatim with ref_label_idx = n_targets.
    n_groups = n_targets + 1
    at = np.vstack([arith_target_acc, arith_ref[None, :]])
    counts = np.concatenate([target_counts, [n_ref]])
    ot = (np.vstack([other_target_acc, other_ref[None, :]])
          if other_target_acc is not None else None)
    if group_libtot is not None:
        glib = np.concatenate([group_libtot, [ref_libtot]])
    else:
        glib = None
    out = np.zeros((n_groups, n_genes), dtype=bool)
    for start in range(0, n_genes, chunk):
        stop = min(start + chunk, n_genes)
        out[:, start:stop] = _refmode_chunk_keep(
            start, stop, stop - start, at, arith_ref, ot, other_ref,
            counts, n_targets, glib, target_sum,
            min_mean, min_total, min_cpm_cell, min_cpm_bulk, keep_genes_arr)
    return out[:n_targets]                       # drop the synthetic reference row
