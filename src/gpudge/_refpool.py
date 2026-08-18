# src/gpudge/_refpool.py
"""Shared reference-pool differential-expression core.

Three callers run this identical core, differing ONLY in where the target groups
come from: the streaming driver (de(archive=...), _shard_stream.stream_de /
_cell_stream) off archive shards, the in-memory external-reference path
(de(adata=..., reference=<AnnData>), inmem_external_ref_de) off adata groups,
and the bring-your-own cell source (de(cell_source=...), cell_source_de) off
whatever the caller yields. Bit-parity between all three is therefore by
construction.
"""
from __future__ import annotations

import warnings

import numpy as np
import polars as pl
import torch
from scipy.sparse import issparse

from ._cell_source import _check_2d, make_target_source
from ._csr_dense import (
    HAS_NUMBA,
    csr_row_sums,
    csr_rows_col_range_to_dense,
    ensure_csr,
    is_cupy_csr,
)
from ._fdr import bh_per_group
from ._filter import _row_scale_needs, validate_keep_genes, x_has_noncount_signal
from ._gpu_mem import _release_gpu_memory
from ._ingest import ALL_OTHERS, ingest
from ._lfc import lfc_base_names, lfc_column_names
from ._normalize import resolve_target_sum
from ._output import (
    DEFAULT_OUTPUT_COLUMNS,
    assemble_dataframe,
    effect_size_from_u,
    empty_output_frame,
    log2_ratio,
)
from ._shard_stream import (
    _reference_prepass,
    _streaming_keep_mask,
    group_chunk_stats,
)
from ._stream import (
    _LFC_TILE_BYTES,
    _TAUSTAR_TILE_BYTES,
    _auto_gene_chunk_size,
    _pinned_buf_width,
    run_gene_chunks_with_recovery,
)
from ._taustar import taustar_column_names


# In-memory external-ref chunk sizing. Distinct from _stream._auto_gene_chunk_size
# (which is shared with streaming and stays unchanged): the in-mem path holds the
# reference resident (sorted_ref_full = n_ref x n_genes f32, ~5.7 GB on CCL_2) for
# the whole run, and its Phase-1 target tile is what the group loop stresses. So
# reserve the resident reference out of free BEFORE budgeting, and size the
# per-gene transient on the larger of the Phase-0 reference sort and the Phase-1
# target tile. The peak transient is ~= budget (chunk = budget / bytes_per_gene),
# so budget + resident must fit in free; the fraction-of-available term keeps that
# safe as GPUs/references shrink, and oom_recovery is the backstop. #72.
#
# Budget = min(FRACTION x available, CAP). Raising both terms (fraction 0.35->0.5,
# cap 16->32 GiB) lets the chunk use more of a large GPU's free memory: on the
# 5.54M-cell CCL_2 shape (#72, ~80 GiB H100) it moves ~2304 -> ~4608, ~halving the
# gene-chunk count and recovering the H2D-overlap win the old cap left on the table
# (1.38x vs the ~1.57x possible at a larger chunk). Peak transient ~= budget +
# resident (~37 GiB here), well within the card. On a small GPU the CAP never binds
# (FRACTION x available stays below it), so it remains fraction-limited and cannot
# over-provision — the fraction bump does enlarge its chunk too, but peak scales
# with its own free memory. oom_recovery remains the backstop. #74-followup.
_INMEM_RESERVE_FRACTION = 0.5      # of free AFTER reserving the resident reference
_INMEM_BUDGET_CAP = 32 * 1024**3   # transient budget ceiling (peak ~= this + resident)
_INMEM_REF_SORT_BYTES = 40         # Phase-0 per ref-cell/gene: f32 in + f64 copy
                                   # + f32 sorted values + int64 sort idx + margin
_INMEM_TILE_BYTES = 96             # Phase-1 per target-cell/gene: f32 tile +
                                   # the ~10 (m, chunk) f64/int64 MWU working arrays
_INMEM_HEADROOM_BYTES = 1 * 1024**3   # reserved out of free for external handles
                                      # (cuBLAS et al.) + peak-model slack; #76


def _auto_gene_chunk_size_inmem(
    *, free_bytes, n_ref, n_genes, max_group_rows, n_combos=0, n_levels=0
):
    """Pick a gene-chunk size for the in-memory external-ref path (pure; GPU-free).

    Reserves a fixed ``_INMEM_HEADROOM_BYTES`` floor (for external handles /
    peak-model slack) and the resident sorted reference (n_ref x n_genes f32 +
    n_genes f64 tie terms) out of ``free_bytes`` first, then budgets
    ``_INMEM_RESERVE_FRACTION`` of what remains (capped at ``_INMEM_BUDGET_CAP``).
    ``bytes_per_gene`` is the max of
    the Phase-0 reference-sort transient (n_ref cells) and the Phase-1 target-tile
    transient (max_group_rows cells). Floored at 64, then capped at n_genes and
    (when the result is >= 64) rounded down to a multiple of 64 — so a
    smaller-than-64 n_genes (only in tests) returns n_genes, EXCEPT n_genes == 0,
    which returns 1 so the chunk driver no-ops instead of raising about
    ``initial_chunk``.

    ``n_levels`` is the tau* accumulator ROW count, matching
    ``_stream._auto_gene_chunk_size``: callers pass
    ``len(levels) + 3*taustar_se``, since the SE block appends lo/hi/se rows.
    """
    resident = n_ref * n_genes * 4 + n_genes * 8
    # Per-group directional accumulators are (n_combos, n_genes) f64 x 2 --
    # chunk-INDEPENDENT, so reserve them out of free like `resident`.
    resident += 2 * n_combos * n_genes * 8
    # tau*'s per-group accumulator is (n_levels, n_genes) f64 -- ONE, not the
    # lfc U/p pair, hence the factor 1. Chunk-INDEPENDENT, so reserved out of
    # free like `resident`.
    resident += 1 * n_levels * n_genes * 8
    available = max(0, int(free_bytes) - _INMEM_HEADROOM_BYTES - resident)
    budget = min(int(available * _INMEM_RESERVE_FRACTION), _INMEM_BUDGET_CAP)
    tile_bytes = (_INMEM_TILE_BYTES
                  + (_LFC_TILE_BYTES if n_combos else 0)
                  + (_TAUSTAR_TILE_BYTES if n_levels else 0))
    bytes_per_gene = max(n_ref * _INMEM_REF_SORT_BYTES,
                         # + the kernel's own (n_combos, chunk) f64 dir_u1/dir_p,
                         # (16 B/gene/combo) and (n_levels, chunk) f64 taustar
                         # (8 B/gene/level), live for their whole loop.
                         max_group_rows * tile_bytes
                         + 16 * n_combos + 8 * n_levels, 1)
    chunk = max(64, budget // bytes_per_gene)
    # See the twin floor in `_stream._auto_gene_chunk_size`: only n_genes == 0
    # can drive this to 0, and a 0 makes the chunk driver raise about an
    # internal parameter name instead of returning the empty frame.
    chunk = max(1, min(int(chunk), n_genes))
    if chunk >= 64:
        chunk = (chunk // 64) * 64
    return chunk


class _PinnedTileUploader:
    """Double-buffered pinned-host arena for async H2D of in-memory target tiles.

    Mirrors the Mode-1 arena (__init__.py:919-1059), transposed to the refpool
    (outer=groups) nesting: two pinned host buffers alternate per tile so the
    next tile's CSR densify + async H2D overlaps the current tile's GPU compute.
    Per-buffer CUDA events guard reuse. Only constructed by the in-memory caller
    when numba+CSR is available; the streaming caller never passes one. #72.
    """

    def __init__(self, max_rows, chunk, device):
        self._bufs = [
            torch.empty(max_rows, chunk, dtype=torch.float32, pin_memory=True)
            for _ in range(2)
        ]
        self._bufs_np = [b.numpy() for b in self._bufs]
        self._events = [torch.cuda.Event(), torch.cuda.Event()]
        self._device = device
        self._i = 0

    def upload(self, X_source, rows, start, stop):
        m = int(rows.shape[0])
        ch = int(stop) - int(start)
        k = self._i % 2
        # Wait for the PREVIOUS H2D queued from this buffer (no-op the first two
        # calls: synchronize() on a never-recorded event returns immediately).
        self._events[k].synchronize()
        # Contiguous (m, ch) view of the buffer's flat prefix. A plain
        # buf[:m, :ch] view is non-contiguous when ch < chunk (trailing tile /
        # post-downshift), which makes .to(device, non_blocking=True) fall back to
        # a synchronous staged copy; the flat-prefix reshape keeps it async.
        # m*ch <= max_rows*chunk always fits, and the event above guards reuse.
        packed = self._bufs_np[k].reshape(-1)[:m * ch].reshape(m, ch)
        dense = csr_rows_col_range_to_dense(X_source, rows, start, stop, out=packed)
        t = torch.from_numpy(dense).to(self._device, non_blocking=True)
        self._events[k].record()
        self._i += 1
        return t


def _accumulate_target_group(
    g, X_source, rows, Ls_for_rows, chunk, *,
    n_genes, device, sorted_ref_full, ref_tie_term_full, n_ref,
    mean_calc, scale_main, scale_num, need_row_scales,
    target_mean_acc, arith_target_acc, other_target_acc,
    U_acc, p_acc, target_counts, group_libtot, oom_recovery,
    uploader=None,
    lfc_combos=None, dir_U_acc=None, dir_p_acc=None,
    taustar_levels=None, taustar_iters=None, taustar_se=False,
    taustar_acc=None,
):
    """One target group's Phase-1 accumulation (densify -> group_chunk_stats ->
    write accumulators by absolute index), driven under the OOM-recovery loop.

    A mechanical lift of the streaming Phase-1 per-group body (previously inlined
    in stream_de, now shared), generalized over (X_source, rows, Ls-for-rows) so
    both the streaming and in-memory target sources reuse it verbatim. Returns
    the (possibly downshifted) chunk so the driver carries a downshift forward.

    When ``uploader`` is supplied (in-memory external-ref path only) AND the
    source is host CSR, each tile is uploaded via a double-buffered pinned async
    H2D and the per-gene-chunk results accumulate device-to-device into per-group
    (n_genes,) tensors, copied to host ONCE per group. This removes the per-tile
    ``.cpu()`` sync that otherwise serializes densify -> H2D -> compute (#72).
    Bit-identical to the legacy path: same math, same absolute-index writes.
    """
    target_counts[g] = rows.size
    if group_libtot is not None:
        group_libtot[g] = float(Ls_for_rows.sum())
    grp_scales_t = None
    if need_row_scales:
        safe = np.where(Ls_for_rows == 0, 1.0, Ls_for_rows)
        grp_scales_t = torch.from_numpy(
            (scale_num / safe).astype(np.float32)).to(device)

    want_other = other_target_acc is not None
    n_combos = 0 if lfc_combos is None else len(lfc_combos)
    n_taustar_rows = (0 if taustar_levels is None
                      else len(taustar_column_names(taustar_levels, taustar_se)))

    # Optimized in-memory path: async pinned H2D + per-group device accumulators
    # with ONE batched D2H per accumulator at group end. Only when the caller
    # injected an uploader AND this source is host CSR (never the device-cupy
    # source, which densifies on-GPU with no H2D — the uploader would add nothing).
    if uploader is not None and not is_cupy_csr(X_source):
        geom = mean_calc == "geometric"
        arith_dev = torch.zeros(n_genes, dtype=torch.float64, device=device)
        reported_dev = (torch.zeros(n_genes, dtype=torch.float64, device=device)
                        if geom else arith_dev)
        other_dev = (torch.zeros(n_genes, dtype=torch.float64, device=device)
                     if want_other else None)
        u_dev = torch.zeros(n_genes, dtype=torch.float64, device=device)
        p_dev = torch.zeros(n_genes, dtype=torch.float64, device=device)
        if lfc_combos is not None:
            dir_u_dev = torch.zeros(
                (n_combos, n_genes), dtype=torch.float64, device=device)
            dir_p_dev = torch.zeros(
                (n_combos, n_genes), dtype=torch.float64, device=device)
        if taustar_levels is not None:
            taustar_dev = torch.zeros(
                (n_taustar_rows, n_genes), dtype=torch.float64, device=device)

        def _chunk(start, stop):
            gt = uploader.upload(X_source, rows, start, stop)   # (m, w) f32, async H2D
            (arith, reported, other, u1, p, dir_u1, dir_p,
             taustar) = group_chunk_stats(
                 gt, sorted_ref_full[start:stop],
                 ref_tie_term_full[start:stop],
                 n_ref, mean_calc=mean_calc, scale_main=scale_main,
                 group_scales=grp_scales_t, want_other=want_other,
                 lfc_combos=lfc_combos, taustar_levels=taustar_levels,
                 taustar_iters=taustar_iters, taustar_se=taustar_se)
            arith_dev[start:stop] = arith            # device-to-device (no sync)
            if geom:
                reported_dev[start:stop] = reported
            if other_dev is not None:
                other_dev[start:stop] = other
            u_dev[start:stop] = u1
            p_dev[start:stop] = p
            if lfc_combos is not None:
                dir_u_dev[:, start:stop] = dir_u1
                dir_p_dev[:, start:stop] = dir_p
            if taustar_levels is not None:
                taustar_dev[:, start:stop] = taustar

        final = run_gene_chunks_with_recovery(
            n_genes, chunk, _chunk, oom_recovery=oom_recovery)
        # One batched D2H per accumulator. When not geometric, reported_dev
        # aliases arith_dev (identical values), so reuse the single arith host
        # copy for target_mean_acc rather than paying a redundant second D2H.
        arith_host = arith_dev.cpu().numpy()
        arith_target_acc[g] = arith_host
        target_mean_acc[g] = reported_dev.cpu().numpy() if geom else arith_host
        if other_dev is not None:
            other_target_acc[g] = other_dev.cpu().numpy()
        U_acc[g] = u_dev.cpu().numpy()
        p_acc[g] = p_dev.cpu().numpy()
        if lfc_combos is not None:
            dir_U_acc[:, g, :] = dir_u_dev.cpu().numpy()
            dir_p_acc[:, g, :] = dir_p_dev.cpu().numpy()
        if taustar_levels is not None:
            taustar_acc[:, g, :] = taustar_dev.cpu().numpy()
        return final

    # Legacy path (streaming host-CSR + device-cupy sources): unchanged.
    # Pick the densify backend ONCE per group (X_source is fixed across this
    # group's gene-chunks): device cupy CSR (shardad x_cupy — on-device densify,
    # zero-copy to torch, no H2D) vs host scipy CSR (numba densify + H2D). Keeps
    # the per-gene-chunk hot loop free of a repeated predicate + module import.
    if is_cupy_csr(X_source):
        from ._csr_dense_gpu import cupy_csr_rows_col_range_to_torch

        def _densify(start, stop):
            return cupy_csr_rows_col_range_to_torch(X_source, rows, start, stop)
    else:
        def _densify(start, stop):
            dense = csr_rows_col_range_to_dense(X_source, rows, start, stop)  # (m, w) f32
            return torch.from_numpy(dense).to(device)

    def _chunk(start, stop):
        gt = _densify(start, stop)                          # (m, w) f32 UNSCALED, on device
        (arith, reported, other, u1, p, dir_u1, dir_p,
         taustar) = group_chunk_stats(
             gt, sorted_ref_full[start:stop], ref_tie_term_full[start:stop],
             n_ref, mean_calc=mean_calc, scale_main=scale_main,
             group_scales=grp_scales_t, want_other=want_other,
             lfc_combos=lfc_combos, taustar_levels=taustar_levels,
             taustar_iters=taustar_iters, taustar_se=taustar_se)
        arith_target_acc[g, start:stop] = arith.cpu().numpy()
        target_mean_acc[g, start:stop] = reported.cpu().numpy()
        if other_target_acc is not None:
            other_target_acc[g, start:stop] = other.cpu().numpy()
        U_acc[g, start:stop] = u1.cpu().numpy()
        p_acc[g, start:stop] = p.cpu().numpy()
        if lfc_combos is not None:
            dir_U_acc[:, g, start:stop] = dir_u1.cpu().numpy()
            dir_p_acc[:, g, start:stop] = dir_p.cpu().numpy()
        if taustar_levels is not None:
            taustar_acc[:, g, start:stop] = taustar.cpu().numpy()

    return run_gene_chunks_with_recovery(
        n_genes, chunk, _chunk, oom_recovery=oom_recovery)


def refpool_de_core(
    *,
    ref_X,
    target_source,
    targets,
    n_genes,
    var_names,
    device,
    mean_calc,
    epsilon,
    gpu_gene_chunk_size,
    oom_recovery,
    target_sum,
    output_columns,
    filter_gene_min_mean_value,
    filter_gene_min_total_value,
    filter_gene_min_cpm_cell,
    filter_gene_min_cpm_bulk,
    keep_genes_arr,
    warn_noncount=True,
    uploader=None,
    lfc_combos=None,
    taustar_levels=None,
    taustar_iters=None,
    taustar_se=False,
    max_group_rows=0,
):
    """Reference-pool DE: reference prepass (resident-sorted on GPU) -> iterate
    target groups from ``target_source`` ranking each vs the resident reference
    -> Phase-2 filter/assemble/BH-FDR. See module docstring.

    ``target_source(need_row_sums)`` yields ``(g, X_source, rows, Ls_for_rows)``
    per target group; ``targets[g]`` is that group's output label. ``target_sum``
    is the already-resolved normalization target (or None). Bit-identical across
    callers by construction.
    """
    _scale_main = target_sum is not None
    _scale_num = target_sum if _scale_main else 1.0e6

    _cpm_filter_active = (filter_gene_min_cpm_cell is not None
                          or filter_gene_min_cpm_bulk is not None)
    _count_filter_active = (filter_gene_min_mean_value is not None
                            or filter_gene_min_total_value is not None
                            or filter_gene_min_cpm_bulk is not None)
    need_unscaled_extra = _count_filter_active and _scale_main
    need_scaled_extra = (filter_gene_min_cpm_cell is not None) and not _scale_main
    need_other_unit = need_unscaled_extra or need_scaled_extra
    # median is already resolved into target_sum by the caller; pass
    # median_requested=False (scale_main already forces need_row_sums when a
    # numeric/median target is active).
    need_row_sums, need_row_scales = _row_scale_needs(
        _scale_main, filter_gene_min_cpm_cell, filter_gene_min_cpm_bulk,
        median_requested=False)

    n_targets = len(targets)
    if n_targets == 0:
        # Canonical typed, output_columns-aware empty frame (same schema as a
        # non-empty result). See _output.empty_output_frame.
        return empty_output_frame(
            output_columns,
            extra_names=((lfc_column_names(lfc_combos) if lfc_combos else [])
                         + (taustar_column_names(taustar_levels, taustar_se)
                            if taustar_levels else [])) or None)

    # Empty reference pool ⇒ MWU is all-NaN. Guard in the SHARED core so BOTH
    # callers fail loudly (streaming previously returned a silent all-NaN
    # DataFrame; in-mem already raises in de(), this backstops it). #79b
    if int(ref_X.shape[0]) == 0:
        raise ValueError(
            "reference pool is empty (0 cells); the external reference "
            "(AnnData pool or archive reference shard) must be non-empty.")

    # Chunk size: budget on the reference cell count (ref-mode basis), n_groups=1
    # (Phase 1 holds no (n_groups, chunk) GPU accumulators — group_chunk_stats
    # returns per-group (chunk,) tensors copied straight to host).
    n_combos = 0 if lfc_combos is None else len(lfc_combos)
    n_taustar_rows = (0 if taustar_levels is None
                      else len(taustar_column_names(taustar_levels, taustar_se)))
    if gpu_gene_chunk_size is None:
        free, _ = torch.cuda.mem_get_info(device)
        gpu_gene_chunk_size = _auto_gene_chunk_size(
            free_bytes=free, budget_n=int(ref_X.shape[0]), n_groups=1,
            mean_calc=mean_calc, n_genes=n_genes, ref_mode=True,
            n_combos=n_combos, n_levels=n_taustar_rows,
            max_group_rows=max_group_rows)
    chunk = gpu_gene_chunk_size

    # CPM non-count warning (once), sampling the reference X. Streaming's target
    # shards are always raw uint16 counts, so ref_X is the only realizable
    # non-count input there; kept identical here so both callers warn the same.
    if warn_noncount and _cpm_filter_active and x_has_noncount_signal(ref_X):
        warnings.warn(
            "reference X does not look like raw counts (non-integer or negative "
            "values); the filter_gene_min_cpm_* filters assume raw counts.",
            UserWarning, stacklevel=2)

    # ---- Phase 0: reference resident-sorted on GPU ----
    ref = _reference_prepass(
        ref_X, n_genes, device, chunk, mean_calc=mean_calc,
        scale_main=_scale_main, scale_num=_scale_num,
        need_other_unit=need_other_unit, need_row_sums=need_row_sums,
        need_row_scales=need_row_scales, oom_recovery=oom_recovery)
    sorted_ref_full = ref["sorted_ref_full"]
    ref_tie_term_full = ref["ref_tie_term_full"]
    n_ref = ref["n_ref"]
    # Carry Phase 0's (possibly OOM-downshifted) chunk into Phase 1.
    chunk = ref["final_chunk"]

    # ---- accumulators (target-indexed) ----
    target_mean_acc = np.zeros((n_targets, n_genes), dtype=np.float64)
    arith_target_acc = np.zeros((n_targets, n_genes), dtype=np.float64)
    other_target_acc = (np.zeros((n_targets, n_genes), dtype=np.float64)
                        if need_other_unit else None)
    U_acc = np.zeros((n_targets, n_genes), dtype=np.float64)
    p_acc = np.ones((n_targets, n_genes), dtype=np.float64)
    dir_U_acc = (np.zeros((n_combos, n_targets, n_genes), dtype=np.float64)
                 if n_combos else None)
    dir_p_acc = (np.ones((n_combos, n_targets, n_genes), dtype=np.float64)
                 if n_combos else None)
    taustar_acc = (np.empty((n_taustar_rows, n_targets, n_genes), dtype=np.float64)
                   if taustar_levels is not None else None)
    target_counts = np.zeros(n_targets, dtype=np.int64)
    group_libtot = (np.zeros(n_targets, dtype=np.float64)
                    if filter_gene_min_cpm_bulk is not None else None)

    # ---- Phase 1: iterate target groups, rank each vs the resident reference ----
    for g, X_source, rows, Ls_for_rows in target_source(need_row_sums):
        chunk = _accumulate_target_group(
            g, X_source, rows, Ls_for_rows, chunk,
            n_genes=n_genes, device=device, sorted_ref_full=sorted_ref_full,
            ref_tie_term_full=ref_tie_term_full, n_ref=n_ref,
            mean_calc=mean_calc, scale_main=_scale_main, scale_num=_scale_num,
            need_row_scales=need_row_scales,
            target_mean_acc=target_mean_acc, arith_target_acc=arith_target_acc,
            other_target_acc=other_target_acc, U_acc=U_acc, p_acc=p_acc,
            target_counts=target_counts, group_libtot=group_libtot,
            oom_recovery=oom_recovery, uploader=uploader,
            lfc_combos=lfc_combos, dir_U_acc=dir_U_acc,
            dir_p_acc=dir_p_acc, taustar_levels=taustar_levels,
            taustar_iters=taustar_iters, taustar_se=taustar_se,
            taustar_acc=taustar_acc)
        # Drop this iteration's refs so the target source's per-shard `del`
        # (streaming) releases the shard before the next decode. (memory hygiene)
        del X_source, rows, Ls_for_rows

    # ---- Phase 2: filters -> assemble -> BH-FDR ----
    # See `_output.log2_ratio`: epsilon=0 is documented to yield NaN / +/-inf,
    # so that divide must not warn (or raise under -W error) -- but a negative
    # mean is undefined rather than documented and keeps its warning.
    log2fc = log2_ratio(target_mean_acc, ref["ref_mean"], epsilon)

    keep_mask_acc = _streaming_keep_mask(
        n_targets, n_genes, chunk, arith_target_acc, ref["arith_ref"],
        other_target_acc, ref["other_ref"], target_counts, n_ref,
        group_libtot, ref["ref_libtot"], target_sum,
        filter_gene_min_mean_value, filter_gene_min_total_value,
        filter_gene_min_cpm_cell, filter_gene_min_cpm_bulk, keep_genes_arr)

    extra_columns = None
    if lfc_combos is not None:
        extra_columns = {}
        for k, (p_name, u_name, _q_name) in enumerate(lfc_base_names(lfc_combos)):
            extra_columns[p_name] = dir_p_acc[k]
            extra_columns[u_name] = effect_size_from_u(
                dir_U_acc[k], target_counts, int(n_ref))
    if taustar_levels is not None:
        extra_columns = extra_columns or {}
        for k, name in enumerate(
                taustar_column_names(taustar_levels, taustar_se)):
            extra_columns[name] = taustar_acc[k]

    df = assemble_dataframe(
        target=np.asarray(targets), feature=var_names,
        target_mean=target_mean_acc, ref_mean=ref["ref_mean"],
        target_ncells=target_counts, ref_ncells=int(n_ref),
        log2_fold_change=log2fc, p_value=p_acc,
        test_statistic=effect_size_from_u(U_acc, target_counts, n_ref),
        p_adj=np.zeros_like(p_acc), flat_keep=keep_mask_acc.ravel(),
        extra_columns=extra_columns,
        output_columns=None)

    if df.height > 0:
        post_filter_g = np.nonzero(keep_mask_acc)[0]
        g_idx = torch.from_numpy(post_filter_g)
        adj = bh_per_group(df["p_value"].to_torch(), g_idx, n_targets)
        new_cols = [pl.Series("p_adj", adj.numpy())]
        if lfc_combos is not None:
            # Each (tau, direction) is its own family of hypotheses.
            for p_name, _u_name, q_name in lfc_base_names(lfc_combos):
                q = bh_per_group(df[p_name].to_torch(), g_idx, n_targets)
                new_cols.append(pl.Series(q_name, q.numpy()))
        df = df.with_columns(new_cols)
    elif lfc_combos is not None:
        # ZERO ROWS still needs every directional p_adj to EXIST and be typed,
        # or the schema differs from a populated result and from
        # empty_output_frame.
        df = df.with_columns([
            pl.Series(q_name, [], dtype=pl.Float64)
            for _p, _u, q_name in lfc_base_names(lfc_combos)
        ])

    # Final explicit projection pins the CANONICAL column order. Without it the
    # p_adj columns land after ALL the p/Ueffect columns, giving
    # p1,es1,p2,es2,q1,q2 instead of the required p1,es1,q1,p2,es2,q2.
    if lfc_combos is not None or taustar_levels is not None:
        df = df.select(
            list(DEFAULT_OUTPUT_COLUMNS)
            + (lfc_column_names(lfc_combos) if lfc_combos else [])
            + (taustar_column_names(taustar_levels, taustar_se)
               if taustar_levels else []))

    if output_columns is None:
        return df
    return df.select(list(output_columns)).rename(output_columns)


def inmem_external_ref_de(
    adata, *, groupby, reference,
    mean_calc, epsilon, gpu_gene_chunk_size, oom_recovery,
    cpm_normalize, normalize_target_sum, output_columns, lfc_combos=None,
    taustar_levels=None, taustar_iters=None, taustar_se=False,
    filter_gene_min_mean_value, filter_gene_min_total_value,
    filter_gene_min_cpm_cell, filter_gene_min_cpm_bulk,
    keep_genes, device,
):
    """In-memory external reference pool: rank every group in
    ``adata.obs[groupby]`` against the separate ``reference`` AnnData pool,
    resident-sorted on GPU with NO target-reference concat. Runs the identical
    ``refpool_de_core`` as the streaming Mode-2 path -> bit-identical results.

    Gene-axis / densify_input / empty-reference validation is performed by the
    caller (``de()``) before the CUDA check so it fails fast on CPU.
    """
    n_genes = int(adata.n_vars)
    var_names = adata.var_names.to_numpy()
    # Non-mutating coercion (this path never mutates the caller and rejects
    # densify_input): a non-CSR sparse .X would silently take the scipy slow
    # path in the numba kernels. One UserWarning per non-CSR matrix. #66
    # stacklevel=4: de() -> inmem_external_ref_de -> ensure_csr -> warn points at
    # the user's de() call.
    ref_X = ensure_csr(reference.X, name="reference.X", stacklevel=4)
    X_host = ensure_csr(adata.X, name="adata.X", stacklevel=4)

    # csr_row_sums(X_host) can be needed by up to three consumers below (the
    # target raw-counts warning backstop, the 'median' pre-pass, and the
    # per-group library sizes in _inmem_target_source). Memoize so X_host is
    # scanned at most ONCE regardless of how many are active — a cpm filter on a
    # clean count matrix must not force an extra full O(nnz) pass on top of the
    # one the target source already needs. #79a
    _xhost_row_sums_cache = {}

    def _x_host_row_sums():
        if "v" not in _xhost_row_sums_cache:
            _xhost_row_sums_cache["v"] = csr_row_sums(X_host)
        return _xhost_row_sums_cache["v"]

    # Raw-counts / non-count warning for the cpm_* filters — TARGET side. The
    # shared core samples ref_X; the streaming caller's target shards are
    # provably uint16 counts, so the target sample belongs here (matches the
    # literal-reference in-mem path in de()). #79a
    _cpm_filter_active = (filter_gene_min_cpm_cell is not None
                          or filter_gene_min_cpm_bulk is not None)
    if _cpm_filter_active:
        _noncount = x_has_noncount_signal(X_host)
        if not _noncount:
            _noncount = bool((_x_host_row_sums() < 0).any())
        if _noncount:
            warnings.warn(
                "adata.X does not look like raw counts (non-integer or negative "
                "values); the filter_gene_min_cpm_* filters assume raw counts. "
                "If X is not counts, pass a precomputed keep_genes mask instead.",
                # de() -> inmem_external_ref_de -> warn: 3 points at the user's de().
                UserWarning, stacklevel=3)

    keep_genes_arr = (validate_keep_genes(keep_genes, n_genes)
                      if keep_genes is not None else None)

    # Median pre-pass over the reference cells + all target cells (adata holds
    # only targets here). Mirrors the streaming median pre-pass over ref_X + all
    # shard cells; median over the union is order-independent.
    _median_requested = (isinstance(normalize_target_sum, str)
                         and normalize_target_sum == "median")
    if _median_requested:
        _row_sums_all = np.concatenate(
            [csr_row_sums(ref_X), _x_host_row_sums()])
    else:
        _row_sums_all = None
    target_sum = resolve_target_sum(
        cpm_normalize=cpm_normalize,
        normalize_target_sum=normalize_target_sum, row_sums=_row_sums_all)

    # Encode groups (EVERY group is a target; none excluded). Reuse ingest with
    # ALL_OTHERS purely for its groupby validation (column present, no NaN
    # labels) + label encoding; ref_label_idx is unused here.
    state = ingest(adata, groupby=groupby, reference=ALL_OTHERS)
    unique_labels = state.unique_labels
    labels = state.labels
    n_targets = len(unique_labels)
    # Per-group absolute row indices via one stable argsort + boundary split
    # (== flatnonzero per group, ascending). Same trick as the in-memory ref-mode
    # loop in de().
    _order = np.argsort(labels, kind="stable")
    _bounds = np.searchsorted(labels[_order], np.arange(n_targets + 1))
    group_to_rows = [_order[_bounds[g]:_bounds[g + 1]]
                     for g in range(n_targets)]

    def _inmem_target_source(need_row_sums):
        Ls = _x_host_row_sums() if need_row_sums else None
        for g in range(n_targets):
            g_rows = group_to_rows[g]
            yield (g, X_host, g_rows,
                   (Ls[g_rows] if Ls is not None else None))

    max_group_rows = max((int(r.size) for r in group_to_rows), default=0)

    # Resident-aware auto chunk (only when the user did not pin one). The core's
    # internal auto-sizer budgets the reference sort with n_groups=1 and samples
    # free BEFORE the ~5.7 GB resident reference is allocated; size it here on the
    # target working set instead and pass an explicit chunk. #72.
    chunk = gpu_gene_chunk_size
    if chunk is None:
        # Reclaim any pooled GPU memory (incl. a caller's cupy pool from a prior
        # phase) BEFORE reading free VRAM, so a stale pool cannot starve the
        # chunk (which #22 recovery would then thrash). #76
        _release_gpu_memory(run_gc=True)
        free = int(torch.cuda.mem_get_info(device)[0])
        chunk = _auto_gene_chunk_size_inmem(
            free_bytes=free, n_ref=int(ref_X.shape[0]), n_genes=n_genes,
            max_group_rows=max_group_rows,
            n_combos=len(lfc_combos or ()),
            n_levels=(0 if taustar_levels is None else
                      len(taustar_column_names(taustar_levels, taustar_se))))

    # Async pinned-H2D uploader for the target tiles (numba fast path only, which
    # needs a sparse CSR X_host — the out= pinned buffer is honored only there).
    # A dense adata.X (ensure_csr passes it through unchanged) or non-CSR falls
    # back to the legacy per-tile path. `issparse` guard first so `.format` is
    # never read on a dense ndarray. #72.
    uploader = None
    if (HAS_NUMBA and max_group_rows > 0
            and issparse(X_host) and X_host.format == "csr"):
        # Cap the pinned buffer width at n_genes: a user-pinned chunk above
        # n_genes would over-allocate page-locked host memory (the gene-chunk
        # loop never emits a tile wider than n_genes). The core still receives
        # the raw `chunk` below — the loop caps it. #80b
        uploader = _PinnedTileUploader(
            max_group_rows, _pinned_buf_width(chunk, n_genes), device)

    return refpool_de_core(
        ref_X=ref_X, target_source=_inmem_target_source,
        targets=unique_labels, n_genes=n_genes, var_names=var_names,
        device=device, mean_calc=mean_calc, epsilon=epsilon,
        gpu_gene_chunk_size=chunk, oom_recovery=oom_recovery,
        target_sum=target_sum, output_columns=output_columns,
        lfc_combos=lfc_combos, max_group_rows=max_group_rows,
        taustar_levels=taustar_levels, taustar_iters=taustar_iters,
        taustar_se=taustar_se,
        filter_gene_min_mean_value=filter_gene_min_mean_value,
        filter_gene_min_total_value=filter_gene_min_total_value,
        filter_gene_min_cpm_cell=filter_gene_min_cpm_cell,
        filter_gene_min_cpm_bulk=filter_gene_min_cpm_bulk,
        keep_genes_arr=keep_genes_arr, warn_noncount=True, uploader=uploader)


def cell_source_de(
    cell_source, *, targets, var_names, reference,
    mean_calc, epsilon, gpu_gene_chunk_size, oom_recovery,
    cpm_normalize, normalize_target_sum, output_columns, lfc_combos=None,
    taustar_levels=None, taustar_iters=None, taustar_se=False,
    filter_gene_min_mean_value, filter_gene_min_total_value,
    filter_gene_min_cpm_cell, filter_gene_min_cpm_bulk,
    keep_genes, device,
):
    """Bring-your-own cell source: rank each caller-supplied target group
    against the ``reference`` pool, resident-sorted on GPU. Runs the identical
    ``refpool_de_core`` as the streaming and in-memory external-reference paths
    -> bit-identical results.

    Mode/parameter validation is performed by the caller (``de()``) before the
    CUDA check, so it fails fast on CPU.

    ``uploader`` and ``max_group_rows`` are passed EXPLICITLY at their inert
    values: neither is knowable without draining the caller's source, and
    naming them documents a decision rather than an oversight.
    ``max_group_rows=0`` leaves the core's auto chunk sizer blind to the target
    working set -- see the ``gpu_gene_chunk_size`` advice in ``de()``'s
    docstring.
    """
    var_names = np.asarray(var_names, dtype=str)
    n_genes = int(var_names.size)
    targets = np.asarray(targets, dtype=str)

    _check_2d(reference, "reference")
    # stacklevel=4: de() -> cell_source_de -> ensure_csr -> warn points at the
    # user's de() call, matching inmem_external_ref_de.
    ref_X = ensure_csr(reference, name="reference.X", stacklevel=4)

    keep_genes_arr = (validate_keep_genes(keep_genes, n_genes)
                      if keep_genes is not None else None)

    # normalize_target_sum='median' is rejected by de() before this point (it
    # needs a row-sums pre-pass, i.e. a second drive of the caller's source), so
    # row_sums is never consulted here.
    target_sum = resolve_target_sum(
        cpm_normalize=cpm_normalize,
        normalize_target_sum=normalize_target_sum, row_sums=None)

    # Target-side raw-counts warning, in-band in the adapter. The in-memory
    # external-ref path samples its single X_host up front; there is no single
    # target matrix here, but the adapter sees every group. The core separately
    # samples ref_X via warn_noncount=True.
    _cpm_filter_active = (filter_gene_min_cpm_cell is not None
                          or filter_gene_min_cpm_bulk is not None)

    return refpool_de_core(
        ref_X=ref_X,
        target_source=make_target_source(
            cell_source, targets=targets, n_genes=n_genes,
            warn_noncount_targets=_cpm_filter_active,
            # target_source(1) -> refpool_de_core(2) -> cell_source_de(3) ->
            # de(4) -> the user's call(5). ensure_csr adds one more frame.
            warn_stacklevel=5),
        targets=targets, n_genes=n_genes, var_names=var_names,
        device=device, mean_calc=mean_calc, epsilon=epsilon,
        gpu_gene_chunk_size=gpu_gene_chunk_size, oom_recovery=oom_recovery,
        target_sum=target_sum, output_columns=output_columns,
        lfc_combos=lfc_combos, taustar_levels=taustar_levels,
        taustar_iters=taustar_iters, taustar_se=taustar_se,
        filter_gene_min_mean_value=filter_gene_min_mean_value,
        filter_gene_min_total_value=filter_gene_min_total_value,
        filter_gene_min_cpm_cell=filter_gene_min_cpm_cell,
        filter_gene_min_cpm_bulk=filter_gene_min_cpm_bulk,
        keep_genes_arr=keep_genes_arr, warn_noncount=True,
        uploader=None, max_group_rows=0)
