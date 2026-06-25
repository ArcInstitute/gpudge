# src/gpudge/_shard_stream.py
"""Native shard-streaming driver for de(shard_archive=...)."""
from __future__ import annotations

import warnings

import anndata as ad
import numpy as np
import polars as pl
import torch

from ._csr_dense import csr_row_sums, csr_rows_col_range_to_dense
from ._filter import _row_scale_needs, validate_keep_genes, x_has_noncount_signal
from ._fdr import bh_per_group
from ._mwu import _tie_term_per_gene, mwu_one_group
from ._output import DEFAULT_OUTPUT_COLUMNS, assemble_dataframe, empty_output_frame
from ._stream import _auto_gene_chunk_size, run_gene_chunks_with_recovery


def _import_shardad():
    try:
        import shardad
    except ImportError as e:  # pragma: no cover - exercised via monkeypatch
        raise ImportError(
            "de(shard_archive=...) requires the optional 'streaming' extra. "
            "Install with `pip install gpudge[streaming]` or "
            "`uv sync --extra streaming`."
        ) from e
    return shardad


def _resolve_streaming(arch, groupby, reference):
    """Resolve groupby + reference mode from the archive. See spec §Dispatch.

    Returns (groupby, mode, ref_X, ref_msg_label) where mode is
    "archive_ref" (reference shard pool) or "external_ref" (AnnData pool).
    """
    arch_groupby = arch.manifest.get("group_by")
    if arch_groupby is None:
        raise ValueError(
            "shard_archive was not written with a group_by key (not "
            "target-aware). Re-write with "
            "shardad.write_sharded(..., format='v2', group_by=<obs column>)."
        )
    if groupby is not None and groupby != arch_groupby:
        raise ValueError(
            f"groupby={groupby!r} does not match the archive's group_by="
            f"{arch_groupby!r}."
        )
    groupby = arch_groupby

    if isinstance(reference, ad.AnnData):
        # Mode 2: external reference pool; all archive shards are targets.
        # If the archive ALSO designates its own reference shard, the external
        # pool wins and the archive's reference shard is ignored (Semantics A).
        ref_shard = arch.manifest.get("reference_shard")
        if ref_shard is not None:
            warnings.warn(
                f"de(shard_archive=..., reference=<AnnData>): the archive designates "
                f"its own reference shard (index {ref_shard}), which will be ignored "
                "in favor of the external reference AnnData pool. Omit reference= "
                "(or pass reference=<label>) to use the archive's own reference shard instead.",
                UserWarning,
                # de() -> stream_de() -> _resolve_streaming() -> warn: 4 points at the user's de() call.
                stacklevel=4,
            )
        if reference.n_vars != arch.n_vars:
            raise ValueError(
                f"reference AnnData has {reference.n_vars} genes but the archive "
                f"has {arch.n_vars}; the reference and targets must share the "
                "gene axis."
            )
        arch_vars = np.asarray(arch.var.index)
        if not np.array_equal(np.asarray(reference.var_names), arch_vars):
            raise ValueError(
                "reference AnnData var_names do not match the archive's gene "
                "axis order; align the reference to the archive var_names."
            )
        return groupby, "external_ref", reference.X, "<external AnnData pool>"

    # Mode 1: reference shard pool.
    if arch.manifest.get("reference_shard") is None:
        raise ValueError(
            "shard_archive has no reference shard. Either supply an external "
            "pool via reference=<AnnData>, or re-write the archive with "
            "shardad.write_sharded(..., reference=<label(s)>)."
        )
    ref_adata = arch.read_reference()                 # read once; reused by Phase 0
    if ref_adata is None:
        raise ValueError(
            "shard_archive designates a reference shard but reading it returned "
            "None; the archive's manifest and reference shard are inconsistent "
            "(possibly corrupted)."
        )
    if groupby not in ref_adata.obs:
        raise ValueError(
            f"groupby column {groupby!r} not found in the reference shard's obs "
            f"(available: {list(ref_adata.obs.columns)})."
        )
    ref_label_set = set(np.asarray(ref_adata.obs[groupby]).astype(str).tolist())
    if reference is not None and str(reference) not in ref_label_set:
        raise ValueError(
            f"reference={reference!r} is not among the archive's reference "
            f"labels {sorted(ref_label_set)}."
        )
    msg_label = (str(reference) if reference is not None
                 else "|".join(sorted(ref_label_set)))
    return groupby, "archive_ref", ref_adata.X, msg_label


def _enumerate_targets(arch):
    """Cheap label-only pass over iter_group_shards (no shard data I/O):
    returns the ordered target labels + a label→index map. Labels are disjoint
    across shards (the planner never splits a group), so order is stable."""
    targets: list[str] = []
    for gs in arch.iter_group_shards():
        targets.extend(str(lab) for lab in gs.labels)
    tgt_index = {lab: i for i, lab in enumerate(targets)}
    if len(tgt_index) != len(targets):                 # defensive: should never happen
        raise ValueError("shard_archive has duplicate group labels across shards.")
    return targets, tgt_index


def _free_gpu_bytes(device) -> int:
    free, _ = torch.cuda.mem_get_info(device)
    return int(free)


def _reference_prepass(ref_X, n_genes, device, chunk, *, mean_calc,
                       scale_main, scale_num, need_other_unit, need_row_sums,
                       need_row_scales, oom_recovery):
    """Read the reference once, sort per gene-chunk, keep the sorted form +
    tie terms resident on the GPU; compute reference means. See spec §Phase 0."""
    n_ref = int(ref_X.shape[0])

    # Oversized guard: the resident sorted reference is dense n_ref×n_genes f32.
    projected = n_ref * n_genes * 4
    free = _free_gpu_bytes(device)
    if projected >= free:
        raise RuntimeError(
            f"shard-streaming reference is too large to keep resident on the "
            f"GPU: n_ref={n_ref} × n_genes={n_genes} × 4 B = {projected/1e9:.1f} GB "
            f"vs ~{free/1e9:.1f} GB free. Use the in-memory de(adata=...) path, a "
            f"smaller reference, or a larger-memory GPU."
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
                      mean_calc, scale_main, group_scales=None, want_other=False):
    """One target group's gene-chunk stats. ``group_t`` (m, chunk) f32 is the
    group's dense slice, UNSCALED; it is scaled **in place** iff
    ``scale_main``. ``group_scales`` (m,) = numerator/L is required when
    scale_main, or for the scaled OTHER-unit mean when not scale_main.
    Returns (arith, reported, other|None, u1, p) — each a (chunk,) f64 torch
    tensor ON DEVICE; callers move to host. Mirrors the in-memory ref-mode body
    exactly (unscaled other-unit, scale-in-place, f64 mean + f32 MWU)."""
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
    u1, p = mwu_one_group(sorted_ref_chunk, ref_tie_chunk,
                          group_t.T.contiguous(), n_ref=n_ref)
    return arith, reported, other, u1, p


def stream_de(shard_archive, *, groupby, reference, mean_calc, epsilon,
              gpu_gene_chunk_size, oom_recovery, cpm_normalize,
              normalize_target_sum, output_columns,
              filter_gene_min_mean_value, filter_gene_min_total_value,
              filter_gene_min_cpm_cell, filter_gene_min_cpm_bulk,
              keep_genes, device):
    import warnings

    shardad = _import_shardad()
    arch = shardad.ShardedArchive(shard_archive)
    n_genes = int(arch.n_vars)
    var_names = np.asarray(arch.var.index)

    if output_columns is not None:
        unknown = [k for k in output_columns if k not in DEFAULT_OUTPUT_COLUMNS]
        if unknown:
            raise KeyError(
                f"output_columns keys not in de() output schema: {unknown}. "
                f"Valid keys: {list(DEFAULT_OUTPUT_COLUMNS)}")
    if mean_calc not in ("arithmetic", "geometric"):
        raise ValueError(f"mean_calc must be 'arithmetic' or 'geometric', got {mean_calc!r}.")
    if epsilon < 0:
        raise ValueError(f"epsilon must be >= 0, got {epsilon!r}.")

    groupby, mode, ref_X, _ = _resolve_streaming(arch, groupby, reference)
    keep_genes_arr = (validate_keep_genes(keep_genes, n_genes)
                      if keep_genes is not None else None)

    from ._normalize import resolve_target_sum
    _median_requested = (isinstance(normalize_target_sum, str)
                         and normalize_target_sum == "median")
    if _median_requested:
        # Up-front pass over the reference + every group shard to gather per-cell
        # library sizes, so the global median target can be computed before any
        # scaling. ~2x archive I/O, only when 'median' is requested in streaming.
        _all_sums = [csr_row_sums(ref_X)]
        for gs in arch.iter_group_shards():
            _shard = gs.to_anndata()
            _all_sums.append(csr_row_sums(_shard.X))
            del _shard
        _row_sums_all = np.concatenate(_all_sums)
    else:
        _row_sums_all = None
    target_sum = resolve_target_sum(
        cpm_normalize=cpm_normalize,
        normalize_target_sum=normalize_target_sum, row_sums=_row_sums_all)
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
    need_row_sums, need_row_scales = _row_scale_needs(
        _scale_main, filter_gene_min_cpm_cell, filter_gene_min_cpm_bulk,
        _median_requested)

    targets, tgt_index = _enumerate_targets(arch)
    n_targets = len(targets)
    if n_targets == 0:
        # Canonical typed, output_columns-aware empty frame, so an empty result
        # has the identical schema to a non-empty one (same dtypes + the same
        # select/rename). See _output.empty_output_frame.
        return empty_output_frame(output_columns)

    # Chunk size: budget on the reference cell count (ref-mode basis).
    n_ref_hint = int(ref_X.shape[0])
    if gpu_gene_chunk_size is None:
        free, _ = torch.cuda.mem_get_info(device)
        gpu_gene_chunk_size = _auto_gene_chunk_size(
            free_bytes=free, budget_n=n_ref_hint, n_groups=n_targets,
            mean_calc=mean_calc, n_genes=n_genes, ref_mode=True)
    chunk = gpu_gene_chunk_size

    # CPM non-count warning (once), sampling the reference X. NOTE: archive
    # shards are stored as uint16 by shardad's v2 writer (it rejects fractional/
    # negative/>65535 data with a LossyCastError), so target + reference SHARDS
    # are always raw counts — there is nothing to detect there. The only
    # realizable non-count input is a Mode-2 external reference= AnnData, which
    # is exactly ref_X here. (ultrareview #42: the "also sample target shards"
    # finding has no reachable trigger; verified empirically.)
    if _cpm_filter_active and x_has_noncount_signal(ref_X):
        warnings.warn(
            "reference X does not look like raw counts (non-integer or negative "
            "values); the filter_gene_min_cpm_* filters assume raw counts.",
            UserWarning, stacklevel=2)

    # ---- Phase 0: reference resident-sorted on GPU ----
    ref = _reference_prepass(
        ref_X, n_genes, device, chunk, mean_calc=mean_calc,
        scale_main=_scale_main, scale_num=_scale_num, need_other_unit=need_other_unit,
        need_row_sums=need_row_sums, need_row_scales=need_row_scales,
        oom_recovery=oom_recovery)
    sorted_ref_full = ref["sorted_ref_full"]
    ref_tie_term_full = ref["ref_tie_term_full"]
    n_ref = ref["n_ref"]
    # Carry Phase 0's (possibly OOM-downshifted) chunk into Phase 1 so a
    # downshift isn't rediscovered per target group. (ultrareview #43)
    chunk = ref["final_chunk"]

    # ---- accumulators (target-indexed) ----
    target_mean_acc = np.zeros((n_targets, n_genes), dtype=np.float64)
    arith_target_acc = np.zeros((n_targets, n_genes), dtype=np.float64)
    other_target_acc = (np.zeros((n_targets, n_genes), dtype=np.float64)
                        if need_other_unit else None)
    U_acc = np.zeros((n_targets, n_genes), dtype=np.float64)
    p_acc = np.ones((n_targets, n_genes), dtype=np.float64)
    target_counts = np.zeros(n_targets, dtype=np.int64)
    group_libtot = (np.zeros(n_targets, dtype=np.float64)
                    if filter_gene_min_cpm_bulk is not None else None)

    # ---- Phase 1: stream guide-shards (outer), gene-chunks (inner) ----
    for gs in arch.iter_group_shards():
        shard = gs.to_anndata()
        Xs = shard.X
        Ls = csr_row_sums(Xs) if need_row_sums else None
        for label, sl in gs.groups.items():
            g = tgt_index[str(label)]
            rows = np.arange(sl.start, sl.stop, dtype=np.int64)
            target_counts[g] = rows.size
            if group_libtot is not None:
                group_libtot[g] = float(Ls[rows].sum())
            grp_scales_t = None
            if need_row_scales:
                safe = np.where(Ls[rows] == 0, 1.0, Ls[rows])
                grp_scales_t = torch.from_numpy((_scale_num / safe).astype(np.float32)).to(device)

            def _chunk(start, stop, g=g, rows=rows, grp_scales_t=grp_scales_t,
                       Xs=Xs):
                dense = csr_rows_col_range_to_dense(Xs, rows, start, stop)  # (m, w) f32 UNSCALED
                gt = torch.from_numpy(dense).to(device)
                arith, reported, other, u1, p = group_chunk_stats(
                    gt, sorted_ref_full[start:stop], ref_tie_term_full[start:stop],
                    n_ref, mean_calc=mean_calc, scale_main=_scale_main,
                    group_scales=grp_scales_t, want_other=other_target_acc is not None)
                arith_target_acc[g, start:stop] = arith.cpu().numpy()
                target_mean_acc[g, start:stop] = reported.cpu().numpy()
                if other_target_acc is not None:
                    other_target_acc[g, start:stop] = other.cpu().numpy()
                U_acc[g, start:stop] = u1.cpu().numpy()
                p_acc[g, start:stop] = p.cpu().numpy()

            # Reassign chunk so a downshift in this group persists to the next
            # (and to the next shard) instead of restarting at full width. (#43)
            chunk = run_gene_chunks_with_recovery(
                n_genes, chunk, _chunk, oom_recovery=oom_recovery)
        del shard, Xs

    # ---- Phase 2: filters → assemble → BH-FDR (same tail as in-memory) ----
    rm_b = np.broadcast_to(ref["ref_mean"], target_mean_acc.shape)
    log2fc = np.log2((target_mean_acc + epsilon) / (rm_b + epsilon))

    keep_mask_acc = _streaming_keep_mask(
        n_targets, n_genes, chunk, arith_target_acc, ref["arith_ref"],
        other_target_acc, ref["other_ref"], target_counts, n_ref,
        group_libtot, ref["ref_libtot"], target_sum,
        filter_gene_min_mean_value, filter_gene_min_total_value,
        filter_gene_min_cpm_cell, filter_gene_min_cpm_bulk, keep_genes_arr)

    df = assemble_dataframe(
        target=np.asarray(targets), feature=var_names,
        target_mean=target_mean_acc, ref_mean=ref["ref_mean"],
        target_ncells=target_counts, ref_ncells=int(n_ref),
        log2_fold_change=log2fc, p_value=p_acc, test_statistic=U_acc,
        p_adj=np.zeros_like(p_acc), flat_keep=keep_mask_acc.ravel(),
        output_columns=None)

    if df.height > 0:
        post_filter_g = np.nonzero(keep_mask_acc)[0]
        g_idx = torch.from_numpy(post_filter_g)
        adj = bh_per_group(df["p_value"].to_torch(), g_idx, n_targets)
        df = df.with_columns(p_adj=pl.Series(adj.numpy()))

    if output_columns is None:
        return df
    return df.select(list(output_columns)).rename(output_columns)


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
