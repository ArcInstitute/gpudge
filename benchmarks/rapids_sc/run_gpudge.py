"""Run gpudge de() on a dataset; write result parquet + timing JSON.

Operates on CPM+log1p-normalized data (the matched input rapids-singlecell also
gets) so the statistic lines up; times the de() compute call only and records
peak GPU VRAM.

  python run_gpudge.py --data screen.h5ad --groupby target_gene \
      --reference ntc --collapse-reference-prefix non-targeting --tag screen
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import scanpy as sc

import _common
from _load import GROUP_COL, load_anndata

from gpudge import de


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, help="path to a raw-count h5ad")
    p.add_argument("--groupby", required=True, help="obs column with group labels")
    p.add_argument("--reference", required=True, help="reference group label")
    p.add_argument("--collapse-reference-prefix", default=None,
                   help="pool group labels starting with this prefix into --reference")
    p.add_argument("--normalize", choices=["scanpy", "inplace"], default="scanpy",
                   help="matched CPM+log1p; 'inplace' is the copy-free host path "
                        "for matrices whose scanpy copy would OOM host RAM. MUST "
                        "match the value passed to run_rapids.py.")
    p.add_argument("--tag", default=None, help="output name; defaults to the data stem")
    p.add_argument("--outdir", type=Path, default=Path(__file__).parent / "results")
    args = p.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or Path(args.data).stem

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Need a CUDA GPU")
    device = torch.cuda.get_device_name(0)
    print(f"[env] torch={torch.__version__} device={device}", flush=True)

    t_load = time.perf_counter()
    adata = load_anndata(args.data, args.groupby, args.reference,
                         collapse_prefix=args.collapse_reference_prefix)
    load_sec = time.perf_counter() - t_load

    # CPM + log1p (matched input). de() then runs on already-normalized data.
    t_norm = time.perf_counter()
    if args.normalize == "inplace":
        _common.normalize_log1p_inplace(adata, target_sum=1e6)
    else:
        sc.pp.normalize_total(adata, target_sum=1e6)
        sc.pp.log1p(adata)
    norm_sec = time.perf_counter() - t_norm

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    # filter_gene_min_mean_value=0.0 drops only (gene,comparison) pairs whose
    # target AND reference means are both zero (statistically trivial), so
    # gpudge's gene set is a near-complete strict subset of rapids-sc's
    # n_genes=all — see RESULTS.md "Correctness" for the small coverage delta.
    out = de(adata, groupby=GROUP_COL, reference=args.reference,
             mean_calc="geometric", epsilon=1e-9,
             filter_gene_min_mean_value=0.0)
    de_sec = time.perf_counter() - t0
    peak_vram_bytes = int(torch.cuda.max_memory_allocated())
    peak_vram_reserved_bytes = int(torch.cuda.max_memory_reserved())
    print(f"[de] {de_sec:.2f}s, {out.height:,} rows, "
          f"peak_vram={peak_vram_bytes / 1e9:.2f}GB "
          f"reserved={peak_vram_reserved_bytes / 1e9:.2f}GB", flush=True)

    pq = args.outdir / f"{tag}_gpudge.parquet"
    out.select(["target", "feature", "log2_fold_change", "p_value", "p_adj"]).write_parquet(pq)

    timing = {
        "tool": "gpudge", "tag": tag, "device": device,
        "normalize": args.normalize,
        "de_wall_sec": de_sec, "load_sec": load_sec, "norm_sec": norm_sec,
        "end_to_end_sec": load_sec + norm_sec + de_sec, "rows": out.height,
        "n_cells": int(adata.n_obs), "n_genes": int(adata.n_vars),
        "peak_vram_bytes": peak_vram_bytes,
        "peak_vram_gb": peak_vram_bytes / 1e9,
        "peak_vram_reserved_bytes": peak_vram_reserved_bytes,
        "peak_vram_reserved_gb": peak_vram_reserved_bytes / 1e9,
    }
    (args.outdir / f"{tag}_gpudge_timing.json").write_text(json.dumps(timing, indent=2))
    print(json.dumps(timing, indent=2), flush=True)


if __name__ == "__main__":
    main()
