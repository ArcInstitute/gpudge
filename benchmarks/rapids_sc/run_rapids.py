"""Run rapids-singlecell PR #636 (Wilcoxon refactor / 0.16.0-dev) DE on a
dataset; write result parquet + timing JSON.

PR #636 reworks ``tl.rank_genes_groups`` Wilcoxon onto dedicated CUDA kernels.
Two architectural levers this runner exercises:

  * exact ``wilcoxon`` with a HOST-STREAMING sparse path (``--transfer host``):
    a CPU-resident scipy CSR is streamed to the GPU in column sub-batches, so
    peak VRAM is decoupled from total cell count. (``--transfer device`` instead
    moves the whole matrix to the GPU first -- fast on small data, OOMs on large,
    like the released 0.14.1.) This mirrors gpudge, whose de() also streams from
    host; for both tools the CPU->GPU transfer is counted inside de_wall_sec.
  * approximate ``wilcoxon_binned`` -- histogram-binned ranks, O(n) not
    O(n log n), Dask-compatible. In-memory binned moves X to GPU, so on large
    data it needs ``--dask`` to stay under VRAM.

Matched input = CPM (target_sum=1e6) + log1p, one-vs-reference, tie correction,
BH FDR -- identical to run_gpudge.py. ``--normalize`` MUST match run_gpudge.py.

  micromamba run -n <env> python run_rapids.py --data screen.h5ad \
      --groupby target_gene --reference ntc \
      --collapse-reference-prefix non-targeting --method wilcoxon \
      --transfer host --tag screen
"""
from __future__ import annotations

import argparse
import inspect
import json
import threading
import time
from pathlib import Path

import polars as pl
import scanpy as sc

import _common
from _load import GROUP_COL, load_anndata


class _PeakVram:
    """Poll total device VRAM use (``memGetInfo``) on a background thread and
    record the peak. PR #636's kernels allocate raw, not via cupy's default
    memory pool, so ``pool.total_bytes()`` reads ~0 -- device-level polling is
    the authoritative measure (same quantity nvidia-smi reports). Assumes a
    dedicated GPU; other processes' allocations would inflate the reading."""

    def __init__(self, hz: float = 20.0):
        self._interval = 1.0 / hz
        self._peak = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self) -> None:
        import cupy as cp
        while not self._stop.is_set():
            free, total = cp.cuda.runtime.memGetInfo()
            used = total - free
            if used > self._peak:
                self._peak = used
            self._stop.wait(self._interval)

    def __enter__(self) -> "_PeakVram":
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

    @property
    def peak_bytes(self) -> int:
        return int(self._peak)


def _run_rank(adata, reference: str, method: str, *, n_bins: int | None) -> tuple[pl.DataFrame, float]:
    import rapids_singlecell as rsc

    kw = dict(groupby=GROUP_COL, reference=reference, method=method,
              tie_correct=True, corr_method="benjamini-hochberg",
              n_genes=adata.n_vars)
    sig = inspect.signature(rsc.tl.rank_genes_groups)
    if "use_continuity" in sig.parameters:
        kw["use_continuity"] = True
    if method == "wilcoxon_binned" and n_bins is not None and "n_bins" in sig.parameters:
        kw["n_bins"] = n_bins
    t0 = time.perf_counter()
    rsc.tl.rank_genes_groups(adata, **kw)
    wall = time.perf_counter() - t0
    df = _common.reshape_rapids_uns(adata.uns["rank_genes_groups"])
    return df, wall


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, help="path to a raw-count h5ad")
    p.add_argument("--groupby", required=True, help="obs column with group labels")
    p.add_argument("--reference", required=True, help="reference group label")
    p.add_argument("--collapse-reference-prefix", default=None,
                   help="pool group labels starting with this prefix into --reference")
    p.add_argument("--method", default="wilcoxon",
                   choices=["wilcoxon", "wilcoxon_binned"])
    p.add_argument("--transfer", default="host", choices=["host", "device"],
                   help="exact wilcoxon: host=stream CPU CSR (no anndata_to_GPU); "
                        "device=move whole matrix to GPU first. Ignored for binned.")
    p.add_argument("--dask", action="store_true",
                   help="binned only: feed a dask-backed cupy array (avoids the "
                        "full-matrix X_to_GPU OOM on large data).")
    p.add_argument("--n-bins", type=int, default=None)
    p.add_argument("--chunk-cells", type=int, default=100_000,
                   help="dask binned: cells per GPU block (smaller = less VRAM).")
    p.add_argument("--normalize", choices=["scanpy", "inplace"], default="scanpy",
                   help="matched CPM+log1p; 'inplace' is the copy-free host path "
                        "for matrices whose scanpy copy would OOM host RAM. MUST "
                        "match the value passed to run_gpudge.py. Host/dask paths "
                        "only (device normalizes on the GPU).")
    p.add_argument("--tag", default=None,
                   help="output suffix; files are <tag>_rapids_<suffix>.parquet etc. "
                        "Defaults to the data stem + method/transfer.")
    p.add_argument("--outdir", type=Path, default=Path(__file__).parent / "results")
    args = p.parse_args()
    # On the on-device exact path and in-memory binned path, normalization is
    # always done on the GPU (rsc.pp.*); --normalize inplace only applies to the
    # host/dask paths. Reject the combo up front so the timing JSON can't record
    # a normalization that wasn't used.
    if (args.method == "wilcoxon_binned" or args.transfer == "device") \
            and not args.dask and args.normalize == "inplace":
        p.error("--normalize inplace is not supported for on-device runs "
                "(device normalizes on the GPU); use --transfer host or --dask.")
    args.outdir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.data).stem
    suffix = (f"{args.method}_{args.transfer}" if args.method == "wilcoxon"
              else f"{args.method}{'_dask' if args.dask else '_gpu'}")
    tag = args.tag or f"{stem}_{suffix}"

    import cupy as cp
    import rapids_singlecell as rsc
    print(f"[env] rapids_singlecell={rsc.__version__} cupy={cp.__version__} "
          f"method={args.method} transfer={args.transfer} dask={args.dask} tag={tag}",
          flush=True)

    t_load = time.perf_counter()
    adata = load_anndata(args.data, args.groupby, args.reference,
                         collapse_prefix=args.collapse_reference_prefix)
    load_sec = time.perf_counter() - t_load

    timing = {"tool": "rapids-singlecell", "tag": tag,
              "rapids_singlecell": rsc.__version__, "cupy": cp.__version__,
              "method": args.method, "transfer": args.transfer, "dask": args.dask,
              "normalize": args.normalize,
              "approximate": args.method == "wilcoxon_binned",
              "load_sec": load_sec,
              "n_cells": int(adata.n_obs), "n_genes": int(adata.n_vars)}

    oom_exc = (cp.cuda.memory.OutOfMemoryError, MemoryError)
    vram = _PeakVram()
    try:
        with vram:
            on_gpu = args.method == "wilcoxon_binned" or args.transfer == "device"
            t_norm = time.perf_counter()
            if on_gpu and not args.dask:
                # device exact / in-memory binned: whole matrix to GPU, normalize there.
                rsc.get.anndata_to_GPU(adata)
                rsc.pp.normalize_total(adata, target_sum=1e6)
                rsc.pp.log1p(adata)
            elif args.dask:
                # binned --dask: normalize on CPU first (scipy CSR), THEN wrap as a
                # dask-cupy array so we don't rely on rsc.pp.* supporting dask-cupy.
                # The binned kernel streams blocks to GPU during the DE call.
                from _dask_helpers import to_dask_gpu
                _normalize_host(adata, args.normalize)
                to_dask_gpu(adata, chunk_cells=args.chunk_cells)
            else:
                # host-streaming exact: keep CPU scipy CSR, normalize on CPU (this
                # is exactly gpudge's normalization path -> tightest parity).
                _normalize_host(adata, args.normalize)
            norm_sec = time.perf_counter() - t_norm

            df, de_sec = _run_rank(adata, args.reference, args.method, n_bins=args.n_bins)
        timing.update(oom=False, norm_sec=norm_sec, de_wall_sec=de_sec,
                      norm_plus_de_sec=norm_sec + de_sec,
                      end_to_end_sec=load_sec + norm_sec + de_sec, rows=df.height,
                      peak_vram_bytes=vram.peak_bytes,
                      peak_vram_gb=vram.peak_bytes / 1e9)
    except oom_exc as e:
        print(f"[oom] {tag}: {type(e).__name__}: {e}", flush=True)
        try:
            _free, _total = cp.cuda.runtime.memGetInfo()
            timing["gpu_total_bytes"] = int(_total)
        except Exception:
            pass
        cp.get_default_memory_pool().free_all_blocks()
        timing.update(oom=True, oom_error=f"{type(e).__name__}: {e}", rows=0,
                      peak_vram_bytes=vram.peak_bytes,
                      peak_vram_gb=vram.peak_bytes / 1e9)
        df = pl.DataFrame(schema={"target": pl.String, "feature": pl.String,
                                  "log2_fold_change": pl.Float64,
                                  "p_value": pl.Float64, "p_adj": pl.Float64})

    df.write_parquet(args.outdir / f"{tag}_rapids.parquet")
    (args.outdir / f"{tag}_rapids_timing.json").write_text(json.dumps(timing, indent=2))
    print(json.dumps(timing, indent=2), flush=True)


def _normalize_host(adata, normalize: str) -> None:
    if normalize == "inplace":
        _common.normalize_log1p_inplace(adata, target_sum=1e6)
    else:
        sc.pp.normalize_total(adata, target_sum=1e6)
        sc.pp.log1p(adata)


if __name__ == "__main__":
    main()
