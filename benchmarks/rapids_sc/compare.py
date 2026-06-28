"""Join gpudge + rapids result parquets, compute correlations + coverage, merge
timings, write a combined report JSON. CPU-only.

Reads <gpudge-tag>_gpudge{.parquet,_timing.json} and
<rapids-tag>_rapids{.parquet,_timing.json} from --outdir (the runners' tags).

  python compare.py --gpudge-tag screen --rapids-tag screen_wilcoxon_host
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

import _common


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpudge-tag", required=True,
                   help="run_gpudge.py --tag (reads <tag>_gpudge.parquet etc.)")
    p.add_argument("--rapids-tag", required=True,
                   help="run_rapids.py tag (reads <tag>_rapids.parquet etc.)")
    p.add_argument("--out", type=Path, default=None,
                   help="report path; defaults to <gpudge-tag>_vs_<rapids-tag>.json")
    p.add_argument("--outdir", type=Path, default=Path(__file__).parent / "results")
    args = p.parse_args()

    gp = pl.read_parquet(args.outdir / f"{args.gpudge_tag}_gpudge.parquet")
    rp = pl.read_parquet(args.outdir / f"{args.rapids_tag}_rapids.parquet")
    gp_t = json.loads((args.outdir / f"{args.gpudge_tag}_gpudge_timing.json").read_text())
    rp_t = json.loads((args.outdir / f"{args.rapids_tag}_rapids_timing.json").read_text())

    cmp = _common.compare(gp, rp)
    rapids_oom = rp_t.get("oom", False)
    g_de = gp_t["de_wall_sec"]
    r_de = rp_t.get("de_wall_sec")  # absent when rapids OOMed
    # norm+DE = compute starting from the CPU AnnData, excluding disk load.
    # gpudge: CPU CPM+log1p (norm_sec) + de() which itself includes the
    # CPU->GPU streaming transfer. rapids host: norm_sec is the CPU normalize;
    # its CPU->GPU streaming transfer is inside de_wall_sec too.
    g_nde = gp_t["norm_sec"] + g_de
    r_nde = (rp_t["norm_sec"] + r_de
             if (not rapids_oom and r_de is not None and "norm_sec" in rp_t)
             else None)

    def _ratio(num, den):
        return num / den if (num is not None and den is not None and den != 0) else None

    report = {
        "gpudge_tag": args.gpudge_tag,
        "rapids_tag": args.rapids_tag,
        "rapids_oom": rapids_oom,
        "gpudge_timing": gp_t,
        "rapids_timing": rp_t,
        "speed": {
            "gpudge_de_sec": g_de, "rapids_de_sec": r_de,
            "speedup_gpudge_over_rapids_de": _ratio(r_de, g_de),
            "gpudge_norm_plus_de_sec": g_nde, "rapids_norm_plus_de_sec": r_nde,
            "speedup_gpudge_over_rapids_norm_plus_de": _ratio(r_nde, g_nde),
            "gpudge_end_to_end_sec": gp_t["end_to_end_sec"],
            "rapids_end_to_end_sec": rp_t.get("end_to_end_sec"),
            "rapids_method": rp_t.get("method"),
            "rapids_approximate": rp_t.get("approximate", False),
            "note": "load=disk->CPU (same path both); CPU->GPU transfer is inside "
                    "de_wall_sec for gpudge and for rapids host-streaming.",
        },
        "vram": {
            "gpudge_peak_gb": gp_t.get("peak_vram_gb"),
            "rapids_peak_gb": rp_t.get("peak_vram_gb"),
        },
        "coverage": cmp["coverage"],
        "correlations": cmp["correlations"],
    }
    out = args.out or args.outdir / f"{args.gpudge_tag}_vs_{args.rapids_tag}.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\n[done] wrote {out}")


if __name__ == "__main__":
    main()
