#!/usr/bin/env python
"""gpudge quickstart -- a runnable, ~30-second differential-expression run.

Walks the shortest useful path: load a real CRISPRi screen, run ``de()`` once, read the
result, and check the answer is right. ``docs/tutorial.md`` narrates the same steps with
the expected output inline.

    python examples/quickstart.py

Needs a CUDA GPU -- gpudge has no CPU fallback. The dataset is committed
(``docs/data/H1-VCC-2025-training.h5ad``, ~5 MB), so there is nothing to download.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import polars as pl

import gpudge

# Committed with the repo; see docs/make_tutorial_data.py for how it was cut.
DATA = Path(__file__).resolve().parent.parent / "docs" / "data" / "H1-VCC-2025-training.h5ad"

GROUPBY = "target_gene"     # obs column holding the perturbation label
CONTROL = "non-targeting"   # the label marking control cells

# This perturbation's guides target this gene, and the gene survived the subset's
# 1000-gene cut -- so gpudge should rank it as a strong DOWN hit. A positive control, not
# proof of the whole screen (see docs/tutorial.md), but a failure here means something is
# wrong end to end, so the script asserts it.
ON_TARGET = "TMSB4X"


def run(path: Path = DATA) -> pl.DataFrame:
    """Load the screen, run one ``de()`` call, and print a readable summary."""
    # ------------------------------------------------------------------ 1. load
    adata = ad.read_h5ad(path)
    counts = adata.X
    print(f"loaded {path.name}: {adata.n_obs} cells x {adata.n_vars} genes")
    print(f"  X is {type(counts).__name__} {counts.dtype}, raw counts (integral values)")
    print("  groups:")
    for label, n in adata.obs[GROUPBY].value_counts().sort_index().items():
        marker = "  <- control" if label == CONTROL else ""
        print(f"    {label:>14s}  {n:>4d} cells{marker}")

    # ------------------------------------------------------------- 2. run de()
    #
    # Every perturbation group is tested against `reference` in a single call. de() does
    # iterate the groups, but it loads and sorts the REFERENCE once per gene chunk and
    # reuses it across them, so the expensive part is paid once rather than per
    # perturbation.
    #
    # cpm_normalize=True scales each cell to 1e6 counts on the GPU, inline. Pass RAW
    # counts when you use it -- it does not mutate adata.X, and normalizing twice is the
    # easiest way to get a wrong answer here. If your matrix is already normalized, leave
    # it False. Exactly one normalization, from exactly one place: see the tutorial.
    print(f"\nrunning de() -- {adata.obs[GROUPBY].nunique() - 1} perturbations vs {CONTROL!r}")
    res = gpudge.de(
        adata,
        groupby=GROUPBY,
        reference=CONTROL,
        cpm_normalize=True,
    )

    # ------------------------------------------------- 3. read the result frame
    print(f"\nresult: polars DataFrame, {res.height} rows x {res.width} columns")
    print("  " + ", ".join(res.columns))
    print(f"  {res.height} rows = {adata.obs[GROUPBY].nunique() - 1} perturbations"
          f" x {adata.n_vars} genes")

    # --------------------------------------------- 4. the strongest hits overall
    print("\ntop 5 hits by adjusted p-value:")
    top = res.sort("p_adj").head(5)
    print(f"  {'target':>14s}  {'gene':>10s}  {'log2FC':>8s}  {'p_adj':>10s}")
    for row in top.iter_rows(named=True):
        print(f"  {row['target']:>14s}  {row['feature']:>10s}"
              f"  {row['log2_fold_change']:>8.2f}  {row['p_adj']:>10.2e}")

    # ------------------------------------------------- 5. did it find the truth?
    #
    # Knocking down a gene should reduce that gene's own expression. A positive control:
    # passing means the plumbing works end to end, and does NOT validate the other four
    # perturbations; failing has explanations (a weak guide, low basal expression, heavy
    # dropout, guide misassignment) that are not gpudge's fault. Worth running first
    # anyway -- see docs/tutorial.md.
    on_target = res.filter(
        (pl.col("target") == ON_TARGET) & (pl.col("feature") == ON_TARGET)
    )
    if on_target.height == 0:
        raise SystemExit(
            f"no ({ON_TARGET}, {ON_TARGET}) row in the result -- this check is specific "
            f"to the committed dataset, where {ON_TARGET} is both a perturbation and one "
            f"of the genes. A different --data will not have it.")
    hit = on_target.row(0, named=True)
    rank = (res.filter(pl.col("target") == ON_TARGET)
               .sort("p_adj")["feature"].to_list().index(ON_TARGET) + 1)
    print(f"\non-target check -- the {ON_TARGET} guides should knock down {ON_TARGET}:")
    print(f"  log2FC = {hit['log2_fold_change']:.2f}"
          f"  ({2 ** hit['log2_fold_change']:.3f}x control)")
    print(f"  p_adj  = {hit['p_adj']:.2e}")
    print(f"  rank   = {rank} of {adata.n_vars} genes for that perturbation")

    assert hit["log2_fold_change"] < -2, "on-target knockdown not detected"
    assert hit["p_adj"] < 1e-10, "on-target knockdown not significant"
    assert rank == 1, f"{ON_TARGET} should be the top hit for its own perturbation"
    print("  OK -- strongly down, highly significant, and the single top hit.")

    return res


def main() -> int:
    # A literal rather than `__doc__.splitlines()[0]`: `python -OO` strips docstrings,
    # which would make that an AttributeError. (Academic here -- gpudge cannot run under
    # -OO at all, because anndata's own _settings.py does `__doc__.find(...)` at import
    # time -- but a literal is clearer anyway.)
    ap = argparse.ArgumentParser(
        description="gpudge quickstart: one de() call on a committed CRISPRi screen.")
    ap.add_argument("--data", type=Path, default=DATA,
                    help="another copy of THIS dataset; the group/control/on-target "
                         "labels below are hard-coded for it")
    ap.add_argument("--out", type=Path, default=None,
                    help="optional .parquet to write the full result to")
    args = ap.parse_args()

    res = run(args.data)
    if args.out is not None:
        res.write_parquet(args.out)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
