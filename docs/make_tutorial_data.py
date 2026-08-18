#!/usr/bin/env python
"""Regenerate ``docs/data/H1-VCC-2025-training.h5ad``, the dataset ``docs/tutorial.md``
and ``examples/quickstart.py`` run on.

It is a deterministic subset of the **public** Virtual Cell Challenge 2025 training data
(https://huggingface.co/datasets/arcinstitute/VCC_train): 6 perturbation groups x 100
cells x the 1000 genes with the highest total count across those cells, raw integer
counts. Byte-identical to the subset cell_eval2's tutorial uses -- same recipe, same
seed -- so a reader working through both tools sees the same cells.

This is a **provenance record**, not something you need to run: the ``.h5ad`` is
committed. Run it only to reproduce or re-cut the subset, pointing it at your own copy
of the full training anndata (221,273 cells x 18,080 genes)::

    GPUDGE_VCC_TRAINING_H5AD=/path/to/adata_Training.h5ad \\
        python docs/make_tutorial_data.py

The source path is taken from the environment rather than hard-coded so this file
carries no site-specific path. Download the source from the Hugging Face link above.
"""
from __future__ import annotations

import os
import sys

import anndata as ad
import numpy as np

SOURCE_ENV = "GPUDGE_VCC_TRAINING_H5AD"
OUT = os.path.join(os.path.dirname(__file__), "data", "H1-VCC-2025-training.h5ad")

# The recipe. Do NOT change these without re-cutting the file: tests/test_tutorial.py
# pins the resulting shape, group sizes and gene count.
PERTS = ["non-targeting", "TMSB4X", "STAT1", "MED12", "TET1", "SRC"]
CELLS_PER_PERT = 100
N_GENES = 1000
SEED = 0


def main() -> int:
    source = os.environ.get(SOURCE_ENV)
    if not source:
        print(
            f"set {SOURCE_ENV} to your copy of the VCC 2025 training anndata "
            "(https://huggingface.co/datasets/arcinstitute/VCC_train).\n"
            "You do not need to run this script to use the tutorial -- "
            f"{os.path.relpath(OUT)} is committed.",
            file=sys.stderr,
        )
        return 2

    rng = np.random.default_rng(SEED)
    adata = ad.read_h5ad(source, backed="r")
    target = adata.obs["target_gene"].to_numpy()

    take = []
    for pert in PERTS:
        idx = np.where(target == pert)[0]
        if idx.size < CELLS_PER_PERT:
            raise SystemExit(
                f"{pert!r} has only {idx.size} cells in {source}; the tutorial and its "
                f"tests assume {CELLS_PER_PERT} per group. Silently under-filling a "
                f"group would make the committed subset disagree with the docs.")
        chosen = rng.choice(idx, size=CELLS_PER_PERT, replace=False)
        take.append(np.sort(chosen))
    sel = np.sort(np.concatenate(take))

    sub = adata[sel].to_memory()
    sub.obs = sub.obs[["target_gene", "guide_id", "batch"]].copy()

    # Highest total count across the selected cells, so the subset stays informative at
    # 1000 genes rather than mostly zeros.
    #
    # `np.argsort` keeps its default, UNSTABLE kind deliberately: the whole point of this
    # subset is that it is byte-identical to cell_eval2's, and cell_eval2's recipe sorts
    # exactly this way -- passing kind="stable" here would make the two recipes disagree
    # whenever genes tie on total count at the 1000th place. Such a tie could in principle
    # be broken differently by another NumPy build, so the authority for "the tutorial's
    # data" is the committed .h5ad and the sha256 `tests/test_tutorial.py` pins on it, not
    # a rerun of this script. (Copilot review, PR #129.)
    totals = np.asarray(sub.X.sum(axis=0)).ravel()
    keep_genes = np.sort(np.argsort(totals)[::-1][:N_GENES])
    sub = sub[:, keep_genes].copy()

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    sub.write_h5ad(OUT)
    print(f"wrote {OUT}  shape={sub.shape}  MB={os.path.getsize(OUT) / 1e6:.2f}")
    print("perts:", sorted(map(str, np.unique(sub.obs["target_gene"]))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
