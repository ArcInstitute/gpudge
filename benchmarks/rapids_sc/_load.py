"""Generic dataset loader for the rapids_sc benchmark.

Reads any h5ad and builds a ready-to-use one-vs-reference grouping column from
``--groupby`` / ``--reference`` (optionally pooling many control guides into the
reference via ``--collapse-reference-prefix``). No GPU imports, so both runners
share exactly one load path and the matched input is identical across tools.
"""
from __future__ import annotations

from pathlib import Path

import anndata as ad
import pandas as pd
import scanpy as sc

import _common

#: obs column the runners group on (one-vs-reference). Built by ``load_anndata``.
GROUP_COL = "comparison"


def load_anndata(path: str | Path, groupby: str, reference: str,
                 *, collapse_prefix: str | None = None) -> ad.AnnData:
    """Load ``path`` (raw-count h5ad), derive ``obs[GROUP_COL]`` as a categorical
    one-vs-reference grouping, and return the AnnData.

    ``groupby`` is the obs column holding the per-cell group label; ``reference``
    is the label every other group is compared against. If ``collapse_prefix`` is
    given, any group label starting with it is pooled into ``reference`` first
    (so e.g. ``non-targeting-*`` control guides become one reference group).
    """
    a = sc.read_h5ad(path)
    if groupby not in a.obs:
        raise KeyError(
            f"--groupby {groupby!r} not in obs columns: {list(a.obs.columns)}")
    labels = a.obs[groupby].to_numpy()
    if collapse_prefix:
        labels = _common.collapse_reference(labels, collapse_prefix, reference)
    # Categorical: scanpy/rapids-singlecell rank_genes_groups requires the
    # groupby column to be a 'category' dtype (uses .cat.categories); gpudge's
    # de() is dtype-agnostic, so categorical is safe for both.
    a.obs[GROUP_COL] = pd.Categorical(labels)
    cats = set(a.obs[GROUP_COL].cat.categories)
    if reference not in cats:
        raise ValueError(
            f"--reference {reference!r} not among the {len(cats)} group labels in "
            f"{groupby!r}"
            + ("" if not collapse_prefix
               else f" (after collapsing {collapse_prefix!r}; check the prefix)"))
    n_ref = int((a.obs[GROUP_COL] == reference).sum())
    n_tgt = int((a.obs[GROUP_COL] != reference).sum())
    n_grp = int(a.obs[GROUP_COL].nunique())
    print(f"[load] cells={a.n_obs:,} genes={a.n_vars:,} groups={n_grp:,} "
          f"reference={reference!r} ({n_ref:,} cells) target={n_tgt:,} cells",
          flush=True)
    return a
