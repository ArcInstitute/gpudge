# src/gpudge/_ingest.py
"""Input validation + label encoding for de()."""
from __future__ import annotations

import warnings
from dataclasses import dataclass
import numpy as np
import anndata as ad

ALL_OTHERS = "__all_others__"
# Pre-v0.1 sentinel value. ``de()`` and ``ingest()`` both map this to
# ``ALL_OTHERS`` with a DeprecationWarning so existing callers keep
# working for one release. ``de()`` remaps before calling ``ingest()``,
# so direct ``ingest()`` callers are the only ones who see the warning
# emitted from this file — no double-warning via the ``de() → ingest()``
# path.
LEGACY_ALL_OTHERS = "all_others"


@dataclass(slots=True)
class IngestedState:
    adata: ad.AnnData
    n_cells: int
    n_genes: int
    ref_label: str
    labels: np.ndarray          # int-encoded per cell, length n_cells
    unique_labels: np.ndarray   # str array, length n_groups
    ref_label_idx: int | None   # None when ref_label == ALL_OTHERS
    target_labels: np.ndarray   # str array, length n_targets


def ingest(adata: ad.AnnData, *, groupby: str, reference: str) -> IngestedState:
    if reference == LEGACY_ALL_OTHERS:
        warnings.warn(
            f"reference={LEGACY_ALL_OTHERS!r} is deprecated; pass the "
            f"ALL_OTHERS constant (or the string {ALL_OTHERS!r}) instead. "
            "The legacy spelling will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        reference = ALL_OTHERS
    if groupby not in adata.obs.columns:
        raise ValueError(
            f"groupby column {groupby!r} not in adata.obs (have: "
            f"{list(adata.obs.columns)})"
        )
    col = adata.obs[groupby]
    n_missing = int(col.isna().sum())
    if n_missing:
        raise ValueError(
            f"adata.obs[{groupby!r}] has {n_missing} cell(s) with a missing "
            f"(NaN/None) group label. Assign or drop them before calling de(): "
            f"pandas .astype(str) would otherwise turn them into a literal "
            f"'nan'/'None' group, silently skewing the comparison (a bogus "
            f"target in literal-reference mode, or polluting the rest-of-cells "
            f"reference in ALL_OTHERS mode)."
        )
    raw = col.astype(str).to_numpy()
    unique_labels, inverse = np.unique(raw, return_inverse=True)

    if reference == ALL_OTHERS:
        ref_label_idx = None
        target_labels = unique_labels.copy()
    else:
        ref_pos = np.where(unique_labels == reference)[0]
        if ref_pos.size == 0:
            raise ValueError(
                f"reference {reference!r} not found in adata.obs[{groupby!r}] "
                f"(have: {list(unique_labels)})"
            )
        ref_label_idx = int(ref_pos[0])
        keep = np.ones_like(unique_labels, dtype=bool)
        keep[ref_label_idx] = False
        target_labels = unique_labels[keep]

    return IngestedState(
        adata=adata,
        n_cells=int(adata.n_obs),
        n_genes=int(adata.n_vars),
        ref_label=reference,
        labels=inverse.astype(np.int32),
        unique_labels=unique_labels,
        ref_label_idx=ref_label_idx,
        target_labels=target_labels,
    )
