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



# Stringified spellings of a missing value. pandas/numpy .astype(str) turns
# NaN/None/pd.NA into exactly these, which is how an unassigned cell becomes a
# LITERAL group label instead of an error. The in-memory path catches this at the
# source (it still has the nullable column); the streaming backends only ever see
# an already-stringified group table, so they screen the strings instead.
# (ultrareview 2026-08)
# Exactly the four strings pandas/numpy .astype(str) can produce for a missing
# value. Deliberately NOT "NaN"/"NAN"/"" : those cannot arise from the conversion,
# so rejecting them would refuse archives whose labels the in-memory ingest()
# accepts -- breaking backend parity to guard against nothing. (codex review)
MISSING_LABEL_SPELLINGS = frozenset({"nan", "None", "<NA>", "NaT"})


def reject_missing_group_labels(labels, *, where: str, remedy: str) -> None:
    """Raise if any label is the stringified form of a missing value.

    ``labels`` is an iterable of already-stringified group labels. A group that is
    genuinely *named* 'nan' is indistinguishable at this level and must be
    renamed -- the message says so.
    """
    bad = sorted({lab for lab in labels if lab in MISSING_LABEL_SPELLINGS})
    if bad:
        raise ValueError(
            f"{where}: group label(s) {bad} are the stringified form of a missing "
            f"(NaN/None) value, so those cells were unassigned when the data was "
            f"written. Treating them as a target group silently adds a bogus "
            f"perturbation to the result. {remedy} If a group is genuinely named "
            f"{bad[0]!r}, rename it."
        )

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
