# src/gpudge/_filter.py
"""Per-(guide, gene) keep mask matching vci_dge/de.py:_target_mp_run."""
from __future__ import annotations

import torch


def per_guide_keep_mask(
    target_mean_arith: torch.Tensor,    # (n_guides, n_genes)
    ref_mean_arith: torch.Tensor,       # (n_genes,) for ref mode, or (n_guides, n_genes) for all_others
    *,
    threshold: float,
) -> torch.Tensor:
    """Return a bool (n_guides, n_genes) mask: True = keep.

    Matches CPU pipeline's `(mean_t > min_tpm) | (mean_r > min_tpm)`.
    """
    if ref_mean_arith.dim() == 1:
        # broadcast (n_genes,) → (1, n_genes)
        ref_b = ref_mean_arith[None, :]
    elif ref_mean_arith.shape != target_mean_arith.shape:
        raise ValueError(
            f"ref_mean_arith shape {tuple(ref_mean_arith.shape)} incompatible "
            f"with target_mean_arith {tuple(target_mean_arith.shape)}"
        )
    else:
        ref_b = ref_mean_arith
    return (target_mean_arith > threshold) | (ref_b > threshold)
