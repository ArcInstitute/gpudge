# src/gpudge/_means.py
"""Per-group arithmetic and geometric means of a (cells x genes) dense torch
tensor, computed via index_add_ (segment reduce)."""
from __future__ import annotations

import torch

MEAN_KINDS = ("arithmetic", "geometric")


def group_means(
    X: torch.Tensor,         # (n_cells, n_genes) float32, on GPU
    labels: torch.Tensor,    # (n_cells,) int32, group index per cell
    n_groups: int,
    kind: str = "arithmetic",
    inner_chunk: int = 512,  # genes per inner float64 cast pass
) -> torch.Tensor:
    """Per-(group, gene) mean. Returns (n_groups, n_genes) float64.

    Accumulation is done in float64 (for precision over many cells), but the
    float32 → float64 cast is sub-chunked across genes to bound the transient
    memory spike. At cell line 2 scale (2 M cells × 4 k genes) a single full-chunk
    cast would be 66 GB on GPU; this loop holds ≤ inner_chunk × n_cells × 8 B
    per inner pass instead.

    ``kind="geometric"`` computes ``expm1(mean(log1p(X)))``, defined only for
    ``X > -1``. Transforming/validating X is the caller's responsibility
    (gpudge never mutates X); out-of-domain inputs propagate **deterministically**
    rather than raising or clamping: ``X < -1`` → ``NaN`` (``log1p`` is NaN),
    and exactly ``X == -1`` → ``-1.0`` (``log1p(-1) = -inf`` →
    ``expm1(-inf) = -1.0``). (Pinned by test_geometric_mean_out_of_domain_*.)
    """
    if kind not in MEAN_KINDS:
        raise ValueError(f"kind must be one of {MEAN_KINDS}, got {kind!r}")
    if X.dim() != 2:
        raise ValueError(f"X must be 2-D, got shape {tuple(X.shape)}")

    n_cells, n_genes = X.shape
    if labels.shape != (n_cells,):
        raise ValueError(
            f"labels length {tuple(labels.shape)} != n_cells {n_cells}"
        )

    labels_long = labels.long()
    counts = torch.bincount(labels_long, minlength=n_groups).to(torch.float64)
    safe = counts.clone()
    safe[safe == 0] = 1.0

    sums = torch.zeros((n_groups, n_genes), dtype=torch.float64, device=X.device)
    # Sub-chunked accumulation: bound the float32 → float64 cast spike.
    for s in range(0, n_genes, inner_chunk):
        e = min(s + inner_chunk, n_genes)
        block = X[:, s:e]
        if kind == "geometric":
            block = torch.log1p(block)
        block64 = block.to(torch.float64)
        sub_sums = torch.zeros((n_groups, e - s),
                               dtype=torch.float64, device=X.device)
        sub_sums.index_add_(0, labels_long, block64)
        sums[:, s:e] = sub_sums
        del block64, sub_sums

    means = sums / safe[:, None]
    if kind == "geometric":
        means = torch.expm1(means)
    return means
