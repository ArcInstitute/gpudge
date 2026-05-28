# src/gpudge/_mwu.py
"""Mann–Whitney U (two-sided, tie-corrected, normal approximation) on torch.

Per (target group g, gene j): U1 = Σ over target cells of
  (count(ref < cell) + 0.5 · count(ref == cell))
computed via ``torch.searchsorted`` against a pre-sorted reference (sorted
once per gene chunk, transposed). Mathematically equivalent to per-pair MWU
but ~100× faster than re-sorting (group ∪ ref) per group. Variance includes
the standard tie correction and the continuity correction is applied (z =
(max(U1,U2) - mu - 0.5) / sqrt(var)) to match scipy.stats.mannwhitneyu with
method='asymptotic' and use_continuity=True (its defaults).
"""
from __future__ import annotations

import math
import torch

SQRT2 = math.sqrt(2.0)


def _rank_with_ties(
    X: torch.Tensor,  # (n_cells, n_genes) float32 or float64, on GPU
) -> tuple[torch.Tensor, torch.Tensor]:
    """Average ranks + tie correction term for a 2-D matrix (global ranks).

    Used by the all_others path: ranks are computed across ALL cells at once
    (not pairwise), so rank_sum per group gives U_g for 1-vs-rest.

    Returns:
        ranks   (n_cells, n_genes) float64 — average rank per cell per gene
        tie_term (n_genes,) float64 — Σ(t^3 - t) per gene (for variance correction)
    """
    n_cells, n_genes = X.shape
    device = X.device
    Xd = X.to(torch.float64)

    sorted_vals, sort_idx = torch.sort(Xd, dim=0)  # (n_cells, n_genes)
    N = n_cells

    # Tie groups: boundary where value changes along the cell axis
    eq = torch.zeros_like(sorted_vals, dtype=torch.bool)
    eq[1:] = sorted_vals[1:] == sorted_vals[:-1]
    group_id = (~eq).cumsum(dim=0) - 1   # (n_cells, n_genes) int64

    pos = torch.arange(1, N + 1, dtype=torch.float64,
                       device=device).unsqueeze(1).expand(-1, n_genes)

    max_tgroups = int(group_id[-1].max().item()) + 1
    rank_sum_buf = torch.zeros((max_tgroups, n_genes), dtype=torch.float64,
                               device=device)
    rank_cnt_buf = torch.zeros((max_tgroups, n_genes), dtype=torch.float64,
                               device=device)
    rank_sum_buf.scatter_add_(0, group_id, pos)
    rank_cnt_buf.scatter_add_(0, group_id, torch.ones_like(pos))
    avg_per_group = rank_sum_buf / rank_cnt_buf.clamp_min(1)

    # Ranks in sorted order, then undo sort to get per-cell ranks
    ranks_sorted = avg_per_group.gather(0, group_id)   # (n_cells, n_genes)
    inv_idx = sort_idx.argsort(dim=0)
    ranks = ranks_sorted.gather(0, inv_idx)            # (n_cells, n_genes)

    tie_term = ((rank_cnt_buf ** 3) - rank_cnt_buf).sum(dim=0)  # (n_genes,)
    return ranks, tie_term


def _tie_term_per_gene(sorted_values: torch.Tensor) -> torch.Tensor:
    """Σ(t^3 - t) per gene for a (n_genes, k) sorted tensor.

    Uses torch.unique_consecutive per row. The Python loop adds ~0.1ms per
    gene, negligible relative to the rest of the chunk loop on cell line 2 scale.
    """
    n_genes = sorted_values.shape[0]
    out = torch.empty(n_genes, dtype=torch.float64, device=sorted_values.device)
    for i in range(n_genes):
        _v, c = torch.unique_consecutive(sorted_values[i], return_counts=True)
        c = c.to(torch.float64)
        out[i] = torch.sum(c * c * c - c)
    return out


def mwu_one_group(
    sorted_ref: torch.Tensor,    # (n_genes, n_ref) float32, sorted along dim=1
    ref_tie_term: torch.Tensor,  # (n_genes,) float64
    group_T: torch.Tensor,       # (n_genes, m) float32
    n_ref: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One target group vs reference MWU.

    Returns (u1, p) both (n_genes,) float64. ``u1`` is the U1 statistic for
    the target group (= what scipy.stats.mannwhitneyu returns by default).
    Variance includes tie correction and continuity correction (matches
    scipy method='asymptotic', use_continuity=True).

    Degenerate cases (``m == 0`` or ``n_ref == 0``) return zeros for U and
    NaN for p; downstream BH-FDR preserves NaNs.
    """
    n_genes = group_T.shape[0]
    m = group_T.shape[1]
    if m == 0 or n_ref == 0:
        device = group_T.device
        u1 = torch.zeros(n_genes, dtype=torch.float64, device=device)
        p = torch.full((n_genes,), float("nan"),
                       dtype=torch.float64, device=device)
        return u1, p

    # U1 = Σ over target cells of (count(ref < cell) + 0.5 · count(ref == cell))
    ref_left = torch.searchsorted(sorted_ref, group_T, right=False)
    ref_right = torch.searchsorted(sorted_ref, group_T, right=True)
    ref_less = ref_left.to(torch.float64)
    ref_equal = (ref_right - ref_left).to(torch.float64)
    u1 = (ref_less + 0.5 * ref_equal).sum(dim=1)

    # Tie correction
    group_sorted, _ = torch.sort(group_T, dim=1)
    if m == 1:
        run_start = torch.ones_like(group_sorted, dtype=torch.bool)
    else:
        run_start = torch.empty_like(group_sorted, dtype=torch.bool)
        run_start[:, 0] = True
        run_start[:, 1:] = group_sorted[:, 1:] != group_sorted[:, :-1]

    gl = torch.searchsorted(group_sorted, group_sorted, right=False)
    gr = torch.searchsorted(group_sorted, group_sorted, right=True)
    gc = (gr - gl).to(torch.float64)
    rl = torch.searchsorted(sorted_ref, group_sorted, right=False)
    rr = torch.searchsorted(sorted_ref, group_sorted, right=True)
    rc = (rr - rl).to(torch.float64)
    combined = rc + gc
    delta = combined ** 3 - combined - (rc ** 3 - rc)
    delta = torch.where(run_start, delta, torch.zeros_like(delta))
    tie_term = ref_tie_term + delta.sum(dim=1)

    n1 = float(m)
    n2 = float(n_ref)
    N = n1 + n2
    u2 = n1 * n2 - u1
    u = torch.maximum(u1, u2)
    mu = n1 * n2 / 2.0
    var_inner = (N + 1.0) - tie_term / (N * (N - 1.0))
    s_sq = n1 * n2 / 12.0 * var_inner
    s_sq = s_sq.clamp_min(torch.finfo(torch.float64).tiny)
    z = (u - mu - 0.5) / torch.sqrt(s_sq)
    p = torch.erfc(z / SQRT2).clamp_(0.0, 1.0)
    return u1, p


def mwu_ref(
    X: torch.Tensor,        # (n_cells, n_genes) float32, any device
    labels: torch.Tensor,   # (n_cells,) int32
    n_groups: int,
    *,
    ref_idx: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-(group, gene) U statistic and two-sided p-value vs reference group.

    Uses ``torch.searchsorted`` against a pre-sorted reference (sorted once
    per gene chunk, transposed). Mathematically equivalent to per-pair MWU
    (matches scipy.stats.mannwhitneyu method='asymptotic', use_continuity=True
    when there are enough samples for the normal approximation) but ~100×
    faster than naively re-sorting (group ∪ ref) per group.

    Returns (U, p) both (n_groups, n_genes) float64; the ref_idx row is filled
    with U=0, p=NaN as a sentinel. The U value is the U1 statistic for the
    target group (not max(U1, U2)) — matches what scipy returns by default
    and what CPU pdex stores in target_de.parquet's ``statistic`` column.

    This is a convenience wrapper around ``mwu_one_group`` for small in-memory
    use (tests, ad-hoc). The ``de()`` entry point drives ``mwu_one_group``
    directly for memory-efficient per-group iteration at scale.
    """
    n_cells, n_genes = X.shape
    device = X.device
    labels_long = labels.long()

    ref_mask = (labels_long == ref_idx)
    n_ref = int(ref_mask.sum().item())
    ref_T = X[ref_mask].T.contiguous().to(torch.float32)         # (n_genes, n_ref)
    sorted_ref, _ = torch.sort(ref_T, dim=1)                     # (n_genes, n_ref)
    del ref_T
    ref_tie_term = _tie_term_per_gene(sorted_ref)                # (n_genes,)

    U_out = torch.zeros((n_groups, n_genes), dtype=torch.float64, device=device)
    p_out = torch.full((n_groups, n_genes), float("nan"),
                       dtype=torch.float64, device=device)

    for g in range(n_groups):
        if g == ref_idx:
            U_out[g] = 0.0   # p stays NaN
            continue
        g_mask = (labels_long == g)
        m = int(g_mask.sum().item())
        if m == 0:
            continue
        group_T = X[g_mask].T.contiguous().to(torch.float32)     # (n_genes, m)
        u1, p = mwu_one_group(sorted_ref, ref_tie_term, group_T, n_ref=n_ref)
        U_out[g] = u1
        p_out[g] = p

    return U_out, p_out
