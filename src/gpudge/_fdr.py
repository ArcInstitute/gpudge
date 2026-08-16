# src/gpudge/_fdr.py
"""Per-group Benjamini–Hochberg FDR.

When numba is available (the `[fast]` extra), two numba kernels do the
work:

  1. **counting-sort by group_id** (O(N + G), single-threaded but a
     tight loop with no allocations after the initial counts/cursors).
     ``np.argsort(group_id, kind="stable")`` was ~7 s on a 52.7M-row
     CCL_2 run; counting-sort is ~0.4 s for the same input.
  2. **per-segment BH** (argsort by p, m*p/rank, running-min-from-right,
     clip), parallel across the ~4672 groups. Writes directly into the
     final original-row layout, so no outer inverse-permute pass.

cProfile baseline (pre-T2) was ~6.5 s of de() wall in the 4672-iteration
scipy.false_discovery_control loop.

When numba isn't installed, falls back to ``_bh_per_group_numpy`` —
a pure-numpy implementation of the same algorithm (single-threaded,
per-group Python loop). The fallback handles ±inf inputs the same
way the kernels do; this module no longer calls scipy at runtime
(``scipy`` itself is still a gpudge dependency for other modules).
"""
from __future__ import annotations

import numpy as np
import torch

from ._csr_dense import HAS_NUMBA

if HAS_NUMBA:
    import numba as nb

    @nb.njit(cache=True, boundscheck=False)
    def _counting_sort_by_group(
        g_not_nan: np.ndarray,   # (n_not_nan,) int64, values in [0, n_groups)
        p_not_nan: np.ndarray,   # (n_not_nan,) float64
        n_groups: int,
    ):
        """Counting-sort rows by group_id.

        Returns (p_sorted_by_g, order_g, starts, stops) where:
          - p_sorted_by_g[i] = p_not_nan[order_g[i]]
          - rows with group g occupy [starts[g], stops[g])
          - order_g[i] is the original row index for sorted position i
        """
        n = g_not_nan.shape[0]
        counts = np.zeros(n_groups, dtype=np.int64)
        for i in range(n):
            counts[g_not_nan[i]] += 1
        starts = np.empty(n_groups, dtype=np.int64)
        stops = np.empty(n_groups, dtype=np.int64)
        s = 0
        for k in range(n_groups):
            starts[k] = s
            s += counts[k]
            stops[k] = s
        cursor = starts.copy()
        p_sorted = np.empty(n, dtype=np.float64)
        order = np.empty(n, dtype=np.int64)
        for i in range(n):
            g = g_not_nan[i]
            pos = cursor[g]
            p_sorted[pos] = p_not_nan[i]
            order[pos] = i
            cursor[g] += 1
        return p_sorted, order, starts, stops

    @nb.njit(parallel=True, cache=True, boundscheck=False)
    def _bh_per_segment_to_original(
        p_sorted_by_g: np.ndarray,
        starts: np.ndarray,
        stops: np.ndarray,
        order_g: np.ndarray,
        out_not_nan: np.ndarray,
    ) -> None:
        """Per-segment BH; writes adjusted q-values directly into the
        original-row layout via the ``order_g`` permutation.

        Each segment: argsort by p, m*p/rank, running-min-from-right,
        clip. The inverse permute (counting-sort order → original-row
        order) is fused into the final scatter, saving a second
        full-length permutation pass.
        """
        n_seg = starts.shape[0]
        for k in nb.prange(n_seg):  # ty: ignore[not-iterable]
            s = starts[k]
            e = stops[k]
            m = e - s
            if m == 0:
                continue
            sub = p_sorted_by_g[s:e]
            local_order = np.argsort(sub, kind="quicksort")
            unadj = np.empty(m, dtype=np.float64)
            for i in range(m):
                unadj[i] = m * sub[local_order[i]] / (i + 1)
            for i in range(m - 2, -1, -1):
                if unadj[i + 1] < unadj[i]:
                    unadj[i] = unadj[i + 1]
            for i in range(m):
                v = unadj[i]
                # Clip to [0, 1]. Matches historical scipy behaviour. The
                # low-side clip is defensive for inf/out-of-range inputs;
                # finite p in [0, 1] never produces a negative unadj.
                if v > 1.0:
                    v = 1.0
                elif v < 0.0:
                    v = 0.0
                out_not_nan[order_g[s + local_order[i]]] = v


def bh_per_group(
    p: torch.Tensor,        # (N,) float
    group_id: torch.Tensor, # (N,) int
    n_groups: int,
) -> torch.Tensor:
    """Returns (N,) float adjusted p-values; per-group BH.

    NaN p stays NaN. ``+/-inf`` p flows through the BH math: ``+inf`` ends
    up clipped to 1.0 and ``-inf`` to 0.0, matching the symmetric clip in
    the kernel and what historical scipy.false_discovery_control would
    have produced on the same inputs (modern scipy now rejects inf).
    """
    # copy=False is a no-op when .numpy() already returns float64/int64
    # (the common case from torch.Tensor.cpu().numpy()); without it,
    # astype always copies, which on CCL_2 means an extra ~420 MB float64
    # buffer and ~420 MB int64 buffer per call.
    p_np = p.detach().cpu().numpy().astype(np.float64, copy=False)
    g_np = group_id.detach().cpu().numpy().astype(np.int64, copy=False)
    N = p_np.shape[0]

    out = np.full(N, np.nan, dtype=np.float64)
    # Only NaN passes through — ±inf participates in BH (and gets clipped
    # to 1). isfinite would exclude inf too, which would silently turn it
    # into NaN in the output.
    not_nan = ~np.isnan(p_np)
    if not not_nan.any():
        return torch.from_numpy(out)

    not_nan_idx = np.flatnonzero(not_nan)
    p_not_nan = p_np[not_nan_idx]
    g_not_nan = g_np[not_nan_idx]
    n_not_nan = p_not_nan.shape[0]

    # The numba counting-sort runs with boundscheck=False; an out-of-range
    # group_id would otherwise corrupt memory silently. The cost is one
    # min + one max over n_not_nan int64 (~50 ms on CCL_2, ~1 % of the function).
    if n_groups <= 0:
        raise ValueError(
            f"n_groups={n_groups}; must be >= 1 when any rows have non-NaN p."
        )
    g_min = int(g_not_nan.min())
    g_max = int(g_not_nan.max())
    if g_min < 0 or g_max >= n_groups:
        raise ValueError(
            f"group_id values must be in [0, n_groups={n_groups}); "
            f"observed range [{g_min}, {g_max}]."
        )

    if HAS_NUMBA:
        p_sorted, order_g, starts, stops = _counting_sort_by_group(
            g_not_nan, p_not_nan, n_groups
        )
        out_not_nan = np.empty(n_not_nan, dtype=np.float64)
        _bh_per_segment_to_original(p_sorted, starts, stops, order_g, out_not_nan)
    else:
        out_not_nan = _bh_per_group_numpy(p_not_nan, g_not_nan, n_groups)

    out[not_nan_idx] = out_not_nan
    return torch.from_numpy(out)


def _bh_per_group_numpy(
    p_not_nan: np.ndarray,
    g_not_nan: np.ndarray,
    n_groups: int,
) -> np.ndarray:
    """Pure-numpy BH per group; the no-numba fallback for bh_per_group.

    Mirrors the numba kernel's algorithm exactly so the two paths agree
    bit-for-bit on the same input. Notably handles ±inf p-values (which
    modern scipy.false_discovery_control rejects) by letting them flow
    through the BH math and clipping to [0, 1] — same as the kernel.

    Slower than the numba path (single-threaded, per-group argsort
    overhead in Python). Install ``gpudge[fast]`` for the numba kernels.
    """
    n = p_not_nan.shape[0]
    order_g = np.argsort(g_not_nan, kind="stable")
    g_sorted = g_not_nan[order_g]
    p_sorted = p_not_nan[order_g]
    starts = np.searchsorted(g_sorted, np.arange(n_groups), side="left")
    stops = np.searchsorted(g_sorted, np.arange(n_groups), side="right")
    out_sorted = np.empty(n, dtype=np.float64)
    for gi in range(n_groups):
        s, e = int(starts[gi]), int(stops[gi])
        if s == e:
            continue
        sub = p_sorted[s:e]
        m = e - s
        local_order = np.argsort(sub, kind="quicksort")
        sub_sorted = sub[local_order]
        unadj = m * sub_sorted / np.arange(1, m + 1, dtype=np.float64)
        # Running min from right, in-place on the reversed view.
        np.minimum.accumulate(unadj[::-1], out=unadj[::-1])
        # Symmetric clip to [0, 1]: matches the kernel and historical
        # scipy.false_discovery_control behaviour on inf/out-of-range
        # inputs.
        np.clip(unadj, 0.0, 1.0, out=unadj)
        out_sorted_local = np.empty(m, dtype=np.float64)
        out_sorted_local[local_order] = unadj
        out_sorted[s:e] = out_sorted_local
    out_not_nan = np.empty(n, dtype=np.float64)
    out_not_nan[order_g] = out_sorted
    return out_not_nan
