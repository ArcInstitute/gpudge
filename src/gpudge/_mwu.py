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
from statistics import NormalDist

import torch

from ._lfc import lfc_scale_factor
from ._taustar import TAUSTAR_SE_COLUMNS, TAUSTAR_SE_LEVEL

SQRT2 = math.sqrt(2.0)

# Tie-term scratch budget (elements): _tie_term_per_gene processes genes in
# blocks of `_TIE_BLOCK_ELEMS // k` so the O(block x k) int64 run_id + f64 ones
# scratch stays bounded for a wide reference pool. Exposed as a module constant
# so tests can force the multi-block path (otherwise block >= n_genes). (L10)
_TIE_BLOCK_ELEMS = 64_000_000


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
    """Σ(t^3 - t) per gene for a (n_genes, k) tensor sorted along dim=1.

    Vectorized: mark tie-run boundaries along the sorted axis, count run
    lengths with one scatter_add, then Σ(t^3 - t) per row. Replaces a Python
    per-gene loop over torch.unique_consecutive (thousands of kernel launches
    per chunk at CCL_2 scale). (ultrareview perf.)
    """
    n_genes, k = sorted_values.shape
    out = torch.zeros(n_genes, dtype=torch.float64,
                      device=sorted_values.device)
    if n_genes == 0 or k == 0:
        return out                                      # no ties possible
    # Process genes in blocks so the O(block x k) int64 run_id + f64 ones
    # scratch (~16 B/elem) stays bounded for a wide reference pool — it must
    # not balloon past the gene chunk's own ranking buffers. (Codex review.)
    block = max(1, min(n_genes, _TIE_BLOCK_ELEMS // k))
    for s in range(0, n_genes, block):
        sv = sorted_values[s:s + block]
        new_run = torch.ones_like(sv, dtype=torch.bool)
        new_run[:, 1:] = sv[:, 1:] != sv[:, :-1]        # value changes
        run_id = new_run.cumsum(dim=1) - 1              # (block, k) run index
        n_runs = int(run_id[:, -1].max().item()) + 1
        counts = torch.zeros((sv.shape[0], n_runs), dtype=torch.float64,
                             device=sv.device)
        counts.scatter_add_(1, run_id,
                            torch.ones_like(sv, dtype=torch.float64))
        out[s:s + block] = (counts ** 3 - counts).sum(dim=1)
    return out


def _selfties(group_sorted: torch.Tensor):
    """(gc, run_start) for a (n_genes, m) target block ALREADY sorted on dim=1.

    ``gc`` is each cell's own tie-run length; ``run_start`` marks the first cell
    of each run. TAU-INVARIANT under the float64 scaling of spec 3.2b: adjacent
    float32 values differ by >= 2**-24 relative and float64 rounding perturbs by
    <= 2**-53, so scaling cannot merge them and the tie structure is preserved
    exactly. Callers sweeping a tau grid therefore compute this ONCE. (In a
    float32 implementation it would NOT be invariant -- see spec 3.4a.)
    """
    m = group_sorted.shape[1]
    if m == 1:
        run_start = torch.ones_like(group_sorted, dtype=torch.bool)
    else:
        run_start = torch.empty_like(group_sorted, dtype=torch.bool)
        run_start[:, 0] = True
        run_start[:, 1:] = group_sorted[:, 1:] != group_sorted[:, :-1]
    gl = torch.searchsorted(group_sorted, group_sorted, right=False)
    gr = torch.searchsorted(group_sorted, group_sorted, right=True)
    gc = (gr - gl).to(torch.float64)
    return gc, run_start


def _sorted_and_selfties(group_T: torch.Tensor):
    """(group_sorted, gc, run_start) for a (n_genes, m) target block.

    The SORT ORDER is tau-independent: multiplication by a positive float is
    monotone under round-to-nearest, so sort(g * s) == sort(g) * s elementwise
    (spec 3.4c). Callers sweeping a tau grid therefore sort ONCE and derive each
    scaled sorted target with a multiply -- never a re-sort.
    """
    group_sorted, _ = torch.sort(group_T, dim=1)
    gc, run_start = _selfties(group_sorted)
    return group_sorted, gc, run_start


def _bounds(sorted_ref: torch.Tensor, values: torch.Tensor):
    """(left, right) insertion points of ``values`` in ``sorted_ref`` (dim=1).

    ``values`` may be float32 (base path) or float64 (directional path, spec
    3.2b). NEVER hand float64 values straight to torch.searchsorted against a
    float32 boundary: ATen promotes to a common supertype and executes
    ``raw_boundaries.to(common_stype)``, converting the ENTIRE (chunk, n_ref)
    reference on every call -- measured 0.05 ms (f32 query) vs 65.26 ms (f64
    query) against a 160 MB reference, a 1300x slowdown.

    Instead, round the query to the nearest float32 ``q``, do the two float32
    searchsorted, and correct the boundary from the sign of (values - q). This
    is EXACT: ``q`` is the nearest float32 to ``values``, so no float32 value
    lies strictly between them -- everything <= q is below ``values`` when
    ``values > q``, and everything >= q is above when ``values < q``. Verified
    bit-identical to the native (upcasting) result on random bit patterns across
    the whole float32 range, an edge fixture (+/-0.0, min/max subnormal, min
    normal, max finite, +inf) at tau in {0, 0.25, 1, 30} both directions, exact
    ties, and the spec 3.2b boundary counterexample. 264x faster (0.24 ms).

    Same-dtype callers take the direct path, so the base path stays
    bit-identical by construction.
    """
    if values.dtype == sorted_ref.dtype:
        return (torch.searchsorted(sorted_ref, values, right=False),
                torch.searchsorted(sorted_ref, values, right=True))
    q = values.to(sorted_ref.dtype)
    lq = torch.searchsorted(sorted_ref, q, right=False)
    rq = torch.searchsorted(sorted_ref, q, right=True)
    q_back = q.to(values.dtype)
    return (torch.where(values > q_back, rq, lq),
            torch.where(values < q_back, lq, rq))


def _u1_against(sorted_ref: torch.Tensor,
                group_like: torch.Tensor) -> torch.Tensor:
    """U1 = sum over target cells of count(ref < cell) + 0.5*count(ref == cell).

    ``group_like`` is the target block in its ORIGINAL (n_genes, m) layout --
    float32 (base path) or float64 (directional path, spec 3.2b); ``_bounds``
    handles the mixed-dtype case without copying the reference. Pass the
    unsorted form: the sum is over the same multiset either
    way, but float64 addition is order-dependent, and using the base path's layout
    keeps the tau=0 directional U1 bit-identical to the base U1.
    """
    ref_left, ref_right = _bounds(sorted_ref, group_like)
    ref_less = ref_left.to(torch.float64)
    ref_equal = (ref_right - ref_left).to(torch.float64)
    return (ref_less + 0.5 * ref_equal).sum(dim=1)


def _cross_tie_delta(sorted_ref: torch.Tensor, group_sorted_like: torch.Tensor,
                     gc: torch.Tensor, run_start: torch.Tensor) -> torch.Tensor:
    """Cross-group tie contribution to sum(t^3 - t), per gene.

    MUST be recomputed against the SCALED target for each (tau, direction):
    0 * s == 0 exactly for any s > 0, so the zero-tie mass survives the scaling
    and this term is large, not negligible, on zero-inflated counts (spec 3.4b).
    This is the ONLY per-combo tie work -- ``gc``/``run_start`` are tau-invariant
    under float64 scaling (spec 3.4a) and are passed in unchanged.
    """
    rl, rr = _bounds(sorted_ref, group_sorted_like)
    rc = (rr - rl).to(torch.float64)
    return _tie_delta_from_rc(rc, gc, run_start)


def _tie_delta_from_rc(rc: torch.Tensor, gc: torch.Tensor,
                       run_start: torch.Tensor) -> torch.Tensor:
    """sum(t^3 - t) contributed by merging reference counts ``rc`` into the
    target's tie runs. Shared by the measured (``_cross_tie_delta``) and the
    generic (``_cross_tie_generic``) callers so the two cannot drift.
    """
    combined = rc + gc
    delta = combined ** 3 - combined - (rc ** 3 - rc)
    delta = torch.where(run_start, delta, torch.zeros_like(delta))
    return delta.sum(dim=1)


def _cross_tie_generic(group_sorted: torch.Tensor, gc: torch.Tensor,
                       run_start: torch.Tensor,
                       n0: torch.Tensor) -> torch.Tensor:
    """Cross-group tie contribution at a GENERIC (non-coincident) shift.

    tau*'s load-bearing simplification (tau* spec 3.4). For a target tie-run
    whose value is NONZERO, the scaled value generically equals no reference
    value, so its ``rc`` is 0 and the contribution collapses to ``gc**3 - gc``
    -- a constant. The ZERO run is the exception and is constant for the
    opposite reason: ``0 * s == 0`` for every ``s > 0``, so its ``rc == n0``
    never moves either. The whole term is therefore INDEPENDENT of the shift,
    which is what lets sigma hoist out of the bisection loop and turns the
    crossing level into a constant.

    NOT obtainable by evaluating ``_cross_tie_delta`` at delta = 0: at
    ``s == 1.0`` the raw target values coincide with raw reference values in
    bulk, which is precisely the BASE test's tie structure and is not generic.
    ``tests/test_mwu_taustar.py`` pins both halves of that statement.

    In floating point the coincidence set has POSITIVE width (a plateau), not
    measure zero -- this is exactly why tau* is defined against the
    coincidence-free p-function ``p~`` rather than the measured one.
    """
    rc = torch.where(
        group_sorted == 0,
        n0.unsqueeze(1).expand_as(group_sorted),
        torch.zeros((), dtype=torch.float64, device=group_sorted.device),
    )
    return _tie_delta_from_rc(rc, gc, run_start)


def _zero_counts(sorted_ref: torch.Tensor,
                 group_sorted: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(n0, m0) per gene: reference zeros and target zeros, both float64.

    ``n0`` comes from ``_bounds`` rather than an ``== 0`` reduction so it costs
    one searchsorted against the already-sorted reference instead of a full
    (n_genes, n_ref) comparison, and so negative values (not produced by the
    count pipeline, but not forbidden either) are counted correctly.
    """
    n_genes = sorted_ref.shape[0]
    zeros = torch.zeros((n_genes, 1), dtype=sorted_ref.dtype,
                        device=sorted_ref.device)
    zl, zr = _bounds(sorted_ref, zeros)
    n0 = (zr - zl).squeeze(1).to(torch.float64)
    m0 = (group_sorted == 0).sum(dim=1).to(torch.float64)
    return n0, m0


def _u1_reach_limits(m: int, m0: torch.Tensor, n_ref: int,
                     n0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(U1(+inf), U1(-inf)) per gene -- the reachable range of U1 (spec 3.5).

    As s -> 0+ every POSITIVE target value stays strictly positive, so it still
    outranks the zero references while falling below every positive one; every
    ZERO target value stays tied with the zero references. As s -> inf every
    positive target value passes every reference. Hence:

        U1(+inf) = n0 * (m - m0/2)                  the minimum
        U1(-inf) = (m - m0) * n_ref + m0 * n0 / 2   the maximum

    A level outside this range has no crossing and is resolved to a signed
    infinity WITHOUT probing. The two coincide exactly when m0 == m or n0 ==
    n_ref (solve n0*(m-m0) == n_ref*(m-m0)); most of those genes still have a
    perfectly informative infinite answer, so do NOT blanket them to NaN.
    """
    mf = float(m)
    nf = float(n_ref)
    u1_min = n0 * (mf - m0 / 2.0)
    u1_max = (mf - m0) * nf + m0 * n0 / 2.0
    return u1_min, u1_max


def _z_from_level(q: float) -> float:
    """Phi^-1(1 - q), the upper-tail normal quantile, in float64.

    Computed as ``-Phi^-1(q)`` via stdlib ``statistics.NormalDist`` (Wichura
    AS241), so gpudge takes no scipy runtime dependency for it.
    q = 0.5 -> 0.0 exactly (the Hodges-Lehmann point); q = 0.025 -> 1.959964.

    Deliberately NOT ``sqrt(2) * erfinv(1 - 2q)``. That form evaluates the
    argument at ``1 - 2q``, which loses the whole tail: it is already wrong in
    the 3rd significant figure at q = 1e-16 (8.2095 vs 8.2221) and rounds to
    exactly 1.0 for q <= 1e-17, making erfinv return +inf. The level then
    becomes +/-inf and EVERY gene resolves to a signed infinity -- silently,
    for a q that ``normalize_taustar_spec`` accepts as valid. Reflecting
    through the lower tail keeps full precision down to q ~ 1e-300 and agrees
    with ``scipy.stats.norm.isf`` to within 1-2 ULP across that range.
    """
    # + 0.0 normalises the -0.0 that inv_cdf(0.5) returns, so the documented
    # "q = 0.5 -> 0.0 exactly" is literally true.
    return -NormalDist().inv_cdf(float(q)) + 0.0


def _s_sq_of(tie_term: torch.Tensor, m: int, n_ref: int) -> torch.Tensor:
    """Tie-corrected variance of U, clamped away from zero."""
    n1 = float(m)
    n2 = float(n_ref)
    N = n1 + n2
    var_inner = (N + 1.0) - tie_term / (N * (N - 1.0))
    s_sq = n1 * n2 / 12.0 * var_inner
    return s_sq.clamp_min(torch.finfo(torch.float64).tiny)


def _p_two_sided(u1: torch.Tensor, tie_term: torch.Tensor,
                 m: int, n_ref: int) -> torch.Tensor:
    """Two-sided continuity-corrected p from max(U1, U2). Clamped to [0, 1]
    because erfc in [0, 2] exceeds 1 when z < 0."""
    n1 = float(m)
    n2 = float(n_ref)
    u2 = n1 * n2 - u1
    u = torch.maximum(u1, u2)
    mu = n1 * n2 / 2.0
    z = (u - mu - 0.5) / torch.sqrt(_s_sq_of(tie_term, m, n_ref))
    return torch.erfc(z / SQRT2).clamp_(0.0, 1.0)


def _p_one_sided(stat: torch.Tensor, tie_term: torch.Tensor,
                 m: int, n_ref: int) -> torch.Tensor:
    """Upper-tail continuity-corrected p for ``stat``.

    NO clamp: 0.5 * erfc(...) is in [0, 1] by construction. Do not copy the
    two-sided clamp here.
    """
    mu = float(m) * float(n_ref) / 2.0
    z = (stat - mu - 0.5) / torch.sqrt(_s_sq_of(tie_term, m, n_ref))
    return 0.5 * torch.erfc(z / SQRT2)


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

    u1 = _u1_against(sorted_ref, group_T)
    group_sorted, gc, run_start = _sorted_and_selfties(group_T)
    tie_term = ref_tie_term + _cross_tie_delta(sorted_ref, group_sorted, gc,
                                               run_start)
    p = _p_two_sided(u1, tie_term, m, n_ref)
    return u1, p


def mwu_one_group_lfc(
    sorted_ref: torch.Tensor,    # (n_genes, n_ref) float32, sorted along dim=1
    ref_tie_term: torch.Tensor,  # (n_genes,) float64 -- the UNSHIFTED term
    group_T: torch.Tensor,       # (n_genes, m) float32
    n_ref: int,
    *,
    lfc_combos,                  # tuple[(tau: float, direction: str), ...]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """MWU with an effect-size floor: the base two-sided test plus one-sided
    tests against the composite nulls H0 log2FC <= +tau (up) / H0 log2FC >=
    -tau (down).

    The shift is applied to the **target**, not the reference (spec 3.2a):
    U(T, R*f) == U(T/f, R) exactly, because the MWU depends only on the ordering
    of the pooled sample. So ``sorted_ref`` and ``ref_tie_term`` are used
    verbatim for every combo -- there is no scaled reference anywhere, no
    chunk-level precompute, and no (chunk, n_ref) per-combo buffer.

    The scaling is done in **float64** (spec 3.2b). float32 scaling lands on tie
    boundaries often enough to change p-values by orders of magnitude (measured:
    p = 2.7e-18 vs p = 0.50 on a constructed fixture), and float64 scaling is
    additionally INJECTIVE on float32 inputs -- adjacent float32 values differ by
    >= 2**-24 relative while float64 rounding perturbs by <= 2**-53 -- which is
    why ``gc``/``run_start`` below are computed ONCE instead of per combo.
    All mixed-dtype comparisons go through ``_bounds``, never through a raw
    ``torch.searchsorted`` -- torch would upcast and COPY the whole float32
    reference on every call (0.05 ms -> 65.26 ms against a 160 MB reference).

    Returns ``(u1, p, dir_u1, dir_p)``. ``u1``/``p`` are bit-identical to
    ``mwu_one_group`` (same helpers, no scaling). ``dir_u1``/``dir_p`` are
    ``(len(lfc_combos), n_genes)`` float64; ``dir_u1`` is U1 of the shifted
    comparison for BOTH directions (U2 = m*n_ref - U1 is derivable, and one
    scale keeps every statistic column comparable).
    """
    n_genes = group_T.shape[0]
    m = group_T.shape[1]
    n_combos = len(lfc_combos)
    device = group_T.device

    if m == 0 or n_ref == 0:
        u1 = torch.zeros(n_genes, dtype=torch.float64, device=device)
        p = torch.full((n_genes,), float("nan"),
                       dtype=torch.float64, device=device)
        dir_u1 = torch.zeros((n_combos, n_genes), dtype=torch.float64,
                             device=device)
        dir_p = torch.full((n_combos, n_genes), float("nan"),
                           dtype=torch.float64, device=device)
        return u1, p, dir_u1, dir_p

    # --- base two-sided test: byte-identical to mwu_one_group ---------------
    u1 = _u1_against(sorted_ref, group_T)
    group_sorted, gc, run_start = _sorted_and_selfties(group_T)
    tie_term = ref_tie_term + _cross_tie_delta(sorted_ref, group_sorted, gc,
                                               run_start)
    p = _p_two_sided(u1, tie_term, m, n_ref)

    # --- hoisted once for the whole grid (spec 3.2b) ------------------------
    # gc / run_start are reused UNCHANGED below: float64 scaling cannot merge
    # distinct float32 values, so the target's self-tie structure is
    # tau-invariant (spec 3.4a). Do NOT drop the .to(float64) -- it is what
    # makes that true, and what keeps the comparison off the float32 tie
    # boundary.
    gT64 = group_T.to(torch.float64)
    gs64 = group_sorted.to(torch.float64)

    # --- per (tau, direction): two (n_genes, m) multiplies + 4 searchsorted --
    dir_u1 = torch.empty((n_combos, n_genes), dtype=torch.float64,
                         device=device)
    dir_p = torch.empty((n_combos, n_genes), dtype=torch.float64,
                        device=device)
    mn = float(m) * float(n_ref)
    for k, (tau, direction) in enumerate(lfc_combos):
        s = lfc_scale_factor(tau, direction)   # float64; NEVER 1/(2**tau)
        # U1 sums over the ORIGINAL layout so tau=0 is bit-identical to `u1`
        # (float64 .sum is order-dependent).
        u1_s = _u1_against(sorted_ref, gT64 * s)
        # Sorted order is preserved by a positive scale (spec 3.4c) -- multiply,
        # NEVER re-sort. LOAD-BEARING: 0 * s == 0, so the zero-tie mass
        # survives; the CROSS delta must be recomputed against the scaled
        # target (spec 3.4b). It is the only per-combo tie work.
        tie_s = ref_tie_term + _cross_tie_delta(sorted_ref, gs64 * s, gc,
                                                run_start)
        stat = u1_s if direction == "up" else (mn - u1_s)
        dir_u1[k] = u1_s
        dir_p[k] = _p_one_sided(stat, tie_s, m, n_ref)
    return u1, p, dir_u1, dir_p


def _taustar_root(sorted_ref, gT64, lo0, hi0, up_mask, z_q, sigma, mu,
                  u1_min, u1_max, iters, pos_inf, neg_inf):
    """One endpoint of the tau* inversion, for the direction in ``up_mask``.

    Passing ``is_up`` reproduces the published ``tau*_p<q>`` column exactly --
    this is the untouched pre-SE code path. Passing ``~is_up`` returns the
    OPPOSITE endpoint of the same interval (SE spec 3.3), because every
    direction-dependent expression below is a ``where`` on this one mask.

    Do NOT symmetrize ``to_neg``/``to_pos``. The strict ``>`` in one branch and
    ``>=`` in the other are correct on two of the four equalities and wrong on
    the other two (tau* spec 3.5); expressing them as ``where``s on the mask is
    what carries that asymmetry across the flip for free.
    """
    level = torch.where(up_mask, mu + 0.5 + z_q * sigma, mu - 0.5 - z_q * sigma)
    lo = lo0.clone()
    hi = hi0.clone()
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        u1_mid = _u1_against(sorted_ref, gT64 * torch.exp2(-mid).unsqueeze(1))
        sig = torch.where(up_mask, u1_mid >= level, u1_mid <= level)
        # up: significant for SMALL delta, tau* = sup -> raise lo when sig.
        # down: significant for LARGE delta, tau* = inf -> lower hi when sig.
        move_lo = torch.where(up_mask, sig, ~sig)
        lo = torch.where(move_lo, mid, lo)
        hi = torch.where(move_lo, hi, mid)
    ts = 0.5 * (lo + hi)
    # Unreachable levels, resolved arithmetically and OVERWRITTEN here (the
    # loop is dense over genes, so they were probed with a meaningless bracket
    # first -- a mask would not save a kernel launch).
    to_neg = torch.where(up_mask, level > u1_max, level >= u1_max)
    to_pos = torch.where(up_mask, level <= u1_min, level < u1_min)
    ts = torch.where(to_neg, neg_inf, ts)
    ts = torch.where(to_pos, pos_inf, ts)
    return ts


def mwu_one_group_taustar(
    sorted_ref: torch.Tensor,    # (n_genes, n_ref) float32, sorted along dim=1
    ref_tie_term: torch.Tensor,  # (n_genes,) float64 -- the UNSHIFTED term
    group_T: torch.Tensor,       # (n_genes, m) float32
    n_ref: int,
    *,
    taustar_levels,              # tuple[float, ...] one-sided p_dir, ascending
    taustar_iters: int,
    taustar_se: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """MWU plus tau*: the signed log2 shift at which each gene crosses a
    one-sided ``p_dir`` level.

    Because the shift scales the TARGET (spec 3.2a of the lfc design), with
    m0 = #{T_i == 0} and n0 = #{R_j == 0},

        U1(delta) = n0*(m - m0) + 0.5*m0*n0                       <- constant
                  + #{ (i,j) : T_i>0, R_j>0, log2(T_i/R_j) >  delta }
                  + 0.5 * #{ (i,j) : T_i>0, R_j>0, log2(T_i/R_j) == delta }

    i.e. the upper-tail count of the POSITIVE-POSITIVE pairwise log-ratio matrix
    on a constant zero baseline, so it is MONOTONE in delta and tau* is an ORDER
    STATISTIC of that matrix. The baseline is not cosmetic: log2(0) - log2(0) is
    undefined, and each zero class is delta-invariant because 0 * s == 0. No
    pairwise matrix is ever formed -- the counting oracle is one ``_bounds`` pair
    against the resident ``sorted_ref``.

    The tie term is constant in delta off the coincidence set (spec 3.4), so
    ``sigma`` is hoisted, the crossing level is a CONSTANT, and each bisection
    probe costs 2 searchsorted, not 4. What is returned inverts ``p~`` -- the
    coincidence-free p-function of spec 3.2, whose tie term IS ``tie_generic`` --
    NOT ``_p_one_sided`` measured at the returned delta, which differs on a
    coincidence plateau.

    Returns ``(u1, p, taustar)``. ``u1``/``p`` are bit-identical to
    ``mwu_one_group``. ``taustar`` is
    ``(len(taustar_levels) + 3*taustar_se, n_genes)`` float64, with rows in
    ``taustar_column_names(levels, se)`` order: requested level rows followed
    by lo, hi, and SE when enabled. Level rows are signed: positive for an up
    gene, negative for a down one. ``+/-inf`` marks an unbounded
    confidence-bound endpoint and is a RESULT, not an error code (spec 3.5);
    NaN is reserved for m == 0, n_ref == 0, and a gene that is zero on both
    sides.
    """
    n_genes = group_T.shape[0]
    m = group_T.shape[1]
    n_levels = len(taustar_levels)
    n_rows = n_levels + (TAUSTAR_SE_COLUMNS if taustar_se else 0)
    device = group_T.device

    if m == 0 or n_ref == 0:
        u1 = torch.zeros(n_genes, dtype=torch.float64, device=device)
        p = torch.full((n_genes,), float("nan"),
                       dtype=torch.float64, device=device)
        taustar = torch.full((n_rows, n_genes), float("nan"),
                             dtype=torch.float64, device=device)
        return u1, p, taustar

    # --- base two-sided test: byte-identical to mwu_one_group ---------------
    u1 = _u1_against(sorted_ref, group_T)
    group_sorted, gc, run_start = _sorted_and_selfties(group_T)
    tie_term = ref_tie_term + _cross_tie_delta(sorted_ref, group_sorted, gc,
                                               run_start)
    p = _p_two_sided(u1, tie_term, m, n_ref)

    # --- hoisted once for the whole level set (spec 3.4) --------------------
    n0, m0 = _zero_counts(sorted_ref, group_sorted)
    tie_generic = ref_tie_term + _cross_tie_generic(group_sorted, gc,
                                                    run_start, n0)
    sigma = torch.sqrt(_s_sq_of(tie_generic, m, n_ref))
    u1_min, u1_max = _u1_reach_limits(m, m0, n_ref, n0)

    mu = float(m) * float(n_ref) / 2.0
    # Ueffect == 0 exactly leaves the sign undefined; resolve to UP (spec 3.5).
    is_up = u1 >= mu
    # Zero on BOTH sides: every pair is a 0-vs-0 tie at every shift, so the gene
    # carries no information. This is the ONLY constant-U1 case that is NaN.
    dead = (m0 == float(m)) & (n0 == float(n_ref))

    # --- bracket: every finite breakpoint is log2(T_i/R_j), both positive ---
    # NEVER find the smallest positive reference value by masking. The obvious
    # `torch.where(sorted_ref > 0, ...).min(dim=1)` materialises a full
    # (n_genes, n_ref) copy of the reference and holds it live through the whole
    # level loop -- the exact cost `_bounds` exists to avoid (0.05 ms -> 65.26 ms
    # on a 160 MB pool, LFC spec 3.2b), and neither chunk-sizer budgets for it.
    # Both arrays are already sorted ascending, so the first positive value is
    # one searchsorted away and one gather retrieves it: O(1) per gene, no copy.
    zeros_f32 = torch.zeros((n_genes, 1), dtype=sorted_ref.dtype, device=device)
    r_first_pos = torch.searchsorted(sorted_ref, zeros_f32, right=True)
    t_first_pos = torch.searchsorted(group_sorted, zeros_f32, right=True)
    # A gene with no positive value on a side indexes one past the end; clamp so
    # the gather is in-bounds. Such a gene has constant U1 and its result is
    # OVERWRITTEN by the spec 3.5 assignment below -- it is still probed, since
    # the loop is dense over the gene axis.
    r_min_pos = sorted_ref.gather(
        1, r_first_pos.clamp_max(n_ref - 1)).squeeze(1).to(torch.float64)
    t_min_pos = group_sorted.gather(
        1, t_first_pos.clamp_max(m - 1)).squeeze(1).to(torch.float64)
    t_max = group_sorted[:, -1].to(torch.float64)
    r_max = sorted_ref[:, -1].to(torch.float64)
    # The fallbacks below only keep the arithmetic well defined for those genes.
    # Clamp BOTH sides of both ratios. A gene with no positive value on a side
    # gathers a 0 above, and an unclamped 0 denominator divides by zero while an
    # unclamped 0 numerator gives log2(0) = -inf. Clamping keeps the arithmetic
    # in-range for the dead genes whose results get overwritten anyway.
    tiny = torch.finfo(torch.float64).tiny
    lo0 = torch.log2(t_min_pos.clamp_min(tiny) / r_max.clamp_min(tiny))
    hi0 = torch.log2(t_max.clamp_min(tiny) / r_min_pos.clamp_min(tiny))
    # KEEP nan_to_num even with the clamps. It is not redundant: `tiny` is
    # 2.2e-308, so a ratio like t_min_pos / tiny still OVERFLOWS to +inf for any
    # ordinary t_min_pos, and log2(inf) = inf. The clamps remove the 0/0 and
    # x/0 cases; nan_to_num remains the backstop for overflow.
    lo0 = torch.nan_to_num(lo0, nan=0.0, posinf=0.0, neginf=0.0)
    hi0 = torch.nan_to_num(hi0, nan=0.0, posinf=0.0, neginf=0.0)
    lo0, hi0 = torch.minimum(lo0, hi0), torch.maximum(lo0, hi0)

    gT64 = group_T.to(torch.float64)
    taustar = torch.empty((n_rows, n_genes), dtype=torch.float64,
                          device=device)
    nan = torch.full((n_genes,), float("nan"), dtype=torch.float64,
                     device=device)
    pos_inf = torch.full((n_genes,), float("inf"), dtype=torch.float64,
                         device=device)
    neg_inf = -pos_inf

    for k, q in enumerate(taustar_levels):
        ts = _taustar_root(sorted_ref, gT64, lo0, hi0, is_up, _z_from_level(q),
                           sigma, mu, u1_min, u1_max, taustar_iters,
                           pos_inf, neg_inf)
        taustar[k] = torch.where(dead, nan, ts)

    if taustar_se:
        # Two one-sided inversions at the SAME internal level: the is_up call
        # is exactly tau*_p0.025, the ~is_up call is the opposite endpoint
        # (SE spec 3.3). No new hoisting -- sigma, the tie term, the bracket
        # and gT64 are all reused.
        z_se = _z_from_level(TAUSTAR_SE_LEVEL)
        root_a = _taustar_root(sorted_ref, gT64, lo0, hi0, is_up, z_se, sigma,
                               mu, u1_min, u1_max, taustar_iters,
                               pos_inf, neg_inf)
        root_b = _taustar_root(sorted_ref, gT64, lo0, hi0, ~is_up, z_se, sigma,
                               mu, u1_min, u1_max, taustar_iters,
                               pos_inf, neg_inf)
        lo_e = torch.where(is_up, root_a, root_b)
        hi_e = torch.where(is_up, root_b, root_a)
        # NO constant-U1 override. When U1 is constant at c the raw inversion
        # returns (+inf, +inf) for c >= L_up, (-inf, +inf) for
        # L_dn < c < L_up, and (-inf, -inf) for c <= L_dn -- never an inverted
        # pair, so there is nothing to fix. Which band a gene lands in is set
        # by c versus the levels, NOT by whether the target or the reference is
        # the all-zero side; forcing (-inf, +inf) across the board would
        # flatten the two informative bands into the uninformative one.
        # SE spec 3.4b; measured, not reasoned.
        #
        # Gate the SE on finiteness rather than subtracting and patching:
        # lo == hi == +inf is reachable on a NON-degenerate gene
        # (level_up <= u1_min), and the mirror lo == hi == -inf when
        # level_dn >= u1_max. The raw difference is inf - inf = NaN in both;
        # +inf is the truth (unbounded uncertainty) where NaN would read as
        # "undefined". NOT clamped at zero: spec 3.4 proves the two searches
        # cannot invert, so a negative SE must surface as a bug, not a 0.0.
        both_finite = torch.isfinite(lo_e) & torch.isfinite(hi_e)
        se = torch.where(both_finite, (hi_e - lo_e) / (2.0 * z_se), pos_inf)
        taustar[n_levels] = torch.where(dead, nan, lo_e)
        taustar[n_levels + 1] = torch.where(dead, nan, hi_e)
        taustar[n_levels + 2] = torch.where(dead, nan, se)
    return u1, p, taustar


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
