# src/gpudge/_lfc.py
"""lfc_threshold grid: validation, scale factors, canonical column naming.

Pure (numpy only, no torch) so it can be unit-tested without a GPU and reused
verbatim by the literal-reference, external-reference, and streaming drivers --
and by the test oracles, so implementation and oracle cannot drift.

A "combo" is a ``(tau: float, direction: str)`` pair. The canonical order is tau
ascending, then LFC_DIRECTIONS order -- deterministic regardless of how the
caller ordered the grid, so column layout never depends on input ordering.
"""
from __future__ import annotations

import math

LFC_DIRECTIONS = ("down", "up")

# BIOLOGICAL sanity bound on tau, not a numeric limit. The scaling is done in
# float64 (spec 3.2b), where the only hard wall is 2.0**tau raising
# OverflowError at tau >= 1024 -- everything below that is comfortably inside
# float64's range. 2**30 ~ 1.07e9-fold is already far past anything meaningful
# (useful values are ~0-2), and it keeps the float64 products well inside the
# normal range, which is what spec 3.4a's injectivity argument needs. Spec 3.4d.
MAX_LFC_THRESHOLD = 30.0


def _as_seq(value):
    """Scalar -> 1-tuple; any non-string ITERABLE -> tuple of its items.

    Deliberately not ``isinstance(value, Sequence)``: ``np.ndarray`` is not a
    ``collections.abc.Sequence``, so the Sequence test sent
    ``lfc_threshold=np.array([0.25, 0.5])`` down the scalar branch, where
    ``float()`` on a 2-element array raises a bare TypeError -- whose exact
    wording varies by numpy version -- instead of either working or producing
    this module's ValueError, and sent
    ``lfc_threshold_alt=np.array(["up", "down"])`` to ``d not in
    LFC_DIRECTIONS`` whole, raising numpy's ambiguous-truth ValueError (it takes
    two or more elements to get that one; a one-element array instead compares
    fine and then dies unhashable in ``seen_d``, which is no better).
    An array of taus or directions is a natural thing to pass. The
    try/except also picks up generators and sets for free, and still routes
    genuine scalars -- ``float``, ``np.float64``, a 0-d array -- to the 1-tuple
    branch, since none of them are iterable.

    The catch wraps ``iter()``, NOT ``tuple()``: only *acquiring* the iterator
    distinguishes a scalar from an iterable. Wrapping the whole materialisation
    would also swallow a TypeError raised from ``__next__`` partway through,
    silently discarding the items already yielded and mistaking a half-consumed
    generator for a scalar.

    The ``str`` branch STAYS, unlike the tau-star path, where
    ``normalize_taustar_spec`` rejects ``str``/``bytes`` before ``_as_seq`` ever
    sees them (that helper only wraps them): ``lfc_threshold_alt`` takes a bare
    string, and ``tuple("up")`` would split it into ``('u', 'p')``. ``bytes`` is
    deliberately NOT added to that branch, purely to keep this a bugfix. It is
    iterated byte-by-byte, exactly as before: ``lfc_threshold_alt=b"up"``
    complains about the integer 117, while ``lfc_threshold=bytes([1, 2])``
    quietly yields taus 1.0 and 2.0. No bytes input is *intentionally*
    supported, but guarding it here would not merely improve those outcomes --
    ``float`` parses bytes, so ``lfc_threshold=b"0.5"`` would go from a
    misleading error to silently succeeding. Making bytes fail cleanly needs an
    explicit rejection -- here, or per-parameter if the message is to name which
    parameter was wrong -- i.e. a validation change rather than part of this fix.
    """
    if isinstance(value, str):
        return (value,)
    try:
        items = iter(value)
    except TypeError:
        return (value,)
    return tuple(items)


def normalize_lfc_spec(lfc_threshold, lfc_threshold_alt):
    """Validate and canonicalise the (tau grid, direction set) into combos.

    Returns ``None`` when ``lfc_threshold is None`` (no directional output),
    otherwise a tuple of ``(tau, direction)`` in canonical order.
    """
    if lfc_threshold is None:
        return None

    taus_raw = _as_seq(lfc_threshold)
    if len(taus_raw) == 0:
        raise ValueError(
            "lfc_threshold must be a number or a non-empty sequence of "
            "numbers (got an empty sequence). Pass None to disable.")
    taus: list[float] = []
    for t in taus_raw:
        tv = float(t)
        if not math.isfinite(tv):
            raise ValueError(
                f"lfc_threshold values must be finite; got {t!r}.")
        if tv < 0.0:
            raise ValueError(
                f"lfc_threshold values must be >= 0 (tau is a magnitude in "
                f"log2 fold-change units; direction is selected with "
                f"lfc_threshold_alt); got {t!r}.")
        if tv > MAX_LFC_THRESHOLD:
            raise ValueError(
                f"lfc_threshold values must be <= {MAX_LFC_THRESHOLD:g} "
                f"(2**{MAX_LFC_THRESHOLD:g} is already ~1.1e9-fold; "
                f"biologically meaningful values are ~0-2). It is a sanity "
                f"bound, not a numeric limit -- the scaling is done in float64 "
                f"(2.0**tau only overflows at tau >= 1024); got {t!r}.")
        # Normalise an input -0.0 magnitude so tau=-0/tau=+0 is determined
        # solely by the canonical down/up direction sign.
        if tv == 0.0:
            tv = 0.0
        taus.append(tv)

    dirs = _as_seq(lfc_threshold_alt)
    if len(dirs) == 0:
        raise ValueError(
            f"lfc_threshold_alt must name at least one direction from "
            f"{LFC_DIRECTIONS}; got an empty sequence.")
    seen_d = set()
    for d in dirs:
        if d not in LFC_DIRECTIONS:
            raise ValueError(
                f"lfc_threshold_alt values must be in {LFC_DIRECTIONS}; "
                f"got {d!r}.")
        if d in seen_d:
            raise ValueError(
                f"lfc_threshold_alt contains duplicate direction {d!r}.")
        seen_d.add(d)

    # Deterministic layout: tau ascending, then CANONICAL direction order
    # (LFC_DIRECTIONS), not the caller's order -- so ("down","up") and
    # ("up","down") produce identical column layouts.
    dirs = tuple(d for d in LFC_DIRECTIONS if d in seen_d)
    taus_sorted = sorted(set(taus))
    if len(taus_sorted) != len(taus):
        # Exact duplicates are a caller mistake, not a silent dedup.
        raise ValueError(
            f"lfc_threshold contains duplicate values: {sorted(taus)}.")

    # %g carries 6 significant figures -- distinct taus can format identically.
    formatted = [_fmt_tau(t) for t in taus_sorted]
    if len(set(formatted)) != len(formatted):
        raise ValueError(
            f"lfc_threshold values collide after column-name formatting "
            f"({formatted}); they would produce duplicate column names. Use "
            f"values that differ within 6 significant figures.")

    return tuple((t, d) for t in taus_sorted for d in dirs)


def lfc_scale_factor(tau: float, direction: str) -> float:
    """TARGET scale factor for one combo, in float64.

    The shift is applied to the TARGET, not the reference, and the factor is
    therefore INVERTED (spec 3.2a) -- U(T, R*f) == U(T/f, R) exactly, because
    the MWU depends only on the ordering of the pooled sample:

        up   : test T vs R*2**(+tau)  ==  (T * 2**(-tau)) vs R
        down : test T vs R*2**(-tau)  ==  (T * 2**(+tau)) vs R

    Returned in FLOAT64, not float32 (spec 3.2b): the kernel promotes the target
    to float64 before scaling, because float32 scaling lands on tie boundaries
    often enough to change p-values by orders of magnitude, and because float64
    scaling is injective on float32 inputs (which is what makes gc/run_start
    tau-invariant).

    ULP pinning: ALWAYS ``2.0 ** (-+tau)``, NEVER ``1.0 / (2.0 ** tau)`` -- at
    float64 the two differ for ~27% of taus. The kernel and the scipy test oracle
    both call this, so they multiply by the identical bits.
    """
    if direction == "up":
        return 2.0 ** (-float(tau))
    if direction == "down":
        return 2.0 ** float(tau)
    raise ValueError(
        f"direction must be one of {LFC_DIRECTIONS}; got {direction!r}.")


def _fmt_tau(tau: float) -> str:
    """Canonical tau -> column-name fragment. 0.25 -> '0.25', 1.0 -> '1'."""
    return f"{tau:g}"


def lfc_base_names(combos) -> list[tuple[str, str, str]]:
    """Per-combo ``(p, Ueffect, padj)`` directional column names, in order."""
    out = []
    for tau, direction in combos:
        t = _fmt_tau(tau)
        sign = "+" if direction == "up" else "-"
        out.append((
            f"tau={sign}{t}_p",
            f"tau={sign}{t}_Ueffect",
            f"tau={sign}{t}_padj",
        ))
    return out


def lfc_column_names(combos) -> list[str]:
    """Flat, ordered list of every directional column name."""
    return [name for triple in lfc_base_names(combos) for name in triple]
