# src/gpudge/_taustar.py
"""tau_star levels: validation, canonical ordering, column naming.

Pure (stdlib only, no torch) so it can be unit-tested without a GPU and reused
verbatim by the literal-reference, external-reference, and streaming drivers --
and by the test oracles, so implementation and oracle cannot drift. Mirrors
``_lfc.py``.

A "level" is a ONE-SIDED ``p_dir`` threshold in the OPEN interval (0, 1). The
canonical order is ascending, so column layout never depends on input ordering.

Naming (spec 4.2): ``tau*_p<level>``. The ``p`` is load-bearing -- it marks the
number as a p-LEVEL, because the roles invert relative to the ``lfc_threshold``
grammar. In ``tau=+0.25_p`` the number is a tau and the value is a p; in
``tau*_p0.5`` the number is a p and the value is a tau.
"""
from __future__ import annotations

import math
import operator

DEFAULT_TAUSTAR_ITERS = 20

#: One-sided p_dir level the SE interval is built from. 2 * 0.025 = 0.05, so
#: (tau*_lo, tau*_hi) is the conventional 95% two-sided interval and
#: tau*_p0.5 / tau*_se compares against 1.96 the way lfcSE does. Deliberately
#: NOT a parameter (spec 3.2): exposing it invites q = 0.5, where z = 0 makes
#: the SE 0/0 and the endpoint gap is the +/-0.5 continuity correction -- a
#: discretization artifact, not a confidence width.
TAUSTAR_SE_LEVEL = 0.025

#: Rows tau_star_se appends after the level rows: lo, hi, se.
TAUSTAR_SE_COLUMNS = 3


def _as_seq(value):
    """Scalar -> 1-tuple; any non-``str``/``bytes`` ITERABLE -> tuple of items.

    Deliberately not ``isinstance(value, Sequence)``: ``np.ndarray`` is not a
    ``collections.abc.Sequence``, so the Sequence test sends
    ``tau_star=np.array([0.5, 0.05])`` down the scalar branch, where
    ``float()`` on a 2-element array raises a bare TypeError -- whose exact
    wording varies by numpy version -- instead of either working or producing
    this module's ValueError. An array of levels is a natural thing to pass. The
    try/except also picks up generators and sets for free, and still routes
    genuine scalars -- ``float``, ``np.float64``, a 0-d array -- to the
    1-tuple branch, since none of them are iterable.

    The catch wraps ``iter()``, NOT ``tuple()``: only *acquiring* the iterator
    distinguishes a scalar from an iterable. Wrapping the whole materialisation
    would also swallow a TypeError raised from ``__next__`` partway through,
    silently discarding the items already yielded and mistaking a half-consumed
    generator for a scalar. Kept in step with ``_lfc._as_seq``.
    """
    if isinstance(value, (str, bytes)):
        return (value,)
    try:
        items = iter(value)
    except TypeError:
        return (value,)
    return tuple(items)


def _fmt_level(q: float) -> str:
    """Canonical level -> column-name fragment. 0.05 -> '0.05', 0.5000 -> '0.5'."""
    return f"{q:g}"


def normalize_taustar_se(tau_star_se) -> bool:
    """Validate the ``tau_star_se`` flag: a genuine boolean, not a truthy value.

    Accepts the ``bool`` singletons and any scalar whose ``.item()`` is a
    ``bool`` -- ``numpy.bool_``, a 0-d numpy array, a 0-d torch tensor.
    ``numpy.bool_`` is not a ``bool`` subclass, and rejecting it would be
    inconsistent: ``tau_star`` already takes numpy scalar levels and
    ``tau_star_iters`` numpy integers. The ``.item()`` duck-test keeps this
    module stdlib-only (no numpy import) while staying precise --
    ``numpy.int64(1).item()`` is an ``int``, so it still raises.

    Truthy coercion is refused outright: ``tau_star_se="false"`` would mean
    True.
    """
    if tau_star_se is True:
        return True
    if tau_star_se is False:
        return False
    # ndim == 0 is REQUIRED, not decorative: numpy's and torch's .item() both
    # happily unwrap a ONE-element vector, so np.array([True]) and even
    # np.array([[True]]) would otherwise pass as scalars.
    if getattr(tau_star_se, "ndim", None) == 0:
        try:
            value = tau_star_se.item()
        except (TypeError, ValueError, AttributeError):
            value = None
        if value is True:
            return True
        if value is False:
            return False
    raise ValueError(
        f"tau_star_se must be True or False; got {tau_star_se!r}. Truthy "
        f"values are rejected deliberately -- tau_star_se='false' would "
        f"otherwise mean True. A numpy/torch boolean scalar is accepted; a "
        f"numeric one is not.")


def normalize_taustar_spec(tau_star, taustar_se: bool = False):
    """Validate and canonicalise the level set.

    Returns ``None`` when ``tau_star is None``, otherwise a tuple of levels in
    ascending order. When ``taustar_se`` is set, ``0.5`` is added to the set if
    absent, so the point estimate the SE belongs to is always reported with it
    (spec 4.1) -- the emitted level set can therefore be LARGER than the one
    requested.
    """
    if tau_star is None:
        if taustar_se:
            raise ValueError(
                "tau_star_se=True requires tau_star to be set: the SE is the "
                "standard error OF the tau* point estimate, and with "
                "tau_star=None there is no estimate to attach it to.")
        return None
    # `float | Iterable[float]` does NOT include str, but str IS iterable and
    # _as_seq's str branch would wrap "0.5" as a single item that float() then
    # happily parses. Reject it explicitly rather than silently honouring a type
    # the signature disallows. (_lfc.py keeps its str branch because
    # lfc_threshold_alt genuinely takes a string; tau_star has no such
    # parameter, so the branch here is pure loophole.)
    if isinstance(tau_star, (str, bytes)):
        raise ValueError(
            f"tau_star must be a number or a sequence of numbers, not a "
            f"string; got {tau_star!r}.")

    raw = _as_seq(tau_star)
    if len(raw) == 0:
        raise ValueError(
            "tau_star must be a number or a non-empty sequence of numbers "
            "(got an empty sequence). Pass None to disable.")

    levels: list[float] = []
    for q in raw:
        qv = float(q)
        if not math.isfinite(qv) or qv <= 0.0 or qv >= 1.0:
            raise ValueError(
                f"tau_star levels are one-sided p_dir thresholds and must lie "
                f"in the OPEN interval (0, 1); got {q!r}. The endpoints drive "
                f"the normal quantile to infinity, which makes every gene "
                f"unreachable (spec 4.1).")
        levels.append(qv)

    ordered = sorted(set(levels))
    if len(ordered) != len(levels):
        # Exact duplicates are a caller mistake, not a silent dedup.
        raise ValueError(f"tau_star contains duplicate values: {sorted(levels)}.")

    # AFTER the duplicate check, so the 0.5 union cannot swallow a genuine
    # caller duplicate; BEFORE the name-collision check, so the added level is
    # covered by it too.
    if taustar_se and 0.5 not in ordered:
        ordered = sorted(ordered + [0.5])

    formatted = [_fmt_level(q) for q in ordered]
    if len(set(formatted)) != len(formatted):
        raise ValueError(
            f"tau_star values collide after column-name formatting "
            f"({formatted}); they would produce duplicate column names. Use "
            f"values that differ within 6 significant figures.")

    return tuple(ordered)


def normalize_taustar_iters(iters) -> int:
    """Validate the bisection step count. ``None`` -> the default.

    Validated UNCONDITIONALLY by the caller, including when ``tau_star is
    None``: the value is only USED when tau_star is set, but a nonsense value is
    a caller error worth failing on either way (spec 4.1).
    """
    if iters is None:
        return DEFAULT_TAUSTAR_ITERS
    # bool PASSES operator.index() (it IS an int), and True would mean ONE
    # bisection step -- the same materially wrong answer that motivates
    # rejecting 1.9 below. So it needs its own guard, ahead of the coercion.
    if isinstance(iters, bool):
        raise ValueError(
            f"tau_star_iters must be a positive integer, not a bool; got "
            f"{iters!r} (which would mean {int(iters)} bisection step(s)).")
    # operator.index(), NOT int(). int() SILENTLY TRUNCATES: int(1.9) == 1 and
    # int("7") == 7, so a fractional value would quietly collapse the bisection
    # to a single step -- a materially wrong answer, not a rounding. index()
    # accepts only genuine integers (including numpy integer scalars) and
    # raises TypeError otherwise, which is re-raised as the documented
    # ValueError so callers see one error type from this module.
    try:
        value = operator.index(iters)
    except TypeError:
        raise ValueError(
            f"tau_star_iters must be a positive integer; got {iters!r}. "
            f"A float is rejected rather than truncated -- int(1.9) == 1 "
            f"would silently reduce the bisection to one step.") from None
    if value < 1:
        raise ValueError(
            f"tau_star_iters must be a positive integer; got {iters!r}.")
    return value


def taustar_column_names(levels, se: bool = False) -> list[str]:
    """Flat, ordered list of tau* column names.

    THE layout authority: the kernel writes its output rows in exactly this
    order and every driver shapes its accumulator from ``len()`` of this list.
    """
    names = [f"tau*_p{_fmt_level(q)}" for q in levels]
    if se:
        lv = _fmt_level(TAUSTAR_SE_LEVEL)
        names += [f"tau*_lo_p{lv}", f"tau*_hi_p{lv}", "tau*_se"]
    return names
