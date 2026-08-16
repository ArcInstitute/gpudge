"""Public bring-your-own cell source: the ``CellGroup`` contract and the
adapter turning it into the internal ``target_source`` 4-tuple.

``de(cell_source=...)`` lets a caller feed gpudge cells it reads itself -- the
same reference-pool core ``de(archive=...)`` and ``de(adata=, reference=)``
run, differing only in where the target groups come from. Keeping the reads on
the CALLER's side is the point: a consumer already streaming an archive for its
own aggregation would otherwise pay a second, unfusable pass over the payload.

The public contract deliberately differs from the internal 4-tuple in two
places, both of which were silent-divergence modes:

* it carries the target's **label**, not a positional index into ``targets``
  (a wrong index attributes results to the wrong perturbation, with no error);
* it has **no row_sums field at all** -- gpudge always computes library sizes
  with its own ``csr_row_sums``. A caller-computed one that disagrees (scipy's
  ``X.sum(axis=1)`` reduces in ``X.data``'s dtype) silently shifts every CPM
  scale, and gpudge cannot check the values without doing the work.
"""
from __future__ import annotations

import warnings
from typing import Any, NamedTuple

import numpy as np
from scipy.sparse import issparse

from ._csr_dense import csr_row_sums, ensure_csr
from ._filter import x_has_noncount_signal


class CellGroup(NamedTuple):
    """One target group's cells, yielded by a ``de(cell_source=...)`` source.

    Parameters
    ----------
    label : str
        The target's label. Must appear in ``de(targets=...)``; gpudge maps it
        to that list's order, so the source may yield groups in any order.
    X : scipy.sparse matrix | numpy.ndarray
        This group's cells x genes. Non-CSR sparse input is coerced to CSR with
        a ``UserWarning``, as everywhere else in gpudge. The gene axis must
        match ``de(var_names=...)``. A DENSE ``X`` must be C-contiguous AND
        aligned when ``rows`` re-orders or subsets it and library sizes are
        being computed: gpudge sums those over the row slice, and numpy reduces
        a Fortran-ordered, strided or unaligned array in a different order,
        which would not be byte-identical to the other ``de()`` paths. Pass
        ``np.require(X, requirements=['C', 'A'])``.

        Byte-identity with ``de(adata=, reference=)`` is guaranteed for a
        target matrix that is CSR, or C-contiguous and aligned dense, with
        standard NumPy/SciPy semantics (an ndarray subclass that redefines
        ``sum`` or ``__getitem__``, or an object dtype, is out of contract).
        gpudge sums the matrix it is handed, so if you gather a group out of a
        Fortran-ordered parent and yield the result, its library sizes will
        differ in the last bits from what summing that parent and indexing
        would have given -- gpudge never sees the parent and cannot detect it.
    rows : numpy.ndarray | None, default None
        Row indices into ``X``. ``None`` means all rows. Lets a caller yield one
        shared matrix plus per-group indices instead of materializing a copy.
        Must be 1-D, of integer dtype (a bool mask or float array is rejected,
        not silently cast), in range, and free of duplicates. An empty selector
        is legal and yields a zero-cell group.

    Library sizes are always computed by gpudge, from ``X`` restricted to
    ``rows``. There is deliberately no way to supply them: that is what keeps
    CPM scaling byte-identical to every other gpudge path.
    """

    label: str
    X: Any
    rows: np.ndarray | None = None


def _as_cell_group(obj) -> CellGroup:
    """Coerce a yielded item to ``CellGroup``.

    Accepts a ``CellGroup`` or a plain 2-3 element tuple/list. Deliberately
    does NOT accept an arbitrary iterable: probing with ``iter()``/``tuple()``
    would swallow a ``TypeError`` raised partway through a caller's own
    generator and mis-report it as a shape error (the ``_as_seq`` lesson from
    #108).
    """
    if isinstance(obj, CellGroup):
        return obj
    if not isinstance(obj, (tuple, list)):
        raise TypeError(
            "cell_source must yield gpudge.CellGroup (or a 2-3 element "
            f"tuple of its fields); got {type(obj).__name__}."
        )
    if not 2 <= len(obj) <= 3:
        raise TypeError(
            f"cell_source yielded a {len(obj)}-element tuple; CellGroup takes "
            "2-3 fields (label, X, rows=None)."
        )
    return CellGroup(*obj)


def _check_2d(X, what):
    """``ensure_csr`` passes any non-scipy-sparse object through UNCHANGED
    (``_csr_dense.py:170``), so a 1-D array or a non-matrix would otherwise
    surface much later as an AttributeError or a bad ``X[rows]``.

    The TYPE is checked as well as the shape. A 2-D duck array (a dask array,
    a torch tensor, an h5py dataset) has a plausible ``shape`` and would get
    all the way to ``csr_rows_col_range_to_dense``'s scipy fallback before
    failing on something unrelated -- and gpudge cannot reason about its
    reduction order, which is load-bearing here (see
    ``_check_sliceable_layout``). Documented as sparse-or-ndarray; enforced."""
    shape = getattr(X, "shape", None)
    try:
        ok = shape is not None and len(shape) == 2
    except TypeError:
        # A duck-typed object whose `shape` is a scalar: len() raises, and the
        # bare TypeError would not say which parameter was wrong.
        ok = False
    if not (ok and (issparse(X) or isinstance(X, np.ndarray))):
        raise TypeError(
            f"{what} must be a 2-D cells x genes matrix (scipy sparse or a "
            f"numpy ndarray); got {type(X).__name__} with shape {shape!r}."
        )


def _validate_rows(rows, X, label):
    """Validated int64 row indices into ``X``, plus whether they cover every
    row IN ORDER. ``rows=None`` means all rows."""
    n_rows_X = int(X.shape[0])
    if rows is None:
        return np.arange(n_rows_X, dtype=np.int64), True
    # An UNTYPED empty sequence only. np.asarray([]) is float64, so without
    # this `rows=[]` -- a legal zero-cell group, which the core handles
    # (mwu_one_group returns zeros + NaN at m == 0) -- would trip the dtype
    # guard. A TYPED empty array still goes through the guards below: an empty
    # bool mask against a non-empty matrix is a caller error, not a zero-cell
    # group, and must not be silently accepted.
    if isinstance(rows, (list, tuple)) and len(rows) == 0:
        return np.empty(0, dtype=np.int64), n_rows_X == 0
    arr = np.asarray(rows)
    if arr.ndim != 1:
        raise ValueError(
            f"cell_source group {label!r} rows must be 1-D; got {arr.ndim} "
            "dimensions."
        )
    # Reject bool BEFORE the integer-kind check: np.bool_ is not an integer
    # kind, but astype(int64) turns a mask into indices 0/1 -- a silently
    # different cell selection, which is exactly what this guards.
    if arr.dtype.kind == "b":
        raise TypeError(
            f"cell_source group {label!r} rows is a boolean mask; rows must "
            "be integer indices. Use np.flatnonzero(mask)."
        )
    if arr.dtype.kind not in ("i", "u"):
        raise TypeError(
            f"cell_source group {label!r} rows has dtype {arr.dtype}; rows "
            "must be an integer array (a float array would be truncated to "
            "different cells)."
        )
    arr = arr.astype(np.int64, copy=False)
    if arr.size == 0:                    # a TYPED empty integer selector
        return arr, n_rows_X == 0
    if arr.min() < 0 or arr.max() >= n_rows_X:
        raise ValueError(
            f"cell_source group {label!r} rows are out of bounds for an X "
            f"with {n_rows_X} rows."
        )
    uniq = np.unique(arr)                    # sorted ascending
    if uniq.size != arr.size:
        raise ValueError(
            f"cell_source group {label!r} rows contains duplicate indices; "
            "each cell must appear at most once."
        )
    # arr covers every row in order iff it has n_rows_X distinct in-range
    # values (so uniq == arange(n_rows_X)) AND equals them in that order.
    # Derived from `uniq` -- no second full-size arange allocation. A SET check
    # is not enough: rows=[2,1,0] covers every row but would misalign the sums.
    is_all = bool(arr.size == n_rows_X and np.array_equal(arr, uniq))
    return arr, is_all


def _check_sliceable_layout(X, rows, rows_is_all, label):
    """Reject a DENSE ``X`` that is not C-contiguous AND aligned, under a
    ``rows`` that re-orders or subsets it.

    Only called when library sizes are actually being computed -- without them
    nothing reduces, so nothing can diverge.

    This mode sums library sizes over the SLICE ``X[rows]`` (slicing first is
    what keeps a shared-matrix source off an ``O(n_groups * nnz)`` rescan),
    where ``de(adata=, reference=)`` sums the whole matrix once and indexes the
    sums afterwards (``_refpool._inmem_target_source``). Those two routes agree
    bit-for-bit for CSR and for C-contiguous, ALIGNED dense input -- verified
    across dtypes, byte orders, shapes and view/ownership status -- because both
    reduce the same row's values in the same order.

    They do NOT agree for a dense ``X`` that is Fortran-ordered, strided, or
    C-contiguous but UNALIGNED (a valid ndarray over an offset byte buffer):
    numpy's
    ``sum(axis=1)`` walks a C array row-major with pairwise summation and a
    non-C one differently, and ``X[rows]`` is always C-contiguous. Measured on
    numpy 2.4.6 at 32/32 rows for a 64 x 5000 float64 Fortran array. One ULP in
    a float64 library size becomes one ULP in the float32 CPM scale, which moves
    float32 ties and therefore ranks and p-values.

    gpudge REFUSES rather than papering over it, for the same reason a bool mask
    or a float ``rows`` is refused: the cheap accommodation (copy it, or just
    sum anyway) silently changes numbers the caller cannot see changing.
    Copying to C order is not a fix either -- it would newly break the
    ``rows``-covers-everything route, which sums ``X`` itself and is exact today.

    Two selections are exempt because both routes provably agree on them:
    ``rows`` covering every row IN ORDER (``X`` itself is summed, whatever its
    layout), and an EMPTY ``rows`` (both routes give an empty array). Everything
    else re-orders or drops rows, including a full permutation like ``[2,1,0]``
    -- which is why this is not a "strict subset" rule.

    NOTE the limit of what this can enforce: gpudge sums the matrix it is
    HANDED. A caller who gathers ``parent[rows]`` out of a Fortran-ordered
    parent and yields the (now C-contiguous) result with ``rows=None`` is
    accepted, and its sums will not match ``csr_row_sums(parent)[rows]`` --
    gpudge never sees the parent and cannot know. That is a documented limit of
    the byte-identity claim, not something this guard can close: parity is
    guaranteed against a target matrix that is CSR, or C-contiguous and aligned
    dense.

    The claim is also scoped to ORDINARY numeric ndarrays. An ndarray subclass
    that overrides ``sum`` or ``__getitem__``, or an object-dtype array whose
    elements add statefully, can diverge with every flag looking right --
    ``np.require`` preserves the subclass, so no flags-only predicate can catch
    it. gpudge does not defend against redefined array semantics here or
    anywhere else; that is out of contract rather than unguarded.
    """
    if rows_is_all or issparse(X) or rows.size == 0:
        return
    # ALIGNED as well as C_CONTIGUOUS. An ndarray over an offset byte buffer is
    # C-contiguous but unaligned, gathering rows yields an ALIGNED copy, and
    # numpy reduces the two differently -- so C-contiguity alone would let a
    # divergent input straight through. Measured on numpy 2.4.6 with a
    # 6 x 8193 float64 offset-buffer array.
    if X.flags["C_CONTIGUOUS"] and X.flags["ALIGNED"]:
        return
    raise ValueError(
        f"{label} is a dense array that is not C-contiguous and aligned, and "
        "rows= re-orders or subsets it. gpudge sums library sizes over the row "
        "slice, and numpy reduces a Fortran-ordered, strided or unaligned "
        "array in a different order than it reduces the C-contiguous, aligned "
        "slice -- so the result would not be byte-identical to "
        "de(adata=, reference=). Pass np.require(X, requirements=['C', 'A']), "
        "or a CSR matrix. (np.ascontiguousarray alone is not enough: it "
        "returns an already-C-contiguous array unchanged, alignment included.)"
    )


def make_target_source(cell_source, *, targets, n_genes,
                       warn_noncount_targets=False, warn_stacklevel=2):
    """Adapt a public ``cell_source`` into the internal ``target_source``.

    Returns ``target_source(need_row_sums)``, a generator function yielding the
    ``(g, X, rows, Ls_for_rows)`` 4-tuple ``_refpool.refpool_de_core`` consumes.
    May be called more than once; each call re-drives ``cell_source()``.

    ``warn_noncount_targets`` enables the target-side raw-counts check the
    in-memory external-ref path performs (``_refpool.py:566-582``). The warned
    flag lives HERE, not inside ``target_source``, so "at most one warning"
    holds across every drive of this source -- not once per drive.

    ``warn_stacklevel`` is how far up the warning should point. The generator
    is resumed from ``refpool_de_core``'s loop, not from its creator, so under
    the public entry the chain at warn time is target_source(1) ->
    refpool_de_core(2) -> cell_source_de(3) -> de(4) -> user(5); the default 2
    is for direct use of this helper. Because the depth depends on who drives
    the generator, this is a best-effort attribution, not a guarantee.
    """
    label_to_idx = {str(label): i for i, label in enumerate(targets)}
    n_genes = int(n_genes)
    warned = [False]                 # list, not a bool: rebound from the inner
                                     # generator without a nonlocal declaration

    def target_source(need_row_sums):
        seen = set()
        for item in cell_source():
            grp = _as_cell_group(item)
            label = str(grp.label)
            g = label_to_idx.get(label)
            if g is None:
                raise ValueError(
                    f"cell_source yielded label {label!r}, which is not in "
                    f"targets ({len(label_to_idx)} labels)."
                )
            if label in seen:
                raise ValueError(
                    f"cell_source yielded label {label!r} more than once; "
                    "each target must be yielded exactly once."
                )
            seen.add(label)

            _check_2d(grp.X, f"cell_source group {label!r} X")
            # stacklevel matches the raw-counts warning below: same chain.
            X = ensure_csr(grp.X, name=f"cell_source group {label!r} X",
                           stacklevel=warn_stacklevel + 1)
            if int(X.shape[1]) != n_genes:
                raise ValueError(
                    f"cell_source group {label!r} has {int(X.shape[1])} genes "
                    f"but var_names has {n_genes}; every group must share the "
                    "gene axis."
                )

            rows, rows_is_all = _validate_rows(grp.rows, X, label)
            # AFTER _validate_rows (it needs rows/rows_is_all) and gated on
            # need_row_sums: with no sums nothing reduces, so no layout can
            # diverge and rejecting would be gratuitous. Cheap: a flags read,
            # no copy. See _check_sliceable_layout.
            if need_row_sums:
                _check_sliceable_layout(X, rows, rows_is_all,
                                        f"cell_source group {label!r} X")
            need_sel = need_row_sums or (warn_noncount_targets
                                         and not warned[0])
            # Built at most ONCE per group and shared by the row sums and the
            # warning. Slicing here rather than gathering afterwards is what
            # keeps a shared-matrix source off an O(n_groups * nnz) rescan.
            X_sel = (X if rows_is_all else X[rows]) if need_sel else None
            Ls = csr_row_sums(X_sel) if need_row_sums else None

            if warn_noncount_targets and not warned[0]:
                # X_sel, NOT X: a shared matrix may hold fractional values in
                # rows no group selects, and warning about cells the run never
                # touches is a false positive.
                bad = x_has_noncount_signal(X_sel)
                if not bad and Ls is not None:
                    bad = bool((Ls < 0).any())
                if bad:
                    warned[0] = True
                    warnings.warn(
                        "cell_source target cells do not look like raw counts "
                        "(non-integer or negative values); the "
                        "filter_gene_min_cpm_* filters assume raw counts. If "
                        "X is not counts, pass a precomputed keep_genes mask "
                        "instead.",
                        UserWarning, stacklevel=warn_stacklevel)

            # Drop the slice BEFORE suspending: a generator's locals stay
            # referenced while it is parked at the yield, so for a subset
            # source X_sel would otherwise pin a full copy of the group for
            # the whole of _accumulate_target_group -- and overlap the next
            # group's slice. Matches the core's own `del` at _refpool.py:455.
            del X_sel

            yield g, X, rows, Ls
            # `item` too, not just `grp`: it is the original CellGroup/tuple and
            # holds its own reference to X, so leaving it bound keeps the
            # previous group alive while FOR_ITER asks the source for the next
            # one -- two large matrices resident at once. Safe on the final
            # iteration (the consumer's references are independent) and skipped
            # entirely on GeneratorExit, where frame teardown frees the locals.
            del item, grp, X, rows, Ls

        if len(seen) != len(label_to_idx):
            missing = [lbl for lbl in label_to_idx if lbl not in seen]
            shown = ", ".join(repr(m) for m in missing[:10])
            more = "" if len(missing) <= 10 else f" (+{len(missing) - 10} more)"
            raise ValueError(
                f"cell_source did not yield {len(missing)} of "
                f"{len(label_to_idx)} targets: {shown}{more}. Every target "
                "must be yielded exactly once."
            )

    return target_source
