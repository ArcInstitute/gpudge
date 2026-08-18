"""Single-pass CSR (rows, col-range) → dense float32 extractor.

Numba-jitted when ``numba`` is installed; falls back to scipy's two-step
``X[rows, col_start:col_stop].toarray()`` otherwise. cProfile on CCL_2
identified scipy's ``csr_row_index`` + ``get_csr_submatrix`` + their
intermediate ``numpy.array`` shells as ~280 s of de() wall (90% of the
remaining post-FDR-fix bottleneck). One pass + parallel-rows kernel
removes most of that overhead.

Install with ``pip install gpudge[fast]`` (or ``uv pip install -e
".[fast]"``) to enable; the extra is just ``numba``.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.sparse import issparse

try:
    import numba as nb
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


if HAS_NUMBA:

    @nb.njit(parallel=True, boundscheck=False, cache=True)
    def _csr_row_sums_numba(
        data,                       # (nnz,) any numeric
        indptr,                     # (n_rows+1,) integer
        out,                        # (n_rows,) float64, pre-allocated
    ):
        """Per-row sum of CSR ``data`` written into pre-allocated ``out``.

        scipy's ``X.sum(axis=1)`` falls onto ``np.ufunc.reduceat`` which is
        dramatically slower on narrow integer dtypes (no SIMD path:
        observed ~50× slowdown for uint16 vs float32 on CCL_2 → 292 s of a
        392 s de() call). The numba loop accumulates in float64 in
        parallel across rows.
        """
        n = indptr.shape[0] - 1
        for r in nb.prange(n):  # ty: ignore[not-iterable]
            s = 0.0
            for j in range(indptr[r], indptr[r + 1]):
                s += data[j]
            out[r] = s


    @nb.njit(parallel=True, boundscheck=False, cache=True)
    def _csr_rows_to_dense_numba(
        data,                       # (nnz,) any numeric (uint16, int16, float32, …)
        indices,                    # (nnz,) any integer (uint16, int32, int64)
        indptr,                     # (n_cells+1,) int32 or int64
        row_indices,                # (m,) int64
        col_start: int,
        col_stop: int,
    ):
        """Gather ``row_indices`` rows × ``[col_start:col_stop]`` cols → dense float32.

        One pass over the selected rows in parallel; writes go straight into
        a float32 output buffer. ``data`` is implicitly cast to float32 on
        store, so uint16-on-disk h5ads (h5ad_compression session) work
        without a separate cast pass. ``indices`` may be uint16 too — they're
        compared against ``col_start``/``col_stop`` (Python ints) and
        contribute to an int offset into ``out``.
        """
        m = row_indices.shape[0]
        n_cols = col_stop - col_start
        out = np.zeros((m, n_cols), dtype=np.float32)
        for i in nb.prange(m):  # ty: ignore[not-iterable]
            r = row_indices[i]
            start = indptr[r]
            end = indptr[r + 1]
            for j in range(start, end):
                c = indices[j]
                if col_start <= c < col_stop:
                    # ``+=`` (not ``=``) so duplicate column entries within a
                    # row accumulate, matching scipy's ``.toarray()``. Canonical
                    # CSR has at most one entry per (row, col) so this is
                    # identical to a plain store there; only non-canonical input
                    # differs (and the scipy fallback already sums). prange is
                    # over rows, so no two iterations touch the same out[i, :].
                    out[i, c - col_start] += data[j]
        return out


    @nb.njit(parallel=True, boundscheck=False, cache=True)
    def _csr_rows_to_dense_numba_into(
        data,
        indices,
        indptr,
        row_indices,                # (m,) int64
        col_start: int,
        col_stop: int,
        out,                        # (>=m, col_stop-col_start) float32, pre-allocated
    ):
        """Variant of ``_csr_rows_to_dense_numba`` that writes into ``out``.

        Same algorithm; lets the caller hand in a pre-allocated (and
        possibly pinned) buffer. ``out`` must be at least
        ``(len(row_indices), col_stop - col_start)``; only the
        ``(:m, :n_cols)`` view is written. Cells in ``out`` outside that
        view are left untouched, and cells inside it are zeroed first
        (CSR stores only non-zeros so a previously-used pinned buffer
        could carry stale values into the dense view).
        """
        m = row_indices.shape[0]
        n_cols = col_stop - col_start
        # Single parallel region per row: zero the active sub-view, then
        # immediately populate non-zero columns from the CSR data. Fusing
        # the two prange loops (vs separate zero / populate passes) avoids
        # a second parallel-region launch + sync, and keeps each row in
        # CPU cache between the two writes.
        for i in nb.prange(m):  # ty: ignore[not-iterable]
            for j in range(n_cols):
                out[i, j] = 0.0
            r = row_indices[i]
            start = indptr[r]
            end = indptr[r + 1]
            for j in range(start, end):
                c = indices[j]
                if col_start <= c < col_stop:
                    # ``+=`` to sum duplicate column entries (see the alloc
                    # variant above); the sub-view was zeroed just above.
                    out[i, c - col_start] += data[j]


def is_cupy_csr(x) -> bool:
    """True iff ``x`` is a cupy device sparse CSR matrix, WITHOUT importing cupy.

    Decided by module/class name so CPU-only / no-cupy environments never import
    cupy just to answer the predicate. The streaming device path lazily imports
    the real cupy helpers (``_csr_dense_gpu``) only after this returns True.
    """
    cls = type(x)
    return (cls.__module__.startswith("cupyx.scipy.sparse")
            and cls.__name__ == "csr_matrix")


def ensure_csr(X, *, name: str = "adata.X", stacklevel: int = 2):
    """Return ``X`` as canonical (sorted) CSR when it is a non-CSR *sparse*
    matrix, emitting a one-time ``UserWarning``; dense or already-CSR ``X``
    passes through unchanged (no warning).

    gpudge's numba fast paths (``csr_row_sums``, ``csr_rows_col_range_to_dense``)
    are gated on ``X.format == "csr"``; a non-CSR sparse ``X`` (e.g. CSC) would
    otherwise silently fall back to single-threaded scipy slicing on every
    ``(group x gene-chunk)`` — minutes vs hours at multi-million-cell scale.

    ``stacklevel`` is forwarded to ``warnings.warn`` so the warning lands on the
    user's ``de(...)`` call rather than an internal gpudge line. Count the frames
    from this function to ``de()``'s caller: a direct ``de() -> ensure_csr``
    caller passes ``3``; ``de() -> inmem_external_ref_de -> ensure_csr`` passes
    ``4``. The default ``2`` points at whoever calls ``ensure_csr`` directly.
    """
    if issparse(X) and X.format != "csr":
        warnings.warn(
            f"de(): {name} is a {X.format!r} sparse matrix; converting to CSR "
            "once for the numba fast path. Pass a csr_matrix to avoid this "
            "(a non-CSR sparse X otherwise falls back to single-threaded scipy "
            "slicing per gene-chunk — minutes vs hours at multi-million-cell "
            "scale).",
            UserWarning,
            stacklevel=stacklevel,
        )
        X = X.tocsr()
        X.sort_indices()   # keep the coerced matrix canonical for the kernels
    return X


def csr_row_sums(X) -> np.ndarray:
    """Per-row sum of a CSR sparse OR dense matrix, float64.

    (de() also calls this with a dense ``adata.X``, natively or after
    ``densify_input=True``; the fallback branch handles that.)
    Uses the numba kernel when ``numba`` is installed, otherwise falls back
    to scipy's ``X.sum(axis=1)``. Equivalent results; the numba path is
    massively faster on narrow integer dtypes (uint16/int16/uint8) where
    scipy's ``np.ufunc.reduceat`` has no SIMD support.
    """
    if HAS_NUMBA and issparse(X) and X.format == "csr":
        n_rows = X.shape[0]
        out = np.empty(n_rows, dtype=np.float64)
        _csr_row_sums_numba(X.data, X.indptr, out)
        return out
    # Fallback: scipy/numpy. The reduction must accumulate in float64, not in
    # X.data's dtype: a float32 CSR (what shardad's cell gather_rows returns)
    # would otherwise round where the same counts as a narrow integer dtype
    # (what the shard reader can return) stay exact, and the two streaming
    # layouts would disagree on CPM scales, the median target, and every cpm_*
    # filter. (#110)
    #
    # Passing dtype= to scipy's sparse .sum() is NOT enough -- whether it widens
    # the accumulator or only the result varies by scipy version. CI caught this
    # on Python 3.11 (reduced in float32) while 3.12 happened to be fine. So for
    # sparse X, widen the data ourselves and let scipy reduce a float64 matrix;
    # indices/indptr are shared, so the transient cost is nnz*8 bytes and the
    # numba kernel above -- what any large run uses, via the [fast] extra --
    # never reaches here. numpy's dense .sum(dtype=) does honour the accumulator.
    if issparse(X):
        if X.dtype == np.float64:
            X64 = X
        elif X.format in ("csr", "csc"):
            # Share indices/indptr; only `data` is widened.
            X64 = type(X)((X.data.astype(np.float64), X.indices, X.indptr),
                          shape=X.shape, copy=False)
        else:
            X64 = X.astype(np.float64)      # COO/other: no indptr to share
        return np.asarray(X64.sum(axis=1)).ravel()
    return np.asarray(X.sum(axis=1, dtype=np.float64)).ravel()


def csr_rows_col_range_to_dense(
    X,
    rows: np.ndarray,
    col_start: int,
    col_stop: int,
    *,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Dense float32 ``(len(rows), col_stop-col_start)`` array.

    Uses the numba kernel if available and ``X`` is sparse CSR with
    ``ndarray`` row selectors; otherwise falls back to scipy's slicing path.
    The kernel writes float32 regardless of ``X.data`` dtype, so uint16/int16
    h5ads from the h5ad_compression session can be passed directly without an
    upfront cast (saving ~half the host RAM for the sparse matrix).

    ``out``: optional pre-allocated float32 buffer of shape at least
    ``(len(rows), col_stop - col_start)``. When supplied (and we're on the
    numba+CSR fast path), the kernel writes into the ``[:m, :n_cols]`` view
    of ``out`` and the same view is returned — this lets the caller hand in
    a pinned-memory buffer so the subsequent ``.to(device, non_blocking=True)``
    skips PyTorch's implicit pin+copy step. Falls back to allocating
    internally on any non-CSR / non-numba path.
    """
    n_cols = int(col_stop) - int(col_start)
    if (HAS_NUMBA
            and issparse(X)
            and X.format == "csr"
            and isinstance(rows, np.ndarray)):
        rows64 = rows.astype(np.int64, copy=False)
        m = rows64.shape[0]
        # The kernels run under boundscheck=False, so an out-of-range row would
        # read indptr out of bounds (silent corruption / segfault) instead of
        # scipy's IndexError. Validate once here (O(m), cheap vs the gather).
        if m and (rows64.min() < 0 or rows64.max() >= X.shape[0]):
            raise IndexError(
                f"row index out of range: rows span "
                f"[{int(rows64.min())}, {int(rows64.max())}] but X has "
                f"{X.shape[0]} rows."
            )
        if out is not None:
            if out.dtype != np.float32:
                raise ValueError(
                    f"out.dtype={out.dtype}; expected float32."
                )
            if out.shape[0] < m or out.shape[1] < n_cols:
                raise ValueError(
                    f"out shape {out.shape} too small for "
                    f"{(m, n_cols)} (needs >= each dim)."
                )
            view = out[:m, :n_cols]
            _csr_rows_to_dense_numba_into(
                X.data, X.indices, X.indptr, rows64,
                int(col_start), int(col_stop), view,
            )
            return view
        return _csr_rows_to_dense_numba(
            X.data, X.indices, X.indptr, rows64, int(col_start), int(col_stop)
        )

    # Fallback: scipy slice → toarray → float32. scipy handles dtype mixes
    # but may upcast to int64 indices, etc.
    if issparse(X):
        block = X[rows, col_start:col_stop]
        if hasattr(block, "toarray"):
            return block.toarray().astype(np.float32, copy=False)
        return np.ascontiguousarray(block, dtype=np.float32)
    return np.ascontiguousarray(X[rows, col_start:col_stop], dtype=np.float32)
