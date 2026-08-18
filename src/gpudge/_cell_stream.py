"""``layout='cell'`` backend for de(archive=...)."""
from __future__ import annotations

import logging

import numpy as np

from ._ingest import reject_missing_group_labels
from ._csr_dense import csr_row_sums
from ._stream_backend import validate_archive_reference

logger = logging.getLogger(__name__)

# Target decoded-CSR bytes per gather_rows() call. The planner turns this into a
# row budget using the manifest's mean nnz/row and its decoded bytes/nnz.
# Measured on a production-scale cell-layout archive (6.24 G nnz,
# pfordelta/uint16, Rust engine, 2026-07-31): a ~50 k-row batch -- what this
# budget yields there -- decodes at 2.17 G nnz/s at n_threads=16, against
# 0.40 G nnz/s for a single 489-row group. A lone group is too little work to
# spread over the thread pool; bigger batches buy nothing and only cost host RAM.
_BATCH_TARGET_BYTES = 2 << 30           # 2 GiB
_CSR_INDEX_BYTES = 4                    # CellStore emits int32 indices


def _cell_bytes_per_nnz(manifest):
    """Decoded CSR bytes per nnz for this archive's value dtype.

    ``CellStore`` preserves float64 archives and narrows everything else to
    float32 (``cell/reader.py`` ``_x_out_dtype``), so a float64 zstd archive
    costs 12 B/nnz where a pfordelta/uint16 one costs 8.
    """
    data_bytes = 8 if manifest.get("value_dtype_on_disk") == "float64" else 4
    return data_bytes + _CSR_INDEX_BYTES


def _plan_batches(ranges, *, bytes_per_row, target_bytes=None):
    """Pack consecutive ``(label, start, stop)`` ranges into gather batches.

    Returns ``[(row_start, row_stop, [(label, local_start, local_stop), ...])]``
    where the local offsets index the CSR that one ``gather_rows(row_start ..
    row_stop)`` call returns. A group is **never split**: an oversized group
    forms its own batch, so ``refpool_de_core`` always sees each target group
    exactly once and whole.

    ``target_bytes=None`` resolves ``_BATCH_TARGET_BYTES`` **at call time**, not
    at def time, so a test can monkeypatch the module constant and have it take
    effect -- batch-size invariance is a real gate, and a default bound at def
    time would make it pass vacuously.
    """
    if target_bytes is None:
        target_bytes = _BATCH_TARGET_BYTES
    max_rows = max(1, int(target_bytes // max(1, int(bytes_per_row))))
    out, cur, cur_start, cur_stop = [], [], 0, 0
    for lab, start, stop in ranges:
        if cur and (stop - cur_start) > max_rows:
            out.append((cur_start, cur_stop, cur))
            cur = []
        if not cur:
            cur_start = start
        cur.append((lab, start - cur_start, stop - cur_start))
        cur_stop = stop
    if cur:
        out.append((cur_start, cur_stop, cur))
    return out


def _cell_group_ranges(store):
    """``(reference_labels | None, [(label, start, stop), ...])`` for a CellStore.

    Reads shardad's **private** ``CellStore._load_groups()``. Deliberate: the
    group table is exact, needs no I/O and no RAM, while the only public route
    to it (run detection over ``store.obs[group_by]``) would load the whole obs
    DataFrame -- 36 columns x 1.27 M rows on a production archive -- just to
    recover a table the archive already stores. An upstream ask is filed for a
    public ``group_ranges()``; ``test_cell_group_ranges_private_api_gone`` makes
    its removal fail loudly rather than silently.

    Validates what shardad's own ``read_reference()`` silently assumes: the
    reference groups lead contiguously from row 0, so its ``[0, max stop)``
    gather cannot pull target cells into the reference pool.
    """
    # The per-record parse is INSIDE the try: a record missing 'label'/'start'/
    # 'stop', or carrying a non-numeric bound, is the same class of failure as
    # the table itself changing shape, and should get the same explanatory
    # error rather than a bare KeyError/ValueError from a list comprehension.
    try:
        rec = store._load_groups()
        raw = rec["groups"]
        ref = rec["reference"]
        ranges = [(str(g["label"]), int(g["start"]), int(g["stop"])) for g in raw]
    except (AttributeError, KeyError, TypeError, ValueError) as e:
        raise RuntimeError(
            f"gpudge cannot read the group table of {store.path}: shardad's "
            f"private CellStore._load_groups() did not return the expected "
            f"{{'reference': ..., 'groups': [...]}} shape. gpudge pins "
            f"shardad>=0.7.1; see gpudge _cell_stream._cell_group_ranges."
        ) from e

    if not ranges:
        raise ValueError(
            f"{store.path}: cell archive has no group table (was it written "
            f"with group_by=?)."
        )
    labels = [lab for lab, _, _ in ranges]
    if len(set(labels)) != len(labels):
        seen, dup = set(), set()
        for lab in labels:
            (dup if lab in seen else seen).add(lab)
        raise ValueError(
            f"{store.path}: duplicate group labels {sorted(dup)} in the group table."
        )
    # An unassigned cell reaches the archive as a group literally named 'nan'
    # (shardad stringifies the obs column at write time), which the in-memory
    # path refuses at _ingest.py's source-level guard. Screen the strings here so
    # the two paths agree instead of streaming a bogus perturbation block.
    # Reading store.obs[group_by] to test isna() is exactly the whole-obs load
    # this function exists to avoid, so the string level is the right level.
    # (ultrareview 2026-08)
    reject_missing_group_labels(
        labels, where=str(store.path),
        remedy="Drop or assign the unassigned cells before writing the archive.")
    pos = 0
    for lab, start, stop in ranges:
        if start != pos or stop < start:
            raise ValueError(
                f"{store.path}: group ranges do not tile [0, n_obs) -- group "
                f"{lab!r} is [{start}, {stop}) but the previous group ended at "
                f"{pos}."
            )
        pos = stop
    if pos != int(store.n_obs):
        raise ValueError(
            f"{store.path}: group ranges cover {pos} rows but the archive has "
            f"{int(store.n_obs)}."
        )

    if ref is None:
        return None, ranges
    ref_labels = [str(x) for x in (ref if isinstance(ref, list) else [ref])]
    label_set = set(labels)          # hoisted: rebuilt per reference label otherwise
    missing = [lab for lab in ref_labels if lab not in label_set]
    if missing:
        raise ValueError(
            f"{store.path}: the manifest names reference label(s) {missing} that "
            f"are not in the group table."
        )
    lead = {lab for lab, _, _ in ranges[:len(ref_labels)]}
    if lead != set(ref_labels):
        raise ValueError(
            f"{store.path}: reference labels {sorted(ref_labels)} are not the "
            f"leading groups (rows from 0 start with {sorted(lead)}). shardad's "
            f"read_reference() gathers [0, max reference stop), so a non-leading "
            f"reference block would silently include target cells in the "
            f"reference pool."
        )
    return ref_labels, ranges


class _CellBackend:
    """``layout='cell'`` backend: one flat per-cell store, contiguous group
    ranges, batched thread-parallel ``gather_rows``.

    Unlike the shard backend there is no shard planner and no decode-ahead
    worker pool: ``gather_rows(n_threads=)`` parallelises the Rust decoder
    in-process at no extra host RAM, and decode is far off the critical path
    (see ``_BATCH_TARGET_BYTES``).
    """

    def __init__(self, store, *, n_workers):
        self._store = store
        self._n_threads = max(1, int(n_workers))
        man = store.manifest
        self.n_vars = int(store.n_vars)
        self.var_names = np.asarray(store.var.index)
        self.group_by = man.get("group_by")
        # No x_cupy equivalent exists for the cell codec anywhere in shardad;
        # stated rather than inferred from a missing schema_version attribute.
        self.supports_device_decode = False

        ref_labels, ranges = _cell_group_ranges(store)
        self._ref_labels = ref_labels
        self.has_archive_reference = ref_labels is not None
        ref_set = set(ref_labels or ())
        self._ref_stop = max((stop for lab, _, stop in ranges if lab in ref_set),
                             default=0)
        self._target_ranges = [r for r in ranges if r[0] not in ref_set]
        self._targets = [lab for lab, _, _ in self._target_ranges]
        self._tgt_index = {lab: i for i, lab in enumerate(self._targets)}
        self._max_group_rows = max(
            (stop - start for _, start, stop in self._target_ranges), default=0)

        # Mean nnz/row straight off the manifest -- no scan. total_nnz is
        # required by the cell format; the fallback only keeps a hand-built
        # manifest from crashing the planner.
        n_obs = max(1, int(store.n_obs))
        # `.get(key, default)` returns None when the key EXISTS and is null, so
        # a hand-built or future manifest carrying `"total_nnz": null` would
        # crash float(None) here rather than fall back. `or n_obs` also absorbs
        # a 0, which would make bytes_per_row 0 and the batch unbounded.
        _total_nnz = man.get("total_nnz") or n_obs
        mean_nnz = float(_total_nnz) / n_obs
        self._batches = _plan_batches(
            self._target_ranges,
            bytes_per_row=max(1, int(round(mean_nnz * _cell_bytes_per_nnz(man)))))
        logger.info(
            "cell-streaming: %d target groups in %d gather batches "
            "(n_threads=%d, ~%.0f nnz/row)",
            len(self._targets), len(self._batches), self._n_threads, mean_nnz)

    def close(self):
        # Close FIRST, clear only on success. CellStore.close() raises
        # BufferError while an mmap view is still exported (shardad
        # cell/reader.py); clearing the attribute first would throw away the
        # only handle that could retry, leaking the mapping.
        if self._store is not None:
            self._store.close()
            self._store = None

    def _gather(self, row_start, row_stop):
        rows = np.arange(row_start, row_stop, dtype=np.int64)
        return self._store.gather_rows(rows, n_threads=self._n_threads)

    def resolve_archive_reference(self, groupby, reference):
        """The archive's own reference pool as a host CSR.

        ``groupby`` is unused here -- unlike shard layout, the labels come from
        the group table, so there is no obs read and no NaN-label case. Kept in
        the signature so both backends share one contract.

        Deliberately does its own gather rather than calling
        ``store.read_reference()``: gpudge needs the same ``n_threads`` it uses
        everywhere else, and doing the gather here keeps the leading-reference
        guard in ``_cell_group_ranges`` on gpudge's side of the boundary.
        """
        del groupby
        labels = self._ref_labels
        if labels is None:
            # _resolve_streaming gates this on has_archive_reference, so this is
            # a direct-call guard: fail with the same message the shard backend
            # would, not a TypeError from set(None).
            raise ValueError(
                "archive has no reference. Either supply an external pool via "
                "reference=<AnnData>, or re-write the archive with "
                "shardad.write_sharded(..., reference=<label(s)>)."
            )
        msg_label = validate_archive_reference(reference, labels)
        ref_X = self._gather(0, self._ref_stop)
        return ref_X, msg_label

    def targets(self):
        return list(self._targets), int(self._max_group_rows)

    def target_row_sums(self):
        parts = [csr_row_sums(self._gather(rs, rt)) for rs, rt, _ in self._batches]
        return (np.concatenate(parts) if parts
                else np.zeros(0, dtype=np.float64))

    def target_source(self, need_row_sums):
        # Outer: gather batches of consecutive whole groups. Yields the absolute
        # target index g so the core writes accumulators by index; `del X, L`
        # releases the batch before the next gather, matching the shard path's
        # per-shard hygiene.
        for row_start, row_stop, members in self._batches:
            X = self._gather(row_start, row_stop)
            L = csr_row_sums(X) if need_row_sums else None
            for lab, lstart, lstop in members:
                rows = np.arange(lstart, lstop, dtype=np.int64)
                yield (self._tgt_index[lab], X, rows,
                       (L[rows] if L is not None else None))
            del X, L
