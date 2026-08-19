"""Layout dispatch + the internal contract ``stream_de`` drives.

``stream_de`` names no cellstream type. It opens a backend via ``open_backend``
and talks only to the surface below, so a new archive layout is a new backend
rather than a new branch in the driver.

Backend surface (plain instance attributes unless noted):

    n_vars                  int -- gene count
    var_names               (n_vars,) ndarray -- archive gene order
    group_by                str | None -- the archive's group_by obs column
    has_archive_reference   bool -- metadata only, NO data I/O
    supports_device_decode  bool

    resolve_archive_reference(groupby, reference) -> (ref_X, msg_label)
        Validate + read the archive's own reference pool ONCE and return it as
        a host CSR. Folded into the backend (rather than split into a
        ``reference_labels`` property plus a separate read) precisely so the
        shard path keeps reading its reference shard exactly once: its labels
        live in that shard's own obs.

    targets() -> (labels: list[str], max_group_rows: int)
        Ordered target labels + the largest target group's row count. Cached
        and idempotent; ``target_source`` relies on the label->index map it
        builds.

    target_row_sums() -> (n_target_cells,) float64
        Per-cell library sizes over TARGET cells only, in ``target_source``
        order. Used only by ``normalize_target_sum='median'``. The reference's
        own row sums are prepended by ``stream_de``, because in Mode 2 the
        reference is an external AnnData the backend never sees.

    target_source(need_row_sums) -> yields (g, X, rows, Ls_for_rows)
        Exactly the 4-tuple ``_refpool.refpool_de_core`` consumes.

    close()
        Release the archive's OS resources. ``CellStore`` owns an mmap and an
        open file handle; refcounting is not enough, because an exception
        raised during DE keeps the backend alive through the traceback.
        ``stream_de`` wraps the backend lifetime in try/finally.

Implementations:

- ``_shard_stream._ShardBackend`` -- ``layout='shard'``, wrapping
  ``cellstream.ShardedArchive``. **Legacy**: shard layout is slated for
  deprecation, and removing it is deleting that class and its module.
- ``_cell_stream._CellBackend`` -- ``layout='cell'``, wrapping
  ``cellstream.CellStore``.
"""
from __future__ import annotations


def validate_archive_reference(reference, labels):
    """Validate ``reference=`` against an archive's OWN reference labels.

    Returns the message label to report. Raises ``ValueError`` on an unknown
    label.

    Both backends gather the reference pool WHOLE -- the cell backend reads
    ``[0, ref_stop)``, the shard backend reads its reference shard -- so
    ``reference=`` here NAMES the pool rather than subsetting it. That is
    deliberate and specified: "``reference=<label>`` is validated for membership
    in the reference labels and otherwise does not subset -- the pool is all
    reference rows".
    A single label against a multi-label pool is therefore ACCEPTED, unchanged.

    What this helper fixes is the sequence case. ``reference=['ntc', 'safe']``
    was stringified whole by the old membership test, so a list that exactly
    matched the archive's labels produced the self-contradictory
    ``reference=['ntc', 'safe'] is not among the archive's reference labels
    ['ntc', 'safe']``. A sequence is now checked element-wise.
    """
    def _label(x):
        return x.decode() if isinstance(x, (bytes, bytearray)) else str(x)

    label_list = sorted(labels)
    if reference is None:
        return "|".join(label_list)
    # str/bytes are sequences; test them first or 'ntc' iterates to
    # {'n','t','c'}. `str(...)` on the str arm too: np.str_ passes the
    # isinstance check (it subclasses str) and would otherwise leak out as
    # np.str_ rather than a plain label.
    if isinstance(reference, (str, bytes, bytearray)):
        named = [_label(reference)]
    else:
        # iter(), not isinstance(_, Iterable): a 0-d np.array('ntc') passes the
        # Iterable check via __getitem__ and then raises TypeError when
        # iterated. Materialized in one pass -- `reference` may be a generator,
        # and it is read twice below (membership, then the message).
        try:
            named = [_label(x) for x in iter(reference)]
        except TypeError:
            # A 0-d array: unwrap it rather than str()-ing the array, or
            # np.array(b"ntc") becomes the literal "np.bytes_(b'ntc')".
            item = (reference.item()
                    if getattr(reference, "ndim", None) == 0 else reference)
            named = [_label(item)]
    if not named:
        raise ValueError(
            f"reference={reference!r} is an empty sequence. Pass reference=None "
            f"to use the archive's reference pool {label_list}, or name it."
        )
    unknown = sorted(set(named) - set(label_list))
    if unknown:
        raise ValueError(
            f"reference={reference!r} is not among the archive's reference "
            f"labels {label_list}."
        )
    # The pool is used whole either way, so the label reported for it is the
    # caller's when they named exactly one and the joined set otherwise.
    return named[0] if len(set(named)) == 1 else "|".join(label_list)


def _import_cellstream():
    try:
        import cellstream
    except ImportError as e:  # pragma: no cover - exercised via monkeypatch
        raise ImportError(
            "de(archive=...) requires the optional 'streaming' extra, which "
            "installs cellstream (>=0.9.0, the former shardad) from PyPI. "
            "Reinstall gpudge with the extra -- e.g. `pip install '.[streaming]'` "
            "from a checkout, or add `streaming` to the pin in the README's "
            "Install section."
        ) from e
    return cellstream


def open_backend(archive, *, n_workers, prefetch):
    """Open ``archive`` and return the backend for its layout.

    Dispatch is by attempt, not by file extension or manifest peek:
    ``ShardedArchive.__init__`` already raises ``IncompatibleSchemaError`` on a
    ``layout='cell'`` archive and ``cellstream.open`` raises the same on a
    shard-layout one, so one cheap failed open resolves the layout in both
    directions -- and an archive with a misleading extension still works.

    On the cell branch the shard error is chained under whatever
    ``cellstream.open`` reports, so a genuinely corrupt archive surfaces its real
    error rather than a misleading cell-flavoured one. A store that opens but
    whose backend construction then fails is closed before the error escapes.
    """
    cellstream = _import_cellstream()
    from cellstream.errors import IncompatibleSchemaError

    try:
        arch = cellstream.ShardedArchive(archive)
    except IncompatibleSchemaError as shard_err:
        # Import BEFORE opening the store: an ImportError here (a partial
        # install, a circular-import regression) would otherwise strand an
        # already-open CellStore -- the one leak path the try/finally below
        # cannot cover, because there is nothing to close yet.
        from ._cell_stream import _CellBackend
        try:
            store = cellstream.open(archive)
        except Exception as cell_err:
            raise cell_err from shard_err
        try:
            return _CellBackend(store, n_workers=n_workers)
        except BaseException:
            store.close()
            raise
    from ._shard_stream import _ShardBackend
    return _ShardBackend(arch, n_workers=n_workers, prefetch=prefetch)
