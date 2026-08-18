"""Layout dispatch + the internal contract ``stream_de`` drives.

``stream_de`` names no shardad type. It opens a backend via ``open_backend``
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
  ``shardad.ShardedArchive``. **Legacy**: shard layout is slated for
  deprecation, and removing it is deleting that class and its module.
- ``_cell_stream._CellBackend`` -- ``layout='cell'``, wrapping
  ``shardad.CellStore``.
"""
from __future__ import annotations


def _import_shardad():
    try:
        import shardad
    except ImportError as e:  # pragma: no cover - exercised via monkeypatch
        raise ImportError(
            "de(archive=...) requires the optional 'streaming' extra, which "
            "installs shardad. shardad is hosted privately at "
            "ArcInstitute/shardad and is not published on PyPI, so the extra "
            "only resolves with access to that repository -- see the Install "
            "section of the README for the exact pin."
        ) from e
    return shardad


def open_backend(archive, *, n_workers, prefetch):
    """Open ``archive`` and return the backend for its layout.

    Dispatch is by attempt, not by file extension or manifest peek:
    ``ShardedArchive.__init__`` already raises ``IncompatibleSchemaError`` on a
    ``layout='cell'`` archive and ``shardad.open`` raises the same on a
    shard-layout one, so one cheap failed open resolves the layout in both
    directions -- and an archive with a misleading extension still works.

    On the cell branch the shard error is chained under whatever
    ``shardad.open`` reports, so a genuinely corrupt archive surfaces its real
    error rather than a misleading cell-flavoured one. A store that opens but
    whose backend construction then fails is closed before the error escapes.
    """
    shardad = _import_shardad()
    from shardad.errors import IncompatibleSchemaError

    try:
        arch = shardad.ShardedArchive(archive)
    except IncompatibleSchemaError as shard_err:
        # Import BEFORE opening the store: an ImportError here (a partial
        # install, a circular-import regression) would otherwise strand an
        # already-open CellStore -- the one leak path the try/finally below
        # cannot cover, because there is nothing to close yet.
        from ._cell_stream import _CellBackend
        try:
            store = shardad.open(archive)
        except Exception as cell_err:
            raise cell_err from shard_err
        try:
            return _CellBackend(store, n_workers=n_workers)
        except BaseException:
            store.close()
            raise
    from ._shard_stream import _ShardBackend
    return _ShardBackend(arch, n_workers=n_workers, prefetch=prefetch)
