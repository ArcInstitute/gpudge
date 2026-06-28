"""Build a dask-backed cupy-CSR ``adata.X`` so rapids-singlecell's binned
Wilcoxon can stream cell-blocks to the GPU instead of moving the whole CSR at
once (which OOMs on a large matrix). In-memory binned calls X_to_GPU on the full
matrix; the dask path moves one block at a time.

Normalize on the CPU (scipy) BEFORE calling this so we don't depend on
rapids-singlecell's pp.* supporting dask-cupy; the binned kernel itself does
support dask arrays (uses bin_range='log1p' to skip the data scan).
"""
from __future__ import annotations


def to_dask_gpu(adata, chunk_cells: int = 100_000) -> None:
    """Replace ``adata.X`` (host scipy CSR) with a dask array of cupy CSR
    blocks, chunked along the cell axis. Mutates ``adata`` in place."""
    import cupyx.scipy.sparse as cpsp
    import dask
    import dask.array as da

    X = adata.X
    n = X.shape[0]
    meta = cpsp.csr_matrix((0, X.shape[1]), dtype=X.dtype)

    def _block_to_gpu(block):
        return cpsp.csr_matrix(block)

    blocks = []
    for start in range(0, n, chunk_cells):
        host_block = X[start : start + chunk_cells]
        delayed = dask.delayed(_block_to_gpu)(host_block)
        blocks.append(
            da.from_delayed(delayed, shape=host_block.shape, dtype=X.dtype, meta=meta)
        )
    adata.X = da.concatenate(blocks, axis=0)
