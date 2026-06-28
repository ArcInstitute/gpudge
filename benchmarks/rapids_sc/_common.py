"""Pure (CPU, no GPU/rapids) helpers for the rapids_sc benchmark.

Kept import-light (numpy/polars/scipy/sklearn only) so the GPU runners and any
test suite can import the data-shaping + comparison logic without pulling in
torch, cupy, or rapids-singlecell.
"""
from __future__ import annotations

import numpy as np
import polars as pl
from scipy.stats import pearsonr, spearmanr

#: Result columns shared by both tools' long-form parquets.
VALUE_COLS = ("log2_fold_change", "p_value", "p_adj")


def collapse_reference(labels, prefix: str, reference: str) -> np.ndarray:
    """Map any group label starting with ``prefix`` to ``reference``; pass the
    rest through unchanged.

    Useful when a screen's control is split across many guides (e.g.
    ``non-targeting-1``, ``non-targeting-2``, ...) that should be pooled into a
    single reference group before the one-vs-reference test.
    """
    g = np.asarray(labels).astype(str)
    return np.where(np.char.startswith(g, prefix), reference, g)


def normalize_log1p_inplace(adata, target_sum: float = 1e6) -> None:
    """Copy-free CPM (``target_sum``) + log1p on a CSR AnnData, host-side, IN PLACE.

    Numerically equivalent to ``sc.pp.normalize_total(adata, target_sum) +
    sc.pp.log1p(adata)`` (float32) but WITHOUT the large transients that blow a
    fixed host-RAM budget on a many-billion-nnz matrix:

      * cast ``X.data`` uint*/int* -> float32 once, then drop the original buffer;
      * per-row CPM scaling via ``inplace_csr_row_scale`` (NO nnz-length
        ``np.repeat`` temporary -- that would materialize a buffer the size of
        the whole data array);
      * ``np.log1p(..., out=...)`` in place.

    Both runners call this (via ``--normalize inplace``) so the matched input is
    byte-identical across tools. X stays CSR float32.
    """
    import scipy.sparse as sp
    from sklearn.utils.sparsefuncs import inplace_csr_row_scale

    X = adata.X
    if not sp.isspmatrix_csr(X):
        X = sp.csr_matrix(X)
        adata.X = X
    # 1. single float32 cast; free the original (e.g. uint32) data array.
    if X.data.dtype != np.float32:
        old = X.data
        X.data = X.data.astype(np.float32)
        del old
    # 2. CPM row scaling in place. row_sums = per-cell library size via the
    #    CSR axis=1 sum (n_cells-length result, NOT an nnz-length temporary --
    #    this is the same quantity scanpy.normalize_total uses). Guard empty
    #    rows (sum 0 -> scale 1.0 so log1p(0)=0 is preserved).
    row_sums = np.asarray(X.sum(axis=1)).ravel().astype(np.float64)
    scale = np.divide(float(target_sum), row_sums,
                      out=np.ones_like(row_sums),
                      where=row_sums > 0).astype(np.float32)
    inplace_csr_row_scale(X, scale)
    # 3. log1p in place on the data buffer.
    np.log1p(X.data, out=X.data)


def reshape_rapids_uns(uns: dict) -> pl.DataFrame:
    """Reshape a scanpy/rapids-sc ``rank_genes_groups`` uns dict (structured
    arrays keyed by group) into a long polars frame with columns
    target, feature, log2_fold_change, p_value, p_adj (+ score if present)."""
    names = uns["names"]
    groups = list(names.dtype.names)
    pieces = []
    for g in groups:
        cols = {
            "target": [g] * len(names),
            # .astype(str) (not str(x)) so bytes-stored gene names decode to
            # 'GENE_A' rather than "b'GENE_A'", which would never join.
            "feature": np.asarray(names[g]).astype(str),
            "log2_fold_change": np.asarray(uns["logfoldchanges"][g], dtype="f8"),
            "p_value": np.asarray(uns["pvals"][g], dtype="f8"),
            "p_adj": np.asarray(uns["pvals_adj"][g], dtype="f8"),
        }
        if "scores" in uns:
            cols["score"] = np.asarray(uns["scores"][g], dtype="f8")
        pieces.append(pl.DataFrame(cols))
    return pl.concat(pieces)


def _corr(x: np.ndarray, y: np.ndarray) -> dict:
    m = np.isfinite(x) & np.isfinite(y)
    n = int(m.sum())
    if n > 1:
        pe = float(pearsonr(x[m], y[m]).statistic)
        sp = float(spearmanr(x[m], y[m]).correlation)
    else:
        pe = sp = None
    return {"pearson": pe, "spearman": sp, "n": n}


def compare(gpudge: pl.DataFrame, rapids: pl.DataFrame) -> dict:
    """Join two long result frames on (target, feature) and report row coverage
    plus Pearson/Spearman per value column."""
    gp = gpudge.select(["target", "feature"]).unique()
    rp = rapids.select(["target", "feature"]).unique()
    matched = gp.join(rp, on=["target", "feature"], how="inner").height
    coverage = {
        "gpudge_rows": gpudge.height,
        "rapids_rows": rapids.height,
        "matched_pairs": matched,
        "matched_pct_of_gpudge": 100 * matched / gpudge.height if gpudge.height else 0.0,
        "gpudge_only_pct": 100 * (gpudge.height - matched) / gpudge.height if gpudge.height else 0.0,
        "rapids_only_pct": 100 * (rapids.height - matched) / rapids.height if rapids.height else 0.0,
    }
    r_ren = rapids.rename({c: f"rapids_{c}" for c in VALUE_COLS if c in rapids.columns})
    j = gpudge.join(r_ren, on=["target", "feature"], how="inner")
    correlations = {}
    for c in VALUE_COLS:
        if c in gpudge.columns and f"rapids_{c}" in j.columns:
            correlations[c] = _corr(j[c].to_numpy(), j[f"rapids_{c}"].to_numpy())
    return {"coverage": coverage, "correlations": correlations}
