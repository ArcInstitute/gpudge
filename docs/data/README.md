# Example data

## `H1-VCC-2025-training.h5ad`

The dataset [`docs/tutorial.md`](../tutorial.md) and
[`examples/quickstart.py`](../../examples/quickstart.py) run on. Committed so a reader
has nothing to download.

| | |
|---|---|
| **Source** | Arc Virtual Cell Atlas — Virtual Cell Challenge 2025 **training** perturbation dataset (`adata_Training.h5ad`, 221,273 cells × 18,080 genes) |
| **Licence** | **CC0 1.0** (Creative Commons Zero, Public Domain Dedication) — <https://creativecommons.org/publicdomain/zero/1.0/> |
| **Where** | <https://arcinstitute.org/tools/virtualcellatlas> · <https://huggingface.co/datasets/arcinstitute/VCC_train> |
| **sha256** | `eb36c766cbf76353f9981cb3a3aa32137622d1de53b29d861c483742bcd4dec7` |
| **Size** | 4,991,092 bytes (600 cells × 1000 genes, CSR float32) |

CC0 places the data in the public domain and imposes no attribution requirement. The
attribution above is given anyway, because a reader should be able to find the source and
because gpudge's own MIT licence covers the *code*, not this file.

### What was changed

`../make_tutorial_data.py` is the exact recipe. Running it against the source above
reproduces the same **logical** subset — the same cells, genes and counts. The h5ad's
exact bytes, and so the checksum below, additionally depend on the anndata/h5py versions
that wrote it, so treat the checksum as identifying *this artifact* rather than as
something a rerun must reproduce. In summary:

- **Cells** — 100 drawn per group, without replacement, from
  `numpy.random.default_rng(0)`, for the six groups `non-targeting`, `TMSB4X`, `STAT1`,
  `MED12`, `TET1`, `SRC`. 600 rows.
- **Genes** — the 1000 with the highest total count *across the selected cells*, kept in
  original order.
- **obs** — reduced to `target_gene`, `guide_id`, `batch`.
- **X** — untouched: raw integer counts, exactly as in the source.

Nothing was recomputed, rescaled or imputed, so any statistic taken here is a statistic
of the public data.

### Relationship to cell_eval2

Byte-identical to the file the sibling tool cell_eval2 uses in its own tutorial — same
recipe, same seed — so the two can be followed against the same cells. Compare with the
sha256 above rather than assuming it.

### Why it is committed rather than downloaded

At 5 MB it is small enough for git, and a tutorial whose first step is a gated download
is a tutorial most readers never run. `tests/test_tutorial.py` asserts the file is
present, has this shape, and is still raw counts — a normalized replacement would make
the quickstart double-normalize and silently teach the mistake the tutorial warns about.
