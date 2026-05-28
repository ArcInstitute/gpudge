# gpudge

Lightweight, GPU-only Wilcoxon (Mann–Whitney U) differential expression
for single-cell CRISPR perturbation screens. Built as a slim replacement
for the parts of [pdex](https://github.com/ArcInstitute/pdex) that the
Arc VCI pipeline uses.

## Why

- **One backend** (CUDA torch), no numba/CPU dispatch
- **Inline per-feature filter** (`min_feature_filter`) so the output
  matches the production CPU pipeline without a post-hoc step
- **Configurable output schema** via the `output_columns` rename/select dict
- **Sparse-aware**: streams gene-blocks from sparse `X` to GPU per chunk,
  no host densify spike

## Install

Requires CUDA 12.6+ and an H100 / A100 / Hopper GPU.

```bash
# Recommended: includes the numba CSR slicer (~3x faster on big sparse inputs)
pip install "gpudge[fast]"

# Or, scipy-only fallback:
pip install gpudge
```

The `[fast]` extra adds `numba` and enables a parallel single-pass CSR
row-gather kernel. Without it the scipy two-step
`X[rows, cols].toarray()` is used; correctness is identical.

## Usage

```python
import numpy as np
import scanpy as sc
from gpudge import de

adata = sc.read_h5ad("filtered.h5ad")
# Either pre-normalize, or pass cpm_normalize=True to do it inline:
sc.pp.normalize_total(adata, target_sum=1e6)
adata.obs["comparison"] = np.where(
    adata.obs["target_guide"].astype(str).str.startswith("non-targeting"),
    "ntc",
    adata.obs["target_guide"].astype(str),
)

result = de(
    adata,
    groupby="comparison",
    reference="ntc",
    min_feature_filter=1.0,       # CPM threshold on arithmetic means
)
result.write_parquet("target_de.parquet")
```

### One-vs-rest

```python
from gpudge import ALL_OTHERS
result = de(adata, groupby="comparison", reference=ALL_OTHERS)
```

(The pre-v0.1 spelling `reference="all_others"` is still accepted with a
`DeprecationWarning` and will be removed in v0.1.0.)

### Rename / select output columns

```python
result = de(
    adata, groupby="comparison", reference="ntc",
    output_columns={
        "target": "guide", "feature": "gene",
        "log2_fold_change": "log2fc",
        "p_value": "p",
        "p_adj": "fdr",
    },
)
# columns: guide, gene, log2fc, p, fdr
```

## API

See `de()` in `src/gpudge/__init__.py`.

| Kwarg | Default | Meaning |
|---|---|---|
| `groupby` | required | column in `adata.obs` |
| `reference` | required | group name or `ALL_OTHERS` (= `"__all_others__"`); legacy `"all_others"` accepted with `DeprecationWarning` |
| `mean_calc` | `"arithmetic"` | one of `arithmetic`, `geometric` |
| `epsilon` | `1e-9` | matches `scanpy.tl.rank_genes_groups` |
| `min_feature_filter` | `1.0` | per-(group, gene) arith-mean filter |
| `gpu_gene_chunk_size` | `None` | auto-pick from free GPU memory |
| `densify_input` | `False` | mutate `adata.X` to dense in place (emits `UserWarning`); faster per-group slicing when host RAM permits |
| `cpm_normalize` | `False` | inline CPM scaling (skips an upfront `sc.pp.normalize_total`); does *not* mutate `adata.X` |
| `output_columns` | `None` | rename + select dict; raises `KeyError` on unknown keys |

## Design

See the module docstrings in `src/gpudge/` for the algorithm and design notes.
