# gpudge

Lightweight, GPU-only Wilcoxon (Mann–Whitney U) differential expression
for single-cell CRISPR perturbation screens. Built as a slim replacement
for the parts of [pdex](https://github.com/ArcInstitute/pdex) that the
Arc VCI pipeline uses.

## Why

- **One backend** (CUDA torch), no numba/CPU dispatch
- **Opt-in per-gene filters** (`filter_gene_min_mean_value`,
  `filter_gene_min_cpm_cell`, etc.) — unit-named and AND-combinable;
  see the `de()` docstring for the full set
- **Configurable output schema** via the `output_columns` rename/select dict
- **Sparse-aware**: streams gene-blocks from sparse `X` to GPU per chunk,
  no host densify spike

## Install

Requires CUDA 12.6+, an H100 / A100 / Hopper GPU, and an NVIDIA driver. The
core library takes an **in-memory `AnnData`** and has **no private
dependencies**.

> **Not on PyPI yet.** `gpudge` is currently a private repo, so install from the
> git tag — requires GitHub SSH access to ArcInstitute. (PyPI distribution is TBD;
> once published, `pip install gpudge` will work directly.)

### pip

```bash
# torch's CUDA 12.6 build is pulled from the PyTorch index:
pip install --extra-index-url https://download.pytorch.org/whl/cu126 \
  "gpudge[fast] @ git+ssh://git@github.com/ArcInstitute/gpudge.git@v0.2.0"
```

Extras:
- **`[fast]`** — adds `numba` (parallel single-pass CSR row-gather; ~3× faster on
  big sparse inputs). Without it the scipy `X[rows, cols].toarray()` fallback is
  used; correctness is identical.
- **`[dev]`** — `pytest`, `ruff`, `scanpy`.
- **`[streaming]`** — adds [`shardad`](https://github.com/ArcInstitute/shardad)
  for `de(shard_archive=…)`. shardad is also private, so pip can't resolve it from
  PyPI — install it first from its git tag, then add the extra:
  ```bash
  pip install "shardad @ git+ssh://git@github.com/ArcInstitute/shardad.git@v0.2.0"
  ```

### uv (recommended for development)

```bash
git clone git@github.com:ArcInstitute/gpudge.git && cd gpudge
uv sync --extra fast          # reads [tool.uv.sources]: torch=cu126 index, shardad git pin
```
`uv sync --extra streaming` additionally pulls shardad from its git pin. uv reads
`[tool.uv.sources]`, so it resolves the cu126 torch and the private shardad
automatically (no manual `--extra-index-url` needed).

### conda / mamba

There is no conda package (gpudge is private; conda-forge is for public
packages). Use conda/mamba to **manage the environment**, then let pip install
gpudge into it — [`environment.yml`](environment.yml) does both:

```bash
mamba env create -f environment.yml   # or: conda env create -f environment.yml
mamba activate gpudge
```

(torch's CUDA libraries ship inside the pip wheel, so the conda env doesn't need
a CUDA toolkit — only a host NVIDIA driver + GPU.)

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
    # Filtering is opt-in; all filter_gene_* params default to None (no filter).
    # Example: keep genes with mean CPM ≥ 1 in target or reference group:
    # filter_gene_min_cpm_cell=1.0,
)
result.write_parquet("target_de.parquet")
```

### One-vs-rest

```python
from gpudge import ALL_OTHERS
result = de(adata, groupby="comparison", reference=ALL_OTHERS)
```

(The pre-v0.1 spelling `reference="all_others"` is still accepted with a
`DeprecationWarning` and will be removed in a future release.)

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
| `filter_gene_min_mean_value` | `None` | per-(group, gene) mean filter on `adata.X` as supplied; unit-agnostic |
| `filter_gene_min_total_value` | `None` | per-(group, gene) sum filter on `adata.X` as supplied; unit-agnostic |
| `filter_gene_min_cpm_cell` | `None` | per-cell CPM filter (assumes raw counts; warns on non-integer X) |
| `filter_gene_min_cpm_bulk` | `None` | pooled bulk CPM filter (assumes raw counts; same warning) |
| `keep_genes` | `None` | per-gene `np.bool_` mask aligned to `var_names`; AND-combined with other filters |
| `gpu_gene_chunk_size` | `None` | auto-pick from free GPU memory |
| `oom_recovery` | `True` | on CUDA OOM, halve the gene-chunk and retry (floor 64, or `chunk//2` if smaller); `False` = strict raise (benchmarking) |
| `densify_input` | `False` | mutate `adata.X` to dense in place (emits `UserWarning`); faster per-group slicing when host RAM permits |
| `cpm_normalize` | `False` | inline CPM scaling (skips an upfront `sc.pp.normalize_total`); does *not* mutate `adata.X` |
| `output_columns` | `None` | rename + select dict; raises `KeyError` on unknown keys |

## Design

See the module docstrings in `src/gpudge/` for the algorithm and design notes.
