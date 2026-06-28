# gpudge

Lightweight, GPU-only Wilcoxon rank-sum (Mann–Whitney U) differential expression
for single-cell CRISPR perturbation screens. Built as a slim replacement
for the parts of [pdex](https://github.com/ArcInstitute/pdex) that the
Arc VCI pipeline uses.

## Why

- **Fast:** GPU Mann–Whitney in a single pass — on the cell line 1/500-perturbation DE stage, **~4,500× vs scanpy**, **~600× vs CPU pdex**, **~140× vs rapids-singlecell**; DE time stays near-constant as the perturbation count grows
- **Low host RAM**: can use shardad to stream AnnData shards without loading full matrix into RAM, enabling scaling to tens of millions of cells
- **Opt-in per-gene filters** (`filter_gene_min_mean_value`,
  `filter_gene_min_cpm_cell`, etc.) — unit-named and AND-combinable;
  see the `de()` docstring for the full set
- **Configurable output schema** via the `output_columns` rename/select dict


## Performance

Benchmarked on **NVIDIA H100 80 GB** unless noted (gpudge v0.1–v0.2; the
Mann–Whitney DE core is unchanged in v0.3.0). gpudge runs the whole test as
**one GPU pass**, so its **DE time is near-constant in the number of
perturbations** — where per-group tools (scanpy, rapids-singlecell) scale
roughly linearly.

**DE-stage time** (the engine's DE call; excludes host load + normalize), at the
highest perturbation rung where all four engines complete:

| Dataset | Perturbations | scanpy (CPU) | pdex (CPU) | rapids-singlecell (GPU) | gpudge (GPU) |
|---|--:|--:|--:|--:|--:|
| cell line 1 | 500 | 3.3e4 s (9.0 h) | 4.3e3 s (71 min) | 1.0e3 s (17 min) | **7.2 s** |

pdex — the optimized parallel-CPU reference gpudge replaces — does the same
cell line 1/500 rung in **71 min on 32 cores** (4,285 s) vs gpudge's **7.2 s**: a
**~595×** DE-stage speedup, with **bit-identical** results (log2FC and p-value
Pearson > 0.999999999 over all 5.5 M gene–perturbation pairs gpudge reports).

Above these rungs the other engines drop out — rapids-singlecell's VRAM grows
with the perturbation count and **OOMs** (cell line 1 full, cell line 2 ≥200), and the scanpy
CPU runs become multi-day. gpudge keeps going at near-constant DE cost:

| Full guide/perturbation set | gpudge DE | Platform |
|---|--:|---|
| cell line 2 — 2.06 M cells × 4,672 guides | **51 s** | H100 80 GB |
| cell line 2 — 2.06 M cells × 4,672 guides | **92 s** (176 s end-to-end) | GCP A100 40 GB |

gpudge VRAM stays flat (~15 GB cell line 1 / ~29 GB cell line 2); the A100-40 GB run fits in
< 85 GB host RAM and reproduces the CPU pdex baseline bit-for-bit. The same cell line 2
DE takes **4 h 11 min on the production CPU pdex pipeline** (~650 GB RAM).

**Full 5.54 M-cell cell line 2 — RAM by data layout.** The full raw cell line 2 (5.54 M
cells × 18,533 genes × 5,095 perturbations; 36 B non-zeros, 94 M result rows)
runs three ways on a single H100, with **bit-identical** results across all
layouts (max abs diff 0.0 on log2FC / p-value / p_adj across all 94 M rows):

| Path | Host RAM | Load | DE | Wall |
|---|--:|--:|--:|--:|
| **Full matrix** (float32 data + int64 indices; standard scipy CSR) | 448 GB | 253 s | 181 s | 7.2 min |
| **Narrowed in-memory** (uint16 data + int32 indices via shardad; normalize inside gpudge) | **236 GB** | 88 s | 75 s | **2.7 min** |
| **Streaming** (`de(shard_archive=…)`; reference GPU-resident, shards streamed) | **31 GB** | — | — | ~20 min |

Narrowing the in-memory CSR is the key trick: loading **uint16 counts + int32
indices** (via shardad) and letting gpudge normalize on-GPU
(`normalize_target_sum=`) — instead of materializing a float copy on the host —
roughly **halves both host RAM (448→236 GB) and DE time** versus the full
float32/int64 matrix, for the same bit-identical output. The DE speedup is a
host-side effect: the per-chunk CSR→dense gather and the row-sum normalize pass
are memory-bandwidth-bound over `X.data` + `X.indices`, so the narrow layout
(6 B/non-zero vs 12) halves the bytes streamed — the on-GPU work and the
host→GPU transfer are unchanged (the gather emits float32 either way). The full
matrix is only worth its wider dtypes if a downstream non-gpudge consumer needs
them.

Streaming interleaves shard I/O with compute (so it has no separate load/DE
split) and trades ~7× wall (vs narrowed) for an order-of-magnitude lower
host-RAM floor, for datasets too large to materialize at all.

Results are concordant with scanpy/pdex (log-fold-change sign-agreement 1.0,
top-50 Jaccard 1.0, identical significant-gene sets, matched NTC false-positive
rate) and bit-perfect vs CPU pdex on log2FC/p-value.

## Install

Requires CUDA 12.6+, an H100 / A100 / Hopper GPU, and an NVIDIA driver. The
core library takes an **in-memory `AnnData`** and has **no private
dependencies**.

### pip

```bash
# torch's CUDA 12.6 build is pulled from the PyTorch index:
pip install --extra-index-url https://download.pytorch.org/whl/cu126 \
  "gpudge[fast] @ git+ssh://git@github.com/ArcInstitute/gpudge.git@v0.3.0"
```

Extras:
- **`[fast]`** — adds `numba` (parallel single-pass CSR row-gather; ~3× faster on
  big sparse inputs). Without it the scipy `X[rows, cols].toarray()` fallback is
  used; correctness is identical.
- **`[dev]`** — `pytest`, `ruff`, `scanpy`.
- **`[streaming]`** — adds [`shardad`](https://github.com/ArcInstitute/shardad)
  for `de(shard_archive=…)`. shardad is also private, so pip can't resolve it from
  PyPI — install it first, then add the extra. The shard-streaming reader API
  postdates shardad's `v0.2.0` tag, so pin the commit gpudge tracks (the same rev
  as `[tool.uv.sources]` in `pyproject.toml`):
  ```bash
  pip install "shardad @ git+ssh://git@github.com/ArcInstitute/shardad.git@35e82bf4bd3b0b297847fc2c4fee640620ee78b9"
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
import scanpy as sc
from gpudge import de

# cell line 2 CRISPRi screen (2.06 M cells), raw counts. Perturbation labels are in
# adata.obs["target_gene"]; non-targeting controls are labelled "non-targeting".
adata = sc.read_h5ad("screen.h5ad")

result = de(
    adata,
    groupby="target_gene",
    reference="non-targeting",
    # The four parameters below reproduce the production CPU pdex output:
    mean_calc="geometric",          # pdex uses the geometric-mean fold change
    epsilon=0.0,                    # pdex adds no pseudocount (de()'s default is 1e-9)
    cpm_normalize=True,             # CPM-normalize on the GPU — pass raw counts in
    filter_gene_min_cpm_cell=1.0,   # pdex's per-(group, gene) 1-CPM filter (filters default None)
)
result.write_parquet("de_results.parquet")
# columns (10): target, feature, target_mean, ref_mean, target_ncells,
#   ref_ncells, log2_fold_change, p_value, test_statistic, p_adj
```

> **pdex parity.** The four annotated parameters above are the recipe to match
> the production CPU [pdex](https://github.com/ArcInstitute/pdex) pipeline —
> together they reproduce its output bit-for-bit (validated on cell line 1 and cell line 2).
> `de()`'s own defaults are `epsilon=1e-9` with no normalization or filtering;
> with pdex's `epsilon=0.0`, genes whose target or reference group mean is
> exactly 0 yield ±inf / NaN `log2_fold_change` (matching pdex).

### One-vs-rest

```python
from gpudge import ALL_OTHERS
# each target_gene vs all other cells (ALL_OTHERS requires mean_calc="arithmetic", the default)
result = de(adata, groupby="target_gene", reference=ALL_OTHERS)
```

(The pre-v0.1 spelling `reference="all_others"` is still accepted with a
`DeprecationWarning` and will be removed in a future release.)

### Streaming a shardad archive

```python
import gpudge

# Stream the full 5.54 M-cell cell line 2 archive — groupby/reference are auto-resolved
# from its manifest (target_gene / non-targeting); host RAM stays ~one shard:
df = gpudge.de(shard_archive="/path/to/archive.shad")

# Optionally validate the archive's reference label:
df = gpudge.de(shard_archive="/path/to/archive.shad", reference="non-targeting")

# Mode 2 — external control pool as the reference (ntc_adata = any AnnData of NTC
# cells); all archive shards become targets. If the archive has its own reference
# shard it is ignored (with a UserWarning) in favor of the external pool:
df = gpudge.de(shard_archive="/path/to/guides", reference=ntc_adata)
```

Bounds host RAM to ~reference + one shard; requires `pip install gpudge[streaming]`.
A memory/feasibility path (not a speedup).

### Rename / select output columns

```python
result = de(
    adata, groupby="target_gene", reference="non-targeting",
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

See `de()` in `src/gpudge/__init__.py`. gpudge also exports the `MeanCalc`
type alias (the `mean_calc` literal) and `__version__`.

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
| `normalize_target_sum` | `None` | inline scanpy-compatible library-size normalization; pass a positive number or `"median"`; mutually exclusive with `cpm_normalize=True` |
| `output_columns` | `None` | rename + select dict; raises `KeyError` on unknown keys |

## Design

See the module docstrings in `src/gpudge/` for the algorithm and design notes.
