---
name: gpudge-usage
description: Use when installing gpudge or running GPU Mann–Whitney differential expression with it on single-cell CRISPR perturbation screens — covers pip/uv/conda install + extras, the de() entry point, one-vs-rest (ALL_OTHERS), shardad archive streaming (de(shard_archive=)), CPM / library-size normalization, and opt-in per-gene filters.
---

# Using gpudge

GPU-only Wilcoxon (Mann–Whitney U) differential expression for single-cell CRISPR
screens. One public entry point: `de()`. Bit-perfect vs CPU `pdex` on log2FC/p-value;
~100–600× faster on the DE stage. **Requires CUDA 12.6+, an H100/A100/Hopper GPU,
and an NVIDIA driver** — there is no CPU fallback.

## Install

Private repo → install from the git tag (needs Arc GitHub SSH access). Use the latest
release tag (shown here as `vX.Y.Z`).

```bash
# pip — torch's CUDA 12.6 build comes from the PyTorch index:
pip install --extra-index-url https://download.pytorch.org/whl/cu126 \
  "gpudge[fast] @ git+ssh://git@github.com/ArcInstitute/gpudge.git@vX.Y.Z"
```

```bash
# uv (development): clone, then
uv sync --extra fast        # or: --extra streaming
```

```bash
# conda/mamba: manages the env, pip installs gpudge into it
mamba env create -f environment.yml && mamba activate gpudge
```

**Extras:** `[fast]` = numba single-pass CSR kernel (~3× faster on big sparse X;
correctness identical without it). `[streaming]` = `shardad` for `de(shard_archive=)`
(also private — `pip install "shardad @ git+ssh://git@github.com/ArcInstitute/shardad.git@vX.Y.Z"` first).
`[dev]` = pytest/ruff/scanpy.

## Quickstart

`de()` takes an in-memory `AnnData` of **raw counts** (sparse CSR is fine — streamed
to GPU per gene-chunk, no host densify spike).

```python
import scanpy as sc
from gpudge import de

adata = sc.read_h5ad("filtered.h5ad")          # raw counts
# label each cell: control group vs per-target groups
adata.obs["comparison"] = ...                  # e.g. "ntc" vs target names

result = de(
    adata,
    groupby="comparison",
    reference="ntc",
    cpm_normalize=True,                        # CPM-normalize on GPU inline
)
result.write_parquet("de.parquet")
# columns (10): target, feature, target_mean, ref_mean, target_ncells,
#   ref_ncells, log2_fold_change, p_value, test_statistic, p_adj
```

**One-vs-rest:** `from gpudge import ALL_OTHERS; de(adata, groupby=..., reference=ALL_OTHERS)`.

## Streaming a shardad archive (low host RAM)

For datasets too large to materialize, stream a target-aware shardad archive — bounds
host RAM to ~reference + one shard. A **memory/feasibility** path, not a speedup.

```python
import gpudge
# Mode 1 — use the archive's own reference (non-targeting) shard:
df = gpudge.de(shard_archive="/path/to/archive")
# Mode 2 — external control pool as the reference (ntc_adata = any AnnData of NTC cells):
df = gpudge.de(shard_archive="/path/to/guides", reference=ntc_adata)
```

## Normalization — pick exactly one

- Pre-normalize yourself (`sc.pp.normalize_total`) and pass the result, **or**
- `cpm_normalize=True` — inline CPM (per-cell sum → 1e6), on GPU, does not mutate `adata.X`, **or**
- `normalize_target_sum=<number | "median">` — scanpy `normalize_total` parity.

`cpm_normalize` and `normalize_target_sum` are mutually exclusive. The two inline
options (and the `cpm_*` filters) **assume raw counts** — do **not** also pass them on
data you already normalized, or you double-normalize. Pick exactly one path.

## Key parameters

| Param | Default | Notes |
|---|---|---|
| `groupby` | required (in-memory) | obs column; auto-resolved from the archive manifest when streaming |
| `reference` | required (in-memory) | group name or `ALL_OTHERS` (`"__all_others__"`); when streaming, optional (uses the archive's reference shard) or an external `AnnData` |
| `mean_calc` | `"arithmetic"` | or `"geometric"` (matches pdex fold-change) |
| `epsilon` | `1e-9` | log pseudocount; matches scanpy |
| `filter_gene_min_mean_value` / `filter_gene_min_total_value` | `None` | opt-in per-(group,gene) filter on X as supplied |
| `filter_gene_min_cpm_cell` / `filter_gene_min_cpm_bulk` | `None` | opt-in; assume raw counts |
| `keep_genes` | `None` | bool mask on `var_names`; AND-combined with filters |
| `gpu_gene_chunk_size` | `None` | auto from free VRAM |
| `oom_recovery` | `True` | on CUDA OOM, halve chunk and retry |
| `densify_input` | `False` | dense per-group slicing; **mutates `adata.X` in place** |
| `cpm_normalize` | `False` | inline CPM normalization (per-cell sum → 1e6) on GPU; does not mutate `adata.X` |
| `normalize_target_sum` | `None` | inline library-size normalization (a number or `"median"`); mutually exclusive with `cpm_normalize` |
| `shard_archive` | `None` | path to a shardad archive → stream instead of an in-memory `adata` (low host RAM) |
| `output_columns` | `None` | rename/select dict, e.g. `{"log2_fold_change": "log2fc"}` |

All `filter_gene_*` are **opt-in** (default `None` = no filter) and AND-combined.

## Gotchas

- **GPU required** — no CPU path; needs a CUDA GPU even for tiny inputs.
- `cpm_*` filters and `cpm_normalize` **assume raw counts**; the cpm *filters* warn once on non-integer/negative X (`cpm_normalize` alone does not warn).
- `densify_input=True` **mutates `adata.X`** and emits a `UserWarning` — pass
  `adata.copy()` first if you need the sparse matrix preserved.
- Legacy `reference="all_others"` still works but is deprecated (use `ALL_OTHERS`).
- See the `de()` docstring in `src/gpudge/__init__.py` for the full parameter set.
