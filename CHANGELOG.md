# Changelog

All notable changes to gpudge are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-05-27

First tagged release. Consolidates the foundational `de()` API, the
sparse-aware streaming pipeline, the shardad v0.2.0 integration, and
the post-numba performance series (T1–T5) on a single H100.

### Added

- `de(adata, groupby, reference, ...)` — public entry point for GPU
  Mann–Whitney U differential expression with per-group BH-FDR.
- Two reference modes:
  - Literal reference group (supports `mean_calc="arithmetic"` and
    `"geometric"`).
  - `ALL_OTHERS` sentinel for one-vs-rest comparisons (arithmetic
    means only — geometric is rejected up front).
- Inline per-`(group, gene)` `min_feature_filter` (default 1.0 CPM)
  that matches the production CPU pdex output without a post-hoc
  filtering step.
- `cpm_normalize=True` kwarg — inline CPM scaling on the GPU; skips
  an upfront `sc.pp.normalize_total` without mutating `adata.X`.
- `densify_input=True` kwarg — opt-in in-place rebind of sparse
  `adata.X` to a dense numpy array. Faster per-group slicing when
  host RAM permits (~154 GB for cell line 2); emits a `UserWarning`.
- `output_columns` rename/select dict — configurable output schema,
  raises `KeyError` on unknown keys.
- Sparse-aware gene-chunk streaming: chunks are sliced from CSR `X`
  to a pre-pinned host dense buffer, then async H2D to a torch
  tensor — no host densify spike for cell line 2-scale inputs.
- Optional `[fast]` extra: numba-accelerated CSR row + col-range
  slicer (`_row_col_slice_np`). With it, per-group slicing is a
  single parallel CSR-gather kernel. Without it, the scipy
  two-step `X[rows, cols].toarray()` fallback is used; correctness
  is bit-identical.
- shardad v0.1 / v0.2 archive loading via the top-level
  `shardad.read_h5ad` API. v1 archives are recommended for cell line 2-scale
  inputs (v2 load is ~16× slower at that scale — see Known issues).
- Per-group BH-FDR via a fused counting-sort + per-segment kernel
  (vectorised across all groups in a single pass).
- Double-buffered pinned H2D + per-chunk GPU accumulators for
  per-group ranks; batched D2H once per gene-chunk.
- Public sentinel `ALL_OTHERS = "__all_others__"` (the legacy
  spelling `"all_others"` is still accepted — see Known issues).

### Performance

Benchmark dataset: **cell line 2 deep CRISPRi screen** — 2,064,002 cells ×
18,533 genes, 4,672 target guides, ~10% density, 13.4B nonzeros.
Hardware: single **NVIDIA H100 80GB HBM3**, CUDA 12.6, torch
2.12.0+cu126. Numbers are warm-cache, single-run `de()` wall times
on a clean GPU node.

| Configuration                                          | `de()` wall |
| :----------------------------------------------------- | ----------: |
| **v0.1.0 (this release, `[fast]` extra)**              |    **~51s** |
| Post-numba CSR kernel, pre-perf series (T0)            |       ~108s |
| scipy two-step fallback (no `[fast]` extra)            |       ~313s |

The `[fast]` extra is required to hit the headline number. The
scipy fallback path remains correct but is roughly 6× slower at
cell line 2 scale.

The benchmark harness is maintained separately and is not included in this release.

### Accuracy

Compared against the production CPU pdex pipeline on the same cell line 2
dataset (`target_de.parquet`, 52,703,861 output rows):

| Metric                                  | Value                          |
| :-------------------------------------- | :----------------------------- |
| Row coverage of CPU pdex                | **100.00%** (identical row set) |
| `log2_fold_change` pearson              | 0.999_999_998_4 (n=52.69M)     |
| `log2_fold_change` spearman             | 0.999_999_993_6                |
| `p_value` pearson                       | 0.999_999_999_9 (n=52.70M)     |
| `p_value` spearman                      | 0.999_999_999_9                |
| `p_adj` pearson                         | 0.999_999_999_6 (n=52.70M)     |
| `p_adj` spearman                        | 0.999_999_977_6                |

Residual numerical drift is dominated by float32 → float64
rank-sum precision and BH-FDR tie ordering, both within float64
ULP of the CPU baseline on every assertable column.

### Dependencies

- Python ≥ 3.12
- CUDA 12.6+ with an H100 / A100 / Hopper GPU
- `torch ≥ 2.5`, `shardad ≥ 0.2.0`, `anndata ≥ 0.12`,
  `polars ≥ 1.38`, `numpy ≥ 2`, `scipy ≥ 1.17`, `pyarrow ≥ 20`,
  `hdf5plugin ≥ 6.0`
- Optional `[fast]`: `numba ≥ 0.59`
- Optional `[dev]`: `pytest ≥ 8`, `ruff ≥ 0.6`, `scanpy ≥ 1.12`

### Known issues

- `reference="all_others"` (lowercase string) is still accepted
  with a `DeprecationWarning`. The deprecation message currently
  states removal in v0.1.0; that removal is **deferred** to a
  later release. New code should pass the `ALL_OTHERS` constant
  (or the string `"__all_others__"`) instead.
- shardad v2 archive loading is roughly 16× slower than v1 at cell line 2
  scale (237s vs 17s). Stay on v1 archives for cell line 2-scale inputs
  until shardad v0.3 lands an on-GPU read path.

[0.1.0]: https://github.com/ArcInstitute/gpudge/releases/tag/v0.1.0
