# Changelog

All notable changes to gpudge are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

_Nothing yet._

## [0.3.1] — 2026-06-27

Maintenance + licensing release. **No API or numeric changes** to the validated
Mann–Whitney DE path — every fix touches degenerate/unreachable cases, input
validation, or metadata (GPU-verified on H100: 220 passed, 1 skipped).
Distribution remains **Arc-internal** — install from the `v0.3.1` git tag.

### Changed

- **Relicensed BSD-3-Clause → MIT.** Added a top-level `LICENSE` file and
  reconciled the `pyproject.toml` `license` field to MIT. Copyright holder: Arc
  Research Institute. (#60)

### Fixed

- Hardening pass — a second multi-agent "ultrareview" addressing all 22
  confirmed findings (report under
  `docs/reviews/2026-06-27-gpudge-ultrareview.md`): reject missing (NaN/None)
  `groupby` labels in both the in-memory and shard-streaming reference paths;
  degenerate-case sentinels (single-cell `N==1` groups, empty groups); tighter
  `epsilon` / `output_columns` / `reference` validation; `ALL_OTHERS` and
  shard-streaming auto-chunk-budget fixes; `/dev/shm` cleanup; and added
  CPU-runnable MWU-vs-scipy, log2FC, and multi-block test coverage. The
  validated path is unchanged (numeric fixes touch only degenerate/unreachable
  cases). (#59)
- `stream_de()` now enforces epsilon-finiteness parity with `de()` and
  fail-fast, archive-free input validation — empty / unknown / duplicate
  `output_columns`, an unknown `mean_calc`, and a non-finite `epsilon` are all
  rejected before the archive is opened — covered by a CI-runnable parity test.
  (#60)

## [0.3.0] — 2026-06-24

Feature + tooling release: native shard-streaming `de(shard_archive=…)`,
scanpy-compatible library-size normalization, the multi-agent "ultrareview"
hardening pass, a published benchmark campaign (vs scanpy / CPU pdex /
rapids-singlecell), and packaging / CI / docs work. **No breaking changes since
v0.2.0** (the Mann–Whitney DE core is unchanged).

Distribution remains **Arc-internal** — install from the `v0.3.0` git tag
(requires ArcInstitute SSH access); not on PyPI.

### Added

- `de(shard_archive=...)`: native shard-streaming over a shardad target-aware v2
  archive — keeps the reference pool resident-sorted on the GPU and streams
  guide-shards past it, bounding host RAM to ~reference + one shard. A
  memory/feasibility path for datasets too large to materialize in host RAM
  (e.g. the full 5.54 M-cell cell line 2 runs in ~31 GB host), **not a speedup** — it is
  ~7× slower wall-clock than the narrowed in-memory path. Supports Mode 1
  (archive reference shard as the pooled control) and Mode 2 (external
  `reference=<AnnData>` pool). `ALL_OTHERS` and oversized references are
  unsupported in v1. Requires the optional `[streaming]` extra. (#36)
- `de(normalize_target_sum=...)` — scanpy-compatible on-the-fly library-size
  normalization. Accepts a positive number (`normalize_total(target_sum=N)`) or
  `"median"` (scanpy's default `target_sum=None`). `cpm_normalize=True` is
  equivalent to `normalize_target_sum=1e6`; exactly one of the two may be set.
  Supported in both in-memory and shard-streaming modes (streaming `"median"`
  adds an up-front row-sum pass over the archive). (#52)

### Changed

- Shard-streaming Mode 2: passing an external `reference=<AnnData>` against an
  archive that *also* carries its own reference shard now **warns and uses the
  external pool** (the archive reference is ignored — "Semantics A") instead of
  raising. The earlier no-reference-archive requirement is relaxed to a
  `UserWarning`. (#56)

### Fixed

- ultrareview hardening — shared-validation parity between the in-memory and
  shard-streaming paths: the legacy `reference="all_others"` remap and the
  `mean_calc` / `epsilon` / `output_columns` validation are hoisted above the
  streaming dispatch, so both paths reject identical inputs before any GPU work;
  the empty-archive early return now emits the typed canonical output schema
  (and honours `output_columns`) instead of all-`Null` columns. (#48)
- `_csr_dense` numba kernels now **sum duplicate column indices within a row**
  (matching scipy `.toarray()`; previously last-write-wins, so the `[fast]` path
  could silently disagree with the scipy fallback on non-canonical CSR), and
  validate row indices are in `[0, n_rows)` (raising `IndexError` instead of
  reading `indptr` out of bounds under `boundscheck=False`). Canonical CSR is
  unaffected. (#48)
- OOM-recovery driver returns its final (possibly downshifted) gene-chunk width
  so the shard-streaming driver carries it across target groups instead of
  re-discovering — and re-paying — the downshift on every group; trailing-chunk
  H2D in the ref-mode group loop is repacked contiguous to preserve the async
  pinned copy; geometric-mean out-of-domain contract pinned + documented
  (`X < -1` → NaN, `X == -1` → -1.0). (#42, #43, #46, #47, #49)

### Performance / Accuracy

- Published a benchmark campaign comparing
  gpudge against scanpy (CPU), CPU pdex, and rapids-singlecell (GPU) on cell line 1 and
  cell line 2. Because gpudge runs the whole screen as one GPU pass, its DE-stage time
  is **near-constant in the perturbation count**, where per-group tools scale
  ~linearly. Headline DE-stage results (H100 80 GB):
  - **cell line 1 / 500 perturbations** (239,054 cells × 18,533 genes): gpudge
    **7.2 s** vs CPU pdex **4,285 s** (71 min, 32 workers) = **~595×**, results
    **bit-identical** (log2FC & p-value Pearson > 0.999999999 over 5.5 M
    gene–perturbation pairs); vs rapids-singlecell ~1.0e3 s, vs scanpy ~9 h.
  - rapids-singlecell VRAM grows with the perturbation count and **OOMs** at
    higher rungs (cell line 1 full, cell line 2 ≥ 200); gpudge VRAM stays flat (~15 GB cell line 1 /
    ~29 GB cell line 2). (#50, #51)

### Packaging / CI

- First CI workflow (`.github/workflows/ci.yml`): `ruff` + CPU `pytest` on every
  push / PR (ubuntu-latest, Python 3.12). Installs via
  `uv pip install -e ".[dev,fast]"` so the private shardad git-SSH source is
  never resolved on the runner; GPU (`needs_cuda`), shardad streaming
  (`importorskip`), and real-data tests auto-skip. The `ruff` gate is blocking;
  the repo was brought to zero lint errors (47 fixed). Green at 121 passed /
  53 skipped. (#53)

### Docs / Tooling

- `docs/THIRD_PARTY_LICENSES.md` — third-party license audit. (#38)
- `gpudge-usage` reference skill (`.claude/skills/gpudge-usage/`) — install +
  `de()` usage guide covering pip/uv/conda + extras, one-vs-rest (`ALL_OTHERS`),
  shardad archive streaming, CPM / library-size normalization, and the opt-in
  per-gene filters. (#55)
- README gains a **Performance** section (4-engine DE-stage comparison incl. CPU
  pdex, a full-5.54 M-cell cell line 2 "RAM by data layout" table, and a concrete
  streaming result), cell line 2-based usage examples, and reconciled install pins
  (gpudge `@v0.3.0`; the `[streaming]` extra's shardad pinned to the commit
  carrying the target-aware reader API, which postdates the shardad v0.2.0 tag).

## [0.2.0] — 2026-06-01

### Changed (BREAKING)
- `de()` filtering is now **opt-in**. `min_feature_filter` is **removed** and
  replaced by explicit, unit-named, AND-combinable per-gene filters:
  `filter_gene_min_mean_value`, `filter_gene_min_total_value` (operate on
  `adata.X` as supplied), `filter_gene_min_cpm_cell`, `filter_gene_min_cpm_bulk`
  (assume raw counts), plus a `keep_genes` boolean-mask escape hatch. All default
  `None` (no filtering). Passing `min_feature_filter=` now raises `ValueError`
  with migration guidance: `=v` under `cpm_normalize=False` -> `filter_gene_min_mean_value=v`;
  under `cpm_normalize=True` -> `filter_gene_min_cpm_cell=v`.
- A negative threshold = keep-all for that filter; `0.0` keeps only
  strictly-positive genes. (Closes #26.)

### Added

- `de(oom_recovery=True)` (default): on a CUDA OOM while processing a
  gene-chunk, `gpu_gene_chunk_size` is halved (down to a floor of 64) and
  retried, logging each downshift — for both `auto` and explicit chunk sizes.
  Pass `oom_recovery=False` for strict mode (the first OOM raises; an explicit
  chunk is honored exactly). Wraps both the literal-reference and `ALL_OTHERS`
  (one-vs-rest) paths. (#22, #28)

### Changed

- `auto` `gpu_gene_chunk_size` budget fraction raised 0.18 → 0.20 **for the
  literal-reference path** — lands ~4096 on A100-40GB/cell line 2, the bench-measured
  throughput knee. Measured `de()` speedup (result-invariant): **~7% on
  A100-40GB** (GCP sweep `20260530T012835Z`, where 0.18 under-sizes to ~3712)
  and **~3% on H100-80GB** (slurm: 47.6s → 46.2s, chunk 8256 → 9152); neutral /
  no regression on larger GPUs (the curve is flat past the knee). Still scales
  inversely with the reference-pool size. The `ALL_OTHERS` path now uses 0.20
  as well, once #28 gave it the same OOM-recovery backstop. (#22, #28)
- `de()` now validates `mean_calc`, `epsilon`, and `output_columns` at entry,
  before any GPU work: an unknown `mean_calc` (previously silently treated as
  arithmetic), a negative `epsilon`, and unknown or duplicate-destination
  `output_columns` keys now raise immediately. The legacy `"all_others"`
  deprecation message no longer names a specific removal version (it wrongly
  said v0.1.0).

### Performance

- Vectorized the Mann–Whitney tie-correction term and trimmed per-chunk GPU
  transients in both the literal-reference and `ALL_OTHERS` paths; output is
  bit-identical (result-invariant). (#30)
- A `filter_gene_min_cpm_bulk`-only run no longer allocates the per-cell
  `row_scales` tensor or the per-group row-index GPU tensors it never uses —
  pooled-bulk CPM is derived from per-group library totals, so those GPU
  allocations are skipped. (#32, #33)

### Packaging

- `shardad` is now an **optional** dependency — the `[streaming]` extra, for the
  `de(shard_archive=…)` path — not a hard requirement. The core `de()` takes an
  in-memory `AnnData` and never imports `shardad`, so a default install no longer
  pulls the private git dependency.
- Install instructions for **pip** (git tag + cu126 torch index +
  `[fast]`/`[dev]`/`[streaming]` extras), **uv**, and **conda/mamba** added to the
  README; new `environment.yml`. Not yet on PyPI — install from the `v0.2.0` git
  tag.

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
  with a `DeprecationWarning`. The deprecation message no longer
  names a specific version; that removal is **deferred** to a
  later release. New code should pass the `ALL_OTHERS` constant
  (or the string `"__all_others__"`) instead.
- shardad v2 archive loading is roughly 16× slower than v1 at cell line 2
  scale (237s vs 17s). Stay on v1 archives for cell line 2-scale inputs
  until shardad v0.3 lands an on-GPU read path.

[Unreleased]: https://github.com/ArcInstitute/gpudge/compare/v0.3.1...HEAD
[0.3.1]: https://github.com/ArcInstitute/gpudge/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/ArcInstitute/gpudge/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/ArcInstitute/gpudge/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/ArcInstitute/gpudge/releases/tag/v0.1.0
