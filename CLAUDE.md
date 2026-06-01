# CLAUDE.md

Guidance for Claude Code sessions working in this repository.

## What this is

`gpudge` is a GPU-only Mann–Whitney U differential expression library
for single-cell CRISPR screens. One public entry point: `de()` in
`src/gpudge/__init__.py`. Designed as a slim replacement for the parts
of [pdex](https://github.com/ArcInstitute/pdex) the Arc VCI pipeline uses.

## Quick orientation

- Public API: `de()`, `ALL_OTHERS` sentinel (= `"__all_others__"`),
  `MeanCalc` literal, `__version__`. The pre-v0.1 spelling `"all_others"`
  is accepted with `DeprecationWarning`; will be removed in a future release.
- Gene-level filtering is **opt-in** (all `filter_gene_*` params default to
  `None`); `min_feature_filter` is removed. See the `de()` docstring for the
  full set of per-gene filter params.
- Internal modules (underscore-prefixed): `_mwu` (MWU stats), `_fdr` (per-group
  BH), `_means`, `_ingest`, `_stream`, `_output`, `_csr_dense` (numba CSR
  slicer; optional dep).
- Optional `[fast]` extra installs `numba` and enables the CSR kernel.

## Validated against

- CPU pdex on cell line 1 and cell line 2 datasets — bit-perfect on log2FC and p-value
  (~1e-8 numerically equivalent on p_adj when `cpm_normalize=True` is used due
  to float32 multiply ordering).
- scipy.stats.mannwhitneyu with `method='asymptotic'`, `use_continuity=True`.

## Performance reference (cell line 2: 2.06M cells × 18.5k genes × 4672 guides)

- Default scipy path: ~313 s on H100.
- `[fast]` (numba CSR kernel): **~51 s** on H100 (v0.1.0, after the T1–T5 perf
  series; the older ~108 s was the pre-series T0 baseline). ~12× faster than the
  `pdex.pdex` reference (~597 s). The CHANGELOG perf table is the source of truth.
- `densify_input=True` opt-in: ~225 s but needs ~310 GB host RAM transient.

## How to extend

- Always run the test suite (`pytest tests/`) — most tests run without a GPU.
- GPU-gated tests use the `needs_cuda` marker in `tests/conftest.py`.
- For non-trivial changes: branch + multi-agent review + PR + Gemini reviewer
  before merging to `main`. See user memory `feedback_review_procedure`.

## Where the bodies are buried

- `_csr_dense.py` emits float32 regardless of input dtype — supports uint16/
  int16/uint8 X.data from h5ad_compression-style narrow h5ads with no upfront
  cast.
- `bh_per_group` (in `_fdr.py`) sorts by group then iterates contiguous
  segments. The previous mask-scan version was O(N × n_groups) and ate 28% of
  cell line 2 wall.
- `mwu_one_group` returns early with zeros + NaN p-values when `m == 0` or
  `n_ref == 0`. Don't strip this guard without also handling the chunk-loop
  edge case.
- `densify_input=True` mutates `adata.X` in place and emits a `UserWarning`.
  Pass `adata.copy()` first if you need to preserve the sparse matrix.
