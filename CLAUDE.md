# CLAUDE.md

Guidance for Claude Code sessions working in this repository.

## What this is

`gpudge` is a GPU-only Mann–Whitney U differential expression library
for single-cell CRISPR screens. One public entry point: `de()` in
`src/gpudge/__init__.py`. Designed as a slim replacement for the parts
of [pdex](https://github.com/ArcInstitute/pdex) the Arc VCI pipeline uses.

## Quick orientation

- Public API — the whole of `__all__`: `de()`, `ALL_OTHERS` sentinel
  (= `"__all_others__"`), `CellGroup`, `MeanCalc` literal, `__version__`. The
  pre-v0.1 spelling `"all_others"` is accepted with `DeprecationWarning`; will
  be removed in a future release.
- Gene-level filtering is **opt-in** (all `filter_gene_*` params default to
  `None`); `min_feature_filter` is removed. See the `de()` docstring for the
  full set of per-gene filter params.
- `de(lfc_threshold=τ|[τ…], lfc_threshold_alt=("up","down"))` adds rank-level
  effect-size-floor tests (`_lfc.py` for validation/naming/scale factors,
  `_mwu.mwu_one_group_lfc` for the kernel). The shift is applied to the
  **target**, not the reference (`U(T, R·f) ≡ U(T/f, R)`) — so `sorted_ref` and
  `ref_tie_term` are never rebuilt and nothing scaled is held resident. Base
  two-sided columns are always emitted unchanged — that byte-identity is the
  release gate. ALL_OTHERS is unsupported.
- Internal modules (underscore-prefixed): `_mwu` (MWU stats), `_fdr` (per-group
  BH), `_means`, `_ingest`, `_stream`, `_output`, `_csr_dense` (numba CSR
  slicer; optional dep), `_refpool` (shared reference-pool DE core
  `refpool_de_core`, run by ALL THREE external-ref paths: streaming, in-memory,
  and `cell_source`).
- Optional `[fast]` extra installs `numba` and enables the CSR kernel.
- `de(adata=<targets>, groupby=, reference=<AnnData>)` accepts a **separate
  control AnnData** in-memory: ranked resident-sorted on GPU with NO
  target∪reference concat, every group a target. Bit-identical to the streaming
  Mode-2 path (same `refpool_de_core`) — the GPU parity test is the merge gate.
- `de(cell_source=<callable>, targets=, var_names=, reference=)` is the public
  **bring-your-own cell source**: the caller yields `CellGroup(label, X,
  rows=None)` per target group and gpudge never opens the payload. Same
  `refpool_de_core` as the other external-ref paths (byte-identity is the merge
  gate). `label` not an index, and no `row_sums` field at all, are deliberate —
  both were silent-divergence modes. The reference-type guard sits ABOVE the
  `ALL_OTHERS` feature guards on purpose. `normalize_target_sum='median'`
  raises; `max_group_rows=0` leaves the auto chunk sizer blind, so large groups
  want a pinned `gpu_gene_chunk_size`.
- `de(archive=<path>)` streams a target-aware shardad archive (reference resident
  on GPU); `de(shard_archive=)` is the deprecated spelling. **Both layouts**:
  `layout='shard'` (`_shard_stream.py` `_ShardBackend`, legacy — slated for
  deprecation) and `layout='cell'` (`_cell_stream.py` `_CellBackend`), behind the
  `_stream_backend.py` seam that `stream_de` drives. Two reference modes each
  (archive's own reference vs external `reference=<AnnData>`). Shard layout:
  decode-ahead prefetch via `stream_n_workers=` (default 16, ~14 GB/worker) and
  `stream_prefetch=` (default 2). Cell layout: `stream_n_workers` is the Rust
  gather's `n_threads`, `stream_prefetch` is inert, no device decode.

## Validated against

- CPU pdex on CCL_1 and CCL_2 datasets — bit-perfect on log2FC and p-value
  (~1e-8 numerically equivalent on p_adj when `cpm_normalize=True` is used due
  to float32 multiply ordering).
- scipy.stats.mannwhitneyu with `method='asymptotic'`, `use_continuity=True`.

## Performance reference (CCL_2: 2.06M cells × 18.5k genes × 4672 guides)

- Default scipy path: ~313 s on H100.
- `[fast]` (numba CSR kernel): **~51 s** on H100 (v0.1.0, after the T1–T5 perf
  series; the older ~108 s was the pre-series T0 baseline). ~12× faster than the
  `pdex.pdex` reference (~597 s). The CHANGELOG perf table is the source of truth.
- `densify_input=True` opt-in: ~225 s but needs ~310 GB host RAM transient.

## How to extend

- Always run the test suite (`pytest tests/`) — most tests run without a GPU.
- GPU-gated tests use the `needs_cuda` **`skipif`** decorator in
  `tests/conftest.py` — a `skipif`, not a registered marker, which is why
  `pytest -m needs_cuda` selects nothing and exits 5. Run the whole suite.
- **Run the suite on a CUDA host for any change touching GPU code paths.** CI is
  CPU-only, so the GPU bit-identity gates, the `lfc_threshold` GPU surface and
  the GPU-backed scanpy-parity assertions never execute there — but "CI covers
  none of that ground" is too strong: `tests/test_scanpy_median_contract.py`
  exists *precisely* because every other test of that parity claim is
  `needs_cuda`, and the kernel-level `tau_star_se` release gate in
  `tests/test_mwu_taustar.py` runs on CPU by design. CI also installs only
  `[dev,fast]`, so the **three** shardad-gated suites (`test_shard_stream.py`,
  `test_cell_stream.py`, `test_inmem_external_ref_gpu.py` — each a module-level
  `pytest.importorskip("shardad")`) are not collected at all.

  Counts, measured 2026-08-18 — do not propagate them without re-measuring;
  two upstream changes moved every figure here within one day. This tree with
  **no** GPU reports **615 passed / 200 skipped** (815 collected), and its CI
  reports **545 passed / 132 skipped** — 128 CUDA-gated cases, the 3
  module-level shardad skips, and the one real-data test. The three shardad
  suites hold **141 cases,
  70 of which need no GPU at all** (33 + 37 + 0 pass on a GPU-less host), so what
  CI misses there is not only GPU coverage. A GPU run has to hard-fail when torch
  cannot see a GPU, or when shardad/numba are absent — otherwise those tests
  silently skip, pytest exits 0, and the result is a green gate that tested
  nothing.
- For non-trivial changes: branch, get the diff reviewed, open a PR, and merge
  only once review and CI are green.
- Comments and tests cite design-spec sections by label (`spec 3.2b`,
  `spec 3.5`, `Semantics A`). Those documents are not part of the
  distribution; where a label carries reasoning, the reasoning is restated
  inline beside it.

## Where the bodies are buried

- `_cell_stream._cell_group_ranges` reads shardad's **private**
  `CellStore._load_groups()`. Deliberate — the public alternative loads the whole
  obs DataFrame to recover a table the archive already stores. It validates that
  the reference groups lead contiguously from row 0, because shardad's own
  `read_reference()` gathers `[0, max reference stop)` and would otherwise
  silently pull target cells into the reference pool.
- `csr_row_sums`'s non-numba fallback **widens a sparse `X.data` to float64
  itself** and lets scipy reduce that. Passing `dtype=` to scipy's *sparse*
  `.sum()` is **not** enough — whether it widens the accumulator or only the
  result varies by scipy version, and CI caught it reducing in float32 on 3.11
  while 3.12 happened to be fine (`_csr_dense.py` says so at the call site).
  numpy's *dense* `.sum(dtype=)` does honour the accumulator, so the dense branch
  passes it. Without the widening scipy reduces in `X.data`'s dtype: the cell
  gather hands gpudge float32, the shard reader can hand it a narrow integer
  dtype, and the two layouts then disagree on CPM scales once a library size
  passes 2**24.
- Cross-layout byte-identity holds only for archives written with **default
  ordering**. shardad v0.7.1's cell writer converts categorical
  `sort_within_group` keys with `.to_numpy()` (dropping category order) where the
  shard writer uses `.values` — shardad #252, fixed only on unreleased main.
- `_csr_dense.py` emits float32 regardless of input dtype — supports uint16/
  int16/uint8 X.data from h5ad_compression-style narrow h5ads with no upfront
  cast.
- `bh_per_group` (in `_fdr.py`) sorts by group then iterates contiguous
  segments. The previous mask-scan version was O(N × n_groups) and ate 28% of
  CCL_2 wall.
- `mwu_one_group` returns early with zeros + NaN p-values when `m == 0` or
  `n_ref == 0`. Don't strip this guard without also handling the chunk-loop
  edge case.
- `mwu_one_group_lfc` scales the TARGET, never the reference, and does it in
  **float64**. Two things not to "optimise": building a scaled `sorted_ref`
  needs `(1 + K·A)` resident `(n_genes, n_ref)` f32 matrices (~40 GB for a 5×2
  grid at CCL_2) and does not fit `refpool_de_core`'s group-outer/chunk-inner
  nesting (spec §3.2a); and dropping the `.to(torch.float64)` puts the
  comparison back on the float32 tie boundary (measured `p = 2.7e-18` vs
  `p = 0.50`) *and* breaks the invariance that lets `gc`/`run_start` be computed
  once (spec §3.2b). All mixed-dtype comparisons go through `_mwu._bounds`;
  a raw `torch.searchsorted` with an f32 boundary and f64 values upcasts and
  copies the whole reference (0.05 ms → 65.26 ms on a 160 MB reference).
- `densify_input=True` mutates a **materialized sparse** `adata.X` in place and
  emits a `UserWarning`; pass `adata.copy()` first if you need the sparse matrix
  preserved. An already-dense `X` is a no-op — nothing is mutated and nothing is
  warned. Honoured on the in-memory group-label / `ALL_OTHERS` path ONLY: a
  sparse view raises, and so do `archive=` and `adata=` + `reference=<AnnData>`.
  `cell_source=` **ignores** it — that branch returns before the reference-type
  guard, so even a `cell_source=` + AnnData-pool call does NOT raise.
