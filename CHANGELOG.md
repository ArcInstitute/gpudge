# Changelog

All notable changes to gpudge are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.8.0] — 2026-08-18

### Added

- **A runnable quickstart** — `docs/tutorial.md` and `examples/quickstart.py`, over a
  committed 4.99 MB subset of the Virtual Cell Challenge 2025 H1 training data
  (`docs/data/`, CC0 1.0, with provenance and a sha256 recorded in `docs/data/README.md`
  and gated by a test). It runs on real cells straight from a clone instead of a
  synthetic fixture; `docs/make_tutorial_data.py` regenerates the subset.

  The numbers the tutorial publishes are **gated, not transcribed**.
  `tests/test_tutorial.py` recomputes them from an independent SciPy CPU oracle —
  deliberately not written with gpudge, because an oracle that shares the
  implementation cannot catch the implementation drifting — and fails if the prose, the
  example's constants and `de()` ever disagree. Its own NaN-p-value branch, which the
  committed subset never reaches, is driven by injected p-values rather than left
  unexecuted. `examples/` joins the CI lint path.

### Fixed

Eight defects from the 2026-08 whole-codebase ultrareview (8 subsystem reviewers
filed 41 findings; 39 were put through adversarial verifiers executing on an
H100, and 3 were refuted and are not listed here).

- ⚠️ **`de()` p-values are chunk-invariant up to a stated bound, and this MAY
  CHANGE output** by an amount *observed* to be 1–2 ULP on `p_value` / `p_adj`
  (not a guaranteed bound). The tie-correction term is an exact integer wherever
  the tie axis fits int64 — see the fallback note below for where it is not, and
  note that "exact" describes the tie sum; the p-value remains a float64 function
  of it. Whether a given gene moves depends on its whole tie
  profile, not on any single run: the *aggregate* Σ(t³−t) has to exceed 2⁵³, and
  neither a threshold on the largest run nor one on the cell count predicts it.
  Measured both ways — a single 420 000-cell run is unchanged, while four runs of
  195 020 / 188 824 / 195 246 / 116 410 (every one of them *below* that scale)
  shift by 4. **Raw counts are affected**, routinely, through the zero run. (The
  *chunk-dependence* itself was narrower — it needed the block's max run count to
  swing across ~1e5, so it was reachable on the normalized paths and measured at
  0/64 rows on raw counts. Two different scopes; do not conflate them.)
  `_mwu._tie_term_per_gene` accumulated Σ(t³−t) in
  float64 into a `(block_height, n_runs)` buffer whose **both** dimensions are
  per-block quantities; torch's row sum is shape-sensitive, so past 2⁵³ a gene's
  tie term depended on which genes shared its chunk, and two `de()` calls
  differing only in `gpu_gene_chunk_size` returned bit-differing p-values —
  falsifying the invariant `tests/test_api.py` and
  `tests/test_taustar_integration.py` assert. Now accumulated in **int64**, exact
  for a tie axis ≤ 2 097 151 and order-free, with a float64 fallback above that.
  `_rank_with_ties` (the ALL_OTHERS tie term) had the same coupling through
  `max_tgroups` and is fixed the same way. Fixing only the zero-pad *width* does
  not work — block height still varies. The cross-tie decomposition
  (`ref_tie_term + Σ[f(rc+gc) − f(rc)]`) is now summed in int64 as well and cast
  **once**: leaving the reference term exact while the delta subtracted a float64
  `f(rc)` broke the cancellation and left a full pooled ULP of error — measured
  at −16 on 7.2e16 for a 416 134-cell zero run with one tied target cell, and
  pinned by `test_pooled_tie_term_cancels_exactly_at_a_large_reference_run`.
  Above the int64 bound the reduction falls back to float64 and chunk-invariance
  no longer holds. There are two routes past the bound — the reference tie axis
  itself (`n_ref`, or all cells under `ALL_OTHERS`) and the pooled `n_ref + m` —
  and **both are documented rather than warned about**: `de()`'s
  `gpu_gene_chunk_size` docstring states the condition and the remedy (pin a chunk
  that fits **and** pass `oom_recovery=False` — that combination fixes the
  reduction shape, which is what makes the inexact float64 result reproducible; a
  pinned chunk alone is not enough, because the default `oom_recovery=True` halves
  even an explicit chunk on OOM and can do so mid-run). A runtime warning was tried and dropped: it cannot be
  made both per-invocation and free inside the per-(group, chunk) inner loop, and
  it guarded a condition that needs a >2e6-cell tie axis and never changed a
  computed value.
- **`de(adata=<AnnData view>, …)` no longer silently loses the CSC→CSR
  coercion.** Assigning to a view's `.X` writes through to the parent instead of
  rebinding, so the coercion vanished while its warning claimed success —
  dropping every gather tile onto scipy slicing, the regression
  [#66](https://github.com/ArcInstitute/gpudge_arc/issues/66) added it to
  prevent. On an already-CSR view the assignment also ran a full
  O(n_obs × n_var) sparse scatter into the caller's matrix. The coerced matrix is
  now bound to a local **for views only**. A materialized `AnnData` is still
  coerced in place, unchanged from 0.7.0: that is deliberate and load-bearing for
  memory — it drops the caller's CSC/COO refcount so only one sparse encoding is
  resident, and holding a local copy instead would keep both live for the whole
  run and could turn a large supported CSC input into a host OOM. The assignment
  now also happens **only when `ensure_csr` actually returned a different
  object**, so an already-CSR input no longer pays a pointless
  O(n_obs × n_var) sparse scatter.
- **`densify_input=True` now raises on a *sparse* AnnData view** (a view whose
  `.X` is already dense has nothing to densify and is untouched) instead of paying the
  full dense allocation (~310 GB peak at CCL_2 scale) and discarding it, having
  already warned that the caller's matrix was mutated. Pass `adata.copy()`.
- **Cell- and shard-layout streaming reject unassigned cells.** A cell with a
  NaN/None group label reaches an archive as a group literally named `'nan'`,
  which the in-memory path has always refused; streaming emitted it as a target
  group with no warning at all. Both backends now screen the group table through
  one shared policy (`_ingest.reject_missing_group_labels`), which also covers
  the shard layout's target side — its reference side was already guarded.
- **The streaming auto gene-chunk sizer budgets the Phase-1 target tile
  unconditionally**, not only when `lfc_threshold` or `tau_star` is active. A
  target-dominated workload previously got a chunk sized for the reference sort
  alone and paid an OOM downshift, or an outright OOM under
  `oom_recovery=False`. This also makes it agree with
  `_auto_gene_chunk_size_inmem`. Callers that cannot know the tile height still
  pass `max_group_rows=0` and are unaffected.
- **The reference-residency guard trims the caching allocators before reading
  free VRAM.** It is a hard guard, and it was refusing runs with "use a
  larger-memory GPU" for references that fit, because torch/cupy still held
  cached-but-unused blocks from an earlier `de()` in the same process. The two
  *soft sizer* read sites are deliberately left alone — deferring the reclaim
  there is a documented decision in the #76 design.
- **`_filter.x_has_noncount_signal` no longer copies the whole dense matrix** to
  sample ≤100 000 values. `np.ravel` defaults to `order='C'` and so copied any
  non-C-contiguous input in full (~40 GB for a 500k × 20k f32 group); it now uses
  `order='K'` — a view for both C and F layouts — and `unravel_index` sampling
  for genuinely strided input.
- **Byte-identity gates now actually check byte-identity.** Every
  `assert_frame_equal` in the suite passes `check_exact=True`: polars defaults to
  `check_exact=False` with `rel_tol=1e-5`, which had silently turned the
  in-memory external-reference **merge gate**
  (`tests/test_inmem_external_ref_gpu.py`, 8 assertions) and three other identity
  claims into tolerance checks that a 1-float32-ULP relative drift passed at
  every magnitude. Third occurrence of this trap in the repo. All of them pass
  exactly on an H100, so the paths were identical — only the gates were blind.

Five more from the same review's LOW tier (verified but unfixed at the time),
plus the test-quality gaps that let them through. All degenerate-input or
diagnostic fixes — **no change to any computed value on a well-formed input**,
measured rather than asserted: 13 `de()` configurations (base; cpm; median;
epsilon=0 arithmetic and geometric; ALL_OTHERS ±cpm; the four-filter set; an lfc
grid; a tau* grid with SE; a pinned chunk size; and both external-reference
modes ±cpm) hash byte-for-byte identically against `main` on an H100.

- **`cpm_normalize` must now be a real boolean.** It was tested for bare
  truthiness, so `cpm_normalize='false'` (or `'False'`, `'no'`, `'0'`, or any
  non-zero number) silently turned CPM normalization ON — byte-identical to
  `cpm_normalize=True` and materially different from the `False` it reads as
  (measured on a 300×40 fixture: `max|Δ log2FC| = 0.105`, `max|Δ p| = 0.275`
  against the real `False`). Now raises, in both directions, matching the
  existing `tau_star_se` guard verbatim — including its rationale, that
  `tau_star_se='false'` would otherwise mean True. Validated in `de()`'s
  fail-fast block (above the CUDA probe, so a typo costs nothing) and again in
  the shared `resolve_target_sum`, which direct callers reach without passing
  through `de()`.
- **A reference-only input returns the typed empty frame** instead of running
  the whole GPU pass and then dying on
  `IndexError: arrays used as indices must be of integer (or boolean) type`.
  `ingest` permits a single group that IS the reference — it checks the groupby
  column, NaN labels and reference membership, not "at least one target" — and
  the untyped `np.array([])` the post-loop assembly built is float64. The index
  array now carries `dtype=np.intp`, and the tail then produces a frame
  schema-identical to `empty_output_frame()`, `lfc` and `tau*` extras included
  (measured, not assumed). All four target-enumeration paths answer this
  degenerate object the same way. `de(cell_source=..., targets=[])` still
  raises — an explicitly empty target list is a caller error, not a degenerate
  object. The full GPU pass still runs for such an input: an early return was
  written and then dropped, because it diverged `densify_input`'s documented
  in-place contract and bypassed late parameter validation, for a saving nobody
  collects on a degenerate object (codex review).
- **A zero-gene `adata` returns the typed empty frame** instead of
  `ValueError: initial_chunk must be > 0, got 0`, which named an internal
  parameter the caller never passed. Both auto chunk sizers are floored at 1;
  the floor is reachable only at `n_genes == 0`, since both produce >= 16 before
  the `min(chunk, n_genes)` clamp. The same input WITH a pinned
  `gpu_gene_chunk_size` already returned a correct 0-row frame, so this makes
  the auto path agree with the pinned one it contradicted. Fixes the
  literal-reference, in-memory external-reference and `ALL_OTHERS` paths at once.
- **The documented `epsilon=0` path is silent.** It is a supported input whose
  NaN / ±inf outcomes the `de()` docstring and README both promise, but the
  divide emitted three unsuppressed numpy RuntimeWarnings and *raised* under
  `-W error::RuntimeWarning` (which took four of the repo's own tests with it).
  Both log2FC sites now go through a shared `_output.log2_ratio`, whose
  suppression is **conditional**: it applies only when `epsilon == 0` and both
  means are finite and non-negative. Each clause earns its place, and the
  blanket `np.errstate` this change first used *would have* hidden all three —
  gpudge accepts arbitrary X, so a centered or
  log-transformed input can carry a negative mean, a pathological one an
  infinite mean (`inf/inf`), and even with a positive epsilon the *quotient* can
  underflow to zero and make `log2(0)` warn. Three rounds of codex review; the
  underflow case in particular refuted the reasoning that had the `epsilon`
  clause down as a mere fast path. It is that too: the default `epsilon=1e-9`
  now pays no scan at all (~87 ms at CCL_2 shape).
- **`de(archive=..., reference=...)` validates against the whole pool.** Both
  backends gather the archive's reference pool WHOLE, so `reference=` can only
  NAME it, never subset it, and two inputs went wrong. A sequence was
  stringified by the membership test, so a list that exactly matched the
  archive's labels produced the self-contradictory `reference=['ntc', 'safe'] is
  not among the archive's reference labels ['ntc', 'safe']`; it is now accepted.
  Naming ONE label of a multi-label pool stays legal and still uses the pool
  whole — that is specified ("`reference=<label>` is validated for membership in
  the reference labels and otherwise does not subset", 2026-07-31 cell-layout
  design), and a round of this change that "fixed" it was reverted. Shared
  between both layouts as `_stream_backend.validate_archive_reference`.
  `reference=None` is unaffected; a `np.str_` no longer leaks into the reported
  label, a 0-d `np.array('ntc')` no longer raises `TypeError` on iteration,
  numpy byte-string elements decode like a scalar `bytes`, a generator is
  materialized before being read twice, and an empty sequence gets its own
  message.

- **`gpu_gene_chunk_size` is validated at entry.** It was unchecked, so a `0`
  or a negative reached `run_gene_chunks_with_recovery` and raised
  `initial_chunk must be > 0` — an internal parameter name no caller passes.
  Now a clear `ValueError` naming the public parameter, before any GPU work.
  `True` is rejected explicitly (`bool` is an `Integral`).

### Tests

Six release-gate tests could not fail, or could not fail at the magnitude that
matters. Each fix below was verified by breaking the code it guards and watching
it go red.

- **`test_cpm_normalize_matches_external`'s p-value assertion no longer
  vanishes.** It sat inside `if finite.sum() > 10:`, so the test PASSED with
  all-NaN p-values, all-inf p-values, ten finite anti-correlated values, and a
  0-row frame. Now `assert`, matching the two siblings 50 lines down in the same
  file that already used the unconditional form.
- **`test_zero_denominator_cpm_bulk_all_others_empty_rest` asserts something.**
  Its whole body was `assert out is not None`, and `de()` has no `return None`
  on any path. The zero-library-total guards it exists to protect are
  *provably* invisible to the keep mask — a guarded 0 and an unguarded NaN
  compare identically under `_one_filter_mask`'s strict `>` for every threshold
  >= 0, and a threshold < 0 short-circuits to all-True — so what is pinned is
  the absence of the 0/0 RuntimeWarning, plus the frame's real contents. Four
  **CPU-runnable** pins for the same two guards were added to
  `tests/test_filter.py`; CPU CI covered none of this before.
- **The documented `epsilon=0` degenerate log2FC is pinned.** README and the
  `de()` docstring both promise NaN for a both-zero gene and ±inf for a
  one-sided one, "matching pdex". Nothing asserted it: replacing the whole
  contract with `0`/`±30` left the entire suite green when the review measured
  it (737 cases at that commit).
- **`group_means` is validated on CPU.** Its only oracle tests were `@needs_cuda`
  with a hard-coded `.cuda()` though the function is device-agnostic, so CPU CI
  validated no part of `target_mean` / `ref_mean` / `log2_fold_change`. Now
  parametrized over both devices, with an empty-group case and **float64**
  oracles — reducing the oracle in float32, as it did, made a
  float32-accumulation regression invisible on every device. All four mutations
  (segment reduction, mean division, empty-group guard, float32 accumulation)
  now fail on CPU.
- **The normalization non-mutation contract is asserted.** `de()` documents in
  four places that `cpm_normalize` / `normalize_target_sum` do not touch the
  caller's `X` — unlike `densify_input` and the CSC→CSR coercion, which say the
  opposite and each have an explicit assertion. The one variant with no indirect
  detector anywhere is the classic literal-reference in-memory path, which is
  what the new test covers.
- **Oracle tolerances brought within range of the measured agreement.** The
  end-to-end scipy p-value check used a `1e-3` ABSOLUTE bound (~4 orders looser
  than the measured 6e-8 / 4e-7 residual, and enough to hide a z-shift of order
  1e-3 at p~0.5) on 5 sampled rows of 250 — now all 250 rows at
  `rtol=1e-5, atol=1e-7`. `test_mwu.py`'s U check allowed `atol=0.5`, exactly
  one U lattice step and therefore the smallest representable U regression,
  against a measured agreement of 0.0 — now `1e-6`; its p check moved from
  `rtol=1e-3` (inside 15% of the 1.15e-3 a tie-divisor regression produces) to
  `1e-5`. The real-data CPU-pdex baseline rested on Pearson `r > 0.999` alone,
  which is exactly invariant to any affine transform: a log2FC max-abs
  difference of 1.351 passed, as did losing 99.9% of the joined rows. It now
  asserts join coverage in both directions and absolute value bounds
  (log2FC 1e-3, p_value 1e-9, both >= 10x the measured 5.2e-5 / 1.6e-15 on
  22,049 rows of CCL_1 chunk_0000).

## [0.7.0] — 2026-08-02

### Added

- **`de(cell_source=…)` — a public bring-your-own cell source**
  ([#86](https://github.com/ArcInstitute/gpudge_arc/issues/86)). A third input
  mode, alongside `adata=` and `archive=`, taking a callable that yields one
  public `CellGroup(label, X, rows=None)` per target group plus `targets=` /
  `var_names=`. It runs the same `refpool_de_core` as the other
  external-reference paths, so output is byte-identical — pinned by a GPU
  parity gate against `de(adata=, reference=<AnnData>)` — and every statistical
  and output-shaping `de()` feature works in it unchanged. What does not carry
  over is the transport: `normalize_target_sum='median'` raises (see below),
  and `densify_input`, `stream_n_workers` and `stream_prefetch` are **ignored**
  — they describe fetching that gpudge is no longer doing in this mode.
  Consumers that were driving the private `_refpool.refpool_de_core` (plus
  `_csr_dense.ensure_csr` / `csr_row_sums`) can now do so through supported
  API. That private entry stays **backward compatible** and its signature is
  now pinned by a test — but it is not frozen: it took three new keyword-only
  parameters this cycle for `tau_star`/`tau_star_se` (21 → 24), all optional,
  so an existing call site keeps working untouched.

  The public contract closes three silent-divergence modes the internal tuple
  carries: it takes the target's **label** rather than a positional index,
  gpudge always computes library sizes itself (there is deliberately no way to
  supply them — a caller-computed one that disagrees silently shifts every CPM
  scale), and a target the source never yields now raises instead of emitting a
  plausible `U=0`/`p=1` row. `rows` is validated for integer dtype, range and
  uniqueness rather than cast. On the same principle, a *dense* `X` that is not
  C-contiguous **and aligned** is rejected when `rows` re-orders or subsets it
  and library sizes are being computed: gpudge sums those over the row slice,
  and numpy reduces a Fortran-ordered, strided or unaligned array in a different
  order — a float64 ULP there becomes a float32 CPM-scale ULP, which moves
  float32 ties. (`np.ascontiguousarray` is not a sufficient remedy for the
  unaligned case; the error names `np.require(X, requirements=['C','A'])`.)
  `rows=None` is accepted for any layout, because it sums `X` itself exactly as
  `de(adata=, reference=)` does. The byte-identity guarantee is scoped to a
  target matrix that is CSR, or C-contiguous and aligned dense, with standard
  NumPy/SciPy semantics (an ndarray subclass that redefines `sum`/`__getitem__`,
  or an object dtype, is out of contract — no flags check can catch it, since
  `np.require` preserves the subclass): gpudge sums the matrix it is handed and
  never sees where the caller gathered it from.

  `normalize_target_sum='median'` raises `NotImplementedError` in this mode —
  it needs a row-sums pre-pass — but the contract already permits the second
  pass, so adding it later is not a breaking change. The automatic gene-chunk
  sizer cannot model the target working set here (no source can report its
  largest group without being drained), so pin `gpu_gene_chunk_size=` for large
  groups.

- `de(tau_star_se=True)` — a standard error for the `tau*` rank-shift point
  estimate. Emits `tau*_lo_p0.025`, `tau*_hi_p0.025` (endpoints of the
  **nominal** 95% normal-approximation rank interval, each a one-sided
  inversion at `p_dir = 0.025`) and `tau*_se = (hi - lo) / (2 * Phi^-1(0.975))`,
  and forces `0.5` into the level set so `tau*_p0.5` is always in the default
  output alongside its SE. Costs two extra bisections for the endpoints
  (~150 s at 1.27M cells), plus one more when `0.5` is auto-inserted. Every
  column present without the flag is bit-identical with it, `tau*_p<q>`
  included. `tau*_se` is `+inf` whenever either endpoint is unbounded, which
  zeros make common on small groups — treat it as zero weight, not as a row to
  drop. Closes #112 by quantifying the raw-counts domain caveat on both
  `tau_star` and `tau_star_se`.

- **`de(archive=…)` reads `layout='cell'` shardad archives**
  ([#110](https://github.com/ArcInstitute/gpudge_arc/issues/110)): streaming DE
  now works on `.csad` per-cell archives, the form several VCI production
  datasets are written in, removing the need for a shard-layout twin of a 40+ GB
  archive. `de(shard_archive=…)` is the deprecated spelling of the new
  layout-neutral `de(archive=…)`; dispatch is on the archive's manifest, not its
  file extension. Internally `stream_de` now drives a small backend seam
  (`_stream_backend.py`), with the shard driver moved into `_ShardBackend`
  with every output-affecting expression unchanged — the shard path's output is
  byte-identical, pinned by an H100 gate run (579 passed / 0 failed) and a
  cross-layout `check_exact=True` gate.
  Cell layout takes the **host** CSR decode path: `stream_n_workers` becomes the
  Rust gather's `n_threads` (no per-worker host RAM, unlike shard layout's
  ~14 GB each) and `stream_prefetch` has no effect. Cross-layout byte-identity
  is scoped to archives written with **default ordering** (shardad #252).
  Requires **shardad ≥ 0.7.1** with its `[cell]` extra (was pinned to v0.5.6,
  which predates the format).

- `de(tau_star=(0.5, 0.05))` — per-(target, gene) **signed** log2 shift at which
  the gene crosses a one-sided `p_dir` level, emitted as `tau*_p0.5` /
  `tau*_p0.05`. `q = 0.5` is the Hodges–Lehmann shift (closes the effect-size
  gap in #87); smaller `q` gives the one-sided confidence bound, i.e. the
  largest effect-size floor the gene survives. Computed by a deterministic GPU
  bisection on an **exact** counting oracle — `U1(delta)` is the upper-tail
  count of the implicit pairwise log-ratio matrix, so the true crossing is an
  order statistic of that matrix, and each probe is one exact `_bounds` rank
  count (two `searchsorted`) against the resident `sorted_ref`; no pairwise
  matrix is materialised. Two different things are in play and only one of them
  is exact: every *probe* is an exact rank count, but the *search* is not
  closed-form — it is a fixed `tau_star_iters` bisection (default 20), and the
  **reported** value is the midpoint of the final bracket, an approximation to
  the crossing that bracket contains rather than the crossing itself. Base
  columns are bit-identical with and without it.

### Changed

- Bounded the scanpy CSR-vs-dense median caveat documented in 0.6.2 at the
  upstream fix; **no behaviour change**. scverse/scanpy#4256 — filed from here
  as scverse/scanpy#4251, merged to scanpy `main` 2026-07-27, backported to the
  1.12.x branch as scverse/scanpy#4259 and targeted at 1.12.4 — moves scanpy's
  CSR branch onto `_compute_nnz_median`, the positive-cell median gpudge
  already implements. **No scanpy release carries the fix as of 1.12.3.**
  Affected: 1.11.2 through 1.12.3, plus the 1.13.0a1 prerelease; versions
  before 1.11.2 never had the split. gpudge's rule is unchanged — it was the
  one upstream adopted.

  Stated as a range rather than as `< 1.12.4` because that bound is wrong at
  both ends: PEP 440 orders the 1.13.0a1 prerelease *above* 1.12.4 — and it was
  cut three days before the fix merged, so the bound would mark an affected
  prerelease fixed — while also sweeping in the pre-1.11.2 versions that never
  had the split at all.

  The canary in `tests/test_scanpy_median_contract.py` now pins *which way* the
  two branches agree before skipping: agreement on the positive-cell median
  matches gpudge, agreement on the all-cell median would break the parity claim
  on both paths instead of one, and only the first may skip. Both CI cells
  currently resolve an affected scanpy (1.11.5 on py3.11, 1.12.x on py3.12), so
  the divergence assertions keep running; the py3.11 cell will keep doing so
  unless the fix is also backported to 1.11.x, since it caps at 1.11.5.

### Docs

- **The raw-counts `tau*` caveat moved to where a `tau_star` user reads it**;
  **no behaviour change** (#87). Its quantified form — 87.9% of finite `tau*`
  within 1e-5 of zero on a 1.27 M-cell production archive, only 6.0% of rows
  reaching `|tau*| >= 0.01`, and `normalize_target_sum` lifting that to 46.8%
  — was stated only under `tau_star_se`, so a caller who set `tau_star=`
  without the SE flag read a bare cross-reference and never the number or the
  instruction. The atom is a property of `tau*` itself and the SE only
  inherits it, so the measurement now sits under `tau_star`; `tau_star_se`
  keeps the part that is genuinely sharper for the SE (the same atom, not
  sampling variability, dominates `hi - lo`) and points back. The README's
  `tau_star` example, which showed a call with no normalization at all, now
  passes `normalize_target_sum="median"` and states why.

- **`de()` now documents that expression is staged in float32**, and what that
  costs; **no behaviour change** (#115). Every densify path emits float32
  regardless of `adata.X`'s dtype — a float64 input is downcast — while the
  reductions accumulate in float64, so the quantization loss sits entirely at
  the staging step and is invisible from the returned schema, in which every
  statistic is `Float64`. The new **Numerical precision** note in `Notes`
  states both consequences: `log2_fold_change` carries an *absolute* error
  floor on the order of one float32 ulp (measured max `9.8e-08`, median
  `3.0e-08`) that is independent of the fold change's magnitude, because a
  float32-relative error in a group mean becomes an absolute error in its log;
  and `Ueffect`/`p_value`/`p_adj` inherit float32 *tie* behaviour, which that
  argument does **not** bound — quantization can create or destroy a tie, so a
  gene can cross a significance threshold numerically and a downstream top-K
  that filters on `p_adj` before ranking is not stable to float32 granularity.
  The existing float32-tie sentence under `normalize_target_sum` was scoped to
  row scales and sat inside a parameter a caller may never use; it now
  cross-references the general note.

### Fixed

- **`environment.yml`'s streaming note pointed at a shardad that cannot open a
  cell-layout archive.** Its comment block still read `shardad>=0.5.6` and
  pinned the `v0.5.6` tag while claiming to match "README and pyproject's
  `[tool.uv.sources]`" — both of which moved to `shardad[cell]>=0.7.1` in this
  release cycle, v0.5.6 predating `layout='cell'` entirely. It also named the
  deprecated `de(shard_archive=…)` spelling. The file's own header asks that it
  be kept in sync with README; it now is. Same defect class as the stale
  install pins fixed in 0.6.2, in the one block that pass did not touch.

- **`csr_row_sums` reduces in float64 on the non-numba path**: the fallback was
  `X.sum(axis=1)`, which reduces in `X.data`'s dtype and only widens afterwards,
  so a float32 CSR could round where the same counts as a narrow integer dtype
  stayed exact. The numba kernel needs `numba` **and** a CSR `X`, so `[fast]`
  takes a *sparse CSR* input off this path but not a dense one; and
  `gpudge[streaming]` does not imply `[fast]` at all — while the two streaming
  layouts hand gpudge different dtypes.

- `de(lfc_threshold=np.array([0.25, 0.5]))` raised a bare `TypeError` from
  numpy instead of working (#108). `_lfc._as_seq` gated on
  `collections.abc.Sequence`, which `np.ndarray` is not, so an array of taus
  went down the *scalar* branch into `float()`. The same helper validates
  `lfc_threshold_alt`, which had the mirror-image defect:
  `de(lfc_threshold=0.5, lfc_threshold_alt=np.array(["up", "down"]))` reached
  `d not in LFC_DIRECTIONS` as a whole array and raised numpy's ambiguous-truth
  `ValueError`. (The `lfc_threshold` is load-bearing in that example:
  `normalize_lfc_spec` returns early when it is `None`, so the direction set is
  never validated on its own.) Both now work, as do generators and sets. Scalars
  are unchanged — `float`, `np.float64` and a 0-d array are not iterable, so
  they still form a one-element grid — and a bare string direction is still one
  direction, not its characters. The same `Sequence` defect was caught in
  `_taustar._as_seq` before `tau_star` shipped; this ports that fix.

  Underneath, a non-`Sequence` iterable is now classified by *iteration*
  rather than by scalar-like coercion, and the guard wraps `iter()` rather than
  the whole `tuple()`, so a `TypeError` raised partway through an iterator
  propagates instead of being mistaken for a scalar. Ordinary inputs normalise
  identically — a one-element `ndarray` or tensor still gives the same single
  τ — but an object that supports both readings, or an iterator that fails
  mid-flight, can now reach a different outcome than it did in 0.6.2.

  Accordingly, `lfc_threshold`, `lfc_threshold_alt` and `tau_star` are now
  annotated and documented as `Iterable`, not `Sequence` — `Sequence` was
  precisely the too-narrow test, so leaving it would have had the signature
  contradict the behaviour and a type checker reject the ndarray call.

## [0.6.2] — 2026-07-26

### Added

- `tests/test_scanpy_median_contract.py` — CPU contract tests for
  `normalize_target_sum="median"` against scanpy. Every existing test of that
  parity claim is `needs_cuda`, so neither CI cell ever executed it.

### Changed

- Documented a scanpy divergence the new tests surfaced; **no behaviour
  change**. In scanpy 1.11.5 and 1.12.1, `normalize_total(target_sum=None)`
  uses a different target-selection rule for CSR input than for dense/Dask
  input: the CSR branch medians over all cells, the dense branch over positive
  cells only. So the two *can* differ when zero-total cells are present (not
  always — row sums `[0, 10, 10, 20]` give 10 either way). gpudge implements the
  positive-cell median and its sparse paths use CSR, so a caller comparing
  gpudge against scanpy on the same object may see a different target.

  The `de()` and `resolve_target_sum` docstrings previously implied unqualified
  scanpy parity and now state the caveat. They also correct an assumption worth
  flagging: the target is a common scale only in *exact* arithmetic. gpudge
  applies row scales in float32 and treats equal values as ties, so a different
  target can create or destroy ties and move `Ueffect`, `p_value` and `p_adj`
  (`tests/test_scanpy_median_contract.py` pins a case going from p=5.5e-6 to
  p=1.0). `log2_fold_change` can be target-dependent as well — negligibly when
  both means greatly exceed `epsilon`, materially for zero or near-zero means,
  and regardless of `epsilon` under `mean_calc="geometric"`.

  gpudge's rule is kept deliberately: it matches scanpy's dense/Dask branch and
  its internal `_compute_nnz_median`, and it is the safer definition — on row
  sums `[0, 0, 10]` the all-cell median is 0, which would zero out the only
  populated cell.

### Fixed

- **Install pins now name the current tag.** Both `README.md` and
  `environment.yml` pinned `git+ssh://…@v0.3.0` — a tag that predates nearly
  every feature documented in this file, so the pin installed something quite
  unlike what the docs described. `README.md` was corrected first (closes #81)
  and bumped to `v0.6.2` here; `environment.yml` carried the same stale pin, was
  missed by that pass, and now reads `v0.6.2` too.
- The compare links at the foot of this file were repointed at the repository
  that carries the tags they name; from `v0.4.0` onward they had resolved to a
  missing ref.

### Docs

- `docs/THIRD_PARTY_LICENSES.md` re-audited against installed metadata; all 13
  declared rows verify. The audit had stated gpudge's own license as
  BSD-3-Clause — it is **MIT**, and has been since the relicense in v0.3.1.
  Also corrected: an MPL-2.0 §3.2 characterisation (source availability is
  required on any executable-form distribution, modified or not), two missing
  copyleft components inside SciPy's wheel (`libquadmath` LGPL-2.1-or-later and
  `libgfortran` GPL-3.0-or-later WITH GCC-exception-3.1, both declared in the
  wheel's legacy `License` value), a misgrouped cuDNN license, and unstated
  provenance on the SciPy/numba BSD identifiers.

## [0.6.1] — 2026-07-25

### Changed

- Lowered `requires-python` from `>=3.12` to `>=3.11`. Nothing in the library
  used 3.12-only syntax or stdlib APIs — the old floor was inherited from the
  initial scaffold, not a real constraint. Every runtime dependency, the
  `[fast]` extra (numba) and the `[streaming]` extra (shardad, itself
  `>=3.11`) resolve on 3.11, and the suite is identical there. Requested by a
  downstream consumer, which supports 3.11 and so could not declare gpudge as a
  pinned optional dependency (#95). The floor is 3.11 rather than 3.10 because
  shardad requires `>=3.11`.
- Relaxed the `dev` extra's `scanpy>=1.12` to `scanpy>=1.11`. scanpy 1.12
  itself requires Python `>=3.12`, so the old floor made `[dev]` unresolvable
  on 3.11. scanpy is non-runtime — it is used by the tests and the
  `benchmarks/` scripts, never by `gpudge` itself.

### Added

- CI now runs lint + tests under a Python 3.11 / 3.12 matrix. The matrix job is
  `pytest`; an aggregating `test` job republishes its verdict so the status
  check required by branch protection keeps its name.
- `tests/test_python_floor.py` — fails if the `requires-python` floor and the
  CI matrix's lowest cell ever drift apart.
- `pyyaml` added to the `dev` extra (the new guard test parses the CI
  workflow; PyYAML is not a guaranteed transitive dependency here).

## [0.6.0] — 2026-07-25

### Changed

- **Breaking** — replaced the raw Mann–Whitney `U` statistic columns with
  `Ueffect = 2A − 1`, the signed rank-biserial correlation (Cliff's delta) in
  [−1, 1]. Directional columns encode direction as the sign of τ —
  `tau=<±τ>_{p,Ueffect,padj}` (`+` = up, `-` = down) — replacing the previous
  `_up`/`_down` suffix.

## [0.5.0] — 2026-07-24

### Changed

- **Breaking:** renamed the base statistic column `test_statistic` → `U`, and
  the directional `lfc_threshold` columns from
  `{p_value,test_statistic,p_adj}__log2fc<τ>__<dir>` to
  `tau=<τ>_{p,U,padj}_<dir>`.

## [0.4.0] — 2026-07-24

### Added

- `de(lfc_threshold=…, lfc_threshold_alt=…)`: an effect-size floor applied at
  the rank level. Reports one-sided MWU p-values against two separate composite
  nulls — `H0: log2FC ≤ +τ` (`up`) and `H0: log2FC ≥ −τ` (`down`) — instead of
  only the point null `log2FC = 0`. (Not `H0: |log2FC| ≤ τ`; that two-sided
  form is deliberately not offered, see the docstring.) The rank-based analogue
  of DESeq2's `lfcThreshold`. A post-hoc `|log2FC|` filter cannot do this: it
  applies FDR to the τ=0 null and then filters, so `p_adj` answers the wrong
  question. Accepts a single τ or a **grid** evaluated in one pass (ingest,
  densify, H2D, the reference sort and `ref_tie_term`, the target sort, and the
  means are all shared across the grid). The two-sided columns are always
  emitted, unchanged. New columns are
  `{p_value,test_statistic,p_adj}__log2fc<τ>__{up,down}`, each (τ, direction)
  its own BH family. Not supported with `reference=ALL_OTHERS`.

- Streaming DE (`de(shard_archive=…)`) now decodes each target shard on the **GPU**
  via shardad `GroupShard.x_cupy()` (device-resident cupy CSR) and densifies
  on-device, replacing the host numba densify + per-gene-chunk H2D. Default on
  x_cupy-capable archives (the v0.5.x single-file **packed** container, plus legacy
  v2-directory) when cupy + shardad ≥ 0.5.5 are installed (`gpudge[streaming-gpu]`),
  with automatic fallback to the CPU-parallel prefetch path (v1 archive, no cupy, or
  older shardad). Output is **byte-identical** to the host decode path (GPU parity
  test is the merge gate); `stream_n_workers` / `stream_prefetch` apply to the host
  fallback only. (#69)

- In-memory `de(adata=<targets>, groupby=…, reference=<AnnData control pool>)`
  now accepts a **separate control AnnData** as the reference, ranked
  resident-sorted on GPU with **no target∪reference concatenation** — previously
  only the streaming path (`shard_archive=`) accepted an AnnData reference. This
  removes the forced `concat(targets, control)` that drove a host OOM at scale
  (anndata/scipy vstack promotes indices to int64 once combined `nnz > 2^31`).
  Implemented via a shared reference-pool core (`_refpool.refpool_de_core`) that
  both the streaming and in-memory paths run, so in-memory external-ref output is
  **bit-identical** to the streaming Mode-2 path on the same cells + reference
  (GPU parity test is the merge gate). Full option parity (`cpm_normalize` /
  `normalize_target_sum` incl. `"median"` / all `filter_gene_*` / `mean_calc`).
  `groupby` is required; `reference.var_names` must equal `adata.var_names` in
  order; `densify_input=True` with an AnnData reference raises `ValueError`.

- Streaming `de(shard_archive=…)` now decodes shards ahead in parallel via
  shardad's opt-in decode-ahead (prefetch) reader, overlapping shard decode with
  GPU compute. Two new knobs: **`stream_n_workers`** (default 16) — decode
  concurrency, the speed↔host-RAM dial — and **`stream_prefetch`** (default 2) —
  decode-ahead queue depth (`0` = serial, byte-identical, lowest host RAM).
  Applies to both the Phase-1 consume loop and the `normalize_target_sum="median"`
  pre-pass. Output is bit-identical regardless of either knob. Requires
  `shardad>=0.5.1` (pin bumped to `0667b9d`).

  Measured on the full CCL_2 archive (5.54 M cells, 68 shards, H100): streaming DE
  drops from **1331 s → 282 s (4.7×)** at the default `stream_n_workers=16`
  (median-normalized path 5.8×; its pre-pass alone 8.6×), all bit-identical to
  serial. Peak host RAM is set by the decode batch — roughly one decoded shard
  per worker, **~14 GB × `stream_n_workers`** (≈75 / 126 / 223 GB at 4 / 8 / 16);
  prefetch depth past ~2 adds RAM without speed. Lower `stream_n_workers` on
  memory-constrained nodes (n_workers=4 → ~75 GB, still ~2.8×).

### Changed

- New `gpudge[streaming-gpu]` extra (`shardad[gpu]>=0.5.6`, pulls cupy + nvcomp)
  enables GPU device decode; the base `gpudge[streaming]` extra stays CPU-friendly
  (`shardad>=0.5.6` — no CUDA wheels forced onto host-only / CPU-CI installs; it
  still device-decodes automatically if cupy happens to be importable, else the
  host CSR path). shardad source pin bumped to tag `v0.5.6` — `v0.5.5` shipped
  `x_cupy()` device decode; `v0.5.6` adds `readinto` pinned staging (#190/#193)
  that cuts the GPU decode wall (~465→319 ms/shard CCL_1, ~400→246 ms/shard CCL_2),
  byte-exact. (#69, #73)

- Streaming `de(shard_archive=…)` now reads each shard's count matrix via
  shardad's lightweight `GroupShard.x()` accessor (raw scipy CSR, no AnnData
  wrapper) instead of `GroupShard.to_anndata().X`. The consumer only ever used
  `.X` (the per-group row slices come from `gs.groups`), so building a full
  AnnData per shard just to reach `.X` was indirect; `x()` reads the matrix
  directly. `x()` is byte-identical to `to_anndata().X` for the same decode
  knobs, so **output is unchanged** — GPU-validated bit-identical on the full
  CCL_2 archive (94,425,635 rows; default and `normalize_target_sum="median"`
  paths). Requires the shardad pin bumped to `0dda675c` (post-v0.5.2; adds
  `GroupShard.x()`, shardad #181 / gpudge #177 suggestion 2).

  **This is a clarity/correctness change, not a speed change.** Contrary to the
  initial hypothesis, removing per-shard AnnData construction does **not**
  meaningfully reduce wall time on the default (prefetch) path: full-CCL_2 H100
  measured **285.4 s → 281.7 s** (default) and **423.9 s → 421.8 s** (median),
  i.e. ~1% (within noise). In prefetch mode the AnnData was already built in
  shardad's background producer thread (overlapped with GPU compute) and is cheap
  (~0.05 s/shard), so it was never on the critical path — per-shard decode
  (~16 s/shard) dominates and is unchanged by `x()`. The serial (`stream_prefetch=0`)
  low-host-RAM path, where there is no background producer and `x()` also skips a
  per-shard header read, is not benchmarked here.

### Fixed

- In-memory external-reference `de(reference=<AnnData>)` no longer strands GPU
  memory or starves its own sizer when a caller shares the process/GPU
  (`gpudge_arc#76`). `de()` now returns its GPU caches (torch's caching
  allocator and, if importable, cupy's pools) to the CUDA driver on exit via a
  new `release_gpu_memory=True` kwarg — so a same-process caller's next
  `cudaMalloc` / cuBLAS op no longer OOMs against a full-but-pooled card
  (`CUBLAS_STATUS_ALLOC_FAILED`). Note the in-mem path allocates through torch,
  so the release is `torch.cuda.empty_cache()` (not a cupy-pool trim). The in-mem
  auto-sizer now reclaims pooled memory (with `gc.collect()`) **before** it reads
  free VRAM, so a caller's stale pool (e.g. a prior cupy phase) can't shrink the
  gene-chunk into a 10–20× slowdown, and it reserves a ~1 GiB headroom floor.
  Memory-management only — DE output is unchanged (GPU bit-parity gate green).
  Pass `release_gpu_memory=False` to keep caches resident across repeated calls.

- In-memory external-reference DE (`de(adata, groupby, reference=<AnnData>)`)
  auto `gpu_gene_chunk_size` no longer over-provisions into an OOM→downshift. The
  previous auto-sizer budgeted the reference sort with `n_groups=1` and sampled
  free GPU memory **before** the ~5.7 GB resident sorted reference was allocated,
  so it picked a chunk that OOM'd on the first attempt (then halved via the #22
  recovery). A new resident-aware in-memory sizer reserves the resident reference
  footprint first and sizes on the target working set, so it no longer relies on
  the OOM recovery. Bit-identical results (chunk size is result-invariant). (#72)

- In-memory `de()` now coerces a non-CSR sparse `adata.X` (and an AnnData
  `reference.X`) to canonical CSR once at entry with a one-time `UserWarning`,
  instead of silently falling back to single-threaded scipy slicing on every
  `(group × gene-chunk)`. A CSC `adata.X` (e.g. from an upstream `concat`/cache)
  previously pinned one core with the GPU idle — minutes vs hours at
  multi-million-cell scale. The literal-reference path coerces `adata.X` in place
  (matching the `densify_input` contract); the external-reference path coerces to
  internal copies and leaves the caller's AnnData untouched. The one-vs-rest
  (`ALL_OTHERS`) path is intentionally left unchanged — its all-cells densify
  never uses the numba CSR kernel, so coercing there would only add overhead
  (its parallelization is a separate follow-up). Streaming (`shard_archive=`) is
  unaffected (shardad already returns canonical CSR). (#66)

### Performance

- In-memory external-reference DE (`de(adata, groupby, reference=<AnnData>)`) now
  uploads target tiles through a **double-buffered pinned host arena with async
  H2D** and batches the device→host copy **once per group** (previously a blocking
  `.to(device)` + a `.cpu()` sync per `(group × gene-chunk)` tile, which fully
  serialized densify→H2D→compute). This closes most of the ~1.55× gap vs the
  in-adata Mode-1 label-reference path on a 5.5 M-cell dataset. Ports Mode-1's
  proven optimization into the shared reference-pool core, used only by the
  in-memory caller — the streaming path is unchanged. Output is **bit-identical**
  (GPU in-mem-vs-streaming parity test is the merge gate). (#72)
  - Follow-up: the in-memory auto `gpu_gene_chunk_size` budget was raised (cap
    16 → 32 GiB, fraction 0.35 → 0.5) so the picked chunk captures more of the
    overlap win on large GPUs — on the 5.54 M-cell CCL_2 shape (~80 GiB H100) it now
    picks ~4608 vs 2304 (~half the gene-chunks), lifting the realized speedup
    toward the synthetic ceiling. Smaller GPUs stay fraction-limited (the cap does
    not bind, so they never over-provision; the fraction bump enlarges their chunk
    somewhat too, with peak scaling to their own free memory), and chunk size is
    result-invariant. **Motivated** by the real 5.54 M-cell repro, which at the
    *old* 2304 chunk measured uploader 149.8 s vs forced-legacy 206.2 s = 1.38×
    (49,951,046 rows identical, zero OOM downshifts); the larger chunk is expected
    to lift this further but was not itself re-run at 5.54 M scale (verified via
    the sizer unit test, the GPU parity gate, and the peak-memory headroom
    argument). (#72)

- **GPU device decode (`de(shard_archive=…)`) benchmarked device-vs-host on H100
  80 GB (shardad v0.5.6) — a speed + host-RAM + stability win at scale, output
  byte-identical to the host path.** End-to-end `de(shard_archive=…)` timed with
  GPU device decode (`x_cupy`, the default on an x_cupy-capable archive) vs the
  host CPU-parallel prefetch path (defaults `stream_n_workers=16`,
  `stream_prefetch=2`), Mode 1 (archive `non-targeting` reference), best of 2 warm
  runs:
  - **CCL_2 (5.54 M cells, 68 shards):** device **173 s** vs host **272 s** —
    **~1.57×**. Device is also stable run-to-run and holds only one shard on the
    GPU, whereas the host path's ~14 GB × `stream_n_workers` (~223 GB at the
    default nw=16) decode-ahead batch can evict the page cache and force a cold
    archive re-read (an intermittent host outlier up to ~500 s was observed).
    Device decode needs no `stream_n_workers` / `stream_prefetch` tuning.
  - **CCL_1 (1.21 M cells, 14 shards):** device **66 s** vs host **69 s** —
    **~perf-neutral** (~1.03×); at this size the GPU Mann–Whitney compute floor
    (~67 s) dominates both decode paths.
  - The win **scales with cell count**: GPU decode is only ~7–10 % of the device
    wall (shardad v0.5.6's `readinto` staging trimmed ~2 s / ~10 s off the CCL_1 /
    CCL_2 device leg), so the payoff is the **on-device densify + eliminated dense
    H2D + eliminated ~223 GB host decode-ahead batch**, all of which grow with the
    data. GPU peak was identical device-vs-host (18.4 GB CCL_1 / 39.4 GB CCL_2). This
    corrects the pre-benchmark ~2.5–3× estimate. (#69, #73)

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
  (e.g. the full 5.54 M-cell CCL_2 runs in ~31 GB host), **not a speedup** — it is
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
  gpudge against scanpy (CPU), CPU pdex, and rapids-singlecell (GPU) on CCL_1 and
  CCL_2. Because gpudge runs the whole screen as one GPU pass, its DE-stage time
  is **near-constant in the perturbation count**, where per-group tools scale
  ~linearly. Headline DE-stage results (H100 80 GB):
  - **CCL_1 / 500 perturbations** (239,054 cells × 18,533 genes): gpudge
    **7.2 s** vs CPU pdex **4,285 s** (71 min, 32 workers) = **~595×**, results
    **bit-identical** (log2FC & p-value Pearson > 0.999999999 over 5.5 M
    gene–perturbation pairs); vs rapids-singlecell ~1.0e3 s, vs scanpy ~9 h.
  - rapids-singlecell VRAM grows with the perturbation count and **OOMs** at
    higher rungs (CCL_1 full, CCL_2 ≥ 200); gpudge VRAM stays flat (~15 GB CCL_1 /
    ~29 GB CCL_2). (#50, #51)

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
  pdex, a full-5.54 M-cell CCL_2 "RAM by data layout" table, and a concrete
  streaming result), CCL_2-based usage examples, and reconciled install pins
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
  literal-reference path** — lands ~4096 on A100-40GB/CCL_2, the bench-measured
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
  host RAM permits (~154 GB for CCL_2); emits a `UserWarning`.
- `output_columns` rename/select dict — configurable output schema,
  raises `KeyError` on unknown keys.
- Sparse-aware gene-chunk streaming: chunks are sliced from CSR `X`
  to a pre-pinned host dense buffer, then async H2D to a torch
  tensor — no host densify spike for CCL_2-scale inputs.
- Optional `[fast]` extra: numba-accelerated CSR row + col-range
  slicer (`_row_col_slice_np`). With it, per-group slicing is a
  single parallel CSR-gather kernel. Without it, the scipy
  two-step `X[rows, cols].toarray()` fallback is used; correctness
  is bit-identical.
- shardad v0.1 / v0.2 archive loading via the top-level
  `shardad.read_h5ad` API. v1 archives are recommended for CCL_2-scale
  inputs (v2 load is ~16× slower at that scale — see Known issues).
- Per-group BH-FDR via a fused counting-sort + per-segment kernel
  (vectorised across all groups in a single pass).
- Double-buffered pinned H2D + per-chunk GPU accumulators for
  per-group ranks; batched D2H once per gene-chunk.
- Public sentinel `ALL_OTHERS = "__all_others__"` (the legacy
  spelling `"all_others"` is still accepted — see Known issues).

### Performance

Benchmark dataset: **CCL_2 deep CRISPRi screen** — 2,064,002 cells ×
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
CCL_2 scale.

The benchmark harness is maintained separately and is not included in this release.

### Accuracy

Compared against the production CPU pdex pipeline on the same CCL_2
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
- shardad v2 archive loading is roughly 16× slower than v1 at CCL_2
  scale (237s vs 17s). Stay on v1 archives for CCL_2-scale inputs
  until shardad v0.3 lands an on-GPU read path.

<!-- Compare links resolve against this repository, which is published as
     milestone snapshots: 0.1.0, 0.2.0, 0.3.0, 0.3.1, 0.7.0 and 0.8.0 are tagged
     here. The versions between them are documented above but are not separately
     tagged here, so they carry no compare link. Issue and PR numbers throughout
     refer to the development repository. -->

[Unreleased]: https://github.com/ArcInstitute/gpudge/compare/v0.8.0...HEAD
[0.8.0]: https://github.com/ArcInstitute/gpudge/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/ArcInstitute/gpudge/compare/v0.3.1...v0.7.0
[0.3.1]: https://github.com/ArcInstitute/gpudge/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/ArcInstitute/gpudge/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/ArcInstitute/gpudge/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/ArcInstitute/gpudge/releases/tag/v0.1.0
