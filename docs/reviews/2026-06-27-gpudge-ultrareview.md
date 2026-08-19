# gpudge Ultrareview Report

## 1. Executive summary

**22 distinct issues confirmed** (23 findings; two — the `all_others` N==1 tie-correction NaN — are the same bug found by the *numerical* and *edge-cases* lenses and are merged into one entry below).

Severity breakdown:
- **Critical:** 0
- **High:** 1
- **Medium:** 3
- **Low:** 11
- **Nit:** 7

(The Low/Nit split read 10/8 until 2026-08-18. The findings below, and the
resolution table added the same day, are L1–L11 and N1–N7: a transcription
error in this summary, not a change to any finding. The total of 22 was always
right.)

No live correctness bug exists in any realistic, supported code path. The core MWU statistic is bit-perfect vs scipy. The findings cluster into: one genuine silent-corruption risk on real-world input, a band of test-coverage gaps around the numerical core and the v0.3.0 streaming headline, a few version/doc drifts from the release, and a long tail of degenerate-input and budgeting nits.

**Top risks (one line each):**
1. **Missing-label silent corruption** — NaN/None in the `groupby` column are bucketed into bogus `'nan'`/`'None'` groups, silently skewing every comparison; unassigned cells are common in CRISPR screens. (HIGH)
2. **Core MWU correctness is invisible to CI** — the only scipy-equivalence tests are `@needs_cuda`, so the device-agnostic statistic is never exercised on the CPU-only CI; a tie/continuity/variance regression passes green. (MEDIUM)
3. **Streaming equivalence is correlation-only** — `_assert_equiv` checks Pearson `> 0.9999999`, which is blind to any affine scale/offset bug exactly where streaming drift is most plausible. (MEDIUM)
4. **`environment.yml` ships v0.2.0** — the documented conda install path installs the previous release, so conda users miss every 0.3.0 feature. (MEDIUM)

---

## Resolution status

Added 2026-08-18. **All 22 findings were addressed in v0.3.1 ([#59](https://github.com/ArcInstitute/gpudge_arc/pull/59))**, the release immediately following this review; two of them deliberately by deciding no code should change. Statuses below were re-verified against the tree at `v0.8.0`, and the *Evidence* column names something you can open — a regression test that cites the finding ID, or the code that now carries the guard.

| # | Severity | Finding | Status | Evidence |
|---|---|---|---|---|
| H1 | High | NaN/None `groupby` labels bucketed into `'nan'`/`'None'` groups | **Fixed** (v0.3.1); guard broadened in v0.8.0 ([#124](https://github.com/ArcInstitute/gpudge_arc/pull/124)) | `_ingest.MISSING_LABEL_SPELLINGS` and the mirroring guard in `_shard_stream.py`, which was the only streaming layout when this was fixed; `_cell_stream.py` gained the same screen when the cell layout arrived in v0.7.0. ⚠️ One backend-parity gap remains open as [#127](https://github.com/ArcInstitute/gpudge_arc/issues/127) — a group *genuinely* named `nan` is rejected on streaming but accepted in memory |
| M1 | Medium | MWU-vs-scipy correctness tests all `@needs_cuda` | **Fixed** | `test_mwu.py::test_mwu_ref_matches_scipy_cpu`, which cites M1 |
| M2 | Medium | Streaming equivalence checked Pearson correlation only | **Fixed** | `test_shard_stream.py::_assert_equiv` — `allclose(rtol=1e-5, atol=1e-7, equal_nan=True)` over full row coverage, with the M2 reasoning at the call site |
| M3 | Medium | `environment.yml` pinned `gpudge @v0.2.0` | **Fixed** | pins the current release tag; kept in step with README by a note in both |
| L1 | Low | `all_others` N==1 → `tie_corr` divide-by-zero NaN | **Fixed** | `test_api.py`, docstring cites L1 |
| L2 | Low | ref-mode empty target reported `p_value=0.0` | **Fixed** | `_mwu.py` returns zeros + NaN when `m == 0 or n_ref == 0` |
| L3 | Low | `epsilon` accepted NaN and +inf | **Fixed** | `math.isfinite(epsilon)` guard; `test_de_rejects_nonfinite_epsilon` (+ the streaming-dispatch twin) |
| L4 | Low | `output_columns={}` yielded a degenerate frame | **Fixed** | `test_api.py`, docstring cites L4 |
| L5 | Low | Non-string `reference` gave an opaque error | **Fixed** | `test_api.py`, docstring cites L5 |
| L6 | Low | `all_others` chunk budget under-estimated GPU peak | **Fixed** | `_stream.py` comment cites L6 |
| L7 | Low | Streaming heuristic budgeted accumulators Phase 1 never allocates | **Fixed** (v0.3.1); model refined again in v0.8.0 | the `UNCONDITIONAL:` comment on the sizer in `__init__.py`, which states that the Phase-1 target tile is modelled whenever the sizer runs |
| L8 | Low | Phase-1 `del shard, Xs` defeated by the `_chunk` closure | **Fixed** | v0.3.1 "`/dev/shm` cleanup"; `del Xs, Ls` per shard in `_shard_stream.py` |
| L9 | Low | `log2_fold_change` never checked against a closed form | **Fixed** | `test_api.py`, docstring cites L9 |
| L10 | Low | Multi-block path of `_tie_term_per_gene` never exercised | **Fixed** | `test_mwu.py::test_tie_term_per_gene_multi_block`, cites L10 |
| L11 | Low | `environment.yml` recommended `shardad @v0.2.0` | **Fixed** | pins `shardad[cell] v0.7.1`, the floor both archive layouts need |
| N1 | Nit | `epsilon=0` + all-zero gene → NaN/inf `log2_fold_change` | **Documented, then pinned** | documented in the `de()` docstring and README; the degenerate value is pinned by a test added in v0.8.0 |
| N2 | Nit | `.item()` forces per-chunk GPU→CPU syncs | **No change** | a perf nit with no correctness impact; the syncs are still there in `_mwu.py` |
| N3 | Nit | A group literally named `'all_others'` is remapped | **No change, deliberate** | this review's own recommendation was "no action before the legacy spelling is removed"; the behaviour is noted in `__init__.py` |
| N4 | Nit | `mwu_ref` sentinel test checked only the p-value half | **Fixed** | `test_mwu.py` now asserts `(U[ref_idx] == 0).all()`, citing N4 |
| N5 | Nit | `MeanCalc` / `__version__` absent from the README API docs | **Fixed** | both documented in the README ([#126](https://github.com/ArcInstitute/gpudge_arc/pull/126)) |
| N6 | Nit | `csr_row_sums` docstring said "CSR" but accepts dense | **Fixed** | now "Per-row sum of a CSR sparse OR dense matrix" |
| N7 | Nit | README/SKILL listed 5 of the 10 default output columns | **Fixed** | both now list all ten under a `columns (10):` heading |

Two entries are **not** code changes, and are listed that way on purpose: N2 was
judged a perf nit not worth the churn, and N3 was a reserved-word behaviour this
review itself recommended leaving alone until the legacy `"all_others"` spelling
is removed. "Addressed" is not the same as "changed", and a resolution table
that blurred the two would be worth less than none.

---

## 2. Findings by severity

### HIGH

#### H1. NaN/None `groupby` labels silently bucketed into `'nan'`/`'None'` groups
- **File:** `src/gpudge/_ingest.py:47-48` (mirrored at `src/gpudge/_shard_stream.py:102`)
- **Lenses:** api-validation
- **What's wrong / why it matters:** `ingest()` builds labels via `adata.obs[groupby].astype(str).to_numpy()`. Pandas `astype(str)` turns `np.nan`/`None`/`pd.NA` into the literal strings `'nan'`/`'None'`, and `np.unique` then treats them as ordinary groups. Cells with a *missing* group assignment are silently swept into a real group: in literal-reference mode they surface as a bogus target named `'nan'`/`'None'`; in `reference=ALL_OTHERS` mode they are folded into the "rest" of every one-vs-rest comparison, perturbing every target's reference distribution. There is no validation, no warning, and no test. Unassigned/ambiguous-guide cells are common in this library's stated domain (CRISPR screens), so a real user gets numerically wrong DE with no signal.
- **Fix:** Before `astype(str)`, detect missing labels (`adata.obs[groupby].isna().any()`) and either raise a `ValueError` naming the count of unassigned cells, or emit a `UserWarning` and drop them explicitly. Mirror the guard in the streaming reference-shard obs read (`_shard_stream.py:102`).

---

### MEDIUM

#### M1. Core MWU-vs-scipy correctness tests are gated behind `@needs_cuda` despite a CPU-runnable kernel
- **File:** `tests/test_mwu.py:32-66`
- **Lenses:** tests
- **What's wrong / why it matters:** The only tests validating the U statistic and p-value against scipy ground truth (`test_mwu_ref_matches_scipy_on_synthetic`, `test_mwu_ref_ignores_ref_row`) are `@needs_cuda`, yet `_mwu.py` is fully device-agnostic (reads `X.device`, never hardcodes `.cuda()`). The `.cuda()` calls in the tests are pure convenience. On the CPU-only CI (`.github/workflows/ci.yml`), the central numerics — tie-correction delta (`_mwu.py:138-147`), continuity term (line 158), variance, two-sided p — are never exercised; a regression there passes CI green. The math is correct *today* (bit-perfect vs scipy), so this is a regression-protection gap rather than a live defect, with the documented "GPU-verified before merge" workflow as the only backstop.
- **Fix:** Add a CPU-runnable test that builds CPU tensors and asserts `mwu_one_group`/`mwu_ref` match `scipy.stats.mannwhitneyu(method='asymptotic', use_continuity=True)` on a small tie-heavy example. Keep a `@needs_cuda` variant for the device path if desired; the math check should not need a GPU.

#### M2. Streaming-vs-in-memory equivalence harness checks only Pearson correlation
- **File:** `tests/test_shard_stream.py:298-313`
- **Lenses:** tests
- **What's wrong / why it matters:** `_assert_equiv` — the shared harness for the v0.3.0 shard-streaming feature (used by mode-1/mode-2 equivalence, chunk-size invariance, 5× CPM-parity, keep_genes parity, geometric parity) — asserts only `np.corrcoef(x, y)[0,1] > 0.9999999` for `log2_fold_change`, `p_value`, `p_adj`. Pearson r is invariant under any affine transform `y = a*x + b`, so a streaming bug applying a consistent multiplicative scale (e.g. a normalization constant divided once vs twice) or a consistent additive log2fc offset yields `r = 1.0` and passes. This is markedly weaker than the in-memory chunk-invariance tests (`test_api.py:137-138, 204-205`) which assert `.abs().max() < 1e-6` — and streaming (separate reference pre-pass, per-shard accumulation, separate median pre-pass) is exactly where numerical drift is most plausible. (The harness does check `target_ncells`/`ref_ncells` exactly and two adjacent streaming tests do use `np.allclose`, so it is a coverage gap, not a demonstrated bug.)
- **Fix:** Strengthen `_assert_equiv` to `np.allclose(x, y, rtol=1e-5, atol=1e-7, equal_nan=True)` (or `assert_frame_equal` on a sorted join) for the three result columns, matching the in-memory tolerance; keep the correlation check as a coarse secondary guard.

#### M3. `environment.yml` pins `gpudge @v0.2.0` while the rest of the v0.3.0 release points conda users here
- **File:** `environment.yml:25` (cited as 20-23; the dependency is at line 25)
- **Lenses:** docs
- **What's wrong / why it matters:** `environment.yml` installs `gpudge[fast] @ ...@v0.2.0`, but `pyproject.toml` is `0.3.0`, the CHANGELOG documents a 0.3.0 release, and the README pip/uv snippets use `@v0.3.0`. README's conda/mamba section (lines 121-130) explicitly tells users to `mamba env create -f environment.yml`, so a user following the documented conda flow gets the *previous* release and misses every 0.3.0 feature (shard-streaming, `normalize_target_sum`, ultrareview fixes). The three documented install paths disagree on version. Distribution is Arc-internal and 0.3.0 has no breaking changes, so the conda user still gets a functional (if outdated) install — hence medium, not high.
- **Fix:** Bump the git ref in `environment.yml` to `@v0.3.0`; consider deriving the tag from the package version to prevent future drift.

---

### LOW

#### L1. `all_others` single-cell input (N==1) yields NaN p-value via `tie_corr` divide-by-`(N-1)`
- **File:** `src/gpudge/__init__.py:721-727`
- **Lenses:** numerical + edge-cases (two findings merged — same bug, same lines)
- **What's wrong / why it matters:** In the in-memory ALL_OTHERS path, `tie_corr = mn * tie_term[None,:] / (12 * N_t * (N_t - 1))`. With a single-cell AnnData, `N_t == 1` makes the divisor `0`; even though `mn = m_t * n_rest = 1*0 = 0`, the result is `0/0 = NaN`, which `clamp_min` does not rescue, so `var` and `p` become NaN. Unlike `mwu_one_group` (`_mwu.py:115-120`), this path has no degenerate-case guard, and no upstream check rejects `n_cells == 1`. Single-cell DE is meaningless input and the only consequence is `p=NaN` instead of the graceful `p=1.0`, so impact is minimal — but it is silent and untested.
- **Fix:** Clamp the denominator (`(N_t * (N_t - 1)).clamp_min(1.0)`) or short-circuit when `state.n_cells <= 1` to emit the documented NaN-p / zero-U sentinel explicitly. Add a CPU test mirroring `test_mwu_one_group_m_zero` for the all_others N==1 path.

#### L2. ref-mode empty target group would report `p_value=0.0` (maximally significant) instead of the `p=1.0`/NaN sentinel
- **File:** `src/gpudge/__init__.py:561, 876-877, 889-890, 951`
- **Lenses:** edge-cases
- **What's wrong / why it matters:** `p_acc = np.ones(...)` (561) is initialized, but per chunk `p_chunk = torch.zeros(...)` (876); empty target groups `continue` (889) so their `p_chunk` row stays 0.0; line 951 then overwrites the whole column slice into `p_acc`, clobbering the `ones` init with `0.0`. A `0.0` p (and `0.0` p_adj after BH) marks every gene of that group as maximally significant — the opposite of "no signal". The `np.ones` init is effectively dead for any group the loop touches; the zeros default is wrong-signed. **Not currently reachable** (`np.unique` of observed labels guarantees every group has ≥1 cell; the skipped ref-label row is dropped before output), so this is a latent footgun. Note: the finding's secondary claim that streaming has the same shape is **inaccurate** — `_shard_stream.py:372` uses a per-group scatter write, so the streaming `np.ones` init is meaningful and not bugged.
- **Fix:** Initialize `p_chunk = torch.full((n_groups, ch), float('nan'))` so untouched rows read as the NaN sentinel BH preserves, or scatter only active group rows into `p_acc` rather than overwriting full columns.

#### L3. `epsilon` validation accepts NaN and +inf (only `epsilon < 0` is checked)
- **File:** `src/gpudge/__init__.py:405-406` (duplicated at `_shard_stream.py:249`)
- **Lenses:** api-validation
- **What's wrong / why it matters:** `if epsilon < 0: raise` lets `nan` and `inf` through (`nan < 0` and `inf < 0` are both False). `epsilon=nan` produces all-NaN `log2_fold_change`; `epsilon=inf` gives `inf/inf = NaN` (the finding's "collapses toward 0" wording is the lone inaccuracy — output is all-NaN either way). This is inconsistent with `normalize_target_sum`, which is rigorously finiteness-checked in `resolve_target_sum` (`_normalize.py:55`). Low: the user must deliberately pass a non-finite value, and the failure is obviously-broken all-NaN output, not silently-plausible numbers.
- **Fix:** `if not math.isfinite(epsilon) or epsilon < 0: raise ValueError("epsilon must be a finite value >= 0")`. Apply the same at `_shard_stream.py:249` for parity.

#### L4. `output_columns={}` (empty dict) passes validation and silently yields a degenerate DataFrame
- **File:** `src/gpudge/__init__.py:412-423` (mirrored at `_shard_stream.py:405-407`, `_output.py:47-49`)
- **Lenses:** api-validation
- **What's wrong / why it matters:** `output_columns` is treated as "provided" whenever non-None. `{}` has no unknown keys and no duplicate destinations, so both entry-point checks pass, then `df.select(list({})).rename({})` (line 1048) returns a frame with **zero columns** (with polars 1.41 it collapses to shape `(0,0)` — rows are lost too, contra the finding's "correct number of rows"). Almost certainly always a user mistake (meant `None`, or forgot to populate the dict), accepted silently rather than rejected.
- **Fix:** Inside the existing block: `if output_columns is not None and not output_columns: raise ValueError("output_columns must be a non-empty dict mapping default column names to output names, or None.")`. Mirror in `stream_de`.

#### L5. Non-string `reference` (list/array) produces an opaque error instead of a clear ValueError
- **File:** `src/gpudge/__init__.py:458`
- **Lenses:** api-validation
- **What's wrong / why it matters:** On the in-memory path, `reference` is type-checked only for `AnnData` (369) and `None` (456). A numpy-array `reference` makes `reference == ALL_OTHERS` (line 458) an array, so `if <array> and ...` raises the opaque `ValueError: The truth value of an array ... is ambiguous` — unrelated to the real mistake. Realistic because shardad's writer API uses `reference=[label]` (a list), so a user may pass a list by analogy. **Caveat:** the finding's lead case `reference=["ntc"]` actually works correctly (`np.where` broadcasts a length-1 list to the right position); only a list of length `≠1, ≠n_groups` errors opaquely, and only `len==n_groups` produces a garbage mask. So in supported usage this is an API-polish gap, not a correctness bug — hence low.
- **Fix:** After the legacy-remap block, add an explicit check: if `reference` is not the ALL_OTHERS sentinel and not `str`/`None`, raise `ValueError("de(): in-memory reference= must be a group-label string or the ALL_OTHERS sentinel; got <type>")`.

#### L6. `all_others` auto gene-chunk budget under-estimates GPU peak (~2.5×), risking a first-chunk OOM + recovery tax
- **File:** `src/gpudge/_stream.py:62-71`
- **Lenses:** gpu-memory
- **What's wrong / why it matters:** `_auto_gene_chunk_size` uses `24 B/cell/gene` (calibrated for ref-mode against the *small* pre-sorted reference) and `accumulator_bytes=0` for the all_others path. But `_rank_with_ties` (`_mwu.py:33-63`) materializes ~6 simultaneous full `(n_cells, ch)` f64/int64 arrays (Xd, sorted_vals, sort_idx, group_id, ranks_sorted, inv_idx, ranks; `pos` is an `.expand` view, not materialized — a minor over-count in the finding) plus the f32 `X_chunk` — ~57-61 B/cell/gene vs the 24 assumed (~2.5×). The per-chunk `(n_groups, ch)` f64 accumulators (rank_sums/U/p) are also unbudgeted, doubly under-sizing many-group inputs. **Not a crash** — `run_gene_chunks_with_recovery` is idempotent and the 0.20 free-memory cap absorbs much of the error — but on a large all_others run a first chunk *may* OOM, then pay `gc.collect()` + `empty_cache()` + halved retry. The finding's "essentially guaranteed" framing overstates; it is a bounded efficiency wart.
- **Fix:** Give the all_others path its own coefficient (~64-72 B/cell/gene) and add the per-chunk `(n_groups, ch)` accumulators to `bytes_per_gene` for `ref_mode=False`. Alternatively sub-chunk the f64 cast inside `_rank_with_ties` (as `_means.group_means` already does), lowering the real peak.

#### L7. Streaming chunk-size heuristic budgets GPU accumulators that streaming Phase 1 never allocates
- **File:** `src/gpudge/_shard_stream.py:299-303`
- **Lenses:** streaming
- **What's wrong / why it matters:** `stream_de` calls `_auto_gene_chunk_size(..., n_groups=n_targets, ref_mode=True)`, so the heuristic adds `8*3*n_groups` (or `*4` geometric) for per-chunk `(n_groups, chunk)` f64 GPU accumulators. But streaming Phase 1 holds **no** such GPU buffers — `group_chunk_stats` returns per-group `(chunk,)` tensors copied straight to host accumulators (`_shard_stream.py:362-372`). So the heuristic charges streaming for memory it never uses, shrinking the auto chunk (conservative — never OOMs). The finding's "dominates" framing overstates for typical CRISPR screens (large NTC reference: e.g. CCL_2/c50 the phantom term is ~6% of the reference-ranking term, ~7% chunk-width change inside the throughput plateau); it only truly bites in the atypical `n_targets >> n_ref` regime.
- **Fix:** Pass `ref_mode=True` but `n_groups=1` (or a dedicated streaming flag) so only the single resident `(n_ref, chunk)`/`(m, chunk)` working set is budgeted. Verify with an OOM-recovery run on a many-target archive.

#### L8. Streaming Phase-1 `del shard, Xs` is defeated by the `_chunk` closure; `/dev/shm` segments not reclaimed when intended
- **File:** `src/gpudge/_shard_stream.py:344-378`
- **Lenses:** concurrency-resources
- **What's wrong / why it matters:** Each shard's `shard = gs.to_anndata()` allocates three POSIX `SharedMemory` segments in `/dev/shm`, attaching a strong-ref bundle to `adata.X` (reclaimed only via `weakref.finalize` on GC). The inner loop defines `_chunk(..., Xs=Xs)` (default-arg capture), and `_chunk` stays bound at function scope after the loop, holding a strong ref to `Xs`. So `del shard, Xs` (line 378) cannot drop the bundle — it is freed only when `_chunk` is rebound next iteration (and the **final** shard's segments persist until `stream_de` returns). Net: ~2 shards' worth of `/dev/shm` at steady state instead of 1; the `del` is a no-op for its stated purpose. Contrast the median pre-pass (264-267) where `del _shard` *is* effective (no closure capture). Bounded (`target_shard_bytes` per shard) and no correctness impact, but can pressure RAM-backed tmpfs since each segment holds the full decompressed f32 CSR.
- **Fix:** Call `shard.close_shared_memory()` after the inner loop (immediate munmap+unlink, doesn't rely on prompt refcount drop), or extend the cleanup to `del shard, Xs, _chunk`. Add a test asserting `/dev/shm` segment count returns to baseline between shards.

#### L9. `log2_fold_change` and the `epsilon` pseudocount are never validated against an independent ground truth
- **File:** `tests/test_api.py:82-93`
- **Lenses:** tests
- **What's wrong / why it matters:** `log2_fold_change` is the one output column scipy doesn't provide, and the README claims "bit-perfect vs CPU pdex on log2FC". Yet every log2fc assertion is gpudge-vs-gpudge self-consistency or a divergence inequality (`test_de_geometric_mean_option` asserts arithmetic vs geometric *differ* — a sanity check). No test pins log2fc to a closed-form `log2((target_mean+epsilon)/(ref_mean+epsilon))`, and `epsilon`'s numerical effect is untested (every test passes `epsilon=0.0`; only the negative-value guard is checked). The default could be changed, or `epsilon` dropped from numerator/denominator, and the suite stays green except the normally-skipped `@needs_data` real-data test (loose Pearson `r>0.999`). A coverage hole, not evidence of a bug — `target_mean`/`ref_mean` are emitted as columns, making the closed-form check trivial.
- **Fix:** Add a test that recomputes log2fc directly from the returned `target_mean`/`ref_mean` and asserts near-exact equality; add a test that two distinct `epsilon` values produce the predicted shift on a near-zero-mean gene.

#### L10. The multi-block path of `_tie_term_per_gene` is never exercised
- **File:** `tests/test_mwu.py:71-102`
- **Lenses:** tests
- **What's wrong / why it matters:** `_tie_term_per_gene` (`_mwu.py:66-94`) processes genes in blocks of `block = max(1, min(n_genes, 64_000_000 // k))`, writing `out[s:s+block]`. The block loop/slice logic only runs when `n_genes > block`; every test (and every internal call site) uses tiny tensors where `block >= n_genes`, so the loop always executes exactly once with `s=0`. A regression in the block stride, per-block `run_id` reset, or out-slice indexing would go uncaught. Memory-bounding correctness path with no coverage.
- **Fix:** Add a CPU test forcing multi-block — e.g. monkeypatch the `64_000_000` constant low, or pass a tall tensor — and assert equality to a single-block / numpy reference.

#### L11. `environment.yml` comment recommends `shardad @v0.2.0`, which predates the streaming reader API
- **File:** `environment.yml:16` (cited as 18-19)
- **Lenses:** docs
- **What's wrong / why it matters:** The streaming-install note suggests `shardad @ ...@v0.2.0`, but README (104-108) and `pyproject.toml` `[tool.uv.sources]` (44-49) both state the target-aware reader API (`read_reference`/`iter_group_shards`/`write_sharded(group_by=, reference=)`) landed *after* the v0.2.0 tag, so callers must pin commit `35e82bf`. `_shard_stream.py` calls `arch.read_reference()`/`arch.iter_group_shards()`/`arch.manifest.get("group_by")` (lines 39/90/118/264/344), so a conda user following the comment installs a shardad lacking those methods and streaming fails with a loud `AttributeError`. Low: a comment in a secondary install file for an opt-in, source-only path; the primary docs are correct and the failure is loud, not silent.
- **Fix:** Update the comment to pin `35e82bf` (matching README and `[tool.uv.sources]`), or reference README's streaming-install section instead of the stale `@v0.2.0` string.

---

### NIT

#### N1. `epsilon=0` with an all-zero gene emits NaN/inf `log2_fold_change` (silent except a numpy RuntimeWarning)
- **File:** `src/gpudge/__init__.py:983` (same idiom at `_shard_stream.py:382`)
- **Lenses:** edge-cases
- With the documented-allowed `epsilon=0`, an all-zero gene gives `log2(0/0)=NaN` and a target-only gene gives `+inf`; with no filter active these are emitted verbatim (`_output.py:114,141`). The default `1e-9` avoids it. The existing all-zero-gene test sets `epsilon=0.0` but never checks `log2_fold_change`.
- **Fix:** Document that `epsilon=0` can produce NaN/inf log2fc for zero-mean genes (or post-process to a defined value), and extend the all-zero-gene test to pin log2fc behavior under `epsilon=0`.

#### N2. `_rank_with_ties` / `_tie_term_per_gene` force per-chunk GPU→CPU syncs via `.item()`
- **File:** `src/gpudge/_mwu.py:48, 88`
- **Lenses:** gpu-memory
- Both call `.item()` on a device scalar to size a data-dependent scatter buffer (`max_tgroups`/`n_runs`) — a blocking GPU→CPU sync that defeats async overlap. Fires once per gene-chunk (line 48) and once per gene-block (line 88, several times per chunk for a wide reference). Cost is real but small vs the sort/scatter; no correctness issue.
- **Fix:** Only if `nsys` shows a material stall, size buffers to the static upper bound (`n_cells` group count; `k` runs) to avoid the round-trip. Otherwise leave as-is.

#### N3. A real group literally named `'all_others'` is unconditionally remapped to one-vs-rest
- **File:** `src/gpudge/__init__.py:391-399` (mirrored `_ingest.py:33-41`)
- **Lenses:** api-validation
- The legacy-sentinel remap rewrites `reference='all_others'` to the ALL_OTHERS sentinel with only a `DeprecationWarning`, so a dataset with a genuine `'all_others'` group can't use it as a literal reference. **Intentional and documented** (code comment + docstring); effectively a reserved word during the deprecation window.
- **Fix:** No action before the legacy spelling is removed (it then becomes a literal label again). Optionally note the reserved-word status in the docstring.

#### N4. `mwu_ref` sentinel test checks only `p[ref_idx]` is NaN, not the documented `U[ref_idx] == 0`
- **File:** `tests/test_mwu.py:56-66`
- **Lenses:** tests
- `mwu_ref` documents (and the parquet `statistic` column depends on) `U=0, p=NaN` for the ref row, but `test_mwu_ref_ignores_ref_row` asserts only the NaN half. Doubly guaranteed by construction (`U_out` zero-init + explicit `U_out[g]=0.0`), so negligible risk — but the contract is half-tested.
- **Fix:** Add `assert (U[ref_idx] == 0).all()`.

#### N5. Public API members `MeanCalc` and `__version__` are in `__all__` but absent from README API docs
- **File:** `README.md:213-233`
- **Lenses:** docs
- `__all__ = ["de", "ALL_OTHERS", "MeanCalc", "__version__"]`, but the README API section documents only `de()` and `ALL_OTHERS`. The `MeanCalc` values are already covered via the `mean_calc` kwarg row, and `__version__` is conventional, so impact is trivial.
- **Fix:** Add a brief README note that gpudge also exports the `MeanCalc` literal and `__version__`, or intentionally drop them from `__all__`.

#### N6. `csr_row_sums` docstring says "CSR sparse matrix" but also accepts dense X
- **File:** `src/gpudge/_csr_dense.py:128-143`
- **Lenses:** docs
- The summary line names only CSR sparse input, but the fallback branch (and its body comment) handle dense arrays, and `de()` calls it with possibly-dense `adata.X` (natively dense or after `densify_input=True`). Cosmetic; behavior is correct.
- **Fix:** Reword the summary to "Per-row sum of a CSR sparse OR dense matrix".

#### N7. README/SKILL streaming usage comment lists only 5 of the 10 default output columns
- **File:** `README.md:156` and `gpudge-usage/SKILL.md` (finding cites `_shard_stream.py:391-407` as the code anchor — the code is correct; the defect is in the docs comments)
- **Lenses:** docs
- The default frame has 10 columns (`DEFAULT_OUTPUT_COLUMNS`, `_output.py:8-14`), but the no-`output_columns` example annotates `# columns: target, feature, log2_fold_change, p_value, p_adj` (5, no "…" qualifier); SKILL.md repeats it. A subset listing, so non-misleading on what's present but incomplete. (The title's "no-reference warning relaxation" sub-claim is unsupported; only the column-count issue holds.)
- **Fix:** List all 10 default columns or add "(among others)"/"…", or make the example actually pass `output_columns={...}` for those 5.

---

## 3. Areas that looked solid

- **Core MWU numerics:** No correctness defect was found in the statistic itself — tie correction, continuity, variance, and two-sided p are bit-perfect vs scipy on realistic input. The numerical findings are confined to a degenerate single-cell input (L1/nit) and a documented-but-undocumented `epsilon=0` edge (N1). The *gap* is test coverage of these numerics (M1, L9, L10, N4), not the math.
- **OOM-recovery / chunk-streaming driver:** Verified idempotent — accumulators are written by absolute gene index and retries re-cover `[start, stop)` identically. The GPU-memory findings (L6, L7) are budgeting accuracy, not crashes; the backstop is sound.
- **Streaming correctness paths:** The per-group scatter writes in the streaming accumulator are correct (and notably *not* affected by the in-memory ref-mode footgun in L2). Streaming concerns are a resource-cleanup wart (L8), a conservative budget (L7), and weak equivalence assertions (M2) — not wrong results.
- **`reference`/legacy-sentinel handling:** Behaves exactly as documented; the only items are input-validation polish (L5) and an intentional, documented reserved-word (N3).
- **`normalize_target_sum` validation:** Held up as the *positive* example — rigorously finiteness-checked in `resolve_target_sum`, which is why `epsilon`'s weaker guard (L3) stands out by contrast.