# gpudge multi-agent code review — 2026-06-13

Comprehensive multi-agent ("ultrareview") of the gpudge codebase. 9 review
dimensions fanned out across the modules; every finding was independently
checked by two adversarial verifiers (a correctness lens + a refuter) before
being kept. **66 agents, 28 raw findings → 23 survived verification**
(11 high-confidence, 11 contested, 1 medium).

## Summary

The library is healthy: the in-memory MWU/CPM compute path is well-validated
(bit-perfect vs pdex / scipy) and **no finding identifies a wrong-result bug on
the primary, tested path**. Every confirmed defect lives in the newer
shard-streaming path (`_shard_stream.py`) diverging from the in-memory path, or
in latent/defensive hardening of the numba CSR kernel.

## Resolution status

| # | Severity | Location | Finding | Status |
|---|---|---|---|---|
| 1 | High | `__init__.py` | `reference="all_others"` legacy spelling mishandled on `shard_archive=` path | **Fixed** (this PR) |
| 2 | Medium | `_shard_stream.py:258` | Empty-archive early return ignored `output_columns` (wrong schema) | **Fixed** (this PR) |
| 3 | Medium | `_shard_stream.py:229` | `output_columns` duplicate-destination check missing on streaming path | **Fixed** (this PR) |
| 4 | Medium | `_shard_stream.py:270` | CPM raw-counts warning samples only the reference, never target shards (warning-only, zero numeric impact) | Issue filed |
| 5 | Low | `_shard_stream.py:298` | OOM chunk-downshift rediscovered per target group (perf-only, rarely triggers) | Issue filed |
| 6 | Low | `_csr_dense.py:74` | Numba kernel last-write-wins on non-canonical CSR (duplicate col indices) vs scipy's sum | **Fixed** (this PR) |
| 7 | Low | `_csr_dense.py:70` | Row indices gathered under `boundscheck=False` with no range validation (latent) | **Fixed** (this PR) |
| 8 | Low | `__init__.py:848` | Last/downshifted gene-chunk uses a non-contiguous pinned-buffer view (perf-only) | Issue filed |
| 9 | Low | `_means.py:47` | Geometric mean: no test/guard for X ≤ −1 where `log1p`→NaN propagates | Issue filed |

The fix cluster (#1/#2/#3) is closed most economically by **hoisting the shared
input-validation + the legacy-`all_others` remap above the `if _streaming:`
dispatch in `de()`**, so both paths validate identically.

## Confirmed finding details

### High

**#1 — `reference="all_others"` mishandled on the `shard_archive=` path.**
The streaming dispatch ran before the legacy→`ALL_OTHERS` remap, and the
streaming guard matched only the new `"__all_others__"` spelling. So
`de(shard_archive=…, reference="all_others")` emitted no `DeprecationWarning`,
skipped the intended `NotImplementedError`, and fell through to a misleading
"…is not among the archive's reference labels" `ValueError`. Worst case: a real
group literally named `all_others` is silently used as a literal reference
instead of triggering 1-vs-rest. The only confirmed defect with a
silent-wrong-behavior mode. **Fix:** hoist the remap above the dispatch.

### Medium

**#2 — Streaming empty-archive return ignores `output_columns`.** When the
archive enumerates zero targets, `stream_de` returned
`DEFAULT_OUTPUT_COLUMNS` verbatim before the select/rename tail, so an empty
result had a different schema than a non-empty one for the same call. **Fix:**
route the empty frame through the same select/rename.

**#3 — `output_columns` duplicate-destination unchecked on streaming.**
In-memory `de()` raised a clear `ValueError`; `stream_de` validated only
unknown keys, so a duplicate destination surfaced as an opaque polars
`DuplicateError` after compute. **Fix:** the hoisted validation now covers both
paths (fail-fast, before compute).

**#4 — Streaming raw-counts warning samples only the reference.** The CPM
non-count `UserWarning` checks `ref_X` only; the in-memory path samples all
cells and checks negative library sizes. Warning-only, **zero numeric impact**.
Filed as an issue.

### Low (latent / perf / hardening)

**#5 — OOM downshift rediscovered per target group** (`run_gene_chunks_with_recovery`
called once per group with the same outer chunk; the halving is local and
discarded). Conservative budget + Phase 0 exercising the same working set first
means the OOM path typically never fires. Perf hardening; filed.

**#6 — Numba CSR kernel last-write-wins on non-canonical CSR.**
`out[i, c] = data[j]` overwrote duplicate column entries; scipy `.toarray()`
sums them, so the `[fast]` path silently disagreed with the scipy fallback on
non-canonical input. Only reachable from hand-built `(data, indices, indptr)`
triplets (every realistic construction path yields canonical CSR). **Fix:**
accumulate with `+=` (matches scipy at negligible cost). Both kernel variants.

**#7 — Row indices unvalidated under `boundscheck=False`.** An out-of-range row
read `indptr` out of bounds (silent corruption / segfault) instead of scipy's
`IndexError`. Purely latent (all callers pass valid `arange` / group splits).
**Fix:** O(m) range check in `csr_rows_col_range_to_dense` before dispatch.

**#8 — Last/downshifted gene-chunk uses a non-contiguous pinned-buffer view**,
defeating the async pinned-copy fast path. Correctness unaffected; perf-only,
exceptional path. Filed.

**#9 — Geometric mean has no test/guard for X ≤ −1** where `log1p` produces
NaN that silently propagates into `log2_fold_change`. A contract-pinning test
gap (geometric mean is mathematically undefined there; the docstring already
disclaims transform responsibility), not a wrong-result bug. Filed.

## Contested findings (one confirm, one refute — needs human judgment)

All 11 share a shape: *"the code does exactly what the finding says, but it's at
the float-noise floor, unreachable from public callers, or a doc-nit, not a
defect."* Notable calls:

- **`t³` tie-term precision ceiling (>~208k identical cells)** (`_mwu.py`): real
  arithmetic, but gpudge uses the byte-identical float64 formula as scipy (its
  validation oracle), so it never *diverges*. At most a clarifying comment.
- **`all_others` scaled "other-unit" mean f32-vs-f64 ordering** (`__init__.py`):
  ~1e-7 relative, exactly the float32-multiply noise the project already
  documents as acceptable. Cosmetic.
- **Stream/in-memory equivalence uses correlation gate, not `assert_allclose`**
  (`tests/`): the correlation gate is what the spec specifies; tightening
  `p_value` would be stricter than the project's own self-consistency test and
  likely flaky. Recommend tightening only `log2_fold_change` (natural zero
  anchor), **not** `p_value`.
- **Severity inflation flagged by the synth:** the `m==1` MWU "test gap"
  (claimed High) and the Phase-1 OOM item (claimed High) are over already-correct
  code — corrected to low/none.

## Coverage notes

Test gaps surfaced (most low/optional; triage):

- Streaming `output_columns` duplicate-destination + empty-archive combos — **added in this PR.**
- Streaming legacy `reference="all_others"` — **added in this PR.**
- Streaming CPM non-count warning (reference-only sampling) — unverified.
- Geometric mean out-of-domain (X ≤ −1) — decide the contract, then pin it (#9).
- CSR kernel adversarial inputs (non-canonical, out-of-range rows) — **added in this PR**; col-overrun still latent. Note: narrow-dtype CSR is already covered.
- MWU boundary sizes (m==1 target, n_ref==1) — untested but verified correct today; low-value regression guards.

---

*Method: `Workflow` orchestration — 9 finder agents (one per dimension) → per-finding
parallel verification (correctness lens + adversarial refuter) → synthesis. Severities
above reflect the verifiers' corrected assessment, not the finders' original tags.*
