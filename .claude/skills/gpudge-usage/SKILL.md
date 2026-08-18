---
name: gpudge-usage
description: Use when installing gpudge or running GPU Mann–Whitney differential expression with it on single-cell CRISPR perturbation screens — covers pip/uv/conda install + extras, the de() entry point, one-vs-rest (ALL_OTHERS), shardad archive streaming, both layouts (de(archive=), shard and cell/.csad), a bring-your-own cell source (de(cell_source=)), CPM / library-size normalization, and opt-in per-gene filters.
---

# Using gpudge

GPU-only Wilcoxon (Mann–Whitney U) differential expression for single-cell CRISPR
screens. One public entry point: `de()`. Bit-perfect vs CPU `pdex` on log2FC/p-value;
~100–600× faster on the DE stage. **Requires CUDA 12.6+, an NVIDIA driver, and a
CUDA GPU with enough VRAM for the reference pool** — there is no CPU fallback.
Developed on H100 80 GB and A100 40 GB; also gated on an L4 (sm_89).

## Install

Install from a release tag (there is no PyPI release yet -- see below). Use the
latest tag, shown here as `vX.Y.Z`.

```bash
# pip — torch's CUDA 12.6 build comes from the PyTorch index:
pip install --extra-index-url https://download.pytorch.org/whl/cu126 \
  "gpudge[fast] @ git+https://github.com/ArcInstitute/gpudge.git@vX.Y.Z"
```

```bash
# uv (development): clone, then
uv venv && uv pip install --torch-backend=cu126 -e ".[dev,fast]"
# NOT `uv sync` without shardad SSH access: its universal lock resolves the
# private shardad source even when you did not ask for the streaming extra.
# --torch-backend is needed because uv pip skips [tool.uv.sources] as well,
# so torch would otherwise come from PyPI rather than the cu126 build.
```

```bash
# conda/mamba: manages the env, pip installs gpudge into it
mamba env create -f environment.yml && mamba activate gpudge
```

**Extras:** `[fast]` = numba single-pass CSR kernel (~3× faster on big sparse X;
correctness identical without it). `[streaming]` = `shardad[cell]>=0.7.1` for
`de(archive=)` (also private, so install it first). The `[cell]` extra is `pyfastpfor`,
required to open a `codec='pfordelta'` archive — **which every production `.csad` is**,
so treat it as required in practice; shardad's zstd cell codec opens without it. The
`0.7.1` floor is where the cell path's `gather_rows(n_threads=)` landed:

```bash
pip install "shardad[cell] @ git+ssh://git@github.com/ArcInstitute/shardad.git@v0.7.1"
```

`[streaming-gpu]` = `shardad[cell,gpu]>=0.7.1`, adding GPU device decode — **shard
layout only**; a cell-layout archive takes the host CSR path either way.
`[dev]` = pytest/ruff/scanpy/pyyaml.

## Quickstart

`de()` takes an in-memory `AnnData` of **raw counts** (sparse CSR is fine — streamed
to GPU per gene-chunk, no host densify spike).

```python
import anndata as ad
from gpudge import de

adata = ad.read_h5ad("filtered.h5ad")          # raw counts
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
#   ref_ncells, log2_fold_change, p_value, Ueffect, p_adj
```

**One-vs-rest:** `from gpudge import ALL_OTHERS; de(adata, groupby=..., reference=ALL_OTHERS)`.

## Streaming a shardad archive (low host RAM)

For datasets too large to materialize, stream a target-aware shardad archive. A
**memory/feasibility** path, not a speedup.

Both shardad layouts work and are detected from the archive's own **manifest**, not
its file extension. On `layout='cell'` (`.csad`), `stream_n_workers` is the Rust
gather's thread count (no extra host RAM) and `stream_prefetch` has no effect; on
`layout='shard'` **host decode**, the two size a decode-ahead worker pool and its
queue depth, and both are ignored under GPU device decode.

Budget host RAM from the worker pool, not from one shard. Three cases:

- **Shard layout, host decode, default** (`stream_n_workers=16` at roughly 14 GB each)
  — about 223 GB at CCL_2 scale. Turn the workers down if that does not fit.
- **Shard layout, host decode, no prefetch** (`stream_prefetch=0`, which is what
  disables decode-ahead — `stream_n_workers` is then unused; turning *it* down to 1
  still prefetches) — `~reference + one decoded shard`.
- **Shard layout, device decode** (`[streaming-gpu]`) — the decoded CSR lives on the
  GPU, so the host holds only the reference plus compressed staging; the per-worker
  batch is gone entirely.

```python
import gpudge
# Mode 1 — use the archive's own reference (non-targeting) shard:
df = gpudge.de(archive="/path/to/archive")
# Mode 2 — external control pool as the reference (ntc_adata = any AnnData of NTC cells):
df = gpudge.de(archive="/path/to/guides", reference=ntc_adata)
```

`de(shard_archive=…)` is the deprecated spelling of `archive=`; it still works and
emits a `DeprecationWarning`.

## The other two input modes

`de()` takes cells from exactly one of three places:

```python
# 2. In-memory, with a SEPARATE control AnnData — no target∪reference concat:
df = gpudge.de(adata=targets, groupby="comparison", reference=ntc_adata)

# 3. Bring your own cells — you yield one CellGroup per target group and gpudge
#    never opens the payload:
from gpudge import CellGroup
def source():
    for label, X in my_groups():          # X: CSR, or C-contiguous ALIGNED dense
        yield CellGroup(label, X)
# `reference` here is the control POOL itself — an AnnData, or a bare
# cells x genes numpy array / scipy sparse matrix:
df = gpudge.de(cell_source=source, targets=[...], var_names=[...], reference=ntc_pool)
```

All three drive the same reference-pool core, so they agree **bit-for-bit on the same
inputs** — the same target cells in the same order against the same control pool.
Sharing the core is not on its own a guarantee that two different runs match.

Two `cell_source` limits worth knowing up front: `normalize_target_sum="median"` raises
`NotImplementedError`, and the automatic gene-chunk sizer is blind here (no source can
report its largest group without being drained) — pin `gpu_gene_chunk_size=` for large
groups.

## Normalization — pick exactly one

- Pre-normalize yourself (`sc.pp.normalize_total`) and pass the result, **or**
- `cpm_normalize=True` — inline CPM (per-cell sum → 1e6), on GPU, does not mutate `adata.X`, **or**
- `normalize_target_sum=<number | "median">` — scanpy `normalize_total` parity.

`cpm_normalize` and `normalize_target_sum` are mutually exclusive. The two inline
options (and the `cpm_*` filters) **assume raw counts** — do **not** also pass them on
data you already normalized, or you double-normalize. Pick exactly one path.

## Effect size — two rank-level options

`log2_fold_change` is a ratio of means while the p-value is a rank statistic, and
the two can disagree. Both additions below report effect size on the rank axis
itself, in **log2 units**, and both leave the base two-sided columns untouched.
Neither works with `ALL_OTHERS`.

- **`lfc_threshold=τ | [τ…]`** — one-sided tests against an effect-size *floor*
  (`H0: log2FC <= +τ`, `H0: log2FC >= -τ`), by comparing the target against
  `reference * 2**(±τ)`. Adds `tau=<±τ>_{p,Ueffect,padj}` per (τ, direction), each
  its own BH family; a whole τ grid costs one pass. Comparable to DESeq2's
  `lfcThreshold`. Restrict the directions with `lfc_threshold_alt`.
- **`tau_star=(0.5, 0.05)`** — the shift the rank test itself estimates.
  `tau*_p0.5` is the Hodges–Lehmann point estimate; `tau*_p0.05` is a one-sided
  95% bound. `tau_star_se=True` adds `tau*_lo_p0.025`, `tau*_hi_p0.025`, `tau*_se`
  (and forces `0.5` into the level set).

Two traps:

- **`tau_star` needs normalized input.** On raw counts `T_i == R_j` is common for
  small integers, so most genes land on a tie atom at zero — measured on a 1.27 M-cell
  archive, 87.9% of finite `tau*` were within 1e-5 of zero and only 6.0% of rows
  reached `|tau*| >= 0.01`. `normalize_target_sum` collapses that plateau to 0.02%
  and lifts usable rows to 46.8%.
- **`lfc_threshold`'s τ is a multiplicative shift** and `0 * 2**τ == 0`, so it moves
  only the nonzero entries: on high-dropout genes the effective floor is weaker than
  τ suggests. `lfc_threshold=0.5` is *not* "drop everything with |log2FC| < 0.5".

## Key parameters

| Param | Default | Notes |
|---|---|---|
| `groupby` | required (in-memory) | obs column; auto-resolved from the archive manifest when streaming |
| `reference` | required (in-memory, and with `cell_source=`) | group name or `ALL_OTHERS` (`"__all_others__"`); when streaming it may be omitted **only if the archive designates its own reference**, otherwise pass an external `AnnData` |
| `mean_calc` | `"arithmetic"` | or `"geometric"` (matches pdex fold-change) |
| `epsilon` | `1e-9` | log pseudocount; matches scanpy |
| `filter_gene_min_mean_value` / `filter_gene_min_total_value` | `None` | opt-in per-(group,gene) filter on X as supplied |
| `filter_gene_min_cpm_cell` / `filter_gene_min_cpm_bulk` | `None` | opt-in; assume raw counts |
| `keep_genes` | `None` | bool mask on `var_names`; AND-combined with filters |
| `lfc_threshold` | `None` | τ or a finite iterable of τ (log2 units, 0 ≤ τ ≤ 30); adds `tau=<±τ>_{p,Ueffect,padj}` |
| `lfc_threshold_alt` | `("up", "down")` | which one-sided alternatives to emit per τ |
| `tau_star` | `None` | one-sided `p_dir` levels in (0, 1); emits a signed `tau*_p<level>` log2 shift |
| `tau_star_iters` | `None` | bisection steps per level (default 20); validated even when `tau_star` is unset |
| `tau_star_se` | `False` | adds `tau*_lo_p0.025` / `tau*_hi_p0.025` / `tau*_se`; requires `tau_star` |
| `gpu_gene_chunk_size` | `None` | auto from free VRAM |
| `oom_recovery` | `True` | on CUDA OOM, halve chunk and retry |
| `densify_input` | `False` | dense per-group slicing; **mutates a materialized sparse `adata.X` in place** (dense input: no-op). In-memory group-label / `ALL_OTHERS` only — see Gotchas |
| `cpm_normalize` | `False` | inline CPM normalization (per-cell sum → 1e6) on GPU; does not mutate `adata.X` |
| `normalize_target_sum` | `None` | inline library-size normalization (a number or `"median"`); mutually exclusive with `cpm_normalize` |
| `archive` | `None` | path to a shardad archive → stream instead of an in-memory `adata` (low host RAM). Both layouts (`layout='shard'`, `layout='cell'`/`.csad`), detected from the manifest, not the extension |
| `shard_archive` | `None` | deprecated spelling of `archive=`; still accepted, emits a `DeprecationWarning` |
| `cell_source` | `None` | callable yielding one `CellGroup(label, X, rows=None)` per target group; needs `targets=` + `var_names=` + a `reference=` control **pool** (AnnData, or a bare cells × genes numpy/scipy matrix) |
| `stream_n_workers` | `16` | shard layout: decode-ahead workers (~14 GB host RAM each). Cell layout: the Rust gather's thread count |
| `stream_prefetch` | `2` | shard layout: decode-ahead queue depth (`0` = serial, lowest RAM). Inert on cell layout |
| `release_gpu_memory` | `True` | hand torch's (and cupy's) caching pools back to the driver on exit, so the caller can allocate after `de()` |
| `output_columns` | `None` | rename/select dict, e.g. `{"log2_fold_change": "log2fc"}` |

All `filter_gene_*` are **opt-in** (default `None` = no filter) and AND-combined.

## Gotchas

- **GPU required** — no CPU path; needs a CUDA GPU even for tiny inputs.
- `cpm_*` filters and `cpm_normalize` **assume raw counts**; the cpm *filters* warn once on non-integer/negative X (`cpm_normalize` alone does not warn).
- `densify_input=True` **mutates a materialized sparse `adata.X`** and emits a `UserWarning`
  — pass `adata.copy()` first if you need the sparse matrix preserved. An
  already-dense `X` is a no-op: no mutation, no warning. Only the in-memory
  group-label / `ALL_OTHERS` path honours it at all — a sparse `AnnData` view
  raises, `archive=` and `adata=` + `reference=<AnnData>` raise, and
  `cell_source=` ignores the flag entirely (even with an AnnData pool — that
  branch returns before the guard that would have raised).
- Legacy `reference="all_others"` still works but is deprecated (use `ALL_OTHERS`).
- See the `de()` docstring in `src/gpudge/__init__.py` for the full parameter set.
