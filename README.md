# gpudge

Lightweight, GPU-only Wilcoxon rank-sum (Mann–Whitney U) differential expression
for single-cell CRISPR perturbation screens. Built as a slim replacement
for the parts of [pdex](https://github.com/ArcInstitute/pdex) that the
Arc VCI pipeline uses.

## Why

- **Fast:** the whole comparison in one GPU call, with the reference sorted once and
  reused across perturbations — on the CCL_1/500-perturbation DE stage, **~4,500× vs scanpy**, **~600× vs CPU pdex**, **~140× vs rapids-singlecell**; DE time stays near-constant as the perturbation count grows
- **Low host RAM**: can use cellstream to stream AnnData shards without loading full matrix into RAM, enabling scaling to tens of millions of cells
- **Opt-in per-gene filters** (`filter_gene_min_mean_value`,
  `filter_gene_min_cpm_cell`, etc.) — unit-named and AND-combinable;
  see the `de()` docstring for the full set
- **Configurable output schema** via the `output_columns` rename/select dict


## Performance

> **`CCL_1` and `CCL_2`** are two in-house CRISPRi perturbation screens on human
> cancer cell lines, anonymised for this release. Every table below gives their
> dimensions; nothing in the library depends on them, they are simply the data
> the timings and the parity checks were measured on.

Benchmarked on **NVIDIA H100 80 GB** unless noted. These numbers were taken at
different points in the project's history — the four-engine comparison below at
v0.1–v0.2; the full-matrix and narrowed-in-memory rows at v0.3.0; the
decode-ahead default-streaming row, the current serial timing and the
`stream_n_workers` trade at v0.4.0 — and none has been re-measured since. The CHANGELOG's
per-release performance entries are the source of truth. gpudge runs
the whole test as
**one GPU pass**, so its **DE time is near-constant in the number of
perturbations** — where per-group tools (scanpy, rapids-singlecell) scale
roughly linearly.

**DE-stage time** (the engine's DE call; excludes host load + normalize), at the
highest perturbation rung where all four engines complete:

| Dataset | Perturbations | scanpy (CPU) | pdex (CPU) | rapids-singlecell (GPU) | gpudge (GPU) |
|---|--:|--:|--:|--:|--:|
| CCL_1 | 500 | 3.3e4 s (9.0 h) | 4.3e3 s (71 min) | 1.0e3 s (17 min) | **7.2 s** |

pdex — the optimized parallel-CPU reference gpudge replaces — does the same
CCL_1/500 rung in **71 min on 32 cores** (4,285 s) vs gpudge's **7.2 s**: a
**~595×** DE-stage speedup, with **bit-identical** results (log2FC and p-value
Pearson > 0.999999999 over all 5.5 M gene–perturbation pairs gpudge reports).

Above these rungs the other engines drop out — rapids-singlecell's VRAM grows
with the perturbation count and **OOMs** (CCL_1 full, CCL_2 ≥200), and the scanpy
CPU runs become multi-day. gpudge keeps going at near-constant DE cost:

| Full guide/perturbation set | gpudge DE | Platform |
|---|--:|---|
| CCL_2 — 2.06 M cells × 4,672 guides | **51 s** | H100 80 GB |
| CCL_2 — 2.06 M cells × 4,672 guides | **92 s** (176 s end-to-end) | GCP A100 40 GB |

gpudge VRAM stays flat (~15 GB CCL_1 / ~29 GB CCL_2); the A100-40 GB run fits in
< 85 GB host RAM and reproduces the CPU pdex baseline bit-for-bit. The same CCL_2
DE takes **4 h 11 min on the production CPU pdex pipeline** (~650 GB RAM).

**Full 5.54 M-cell CCL_2 — RAM by data layout.** The full raw CCL_2 (5.54 M
cells × 18,533 genes × 5,095 perturbations; 36 B non-zeros, 94 M result rows)
runs three ways on a single H100, with **bit-identical** results across all
layouts (max abs diff 0.0 on log2FC / p-value / p_adj across all 94 M rows):

| Path | Host RAM | Load | DE | Wall |
|---|--:|--:|--:|--:|
| **Full matrix** (float32 data + int64 indices; standard scipy CSR) | 448 GB | 253 s | 181 s | 7.2 min |
| **Narrowed in-memory** (uint16 data + int32 indices via cellstream; normalize inside gpudge) | **236 GB** | 88 s | 75 s | **2.7 min** |
| **Streaming — default** (`de(archive=…)`, `stream_n_workers=16`; reference GPU-resident, shards streamed) | 223 GB | — | — | 4.7 min |
| **Streaming — serial** (add `stream_prefetch=0`; lowest host-RAM floor, byte-identical) | **31 GB** | — | — | 22 min |

Narrowing the in-memory CSR is the key trick: loading **uint16 counts + int32
indices** (via cellstream) and letting gpudge normalize on-GPU
(`normalize_target_sum=`) — instead of materializing a float copy on the host —
roughly **halves both host RAM (448→236 GB) and DE time** versus the full
float32/int64 matrix, for the same bit-identical output. The DE speedup is a
host-side effect: the per-chunk CSR→dense gather and the row-sum normalize pass
are memory-bandwidth-bound over `X.data` + `X.indices`, so the narrow layout
(6 B/non-zero vs 12) halves the bytes streamed — the on-GPU work and the
host→GPU transfer are unchanged (the gather emits float32 either way). The full
matrix is only worth its wider dtypes if a downstream non-gpudge consumer needs
them.

Streaming interleaves shard I/O with compute (no separate load/DE split). Its
host RAM is set by `stream_n_workers` — the decode-concurrency dial, ~14 GB per
worker. The **default** `stream_n_workers=16` runs in 282 s at ~223 GB; adding
`stream_prefetch=0` (serial) drops the floor to ~31 GB (one resident shard) at
~1331 s (~22 min) — an order-of-magnitude lower RAM for datasets too large to
materialize at all. The two figures above are the two ends of that dial; see the
[streaming-knobs section](#streaming-a-cellstream-archive) for the full trade
(`stream_n_workers` 4 / 8 / 16 → ~75 / 126 / 223 GB, ~2.8× / 3.8× / 4.7× over
serial). The GPU device-decode path (`[streaming-gpu]`) keeps only one shard
resident and is byte-identical (see [Install](#install)).

Results are concordant with scanpy/pdex (log-fold-change sign-agreement 1.0,
top-50 Jaccard 1.0, identical significant-gene sets, matched NTC false-positive
rate) and bit-perfect vs CPU pdex on log2FC/p-value.

## Install

Requires CUDA 12.6+, an NVIDIA driver, and a CUDA GPU with enough VRAM for the
reference pool — there is no CPU fallback. Developed and benchmarked on H100
80 GB and A100 40 GB, and the pre-release GPU gate runs the `[dev,fast]` suite
on an **L4 (sm_89)**, where 13 `de()` scenarios are byte-identical to sm_90. The
core library takes an **in-memory `AnnData`** and has **no private
dependencies**.

### pip

```bash
# torch's CUDA 12.6 build is pulled from the PyTorch index:
pip install --extra-index-url https://download.pytorch.org/whl/cu126 "gpudge[fast]"
```

Requires **Python ≥ 3.11**. gpudge is a pure-Python wheel; the GPU comes from
torch.

To pin an exact commit instead of a release — or to install a change that has
not been released yet:

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cu126 \
  "gpudge[fast] @ git+https://github.com/ArcInstitute/gpudge.git@v0.9.1"
```

Extras:
- **`[fast]`** — adds `numba` (parallel single-pass CSR row-gather; ~3× faster on
  big sparse inputs). Without it the scipy `X[rows, cols].toarray()` fallback is
  used; correctness is identical.
- **`[dev]`** — `pytest`, `ruff`, `scanpy`, `pyyaml`.
- **`[streaming]`** — adds [`cellstream`](https://github.com/ArcInstitute/cellstream)
  for `de(archive=…)`. CPU-friendly: pulls no CUDA wheels. (Device decode
  auto-engages whenever cupy is already importable — e.g. via conda — and
  cellstream exposes `x_cupy()`; the `[streaming-gpu]` extra below just
  guarantees those GPU deps are installed.) It resolves from PyPI like any other
  dependency, so the extra is all you need:
  ```bash
  pip install "gpudge[fast,streaming]"
  ```

  **Platform:** cellstream ships Linux **x86_64** wheels only (`manylinux_2_28`,
  i.e. glibc ≥ 2.28, cp311 and cp312), and the vendored FastPFor is built for
  **SSE4.1** — check `grep sse4_1 /proc/cpuinfo` rather than assuming by CPU age.
  There is no wheel for Windows, for aarch64, or for Python 3.13, and a source
  build needs a Rust toolchain and a C++11 compiler. So `de(archive=…)` is
  effectively Linux-x86_64-only; the other three input modes are unaffected.
- **`[streaming-gpu]`** — `cellstream[gpu]>=0.9.0`, i.e. `[streaming]` plus
  cellstream's `[gpu]` extra (cupy + nvcomp), for GPU **device decode**: each shard
  is decompressed on the GPU by `GroupShard.x_cupy()`, which hands back a device
  cupy **CSR** that gpudge then densifies per gene-chunk — so the host never
  materialises the decoded shard (the compressed bytes still pass through it on
  their way to the device). Used automatically on an x_cupy-capable archive when cupy is
  importable (else the host `[streaming]` path). Byte-identical to host decode; faster and far more
  host-RAM-frugal at scale (see the CHANGELOG performance section). **Shard
  layout only** — there is no `x_cupy` equivalent for the cell codec, so a
  cell-layout archive takes the host CSR path either way:
  ```bash
  pip install "gpudge[fast,streaming-gpu]"
  ```

### uv (recommended for development)

```bash
git clone https://github.com/ArcInstitute/gpudge.git && cd gpudge
uv venv && uv pip install --torch-backend=cu126 -e ".[dev,fast]"
```

`--torch-backend=cu126` (uv ≥ 0.7.3) is needed because `uv pip install` skips
`[tool.uv.sources]`, so without it torch comes from PyPI. CI omits the flag
deliberately — it is CPU-only.

`uv sync --extra streaming` also works and pulls the cu126 torch build from its
pin with no manual `--extra-index-url`. cellstream resolves from PyPI like any
other dependency, so no repository access is involved.

### conda / mamba

There is no conda-forge package. Use conda/mamba to **manage the environment**, then let pip install
gpudge into it — [`environment.yml`](environment.yml) does both:

```bash
mamba env create -f environment.yml   # or: conda env create -f environment.yml
mamba activate gpudge
```

(torch's CUDA libraries ship inside the pip wheel, so the conda env doesn't need
a CUDA toolkit — only a host NVIDIA driver + GPU.)

## Usage

> **New here?** [`docs/tutorial.md`](docs/tutorial.md) is a runnable quickstart on a
> committed 5 MB slice of the public
> [VCC 2025](https://huggingface.co/datasets/arcinstitute/VCC_train) screen — one `de()`
> call, how to read the ten columns, and how to check the answer is right. Run it with
> `python examples/quickstart.py`.

```python
import anndata as ad
from gpudge import de

# CCL_2 CRISPRi screen (2.06 M cells), raw counts. Perturbation labels are in
# adata.obs["target_gene"]; non-targeting controls are labelled "non-targeting".
adata = ad.read_h5ad("screen.h5ad")

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
#   ref_ncells, log2_fold_change, p_value, Ueffect, p_adj
```

> **pdex parity.** The four annotated parameters above are the recipe to match
> the production CPU [pdex](https://github.com/ArcInstitute/pdex) pipeline —
> together they reproduce its output bit-for-bit (validated on CCL_1 and CCL_2).
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

### Effect-size floor: `lfc_threshold`

A small p-value only says "not exactly equal". `lfc_threshold` adds one-sided
tests against a real effect-size floor, applied at the rank level — the target
is compared against `reference * 2**(±τ)`:

```python
res = de(adata, groupby="target_gene", reference="non-targeting",
         lfc_threshold=[0.25, 1.0],          # a whole grid, in ONE pass
         lfc_threshold_alt=("up", "down"))   # the default; either alone is fine
# adds, per (τ, direction):  tau=+0.25_p / _Ueffect / _padj   (H0: log2FC <= +0.25)
#                            tau=-0.25_p / _Ueffect / _padj   (H0: log2FC >= -0.25)
#                            tau=+1_…      tau=-1_…
```

The base two-sided `p_value` / `Ueffect` / `p_adj` columns are always emitted
unchanged. Each (τ, direction) is its own BH family. τ is in log2 units, so it
is directly comparable to `tau_star` and to DESeq2's `lfcThreshold`. Not
supported with `ALL_OTHERS`.

Three caveats the `de()` docstring spells out and you should read before
relying on this: the rank direction can flatly contradict the sign of
`log2_fold_change`; τ is a *multiplicative* shift and `0 * 2**τ == 0`, so on
high-dropout genes the effective floor is weaker than τ suggests; and the
p-values are **not** guaranteed monotone in τ across tie transitions.

### Effect size on the rank axis: `tau_star`

`log2_fold_change` is a ratio of means; the p-value is a rank statistic. They
can disagree — in one measured screen, on 11% of gene tests. `tau_star` reports
the effect size the rank test itself estimates:

```python
res = de(adata=adata, groupby="guide", reference=control,
         normalize_target_sum="median",   # not optional here — see below
         tau_star=(0.5, 0.05))
# tau*_p0.5   signed log2 shift = the Hodges–Lehmann estimate
# tau*_p0.05  one-sided 95% bound = the largest floor this gene survives
```

Both are in log2 units, so they are directly comparable to `lfc_threshold` and
to DESeq2's `lfcThreshold`. `+/-inf` means the bound is unbounded (common on
zero-heavy genes); NaN means undefined.

**Normalize first — on raw counts `tau*` measures almost nothing.** The modal
pairwise log2 ratio is exactly 0, because `T_i == R_j` is common for small
integers, so most genes land on a tie atom at zero rather than on a resolvable
shift. Measured on a 1.27 M-cell production archive: 87.9% of finite `tau*` lie
within 1e-5 of zero and only 6.0% of rows reach `|tau*| >= 0.01`.
`normalize_target_sum` collapses that plateau to 0.02% and lifts usable rows to
46.8%. The same atom dominates the `tau_star_se` interval width.

### Bring your own cell source: `cell_source`

If you already stream cells yourself, hand them to gpudge directly instead of
pointing it at an archive it would re-read:

```python
from gpudge import de, CellGroup

def my_source():                     # may be called more than once
    for label, X in my_reader():     # X = that group's cells x genes
        yield CellGroup(label, X)

res = de(cell_source=my_source,
         targets=labels,             # ordered; defines output row order
         var_names=var_names,
         reference=control_pool,     # AnnData or a cells x genes matrix
         normalize_target_sum=1e6)
```

Same reference-pool core as `de(archive=)` and `de(adata=, reference=)`, so the
output is byte-identical — and every `de()` feature (`lfc_threshold`,
`tau_star`, the `filter_gene_*` set) works unchanged.

Three things to know. `normalize_target_sum="median"` is not yet supported in
this mode; pass the target as a number. Because a source cannot report its
largest group without being drained, the automatic gene-chunk sizer cannot
model the target working set here — **pin `gpu_gene_chunk_size=` if your groups
are large**, or keep the default `oom_recovery=True` and accept a downshift.
And byte-identity is scoped to a target matrix that is **CSR, or C-contiguous
and aligned dense**, with standard NumPy/SciPy semantics: gpudge sums library
sizes over the rows you select, and
numpy reduces a Fortran-ordered, strided or unaligned array in a different
order — enough to move a float32 tie. Such an `X` under a re-ordering `rows=`
is rejected rather than silently summed; pass
`np.require(X, requirements=["C", "A"])`.

### Streaming a cellstream archive

Both cellstream layouts are accepted and detected automatically from the archive's
manifest (not its file extension): `layout='shard'` and `layout='cell'`
(`.csad`). The knobs differ by layout — on shard layout `stream_n_workers` sizes
a decode-ahead worker pool (~14 GB host RAM **per worker**) and `stream_prefetch`
sets its queue depth; on cell layout `stream_n_workers` is the in-process Rust
gather's thread count (no extra host RAM) and `stream_prefetch` has no effect.
Cell layout uses the host CSR decode path; GPU device decode (`x_cupy`) exists
only for shard layout.

```python
import gpudge

# Stream the full 5.54 M-cell CCL_2 archive — groupby/reference are auto-resolved
# from its manifest (target_gene / non-targeting). Decode-ahead prefetch is on by
# default (stream_n_workers=16, stream_prefetch=2); pass stream_prefetch=0 for the
# serial, lowest-host-RAM path:
df = gpudge.de(archive="/path/to/archive.shad")

# Optionally validate the archive's reference label:
df = gpudge.de(archive="/path/to/archive.shad", reference="non-targeting")

# Mode 2 — external control pool as the reference (ntc_adata = any AnnData of NTC
# cells); all archive shards become targets. If the archive has its own reference
# shard it is ignored (with a UserWarning) in favor of the external pool:
df = gpudge.de(archive="/path/to/guides", reference=ntc_adata)

# Trade speed vs host RAM with stream_n_workers (the decode-concurrency dial);
# lower it on memory-constrained nodes:
df = gpudge.de(archive="/path/to/archive.shad", stream_n_workers=8)
```

Two streaming knobs tune the reader. **Their meaning depends on the archive's
layout** — the figures below are for `layout='shard'`, which is the only layout
with a decode-ahead reader:

- **`stream_n_workers`** (default 16) — decode concurrency, the main speed↔host-RAM
  dial. Peak host RAM ≈ one decoded shard per worker. Measured on full CCL_2
  (5.54 M cells, H100): `stream_n_workers` 4 / 8 / 16 → ~75 / 126 / 223 GB host
  RAM and ~2.8× / 3.8× / **4.7×** over serial (1331 s → 282 s at 16). Decode
  saturates by ~16 here, so more workers add RAM, not speed.
- **`stream_prefetch`** (default 2) — decode-ahead queue depth. A shallow queue is
  all that's needed (GPU compute ≪ decode); raising it past ~2 adds host RAM
  without speed. **`stream_prefetch=0`** is the serial, lowest-memory path
  (byte-identical output, ~one resident shard).

On **`layout='cell'`** neither figure applies: the concurrency is in-process,
so `stream_n_workers` becomes the Rust cell
gather's `n_threads` and costs **no extra host RAM** (16 is the measured
sweet spot: 2.17 G nnz/s vs 0.40 G nnz/s for an unbatched single group), and
`stream_prefetch` has **no effect at all** — the cell path gathers
synchronously in batches sized from the archive's own manifest. There is no
GPU device decode for cell layout.

Output is bit-identical regardless of either knob. Requires the optional
`streaming` extra (`pip install "gpudge[streaming]"`), which pulls cellstream
from PyPI.

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

Every keyword parameter, grouped. `adata` is the sole positional.

| Kwarg | Default | Meaning |
|---|---|---|
| **Comparison** ||
| `groupby` | `None` | column in `adata.obs`. Required with `adata=`; auto-resolved from the manifest with `archive=`; must be omitted with `cell_source=`, which raises if it is passed |
| `reference` | `None` | a group name, `ALL_OTHERS` (= `"__all_others__"`), or a control-pool `AnnData`. Required with `adata=` and with `cell_source=`. With `archive=` it may be omitted **only if the archive designates its own reference**; otherwise pass an external `AnnData` pool. With `cell_source=` it is the pool itself and may also be a bare cells × genes numpy/scipy matrix. Legacy `"all_others"` accepted with `DeprecationWarning` |
| `mean_calc` | `"arithmetic"` | one of `arithmetic`, `geometric` |
| `epsilon` | `1e-9` | matches `scanpy.tl.rank_genes_groups` |
| **Where the cells come from** ||
| `archive` | `None` | stream a cellstream archive instead of an in-memory `adata`; both layouts, detected from the manifest |
| `shard_archive` | `None` | deprecated spelling of `archive=`; still accepted, emits a `DeprecationWarning` |
| `cell_source` | `None` | callable yielding one `CellGroup(label, X, rows=None)` per target group |
| `targets` | `None` | ordered target labels; required with `cell_source=`, and defines output row order |
| `var_names` | `None` | gene names; required with `cell_source=` |
| **Per-gene filters** (all opt-in, AND-combined) ||
| `filter_gene_min_mean_value` | `None` | per-(group, gene) mean filter on `adata.X` as supplied; unit-agnostic |
| `filter_gene_min_total_value` | `None` | per-(group, gene) sum filter on `adata.X` as supplied; unit-agnostic |
| `filter_gene_min_cpm_cell` | `None` | per-cell CPM filter (assumes raw counts; warns on non-integer X) |
| `filter_gene_min_cpm_bulk` | `None` | pooled bulk CPM filter (assumes raw counts; same warning) |
| `keep_genes` | `None` | per-gene `np.bool_` mask aligned to `var_names` |
| **Effect size** ||
| `lfc_threshold` | `None` | τ, or a finite iterable of τ, in log2 units (0 ≤ τ ≤ 30); adds `tau=<±τ>_{p,Ueffect,padj}`. Not supported with `ALL_OTHERS` |
| `lfc_threshold_alt` | `("up", "down")` | which one-sided alternatives to emit per τ |
| `tau_star` | `None` | one-sided `p_dir` levels in the open interval (0, 1); emits a signed `tau*_p<level>` log2 shift per (target, gene). Not supported with `ALL_OTHERS` |
| `tau_star_iters` | `None` | bisection steps per level (default 20); validated even when `tau_star` is unset |
| `tau_star_se` | `False` | adds `tau*_lo_p0.025`, `tau*_hi_p0.025`, `tau*_se`; requires `tau_star`, and forces `0.5` into the level set |
| **Normalization** ||
| `cpm_normalize` | `False` | inline CPM scaling (skips an upfront `sc.pp.normalize_total`); does *not* mutate `adata.X` |
| `normalize_target_sum` | `None` | inline scanpy-compatible library-size normalization; pass a positive number or `"median"`; mutually exclusive with `cpm_normalize=True` |
| **Execution** ||
| `gpu_gene_chunk_size` | `None` | auto-pick from free GPU memory |
| `oom_recovery` | `True` | on CUDA OOM, halve the gene-chunk and retry (floor 64, or `chunk//2` if smaller); `False` = strict raise (benchmarking) |
| `densify_input` | `False` | in-memory group-label / `ALL_OTHERS` path only. There: materialized sparse → densified in place with a `UserWarning` (faster per-group slicing when host RAM permits); sparse view → raises; already-dense → no-op. `archive=`, and `adata=` with `reference=<AnnData>`, both raise. `cell_source=` **ignores** it, including when its pool is an `AnnData` |
| `release_gpu_memory` | `True` | return torch's (and cupy's) caching-allocator pools to the driver on exit, so a same-process caller can allocate afterwards |
| `stream_n_workers` | `16` | shard layout: decode-ahead workers (~14 GB host RAM each). Cell layout: the Rust gather's thread count (no extra host RAM) |
| `stream_prefetch` | `2` | shard layout: decode-ahead queue depth; `0` is the serial, lowest-RAM path. No effect on cell layout |
| **Output** ||
| `output_columns` | `None` | rename + select dict; raises `KeyError` on unknown keys |

## Design

See the module docstrings in `src/gpudge/` for the algorithm and design notes.
