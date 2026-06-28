# rapids-singlecell PR #636 (Wilcoxon refactor) vs gpudge — results

Benchmark of [scverse/rapids-singlecell#636](https://github.com/scverse/rapids-singlecell/pull/636)
("Wilcoxon refactor", branch `wilcoxon-refactor`, the unreleased 0.16.0-dev),
which reworks `tl.rank_genes_groups` Wilcoxon onto dedicated **CUDA kernels**
(CUB segmented radix sort) with a **host-streaming sparse path** and a new
**approximate `wilcoxon_binned`** method. gpudge `de()` is the reference. Both
tools run one-vs-reference (perturbation vs non-targeting), exact tie-corrected
asymptotic Wilcoxon, per-group BH FDR, on matched CPM(`target_sum=1e6`)+log1p
input, on the same H100 80 GB GPU.

Three perturbation-screen datasets, two cancer cell lines, at increasing scale:

- **CCL_1** — 36,346 cells × 18,533 genes, 82 perturbations
- **CCL_2 (2.06M-cell subset)** — 2,064,002 cells × 18,533 genes, 4,672 perturbations
- **CCL_2 (full, 5.54M cells)** — 5,542,923 cells × 18,533 genes, 5,096 groups (1 reference + 5,095 target genes)

## Why host-streaming matters

The released rapids-singlecell **0.14.1** exact Wilcoxon is **non-streaming**: it
moves the entire sparse matrix to the GPU at once. That scales ~2 s per
perturbation on small data and **OOMs outright** on the larger matrices — at
2.06M cells it tries to allocate the whole ~107.5 GB CSR on an 80 GB H100 and
fails. PR #636's **host-streaming** path keeps the CSR host-resident and streams
column sub-batches to the GPU, so peak VRAM is decoupled from cell count. This is
architecturally the same approach gpudge uses (gpudge streams host→GPU chunks
inside `de()`), which is what makes the comparison below a real head-to-head
rather than a fits/doesn't-fit story.

## The crossover (headline)

The faster tool changes with scale — the exact-Wilcoxon advantage moves from
rapids' device path → rapids' host-streaming → gpudge as the matrix grows:

| Dataset | Fastest exact path | DE-only speed | Peak VRAM | Why |
|---|---|---|---|---|
| CCL_1 (36k cells) | rapids #636 **device** | 0.50 s vs gpudge 1.39 s → **2.8× faster** | both fit | small matrix fits on-device; streaming's PCIe overhead isn't worth paying |
| CCL_2 (2.06M cells) | rapids #636 **host-stream** | 41.5 s vs gpudge 50.7 s → **1.22× faster** | 55.5 GB | device path OOMs (107 GB > 80 GB); host-streaming fits *and* leads |
| CCL_2 (5.54M cells) | **gpudge** | 72.8 s vs rapids 94.3 s → **1.30× faster** | gpudge **25.7** vs rapids 58.8 GiB (**½**) | both stream from host & fit; gpudge pulls ahead, at half the VRAM |

p-values are **bit-identical** across all three (Pearson 1.000000); log2FC
Pearson ≈ 0.9999. Details below.

## CCL_1 (36,346 cells, 82 perturbations)

gpudge DE = **1.385 s** (the old rapids 0.14.1 exact path here was 162.89 s; PR
#636 turns that into sub-second).

| rapids #636 path | DE (s) | norm+DE (s) | vs gpudge DE | p-value Pearson |
|---|---|---|---|---|
| exact, device (whole matrix on GPU) | **0.495** | 2.60 | **2.80× faster** | 1.000000 |
| binned, GPU (approximate) | 0.809 | 2.45 | 1.71× faster | 0.999982 |
| exact, host-streaming | 2.02 | 4.76 | 1.46× *slower* | 1.000000 |

On a matrix this small the whole thing fits on the GPU, so the device path is
fastest — **2.8× faster than gpudge**. Host-streaming carries PCIe-transfer
overhead that dominates here (it pays off at scale — see below).

## CCL_2, 2.06M-cell subset (4,672 perturbations)

gpudge DE = **50.68 s**; old rapids 0.14.1 exact = **OOM** (107.5 GB CSR > 80 GB).

| rapids #636 path | DE (s) | norm+DE (s) | peak VRAM | p-value Pearson | result |
|---|---|---|---|---|---|
| **exact, host-streaming** | **41.53** | 91.05 | **55.5 GB** | 1.000000 (80.2 M pairs) | ✅ fits, **1.22× faster than gpudge** |
| binned, dask (approximate) | — | — | — | — | ❌ OOM |

Where 0.14.1 OOMed, #636's exact host-streaming runs in **41.5 s** at a bounded
**55.5 GB** peak and is **faster than gpudge** on the DE stage, with bit-perfect
p-values. The approximate **binned path OOMs even via dask cell-streaming**: the
blow-up is in the *group* dimension (4,672 groups), not cells, so smaller cell
blocks don't help. Binned is excellent for low-cardinality (clustering) DE — see
CCL_1 — but unsuited to large perturbation screens.

## CCL_2, full 5.54M cells (5,096 groups)

Both tools complete end-to-end. The matched normalize here uses the **copy-free
in-place CPM+log1p** path (`--normalize inplace`) for both tools — at ~36 billion
nonzeros the scanpy copy path would blow the host-RAM budget; in-place keeps the
peak bounded (~541 GiB) and the input byte-identical across tools.

| tool | load (s) | norm (s) | DE (s) | peak VRAM | peak host RAM | OOM? |
|---|---|---|---|---|---|---|
| **gpudge** | 53.3 | 127.2 | **72.8** | **25.7 GiB** | 541.5 GiB | no |
| rapids #636 exact host | 52.9 | 127.2 | 94.3 | 58.8 GiB | 541.4 GiB | no |

- **DE only:** gpudge 72.8 s vs rapids 94.3 s → **gpudge 1.30× faster**
- **norm + DE:** 200.0 s vs 221.5 s → 1.11×
- **end-to-end:** 253.3 s vs 274.4 s

At the largest scale rapids #636's host-streaming **does not OOM** (unlike 0.14.1
at 2.06M) — it fits in ~58.8 GiB. But gpudge is **1.30× faster on DE and uses
less than half the VRAM** (25.7 vs 58.8 GiB). gpudge's peak is from
`torch.max_memory_allocated`; rapids' is from a device-level VRAM poller
(`memGetInfo`), since #636's kernels allocate outside cupy's default pool.

## Correctness (vs gpudge / scipy asymptotic Wilcoxon)

Exact `wilcoxon` is **bit-perfect on p-values**: Pearson and Spearman =
**1.000000** at every scale (CCL_1; CCL_2 over 80.2 M matched
gene×group pairs at 2.06M and 87.9 M at 5.54M). log2FC Pearson ≈ **0.9999**
(0.999904 at 5.54M). The binned approximation is p-value Pearson 0.99998.

`p_adj` Spearman is lower (0.49 CCL_1 / 0.78 at 5.54M) — the known
**BH-denominator difference**: gpudge runs per-group BH over its CPM-cell-filtered
gene set, rapids over all genes. Raw p-values match exactly, and `p_adj` Pearson
stays high (0.987–0.998). Not a regression — identical behavior across releases.

Coverage: **100% of gpudge's rows match** rapids; rapids reports a few percent
more rows (6.9% at 5.54M, ~14% at CCL_1) because gpudge's
`filter_gene_min_mean_value=0.0` still drops genes with zero mean expression in a
comparison (statistically trivial), so its output is a strict subset.

## Bottom line

PR #636 closes the gap the non-streaming 0.14.1 exposed: it kills the
~2 s/perturbation scaling (→ sub-second on CCL_1) **and** removes the OOM
via an exact host-streaming path that is correct and VRAM-bounded. On these
datasets rapids #636 is **faster than gpudge up to ~2M cells** (2.8× on the small
matrix, 1.22× at 2.06M), and gpudge is **faster at 5.54M (1.30×) using half the
VRAM**. The one caveat is the approximate binned path, which doesn't fit
high-cardinality perturbation screens. Both tools produce bit-identical
p-values throughout.

## Reproduce

See [`README.md`](README.md) for environment setup. With both envs ready:

```bash
# gpudge reference (use --normalize inplace for very large matrices)
python run_gpudge.py --data <screen.h5ad> --groupby <col> --reference <ref> \
    --collapse-reference-prefix non-targeting --tag <ds>

# rapids #636 — the three paths
M="micromamba run -n <rapids-env> python"
$M run_rapids.py --data <screen.h5ad> --groupby <col> --reference <ref> \
    --collapse-reference-prefix non-targeting --method wilcoxon --transfer device
$M run_rapids.py ... --method wilcoxon --transfer host
$M run_rapids.py ... --method wilcoxon_binned --dask

# compare each vs gpudge
python compare.py --gpudge-tag <ds> --rapids-tag <ds>_wilcoxon_host
```
