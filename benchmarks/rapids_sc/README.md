# gpudge vs rapids-singlecell `rank_genes_groups` benchmark

Speed + correctness head-to-head between gpudge's `de()` and
rapids-singlecell's GPU Wilcoxon `rank_genes_groups`, one-vs-reference (e.g. a
perturbation vs non-targeting controls). Results and the full analysis are in
[`RESULTS.md`](RESULTS.md).

The benchmark targets [scverse/rapids-singlecell#636](https://github.com/scverse/rapids-singlecell/pull/636)
("Wilcoxon refactor", the unreleased 0.16.0-dev), which reworks the Wilcoxon
test onto dedicated CUDA kernels with a **host-streaming** sparse path and an
approximate **`wilcoxon_binned`** method.

## Dataset-agnostic

Nothing here is tied to a specific dataset. Each runner takes any raw-count
h5ad plus the obs column to group on and the reference label:

```
--data <path.h5ad> --groupby <obs column> --reference <label>
```

If the control is split across many guides (`non-targeting-1`, `-2`, …), pool
them into the reference with `--collapse-reference-prefix non-targeting`.

## Setup

Two environments, because the tools have incompatible CUDA stacks:

- **gpudge** (its own venv): `pip install "gpudge[fast]"` (the `[fast]` extra
  brings in the numba CSR kernel) plus a CUDA build of PyTorch.
- **rapids-singlecell #636** (a separate conda/micromamba env): install the
  branch's CI-built `rapids-singlecell-cu12` wheel from
  [PR #636](https://github.com/scverse/rapids-singlecell/pull/636) on top of a
  **recent RAPIDS** (RAPIDS ≥ 26.06 / cuML, cuDF, cuPy for CUDA 12). The CI
  wheel's kernels link a recent `librmm-cu12`; older RAPIDS fails to load them
  with an `undefined symbol` error. Use the same H100-class GPU for both tools.

## Run

```bash
# 1. gpudge (in its venv), --tag names all outputs
python run_gpudge.py --data screen.h5ad --groupby target_gene \
    --reference ntc --collapse-reference-prefix non-targeting --tag screen

# 2. rapids #636 (in its env) — the three paths PR #636 adds
M="micromamba run -n <rapids-env> python"
$M run_rapids.py --data screen.h5ad --groupby target_gene --reference ntc \
    --collapse-reference-prefix non-targeting --method wilcoxon --transfer device   # whole matrix on GPU
$M run_rapids.py ... --method wilcoxon --transfer host                              # host-streaming
$M run_rapids.py ... --method wilcoxon_binned --dask                               # approximate, cell-streamed

# 3. compare (CPU) — pass the gpudge tag and the rapids tag
python compare.py --gpudge-tag screen --rapids-tag screen_wilcoxon_host
```

`run_rapids.py` derives a tag of `<data-stem>_<method>_<transfer>` (e.g.
`screen_wilcoxon_host`) unless you pass `--tag`. Outputs land in `results/`
(git-ignored): `<tag>_{gpudge,rapids}.parquet`, the matching `_timing.json`, and
the combined `<gp>_vs_<rp>.json`.

## Matched input & fairness

- Both tools get the **same CPM (`target_sum=1e6`) + log1p** input so the
  Wilcoxon statistic lines up. Pick `--normalize scanpy` (default) or, for
  matrices whose scanpy copy would OOM host RAM, the copy-free
  `--normalize inplace` — and pass the **same** choice to both runners.
- **DE-only timing** is the headline; it excludes disk load (identical path for
  both) but includes each tool's CPU→GPU transfer (inside `de_wall_sec` for
  gpudge and for rapids host-streaming). `norm+DE` is the secondary "compute
  from a CPU AnnData" view.
- gpudge's `de(filter_gene_min_mean_value=0.0)` drops only all-zero
  (gene,comparison) pairs (both target and reference mean == 0), so its gene set
  is a strict subset of rapids' `n_genes=all` (see RESULTS.md for the small
  coverage delta). Single GPU, single run — not a timing distribution.
