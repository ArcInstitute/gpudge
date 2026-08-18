# gpudge quickstart

A runnable, roughly thirty-second tour: load a real CRISPRi screen, run one differential
expression call, read the result, and check the answer is right.

Everything below is in [`examples/quickstart.py`](../examples/quickstart.py). Run it
**from a clone, at the repository root** — the wheel packages `src/gpudge` only, so
neither the example nor its data comes with a `pip install`:

```bash
git clone https://github.com/ArcInstitute/gpudge.git && cd gpudge
uv venv && uv pip install --torch-backend=cu126 -e ".[fast]"

.venv/bin/python examples/quickstart.py     # or: uv run --no-sync python examples/…
```

Every path below is relative to that repository root.

> **You need a CUDA GPU.** gpudge is GPU-only — there is no CPU fallback, not even for a
> dataset this small. Without one, `de()` raises `RuntimeError: gpudge requires a CUDA
> GPU`. See the [README's Install section](../README.md#install) for the pin; the core
> library has no private dependencies.

---

## The data

The example dataset is committed with the repo, so there is nothing to download:
`docs/data/H1-VCC-2025-training.h5ad`, about 5 MB.

It is a deterministic subset of the Virtual Cell Challenge 2025 training data from the
[Arc Virtual Cell Atlas](https://arcinstitute.org/tools/virtualcellatlas), which is
released under **CC0 1.0** (public domain) — **600 cells** × **1000 genes** of raw
integer counts, six groups of 100: a `non-targeting` control pool and five CRISPRi
perturbations. The perturbation label lives in `obs["target_gene"]`. Full provenance,
licence and checksum: [`docs/data/README.md`](data/README.md).

```python
import anndata as ad

adata = ad.read_h5ad("docs/data/H1-VCC-2025-training.h5ad")
print(adata.shape)                                # (600, 1000)
print(adata.obs["target_gene"].value_counts())
```

which the quickstart formats as:

```
             MED12   100 cells
               SRC   100 cells
             STAT1   100 cells
              TET1   100 cells
            TMSB4X   100 cells
     non-targeting   100 cells  <- control
```

[`docs/make_tutorial_data.py`](make_tutorial_data.py) is the recipe: running it against
the source reproduces the same logical subset (the h5ad's exact bytes also depend on your
anndata/h5py versions). You do not need to run it. It is
the same subset the cell_eval2 tutorial uses, so the two tools can be followed against
identical cells.

## Your first run

One call tests every perturbation group against the reference. `de()` does iterate the
groups, but it loads and sorts the *reference* once per gene chunk and then reuses it for
every group in that chunk — so you pay the expensive part once rather than once per
perturbation, which is what a per-perturbation DE call would do. That is where the
measured speedups in the [README](../README.md#performance) partly come from — one
contributor among several, not the whole explanation. It is not an asymptotic claim
either: the output still grows as perturbations x genes.

```python
import gpudge

res = gpudge.de(
    adata,
    groupby="target_gene",
    reference="non-targeting",
    cpm_normalize=True,
)
```

`cpm_normalize=True` scales each cell to 1e6 counts **on the GPU, inline**, and does not
mutate `adata.X`. Pass raw counts when you use it — see
[Normalization](#normalization-pick-exactly-one) below, which is the easiest thing to get
wrong here.

## Reading the result

`de()` returns a [polars](https://pola.rs) DataFrame in long format: one row per
(perturbation, gene). Here that is 5 × 1000 = **5000 rows**, ten columns:

```
result: polars DataFrame, 5000 rows x 10 columns
  target, feature, target_mean, ref_mean, target_ncells, ref_ncells,
  log2_fold_change, p_value, Ueffect, p_adj
```

| column | what it is |
|---|---|
| `target`, `feature` | the perturbation and the gene |
| `target_mean`, `ref_mean` | group means **in the units actually analysed** — CPM here, because `cpm_normalize=True` scaled the input |
| `target_ncells`, `ref_ncells` | cells behind each mean |
| `log2_fold_change` | `log2((target_mean + epsilon) / (ref_mean + epsilon))`, `epsilon=1e-9` by default — an *effect on the mean* |
| `p_value` | **two-sided** Mann–Whitney U, asymptotic, with tie and continuity corrections (matches `scipy.stats.mannwhitneyu(method="asymptotic")`) |
| `Ueffect` | signed rank-biserial correlation (Cliff's δ) = `2A − 1` ∈ [−1, 1], where `A` is the probability of superiority |
| `p_adj` | Benjamini–Hochberg, **per perturbation** — each perturbation is its own family |

`log2_fold_change` and `p_value` measure different things — a ratio of means and a rank
statistic — so they can disagree, and on skewed or zero-heavy genes they sometimes
flatly contradict each other. The p-value is two-sided and so carries no direction at
all; `Ueffect` is the signed effect size on the same rank axis the test uses, so read the
two together.

Sorting by `p_adj` shows the strongest hits. Note that `p_adj` was computed **within**
each perturbation, so this ordering is descriptive — it does not give you FDR control
across all 5000 rows at once:

```python
print(res.sort("p_adj").head(5))
```

```
          target        gene    log2FC       p_adj
          TMSB4X      TMSB4X     -4.91    2.56e-31
           MED12       PODXL     -0.52    2.45e-14
           MED12      DNMT3B     -0.50    1.86e-12
           MED12       TERF1     -0.42    7.32e-11
           MED12        CD24     -0.33    2.32e-10
```

## Did it work?

Before reading biology out of a screen, check the one thing you already have an
expectation for: **knocking down a gene should reduce that gene's own expression.**

`TMSB4X` is both a perturbation here and one of the 1000 genes, so it is directly
checkable (its group pools two paired-guide constructs):

```
on-target check -- the TMSB4X guides should knock down TMSB4X:
  log2FC = -4.91  (0.033x control)
  p_adj  = 2.56e-31
  rank   = 1 of 1000 genes for that perturbation
```

A 30-fold knockdown, and the smallest `p_adj` in the run. That is a **positive
control passing**, which is worth having — but keep its weight straight: it is one
perturbation on one dataset. A *failed* on-target check does not by itself implicate
gpudge (an ineffective guide, low basal expression, heavy dropout, or a guide-assignment
problem all produce the same symptom), and a *passing* one does not validate the other
perturbations. It tells you the plumbing works end to end.

(The other four targets — `STAT1`, `MED12`, `TET1`, `SRC` — did not survive the subset's
1000-gene cut, so the same check cannot be run on them here. On a full dataset, run it on
every perturbation you can.)

## Normalization — pick exactly one

The single most common way to get a wrong answer. Choose one and only one:

1. **Normalize beforehand** yourself and pass the result, leaving
   `cpm_normalize=False` — e.g. `scanpy.pp.normalize_total`, though scanpy is not a
   gpudge runtime dependency and `[fast]` does not install it; or
2. **`cpm_normalize=True`** — inline CPM on the GPU, as above; or
3. **`normalize_target_sum=<number | "median">`** — inline library-size normalization
   with scanpy `normalize_total` parity.

`cpm_normalize` and `normalize_target_sum` are mutually exclusive, and both **assume raw
counts**. Passing either on data you already normalized double-normalizes it silently.

## Where to go next

- The [README's API table](../README.md#api) lists all 28 keyword parameters.
- **Filtering**: the `filter_gene_*` family is opt-in and AND-combined.
- **One-vs-rest**: `reference=gpudge.ALL_OTHERS` instead of a control label.
- **Effect-size floors**: `lfc_threshold=` tests against a real floor rather than against
  zero; `tau_star=` reports the shift the rank test estimates. Both are in log2 units.
- **Larger-than-RAM data**: `de(archive=…)` streams a shardad archive, and
  `de(cell_source=…)` takes cells from a callable you write.
- The `de()` docstring in `src/gpudge/__init__.py` is the reference for all of it.
