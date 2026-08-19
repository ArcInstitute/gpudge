# Contributing to gpudge

Thanks for looking. gpudge is a small, deliberately narrow library — one public
entry point, `de()` — so most contributions are bug reports, test cases, and
documentation fixes rather than new features.

## What is and is not in this repository

`src/` is the library, `tests/` its suite, and `docs/` and `examples/` its
documentation. `benchmarks/` holds performance and comparison harnesses that
are not part of the installed package.

Some code comments cite design specs by label (`spec 3.2b`, `Semantics A`).
Those documents are not shipped with the package and may not be in your
checkout at all. Where such a label carries reasoning you need, that reasoning
is restated in the code — if you find one that is a bare pointer with nothing
behind it, that is a bug worth reporting.

## Setting up

Python **≥ 3.11**, and a **CUDA GPU** for anything that calls `de()`. There is
no CPU fallback; without a GPU it raises rather than running slowly.

```bash
git clone https://github.com/ArcInstitute/gpudge.git && cd gpudge
uv venv && uv pip install --torch-backend=cu126 -e ".[dev,fast]"
```

Use `uv pip install`, **not `uv sync`** — `uv sync` builds uv's universal lock,
which resolves every entry in `[tool.uv.sources]`, including the private
`shardad` source, even when you did not ask for the `streaming` extra; without
SSH access to that repository it fails with `Permission denied (publickey)`.
`uv pip install` resolves only the extras you name. `[fast]` adds numba, which
the fast CSR kernel needs; `[dev]` adds pytest, ruff, scanpy and pyyaml.

## Running the tests

```bash
uv run --no-sync pytest tests/                      # most of it needs no GPU
uv run --no-sync ruff check src/ tests/ examples/   # exactly what CI runs
```

`uv venv` creates `.venv` but does not activate it, so `uv run --no-sync` is
what makes a bare shell use it — these are the two commands CI runs verbatim.

Two things that surprise people:

- **`pytest -m needs_cuda` selects nothing and exits 5.** `needs_cuda` is a
  `skipif` decorator in `tests/conftest.py`, not a registered marker. Run the
  whole suite on a GPU host instead.
- **CI is CPU-only**, so the GPU bit-identity gates and the GPU-backed parity
  assertions never execute there. Green CI is necessary, not sufficient, for a
  change touching a GPU path — say in your PR whether you ran the suite on a
  CUDA host, and what it reported.

Three suites (`test_shard_stream.py`, `test_cell_stream.py`,
`test_inmem_external_ref_gpu.py`) call `importorskip("shardad")` at module
level, so without that optional dependency they are not collected at all — they
appear as 3 skips rather than as their real case count.

## The bar for tests

This repository has been bitten repeatedly by tests that could not fail: a
byte-identity gate that was silently a tolerance check, because polars'
`assert_frame_equal` defaults to `check_exact=False`; assertions with tolerances
four orders looser than the measured agreement; an oracle branch that had never
once executed and was wrong when it finally did. Six such gates were repaired in
a single release.

So the expectation for a test accompanying a fix is concrete:

> **Break the fix and watch the test go red.** Then put it back.

If a test cannot be made to fail by reverting the thing it covers, it is not
testing that thing. Saying so in the PR — "reverting X turns this red" — is the
most useful sentence you can write, and reviewers here will ask for it.

## Documentation that is enforced

Some prose is checked by tests, and changing it means changing the test too:

- `docs/tutorial.md`'s published numbers are recomputed by
  `tests/test_tutorial.py` from an independent SciPy oracle. Edit the transcript
  and the test fails.
- The committed tutorial dataset is pinned by sha256.

## Pull requests

Branch, open a PR, get CI green. Keep commits and PR descriptions factual about
what was verified and what was not — "I did not run the GPU suite" is a fine
thing to write, and much better than silence. If a reviewer's suggestion looks
wrong to you, say why rather than applying it; several suggestions in this
repository's history were technically reasoned and empirically false.

## Reporting a bug

Include the gpudge version, the Python and torch versions, the GPU, and the
shape of the input (`n_obs × n_vars`, number of groups, sparse or dense, dtype).
For a numerical disagreement the most useful report is a small reproducer plus
what you expected and why — a comparison against `scipy.stats.mannwhitneyu` with
`method="asymptotic"` is the reference gpudge is built to match.

Security issues go through [SECURITY.md](SECURITY.md), not the public tracker.
Conduct concerns go through [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
