# Security policy

## Reporting a vulnerability

Please report privately, through GitHub's **[private vulnerability
reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability)**
— the *Security* tab of this repository, then *Report a vulnerability*. That
opens a draft advisory only the maintainers can see.

Please do not open a public issue for a suspected vulnerability.

What is useful in a report: what an attacker controls, what they achieve, and a
reproducer if you have one. gpudge is maintained by a small team alongside other
work, so expect an acknowledgement in days rather than hours. If a report turns
out to be a plain bug rather than a vulnerability we will say so and move it to
the public tracker, with your agreement.

## Supported versions

The latest release only. gpudge is pre-1.0 and there are no maintenance
branches: fixes land on `main` and go out in the next release rather than being
backported.

## What the threat model actually is

gpudge is a compute library. It does not open sockets, spawn processes, or
authenticate anyone. Realistically the interesting surface is **the input you
hand it**:

- an untrusted `.h5ad` or `cellstream` archive — parsing is done by `anndata`,
  `h5py` and `cellstream`, so a malicious file is mostly *their* attack surface,
  but a crash or a wild allocation reachable through gpudge's own slicing and
  chunk-sizing code is in scope here;
- `de(cell_source=…)`, which runs a callable you supply — that is by design, and
  passing an untrusted callable is equivalent to running untrusted code;
- resource exhaustion: gpudge deliberately sizes GPU chunks against available
  VRAM and recovers from OOM, so a hostile input shaped to defeat that sizing is
  a legitimate report.

Out of scope: that `de()` requires a CUDA GPU and raises without one; that
`densify_input=True` mutates `adata.X` in place and needs host RAM proportional
to the input (both documented); and anything that requires already being able to
run arbitrary code in the same process.

## Dependencies

Third-party licences and the environment a release was audited against are
recorded in [`docs/THIRD_PARTY_LICENSES.md`](docs/THIRD_PARTY_LICENSES.md). A
vulnerability in a dependency is best reported upstream first; tell us too, so
the floor can be raised here.
