# Third-party licenses

License audit of the packages `gpudge` depends on. **`gpudge` itself is
licensed `BSD-3-Clause`** (see `pyproject.toml`).

> **Audited 2026-06-09** against the resolved `.venv` (Python 3.12, the
> `cu126` PyTorch build). Versions below reflect that environment; re-audit
> after dependency bumps. Licenses were read from installed package metadata
> (`importlib.metadata`), not assumed.

## How to regenerate

```bash
.venv/bin/python - <<'PY'
import importlib.metadata as M
for d in sorted(M.distributions(), key=lambda d: (d.metadata.get("Name") or "").lower()):
    md = d.metadata
    name = md.get("Name")
    if not name:
        continue
    lic = (md.get("License-Expression")
           or "; ".join(dict.fromkeys(
               v.split("::")[-1].strip() for k, v in md.items()
               if k == "Classifier" and v.startswith("License ::")))
           or (md.get("License") or "").strip().splitlines()[0:1] or ["(none declared)"])
    print(f"{name}\t{d.version}\t{lic if isinstance(lic, str) else lic[0]}")
PY
```

## Declared dependencies (`pyproject.toml`)

| Package | Version | Extra | License | Notes |
|---|---|---|---|---|
| anndata | 0.12.16 | core | BSD-3-Clause | permissive |
| hdf5plugin | 6.0.0 | core | MIT (wrapper, ESRF) | bundled HDF5 filters carry their own licenses (BSD / Zlib / etc.); see `doc/information.rst` in the package |
| numpy | 2.4.6 | core | BSD-3-Clause | vendored components: 0BSD / MIT / Zlib / CC0-1.0 — all permissive |
| polars | 1.41.0 | core | MIT | permissive |
| pyarrow | 24.0.0 | core | Apache-2.0 | permissive |
| scipy | 1.17.1 | core | BSD-3-Clause | permissive |
| torch | 2.12.0+cu126 | core | BSD-3-Clause | PyTorch itself is BSD-3; the `cu126` build pulls in **proprietary** NVIDIA CUDA libraries — see below |
| pytest | 9.0.3 | dev | MIT | permissive |
| ruff | 0.15.14 | dev | MIT | permissive |
| scanpy | 1.12.1 | dev | BSD-3-Clause | permissive |
| numba | 0.65.1 | fast | BSD-2-Clause | permissive |
| **shardad** | 0.2.0 | streaming | **none declared** | Arc Institute private package; optional `streaming` extra only — see flag below |

## Transitive dependencies requiring attention

### NVIDIA CUDA runtime — proprietary

The `cu126` PyTorch build pulls these in, and they are **used at runtime**
(gpudge is GPU-only). They are **proprietary** (NVIDIA EULA), not open source:

| Package(s) | License |
|---|---|
| `nvidia-cublas` / `cuda-cupti` / `cuda-nvrtc` / `cuda-runtime` / `cufft` / `cufile` / `curand` / `cusolver` / `cusparse` / `cusparselt` / `nvjitlink` `-cu12` | NVIDIA Proprietary Software (EULA) |
| `nvidia-cudnn-cu12`, `nvidia-nccl-cu12`, `nvidia-nvshmem-cu12`, `cuda-bindings` | `LicenseRef-NVIDIA-Proprietary` |
| `nvidia-nvtx-cu12` | Apache-2.0 ✅ |
| `triton` | MIT ✅ |

### MPL-2.0 (weak, file-level copyleft)

Fine to use and redistribute **unmodified**; only triggers source-disclosure
obligations on files you modify:

- `certifi`, `tqdm` (MPL-2.0 ∧ MIT)
- `fast-array-utils`, `legacy-api-wrap`, `session-info2` (pulled in via the `scanpy` dev extra)

### Everything else

The remaining transitive packages (~60 of 84 installed) are all permissive —
MIT / BSD / Apache-2.0 / PSF.

## Compliance flags

1. **`shardad` declares no license.** Not in its installed metadata, its
   `pyproject.toml`, or as a `LICENSE` file in the repo. By default that means
   all rights reserved. It is Arc's own internal package and the *optional*
   `streaming` extra, so this is a non-issue for internal use — but it should
   be given an explicit license (BSD-3-Clause, to match gpudge) **before any
   public / PyPI release** that references it.

2. **NVIDIA CUDA libraries are proprietary.** gpudge's own source stays cleanly
   BSD-3-Clause, but a *deployed* gpudge environment includes proprietary NVIDIA
   components — standard for any CUDA PyTorch stack. This does not affect using
   gpudge; it only blocks redistributing the assembled binary environment under
   open terms.

**Net:** everything gpudge authored and declares is permissive (BSD / MIT /
Apache). The only open items are shardad's missing license (internal package)
and the inherent NVIDIA-proprietary CUDA runtime.
