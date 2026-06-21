# GB10 TeaCache Extension

GB10-maintained vendored derivative of `feffy380/sd-webui-teacache`, packaged as a first-class owned extension for this A1111 fork.

TeaCache accelerates SDXL sampling by estimating when the UNet output can reuse cached residuals from nearby timesteps. The upstream implementation targets SDXL and includes fitted polynomial coefficients for SDXL-style models; without those coefficients the behavior is effectively FBCache-like.

## Provenance

- Upstream: https://github.com/feffy380/sd-webui-teacache
- Vendored from upstream commit: `a8cecf28cf77b608e4423d545d564264e0baa284`
- Upstream latest observed at adoption: 2025-08-04
- License: MIT, preserved in `LICENSE`
- Original copyright: Copyright (c) 2025 feffy380
- Upstream credits: official TeaCache repository (`ali-vilab/TeaCache`), TeaCache paper (`arXiv:2411.19108`), and SDXL compatibility work based on `chengzeyi/Comfy-WaveSpeed`.

## GB10 ownership doctrine

Treat this directory as first-class GB10 source:

- do not depend on a live external TeaCache checkout for runtime behavior
- keep local changes reviewable in this repository
- preserve the MIT license and upstream attribution
- keep the extension disabled by default until SDXL quality/performance benchmarking validates preferred thresholds for Schwi's workflows
- consider Flux behavior unsupported here unless this fork intentionally adopts and validates a Flux path later

## A1111 API argument order

The always-on script title is `TeaCache`. Positional API callers should preserve this argument order:

1. `enabled`
2. `threshold`
3. `max_consecutive`
4. `start`
5. `end`

Current defaults are conservative and disabled-by-default through the closed accordion:

- `enabled`: `False`
- `threshold`: `0.3`
- `max_consecutive`: `0`
- `start`: `0.3`
- `end`: `1.0`

## Local compatibility notes

- Uses this fork's `modules.headless_ui` shim instead of importing Gradio directly, so API/headless startup can discover the script without a UI-only dependency path.
- Restores the patched UNet forward method from an explicit `_openclaw_teacache_original_forward` attribute to reduce the chance of a stale patch after exception recovery.
- Keeps TeaCache state per sampling pass and records enabled settings in infotext only when the script is active.
