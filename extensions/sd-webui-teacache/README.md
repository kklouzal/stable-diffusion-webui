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
- `threshold`: `0.25`
- `max_consecutive`: `4`
- `start`: `0.35`
- `end`: `0.90`

## Local compatibility notes

- Uses this fork's `modules.headless_ui` shim instead of importing Gradio directly, so API/headless startup can discover the script without a UI-only dependency path.
- Restores the patched UNet forward method from an explicit `_openclaw_teacache_original_forward` attribute to reduce the chance of a stale patch after exception recovery. Restoring only happens while TeaCache's patch is the live `unet.forward`: when a failed generation left the patch installed and another hook (ControlNet) wrapped above it, the patch stays as a pass-through until that hook restores its baseline, so TeaCache never drops another owner's wrapper.
- Keeps TeaCache state per sampling pass and records enabled settings in infotext only when the script is active.
- Keeps cached residuals, refresh decisions, accumulated distances and the `max_consecutive` hit counter per UNet call lane: the call signature (shape/dtype/device of the hidden state, timesteps, context and vector conditioning) plus the call's ordinal among that denoiser step's calls with the same signature. Conditional, unconditional, batch, dtype and shape changes never share residuals; a step layout that alternates (NGMS dropping the uncond on every other step) keeps one lane per layout instead of missing on every call; every lane, including PAG's hidden pass, is bounded by `max_consecutive`, and lanes with identical inputs refresh on the same steps.
- Refreshes a lane whenever its cross-attention context or vector conditioning differs from the conditioning its cached residual was computed with (prompt editing, lanes shifted by skipped calls). SDXL's first block never sees the context, so the first-block distance alone cannot notice such changes. UNet-internal hook state that changes without changing the inputs (for example a SEG/PAG step interval boundary) is not part of the cache key.
- Disables residual reuse for non-SDXL models and masked/inpaint denoising paths; those modes need empirical validation before enabling because partial-latent blending and unsupported model coefficients can amplify quality drift.
- Rescales the first-block distance with upstream's only SDXL polynomial, fit on NoobAI-XL vpred v1.0. No fit exists for eps-prediction SDXL checkpoints (production's), so the threshold is effectively calibrated against the v-pred fit; refitting needs a reproducible GPU calibration run and has not been done. Every pass that can reuse residuals records `TeaCache rescale: NoobAI-XL v-pred fit` in infotext.
- Measures the `start`/`end` window in sampler steps of the full schedule: each denoiser call maps to its sampler step through the CFG denoiser's call count (`step * steps // total_steps`, the same mapping as Incantations' `sampler_step`), so second-order samplers (two calls per step) and "Multi" chains keep the requested window. img2img and hires passes start at their first scheduled step.
- Uses conservative quality-preserving defaults (`threshold=0.25`, `max_consecutive=4`, `start=0.35`, `end=0.90`) to avoid early composition and late detail/final-cleanup reuse by default.
