# GB10 Incantations Extension

This is the GB10-owned vendored guidance extension, combining Incantations PAG/SEG/CFG-combiner behavior with Dynamic Thresholding / CFG-Fix for the GB10 A1111 image.

## Provenance

### Dynamic Thresholding / CFG-Fix provenance

- Upstream: https://github.com/mcmonkeyprojects/sd-dynamic-thresholding
- Vendored from local upstream checkout commit: `73e4e04565aa86237d66764ac58ffae1f7e40e48`
- License: MIT, preserved in `DYNAMIC-THRESHOLDING-LICENSE.txt`
- Note: the combined extension as a whole remains GPL-3.0-compatible because the Incantations code is GPL-3.0; Dynamic Thresholding sources retain their MIT notice.
- Upstream documentation was intentionally not retained as shipped docs because this repo now owns a trimmed A1111-only integration.

### Incantations provenance

- Upstream: https://github.com/v0xie/sd-webui-incantations
- Vendored from local upstream-derived checkout commit: `769a55cade195c3c0718c41930adff9865052aac`
- Local source branch at adoption time: `gb10-local`
- License: GPL-3.0, preserved in `LICENSE`
- Upstream documentation was intentionally not retained as shipped docs because removed legacy features would make it misleading.
- Upstream watch note: even though the upstream appears effectively stale for GB10 runtime needs, periodically check `v0xie/sd-webui-incantations` for relevant issue reports or fixes before/while changing PAG, SEG, CFG-combiner, or hook lifecycle behavior.

## Source map

- `scripts/dynamic_thresholding.py` and `dynthres_core.py`: A1111 Dynamic Thresholding / CFG-Fix source for k-diffusion samplers (the timestep samplers DDIM, DDIM CFG++, PLMS and UniPC are rejected with an error). It wraps the hires pass's `hr_sampler_name` as well (it used to run without Dynamic Thresholding when one was set), and its renamed sampler is a `_replace` copy that keeps the sampler data's class and fields (a Multi chain's `MultiSamplerData.total_steps`). ComfyUI/SwarmUI entrypoints from the old standalone extension were removed.
- `scripts/pag.py` and `scripts/smoothed_energy_guidance.py`: Incantations guidance source with GB10 lifecycle fixes.
- `scripts/cfg_combiner.py`: GB10-owned CFG composition glue for PAG and CFG-Fix coexistence.
- `scripts/incantation_base.py`: GB10-trimmed A1111 entrypoint that exposes only the supported combined guidance stack.
- `scripts/ui_wrapper.py`: the submodule base class (per-batch callback and hook bookkeeping) and shared X/Y/Z and conditioning helpers; `scripts/incant_utils/`: module field/lookup helpers and the timing records behind the API's `openclaw_extension_timings`.
- `tests/`: CPU unit tests with A1111 stubbed; run them in their own pytest process.
- Removed abandoned A1111-discovered Incantations scripts: legacy prompt incanting, S-CFG, T2I-Zero, and attention-map saving. They were not part of the GB10 active guidance path and still used stale callback cleanup / debug code.
- Removed PAG's CFG Scheduler ("CFG Interval": noise-interval CFG and CFG weight schedules). Its four script args stay as placeholders, see below.
- PAG's perturbed pass replaces the middle-block self-attention map with the identity, so the layer outputs `to_out(to_v(x))` as in the paper, diffusers' `PAGIdentitySelfAttnProcessor` and ComfyUI. Upstream Incantations (and this tree until 2026-10-09) returned `to_v(x)` and skipped the output projection; the fix changes PAG images.
- PAG and SEG Start/End Step compare the sampler step of the denoiser call in progress (`ui_wrapper.sampler_step`: the denoiser's call count over its `total_steps`, as the core measures progress). They read `state.sampling_step` before, which lags one step behind the model call, so every interval started and ended one step late. The default 0-150 interval is unaffected.
- SEG blurs the uncond rows of every UNet call of the CFG batch, located by the call's row offset in the batch: one call, `batch_cond_uncond` off, and prompt/negative prompt of different token lengths (separate cond and uncond calls) all blur the same rows. Before, SEG switched itself off for the whole request when `batch_cond_uncond` was off and skipped the steps whose batch was split, silently. A cond-only call (skip-uncond: NGMS, skip early CFG) has no uncond rows to blur.
- SEG's attention grid comes from the shape of the latent being denoised, not `p.height`/`p.width`, so a cropped hires pass blurs on its real grid (it could be transposed); first passes are bit-identical.
- PAG SANF chooses per element between CFG and PAG guidance. With the core combiner each cond's raw CFG term competes with its PAG term. With Dynamic Thresholding (a replaced combiner) the CFG candidate is the thresholded guidance (the combiner's result minus uncond) and the PAG candidate is the image's summed PAG terms; the infotext records `PAG SANF note: PAG SANF blends the dynamically thresholded CFG`. Until 2026-10-10 SANF discarded Dynamic Thresholding's result, so DT never applied with SANF on although the infotext said `Dynamic thresholding enabled: True`; this changes those images.
- SEG fails the request instead of blurring wrong rows when: an extension replaces the CFG denoiser's or inner model's forward or `sd_model.apply_model` (Tiled Diffusion: MultiDiffusion, DemoFusion, Mixture of Diffusers; the CUDA-graph bypass uses the same test); a UNet call is not a row slice of the CFG batch; hypertile tiles the middle block; the CFG batch is not [cond, uncond] (InstructPix2Pix image CFG).


## A1111 API argument order

Keep the `incantations` always-on script argument order stable unless a caller migration is planned. The UI labels and `elem_id`s may be clarified, but positional API callers depend on this order:

1. `seg_active`
2. `seg_blur_sigma`
3. `seg_start_step`
4. `seg_end_step`
5. `pag_active`
6. `pag_scale`
7. `pag_start_step`
8. `pag_end_step`
9. `cfg_interval_enable`
10. `cfg_interval_schedule`
11. `cfg_interval_low`
12. `cfg_interval_high`
13. `pag_sanf`

This ordering intentionally differs from the PAG UI visual order because `pag_sanf` was appended last historically. Preserve behavior over cosmetic ordering.

Arguments 9-12 belonged to the removed CFG Scheduler and are kept only so `pag_sanf` keeps its position (the controller sends `false, "Constant", 0.0, 100.0` there). `cfg_interval_enable` must be false: true fails the request, also when it comes from the infotext of an image made with the scheduler. The other three are ignored.

## GB10 ownership doctrine

The upstream extension appears effectively abandoned for our runtime needs, while PAG/SEG/CFG-combiner and CFG-Fix behavior are now quality-critical for this image. Treat this directory as first-class GB10 source:

- do not depend on a live external extension checkout for runtime behavior
- do not load a separate `sd-dynamic-thresholding` extension alongside this combined owned extension
- do not restore old upstream README/ComfyUI/SwarmUI/package-manager surfaces unless this repo actually supports them again
- keep local changes reviewable in this repository
- preserve GPL-3.0 notices and upstream provenance
- prefer conservative, testable fixes before changing guidance math
- audit hook cleanup and CFG-combiner interactions carefully because stale hooks or wrapper-order drift can materially affect image quality
- keep `scripts/cfg_combiner.py` compatible with CFG-Fix / Dynamic Thresholding by delegating the base CFG result to the captured original `combine_denoised` callable before adding PAG guidance
- avoid reintroducing abandoned patch-stack cleanup machinery in PAG/SEG; hook cleanup should be explicit and local
- do not add experimental scripts under `scripts/` unless they are actively maintained; A1111 auto-discovers them
