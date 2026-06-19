# GB10 A1111 latest math/logic generation-quality audit

Started: 2026-06-19 UTC
Host/repo/branch: GB10 `/home/kklouzal/stable-diffusion-webui` branch `latest`
Starting HEAD: e67054caa89dd79fe84613b828321e729f69b5d0
Scope: function-by-function audit for math errors and logical inconsistencies that could hinder final generation quality.

## Audit policy
- Prioritize sampling, conditioning, attention/UNet call paths, CFG/NEG, hires/img2img/inpaint, precision/caching/replay, mask/latent/image geometry, scheduler/sampler integration, prompt shape handling, model loading/precision, generation-altering extensions, and API parameter handling.
- For each coherent defect: verify locally and against authoritative semantics where needed, patch narrowly, validate, commit, and record here.
- Preserve GB10 intentional CUDA graph behavior: full-window SEG replay may be enabled by `OPENCLAW_CUDA_GRAPH_ALLOW_SEG=1`; masks/inpaint bypass replay.

## Current status
- Initial repo state: clean `latest` at `e67054caa89dd79fe84613b828321e729f69b5d0`, tracking `origin/latest`.
- Next unchecked scope: `modules/processing.py` top-level helpers and processing classes.

## Checked scope

## Findings and fixes

## Validation log

## References consulted

## Remaining high-priority scope
- `modules/processing.py`
- `modules/sd_samplers*.py`
- `modules/prompt_parser.py`
- `modules/openclaw_cuda_graphs.py`
- `modules/masking.py`, `modules/images.py`, `modules/img2img.py`, `modules/txt2img.py`
- precision/cache modules (`mxfp8*`, `nvfp4*`, `cache.py`)
- generation-altering extensions under `extensions/`

### 2026-06-19 pass 1 - CFG denoiser skip-uncond conditioning alignment
Checked:
- `modules/sd_samplers_cfg_denoiser.py`: `catenate_conds`, `subscript_cond`, `pad_cond`, `CFGDenoiser.combine_denoised`, `combine_denoised_for_edit_model`, `get_pred_x0`, `update_inner_model`, `run_inner_model`, `pad_cond_uncond`, `pad_cond_uncond_v0`, and the high-risk first half of `CFGDenoiser.forward` through skip-uncond input assembly.
- `modules/openclaw_cuda_graphs.py`: `_env_flag`, `_allow_seg_graphs`, `status`, `set_enabled`, `clear`, signature helpers, SEG graph-key helpers, `_graph_denoiser_bypass_reason`, and `run`; no defect found in this pass. Mask/inpaint bypass and explicit full-window SEG opt-in match GB10 policy.
- `modules/prompt_parser.py`: schedule parsing/reconstruction functions through `reconstruct_multicond_batch`; no defect found in this pass. Prompt schedule step math preserves existing A1111 hires scheduling semantics.

Finding:
- In `CFGDenoiser.forward`, when `skip_uncond` was true, `x_in` and `sigma_in` were trimmed to remove the uncond branch but `image_cond_in` was not. The next inner-model call could therefore receive text/noise batches shorter than concat/image conditioning. This is a logical shape inconsistency on image-conditioning paths and CFG callbacks; it can error or feed mismatched conditioning and harm generation quality.

Fix:
- Trim `image_cond_in` alongside `x_in` and `sigma_in` in the skip-uncond branch.

Validation:
- `python3 -m py_compile modules/sd_samplers_cfg_denoiser.py`
- Dependency-free static harness verified the branch contains the coupled `x_in`/`sigma_in`/`image_cond_in` trim.

References:
- Local A1111 CFG denoiser implementation and callback contract in `modules/script_callbacks.py` (`CFGDenoiserParams` carries `x`, `image_cond`, and `sigma` as coupled inner-model inputs).
- Official PyTorch tensor concatenation/indexing semantics: tensors passed to model branches must agree on leading batch dimension for downstream operations.

Commits:
- Pending at time of ledger entry.

Next unchecked scope:
- Continue `CFGDenoiser.forward` after inner-model output assembly, then `modules/sd_samplers_kdiffusion.py` sampler schedule/noise paths.

### 2026-06-19 pass 2 - samplers, processing geometry, incantations guidance
Checked:
- `modules/sd_samplers_cfg_denoiser.py`: remainder of `CFGDenoiser.forward`; no additional defect found. Output assembly, fake-uncond replacement, preview latent selection, mask blending, and callback ordering are internally consistent after commit `9b566e48620009087edcacdce2804cd9b42bd3ad`.
- `modules/sd_samplers_kdiffusion.py`: `CFGDenoiserKDiffusion.inner_model`, `KDiffusionSampler.__init__`, `get_sigmas`, `sample_img2img`, and `sample`; no defect found. Sigma list handling, penultimate discard, SGM noise scaling, img2img `t_enc`, Brownian noise setup, and sampler extra args match local k-diffusion/A1111 contracts.
- `modules/sd_samplers_common.py`: `SamplerData.total_steps`, `setup_img2img_steps`, latent preview encode/decode helpers, `apply_refiner`; no defect found in checked math paths.
- `modules/sd_schedulers.py`: scheduler helpers through `beta_scheduler`; no defect found. Step-count validation, appended terminal zero, timestep-to-sigma conversion, and AYS/Beta/Karras-style schedule call shapes are coherent.
- `modules/processing.py`: top-level color/overlay/mask/cache helpers, text/image conditioning helpers, prompt setup/conditioning cache, hires target resolution and `sample_hr_pass`, img2img init/cache/mask/latent setup and sampling; no new defect found. Mask/latent broadcasts and cache key invalidators include geometry, VAE, checkpoint, mask, dtype, resize, and inpainting weight inputs.
- `modules/masking.py`: `get_crop_region_v2`, compatibility `get_crop_region`, `expand_crop_region`, `fill`; no defect found.
- `modules/images.py`: `resize_image` geometry for modes 0/1/2; no defect found. Verified PIL accepts the edge-fill zero-width/zero-height source boxes used for mode 2 border extension.
- `extensions/sd-webui-incantations/dynthres_core.py`: `DynThresh` schedule/statistics paths; no defect found in this pass. fp32 statistics and safe denominators protect fp16/bf16 reduction math.
- `extensions/sd-webui-incantations/scripts/cfg_combiner.py`: SANF blur/blend, wrapper lifecycle, callback registration, combine delegation paths; no defect found.
- `extensions/sd-webui-incantations/scripts/pag.py`: PAG hidden-pass batching helpers, condition padding, noise-level/index helpers, CFG schedules; no defect found.
- `extensions/sd-webui-incantations/scripts/smoothed_energy_guidance.py`: SEG state/timing helpers and `_blur_seg_cond_queries`; no defect found. Full-window SEG replay policy remains preserved in `modules/openclaw_cuda_graphs.py`.

Validation:
- `docker run --rm -v /home/kklouzal/stable-diffusion-webui:/work:ro -w /work local/gb10-a1111:latest python extensions/sd-webui-incantations/tests/test_guidance_core.py` -> 29 tests OK.
- Host `python3` PIL check: zero-height and zero-width `Image.resize(..., box=...)` edge boxes both returned expected sizes.
- Host `python3 -m pytest -q extensions/sd-webui-incantations/tests/test_guidance_core.py` could not run because host Python lacks `torch`; no host installs performed.

References:
- Local k-diffusion sampler signatures via `inspect.signature(self.func)` and checked call sites.
- PyTorch broadcasting/indexing and tensor batch-dimension semantics as exercised by local unit tests.
- PIL `Image.resize` behavior for zero-width/zero-height source boxes was verified locally.

Commits:
- `9b566e48620009087edcacdce2804cd9b42bd3ad` - Fix skip-CFG image conditioning alignment.

Next unchecked scope:
- `modules/sd_samplers_timesteps.py`, `modules/sd_samplers_timesteps_impl.py`, `modules/sd_samplers_compvis.py`, `modules/sd_samplers_lcm.py`, then precision/model-cache modules.

### 2026-06-19 pass 3 - timestep/LCM samplers and precision caches
Checked:
- `modules/sd_samplers_timesteps.py`: `CompVisTimestepsDenoiser`, `CompVisTimestepsVDenoiser`, `CFGDenoiserTimesteps`, `_make_timesteps`, `CompVisSampler.get_timesteps`, `sample_img2img`, and `sample`; no defect found. V-prediction epsilon conversion and x0 prediction use local alphas consistently.
- `modules/sd_samplers_timesteps_impl.py`: `_step_value`, `_model_timestep`, `_ddim_sigmas`, `ddim`, `ddim_cfgpp`, `plms`, `UniPCCFG`, `unipc`; no defect found. The `len(timesteps)-1` DDIM/PLMS loop matches current upstream A1111 implementation, so it is recorded as intended behavior rather than patched.
- `modules/sd_samplers_lcm.py`: `LCMCompVisDenoiser` sigma/timestep mapping, `sample_lcm`, `CFGDenoiserLCM`, `LCMSampler`; no defect found.
- `modules/sd_samplers_extra.py`: `restart_sampler` restart-list construction and Heun restart step math; no defect found.
- `modules/mxfp8_config.py`, `modules/nvfp4_config.py`: Linear eligibility checks and config validators; no defect found.
- `modules/mxfp8_model_cache.py`, `modules/nvfp4_model_cache.py`: source/cache sidecar metadata, device/shape/dtype checks, safe tensor subclass loading, eligible module validation; no defect found in checked load paths.
- `modules/cache.py`: diskcache subsection creation, old cache conversion, and file data invalidation on mtime+size; no defect found.
- `modules/sd_models.py`: model listing/checkpoint alias selection and checkpoint dict key transforms through `get_state_dict_from_checkpoint`; no defect found in this partial pass.

Validation:
- Source inspection plus upstream comparison for timestep implementation loop behavior using AUTOMATIC1111 master source view.
- No precision-cache execution test was run because it would require model/cache artifacts and TorchAO runtime paths; no code change was made.

References:
- AUTOMATIC1111 upstream `modules/sd_samplers_timesteps_impl.py` master source for DDIM/PLMS loop shape.
- Local TorchAO cache loaders and sidecar metadata schema in `modules/mxfp8_model_cache.py` and `modules/nvfp4_model_cache.py`.

Next unchecked scope:
- Continue `modules/sd_hijack_clip.py`, `modules/sd_hijack_open_clip.py`, `modules/sd_hijack_optimizations.py`, API parameter mapping, and remaining generation-altering scripts/extensions.


### 2026-06-19 pass 4 - model reload and RNG subseed blending
Checked:
- `modules/sd_models.py`: checkpoint metadata/list/select helpers, state-dict read/transform paths, FP8/MXFP8/NVFP4 policy selectors and quantization application, model type detection, SDXL CLIP key remap, `load_model_weights`, alpha schedule override, model device movement/trash/reuse/reload paths, and token merging. No additional defect found; TorchAO cache/reload guards remain logically consistent with prior GB10 cache-hardening policy.
- `modules/rng.py`: `randn`, `randn_local`, `randn_like`, `randn_without_seed`, `manual_seed`, `create_generator`, `slerp`, and `ImageRNG.first/next` seed resize/subseed paths.

Finding:
- `rng.slerp` used a batch/global `dot.mean() > 0.9995` fallback. Mixed batches can contain one identical or antipodal low/high pair and another ordinary pair; the mean may take the spherical path, and the identical pair then divides by `sin(acos(1)) == 0`, producing NaNs during subseed blending. In image/latent tensors the dot product is also spatial after the channel-dimension norm, so the fallback must be elementwise rather than one decision for the whole tensor.

Fix:
- Clamp the dot product into the valid `acos` domain and select between spherical interpolation and linear interpolation elementwise with `torch.where`, preserving the existing channel-dimension norm behavior while avoiding singularities at nearly identical or opposite vectors.

Validation:
- `docker run --rm -v /home/kklouzal/stable-diffusion-webui:/work:ro -w /work local/gb10-a1111:latest python -c <AST-extracted rng.slerp finite/expected-value harness>` -> passed for identical, antipodal, orthogonal, and 3D channel-dimension cases.
- `python3 -m py_compile modules/rng.py` -> passed.

References:
- PyTorch `torch.acos` domain is [-1, 1], and division by `torch.sin(omega)` is singular at omega 0/pi; local harness reproduced the old NaN path and verified the patched finite path.

Commits:
- `8cf3573d484634eafe5b081c638c405d5a531333` - Fix subseed slerp singularities.

Next unchecked scope:
- Continue `modules/rng_philox.py`, then `modules/sd_hijack*.py`, `modules/sd_unet.py`, `modules/sd_vae*.py`, API parameter mapping, and remaining generation-altering scripts/extensions.


### 2026-06-19 pass 5 - Philox RNG, hijack shell, UNet/VAE load paths
Checked:
- `modules/rng_philox.py`: `uint32`, `philox4_round`, `philox4_32`, `box_muller`, and `Generator.randn`; no defect patched. Counter/key shapes and Box-Muller clamping offset avoid log(0), and behavior is intentionally CUDA-RNG-emulation oriented.
- `modules/sd_hijack.py`: optimizer selection/undo, weighted loss forwarding, SDXL/SSD conditioner wrapping, hijack/undo/redo, circular padding, embeddings replacement, and buffer registration. No defect found in checked generation-affecting logic.
- `modules/sd_unet.py`: UNet option lookup, activation/deactivation, and patched UNet forward dispatch. No defect found; custom UNet routing preserves original forward when inactive.
- `modules/sd_vae.py`: VAE discovery, resolution precedence, base VAE store/restore, VAE checkpoint cache, load/reload device transitions, and hijack callback ordering. No defect found.
- `modules/sd_vae_approx.py`: preview VAE architecture, model selection, load/cache, cheap RGB approximations for SD1/SDXL/SD3. No defect found in checked preview-quality math.
- `modules/sd_vae_taesd.py`: began architecture/model selection read; remaining TAESD functions still need full audit.

Validation:
- Source inspection only for this no-change checkpoint. Previous `modules/rng.py` py_compile and slerp harness remain the active validation for the only code change in this pass group.

References:
- Local Philox implementation comments and expected-output contract.
- Local VAE/UNet/hijack call graph from `modules/sd_models.py` reload/load paths checked in pass 4.

Next unchecked scope:
- Finish `modules/sd_vae_taesd.py`, then `modules/sd_hijack_clip.py`, `modules/sd_hijack_open_clip.py`, `modules/sd_hijack_optimizations.py`, API parameter mapping, and remaining generation-altering scripts/extensions.

### 2026-06-19 pass 6 - TAESD, CLIP hijacks, attention optimizers, API mapping, extension sampler controls
Checked:
- `modules/sd_vae_taesd.py`: TAESD encoder/decoder architecture, SD3 latent-channel selection, model-name dispatch for SD1/SDXL/SD3, download/cache/load, eval/device/dtype movement, and call sites through `modules/sd_samplers_common.py`; no defect found. TAESD decode remains correctly remapped from `[0, 1]` to `[-1, 1]`, while TAESD encode consumes `[0, 1]` image tensors as expected by the TAESD encoder path.
- `modules/sd_hijack_clip.py`: prompt chunking, comma backtracking, textual inversion placeholder insertion, empty chunk padding, pooled-output preservation, emphasis handoff, CLIP skip selection for SD1/SDXL, and infotext metadata; no generation defect found.
- `modules/sd_hijack_open_clip.py`: OpenCLIP tokenization, layer/pooled output handling, and embedding lookup path; no inference-conditioning defect found. `encode_embedding_init_text` does not cap to `nvpt` like the CLIP implementation, but this is a training/embedding initialization inconsistency rather than an inference generation-quality defect in this pass.
- `modules/sd_hijack_optimizations.py`: split attention, InvokeAI split attention, sub-quadratic attention, SDPA backend selection, attention block replacements, scale placement, dtype/upcast behavior, chunk/slice sizing, and mask handling where supported; no defect found.
- `modules/api/models.py`, `modules/api/api.py`: txt2img/img2img generation parameter mapping, sampler/scheduler alias normalization, infotext parameter import, override settings, init image/mask decode, selectable and always-on script argument vector handling, GB10 variable-length script arg override path, and response-only fields; no defect found.
- `extensions/openclaw-denoise-ramp/scripts/openclaw_denoise_ramp.py`: sigma interpolation and img2img tail remapping; no defect found. The ramp preserves A1111 img2img start/end sigma points and only bends spacing inside the active tail.
- `extensions/openclaw-multi-sampler/scripts/openclaw_multi_sampler.py`: custom sampler definition normalization, stage boundary validation, scheduler-chain sigma slicing, denoise-ramp composition, sampler kwargs dispatch, brownian-noise forwarding, state/tqdm callbacks, one-step DPM++ 2M SDE final-stage workaround, and snapshot decoding; no defect found.
- `extensions/sd-webui-incantations/scripts/cfg_combiner.py`, `pag.py`, `smoothed_energy_guidance.py`, `dynthres_core.py`, `dynthres_unipc.py`: CFG combiner wrapper ownership, PAG hidden-pass cond/uncond padding, SEG paired-batch guard and hook cleanup, dynamic thresholding denominator/std/quantile safeguards, and UniPC dynamic-threshold integration; no new defect found.

Validation:
- `python3 -m pytest -q extensions/sd-webui-incantations/tests/test_guidance_core.py extensions/openclaw-multi-sampler/tests` could not collect because host Python lacks `torch`; no host installs performed.
- `docker run --rm -v /home/kklouzal/stable-diffusion-webui:/work:ro -w /work local/gb10-a1111:latest python -m pytest -q extensions/sd-webui-incantations/tests/test_guidance_core.py extensions/openclaw-multi-sampler/tests` could not run because the existing validation image lacks `pytest`; no image/toolchain mutation performed.
- Source inspection only for this checkpoint because no code defect was confirmed.

References:
- Local A1111 TAESD call sites in `modules/sd_samplers_common.py`.
- Local sampler/scheduler normalization in `modules/sd_samplers.py` and API request construction in `modules/api/api.py`.
- Local extension tests in `extensions/sd-webui-incantations/tests/test_guidance_core.py` and `extensions/openclaw-multi-sampler/tests/test_openclaw_multi_sampler.py` were used as behavioral specifications where executable tooling was unavailable.

Next unchecked scope:
- Remaining built-in generation-altering scripts: `scripts/img2imgalt.py`, `scripts/loopback.py`, `scripts/outpainting_mk_2.py`, `scripts/poor_mans_outpainting.py`, `scripts/prompt_matrix.py`, `scripts/prompts_from_file.py`, `scripts/sd_upscale.py`, and `scripts/xyz_grid.py`; then non-generation postprocessing/diagnostic scripts can be recorded as out of generation-quality scope or lightly checked.

