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

### 2026-06-19 pass 7 - built-in generation scripts
Checked:
- `scripts/img2imgalt.py`: reverse-noise reconstruction loops, sigma-adjustment branch, CFG decode blend, cached recovered-noise reuse, random/recovered noise normalization, sampler handoff, and seed mutation; no new defect found. The script remains an explicitly experimental legacy img2img alternative and keeps Euler-oriented behavior by design.
- `scripts/loopback.py`: per-loop denoising curves, seed progression, prompt interrogation append path, inpainting-fill restoration, grid/history construction, and interrupt/skip behavior; no defect found.
- `scripts/outpainting_mk_2.py`: FFT/IFFT shaping, zero-normalization guards, phase normalization, histogram matching mask/reference split, directional expansion sizing, process crop selection, image/latent mask construction, seed/info capture, and final grid/save behavior; no new defect found.
- `scripts/poor_mans_outpainting.py`: directional target sizing, mask/latent-mask construction, grid tile selection, per-tile seed progression, and recombination; no defect found.
- `scripts/prompt_matrix.py`: prompt combination enumeration, positive/negative prompt replacement, seed-list generation, batch count, grid construction, and infotext insertion; no defect found.
- `scripts/prompts_from_file.py`: command-line prompt tag parsing, boolean parsing, sampler alias mapping, model override handling, per-line processing copy, prompt prepend/append behavior, seed iteration, and result aggregation; no defect found.
- `scripts/sd_upscale.py`: upscaler selection, tile splitting, batch/seed progression, recombination, and infotext/save behavior; no defect found.
- `scripts/xyz_grid.py`: axis application functions, sampler/checkpoint/VAE validation, int/float range parsing, seed fixing/varying, total-step accounting, grid/subgrid assembly, checkpoint/VAE reload cleanup, and metadata handling; no generation defect found. Existing TODO about grid metadata array alignment is documented local behavior and was not changed.

Validation:
- `python3 -m py_compile scripts/img2imgalt.py scripts/loopback.py scripts/outpainting_mk_2.py scripts/poor_mans_outpainting.py scripts/prompt_matrix.py scripts/prompts_from_file.py scripts/sd_upscale.py scripts/xyz_grid.py` -> passed.

References:
- Local script implementations and call paths through `modules.processing.process_images` and `modules.images` grid helpers.

Next unchecked scope:
- Remaining non-core generation-adjacent scripts/extensions: `scripts/custom_code.py`, `scripts/mxfp8_diagnostics_api.py`, postprocessing scripts, model-converter UI hooks, clear-cond-cache UI/API, and any remaining extension script surfaces that affect generation state indirectly. Then do a final ledger pass to mark the exhaustive audit boundary and any residual validation gaps.

### 2026-06-19 pass 8 - generation-adjacent scripts and final resumed-slice boundary
Checked:
- `scripts/custom_code.py`: generation mutation is intentionally user-authored and gated behind `--allow-code`; no fork math defect found in the wrapper path.
- `scripts/mxfp8_diagnostics_api.py`: diagnostics API/startup probe registration only; no generation-state mutation found beyond launching diagnostics.
- `scripts/postprocessing_codeformer.py`, `scripts/postprocessing_gfpgan.py`, `scripts/postprocessing_upscale.py`: postprocessing-only image transforms and extras upscaling; no diffusion-generation math path to remediate in this audit slice.
- `extensions/openclaw-clear-cond-cache/scripts/openclaw_clear_cond_cache.py`: cache/status/compile/backend activity helpers and token estimation surface; no new generation-quality defect found in the inspected state-management portions.
- `extensions/sd-webui-incantations/scripts/dynamic_thresholding.py`, `incantation_base.py`, `incant_utils/module_hooks.py`, `ui_wrapper.py`: dynamic-threshold sampler wrapping/restoration, incantation submodule dispatch ordering, hook registration/removal helpers, and wrapper no-op defaults; no new defect found beyond the already-audited core math in `dynthres_core.py`, `pag.py`, `cfg_combiner.py`, and `smoothed_energy_guidance.py`.
- `extensions/sd-webui-model-converter/scripts/convert.py`, `ui.py`: checkpoint/LoRA conversion utility, precision conversion helpers, CLIP key/position-id repair, safe output naming/path checks, and API/UI hooks; generation-adjacent artifact tooling only, no runtime generation defect found.

Validation:
- `python3 -m py_compile scripts/custom_code.py scripts/mxfp8_diagnostics_api.py scripts/postprocessing_codeformer.py scripts/postprocessing_gfpgan.py scripts/postprocessing_upscale.py` -> passed.
- Direct `python3 -m py_compile` for extension scripts failed to write an existing extension `__pycache__` file due to permissions, so compilation was rerun with explicit `/tmp` bytecode outputs.
- `python3 - <<PY ... py_compile.compile(..., cfile=/tmp/a1111-pycompile-hwkpf2z4/*.pyc, doraise=True) ... PY` for `openclaw_clear_cond_cache.py`, Incantations wrapper/helper scripts, and model-converter scripts -> passed.

References:
- Local extension entrypoints and helper implementations listed above.
- Existing executable tests remain blocked as recorded in pass 6 by host missing `torch` and validation image missing `pytest`.

Resumed-slice status:
- The resume scope from `modules/sd_vae_taesd.py` through CLIP hijacks, attention optimizers, API mapping, GB10 generation extensions, built-in generation scripts, and generation-adjacent scripts/extensions has been inspected and ledgered.
- No additional code defect was confirmed in this resumed slice, so only ledger commits were made.

Next unchecked scope:
- A final owner review can decide whether to broaden beyond generation-quality math/logic into non-generation surfaces, documentation, UI-only behavior, or full integration tests after installing/using an approved test surface with both `torch` and `pytest` available. Do not push until the owner reviews the ahead commits.



### 2026-06-20 pass 9 - broadened UI/config/save/history HTML surfaces
Preflight:
- Rechecked `git status --short --branch`: `latest...origin/latest [ahead 9]` before edits.
- Rechecked HEAD: `815237f3c64ae1b14539554912b5e673a44ce40f` before edits.
- Read ledger tail; previous generation-adjacent pass had stopped at the audit boundary and explicitly left non-generation surfaces for owner-approved broadening.

Checked:
- `modules/options.py`, `modules/shared_options.py`: option registration, load migrations, bad-type warnings, restricted/frozen setting guards, save/load behavior; no data-loss or state-consistency defect found in the inspected paths.
- `modules/config_states.py`: config-state discovery, webui/extension snapshot collection, webui restore, extension restore result aggregation; no source edit in this pass. Noted restore remains intentionally destructive because it is the feature contract for restoring a saved commit set.
- `modules/ui_loadsave.py`: UI defaults load/save, component mapping, dropdown/radio validation, default review/apply flow. Found and fixed unescaped `gr.HTML` table rendering of saved/default UI paths and old/new values.
- `modules/ui_common.py`: output-panel image save/download, `log.csv` header migration/padding, save-selected image handling, zip creation, and output-path message rendering; no defect found. Saved-path status uses `plaintext_to_html`, and CSV writes use `csv.writer`.
- `modules/ui_tempdir.py`: temp-file registration/checking, PNG metadata preservation, custom temp dir registration, cleanup of PNG-only temp files, and temp-path detection; no defect found in this pass.
- `modules/ui_extensions.py`: installed-extension table, commit links, backup/restore config-state table, install result output, and available-extension table escaping. Found and fixed unescaped HTML rendering for branch/status/version/config-backup fields and commit-link text/href.
- `modules/styles.py`: style CSV loading/saving, backup file creation, prompt/style merge and extraction logic; no defect found in inspected paths.

Findings/fixes:
- Fixed UI defaults review HTML injection: `modules/ui_loadsave.py` now escapes table paths and old/new values while preserving the intentional `None` marker span.
- Fixed extension/config backup table HTML injection: `modules/ui_extensions.py` now escapes commit-link text/href, extension status/version/branch, config backup name/path, webui branch, and saved extension branch fields.
- Added focused contract tests in `tests/test_ui_loadsave_contract.py` and `tests/test_ui_extensions_contract.py` for the escaping behavior without requiring Torch or a full Gradio startup.

Commits:
- `15039ba3 Escape UI backup and defaults HTML`

Validation:
- `python3 -m py_compile modules/ui_loadsave.py modules/ui_extensions.py tests/test_ui_loadsave_contract.py tests/test_ui_extensions_contract.py` -> passed.
- `python3 -m pytest -q tests/test_ui_loadsave_contract.py tests/test_ui_extensions_contract.py` -> passed, 3 tests; existing pytest warning remains `Unknown config option: base_url`.
- `python3 -m py_compile modules/options.py modules/shared_options.py modules/config_states.py modules/ui_loadsave.py modules/ui_common.py modules/ui_tempdir.py modules/ui_extensions.py modules/styles.py tests/test_ui_loadsave_contract.py tests/test_ui_extensions_contract.py` -> passed.
- `git diff --check` -> passed after setting this worktree's Git whitespace check to include `cr-at-eol`, matching the existing CRLF style of `modules/ui_extensions.py` and avoiding false positives on CRLF-added lines.

References:
- Python stdlib `html.escape` semantics for escaping `&`, `<`, `>`, and quotes in HTML text/attribute contexts.
- Local existing escaping patterns in `modules/ui_common.py` and `modules/ui_extensions.py`.

Next unchecked scope:
- Continue broadened audit with API/runtime/state surfaces not already covered in generation passes: `modules/call_queue.py`, `modules/progress.py`, `modules/shared_state.py`, `modules/shared.py`, `modules/shared_items.py`, `modules/shared_init.py`, `modules/initialize.py`, `modules/initialize_util.py`, `modules/launch_utils.py`, `modules/restart.py`, `modules/sysinfo.py`, `modules/errors.py`, `modules/localization.py`, and then JavaScript frontend state/history files beginning with `javascript/ui.js`, `generationParams.js`, `localStorage.js`, `progressbar.js`, `imageviewer.js`, and `extensions.js`.


### 2026-06-20 pass 10 - runtime queue/progress/shared init/error handling
Checked:
- `modules/call_queue.py`: queue lock usage, UI/GPU call wrappers, exception-to-HTML path, state reset, memmon/profiling HTML append; no defect found in inspected paths.
- `modules/progress.py`: task queue state, pending/progress API models, active/queued/completed reporting, progress/ETA math, live preview encoding, and result restore window; no defect found in inspected paths.
- `modules/shared_state.py`: job lifecycle, interrupt/skip/stop flags, restart command event, live preview image assignment, and state dict; no defect found in inspected paths.
- `modules/errors.py`: exception recording/reporting/display helpers and wrapper `run`; found and fixed an argument-order bug in `run` that could make the error handler fail while trying to report a wrapped exception.
- `modules/shared.py`, `modules/shared_items.py`, `modules/shared_init.py`: shared singleton setup, option/template wiring, dtype/device initialization, lazy model property, callback-order option generation; no non-generation runtime defect found in inspected paths.
- `modules/initialize_util.py`: restart-config restore, TLS option validation, auth credential parsing, signal handler, options onchange registration, and CORS/GZip middleware setup; no defect found in inspected paths.
- `modules/launch_utils.py`: version/source detection, subprocess wrapper, git clone/pull helpers, extension installer dispatch, requirements check, and extension list handling; no source edit in this pass.
- `modules/restart.py`, `modules/localization.py`, `modules/sysinfo.py`, `modules/util.py`: restart marker/exit, localization JSON merge to JS, sysinfo checksum/env/config/package collection, file walking/sorting/open-folder helpers; no defect found in inspected paths.

Findings/fixes:
- Fixed `errors.run` to call `display(e, task)` instead of `display(task, e)`, preserving the intended error-reporting path when wrapped code raises.
- Added `tests/test_errors_contract.py` to assert `errors.run` reports the wrapped exception without raising a secondary exception.
- Follow-up commit restored `modules/errors.py` CRLF line endings so the net tree diff remains the intended one-line logic fix plus test.

Commits:
- `e6109dde Fix errors.run exception display`
- `c7d0dc5e Restore errors module line endings`

Validation:
- `python3 -m py_compile modules/errors.py tests/test_errors_contract.py` -> passed.
- `python3 -m pytest -q tests/test_errors_contract.py` -> passed, 1 test; existing pytest warning remains `Unknown config option: base_url`.
- `python3 -m py_compile modules/call_queue.py modules/progress.py modules/shared_state.py modules/errors.py modules/shared.py modules/shared_items.py modules/shared_init.py modules/initialize_util.py modules/launch_utils.py modules/restart.py modules/localization.py modules/sysinfo.py modules/util.py tests/test_errors_contract.py` -> passed.
- `python3 -m pytest -q tests/test_errors_contract.py tests/test_ui_loadsave_contract.py tests/test_ui_extensions_contract.py` -> passed, 4 tests; same existing `base_url` warning.
- `git diff --check` -> passed.

References:
- Python `traceback.TracebackException.from_exception` call contract as used locally by `display(e, task)`.
- Local wrapper pattern in `display_once(e, task)` confirmed the intended argument order.

Next unchecked scope:
- Continue broadened audit with JavaScript frontend state/history and browser-side save/send behavior: `javascript/ui.js`, `generationParams.js`, `localStorage.js`, `progressbar.js`, `imageviewer.js`, `imageviewerGamepad.js`, `extraNetworks.js`, `extensions.js`, `settings.js`, `token-counters.js`, `dragdrop.js`, and `edit-attention.js`; then return to remaining API/postprocessing/file-history surfaces not already covered.


### 2026-06-20 pass 11 - JavaScript frontend state/history controls
Checked:
- `javascript/ui.js`: gallery selection helpers, tab switching, submit/restore progress task ids, resolution paste parsing, settings JSON watcher, checkpoint hash display, restart reload polling, seed/dimension helpers, and theme selection. Found malformed theme URL construction when the current URL already had query parameters.
- `javascript/generationParams.js`: gallery click/key listeners and modal-close infotext refresh trigger; no defect found in inspected paths.
- `javascript/localStorage.js`: localStorage set/get/remove wrappers with exception guards; no defect found in current call sites.
- `javascript/progressbar.js`: progress request JSON parsing/error handling, title/progress formatting, live preview loop, wake lock handling, and cleanup; no defect found in inspected paths.
- `javascript/imageviewer.js`, `imageviewerGamepad.js`: modal/gallery navigation, save routing, live-preview toggle, keyboard/gamepad controls, and image setup idempotence; no defect found in inspected paths.
- `javascript/extensions.js`: extension enable/update JSON collection, check/update progress, install-from-index handoff, restore confirmation, and all-toggle behavior; no defect found in inspected paths.
- `javascript/extraNetworks.js`: CSS injection helper, prompt focus tracking, filtering/sorting, prompt insertion/removal, and preview-save handoff. Found a comparison typo that prevented existing injected CSS from being cleared before appending new CSS.
- `javascript/settings.js`, `token-counters.js`, `dragdrop.js`, `edit-attention.js`: settings search/category labeling, token-counter debounce/visibility, image drag/drop/paste routing, prompt image URL fetch, and attention weight editing; no defect found in inspected paths.

Findings/fixes:
- `set_theme` now uses the browser `URL`/`searchParams` API via `make_theme_url`, preserving existing query strings and avoiding malformed `...?foo=bar?__theme=...` URLs.
- `toggleCss` now clears existing style text with assignment before appending the new CSS, instead of evaluating `style.innerHTML == ''` and duplicating CSS on repeated updates.
- Added `tests/test_javascript_ui_contract.py` to lock both frontend source contracts.

Commits:
- `a09cfc4d Fix frontend theme and style update logic`

Validation:
- `node --check` passed for `javascript/ui.js`, `extraNetworks.js`, `generationParams.js`, `localStorage.js`, `progressbar.js`, `imageviewer.js`, `extensions.js`, `settings.js`, `token-counters.js`, `dragdrop.js`, and `edit-attention.js`.
- `python3 -m pytest -q tests/test_javascript_ui_contract.py` -> passed, 2 tests; existing pytest warning remains `Unknown config option: base_url`.
- `git diff --check` -> passed.

References:
- Browser `URL` and `URLSearchParams` behavior for preserving and updating query parameters.
- Local JS contract-test pattern in `tests/test_image_mask_fix_contract.py`.

Next unchecked scope:
- Continue with remaining API/postprocessing/file-history surfaces beyond previous generation passes: `modules/extras.py`, `modules/postprocessing.py`, `modules/scripts_postprocessing.py`, `modules/scripts_auto_postprocessing.py`, `modules/ui_postprocessing.py`, `modules/ui_prompt_styles.py`, `modules/ui_settings.py`, `modules/ui_extra_networks*.py`, `modules/hashes.py`, `modules/cache.py`, `modules/modelloader.py`, `modules/safe.py`, and remaining shell/launch scripts/config files where logic affects state, files, or runtime stability.


### 2026-06-20 pass 12 - postprocessing file-output and caption handling
Checked:
- `modules/extras.py`: PNG info rendering, model-merger config selection/copy, safetensors metadata stringification, merge arithmetic dispatch, inpainting/instruct-pix2pix channel handling, VAE bake-in, metadata recipe construction, and output checkpoint naming; no new defect found in inspected paths.
- `modules/postprocessing.py`: extras input source selection, batch/directory iteration, existing PNG info preservation, postprocessing script execution, output image save path, caption sidecar read/merge/write behavior, and API compatibility wrapper. Found incorrect existing-caption combine order for `Prepend` and `Append`.
- `modules/scripts_postprocessing.py`: postprocessed image suffix collision handling, copied extra-image state, postprocessing script ordering/filtering, UI arg mapping, firstpass/process order, and API arg construction; no defect found in inspected paths.
- `modules/scripts_auto_postprocessing.py`: main-UI postprocessing script bridge and option-based script selection; no defect found.
- `modules/ui_postprocessing.py`: extras tab mode mapping, directory controls, script inputs, submit wiring, output panel, paste field registration, and image-change notification; no defect found.

Findings/fixes:
- Added `combine_caption(existing_caption, new_caption, action)` and fixed caption ordering so `Prepend` writes generated/new caption before existing caption, while `Append` writes existing caption before generated/new caption.
- Added `tests/test_postprocessing_caption_contract.py` using AST extraction of the pure helper to avoid full app/Torch startup on host Python.

Commits:
- `6652986b Fix postprocessing caption combine order`

Validation:
- `python3 -m py_compile modules/extras.py modules/postprocessing.py modules/scripts_postprocessing.py modules/scripts_auto_postprocessing.py modules/ui_postprocessing.py tests/test_postprocessing_caption_contract.py` -> passed.
- `python3 -m pytest -q tests/test_postprocessing_caption_contract.py` -> passed, 2 tests; existing pytest warning remains `Unknown config option: base_url`.
- `git diff --check` -> passed.

References:
- Local option label in `modules/shared_options.py`: `Prepend/Append = combine both`, with conventional prepend/append ordering applied to generated caption relative to existing caption.

Next unchecked scope:
- Continue with remaining file/cache/model metadata and extra-network surfaces: `modules/ui_prompt_styles.py`, `modules/ui_settings.py`, `modules/ui_extra_networks.py`, `modules/ui_extra_networks_user_metadata.py`, `modules/ui_extra_networks_checkpoints*.py`, `modules/ui_extra_networks_textual_inversion.py`, `modules/ui_extra_networks_hypernets.py`, `modules/hashes.py`, `modules/cache.py`, `modules/modelloader.py`, `modules/safe.py`, `modules/paths.py`, and shell/launch/config files not yet covered.

### 2026-06-20 pass 13 - prompt styles, settings, cache, paths, safe load, and extra-network metadata
Checked:
- `modules/ui_prompt_styles.py`: style selection, save/delete, dropdown refresh, materialize/apply wiring, and prompt-copy behavior; no defect found in inspected paths.
- `modules/ui_settings.py`: bulk and quick settings save, refresh components, checkpoint hash worker fanout, sysinfo validation, script-body reload, restart wiring, and settings reload values; no defect found in inspected paths.
- `modules/hashes.py`, `modules/cache.py`: sha256/addnet/partial hash cache invalidation by mtime and size, old cache migration, diskcache initialization, and cached-file helper behavior; no new defect found in inspected paths.
- `modules/modelloader.py`: model directory walk/download fallback, friendly names, upscaler registry de-duplication, spandrel extra arch initialization, dtype/half handling, and descriptor evaluation mode; no defect found.
- `modules/safe.py`: restricted pickle globals, zip filename filtering, legacy and zip checkpoint precheck paths, PyTorch 2.6 `weights_only` compatibility default, and global extra handler context; no defect found.
- `modules/paths.py`: SD/SDXL/BLIP/k-diffusion path discovery and import-path insertion behavior; no defect found.
- `modules/ui_extra_networks.py`, `modules/ui_extra_networks_user_metadata.py`, `modules/ui_extra_networks_checkpoints.py`, `modules/ui_extra_networks_checkpoints_user_metadata.py`, `modules/ui_extra_networks_textual_inversion.py`, `modules/ui_extra_networks_hypernets.py`: extra-network route registration, preview fetch guards, cover-image metadata fetch, item/card/tree/dir HTML construction, hidden-model detection, metadata editor save/preview behavior, and checkpoint/TI/hypernetwork item creation. Found unsafe string-prefix path classification and brittle cover-image metadata defaults/indexing.

Findings/fixes:
- Extra-network search terms and hidden-model local path detection now use `path_is_parent`/`os.path.commonpath` instead of raw string prefixes, preventing sibling paths with matching prefixes from being treated as inside an allowed preview root.
- Cover-image metadata fetch now defaults missing `ssmd_cover_images` to an empty list, rejects non-list decoded values, and rejects negative indexes instead of accidentally selecting from the end.
- Added `tests/test_extra_networks_metadata_contract.py` to lock the cover-image and path-parent contracts.

Validation:
- `python3 -m py_compile modules/ui_extra_networks.py tests/test_extra_networks_metadata_contract.py` -> passed.
- `python3 -m pytest -q tests/test_extra_networks_metadata_contract.py` -> passed, 2 tests; existing pytest warning remains `Unknown config option: base_url`.
- `git diff --check` -> passed.

Next unchecked scope:
- Continue with shell/launch/config/runtime helper files and remaining non-Python/less-traveled source surfaces not yet covered: launch scripts, `webui*.sh`, `modules/paths_internal.py`, API/options/history/save/delete paths, scripts/hooks, tests, and repository JS/config/shell files outside the already audited frontend set.

### 2026-06-20 pass 14 - launch shell and GB10 container helpers
Checked:
- `modules/paths_internal.py`: early `COMMANDLINE_ARGS` parsing with `shlex.split`, data/models/extensions/output path derivation, and default model/config path constants; no defect found.
- `launch.py`: environment preparation/test-server/sysinfo/start dispatch; no defect found.
- `webui-user.sh`, `webui-macos-env.sh`: user override templates and macOS defaults; no defect found.
- `docker/entrypoint.sh`, `docker/launch-a1111.sh`: container-owned config/style initialization, ownership handoff, command override behavior, and `COMMANDLINE_ARGS` single-source launch path; no defect found.
- `gb10/build.sh`, `gb10/run.sh`, `gb10/smoke-test.sh`, `gb10/stop.sh`: BuildKit/cache arguments, host data root setup, output symlink guard, owned extension sync/removal, container args/mounts/env, smoke-test API/import/quantization checks, and stop behavior; no defect found.
- `webui.sh`: install/bootstrap, venv handling, GPU/TCMalloc detection, restart loop, and accelerate launch selection. Found a shell test expression that treated any non-empty `ACCELERATE` value as true.

Findings/fixes:
- Fixed `webui.sh` accelerate branch guard to require `ACCELERATE=True` with a quoted `[[ ... == ... ]]` comparison, instead of `[ ${ACCELERATE}="True" ]` which is a single non-empty test argument.
- Added `tests/test_launch_shell_contract.py` to lock the corrected shell guard.

Validation:
- `bash -n webui.sh webui-user.sh webui-macos-env.sh docker/entrypoint.sh docker/launch-a1111.sh gb10/build.sh gb10/run.sh gb10/smoke-test.sh gb10/stop.sh` -> passed.
- `python3 -m pytest -q tests/test_launch_shell_contract.py` -> passed, 1 test; existing pytest warning remains `Unknown config option: base_url`.
- `git diff --check` -> passed.

Next unchecked scope:
- Continue through remaining API/options/history/save/delete paths, scripts/hooks, tests, repository JS/config files outside the already audited frontend set, and any source files not yet listed in ledger passes.

### 2026-06-20 pass 15 - options, config state, UI defaults/temp files, and API generation entrypoints
Checked:
- `modules/options.py`: option metadata helpers, freeze/restrict guards, callback rollback, default/type handling, legacy settings migrations, JSON dump categories, option insertion/reorder, and string-to-type casting; no defect found in inspected paths.
- `modules/config_states.py`: config-state listing/load ordering, webui/extension config capture, webui hard-reset restore path, extension fetch/reset/enable state restore, and disabled-extension persistence; no new defect found in inspected paths. Destructive restore behavior is existing explicit UI functionality, not changed in this audit slice.
- `modules/ui_loadsave.py`: UI component default capture, value/type normalization, dropdown/tab validation, change review/apply, and corrupted-load protection; no defect found.
- `modules/ui_tempdir.py`: Gradio temp-file registration, PNG metadata preservation, custom temp-dir registration/cleanup, and temp path classification; no defect found.
- `modules/api/api.py`: precision-map cache signatures and serialization, script default args, script selection and always-on arg expansion, infotext field application, API image encode/decode, request URL guard, API middleware/auth, route registration, OpenClaw runtime env defaults, and txt2img/img2img entrypoint setup through the inspected region; no new defect found in inspected paths.

Validation:
- `python3 -m py_compile modules/options.py modules/config_states.py modules/ui_loadsave.py modules/ui_tempdir.py modules/api/api.py` -> passed.
- `git diff --check` -> passed.

Next unchecked scope:
- Continue deeper through the remainder of `modules/api/api.py` methods after txt2img/img2img setup, API models, remaining script/hook surfaces (`modules/scripts.py`, `modules/script_callbacks.py`, `modules/script_loading.py`), and remaining source/test/config files not yet explicitly listed in ledger passes.

### 2026-06-20 pass 16 - API response models and remaining API methods
Checked:
- Remainder of `modules/api/api.py`: txt2img/img2img completion and task cleanup, extras single/batch image endpoints, PNG info parsing, progress response construction, interrogate, interrupt/skip/unload/reload, options get/set, metadata listing endpoints, embedding/hypernetwork create/train wrappers, memory reporting, extension listing, and server control endpoints. Found a response-model mismatch for `current_task`.
- `modules/api/models.py`: dynamic processing models, extras/PNG/progress/interrogate/train/create/listing response models, options/flags dynamic models, and extension/script metadata schemas. Found the progress schema omitted the task id returned by the API implementation.

Findings/fixes:
- Added `current_task: Optional[str]` to `ProgressResponse` so `/sdapi/v1/progress` exposes the active task id that `progressapi()` already returns, instead of having it dropped by the response model.
- Added `tests/test_api_progress_contract.py` to lock the API/model contract.

Validation:
- `python3 -m py_compile modules/api/models.py modules/api/api.py tests/test_api_progress_contract.py` -> passed.
- `python3 -m pytest -q tests/test_api_progress_contract.py` -> passed, 1 test; existing pytest warning remains `Unknown config option: base_url`.
- `git diff --check` -> passed.

Next unchecked scope:
- Continue through script/hook surfaces (`modules/scripts.py`, `modules/script_callbacks.py`, `modules/script_loading.py`), extension management internals, image save/delete/history paths, and remaining source/test/config files not yet explicitly listed in ledger passes.

### 2026-06-20 pass 17 - script loading, callback ordering, and extension metadata
Checked:
- `modules/scripts.py`: base `Script` hook contracts, script discovery from built-in/extensions/processing scripts, extension metadata dependency expansion, topological ordering, module load/reload, UI arg collection/API metadata, script dropdown visibility, selectable script execution, always-on hook dispatch/timing, element-specific before/after component callbacks, source reload, setup/before-hr hooks, and named arg mutation; no defect found in inspected paths.
- `modules/script_callbacks.py`: callback registration naming/de-duplication, metadata/user ordering, ordered callback cache invalidation, all callback dispatchers, unload/reload behavior, and callback removal helpers; no defect found.
- `modules/script_loading.py`: module import-by-path and extension preload handling; no defect found.
- `modules/extensions.py`: extension metadata parsing, canonical-name de-duplication, requirements/callback-order parsing, Git info caching, update checks, hard reset helper, active-extension filters, file listing, and path-to-extension lookup; no defect found in inspected paths.

Validation:
- `python3 -m py_compile modules/scripts.py modules/script_callbacks.py modules/script_loading.py modules/extensions.py` -> passed.
- `git diff --check` -> passed.

Next unchecked scope:
- Continue through image save/delete/history paths (`modules/images.py`, `modules/ui_common.py`, `modules/infotext_utils.py` where not already covered), model load/config/vae surfaces, remaining `modules/sd_*` files, tests, and repo config/JS files not yet explicitly listed in ledger passes.

### 2026-06-20 pass 18 - image save/history/infotext and model load/config/VAE surfaces
Checked:
- `modules/ui_common.py`: gallery infotext selection, save button argument mapping, selected-image-only handling, save CSV migration/padding, zip packaging from saved file list, output panel folder-open behavior, paste-button registration, refresh button update propagation, and dialog show/close wiring. No defect found in inspected paths.
- `modules/images.py`: grid/split/combine helpers, resize modes, filename token expansion/sanitization, sequence numbering, image metadata writes for PNG/JPEG/WebP/AVIF/GIF, atomic image saves, alternate 4chan export path, sidecar txt writes, metadata reads, image data parsing, transparency flattening, EXIF transpose, and PNG transparency repair. No defect found in inspected paths.
- `modules/infotext_utils.py`: image-from-gallery/temp-file decoding, send-image dimensions, old hires-fix restoration, indexed inpaint infotext mappings, generation-parameter parsing/defaults/backcompat, override settings extraction, and paste-field value casting. No defect found in inspected paths.
- `modules/sd_models_config.py`: SD/SDXL/SD3/config inference from state dict keys, v-parameterization probe, and near-checkpoint config discovery; no defect found.
- `modules/sd_vae.py`: VAE listing, setting/user-metadata/near-checkpoint resolution order, VAE cache/load/restore behavior, and reload-device/hijack callback flow; no defect found.
- `modules/sd_models.py`: checkpoint registration/listing/alias selection, safetensors metadata/state-dict load, checkpoint state cache behavior, FP8/MXFP8/NVFP4 policy selection and reload guards, model type/config/load/reload/unload/cache reuse, VAE integration, token merging, and TorchAO move/trash helpers. Found a path-boundary defect in checkpoint display-name classification.
- `modules/sd_models_types.py`, `modules/sd_models_xl.py`, `modules/sd_unet.py`, `modules/sd_vae_approx.py`, `modules/sd_vae_taesd.py`, `modules/sd_disable_initialization.py`, `modules/sd_hijack_checkpoint.py`, `modules/sd_hijack_utils.py`, `modules/sd_hijack_unet.py`: type annotations, SDXL compatibility shims, UNet option activation, VAE approximation/TAESD helper surfaces, initialization suppression/meta-load hooks, and hijack utility wrappers; no defect found in inspected paths.

Findings/fixes:
- `CheckpointInfo` used raw `abspath.startswith(...)` checks to decide whether a checkpoint lived under `--ckpt-dir` or the default model root. A sibling such as `/models/Stable-diffusion-extra/...` could be misclassified as inside `/models/Stable-diffusion`, corrupting display names/aliases. Added `path_is_parent()` using `os.path.commonpath()` and switched relative-name derivation to `os.path.relpath()`.
- Added `tests/test_sd_models_checkpoint_info_contract.py` to lock the checkpoint root-boundary contract.

Validation:
- `python3 -m py_compile modules/sd_models.py tests/test_sd_models_checkpoint_info_contract.py` -> passed.
- `python3 -m pytest -q tests/test_sd_models_checkpoint_info_contract.py` -> passed, 2 tests; existing pytest warning remains `Unknown config option: base_url`.

Next unchecked scope:
- Continue through the remaining `modules/sd_*` sampler/hijack/emphasis/CLIP files not listed in pass 18, then tests, repository config files, and JS files not yet explicitly listed in ledger passes.

### 2026-06-20 pass 19 - remaining SD hijack, CLIP, sampler, and scheduler surfaces
Checked:
- `modules/sd_emphasis.py`: emphasis option selection and normalization variants; no defect found.
- `modules/sd_hijack.py`: optimizer selection/undo, checkpoint patching, weighted loss/forward hooks, model hijack/undo for SD1/SD2/SDXL/AltDiffusion/OpenCLIP/XLMR, textual inversion embedding injection, prompt-length reporting, circular convolution patch, and buffer registration device handling; no defect found in inspected paths.
- `modules/sd_hijack_clip.py`, `modules/sd_hijack_clip_old.py`, `modules/sd_hijack_open_clip.py`, `modules/sd_hijack_xlmr.py`, `modules/sd_hijack_ip2p.py`: prompt chunking, comma backtracking, textual inversion placement, emphasis multiplier flow, pooled-output handling, old emphasis compatibility, OpenCLIP/XLMR tokenization/embedding-init paths, and InstructPix2Pix config heuristic; no defect found.
- `modules/sd_hijack_optimizations.py`: attention optimizer registration/selection, split attention memory slicing, InvokeAI/sub-quadratic/SDPA attention paths, SDPA backend normalization/status setters, and VAE attention block replacements; no defect found in inspected paths.
- `modules/sd_samplers.py`: sampler registry/visibility, infotext sampler/scheduler normalization, hires sampler/scheduler extraction, and invalid sampler autocorrection; no defect found.
- `modules/sd_samplers_common.py`: img2img step math, latent/image encode/decode approximation paths, live preview storage, eta-noise-delta decision, refiner switch timing, Torch RNG hijack, sampler setup/extra params/noise sampler creation, and callback state handling; no defect found in inspected paths.
- `modules/sd_samplers_cfg_denoiser.py`: condition concatenation/subscript/padding, CFG denoiser inner-model dispatch, refiner refresh, mask blending hook, edit-model guidance, skip-uncond decisions, cond/uncond batching, callback dispatch, preview latent selection, and after-CFG callback; no defect found in inspected paths.
- `modules/sd_samplers_kdiffusion.py`, `modules/sd_samplers_lcm.py`, `modules/sd_samplers_timesteps.py`, `modules/sd_samplers_timesteps_impl.py`, `modules/sd_samplers_extra.py`, `modules/sd_schedulers.py`: K-diffusion sigma scheduling/extra args/noise samplers, LCM denoiser/sample loop, CompVis timestep scheduling/img2img start math, DDIM/DDIM CFG++/PLMS/UniPC implementations, Restart sampler step list, and scheduler functions for uniform/SGM/Karras-like/simple/normal/DDIM/beta/AYS/KL optimal schedules; no defect found in inspected paths.

Validation:
- `python3 -m py_compile modules/sd_disable_initialization.py modules/sd_emphasis.py modules/sd_hijack.py modules/sd_hijack_checkpoint.py modules/sd_hijack_clip.py modules/sd_hijack_clip_old.py modules/sd_hijack_ip2p.py modules/sd_hijack_open_clip.py modules/sd_hijack_optimizations.py modules/sd_hijack_unet.py modules/sd_hijack_utils.py modules/sd_hijack_xlmr.py modules/sd_models.py modules/sd_models_config.py modules/sd_models_types.py modules/sd_models_xl.py modules/sd_samplers.py modules/sd_samplers_cfg_denoiser.py modules/sd_samplers_common.py modules/sd_samplers_compvis.py modules/sd_samplers_extra.py modules/sd_samplers_kdiffusion.py modules/sd_samplers_lcm.py modules/sd_samplers_timesteps.py modules/sd_samplers_timesteps_impl.py modules/sd_schedulers.py modules/sd_unet.py modules/sd_vae.py modules/sd_vae_approx.py modules/sd_vae_taesd.py` -> passed.

Next unchecked scope:
- Continue through tests, repository config files, and JS files not yet explicitly listed in ledger passes.

### 2026-06-20 pass 20 - tests, auxiliary JavaScript, and repo config manifests
Checked:
- `tests/*.py`: current lightweight audit contract tests, including API progress, errors, extension metadata, extra-network path/metadata, JavaScript UI contracts, launch shell guard, postprocessing captions, processing infotext alignment, save serialization, SD checkpoint path classification, textual-inversion preview save, UI extension escaping, and UI load/save escaping. Found a test isolation defect.
- `script.js`, `.eslintrc.js`, `javascript/aspectRatioOverlay.js`, `contextMenus.js`, `edit-order.js`, `hints.js`, `hires_fix.js`, `imageMaskFix.js`, `inputAccordion.js`, `localization.js`, `notification.js`, `profilerVisualization.js`, `resizeHandle.js`, `textualInversion.js`, and `ui_settings_hints.js`: global Gradio helpers/callbacks, keyboard shortcuts, aspect-ratio overlay, context menus/repeat generation, prompt order editing, tooltips, hires resolution state, mask canvas resize, accordion state sync, localization dump/RTL handling, browser notifications, profiler table expansion, resize handle behavior, textual inversion progress startup, and settings comments/quicksettings hints. No defect found in inspected paths.
- `package.json`, `pyproject.toml`, `_typos.toml`, `environment-wsl2.yaml`, and `configs/*.yaml`: lint/test manifest settings, dependency environment metadata, and model config targets/shape constants used by `sd_models_config`; no defect found.

Findings/fixes:
- `tests/test_extensions_metadata_contract.py` installed fake `modules.*` entries in `sys.modules` without restoring them, so running the whole lightweight suite could make later tests fail to import real `modules.ui_extensions`. Added an autouse fixture that snapshots/restores `modules` and `modules.*` entries around each test.

Validation:
- `python3 -m py_compile tests/test_extensions_metadata_contract.py` -> passed.
- `python3 -m pytest -q tests` -> passed, 31 tests; existing pytest warning remains `Unknown config option: base_url`.
- `node --check script.js .eslintrc.js javascript/aspectRatioOverlay.js javascript/contextMenus.js javascript/edit-order.js javascript/hints.js javascript/hires_fix.js javascript/imageMaskFix.js javascript/inputAccordion.js javascript/localization.js javascript/notification.js javascript/profilerVisualization.js javascript/resizeHandle.js javascript/textualInversion.js javascript/ui_settings_hints.js` -> passed.
- JSON/TOML/YAML parse check for `package.json`, `pyproject.toml`, `_typos.toml`, `environment-wsl2.yaml`, and `configs/*.yaml` -> passed.
- `git diff --check` -> passed.

Next unchecked scope:
- Broad source audit still is not complete: remaining not-explicitly-listed surfaces include deeper training/hypernetwork/textual-inversion internals, checkpoint merger/model conversion paths, face restoration/GFPGAN/CodeFormer helpers, lowvram/devices/mac-specific paths, interrogate/deepbooru, UI construction modules not named in earlier passes, and extension source trees beyond the already-audited extension slices.

### 2026-06-20 pass 21 - training, textual inversion, hypernetwork, and checkpoint merger internals
Checked:
- `modules/textual_inversion/autocrop.py`: focal-point weighting, entropy/corner/face point collection, YuNet model path selection, model download/cache path, and crop coordinate clamping. Found an OpenCV array truthiness defect in Haar face detection.
- `modules/textual_inversion/dataset.py`: dataset image/text loading, latent sampling modes, alpha-channel loss weighting normalization, tag shuffle/dropout, varsize grouping, grouped batch sampling, and collate wrappers; no additional defect found.
- `modules/textual_inversion/image_embedding.py`: embedding JSON/base64 encode/decode, PNG side-band embedding insert/extract, deterministic xor/style blocks, black-border cropping, and preview caption overlay; no defect found in inspected paths.
- `modules/textual_inversion/learn_schedule.py`, `modules/textual_inversion/saving_settings.py`, and `modules/textual_inversion/ui.py`: learning-rate schedule parsing/application, training settings serialization, embedding creation UI, and optimization restore around embedding training; no defect found.
- `modules/textual_inversion/textual_inversion.py`: template discovery, embedding database reload/cache/hash/shape handling, embedding creation/loading, train-input validation, textual-inversion training loop, preview image/embed save path, loss/tensorboard writes, and checkpoint metadata on save; no additional defect found.
- `modules/hypernetworks/hypernetwork.py` and `modules/hypernetworks/ui.py`: hypernetwork module construction/load/save, dropout parsing, optimizer resume metadata, attention patch application, hypernetwork list/load/multiplier behavior, creation UI contract, and training loop/preview save path. Found a create helper return-value defect.
- `modules/extras.py` and `modules/ui_checkpoint_merger.py`: PNG info display, checkpoint config copy selection, metadata normalization/read/merge recipe construction, weighted/add-difference/no-interpolation merge flows, VAE baking, discard regex, safetensors metadata save, and checkpoint merger UI wiring; no defect found in inspected paths.

Findings/fixes:
- `image_face_points()` used `if faces:` on OpenCV `detectMultiScale()` results. For NumPy arrays with multiple detections this raises ambiguous-truth errors instead of returning face focal points. Switched to `len(faces) > 0` and added `tests/test_textual_inversion_autocrop_contract.py`.
- `create_hypernetwork()` saved and reloaded hypernetworks but returned `None`, while `modules/hypernetworks/ui.py` reports the returned filename like textual-inversion embedding creation does. Returned `fn` and added `tests/test_hypernetwork_creation_contract.py`.

Validation:
- `PYTHONPYCACHEPREFIX=/tmp/gb10-a1111-pycache python3 -m py_compile modules/textual_inversion/autocrop.py modules/hypernetworks/hypernetwork.py tests/test_textual_inversion_autocrop_contract.py tests/test_hypernetwork_creation_contract.py` -> passed.
- `PYTHONPYCACHEPREFIX=/tmp/gb10-a1111-pycache python3 -m pytest -q tests/test_textual_inversion_autocrop_contract.py tests/test_hypernetwork_creation_contract.py tests/test_textual_inversion_preview_save_contract.py` -> passed, 3 tests; existing pytest warning remains `Unknown config option: base_url`.
- `git diff --check` -> passed.

Next unchecked scope:
- Continue through face restoration/GFPGAN/CodeFormer helpers, lowvram/devices/mac-specific paths, interrogate/deepbooru, remaining UI construction modules not named in earlier passes, and extension source trees beyond already-audited extension slices. The broad source audit is still not complete.
