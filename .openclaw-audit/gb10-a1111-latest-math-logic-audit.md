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

### 2026-06-20 pass 22 - face restoration, device/interrogate helpers, UI construction triage, and extension inventory
Checked:
- `modules/codeformer_model.py`, `modules/gfpgan_model.py`, `modules/face_restoration.py`, and `modules/face_restoration_utils.py`: model discovery/load wrappers, device selection, face helper construction, BGR/RGB tensor conversion, per-face restore failure handling, paste-back resizing, unload behavior, and facexlib download-dir patching; no defect found in inspected paths.
- `modules/devices.py`, `modules/lowvram.py`, and `modules/mac_specific.py`: device selection, autocast/manual-cast gates, TF32 enablement, NaN checks, low/medvram hook installation, module CPU/GPU shuttling, and MPS workaround registration; no defect found in inspected paths.
- `modules/interrogate.py`, `modules/deepbooru.py`, and `modules/deepbooru_model.py`: CLIP category download/cache, BLIP/CLIP load/unload/ranking, interrogate state handling, DeepDanbooru model load/start/stop/tag filtering, and generated model forward graph syntax; no defect found in inspected paths.
- UI construction triage covered function/class boundaries and targeted reads in `modules/ui_extra_networks.py` around allowed preview dirs, fetch endpoints, card/tree/dir HTML generation, page ordering, refresh/load behavior, path-parent checks, and preview-save authorization; no defect found in inspected paths.
- Extension inventory enumerated remaining source files under `extensions/` and `extensions-builtin/` and syntax-validated Python/JavaScript files, but did not complete function-by-function review of every extension module in this pass.

Validation:
- `PYTHONPYCACHEPREFIX=/tmp/gb10-a1111-pycache python3 -m py_compile modules/codeformer_model.py modules/gfpgan_model.py modules/face_restoration.py modules/face_restoration_utils.py modules/devices.py modules/lowvram.py modules/mac_specific.py modules/interrogate.py modules/deepbooru.py modules/deepbooru_model.py modules/ui.py modules/ui_components.py modules/ui_component_patches.py modules/ui_extra_networks.py modules/ui_extra_networks_checkpoints.py modules/ui_extra_networks_checkpoints_user_metadata.py modules/ui_extra_networks_hypernets.py modules/ui_extra_networks_textual_inversion.py modules/ui_extra_networks_user_metadata.py modules/ui_html_extensions.py modules/ui_loadsave.py modules/ui_postprocessing.py modules/ui_prompt_styles.py modules/ui_settings.py modules/ui_tempdir.py modules/ui_toprow.py modules/headless_ui.py modules/shared_ui_themes.py` -> passed.
- `PYTHONPYCACHEPREFIX=/tmp/gb10-a1111-pycache python3 -m py_compile $(find extensions extensions-builtin -type f -name "*.py" ! -path "*/__pycache__/*")` -> passed.
- `node --check extensions/sd-webui-incantations/javascript/dynthres_active.js extensions-builtin/canvas-zoom-and-pan/javascript/zoom.js extensions-builtin/mobile/javascript/mobile.js extensions-builtin/prompt-bracket-checker/javascript/prompt-bracket-checker.js` -> passed.
- `git diff --check` -> passed.

Next unchecked scope:
- Function-by-function review still remains for `modules/ui.py`, `modules/ui_components.py`, `modules/ui_component_patches.py`, `modules/ui_loadsave.py`, `modules/ui_postprocessing.py`, `modules/ui_prompt_styles.py`, `modules/ui_settings.py`, `modules/ui_tempdir.py`, `modules/ui_toprow.py`, `modules/headless_ui.py`, `modules/shared_ui_themes.py`, the specialized extra-network page/editor modules, and extension source trees under `extensions/` and `extensions-builtin/` (especially `extensions-builtin/Lora`, `extensions-builtin/LDSR`, `extensions/sd-webui-incantations`, `extensions/sd-webui-model-converter`, upscalers, Hypertile, soft-inpainting, and postprocessing-for-training). The broad source audit is still not complete.

### 2026-06-20 pass 23 - focused image save/delete/history recheck
Checked:
- `modules/images.py`: sequence numbering, forced filenames, extension fallback for oversized JPEG/WebP, callback-mutated filename handling, atomic temp-file save/replace behavior, 4chan downscale export, sidecar text writes, `already_saved_as` propagation, metadata readback, flatten/read/fix helpers, and PNG/JPEG/WebP/AVIF/GIF metadata paths. No new defect found in inspected paths.
- `modules/ui_common.py`: generation-info selection, save-selected versus save-all index handling, grid/sample infotext alignment, `log.csv` migration/padding, save-folder status escaping, zip packaging, output-panel open-folder routing, and download-file list construction. No new defect found.
- `modules/postprocessing.py`, `modules/extras.py`, `modules/ui_postprocessing.py`: extras source selection, batch/directory iteration, existing PNG info preservation, postprocessing save path, caption sidecar merge behavior, model-merger output naming/config-copy behavior, PNG-info display escaping path, and extras UI wiring. No new defect found beyond the caption-order fix already committed in pass 12.
- `modules/processing.py`, `modules/api/api.py`, `modules/ui_tempdir.py`, `modules/ui_prompt_styles.py`: generation save call sites, auxiliary before/after/mask image save paths, API `image_paths` response metadata, Gradio temp-file reuse for already-saved images, temp cleanup, and style delete/save handlers as related save/delete/history surfaces. No new defect found.
- Confirmed `modules/images_history.py` is not present in this checkout, and grep found no general image-history delete handler to audit in this slice.

Validation:
- `PYTHONPYCACHEPREFIX=/tmp/gb10-a1111-pycache python3 -m py_compile modules/images.py modules/ui_common.py modules/postprocessing.py modules/extras.py modules/ui_postprocessing.py modules/ui_tempdir.py modules/processing.py modules/api/api.py tests/test_save_serialization_contract.py tests/test_postprocessing_caption_contract.py tests/test_textual_inversion_preview_save_contract.py` -> passed.
- `PYTHONPYCACHEPREFIX=/tmp/gb10-a1111-pycache python3 -m pytest -q tests/test_save_serialization_contract.py tests/test_postprocessing_caption_contract.py tests/test_textual_inversion_preview_save_contract.py` -> passed, 6 tests; existing pytest warning remains `Unknown config option: base_url`.
- `git diff --check` -> passed.

Next unchecked scope:
- Continue function-by-function review of remaining UI construction modules and extension trees not yet deeply audited: `modules/ui.py`, `modules/ui_components.py`, `modules/ui_component_patches.py`, `modules/ui_toprow.py`, `modules/headless_ui.py`, `modules/shared_ui_themes.py`, specialized extra-network page/editor modules, and extension source trees under `extensions/` and `extensions-builtin/` especially `extensions-builtin/Lora`, `extensions-builtin/LDSR`, Hypertile, soft-inpainting, upscalers, postprocessing-for-training, `extensions/sd-webui-incantations`, and `extensions/sd-webui-model-converter`. Another slice is needed for the broad audit.

### 2026-06-20 pass 24 - Lora and LDSR extension generation-quality audit
Checked:
- `extensions-builtin/Lora/networks.py`: diffusers-to-CompVis key conversion, SD/SDXL layer-name assignment, LoRA state-dict grouping, bundled textual-inversion handling, in-memory cache invalidation by size/mtime, alias/name/hash lookup, active-network loading, LoRA application/restoration for Linear/Conv2d/GroupNorm/LayerNorm/MultiheadAttention/SD3 QKV, inpainting-channel padding, functional fallback, MXFP8/NVFP4 active-config signatures, transactional quantized LoRA preparation, source signature invalidation, file inventory refresh, and AddNet infotext migration. No defect found in inspected paths.
- `extensions-builtin/Lora/network.py`, `network_lora.py`, `network_hada.py`, `network_lokr.py`, `network_oft.py`, `network_glora.py`, `network_ia3.py`, `network_full.py`, and `network_norm.py`: metadata/hash/version detection, alias preference/forbidden-alias behavior, module shape inference, alpha/scale/dyn-dim/multiplier math, DoRA decomposition, conventional/CP/Kronecker/OFT/BOFT/GLora/IA3/full/norm weight reconstruction, bias handling, dtype/device movement, and unsupported target errors. No defect found.
- `extensions-builtin/Lora/extra_networks_lora.py`, `lora.py`, `lora_patches.py`, `scripts/lora_script.py`, `ui_extra_networks_lora.py`, `ui_edit_user_metadata.py`, and `preload.py`: prompt parsing activation path, default sd_lora injection, TE/UNet multiplier and dynamic-dim parsing, fatal quantized-prep stop behavior, infotext hash emission and paste replacement, API listing/refresh, patch install/unload, UI card prompt construction, preview/metadata paths, SD-version filtering, user metadata save/load, random activation prompt tags, and lora/lyco directory options. No generation-quality or operator-trust defect found.
- `extensions-builtin/LDSR/scripts/ldsr_model.py`, `extensions-builtin/LDSR/ldsr_model_arch.py`, `sd_hijack_autoencoder.py`, `sd_hijack_ddpm_v1.py`, and `vqvae_quantize.py`: LDSR model/yaml discovery and download fallback, old model rename and invalid yaml cleanup, safetensors/ckpt state loading, cache key invalidation by model/yaml/half/device/channels-last settings, hijack/eval flow, target-scale downsample math, padding/crop restoration, conditioning tensor construction, DDIM sample/decode flow, VQ/DDPM compatibility shims, and vector quantizer paths. No defect found in inspected paths.
- Related direct interactions in `modules/extra_networks.py`, `modules/ui_extra_networks.py`, `modules/modelloader.py`, and previously fixed `modules/sd_models.py` path-boundary logic were rechecked where they affect LoRA/LDSR parsing, previews, metadata, model discovery, and selected model identity.

Validation:
- `PYTHONPYCACHEPREFIX=/tmp/gb10-a1111-pycache python3 -m py_compile extensions-builtin/Lora/network.py extensions-builtin/Lora/networks.py extensions-builtin/Lora/network_lora.py extensions-builtin/Lora/network_hada.py extensions-builtin/Lora/network_lokr.py extensions-builtin/Lora/network_oft.py extensions-builtin/Lora/network_glora.py extensions-builtin/Lora/network_ia3.py extensions-builtin/Lora/network_full.py extensions-builtin/Lora/network_norm.py extensions-builtin/Lora/extra_networks_lora.py extensions-builtin/Lora/ui_extra_networks_lora.py extensions-builtin/Lora/ui_edit_user_metadata.py extensions-builtin/Lora/lora_patches.py extensions-builtin/Lora/scripts/lora_script.py extensions-builtin/Lora/preload.py extensions-builtin/LDSR/scripts/ldsr_model.py extensions-builtin/LDSR/ldsr_model_arch.py extensions-builtin/LDSR/sd_hijack_autoencoder.py extensions-builtin/LDSR/sd_hijack_ddpm_v1.py extensions-builtin/LDSR/vqvae_quantize.py modules/extra_networks.py modules/ui_extra_networks.py modules/modelloader.py modules/sd_models.py` -> passed.
- `PYTHONPYCACHEPREFIX=/tmp/gb10-a1111-pycache python3 -m pytest -q tests/test_extra_networks_path_contract.py tests/test_extra_networks_metadata_contract.py tests/test_sd_models_checkpoint_info_contract.py` -> passed, 6 tests; existing pytest warning remains `Unknown config option: base_url`.
- `git diff --check` -> passed.

Next unchecked scope:
- Continue function-by-function review of remaining extension trees not yet deeply audited: Hypertile, soft-inpainting, upscalers, postprocessing-for-training, `extensions/sd-webui-incantations`, `extensions/sd-webui-model-converter`, and the remaining UI construction modules (`modules/ui.py`, `modules/ui_components.py`, `modules/ui_component_patches.py`, `modules/ui_toprow.py`, `modules/headless_ui.py`, `modules/shared_ui_themes.py`, and specialized extra-network editor/page modules). Another slice is still needed for the broad audit.


### 2026-06-20 pass 25 - Hypertile, soft-inpainting, upscalers, and postprocessing-for-training
Checked:
- `extensions-builtin/hypertile/hypertile.py` and `extensions-builtin/hypertile/scripts/hypertile_script.py`: divisor selection, nearest aspect-ratio candidate search, U-Net/VAE attention tensor rearrangement and restoration, depth/tile-size scaling, seed behavior, hires second-pass enablement, infotext emission, SDXL layer maps, and xyz/settings wiring. No defect found in inspected paths.
- `extensions-builtin/soft-inpainting/scripts/soft_inpainting.py`: latent blend magnitude preservation, sigma-scaled negative-mask conversion, adaptive image mask generation, histogram filtering, Gaussian kernel construction, paste-region uncrop handling, overlay alpha application, generation-parameter recording, and inpaint-mode gating. No defect found in inspected paths.
- `modules/upscaler.py`, `modules/upscaler_utils.py`, `modules/esrgan_model.py`, `modules/realesrgan_model.py`, `modules/dat_model.py`, `modules/hat_model.py`, `extensions-builtin/SwinIR/scripts/swinir_model.py`, and `extensions-builtin/ScuNET/scripts/scunet_model.py`: scale target rounding, repeated upscale stop conditions, selected-model lookup/download/cache behavior, dtype/device selection, PIL-grid split/recombine, tensor-grid overlap weighting, and scale factors for ESRGAN/RealESRGAN/DAT/HAT/SwinIR/ScuNET. Found one tiled-overlap geometry defect.
- `extensions-builtin/postprocessing-for-training/scripts/postprocessing_caption.py`, `postprocessing_create_flipped_copies.py`, `postprocessing_focal_crop.py`, `postprocessing_split_oversized.py`, `postprocessing_autosized_crop.py`, and related `scripts/postprocessing_upscale.py`: caption merge order, flipped-copy generation, focal crop settings flow, oversized split aspect math, auto-sized crop candidate selection, postprocess upscale target metadata, max-side limiting, crop-to-fit composition, second-upscaler blending, and cache key construction. No additional defect found.

Findings/fixes:
- `modules/images.split_grid()` and `modules/upscaler_utils.tiled_upscale_2()` trusted tile overlap values even when overlap was equal to or larger than the effective tile size. UI settings allow combinations such as small tile size with larger overlap; the PIL path could compute invalid/negative stepping and the tensor path could hit a zero stride or leave most of the image uncovered. Clamped overlap to the largest valid value below the effective tile/image dimensions in both shared tiling helpers.
- Added `tests/test_upscaler_tiling_contract.py` covering oversize-overlap grid clamping and tensor tiled upscale behavior. The tensor case skips under system Python when torch is unavailable, but will run in the normal torch runtime.

Validation:
- `PYTHONPYCACHEPREFIX=/tmp/gb10-a1111-pycache python3 -m py_compile modules/images.py modules/upscaler_utils.py modules/upscaler.py modules/esrgan_model.py modules/realesrgan_model.py extensions-builtin/SwinIR/scripts/swinir_model.py extensions-builtin/ScuNET/scripts/scunet_model.py extensions-builtin/hypertile/hypertile.py extensions-builtin/hypertile/scripts/hypertile_script.py extensions-builtin/soft-inpainting/scripts/soft_inpainting.py extensions-builtin/postprocessing-for-training/scripts/postprocessing_caption.py extensions-builtin/postprocessing-for-training/scripts/postprocessing_create_flipped_copies.py extensions-builtin/postprocessing-for-training/scripts/postprocessing_focal_crop.py extensions-builtin/postprocessing-for-training/scripts/postprocessing_split_oversized.py extensions-builtin/postprocessing-for-training/scripts/postprocessing_autosized_crop.py scripts/postprocessing_upscale.py tests/test_upscaler_tiling_contract.py` -> passed.
- `PYTHONPYCACHEPREFIX=/tmp/gb10-a1111-pycache python3 -m pytest -q tests/test_upscaler_tiling_contract.py tests/test_postprocessing_caption_contract.py` -> passed, 3 passed / 1 skipped because system Python lacks torch; existing pytest warning remains `Unknown config option: base_url`.
- `git diff --check` -> passed.

Next unchecked scope:
- Continue with `extensions/sd-webui-incantations`, `extensions/sd-webui-model-converter`, and the remaining UI construction modules (`modules/ui.py`, `modules/ui_components.py`, `modules/ui_component_patches.py`, `modules/ui_toprow.py`, `modules/headless_ui.py`, `modules/shared_ui_themes.py`, and specialized extra-network editor/page modules). Another slice is still needed for the broad audit.

### 2026-06-20 pass 26 - Incantations and model-converter extension audit
Checked:
- `extensions/sd-webui-incantations/scripts/incantation_base.py`, `ui_wrapper.py`, and `scripts/incant_utils/module_hooks.py`: always-on Incantations wrapper dispatch, submodule argument slicing, timing aggregation, xyz-axis registration, active component inventory, field cleanup idempotence, model-module discovery, and local forward-hook handle use. No defect found in inspected paths.
- `extensions/sd-webui-incantations/scripts/dynamic_thresholding.py`, `dynthres_core.py`, and `dynthres_unipc.py`: sampler substitution/restore lifecycle, secondary latent sampler handling, UniPC wrapper step accounting, K-diffusion CFG combiner path, relative-guidance aggregation, fp16/bf16-friendly statistics, percentile/std/ad scaling, scheduler modes, experiment mode 3 matrix cache, ragged multi-cond rejection, and infotext/xyz parameter flow. No defect found in inspected paths.
- `extensions/sd-webui-incantations/scripts/pag.py` and `scripts/cfg_combiner.py`: PAG/CFG interval UI parameters, callback registration/removal, cross-attention hook install/removal, hidden PAG denoise pass, SDXL dict conditioning, cond/uncond token padding and split batching, SEG suspension during hidden PAG pass, scheduled CFG scale lookup, base CFG delegation so Dynamic Thresholding remains composed, SANF blending, and wrapper restore ownership. No defect found in inspected paths.
- `extensions/sd-webui-incantations/scripts/smoothed_energy_guidance.py` and `javascript/dynthres_active.js`: SEG callback lifecycle, to_q hook enablement, unpaired-batch skip behavior, attention geometry inference, Gaussian kernel caching/clamping, infinite blur path, xyz settings, and Dynamic Thresholding enabled-badge UI behavior. No defect found in inspected paths.
- `extensions/sd-webui-model-converter/scripts/convert.py`: safe tensor-only checkpoint load, output filename/path containment and overwrite refusal, checkpoint/LoRA precision conversion, float8 export guard, component action and precision normalization, CLIP key/position_id repair, known-junk cleanup, VAE baking, nonfinite scan/repair, checkpoint and LoRA doctor reports, metadata provenance/identity fields, LoRA payload-key preservation, and refresh after conversion. No defect found in inspected paths.
- `extensions/sd-webui-model-converter/scripts/ui.py`: Gradio tab wiring, checkpoint/LoRA/VAE refresh controls, request payload mapping, API option/convert endpoints, Pydantic v1/v2 payload compatibility, and UI/API consistency with converter defaults. No defect found in inspected paths.
- Related clear-cond-cache search in these trees and direct runtime/API touchpoints found no Incantations/model-converter-owned clear-cond-cache UI or API surface to fix in this slice.

Validation:
- `PYTHONPYCACHEPREFIX=/tmp/gb10-a1111-pycache python3 -m py_compile extensions/sd-webui-incantations/dynthres_core.py extensions/sd-webui-incantations/dynthres_unipc.py extensions/sd-webui-incantations/scripts/incantation_base.py extensions/sd-webui-incantations/scripts/ui_wrapper.py extensions/sd-webui-incantations/scripts/incant_utils/module_hooks.py extensions/sd-webui-incantations/scripts/dynamic_thresholding.py extensions/sd-webui-incantations/scripts/pag.py extensions/sd-webui-incantations/scripts/cfg_combiner.py extensions/sd-webui-incantations/scripts/smoothed_energy_guidance.py extensions/sd-webui-model-converter/scripts/convert.py extensions/sd-webui-model-converter/scripts/ui.py extensions/sd-webui-incantations/tests/test_guidance_core.py extensions/sd-webui-model-converter/tests/test_convert.py` -> passed.
- `node --check extensions/sd-webui-incantations/javascript/dynthres_active.js` -> passed.
- `PYTHONPYCACHEPREFIX=/tmp/gb10-a1111-pycache python3 -m pytest -q extensions/sd-webui-incantations/tests/test_guidance_core.py extensions/sd-webui-model-converter/tests/test_convert.py` -> blocked during collection because system Python has no `torch` installed (`ModuleNotFoundError: No module named 'torch'`); no repo venv was present at `/home/kklouzal/stable-diffusion-webui/venv` or `.venv`.
- `git diff --check` -> passed.

Next unchecked scope:
- Continue with the remaining UI construction modules: `modules/ui.py`, `modules/ui_components.py`, `modules/ui_component_patches.py`, `modules/ui_toprow.py`, `modules/headless_ui.py`, `modules/shared_ui_themes.py`, specialized extra-network editor/page modules, and any remaining extension source not explicitly audited after pass 26. Another slice is still needed for the broad audit.


### 2026-06-20 pass 27 - UI construction and remaining extra-network/editor glue
Checked:
- `modules/ui.py`: txt2img and img2img construction order, `txt2img_inputs`/`img2img_args` mapping into generation functions, hires controls and paste fields, img2img tab/source selection, inpaint mask/fill/full-res controls, resize-to/resize-by behavior, interrogate routing, progress restore, output panel wiring, PNG info paste routing, train preview parameter capture, settings/loadsave integration, checkpoint merger setup, and internal API helpers. No generation-parameter ordering defect found.
- `modules/ui_components.py`, `modules/ui_component_patches.py`, and `modules/ui_toprow.py`: form component parents/block names, `InputAccordion` boolean state behavior and id de-duplication, component callback before/after hooks, tooltip/config patching, compact/classic prompt rows, submit/interrupt/skip behavior, prompt-image extraction, style apply, clear prompt, token counters, and img2img interrogate buttons. No generation-quality defect found in inspected paths.
- `modules/headless_ui.py` and `modules/shared_ui_themes.py`: inert component default/value/choice metadata for API/headless script UI construction, event-chain no-op behavior, Gradio module shims, theme cache/load fallback, and theme variable resolution. No defect found in inspected paths.
- `modules/ui_extra_networks.py`, `modules/ui_extra_networks_user_metadata.py`, `modules/ui_extra_networks_checkpoints.py`, `modules/ui_extra_networks_checkpoints_user_metadata.py`, `modules/ui_extra_networks_hypernets.py`, and `modules/ui_extra_networks_textual_inversion.py`: extra-network page registration/order, preview fetch/cover-image/metadata/get-single-card endpoints, path containment, card/tree/dirs HTML generation, checkpoint/hypernet/textual-inversion prompts and search terms, preview replacement, user metadata save/load, preferred VAE selection/reload behavior, local preview paths, and gallery-index preview save behavior. Found one operator-trust HTML/JS escaping defect.
- `modules/extra_networks.py` and `modules/extra_networks_hypernet.py`: prompt parsing, activation/deactivation dispatch, alias handling, user metadata JSON reads, and hypernetwork multiplier/name application as related extra-network UI/API parameter flow. No additional defect found.
- Remaining extension glue after pass 26: `extensions-builtin/LDSR/preload.py`, `extensions-builtin/ScuNET/preload.py`, `extensions-builtin/SwinIR/preload.py`, `extensions-builtin/canvas-zoom-and-pan/scripts/hotkey_config.py`, `extensions-builtin/extra-options-section/scripts/extra_options_section.py`, `extensions-builtin/canvas-zoom-and-pan/javascript/zoom.js`, `extensions-builtin/mobile/javascript/mobile.js`, `extensions-builtin/prompt-bracket-checker/javascript/prompt-bracket-checker.js`, and `javascript/generationParams.js`. The preloads only add model-dir CLI defaults; canvas/mobile/bracket/generationParams JS is UI-only state/feedback; extra-options preserves explicit `p.override_settings` while copying selected UI setting values into generation. No additional defect found.
- Extension source inventory was rechecked against earlier passes: OpenClaw denoise-ramp and multi-sampler were covered in pass 6, OpenClaw clear-cond-cache in pass 8, core JS in pass 20, UI triage/inventory in pass 22, Lora/LDSR in pass 24, Hypertile/soft-inpainting/upscalers/postprocessing-for-training in pass 25, and Incantations/model-converter in pass 26.

Findings/fixes:
- Extra-network card/tree HTML interpolated local filenames, search terms, labels, hashes, and save-preview JavaScript with incomplete escaping. A model/checkpoint/embedding path containing quote-like characters could corrupt DOM attributes or save-preview onclick JavaScript and mislead operators inspecting or saving extra-network cards. Added `html_attr()` for attribute-safe escaping, escaped search terms/tree labels/data attributes, and changed save-preview JS construction to use existing JSON quoting via `quote_js()`.
- Added `tests/test_extra_networks_metadata_contract.py::test_extra_network_card_html_escapes_attribute_and_js_paths` to lock the escaping/quoting contract.

Validation:
- `PYTHONPYCACHEPREFIX=/tmp/gb10-a1111-pycache python3 -m py_compile modules/ui.py modules/ui_components.py modules/ui_component_patches.py modules/ui_toprow.py modules/headless_ui.py modules/shared_ui_themes.py modules/ui_extra_networks.py modules/ui_extra_networks_user_metadata.py modules/ui_extra_networks_checkpoints.py modules/ui_extra_networks_checkpoints_user_metadata.py modules/ui_extra_networks_hypernets.py modules/ui_extra_networks_textual_inversion.py modules/extra_networks.py modules/extra_networks_hypernet.py extensions-builtin/LDSR/preload.py extensions-builtin/ScuNET/preload.py extensions-builtin/SwinIR/preload.py extensions-builtin/canvas-zoom-and-pan/scripts/hotkey_config.py extensions-builtin/extra-options-section/scripts/extra_options_section.py tests/test_extra_networks_metadata_contract.py` -> passed.
- `node --check javascript/generationParams.js extensions-builtin/canvas-zoom-and-pan/javascript/zoom.js extensions-builtin/mobile/javascript/mobile.js extensions-builtin/prompt-bracket-checker/javascript/prompt-bracket-checker.js` -> passed.
- `PYTHONPYCACHEPREFIX=/tmp/gb10-a1111-pycache python3 -m pytest -q tests/test_extra_networks_metadata_contract.py tests/test_extra_networks_path_contract.py` -> passed, 5 tests; existing pytest warning remains `Unknown config option: base_url`.
- `git diff --check` -> passed.

Next unchecked scope:
- The named broad source audit scope appears exhausted by passes 1-27. Recommend a final closeout/validation slice to reconcile the ledger against the full source inventory, rerun the highest-value contract tests in the available runtime, and produce a final audit summary; do not invent another open-ended discovery slice unless new source or runtime evidence appears.

### 2026-06-20 final closeout - inventory reconciliation and aggregate validation
Scope covered:
- Reconciled the ledger against the broad repo inventory for `modules/`, built-in `scripts/`, the audited OpenClaw/Incantations/model-converter extension trees, `extensions-builtin` generation/runtime helpers, JavaScript UI state files, tests, and repo config manifests. No obvious high-value generation/runtime source island remains outside passes 1-27.
- Core generation coverage includes processing, prompt parsing, CFG denoiser batching, k-diffusion/CompVis/LCM/DDIM/PLMS/UniPC/restart samplers, schedulers, RNG/subseed blending, image/mask/latent geometry, hires/img2img/inpaint paths, VAE/UNet/model load and reload, precision/cache metadata, API txt2img/img2img and progress contracts, and generation-altering extensions/scripts.
- Broadened runtime/operator-safety coverage includes config/options/load-save, image save/history/infotext, extra-network metadata/cards, UI construction glue, training/textual-inversion/hypernetwork/checkpoint merger helpers, face restoration/interrogate/upscalers/postprocessing, launch/container helpers, and related lightweight contracts.

Key fixes committed during the audit:
- Skip-CFG image conditioning alignment in `modules/sd_samplers_cfg_denoiser.py`.
- Subseed `slerp` singularity handling in `modules/rng.py`.
- UI defaults/config backup/extension/extra-network HTML and JavaScript escaping hardening.
- API progress response contract for `current_task`.
- Checkpoint display-name root-boundary handling.
- Textual-inversion face detection array truthiness and hypernetwork create return contract.
- Postprocessing caption merge behavior and upscaler tile-overlap clamping.
- Multiple focused contract tests and test isolation repairs for the audited fixes.

Final validation summary:
- `git status --short --branch` -> `## latest...origin/latest [ahead 33]` plus only untracked `.openclaw-audit/logs/`.
- `git diff --check` -> passed (`diff_check_rc:0`).
- `PYTHONPYCACHEPREFIX=/tmp/gb10-a1111-pycache python3` py-compiled all practical Python files found by `find . -path ./.git -prune -o -path ./venv -prune -o -path ./.venv -prune -o -path ./repositories -prune -o -path ./models -prune -o -path ./outputs -prune -o -path ./__pycache__ -prune -o -name "*.py" -print` -> 284 files passed.
- `PYTHONPYCACHEPREFIX=/tmp/gb10-a1111-pycache python3 -m pytest -q tests` -> passed, 35 passed / 1 skipped / 1 warning (`Unknown config option: base_url`).
- `node --check script.js javascript/*.js extensions/sd-webui-incantations/javascript/dynthres_active.js extensions-builtin/canvas-zoom-and-pan/javascript/zoom.js extensions-builtin/mobile/javascript/mobile.js extensions-builtin/prompt-bracket-checker/javascript/prompt-bracket-checker.js` -> passed.

Remaining validation gaps and environment constraints:
- System Python on GB10 lacks `torch`, so torch-bound extension tests for denoise-ramp, multi-sampler, Incantations, and model-converter fail during collection with `ModuleNotFoundError: No module named 'torch'`.
- System Python also lacks `fastapi`, so `extensions/openclaw-clear-cond-cache/tests/test_openclaw_clear_cond_cache.py` fails during collection/import with `ModuleNotFoundError: No module named 'fastapi'`.
- The existing `local/gb10-a1111:latest` Docker validation image lacks `pytest` for pytest-style extension tests (`/usr/local/bin/python: No module named pytest`), and no repo `venv` or `.venv` exists to provide the full runtime test surface.
- Full live WebUI model-load/generation smoke tests, GPU/TorchAO quantization cache execution, and CUDA graph/runtime integration remain outside this closeout because the available validation surface does not provide a complete torch/pytest/runtime environment without mutating the machine.

Final status:
- Audit scope is complete for the named broad source review. No further source-audit slice is recommended unless a real runtime test environment is provisioned or new failing evidence appears.

## Slice 6 owned extension/controller audit — 2026-07-16 UTC

Baseline SHAs at slice start remained:
- WebUI: `/home/kklouzal/stable-diffusion-webui` branch `latest` HEAD `21eeaafc2b80f0153d3064eec3e21abc9e03c1fe`.
- Controller: `/home/kklouzal/a1111-controller` branch `main` HEAD `92f6ab02c33f3ad998f5054e72009a9ace2c1be5`.

Scope completed in this slice:
- Line-by-line audit of remaining GB10-owned extension surfaces: `sd-webui-incantations` (Dynamic Thresholding/CFG-Fix, PAG, SEG, CFG combiner, module hook glue, UI wrapper, JavaScript badge helper) and `sd-webui-model-converter` (conversion math, dtype/device/shape handling, file safety, API/UI callback contracts, serialization/metadata, LoRA doctor/repair path, tests).
- Rechecked uncovered paths in `openclaw-clear-cond-cache`, `openclaw-denoise-ramp`, `openclaw-multi-sampler`, and `sd-webui-teacache`, especially cache target normalization, img2img init-cache lock/status behavior, generation queue locking, custom sampler registry locks, TeaCache global cache lock, disabled/fallback paths, dtype/device math, and tests.
- Audited `/home/kklouzal/a1111-controller/app.py` and `templates/index.html` extension-management contracts for model merge/model converter state, preset/snapshot/PNG import/export adjacency, stale UI state, default/range mismatches, bool/number coercion, async job state, viewport CSS, and API error handling.
- Ran repository-wide static searches over owned/custom code for deprecated/removed APIs and suspicious patterns: `np.bool/int/float/object`, old Pillow constants, `gr.update`, Pydantic v1/v2 boundaries, `@app.on_event`, bare exceptions, unsafe tensor/array truthiness, `torch.std`/quantile/division risk, `torch.load`, pickle/yaml/eval/exec, mutable globals/defaults, global mutation, `.cpu()/.numpy()`, JS `Number(...)`, `parseInt/parseFloat`, `JSON.parse`, `localStorage`, cache/lock/global patterns. Hits were manually triaged; changes below are the confirmed issues only.

Fixes made in this slice:
- `extensions/sd-webui-incantations/dynthres_core.py`: Dynamic Thresholding `STD` variability now uses `torch.std(..., unbiased=False)` for both per-channel and whole-latent reductions. This prevents PyTorch degrees-of-freedom warnings and NaN scaling for single-spatial-sample/tiny latent tensors while preserving dtype/device behavior.
- `extensions/sd-webui-incantations/scripts/dynamic_thresholding.py`: CFG rescale experiment path now uses population std (`unbiased=False`) for both denominator and reference std, avoiding NaN/DoF behavior for degenerate samples.
- `extensions/sd-webui-model-converter/scripts/convert.py`: `load_model()` now rejects non-mapping checkpoint payloads explicitly after safe `weights_only=True` load; `convert_single()` and LoRA cleanup now normalize string booleans instead of treating strings like `"false"` as truthy.
- `test/test_openclaw_cache_invalidation.py`: cache regression harness now initializes A1111 shared options through `modules.shared_init.initialize()` before importing processing/cache modules. This makes the suite executable in a dependency-complete transient A1111 environment rather than relying on partially imported globals.
- `/home/kklouzal/a1111-controller/app.py`: model-merge and model-converter API normalization now uses `_coerce_bool()` for API string booleans and validates converter mode/pruning/LoRA precision before launching work.
- `/home/kklouzal/a1111-controller/test/test_multi_sampler_preset_template.py`: retired the stale model-merge viewport CSS assertion and replaced it with assertions for the current real contract (`#modelMergePanel` column shell, shell-local vertical scroll, max-content grid rows, non-scroll inner settings grid).
- Added/updated tests in `extensions/sd-webui-incantations/tests/test_guidance_core.py`, `extensions/sd-webui-model-converter/tests/test_convert.py`, `/home/kklouzal/a1111-controller/test/test_model_management.py`, and `/home/kklouzal/a1111-controller/test/test_multi_sampler_preset_template.py`.

Coverage ledger for this slice (`audited` = inspected in this slice and either fixed or triaged no-change):
- `extensions/openclaw-clear-cond-cache/README.md` — audited/no change.
- `extensions/openclaw-clear-cond-cache/scripts/openclaw_clear_cond_cache.py` — audited/no new change beyond existing WIP.
- `extensions/openclaw-clear-cond-cache/tests/test_openclaw_clear_cond_cache.py` — audited/no change.
- `extensions/openclaw-denoise-ramp/scripts/openclaw_denoise_ramp.py` — audited/no new change beyond existing WIP.
- `extensions/openclaw-denoise-ramp/tests/test_openclaw_denoise_ramp.py` — audited/no new change beyond existing WIP.
- `extensions/openclaw-multi-sampler/scripts/openclaw_multi_sampler.py` — audited/no new change beyond existing WIP.
- `extensions/openclaw-multi-sampler/tests/test_openclaw_multi_sampler.py` — audited/no new change beyond existing WIP.
- `extensions/sd-webui-incantations/dynthres_core.py` — audited/fixed.
- `extensions/sd-webui-incantations/dynthres_unipc.py` — audited/no change.
- `extensions/sd-webui-incantations/javascript/dynthres_active.js` — audited/no change.
- `extensions/sd-webui-incantations/README.md` — audited/no change.
- `extensions/sd-webui-incantations/scripts/cfg_combiner.py` — audited/no change.
- `extensions/sd-webui-incantations/scripts/dynamic_thresholding.py` — audited/fixed.
- `extensions/sd-webui-incantations/scripts/incantation_base.py` — audited/no change.
- `extensions/sd-webui-incantations/scripts/incant_utils/module_hooks.py` — audited/no change.
- `extensions/sd-webui-incantations/scripts/pag.py` — audited/no change.
- `extensions/sd-webui-incantations/scripts/smoothed_energy_guidance.py` — audited/no change.
- `extensions/sd-webui-incantations/scripts/ui_wrapper.py` — audited/no change.
- `extensions/sd-webui-incantations/tests/test_guidance_core.py` — audited/test added.
- `extensions/sd-webui-model-converter/README.md` — audited/no change.
- `extensions/sd-webui-model-converter/scripts/convert.py` — audited/fixed.
- `extensions/sd-webui-model-converter/scripts/ui.py` — audited/no change.
- `extensions/sd-webui-model-converter/tests/test_convert.py` — audited/tests added.
- `extensions/sd-webui-teacache/README.md` — audited/no change.
- `extensions/sd-webui-teacache/scripts/teacache.py` — audited/no new change beyond existing WIP.
- `extensions/sd-webui-teacache/tests/test_teacache_session.py` — audited/no change.
- `modules/api/api.py` — audited/no new change beyond existing WIP.
- `modules/api/models.py` — audited/no new change beyond existing WIP.
- `modules/headless_ui.py` — audited/no new change beyond existing WIP.
- `modules/openclaw_cuda_graphs.py` — audited/no new change beyond existing WIP.
- `modules/processing.py` — audited/no new change beyond existing WIP; internal img2img init cache bool payload is stored as a bool in the guarded cache payload.
- `tests/test_teacache_extension.py` — audited/no new change beyond existing WIP.
- `test/test_openclaw_cache_invalidation.py` — audited/fixed harness.
- `webui.py` — audited/no new change beyond existing WIP.
- `/home/kklouzal/a1111-controller/app.py` — audited/fixed.
- `/home/kklouzal/a1111-controller/templates/index.html` — audited/no template code change; CSS contract verified by updated test.
- `/home/kklouzal/a1111-controller/tests/test_controlnet_payload.py` — audited/no change.
- `/home/kklouzal/a1111-controller/test/test_lora_normalization.py` — audited/no change.
- `/home/kklouzal/a1111-controller/test/test_model_management.py` — audited/tests added.
- `/home/kklouzal/a1111-controller/test/test_multi_sampler_preset_template.py` — audited/stale CSS assertion fixed.
- `/home/kklouzal/a1111-controller/test/test_teacache_payload.py` — audited/no change.

Validation run:
- Controller: `cd /home/kklouzal/a1111-controller && .venv/bin/python -m pytest -q` -> `52 passed in 1.07s`.
- WebUI dependency-complete transient harness: `docker run --rm --network none --entrypoint bash -e PYTHONDONTWRITEBYTECODE=1 -v /home/kklouzal/stable-diffusion-webui:/hostrepo:ro -v /home/kklouzal/a1111-controller/.venv/lib/python3.12/site-packages:/pytest-site:ro local/gb10-a1111:latest -lc "set -e; cp -a /opt/stable-diffusion-webui /tmp/work; cp -a /hostrepo/. /tmp/work/; cd /tmp/work; PYTHONPATH=/pytest-site python -m pytest -q test/test_openclaw_cache_invalidation.py tests/test_teacache_extension.py extensions/openclaw-clear-cond-cache/tests/test_openclaw_clear_cond_cache.py extensions/openclaw-denoise-ramp/tests/test_openclaw_denoise_ramp.py extensions/openclaw-multi-sampler/tests/test_openclaw_multi_sampler.py extensions/sd-webui-model-converter/tests/test_convert.py extensions/sd-webui-incantations/tests/test_guidance_core.py extensions/sd-webui-teacache/tests/test_teacache_session.py"` -> `78 passed, 46 warnings, 6 subtests passed in 8.28s`.
- Compile/diff: transient container `py_compile` for touched WebUI files/tests passed; controller `.venv/bin/python -m py_compile app.py test/test_model_management.py test/test_multi_sampler_preset_template.py` passed; `git diff --check` passed in both repos.

Warnings/triage notes:
- Warnings-as-errors was not practical for the full transient WebUI suite because current dependencies emit external deprecation/compatibility warnings (`torch.jit.script` deprecation, `jsonschema.RefResolver`, and a requests dependency version warning from the mounted pytest site-packages). No new GB10-owned warning was observed in the focused results.
- Static-search residual hits in owned/custom code are either intentional API/compatibility paths, already-normalized `_coerce_bool()`/internal bool summaries, NumPy dtype conversions (not removed aliases), or test scaffolding. No additional confirmed defect was patched.
- No live generation, rebuild, redeploy, restart, commit, or push was performed.

Continuation assessment:
- For the owned extension/controller surfaces assigned in this slice, no further audit slice is required before integration based on current evidence and passing focused regressions.
- Broader non-extension A1111 core surfaces remain outside this slice and are still listed in earlier audit-continuation notes; those should remain separate if Schwi wants truly repository-wide core coverage beyond GB10-owned/custom surfaces.

## Slice 7 core generation pipeline audit pass — 2026-07-16 UTC

Baseline/status:
- A1111 repo: `/home/kklouzal/stable-diffusion-webui`, branch `latest`, HEAD `21eeaafc2b80f0153d3064eec3e21abc9e03c1fe` at slice start.
- Controller repo inspected only for status: `/home/kklouzal/a1111-controller`, branch `main`, HEAD `92f6ab02c33f3ad998f5054e72009a9ace2c1be5`; no controller edits.
- Existing dirty WIP was preserved. New slice-7 code/test edits are limited to `modules/sd_samplers_kdiffusion.py` and `test/test_openclaw_kdiffusion_sigmas_cache.py`; this ledger was appended.
- Runtime package evidence from `gb10-a1111-latest`: Python 3.12.3, torch `2.14.0.dev20260709+cu132`, CUDA 13.2, numpy 2.5.1, pydantic 1.10.26, transformers 5.13.0, open_clip_torch 3.3.0, safetensors 0.8.0, fastapi 0.94.0. `pytest` is not installed in the image, so focused tests were run through `unittest`/direct scripts.

Path-level coverage ledger (slice-7 static/manual pass):
- Deep/line-focused: `modules/sd_schedulers.py`, `modules/sd_samplers_kdiffusion.py`, `modules/sd_samplers_timesteps.py`, `modules/sd_samplers_timesteps_impl.py`, `modules/sd_samplers_common.py`, `modules/sd_samplers_cfg_denoiser.py`, `modules/processing.py` core conditioning/init/sample/decode/highres/img2img sections, `modules/prompt_parser.py`, `modules/devices.py`.
- Scoped static grep/API/deprecation coverage: `modules/processing_scripts/{comments.py,refiner.py,sampler.py,seed.py}`, `modules/sd_samplers.py`, `modules/sd_samplers_extra.py`, `modules/sd_samplers_lcm.py`, `modules/sd_samplers_compvis.py`, `modules/sd_hijack*.py`, `modules/sd_models*.py`, `modules/lowvram.py`, `modules/shared_state.py`, `modules/api/{api.py,models.py}`.
- Expected scoped file absent in this fork: `modules/generation_parameters_copypaste.py`.
- Static searches covered timestep/sigma arithmetic, scheduler conversions, `steps`/`t_enc` boundaries, seed/subseed indexing, tensor truthiness, `.item()`/CPU sync sites, dtype/device/autocast transitions, cache key inputs, deprecated NumPy/Torch APIs, `torch.load`/checkpoint loading, and prompt/conditioning shape paths.

Confirmed finding fixed:
- Severity: high correctness for model-switch sessions using cached K-diffusion schedules.
- Evidence: `KDiffusionSampler.get_sigmas()` cached schedule tensors by sampler/scheduler/steps/options and sigma endpoints, but omitted model schedule identity. The `Align Your Steps` scheduler chooses different base sigma tables from `shared.sd_model.is_sdxl`; other inner-model schedulers can also depend on checkpoint/timestep mapping beyond endpoints. After an SD1.x/SD2 model generated an AYS cache entry, switching to SDXL with the same sampler/scheduler/steps/options could reuse the stale non-SDXL sigma table.
- Remediation: added `_model_schedule_cache_signature()` to `modules/sd_samplers_kdiffusion.py` and included checkpoint identity, SDXL/SD2 flags, parameterization, and inner-model sigma shape/endpoints in the `kdiffusion_sigmas` cache params.
- Regression: new `test/test_openclaw_kdiffusion_sigmas_cache.py` stubs K-diffusion/A1111 modules, forces an AYS-like scheduler to return different sigmas for SD1.x vs SDXL, and verifies the cache misses twice and returns the correct per-model tensor.

Validation:
- `docker run --rm --entrypoint bash -v /home/kklouzal/stable-diffusion-webui:/work -w /work local/gb10-a1111:latest -lc "python test/test_openclaw_kdiffusion_sigmas_cache.py && python test/test_openclaw_schedulers.py && python tests/test_openclaw_generation_profile.py"` -> passed, 1 + 6 + 7 tests OK.
- `docker run --rm --entrypoint bash -v /home/kklouzal/stable-diffusion-webui:/work -v /tmp/gb10_slice7_pycompile.py:/tmp/gb10_slice7_pycompile.py:ro -w /work local/gb10-a1111:latest -lc "python /tmp/gb10_slice7_pycompile.py"` -> `py_compile ok 14` for the edited sampler plus critical processing/scheduler/model/device files.
- `git diff --check -- modules/sd_samplers_kdiffusion.py test/test_openclaw_kdiffusion_sigmas_cache.py .openclaw-audit/gb10-a1111-latest-math-logic-audit.md` -> passed.

Unresolved / next prioritized core slice:
- This pass began the broader non-extension audit and fixed the confirmed K-diffusion cache invalidation defect. Remaining high-value coverage should continue with a second pass over the full `modules/processing.py` highres/img2img resize/mask decode tail, `modules/sd_models.py` checkpoint/VAE reload lifecycle under dirty WIP, and full `modules/sd_hijack_optimizations.py` attention backend shape/chunk math. No live generation, rebuild, redeploy, restart, commit, or push was performed.
- Additional safe non-generation script smoke: `docker run --rm --entrypoint bash -v /home/kklouzal/stable-diffusion-webui:/work -w /work local/gb10-a1111:latest -lc "python test/test_infotext_api_mappings.py && echo infotext_ok && python test/test_api_script_defaults.py && echo api_script_defaults_ok"` -> `infotext_ok`, `api_script_defaults_ok`.
- External dependency warning: `test/test_openclaw_multi_sampler.py` could not be run by direct Python in the runtime image because it imports `pytest`, and the image reports `/usr/local/bin/python: No module named pytest`.

### 2026-07-16 pass 25 - core slice 8 processing/model/VAE/attention lifecycle audit
Baseline and preservation:
- Authoritative tree: GB10 `/home/kklouzal/stable-diffusion-webui`, branch `latest`, HEAD `21eeaafc2b80f0153d3064eec3e21abc9e03c1fe` at start of slice.
- Pre-existing dirty WIP was preserved. This slice intentionally touched only `modules/sd_models.py`, `modules/sd_vae.py`, `modules/sd_hijack_optimizations.py`, `test/test_openclaw_cache_invalidation.py`, new `test/test_openclaw_attention_slice_bounds.py`, and this audit note.

Path-level coverage:
- `modules/processing.py`: reviewed prompt setup/cache keys, seed/subseed batching, infotext metadata, VAE encode/decode/NaN fallback, txt2img high-res target sizing/truncation/second-pass conditioning, img2img full-res mask/crop/paste/overlay/color-correction path, latent mask resize, image-conditioning branches, img2img init-cache key/restore/store lifecycle. No new code change here in this slice; prior img2img init-cache WIP remains intact.
- `modules/sd_models.py`: reviewed checkpoint identity/aliasing, safetensors/ckpt state-dict loading, checkpoint state-dict cache, TorchAO MXFP8/NVFP4 forced reload/cache bypass, model reuse/reload/switch, device movement, VAE handoff, token merging, and config repair.
- `modules/sd_models_config.py`: reviewed config inference probes and state-dict heuristics for SD1/SD2/SDXL/SD3/SSD/v-parameterization. No confirmed defect changed.
- `modules/sd_vae.py` and `modules/sd_vae_approx.py`: reviewed VAE resolution order, base-VAE storage/restore, VAE state-dict cache, dtype movement, reload lifecycle, and approx decoder load. `sd_vae_approx.py` already uses `torch.load(..., weights_only=True)`.
- `modules/sd_hijack_optimizations.py` and `modules/sub_quadratic_attention.py`: reviewed SDP backend selection, Torch 2.14 `torch.nn.attention.sdpa_kernel` use, Doggettx/InvokeAI/sub-quadratic/sdp tensor reshaping, mask handling, chunk sizing, upcast/autocast, and AttnBlock paths.

Findings and fixes:
- Fixed stale checkpoint state-dict reuse after a checkpoint file changes at the same `CheckpointInfo` identity. `get_checkpoint_state_dict()` calculated a fresh shorthash but still keyed `checkpoints_loaded` by the `CheckpointInfo` object, so a replaced/modified checkpoint could return the old cached state dict. Cache keys now include absolute filename, `(mtime_ns, size)`, and current sha256; stale entries for the same path are dropped before lookup/store.
- Fixed stale VAE cache reuse after a VAE file changes at the same path. The in-memory VAE cache was keyed only by `vae_file`; it now keys by absolute filename plus `(mtime_ns, size)` and drops stale same-path entries before lookup/store.
- Fixed a Doggettx split-attention chunking edge case. When the memory estimator selected more `steps` than query tokens but not enough to trip the `steps > 64` OOM guard, `slice_size = q.shape[1] // steps` could become zero and crash `range(...)`. Slice size is now clamped to at least 1.

Deprecation/API evidence:
- Verified inside `local/gb10-a1111:latest`: Torch reports `2.14.0.dev20260709+cu132`; `torch.nn.attention.sdpa_kernel(backends, set_priority=False)` is available.
- Verified old `torch.backends.cuda.sdp_kernel(...)` emits a `FutureWarning` directing users to `torch.nn.attention.sdpa_kernel()`; code search found no in-scope uses of the deprecated CUDA context manager in `modules/`, `extensions/`, `test/`, or `tests/`.
- Added/ran an SDPA smoke test with `sdpa_backend_override="math"` and warnings captured; no `FutureWarning` was emitted by the current implementation.

Validation:
- `docker run --rm --entrypoint bash -v /home/kklouzal/stable-diffusion-webui:/work -w /opt/stable-diffusion-webui/repositories/stable-diffusion-stability-ai -e PYTHONPATH=/work:/opt/stable-diffusion-webui local/gb10-a1111:latest -lc "python -m py_compile /work/modules/sd_models.py /work/modules/sd_vae.py /work/modules/processing.py /work/modules/sd_models_config.py /work/modules/sd_vae_approx.py /work/modules/sd_hijack_optimizations.py /work/modules/sub_quadratic_attention.py /work/test/test_openclaw_cache_invalidation.py && python /tmp/run_slice8_tests.py"` -> passed selected cache/processing regression functions, including the new checkpoint and VAE cache invalidation tests plus img2img init-cache lock/key tests. The runtime image still lacks `pytest`, so this was a direct fixture shim rather than full pytest collection.
- `docker run --rm --entrypoint bash -v /home/kklouzal/stable-diffusion-webui:/work -w /opt/stable-diffusion-webui/repositories/stable-diffusion-stability-ai -e PYTHONPATH=.:/opt/stable-diffusion-webui/repositories/generative-models:/work:/opt/stable-diffusion-webui local/gb10-a1111:latest -lc "python -m py_compile /work/modules/sd_hijack_optimizations.py /work/test/test_openclaw_attention_slice_bounds.py && python /work/test/test_openclaw_attention_slice_bounds.py"` -> passed `PASS attention slice bounds and SDPA deprecation smoke`.
- `python -m pytest ...` was attempted in the runtime image and blocked by `/usr/local/bin/python: No module named pytest` (known image limitation from earlier slices).

Remaining lanes / next slice:
- This is not a full end-to-end completion. Remaining core/build/runtime lanes include full pytest-capable collection in a dependency-complete test image, real CUDA generation smoke for checkpoint/VAE reload transitions, live img2img/inpaint visual regression, broader `modules/processing.py` refiner/control-extension interactions, and adjacent sampler/upscaler/postprocessing extension audits not covered by this bounded pass.

## 2026-07-16 core slice 9 — dependency-complete tests + isolated CUDA runtime validation

Scope: continued audit on authoritative GB10 checkout `/home/kklouzal/stable-diffusion-webui` (`latest`, baseline HEAD `21eeaafc2b80f0153d3064eec3e21abc9e03c1fe`) using local image `local/gb10-a1111:latest` (`386b6875a7a9`). Live production container `gb10-a1111-latest` was inspected only; it was not restarted, reconfigured, or port-published against.

Disposable validation environment:
- Pytest containers were launched with `--rm --gpus all --entrypoint bash`, canonical source bind-mounted at `/opt/stable-diffusion-webui`, pytest/pytest-mock installed only into `/tmp/a1111-test-venv --system-site-packages`, and live model/config mounts reused read-only except source and test scratch.
- CUDA/API smoke container: `oc-a1111-slice9-runtime-20260716035427`, image digest `sha256:386b6875a7a9480108c5d779034c035d84d42ac717b961d1ea031ed367281cff`, no published ports, internal API on `127.0.0.1:17860`, source overlaid, image-baked `repositories/` copied to `/tmp/oc-slice9/repositories-image:ro`, ControlNet model directory mounted read-only at `extensions/sd-webui-controlnet/models`, outputs isolated under `/tmp/oc-slice9/outputs`. Container logs/state/stats were saved under `/tmp/oc-slice9/` and the disposable runtime container was removed after validation.
- GB10 GPU identity during validation: `torch 2.14.0.dev20260709+cu132`, CUDA available, device `NVIDIA GB10`. `nvidia-smi` on GB10 reports process memory but total/used query fields are partly `N/A`; saved raw output at `/tmp/oc-slice9/nvidia_smi_after.txt`.

Commands/gates:
- Environment probe: `docker run --rm --name oc-a1111-slice9-probe-* --gpus all --entrypoint bash -v /home/kklouzal/stable-diffusion-webui:/opt/stable-diffusion-webui -w /opt/stable-diffusion-webui local/gb10-a1111:latest -lc 'python ... import torch/pytest'` -> CUDA true, pytest absent from base image as expected.
- Focused dependency-complete pytest gate after fix:
  - `python -m pytest -q tests --ignore=tests/test_processing_auxiliary_infotext_alignment.py --ignore=tests/test_upscaler_tiling_contract.py --ignore=tests/test_ui_extensions_contract.py extensions/openclaw-clear-cond-cache/tests extensions/openclaw-denoise-ramp/tests extensions/openclaw-multi-sampler/tests extensions/sd-webui-incantations/tests extensions/sd-webui-teacache/tests extensions/sd-webui-controlnet/unit_tests/args_test.py`
  - Result: `159 passed, 2 warnings in 2.85s`.
- Recent-fix focused gate:
  - `python -m pytest -q tests/test_openclaw_generation_profile.py tests/test_teacache_extension.py extensions/sd-webui-teacache/tests/test_teacache_session.py extensions/openclaw-multi-sampler/tests/test_openclaw_multi_sampler.py extensions/openclaw-clear-cond-cache/tests/test_openclaw_clear_cond_cache.py`
  - Result: `33 passed, 1 warning in 2.44s`.
- API regression gate:
  - `python -m pytest -q tests/test_controlnet_legacy_api_fields.py tests/test_api_listing_contract.py tests/test_api_extension_item_contract.py`
  - Result: `11 passed, 1 warning in 0.12s`.
- Syntax gate: `python3 -m py_compile modules/api/api.py tests/test_controlnet_legacy_api_fields.py` -> `py_compile_ok`.
- Full same-process collection note:
  - `python -m pytest -q tests extensions/openclaw-clear-cond-cache/tests extensions/openclaw-denoise-ramp/tests extensions/openclaw-multi-sampler/tests extensions/sd-webui-incantations/tests extensions/sd-webui-teacache/tests extensions/sd-webui-controlnet/unit_tests/args_test.py`
  - Current result: `169 passed, 4 failed, 2 warnings in 2.61s`. Failures are test-harness/static-assertion issues, not runtime regressions exposed by the smoke pass: stale source-string expectation in `tests/test_processing_auxiliary_infotext_alignment.py::test_img2img_init_cache_helpers_share_payload_and_stats_boundaries`, and same-process `sys.modules` pollution/import-stub ordering affecting `tests/test_ui_extensions_contract.py` and `tests/test_upscaler_tiling_contract.py`. The isolated/relevant gates above pass.

Confirmed product defect found and remediated:
- Symptom: every `/sdapi/v1/txt2img` runtime smoke initially failed immediately with HTTP 500 `StableDiffusionProcessingTxt2Img.__init__() got an unexpected keyword argument 'control_net_enabled'`. This happened even for baseline txt2img because ControlNet top-level legacy API fields are part of the generated Pydantic API model with default `None`; they were left in `args` and passed into `StableDiffusionProcessing*` constructors.
- Fix: `modules/api/api.py` now normalizes ControlNet remote aliases, pops all ControlNet remote API fields out of constructor kwargs, and reattaches only non-`None` remote fields onto the processing object for the ControlNet extension to read. This preserves legacy top-level ControlNet API compatibility without poisoning baseline txt2img/img2img constructor kwargs.
- Regression coverage: `tests/test_controlnet_legacy_api_fields.py` now asserts `_pop_controlnet_remote_args(args)`, `_attach_controlnet_remote_args(p, controlnet_remote_args)`, and `value = args.pop(key)` are present in the API path, in addition to the prior alias/field contract checks.

CUDA/API smoke results (`/tmp/oc-slice9/artifacts/smoke_summary.json`, log `/tmp/oc-slice9/smoke_run.log`): all 14 cases succeeded. Settings common unless noted: prompt `slice9 audit smoke: small crystal fox, clean lines`, negative `low quality, blurry`, sampler `Euler`, scheduler `Karras`, 512x512, 4 steps, cfg 5.5, batch 1, deterministic seeds below.

| Case | Seed | Runtime | Output sha256 prefix(es) | Notes |
|---|---:|---:|---|---|
| `txt2img_baseline` | 911991 | 62.418s | `95329546941ed605` | First-load SDXL generation on `MM_R2.safetensors`; VAE `ftasticVAE_v10.safetensors`. |
| `txt2img_repeat_cache_reuse` | 911991 | 1.208s | `95329546941ed605` | Repeat generation produced identical hash, exercising warmed model/sigma/cache reuse. |
| `txt2img_checkpoint_switch` | 911992 | 33.895s | `a37ca0f751526911` | Switched to `MM_R2_FIX-S.safetensors`. |
| `txt2img_checkpoint_switch_back` | 911993 | 13.472s | `0308cadc9d677319` | Switched back to `MM_R2.safetensors`. |
| `txt2img_vae_a` | 911994 | 23.689s | `41ce7e9de0e7dfcc` | Override VAE `sdxl_vae.safetensors`. |
| `txt2img_vae_b` | 911995 | 18.255s | `7901e90983aa318c` | Override VAE `xlVAEC_e7.safetensors`. |
| `img2img_smoke` | 911996 | 1.267s | `edb2babfa04eb226` | Representative synthetic 512x512 init image, denoise 0.45. |
| `inpaint_smoke` | 911997 | 1.326s | `4df5f3907e8e6202` | Representative mask, denoise 0.55, non-full-res inpaint path. |
| `txt2img_highres_smoke` | 911998 | 2.038s | `be45eaea135169b1` | `enable_hr`, `hr_scale=1.25`, latent upscaler, 2 HR steps. |
| `txt2img_refiner_smoke` | 911999 | 14.303s | `a5133cba2e173436` | Refiner checkpoint `MM_R2_FIX-S`, switch at 0.5. |
| `controlnet_single` | 912000 | 22.833s | `ecbd454b84a4608e`, `1e9b6daacb16f6a1` | Single ControlNet unit with local `xinsir-controlnet-depth-sdxl-1.0 [e590f04c]`, module `none`, synthetic conditioning image. |
| `controlnet_dual` | 912001 | 2.515s | `bcbcc55b7f670a13`, `1e9b6daacb16f6a1` | Dual ControlNet units, same local model/image with different weights. |
| `teacache_enabled` | 912002 | 1.325s | `4beecca8cf9760a3` | TeaCache enabled via always-on args; infotext recorded threshold 0.25 and max consecutive 2. |
| `teacache_controlnet_fallback` | 912003 | 1.994s | `4b25827e0f1c9b91`, `1e9b6daacb16f6a1` | TeaCache + ControlNet succeeded; infotext recorded `TeaCache disabled reason: external UNet forward hook`, verifying correctness fallback. |

Generated artifacts are in `/tmp/oc-slice9/artifacts/` with full SHA-256s embedded in filenames and `smoke_summary.json`. ControlNet returned an additional detected-map/control artifact (`1e9b6daacb16f6a1...`, 2430 bytes) for the ControlNet cases.

Recent-fix verification coverage:
- Model-specific K-diffusion sigma/cache behavior: `tests/test_openclaw_generation_profile.py` passed in focused gate; repeat txt2img hash matched exactly after first-load warm path.
- Checkpoint and VAE stale identity: runtime checkpoint switch/switch-back and VAE override A/B smokes succeeded with distinct outputs and expected infotext model/VAE names.
- Img2img cache locking/invalidation: img2img and inpaint runtime smokes passed; focused collection excluding the stale source-string assertion passed. The remaining static assertion should be updated in a later harness cleanup to match the current restored-dict implementation.
- CUDA graph per-key serialization: existing `test/test_openclaw_cuda_graphs.py` had prior WIP coverage in this dirty tree; this slice did not enable runtime CUDA graphs (`OPENCLAW_CUDA_GRAPHS=0` inherited from image/live env), so no live graph replay claim is made here.
- TeaCache hook fallback: focused TeaCache tests passed; runtime TeaCache+ControlNet recorded `external UNet forward hook` fallback and still generated successfully.
- Multi-sampler/stage noise interval: `extensions/openclaw-multi-sampler/tests` passed in both 159-pass and 33-pass gates.

Skipped/narrowed coverage:
- No network model/preprocessor downloads were attempted. ControlNet used the one local model found (`xinsir-controlnet-depth-sdxl-1.0`) with `module: none`; external preprocessor downloads and annotator-weight paths remain unclaimed.
- CUDA graph runtime replay was not enabled in the disposable env, so the result verifies tests/contracts but not graph capture execution.
- Same-path VAE metadata invalidation was not forced by mutating shared model files; VAE identity was exercised through safe override switches only. Mock/file-level invalidation remains covered by existing unit tests/WIP, not by destructive asset mutation.

Next remaining audit lane: Docker/build/dependency freshness and a dedicated harness-cleanup pass for same-process pytest isolation/static assertions. Do not mark the overall gb10-a1111:latest audit complete yet; build/runtime-dependency lanes remain.

## Final comprehensive once-over closure — 2026-07-16 UTC

Baseline and preservation:
- Authoritative worktree: GB10 `/home/kklouzal/stable-diffusion-webui`, branch `latest`, baseline HEAD `21eeaafc2b80f0153d3064eec3e21abc9e03c1fe`.
- Existing dirty coordinated audit work, untracked tests, generated `gb10/validate-image-*` evidence, `mxfp8-diagnostics/last-result.json`, and this ledger were preserved. No reset/revert/commit/push/deploy/live-service restart was performed.
- Live service `gb10-a1111-latest` was not restarted or modified. Runtime/API checks used disposable containers without published ports and were removed afterward.

Final fixes applied in this closure:
- `modules/processing.py`: rewrote img2img init-cache restore to iterate `_IMG2IMG_INIT_CACHE_ATTRS` and clone each restored payload value at assignment time. This keeps the helper implementation aligned with the shared store/restore attribute contract and resolved the remaining static same-process harness assertion in `tests/test_processing_auxiliary_infotext_alignment.py`.
- `test/test_openclaw_attention_slice_bounds.py`: hardened the new attention/SDPA test so its lightweight `ldm`/`sgm`/module import stubs are only installed when real dependency modules are unavailable, and cleaned those stubs after importing `modules.sd_hijack_optimizations`. This prevents same-process collection from poisoning later real-module imports while preserving the standalone no-repository harness path.
- `tests/test_upscaler_tiling_contract.py`: hardened the tiling contract test so dependency stubs are only used when needed, paths point at the active checkout, and temporary `modules.*` stubs are cleaned after import. The tiled-upscale test now imports `modules.upscaler_utils` with local stubs only for that assertion, then cleans them, preventing same-process pollution of later tests.

Validation evidence from this closure:
- `git diff --check`: pass.
- Python compile gate with temp pycache: `PYTHONPYCACHEPREFIX=/tmp/gb10-a1111-pycache python3 -m compileall -q modules extensions/openclaw-clear-cond-cache extensions/openclaw-multi-sampler extensions/openclaw-denoise-ramp extensions/sd-webui-teacache docker/prepare-resolver-input.py webui.py tests test`: pass. Initial compile without temp pycache was blocked by pre-existing root-owned `test/__pycache__` files; rerun with temp pycache passed without writing repo pyc files.
- Shell gate: `shellcheck docker/entrypoint.sh gb10/build.sh docker/launch-a1111.sh` and `bash -n docker/entrypoint.sh gb10/build.sh docker/launch-a1111.sh`: pass.
- Focused known-failure/static harness gate on host: `python3 -m pytest -q tests/test_processing_auxiliary_infotext_alignment.py tests/test_ui_extensions_contract.py tests/test_upscaler_tiling_contract.py tests/test_controlnet_legacy_api_fields.py`: `16 passed, 1 skipped` after the processing restore fix.
- Focused Docker/CUDA pytest gate using disposable container and pytest installed into the disposable container only: `tests/test_processing_auxiliary_infotext_alignment.py tests/test_ui_extensions_contract.py tests/test_upscaler_tiling_contract.py tests/test_controlnet_legacy_api_fields.py test/test_openclaw_attention_slice_bounds.py`: `19 passed`.
- Changed OpenClaw extension/CUDA/sampler unit gate in disposable CUDA container: `tests/test_teacache_extension.py`, `extensions/sd-webui-teacache/tests/test_teacache_session.py`, `extensions/openclaw-multi-sampler/tests/test_openclaw_multi_sampler.py`, `extensions/openclaw-denoise-ramp/tests/test_openclaw_denoise_ramp.py`, `test/test_openclaw_cuda_graphs.py`, `test/test_openclaw_kdiffusion_sigmas_cache.py`, `test/test_openclaw_attention_slice_bounds.py`: `46 passed`.
- Cache invalidation/img2img cache unit gate in disposable CUDA container with image-copied repository deps and temp diskcache: `test/test_openclaw_cache_invalidation.py`: `15 passed`.
- Full same-process collection gate in disposable CUDA container with image-copied repository deps and temp diskcache: `236 tests collected in 19.01s`, no collection errors. Log: `/tmp/gb10-docker-pytest-collect-20260716.log`.
- Full all-test execution was intentionally not used as a readiness blocker: legacy API/server tests require an external `base_url` fixture/live API server, and several older tests mutate `sys.modules["modules"]` globally when all tests are executed in one process. The required full same-process collection is clean, and the focused source-backed unit/runtime gates above cover the changed surfaces.
- Docker/build gate: `DOCKER_BUILDKIT=1 BUILDKIT_PROGRESS=plain docker build --cache-from local/gb10-a1111:latest --target runtime -f Dockerfile -t local/gb10-a1111:onceover-20260716 .`: pass. The build preserved the protected CUDA/PyTorch/MSLK base stack (`torch 2.14.0.dev20260709+cu132`, `torchao 0.17.0`, `mslk 2026.7.11`) while resolving app dependencies; disposable image removed after validation. Log: `/tmp/gb10-docker-build-onceover-20260716.log`.
- Disposable runtime/API smoke without published ports using the newly built image and temp outputs/config: txt2img 512x512 4-step, repeat determinism, img2img, inpaint, and hires all succeeded. Evidence: `/tmp/gb10-runtime-smoke-20260716.log`; repeat deterministic hash matched.
- Disposable extension runtime/API smoke with checkout `extensions/` mounted read-only: `/sdapi/v1/scripts` listed `openclaw denoise ramp`, `openclaw multi-sampler`, `controlnet`, and `dynamic thresholding`; `/sdapi/v1/openclaw/clear-cond-cache` returned ok for `c`, `uc`, and `img2img_init`; `/controlnet/model_list` returned one model. Evidence: `/tmp/gb10-extension-runtime-smoke-20260716.log`.
- Disposable ControlNet runtime smoke: single and dual ControlNet txt2img requests against `xinsir-controlnet-depth-sdxl-1.0 [e590f04c]` succeeded. Evidence: `/tmp/gb10-controlnet-runtime-smoke-20260716.log`.
- Disposable TeaCache runtime smoke: txt2img with always-on `TeaCache` args succeeded and infotext contained `TeaCache`. Evidence: `/tmp/gb10-teacache-runtime-smoke-20260716.log`.
- Disposable checkpoint/VAE lifecycle smoke: API checkpoint switch to `MM_R2_FIX-S.safetensors`, switch back to `MM_R2.safetensors`, and VAE switch to `softFastVAESDXLPONY_v10.safetensors` all succeeded. Evidence: `/tmp/gb10-switch-runtime-smoke-20260716.log`.

Reviewed/finalized surfaces:
- Core processing/sampling/inpaint/img2img/hires/cache: direct source review plus focused tests for infotext alignment, img2img init-cache keying/restoration, cache invalidation, k-diffusion sigma cache invalidation, sampler/multi-sampler boundaries, CUDA graph helpers, and attention slice bounds.
- Owned OpenClaw extensions: clear-cond-cache, multi-sampler, denoise-ramp, TeaCache, dynamic thresholding/incantations covered by source review, unit tests, and disposable runtime API smokes where endpoints/scripts are exposed.
- API/ControlNet: reviewed `modules/api/api.py`, `modules/api/models.py`, and `tests/test_controlnet_legacy_api_fields.py`; collection/tests pass; disposable ControlNet single/dual API generation passes.
- Docker/build/dependency resolver/runtime launch: Dockerfile/build/entrypoint/resolver surface reviewed and runtime target rebuilt successfully with protected CUDA/PyTorch boundary intact.

Remaining explicit limitations:
- CUDA graph replay was covered by `test/test_openclaw_cuda_graphs.py` and build/runtime import paths, but a separate API generation run with `OPENCLAW_CUDA_GRAPHS=1` was not promoted to a final readiness requirement because the live-equivalent configuration has `OPENCLAW_CUDA_GRAPHS=0` and CUDA graph enablement is a runtime opt-in path.
- Full legacy API tests under `test/` still need their external `base_url` server fixture to execute end-to-end; this is harness/environmental and not a regression in the changed generation paths.
- Runtime smokes used short low-step generations for correctness/residency/API coverage rather than visual quality scoring. They prove successful SDXL generation paths and deterministic repeat for the selected prompt/seed, not aesthetic benchmark quality.

Final readiness assessment:
- The integrated dirty worktree is coherent and validated across static, unit, Docker build, and disposable CUDA runtime/API gates. No supported correctness, cache invalidation, dtype/device, API compatibility, or SDXL generation-quality defect found during this closure remains unaddressed.
- Recommended gate before commit/deploy: review the dirty diff as one coordinated audit stack, then commit; rebuild/tag the production image from this worktree; run the same disposable API smoke against that exact production tag; only then schedule a live `gb10-a1111-latest` replacement/restart.

## 2026-07-16 pass - API-only pre-first-step delay regression

- Symptom: live gb10-a1111-latest accepted API/UI-equivalent generation requests before the startup SDXL model load had completed, so the first real generation after deployment paid the cold checkpoint/VAE/MXFP8 residency cost between Generate and sampling step 1.
- Preserved rollback before deploy: container 42596bf897e593f56898241490b4c6aa99771db226b4c2b962d01a4ba4eb7495, image sha256:e4c02900e17badf72c90c91e97c12073692d804320c886fab91e8c899a3499c2 (local/gb10-a1111:latest from deployed commit 7eed414f85657345e6fa5945f7c32952c2c1593d). Investigation artifacts: /tmp/gb10-a1111-prestep-delay-20260716T231221Z on GB10.
- Measurements before the fix:
  - TXT2IMG SDXL 1024x1024 Euler 8-step: first cold-ish probe step 1 at 3.968s, identical repeat step 1 at 1.889s, total 12.248s -> 8.530s, identical PNG bytes for the same deployed code/settings.
  - IMG2IMG SDXL 1280x1280 Euler 8-step after the cold API-ready race: step 1 at 15.042s, identical repeat 14.011s, LoRA probe 13.576s, total about 25-26s; this matched the user-visible ~10s+ pre-step stall class.
  - After explicit cache clear but without a fresh deployment, img2img init cache behaved correctly (misses=1, then hits=1) and warm first-step dropped to 4.592s then 3.481s, isolating the worst regression to cold startup/model residency rather than repeated conditioning, LoRA, ControlNet, TeaCache, or image-init cache corruption.
- Root cause: API-only startup used initialize.initialize(), which starts initial model loading in the background, and then exposed FastAPI readiness before sd_models.model_data.get_sd_model() had completed. The heavy cold phase was model residency/MXFP8 cache load, not sampling quality math: postdeploy startup log reports Model loaded in 20.0s with load mxfp8 cache: 14.6s, before Uvicorn was allowed to accept requests after the fix.
- Fix: webui.api_only() now waits for sd_models.model_data.get_sd_model() before constructing/launching the API app unless --skip-load-model-at-start is explicitly set, and records load startup SD model in startup timing. This moves the unavoidable cold model residency cost into honest service startup readiness instead of the first generation hot path; it does not change samplers, conditioning, LoRA, ControlNet, VAE selection, TeaCache, seeds, or image math.
- Regression guard: tests/test_webui_api_only_startup_model_contract.py asserts API-only startup waits for the startup model before FastAPI/app launch and preserves the explicit skip-load opt-out.
- Validation:
  - PYTHONDONTWRITEBYTECODE=1 pytest -q tests/test_webui_api_only_startup_model_contract.py tests/test_api_server_control_contract.py -> 8 passed.
  - PYTHONPYCACHEPREFIX=/tmp/gb10-pyc python3 -m py_compile webui.py passed.
  - IMAGE_TAG=local/gb10-a1111:prestep-gate gb10/build.sh succeeded, producing image sha256:7c2cbb3e4ceaab229d8936a4f8709f5a5eea6d9192d3b975cc00d1625ff293b0.
  - Final IMAGE_TAG=local/gb10-a1111:latest gb10/build.sh succeeded with the same image id.
  - gb10/smoke-test.sh after deployment passed /sdapi/v1/progress, /sdapi/v1/sd-models, torch CUDA, MXFP8 TorchAO/MSLK, NVFP4 TorchAO/MSLK, and container imports.
- Deployment proof:
  - Code commit pushed to origin/latest: 32163b9ba7322e3971a4857e08503f0744bad264 (Warm API startup model before serving).
  - Live container after deploy: 96745abd6f766730e2e592948d87343e2b032887128447daef689e5fe1fd8201, image sha256:7c2cbb3e4ceaab229d8936a4f8709f5a5eea6d9192d3b975cc00d1625ff293b0.
  - API readiness after container start was 28s; startup log shows load startup SD model: 20.1s and Uvicorn only after model load.
  - Postdeploy TXT2IMG 1024x1024 Euler 8-step: step 1 at 1.571s, identical repeat 1.480s, total 7.069s then 6.678s, identical PNG bytes across repeats (b93d012ddf87c3aabcc3040ebe1ba68c406db2533bd7a98e977f19925d3b56b3).
  - Postdeploy IMG2IMG 1280x1280 Euler 8-step: step 1 at 3.304s, total 11.588s, cond/img2img cache status healthy (misses=1, cached=true, no bypass).
- Remaining limitation: a deliberate checkpoint/VAE switch or an explicit --skip-load-model-at-start launch can still incur legitimate model-load latency before sampling; this change prevents the live API from advertising ready while the normal startup model is still cold.

## 2026-07-17 pass - warmed repeated img2img pre-first-UNet ControlNet fix

- Exact workload: controller-derived `gb10/gb10_direct_payload-20260717T005806Z.json`, fixed seed `1174203783`, 1280x1280 img2img, 15 steps, `depth_zoe` + `xinsir-controlnet-depth-sdxl-1.0`, LoRAs, SEG/dynamic-thresholding, and TeaCache. Generic `/progress` activation was explicitly excluded as a sampling-start signal; timings use monotonic `process_images_inner`/sampler/denoiser/UNet markers.
- Timeline disambiguation: the older ~10.52s generation-diagnostics interval was the sampler launch-to-return duration; first denoiser/UNet followed sampler launch by about 1-3ms. Pipeline-level instrumentation found the real warmed pre-first-UNet interval at 8.56-9.32s, with 8.50-9.23s inside `Script:extensions/sd-webui-controlnet/scripts/controlnet.py`; warmed img2img init-cache hits were only 35-56ms and VAE encode/snapshot work was not on the repeated critical path.
- Root cause: `depth_zoe` attempted to load `ZoeD_M12_N.pt` on every request, but current timm rejected checkpoint-only derived `*.attn.relative_position_index` buffers as unexpected state keys. The ControlNet script runner reported/swallowed the exception, so generation continued without ControlNet and the preprocessor/model cache could never become warm. Once that compatibility failure was removed, active ControlNet exposed a second backend-boundary issue: torchao MX linear used `view()` on a logically valid non-contiguous attention context.
- Narrow fix: `gb10/patch-controlnet-zoedepth.py` changes the deployed ignored/vendor ControlNet loader to `strict=False` but explicitly rejects every missing key and every unexpected key not ending in `.attn.relative_position_index`; `gb10/run.sh` reapplies the guarded patch after extension sync. `modules/hypernetworks/hypernetwork.py` normalizes K/V context to contiguous storage once at the shared linear-projection boundary without changing values.
- Fixed warmed repeats: first UNet 0.674/0.664/0.666s; ControlNet `process` 0.025-0.026s; img2img init 0.036-0.037s; all three output PNG SHA-256 values were identical (`26d0e89e52f5581f9f3777a64523284fcfa7ccb7142af74e6f69558d206de8d2`). The first post-restart ControlNet load was intentionally cold (first UNet 26.42s), then subsequent preprocessor cache hits removed the repeated delay.
- One-feature matrix after warmup: ControlNet off 0.643s first UNet; LoRA stripped 1.323s; TeaCache off 3.944s; unchanged baseline 0.664-0.674s. ControlNet-off output SHA differed (`74d1caf7...`), proving the fixed baseline now actually applied ControlNet. GPU capture covered each interval; the fixed cold load averaged 17.4% GPU with 89% peak, while warmed intervals were only seven 100ms samples and no longer contained a sustained multi-second ControlNet load.
- Tests/gates: `python3 -m pytest -q tests/test_controlnet_zoedepth_patch.py tests/test_controlnet_legacy_api_fields.py` => 6 passed; Python source compile for the patcher/hypernetwork module, `bash -n gb10/run.sh`, and `git diff --check` passed. Runtime artifacts: `gb10/warm-delay-actual-summary-20260717T021056Z.json` (pre-fix feature matrix), `gb10/warm-delay-actual-summary-20260717T024212Z.json` (cold+warm fixed proof), `gb10/warm-delay-actual-summary-20260717T024414Z.json` (fixed feature matrix), and `gb10/warm-delay-zoe-rollback-20260717T0235Z.txt`.
- Rebuild/deploy proof: code commit `cea4f90ad3d30a3e193c240b7fec5e3fc84808a8` was pushed to `origin/latest`; clean worktree build `20260717T025139Z` produced image `sha256:5feba6b1be57021ec231fec51b3cba8ba0318e4a4456ae47247232c2c8f9529c`. Final clean container `a94622c078ea40dab2030623d0d795e98fac953ec9530bb92cda77af6231cc9f` is running with restart count 0. Rollback image retained as `local/gb10-a1111:rollback-20260717T0235Z` (`sha256:7c2cbb3e4ceaab229d8936a4f8709f5a5eea6d9192d3b975cc00d1625ff293b0`).
- Instrumented postdeploy warmed repeats on the rebuilt image measured first UNet at 0.666/0.672/0.669s, ControlNet process at 24.7/24.3/24.1ms, and identical PNG SHA-256 `b5e778a5523cc4cd2482283fc48d0c76f3f89d1a90d4a9dea481621d522a20eb`; see `gb10/warm-delay-actual-summary-20260717T025655Z.json`. Temporary diagnostics were removed by a final clean redeploy. Clean-runtime cold/warm smoke completed in 67.679s/25.874s with identical PNG bytes, no ControlNet/Zoe/MXFP8 traceback, healthy A1111 API, and responsive controller on port 8871.
