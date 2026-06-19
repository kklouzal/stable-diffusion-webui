# GB10 A1111 latest core generation/math audit ledger

Target: `/home/kklouzal/stable-diffusion-webui` on GB10, branch `latest`.

Baseline:
- 2026-06-19: `git status --short --branch` reported `## latest...origin/latest [ahead 22]` with no dirty or untracked entries before this ledger was created.
- Baseline HEAD: `41ef77d8b165b2bcac32fea7777aed77777e447b` (`Add save serialization contract tests`).

Audit scope:
- Core Python generation/math/logical consistency paths affecting final image quality.
- Priority checks: sampling math, latent encode/decode ranges, VAE/model precision, RNG/seed determinism, cache invalidation, image grid/resize math, script hooks that can alter denoising dataflow.

Reviewed files/functions:
- `modules/rng.py`: `randn`, `randn_local`, `randn_like`, `randn_without_seed`, `manual_seed`, `create_generator`, `slerp`, `ImageRNG.__init__`, `ImageRNG.first`, `ImageRNG.next`.
  - Findings: no confirmed remediation yet. Seed resize center-crop/paste arithmetic and CPU/NV generator routing are internally consistent with existing A1111 behavior. `slerp` still has the upstream-style zero-norm sensitivity, but generated normal noise makes this non-actionable without a reproducible quality failure.
- `modules/sd_samplers_common.py`: `setup_img2img_steps`, `samples_to_images_tensor`, `single_sample_to_image`, `decode_first_stage`, `sample_to_image`, `samples_to_image_grid`, `images_tensor_to_samples`, `store_latent`, `is_sampler_using_eta_noise_seed_delta`, `replace_torchsde_browinan`, `apply_refiner` first half.
  - Findings: no confirmed remediation yet. Latent/image range conversions follow the expected `[-1,1]` decode and `[0,1]` encode paths; OOM fallback preserves per-sample semantics.
- `modules/sd_samplers_kdiffusion.py`: `KDiffusionSampler.sample`, `KDiffusionSampler.sample_img2img`.
  - Findings: no confirmed remediation yet. Sigma slicing, img2img noise injection, SGM multiplier, Brownian noise-sampler hookup, and sampler argument construction are coherent in the reviewed chunk.
- `modules/sd_samplers_timesteps.py`: `CompVisSampler.get_timesteps`, `CompVisSampler.sample`, `CompVisSampler.sample_img2img`.
  - Findings: no confirmed remediation yet. Timestep schedule and DDIM-style latent noising match expected alpha-cumprod formula in reviewed chunk.
- `modules/sd_samplers_lcm.py`: `LCMCompVisDenoiser.__init__`, `get_sigmas`, `sigma_to_t`, `t_to_sigma`, `get_eps`, `get_scaled_out`, `forward`, `sample_lcm`, `CFGDenoiserLCM.inner_model`, `LCMSampler.__init__`.
  - Findings: no confirmed remediation yet. LCM timestep mapping and scaled-output formula remain a high-value area for comparison against upstream Diffusers LCM before final closure.
- `modules/sd_samplers_cfg_denoiser.py`: `catenate_conds`, `subscript_cond`, `pad_cond`, `CFGDenoiser.__init__`, `combine_denoised`, `combine_denoised_for_edit_model`, `get_pred_x0`, `update_inner_model`, `run_inner_model`, `pad_cond_uncond`, `pad_cond_uncond_v0`, `forward`.
  - Findings: no confirmed remediation yet. AND composition indexing, skip-uncond handling, mask blending before/after denoising, and live-preview latent selection were reviewed for shape/index consistency.

Validation run so far:
- `git status --short --branch`
- Function inventory with `rg -n "^(def|class) " ...`
- Targeted source reads with `sed -n` over reviewed functions.

Commits made during this audit:
- None yet.

Remaining areas:
- Continue `modules/processing.py` function-by-function beyond random tensor creation and txt2img/img2img sample paths.
- Complete `modules/sd_models.py` precision/schedule/model reload paths, especially alpha schedule override and quantized reload handling.
- Complete `modules/sd_vae.py`, `modules/sd_vae_approx.py`, `modules/sd_vae_taesd.py`.
- Complete `modules/images.py` resize/grid/save math paths.
- Complete `modules/cache.py`, `modules/patches.py`, `modules/sd_hijack*.py`, `modules/scripts*.py`, `modules/shared*.py` generation-relevant hooks.
- Run focused unit/import tests after any remediation and at checkpoint.


Checkpoint update - cache remediation:
- `modules/cache.py`: `cached_data_for_file` now treats a missing cached `size` field as stale by comparing `cached_size != ondisk_size` rather than only checking mismatches when `size` is present.
  - Finding: legacy/migrated cache entries without `size` could reuse stale file-derived metadata when mtime matched, even if the file contents/size changed. This can affect checkpoint/VAE/safetensors metadata freshness and downstream model-selection diagnostics.
- `test/test_openclaw_cache_invalidation.py`: added `test_cached_data_for_file_invalidates_legacy_entry_without_size`.

Validation for cache remediation:
- `pytest -q test/test_openclaw_cache_invalidation.py` could not collect under GB10 system Python because runtime test dependencies are absent (`numpy` first; narrow imports also lacked `diskcache`, `tqdm`, and `torch`).
- `python3 -m py_compile modules/cache.py test/test_openclaw_cache_invalidation.py` passed.
- Dependency-stubbed direct check of `cached_data_for_file` legacy missing-size invalidation passed: `legacy cache size-missing invalidation passed`.

Commits made during this audit:
- `183d1ff4 Fix legacy file cache invalidation`: tightened file-derived metadata cache invalidation and added a legacy missing-size regression test.


Checkpoint update - additional reviewed paths:
- `modules/processing.py`: `StableDiffusionProcessing.__post_init__`, image conditioning helpers, prompt/cond cache setup, `process_images`, `process_images_inner`, `old_hires_fix_first_pass_dimensions`, `StableDiffusionProcessingTxt2Img.__post_init__`, `calculate_target_resolution`, `init`, `sample`, `sample_hr_pass`, HR condition setup, `StableDiffusionProcessingImg2Img.__post_init__`, mask blur accessors, img2img init-cache helpers, `StableDiffusionProcessingImg2Img.init`, `sample`, `get_token_merging_ratio`.
  - Findings: no additional confirmed remediation. Rechecked the post-decode color-correction branch after terminal output suggested a duplicate application; working tree contains only one `apply_color_correction` call in that branch.
- `modules/sd_models.py`: `rescale_zero_terminal_snr_abar`, `apply_alpha_schedule_override`, `SdModelData`, empty conditioning, TorchAO quantization detection/device/reload helpers, `load_model`, `reuse_model_from_already_loaded`, `reload_model_weights`, `unload_model_weights`, `apply_token_merging`.
  - Findings: no additional confirmed remediation. Noted duplicate `class SdModelData` declaration in source; the first empty version is immediately shadowed by the complete class and does not affect runtime behavior.
- `modules/sd_vae.py`: VAE identity/hash helpers, base VAE store/restore, list/resolve/load/reload paths.
  - Findings: no additional confirmed remediation. VAE cache/reload path preserves base VAE restoration semantics and dtype move after load.
- `modules/images.py`: `image_grid`, `split_grid`, `combine_grid`, `resize_image`.
  - Findings: no additional confirmed remediation in reviewed normal generation paths.
- `modules/sd_hijack.py`: optimizer selection/reset, weighted loss/forward helpers, partial `StableDiffusionModelHijack`, circular conv option, MPS buffer registration.
  - Findings: no remediation. `hijack_ddpm_edit` in `modules/sd_hijack_unet.py` registers `encode_first_stage` twice; reviewed as redundant same-dtype wrapping rather than a numerical output change.
- `modules/sd_hijack_optimizations.py`: split attention variants, einsum slicing, SDPA backend selection and attention forward.
  - Findings: no confirmed remediation.
- `modules/sd_hijack_unet.py`: `TorchHijackForUnet`, UNet apply-model dtype cast, timestep embedding, spatial transformer forward, GELU hijack, first-stage dtype hooks.
  - Findings: no confirmed remediation.
- `modules/scripts.py`: script callback registration/order/timing and generation-affecting hooks through `postprocess_batch` start.
  - Findings: no confirmed remediation in reviewed hook dispatch logic.
- `modules/shared_state.py`, `modules/shared.py`: generation state reset/progress preview and shared latent upscaler/device globals.
  - Findings: no confirmed remediation.
- `modules/sd_models_xl.py`, `modules/sd_models_config.py`: SDXL conditioning/apply-model extensions and checkpoint config inference.
  - Findings: no confirmed remediation.

Checkpoint validation after commit:
- `git status --short --branch` after commit reported branch `latest...origin/latest [ahead 25]` with no dirty entries before continuing reads.
- Later `git status --short` was clean after the false-positive color-correction candidate was rechecked.

Continuation instructions:
- Next slice should start at `modules/sd_hijack_clip.py`, `modules/sd_hijack_open_clip.py`, `modules/sd_hijack_clip_old.py`, `modules/sd_hijack_checkpoint.py`, and finish the remainder of `modules/scripts.py` after `postprocess_batch`.
- Then run focused review of extension scripts that alter sampler, refiner, or conditioning behavior under `modules/processing_scripts/` and relevant built-in extensions.
- Re-run validation in the project runtime environment if available; GB10 system Python lacks the dependencies needed for full pytest collection.

Checkpoint update - CLIP/script/extension audit continuation:
- Pre-edit coordination: `git status --short --branch` reported `## latest...origin/latest [ahead 26]` with a clean working tree at the start of this continuation; recent HEAD was `96c3fbd2 Update core math audit ledger`.
- `modules/sd_hijack_clip.py`: `clip_text_transformer_module`, `clip_text_embeddings`, `PromptChunk`, `TextConditionalModel.empty_chunk`, `get_target_prompt_token_count`, `tokenize_line`, `process_texts`, `forward`, `process_tokens`, `FrozenCLIPEmbedderWithCustomWordsBase.forward`, `FrozenCLIPEmbedderWithCustomWords.__init__`, `tokenize`, `encode_with_transformers`, `encode_embedding_init_text`, `FrozenCLIPEmbedderForSDXLWithCustomWords.encode_with_transformers`.
  - Findings: no confirmed remediation. Prompt chunk padding, textual inversion fix offsets, SD2 pad replacement, emphasis multiplier application, pooled SDXL return handling, and CLIP skip layer selection were internally consistent in the reviewed paths.
- `modules/sd_hijack_open_clip.py`: `FrozenOpenCLIPEmbedderWithCustomWords.__init__`, `tokenize`, `encode_with_transformers`, `encode_embedding_init_text`, `FrozenOpenCLIPEmbedder2WithCustomWords.__init__`, `tokenize`, `encode_with_transformers`, `encode_embedding_init_text`.
  - Findings: no confirmed generation-path remediation. OpenCLIP tokenizer IDs, pad semantics, transformer delegation, and pooled return propagation were coherent for conditioning.
- `modules/sd_hijack_clip_old.py`: `process_text_old`, `forward_old`.
  - Findings: no confirmed remediation. Legacy emphasis truncation, multipliers, and hijack fixes match the old 77-token behavior.
- `modules/sd_hijack_checkpoint.py`: `BasicTransformerBlock_forward`, `AttentionBlock_forward`, `ResBlock_forward`, `add`, `remove`.
  - Findings: no confirmed remediation. Checkpoint wrapper patch/restore state is simple and shape-preserving.
- `modules/scripts.py`: completed remaining `ScriptRunner` hooks after `postprocess_batch`: `postprocess_batch_list`, `post_sample`, `on_mask_blend`, `postprocess_image`, `postprocess_maskoverlay`, `postprocess_image_after_composite`, component callbacks, `reload_sources`, `before_hr`, `setup_scrips`, `set_named_arg`, `reload_script_body_only`.
  - Findings: no confirmed remediation. Hook ordering, arg slicing via `openclaw_script_args_to_overrides`, and timing bookkeeping preserve script dataflow.
- `modules/processing_scripts/seed.py`: `ScriptSeed.ui`, `setup`, `connect_reuse_seed`.
  - Findings: no confirmed remediation. Seed/subseed/resize assignment and reuse parsing are consistent with processing RNG fields.
- `modules/processing_scripts/sampler.py`: `ScriptSampler.ui`, `setup`.
  - Findings: no confirmed remediation. Steps, sampler, and scheduler setup are direct assignments to processing state.
- `modules/processing_scripts/refiner.py`: `ScriptRefiner.ui`, `setup`.
  - Findings: no confirmed remediation. Refiner checkpoint/switch gating matches `apply_refiner` expectations.
- `modules/processing_scripts/comments.py`: `strip_comments`, `ScriptStripComments.process`, `before_token_counter`.
  - Findings: no confirmed remediation. Prompt/comment stripping is applied to generation prompts and token counter consistently when enabled.
- `extensions-builtin/Lora/network.py`: `NetworkOnDisk`, `Network`, `NetworkModule.multiplier`, `calc_scale`, `apply_weight_decompose`, `finalize_updown`, `forward`.
  - Findings: no confirmed remediation. Scale/multiplier and DoRA decomposition paths were reviewed for dtype/device and output-shape consistency.
- `extensions-builtin/Lora/networks.py`: name conversion/mapping, `load_network`, `load_networks`, backup/restore, eager LoRA application, functional fallback, MXFP8/NVFP4 active-config preparation and merged LoRA helpers, patched module forwards/load-state hooks, infotext paste handling, available-network refresh.
  - Findings: no confirmed remediation. Existing file-signature invalidation for loaded LoRAs and quantized active config signatures include the current effective LoRA source and multiplier state.
- `extensions-builtin/Lora/network_lora.py`, `network_hada.py`, `network_lokr.py`, `network_glora.py`, `network_ia3.py`, `network_full.py`, `network_norm.py`, `network_oft.py`, `lora_patches.py`, `scripts/lora_script.py`, `extra_networks_lora.py`.
  - Findings: no confirmed remediation. Reviewed supported LoRA/LyCORIS/OFT module reconstruction, q/k/v split handling, extension activation/deactivation, and patch installation.
- `extensions-builtin/hypertile/hypertile.py`, `extensions-builtin/hypertile/scripts/hypertile_script.py`: divisor/tile candidate helpers, attention wrapper, model hook configuration, UI/XYZ integration.
  - Findings: no confirmed remediation. Tile divisor selection, rearrange inverses, seeded randomization, and first/second-pass configuration were coherent.
- `extensions-builtin/soft-inpainting/scripts/soft_inpainting.py`: inpainting detection, `latent_blend`, mask transforms, adaptive mask generation, histogram filters, UI/process hooks, `on_mask_blend`, `post_sample`, `postprocess_maskoverlay`.
  - Findings: no confirmed remediation. Latent blend dtype promotion/restoration, mask broadcasting, final-blend bypass, overlay rebuild, and per-image mask replacement were reviewed.
- `extensions-builtin/LDSR/scripts/ldsr_model.py`, `sd_hijack_autoencoder.py`, `sd_hijack_ddpm_v1.py`; `extensions-builtin/SwinIR/scripts/swinir_model.py`; `extensions-builtin/ScuNET/scripts/scunet_model.py`; selected postprocessing-for-training image math scripts.
  - Finding/remediation: `extensions-builtin/ScuNET/scripts/scunet_model.py` ignored the selected HTTP model URL in `UpscalerScuNET.load_model` and always passed `self.model_url` to `load_file_from_url`. Selecting the PSNR URL could therefore download/use the GAN model, altering final restoration quality. Fixed the URL branch to pass the selected `path` and derive the cache filename from that URL.

Validation for this continuation:
- `python3 -m py_compile extensions-builtin/ScuNET/scripts/scunet_model.py` passed.
- Static selected-URL check passed: the ScuNET HTTP branch now contains `load_file_from_url(path, model_dir=self.model_download_path, ...)` and no longer contains `load_file_from_url(self.model_url, ...)`.

Commits made during this continuation:
- `83b170bc Fix ScuNET URL model selection`: fixed ScuNET selected-URL handling for built-in restoration model selection.

Working tree note:
- After the ScuNET validation, unrelated concurrent changes appeared in `modules/ui_extra_networks.py` and `tests/test_extra_networks_path_contract.py`. They were not staged or modified by this continuation.

Remaining areas:
- Continue with any not-yet-reviewed generation-quality extension scripts outside this slice if desired, especially broader `scripts/` selectable scripts and deeper LDSR model internals if the audit scope extends to full postprocessing/upscaler architecture internals.
- Full pytest remains blocked under GB10 system Python unless the project/runtime test dependencies are available.
