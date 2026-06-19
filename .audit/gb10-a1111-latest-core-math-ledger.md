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
- Pending: cache remediation not yet committed at this ledger update.
