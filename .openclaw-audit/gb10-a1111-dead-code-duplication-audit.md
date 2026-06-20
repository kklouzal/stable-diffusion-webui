# GB10 A1111 latest dead-code and duplication audit

Started: 2026-06-20T05:35:43Z
Host/repo/branch: GB10 /home/kklouzal/stable-diffusion-webui branch latest
Starting HEAD: 26122d22350999c0edbd1f15821fe38c9a603367
Scope: exhaustive function-by-function source audit for dead code and code duplication, remediating findings immediately when safe.

## Audit policy
- Work in bounded slices; update this ledger each slice.
- Prefer real, behavior-preserving removals/deduplications over cosmetic churn.
- Commit each remediation or ledger checkpoint before continuing.
- Validate touched code with py_compile/node/shell/static checks/focused tests as feasible.
- Keep generated/vendor/third-party-heavy surfaces conservative; record if intentionally excluded from remediation.

## Current status
- Initial source inventory stored by first parent inspection in /tmp/gb10-a1111-source-inventory.txt on GB10.
- Next unchecked scope: core Python entrypoints and module graph triage.

## Checked scope

## Findings and fixes

## Validation log

## Remaining scope
- All source trees not yet explicitly recorded below.


## Pass 1 - core Python entrypoints/module graph (2026-06-20)

### Checked scope
- `launch.py`: `main`; compatibility exports for `args`, `python`, `git`, `index_url`, `dir_repos`, `commit_hash`, `git_tag`, `run`, `is_installed`, `repo_dir`, `run_pip`, `check_run_python`, `git_clone`, `git_pull_recursive`, `list_extensions`, `run_extension_installer`, `prepare_environment`, `configure_for_tests`, `start`.
- `webui.py`: `create_api`, `api_only`, `webui`; import-time startup sequence through `initialize_util`/`initialize`.
- `modules/launch_utils.py`: `check_python_version`, `_webui_source_version_env`, `_script_path_is_git_repo`, `commit_hash`, `git_tag`, `run`, `is_installed`, `repo_dir`, `run_pip`, `check_run_python`, `git_fix_workspace`, `run_git`, `git_clone`, `git_pull_recursive`, `version_check`, `run_extension_installer`, `list_extensions`, `run_extensions_installers`, `requirements_met`, `prepare_environment`, nested `ensure_build_dependencies`, `configure_for_tests`, `start`, `dump_sysinfo`.
- `modules/cmd_args.py`: module-level parser and all launch/runtime argument declarations in this file.
- `modules/paths.py`: `mute_sdxl_imports`, nested `Dummy`, path discovery/import side effects, `paths` mapping.
- `modules/paths_internal.py`: module-level path constants, `normalized_filepath`, pre-parser for `--data-dir`/`--models-dir`.
- `modules/shared.py`: module-level shared state/options placeholders and compatibility re-exports from `util`, `shared_items`, and `shared_ui_themes`.
- `modules/shared_state.py`: `State` and methods `__init__`, `need_restart` getter/setter, `server_command` getter/setter, `wait_for_server_command`, `request_restart`, `skip`, `interrupt`, `stop_generating`, `nextjob`, `dict`, `begin`, `end`, `set_current_image`, `do_set_current_image`, `assign_current_image`.
- `modules/timer.py`: `TimerSubcategory`, `Timer`, all timer methods, module-level parser, `startup_timer`, `startup_record`.
- `modules/errors.py`: `format_traceback`, `format_exception`, `get_exceptions`, `record_exception`, `report`, `print_error_explanation`, `display`, `display_once`, `run`, `check_versions`.
- `modules/devices.py`: `has_xpu`, `has_mps`, `cuda_no_autocast`, `get_cuda_device_id`, `get_cuda_device_string`, `get_optimal_device_name`, `get_optimal_device`, `get_device_for`, `torch_gc`, `torch_npu_set_device`, `enable_tf32`, `cond_cast_unet`, `cond_cast_float`, `manual_cast_forward`, `manual_cast`, `autocast`, `without_autocast`, `NansException`, `test_for_nans`, `first_time_calculation`, `force_model_fp16`, module-level dtype/device flags.

### Findings and fix decision
- No safe source deletion or deduplication was found in this slice.
- `launch.py` intentionally re-exports `modules.launch_utils` helpers. Several exports are referenced by first-party modules (`launch.commit_hash`, `launch.git_tag`, `launch.list_extensions`, `launch.run_extension_installer`) and the rest are part of the legacy launch/import compatibility surface, so they were preserved.
- `webui.webui()` is not dead despite the headless fork: it is the explicit compatibility failure path when the browser UI is requested, and `launch_utils.start()` still dispatches to it when `--nowebui` is absent.
- `modules.launch_utils.list_extensions()` and `modules.extensions.list_extensions()` share a name but are not duplicates: the launch helper reads settings before full app initialization to decide extension installer/preload scope; the extensions helper builds runtime extension metadata.
- `modules.errors.check_versions()` and `modules.initialize.check_versions()` are not duplicate implementations: `initialize.check_versions()` gates the runtime option and delegates to `errors.check_versions()` for the warning body.
- `modules.errors.display_once()` was the only in-scope function with no first-party call found by `git grep`; it is kept as an uncertain dynamic/public `modules.errors` helper that extensions may import.
- `modules/paths.py`, `modules/paths_internal.py`, `modules/cmd_args.py`, and `modules/shared.py` are side-effect/import-surface modules. No module-level constants or re-exports were removed in this pass.

### Static/dynamic audit map notes
- Entrypoint chain: `launch.py` -> `modules.launch_utils.prepare_environment()` -> `modules.launch_utils.start()` -> `webui.api_only()` for `--nowebui`, otherwise intentional `webui.webui()` browser-UI removal error.
- Runtime initialization chain: `webui.py` import-time setup -> `initialize.imports()` -> `shared_init.initialize()` and path/import side effects; `webui.api_only()` then calls `initialize.initialize()` and API callbacks.
- Public/dynamic surfaces to treat conservatively in later slices: `launch` re-exports, `modules.shared` global fields/re-exports, `modules.errors` helpers, `modules.devices` dtype/device globals, `modules.paths*` path constants, and extension/plugin import hooks.

### Validation log
- `PYTHONPYCACHEPREFIX=/tmp/gb10-a1111-pycompile-pass1 python3 -m py_compile launch.py webui.py modules/launch_utils.py modules/cmd_args.py modules/paths.py modules/paths_internal.py modules/shared.py modules/shared_state.py modules/timer.py modules/errors.py modules/devices.py` - passed. Initial run without `PYTHONPYCACHEPREFIX` failed because the repo `__pycache__` path was not writable by this user.
- `git diff --check` - passed.

### Next unchecked scope
- Continue with initialization/shared runtime graph: `modules/initialize.py`, `modules/initialize_util.py`, `modules/shared_init.py`, `modules/shared_cmd_options.py`, `modules/options.py`, `modules/shared_options.py`, `modules/shared_items.py`, `modules/import_hook.py`, `modules/logging_config.py`, and adjacent bootstrap/runtime compatibility helpers.


## Pass 2 - initialization/shared runtime graph (2026-06-20)

### Checked scope
- `modules/initialize.py`: `imports`, `check_versions`, `initialize`, `initialize_rest`, nested `load_model`.
- `modules/initialize_util.py`: `server_name`, `fix_torch_version`, `fix_pytorch_lightning`, `fix_asyncio_event_loop_policy`, nested `AnyThreadEventLoopPolicy.get_event_loop`, `restore_config_state_file`, `validate_tls_options`, `get_gradio_auth_creds`, nested `process_credential_line`, `dumpstacks`, `configure_sigint_handler`, nested `sigint_handler`, `configure_opts_onchange`, `setup_middleware`, `configure_cors_middleware`.
- `modules/shared_init.py`: `initialize`.
- `modules/shared_cmd_options.py`: parser preload/parse side effects, dynamic `cmd_opts.webui_is_non_local` and `cmd_opts.disable_extension_access` fields.
- `modules/options.py`: `OptionInfo` and fluent helpers `link`, `js`, `info`, `html`, `needs_restart`, `needs_reload_ui`; `OptionHTML`; `options_section`; `Options` and methods `__setattr__`, `__getattr__`, `set`, `get_default`, `save`, `same_type`, `load`, `onchange`, `dumpjson`, `add_option`, `reorder`, `cast_value`; `OptionsCategory`; `OptionsCategories.register_category`; module-level `categories`.
- `modules/shared_options.py`: restricted option set, category registration, every built-in option section registration and callback/refresh lambda in the file.
- `modules/shared_items.py`: `realesrgan_models_names`, `dat_models_names`, `postprocessing_scripts`, `sd_vae_items`, `refresh_vae_list`, `cross_attention_optimizations`, `sd_unet_items`, `refresh_unet_list`, `list_checkpoint_tiles`, `refresh_checkpoints`, `list_samplers`, `reload_hypernetworks`, `get_infotext_names`, `ui_reorder_categories`, `callbacks_order_settings`, `Shared.sd_model` property/getter/setter and `modules.shared` class replacement side effect.
- `modules/import_hook.py`: import-time xformers disablement and torchvision `functional_tensor` compatibility shim.
- `modules/logging_config.py`: optional `TqdmLoggingHandler`, `setup_logging`.
- Adjacent callers checked for reachability/duplication: `webui.py`, `modules/launch_utils.py`, `modules/shared.py`, `modules/ui.py`, `modules/ui_settings.py`, `modules/script_callbacks.py`, `modules/sysinfo.py`, `modules/api/api.py`, built-in extension option registration files.

### Findings and fix decision
- No safe source deletion or deduplication was found in this slice.
- `modules.initialize.*` functions are all reached from `webui.py` import/startup (`initialize.imports()`, `initialize.check_versions()`, `initialize.initialize()`) or from `initialize.initialize()` itself (`initialize_rest()`), so none are dead.
- `modules.initialize_util.server_name()` and `setup_middleware()` are reached from `webui.api_only()`. TLS, signal, onchange, torch/pytorch-lightning, and config-state helpers are reached through `initialize.initialize()`. `dumpstacks()` is only called by the installed SIGINT handler when `shared.opts.dump_stacks_on_signal` is enabled, so it is live through configuration.
- `modules.initialize_util.get_gradio_auth_creds()` has no first-party call site in the current headless/API startup path. It was preserved as an uncertain public/legacy command-line auth compatibility helper because `--gradio-auth` and `--gradio-auth-path` still exist on `cmd_opts`, and extensions or downstream code may import the helper.
- `modules.shared_cmd_options` is a side-effect parser/preload module. Its imports from `modules.paths_internal` intentionally re-export legacy path names and supply extension preload directories; no imports were removed.
- `modules.options` and `modules.shared_options` are not dead despite many indirect/fluent references: `shared_init.initialize()` constructs `shared.opts` from `shared_options.options_templates`; `modules.shared` re-exports `OptionInfo`, `OptionHTML`, `options_section`, `opts`, and `options_templates`; built-in extensions register additional options through that public surface.
- The `Options.reorder()` section/category normalization logic overlaps conceptually with `Options.dumpjson()` category emission, but they serve different phases: `reorder()` mutates label ordering after dynamic option registration, while `dumpjson()` serializes metadata for clients. No dedupe was safe.
- `modules.shared_items` helper names are option refresh/callback providers or legacy `modules.shared` re-exports. Several are referenced through lambdas in `shared_options`, so plain textual call-site checks undercount reachability.
- `modules.import_hook` is intentionally import-time behavior, not dead code. The xformers `None` sentinel and torchvision fallback protect legacy model/import paths and are not duplicated elsewhere in this slice.
- `modules.logging_config.setup_logging()` is called by `modules.launch_utils` at import time. `TqdmLoggingHandler` is a local optional wrapper selected when tqdm imports; no duplicate logging setup was found.

### Static/dynamic audit map notes
- Runtime startup chain for this slice: `launch_utils` imports `logging_config.setup_logging()` -> `webui.py` imports `initialize_util.fix_pytorch_lightning()` and `initialize.imports()` -> `shared_init.initialize()` builds options/shared state -> `api_only()` calls `initialize.initialize()` and `initialize_util.setup_middleware()`.
- Option graph: `shared_options.options_templates` -> `options.Options` -> `shared.opts`; extensions/scripts can mutate the graph through `shared.options_templates.update(...)`, `shared.options_section(...)`, and `shared.opts.add_option(...)`.
- Dynamic/public surfaces to continue treating conservatively: `modules.initialize_util.get_gradio_auth_creds`, `modules.shared_cmd_options` parser/path exports, `modules.options` fluent option metadata API, `modules.shared_items` callbacks, `modules.import_hook` side effects, and `modules.logging_config` handler class.

### Validation log
- `PYTHONPYCACHEPREFIX=/tmp/gb10-a1111-pycompile-pass2 python3 -m py_compile modules/initialize.py modules/initialize_util.py modules/shared_init.py modules/shared_cmd_options.py modules/options.py modules/shared_options.py modules/shared_items.py modules/import_hook.py modules/logging_config.py webui.py modules/launch_utils.py modules/shared.py modules/ui.py modules/ui_settings.py modules/script_callbacks.py modules/sysinfo.py modules/api/api.py` - passed.
- `git diff --check` - passed.

### Next unchecked scope
- Continue with model/runtime loading and extension/runtime registries: `modules/sd_models.py`, `modules/sd_models_config.py`, `modules/sd_models_types.py`, `modules/sd_vae.py`, `modules/sd_unet.py`, `modules/sd_hijack.py`, `modules/sd_hijack_optimizations.py`, `modules/extensions.py`, `modules/script_loading.py`, `modules/scripts.py`, and adjacent model/extension compatibility helpers.


## Pass 3A - model metadata/loading core (2026-06-20)

### Checked scope
- `modules/sd_models.py`: `ModelType`, `path_is_parent`, `replace_key`, `CheckpointInfo` and methods `__init__`, `register`, `calculate_shorthash`; `setup_model`, `checkpoint_tiles`, `list_models`, `get_closet_checkpoint_match`, `select_checkpoint`, `transform_checkpoint_dict_key`, `get_state_dict_from_checkpoint`, `read_metadata_from_safetensors`, `read_state_dict`, `get_checkpoint_state_dict`, `SkipWritingToConfig`, `check_fp8`, `check_mxfp8`, `check_nvfp4`, `DisableFastModelLoadingForTorchAOQuant`, `check_weight_quantization_mutual_exclusion`, TorchAO policy/filter/apply helpers for MXFP8/NVFP4, `set_model_type`, `set_model_fields`, `remap_sdxl_clip_text_model_state_dict_if_needed`, `load_model_weights`, `enable_midas_autodownload`, `patch_given_betas`, `repair_config`, `rescale_zero_terminal_snr_abar`, `apply_alpha_schedule_override`, `SdModelData` and methods `get_sd_model`, `set_sd_model`, `get_empty_cond`, model move/reload helpers, `instantiate_from_config`, `get_obj_from_str`, `load_model`, `reuse_model_from_already_loaded`, `reload_model_weights`, `unload_model_weights`, `apply_token_merging`.
- `modules/sd_models_config.py`: config path constants, `is_using_v_parameterization_for_sd2`, `guess_model_config_from_state_dict`, `find_checkpoint_config`, `find_checkpoint_config_near_filename`.
- `modules/sd_models_types.py`: `WebuiSdModel` annotation surface.
- `modules/sd_vae.py`: loaded/base VAE state, VAE discovery/resolution/cache helpers, `VaeResolution`, `load_vae`, `_load_vae_dict`, `reload_vae_weights`.
- `modules/sd_unet.py`: UNet option registry state, `list_unets`, `get_unet_option`, `apply_unet`, `SdUnetOption`, `SdUnet`, `create_unet_forward`, `original_forward` compatibility state.
- Adjacent callers checked for reachability/duplication: `modules/initialize.py`, `modules/initialize_util.py`, `modules/shared_items.py`, `modules/api/api.py`, `modules/processing.py`, `modules/img2img.py`, `modules/extras.py`, `modules/ui.py`, `modules/ui_settings.py`, `modules/sd_hijack.py`, `modules/sd_samplers_common.py`, `modules/processing_scripts/refiner.py`, `modules/ui_extra_networks_checkpoints.py`, `modules/ui_extra_networks_checkpoints_user_metadata.py`, built-in model converter extension call sites, and OpenClaw clear-cond-cache extension references.

### Findings and fix decision
- No safe source deletion or deduplication was found in this bounded slice.
- `modules.sd_models.checkpoint_alisases` is a misspelled alias, but it explicitly points at `checkpoint_aliases` for old-name compatibility and was preserved.
- `get_closet_checkpoint_match` is misspelled, but it is live through first-party callers (`modules/ui.py`, `modules/img2img.py`, `modules/processing.py`, `modules/processing_scripts/refiner.py`) and is part of the public `modules.sd_models` surface used by scripts/extensions, so it was preserved as-is.
- `modules.sd_unet.original_forward` is not used by `sd_unet.py` itself, but `modules/sd_hijack.py` writes it while installing/restoring UNet forward patches. The compatibility comment matches current behavior, so it was preserved.
- `modules.sd_vae.get_filename()` is a tiny wrapper around `os.path.basename`, but it centralizes VAE display/cache naming inside the module and is used by multiple in-scope functions. Inlining it would be cosmetic churn.
- `modules.sd_models_config.find_checkpoint_config_near_filename()` is a small wrapper, but it is used both by `find_checkpoint_config()` and the API model listing path to expose nearby YAML config metadata, so it was preserved.
- VAE resolution has several small staged helpers (`resolve_vae_from_setting`, `resolve_vae_from_user_metadata`, `resolve_vae_near_checkpoint`) that look mechanically similar but encode ordered precedence and distinct messages/sources. No dedupe was safe without making the per-source resolution behavior less clear.
- MXFP8 and NVFP4 helper groups in `sd_models.py` intentionally mirror each other. They are real duplication candidates for a future quantization-focused refactor, but they are recent, policy-heavy runtime code with separate config/cache modules and tensor subclasses. This model metadata/loading pass did not collapse them.
- `modules/sd_models_types.py` is annotation-only and not instantiated; it was preserved because it documents the dynamic fields written by model loading and is imported by type-checking/doc surfaces.

### Static/dynamic audit map notes
- Model list/selection chain: `initialize.initialize_rest()` calls `sd_models.list_models()` and `sd_vae.refresh_vae_list()`; model selection flows through `select_checkpoint()` or `get_closet_checkpoint_match()` into `load_model()`/`reload_model_weights()`.
- Checkpoint metadata identity is maintained through `CheckpointInfo.ids`, `checkpoints_list`, and `checkpoint_aliases`; those aliases intentionally include legacy hash/name/title forms and are consumed by UI/API/processing override paths.
- VAE precedence is command-line `--vae-path`, then per-model preference override when enabled, then user metadata, near-checkpoint discovery, then global setting. The staged helper structure matches that precedence and source reporting.
- UNet replacement is callback-driven: `initialize.initialize_rest()` refreshes `sd_unet.unet_options`, `sd_hijack` installs patched forwards with `create_unet_forward()`, and processing/reload paths call `apply_unet()`.
- Dynamic/public surfaces to continue treating conservatively: `modules.sd_models` globals and compatibility aliases, `CheckpointInfo` fields/IDs, VAE/UNet registries, `WebuiSdModel` dynamic-field annotations, and extension-facing `SdUnetOption`/`SdUnet`.

### Validation log
- `PYTHONPYCACHEPREFIX=/tmp/gb10-a1111-pycompile-pass3a python3 -m py_compile modules/sd_models.py modules/sd_models_config.py modules/sd_models_types.py modules/sd_vae.py modules/sd_unet.py modules/initialize.py modules/initialize_util.py modules/shared_items.py modules/api/api.py modules/processing.py modules/img2img.py modules/extras.py modules/ui.py modules/ui_settings.py modules/sd_hijack.py modules/sd_samplers_common.py modules/processing_scripts/refiner.py modules/ui_extra_networks_checkpoints.py modules/ui_extra_networks_checkpoints_user_metadata.py extensions/openclaw-clear-cond-cache/scripts/openclaw_clear_cond_cache.py extensions/sd-webui-model-converter/scripts/convert.py extensions/sd-webui-model-converter/scripts/ui.py` - passed.
- `git diff --check` - passed.

### Next unchecked scope
- Continue with adjacent model/runtime extension surfaces not inspected in this bounded pass: `modules/sd_hijack.py`, `modules/sd_hijack_optimizations.py`, `modules/extensions.py`, `modules/script_loading.py`, `modules/scripts.py`, script callback registries, and model/runtime plugin compatibility helpers.

## Pass 3B - hijack/runtime extension/script registries (2026-06-20)

### Checked scope
- `modules/sd_hijack.py`: optimizer list/apply/undo flow; checkpoint compatibility stub; weighted forward/loss patch helpers; `StableDiffusionModelHijack` lifecycle (`__init__`, `apply_optimizations`, `convert_sdxl_to_ssd`, `hijack`, `undo_hijack`, `apply_circular`, `clear_comments`, `get_prompt_lengths`, `redo_hijack`); textual inversion embedding wrappers; circular Conv2d helper; MPS register-buffer patch.
- `modules/sd_hijack_optimizations.py`: optimization classes and `apply`/`undo` methods; optimizer registration; split attention, InvokeAI, sub-quadratic, and SDPA attention/attnblock implementations; SDPA backend status/setter compatibility helpers.
- `modules/extensions.py`: extension active filtering, metadata parsing, callback ordering metadata, extension repo/status/update helpers, extension discovery/requirements checks, and extension path lookup.
- `modules/script_loading.py`: dynamic module loading and extension preload hook execution.
- `modules/scripts.py`: base script hook API, UI/API metadata setup, extension script dependency ordering, script discovery/loading/reload, callback ordering cache, processing hook dispatch, per-script component callbacks, named argument helper, and compatibility aliases.
- `modules/script_callbacks.py`: callback parameter classes, callback registration/naming, extension/user sorting, callback enumeration/clearing/dispatch, removal helpers, and all `on_*` public registration helpers.
- Adjacent callers checked for reachability/duplication: `modules/initialize.py`, `modules/shared_items.py`, `modules/processing.py`, `modules/ui_settings.py`, `modules/ui_html_extensions.py`, `modules/api/api.py`, `modules/openclaw_cuda_graphs.py`, `modules/models/sd3/other_impls.py`, built-in extensions, OpenClaw clear-cond-cache, sd-webui-incantations, and focused tests referencing script loading/callback behavior.

### Findings and fix decision
- No safe source deletion or deduplication was found in this bounded slice.
- `sd_hijack.fix_checkpoint()` has no first-party call site, but it is an explicit legacy compatibility stub whose docstring explains the moved behavior. It was preserved as extension-facing `modules.sd_hijack` API.
- `TextualInversionEmbeddings` is live through `modules/models/sd3/other_impls.py`; `EmbeddingsWithFixes` is live through SD/SDXL/XLMR hijacks. The two wrappers share a forward path intentionally but adapt different embedding owner types, so no dedupe was safe.
- `add_circular_option_to_conv_2d()` has no first-party call site in this fork, but it monkeypatches a public runtime option path and was preserved as an uncertain compatibility helper. `StableDiffusionModelHijack.apply_circular()` remains the current in-model circular-padding toggle.
- Attention optimization classes deliberately repeat LDM/SGM monkeypatch assignment pairs. A helper could reduce text, but the current explicit `apply()` bodies are the public optimizer surface and make each backend's patched forwards visible; collapsing them would be cosmetic and riskier than useful.
- `get_xformers_flash_attention_op()` is unused first-party but returns `None` as an xformers compatibility surface after xformers disablement; it was preserved.
- `attention_backend_status()`/`set_attention_backend()` are compatibility wrappers around the newer SDPA-only API. They were preserved because current API/OpenClaw code reaches `sdpa_backend_*`, and older clients may still call the generic names.
- `extensions.Extension.read_info_from_repo()` and `check_updates()` duplicate some Git metadata concepts but serve separate cached display and live remote-check flows. No unreachable extension status path was found.
- `ExtensionMetadata.parse_list()`, script dependency parsing, and callback ordering parsing all normalize metadata lists in adjacent registries. They are coupled to different config sections/callers and were not safely mergeable in this pass.
- `scripts.Script.describe()` is documented as unused and has no first-party call site, but it is a base extension API method and was preserved.
- `scripts.ScriptRunner.setup_scrips()` keeps its misspelled name because `modules/processing.py` calls it and external scripts may mirror the typo. No rename/dedupe was attempted.
- `script_callbacks` contains many mechanically similar dispatch and `on_*` registration helpers. They are intentionally separate public callback APIs with different parameter objects, order direction, and category names; no callback registry helper could be removed safely.
- Targeted line-numbered inspection corrected an apparent duplicate `scripts_data = []` and duplicate `return gr.update(visible=False)` seen in a broad truncated dump; the source currently contains only one of each in the relevant locations.

### Static/dynamic audit map notes
- Hijack startup chain: `initialize.initialize()` registers `sd_hijack_optimizations.list_optimizers`; `sd_hijack.model_hijack.hijack()` wraps text encoders, installs weighted forward, applies selected attention optimization, and records `sd_unet.original_forward` for the active model family.
- Extension/script loading chain: `extensions.list_extensions()` builds canonical extension metadata and requirements; `scripts.list_scripts()` orders root/extension scripts; `scripts.load_scripts()` imports them through `script_loading.load_module()` and refreshes callback registries.
- Callback naming/order chain: `script_callbacks.add_callback()` maps callback source files back to extensions via `extensions.find_extension()`; `sort_callbacks()` combines extension metadata order constraints with user option priority lists.
- Dynamic/public surfaces to continue treating conservatively: all `Script` hook methods, `script_callbacks.on_*` registration functions, callback parameter classes, extension metadata/canonical-name handling, typo-compatible helpers (`setup_scrips`), and hijack/attention monkeypatch helpers exposed under `modules.sd_hijack*`.

### Validation log
- `PYTHONPYCACHEPREFIX=/tmp/gb10-a1111-pycompile-pass3b python3 -m py_compile modules/sd_hijack.py modules/sd_hijack_optimizations.py modules/extensions.py modules/script_loading.py modules/scripts.py modules/script_callbacks.py modules/initialize.py modules/shared_items.py modules/processing.py modules/ui_settings.py modules/ui_html_extensions.py modules/api/api.py modules/openclaw_cuda_graphs.py modules/models/sd3/other_impls.py extensions/openclaw-clear-cond-cache/scripts/openclaw_clear_cond_cache.py extensions/sd-webui-incantations/scripts/pag.py extensions/sd-webui-incantations/scripts/cfg_combiner.py extensions/sd-webui-incantations/scripts/smoothed_energy_guidance.py` - passed.
- `git diff --check` - passed.

### Next unchecked scope
- Continue with processing/sampler/runtime execution surfaces not yet audited in detail: `modules/processing.py`, `modules/processing_scripts/*.py`, `modules/sd_samplers*.py`, `modules/sd_samplers_common.py`, sampler configuration/extra networks activation paths, and directly adjacent generation hook compatibility helpers.


## Pass 4 - processing/sampler/runtime execution surfaces (2026-06-20)

### Checked scope
- `modules/processing.py`: color correction/overlay helpers, mask/latent cache helpers, txt2img/img2img image-conditioning helpers, `StableDiffusionProcessing` lifecycle and cond/cache methods, `Processed`, latent decode/image conversion helpers, seed/infotext helpers, `process_images`/`process_images_inner`, old hires-fix dimensions, `StableDiffusionProcessingTxt2Img`, and `StableDiffusionProcessingImg2Img` including img2img init-cache restore/store/status helpers.
- `modules/processing_scripts/comments.py`: prompt comment stripping script and token-counter callback.
- `modules/processing_scripts/refiner.py`: refiner UI/infotext/setup script surface.
- `modules/processing_scripts/sampler.py`: sampler/scheduler UI, infotext paste helpers, and processing setup.
- `modules/processing_scripts/seed.py`: seed/subseed UI, setup, and reuse-seed callback wiring.
- `modules/sd_samplers.py`: sampler registry, visibility/filter maps, infotext sampler/scheduler conversion, hires sampler/scheduler conversion, and processing autocorrection helper.
- `modules/sd_samplers_common.py`: img2img step calculation, latent/image decode helpers, latent preview storage, eta-noise-seed-delta detection, torchsde Brownian patch, refiner application, `TorchHijack`, and base `Sampler` lifecycle/noise/callback methods.
- `modules/sd_samplers_kdiffusion.py`: k-diffusion sampler table/data construction, `CFGDenoiserKDiffusion`, and `KDiffusionSampler` sigma/sample/sample_img2img paths.
- `modules/sd_samplers_timesteps.py` and `modules/sd_samplers_timesteps_impl.py`: DDIM/PLMS/UniPC sampler table, CompVis timestep denoisers, compatibility aliases for `modules.sd_samplers_compvis`, and timestep sampler/sample paths.
- `modules/sd_samplers_lcm.py`: LCM denoiser, sampler function, CFG denoiser, and sampler data.
- `modules/sd_samplers_extra.py`: restart sampler implementation.
- `modules/sd_samplers_cfg_denoiser.py`: cond concatenate/slice/pad helpers, CFG denoiser lifecycle, mask blending, callback dispatch, cond/uncond padding variants, edit-model branch, and preview storage.
- `modules/sd_samplers_cfg_denoised_callback.py`: checked as present in the requested scope only if it exists; no file exists in this checkout.
- `modules/sd_samplers_compvis.py`: zero-byte compatibility placeholder; live aliasing is supplied by `modules/sd_samplers_timesteps.py`.
- `modules/sd_schedulers.py`: scheduler dataclass, sigma/timestep conversion helpers, all scheduler functions, scheduler table, and maps.
- `modules/extra_networks.py`: registry/alias registration, activation/deactivation, prompt parsing, batch extra-network consistency check, and user metadata loader.
- Adjacent callers checked for reachability/duplication: `modules/api/api.py`, `modules/ui.py`, `modules/img2img.py`, `modules/shared_state.py`, `modules/images.py`, `modules/textual_inversion/textual_inversion.py`, `scripts/img2imgalt.py`, `scripts/loopback.py`, `scripts/prompt_matrix.py`, `scripts/sd_upscale.py`, `scripts/xyz_grid.py`, `extensions/sd-webui-incantations/scripts/dynamic_thresholding.py`, `extensions/sd-webui-incantations/scripts/pag.py`, `extensions/openclaw-multi-sampler/scripts/openclaw_multi_sampler.py`, `extensions/openclaw-denoise-ramp/scripts/openclaw_denoise_ramp.py`, and focused tests touching scheduler/processing/cache/callback behavior.

### Findings and fix decision
- No safe source deletion or deduplication was found in this bounded slice.
- Processing helper functions that may look standalone are reached by generation, script, extension, or test paths: color correction/overlay/masks from img2img/inpainting/soft-inpainting, seed helpers from built-in scripts, infotext helpers from grids and processing, and latent decode helpers from processing and shared preview state.
- `StableDiffusionProcessing` and its txt2img/img2img subclasses contain many method hooks that are only reached dynamically by scripts, API processing objects, infotext paste paths, or sampler callbacks. These were preserved as public/runtime surfaces.
- Img2img init-cache helpers are tightly coupled to recent cache-invalidation behavior and are covered by OpenClaw cache tests; no helper was isolated enough to remove or fold.
- `modules/sd_samplers.py` keeps `samples_to_image_grid` and `sample_to_image` re-export aliases for older callers. First-party `shared_state` still imports through `modules.sd_samplers`, so the aliases are live compatibility surface.
- `modules/sd_samplers_compvis.py` is empty, but current installed extension code imports `modules.sd_samplers_compvis.VanillaStableDiffusionSampler`; `modules/sd_samplers_timesteps.py` installs `sys.modules['modules.sd_samplers_compvis']` and the compatibility alias after sampler import. The placeholder was preserved as part of that compatibility story.
- Exact AST duplicate-body scan across the inspected processing/sampler files found only tiny framework/API-shape duplicates: `show()` methods in built-in processing scripts and identical `__init__()` bodies for the two CompVis timestep denoiser wrappers. These were not collapsed because their separate class/method shapes are script and model-adapter API surfaces.
- K-diffusion and CompVis timestep `sample`/`sample_img2img` methods share lifecycle concepts (step setup, extra noise, sampler args, infotext) but diverge on sigma-vs-timestep math, scheduler selection, noise injection, and callback signatures. No safe common helper was introduced.
- Scheduler functions share validation and sigma tensor shaping, and the shared helpers are already factored (`_validate_step_count`, `_as_sigma`, `_stack_sigmas`, `_append_zero`, `_sigmas_from_timesteps`, `_loglinear_interp_sigmas`). Remaining per-scheduler bodies encode distinct algorithms and were preserved.
- Extra-network activation/deactivation both walk registered networks, but one activates all networks with argument lists and fires script callbacks while the other deactivates by parsed data. The similar error handling is not enough to justify a shared helper in this public extension-facing path.

### Static/dynamic audit map notes
- Generation chain: API/UI/scripts construct `StableDiffusionProcessing*` -> `process_images()`/`process_images_inner()` -> prompt/extra-network parsing and activation -> sampler factory -> sampler `sample`/`sample_img2img` -> CFG denoiser callbacks/refiner/preview storage -> decode/infotext/save hooks.
- Sampler registry chain: `sd_samplers.set_samplers()` builds visible maps from k-diffusion, timestep, and LCM sampler data; infotext/API paths normalize sampler/scheduler pairs through `get_sampler_and_scheduler()` and `fix_p_invalid_sampler_and_scheduler()`.
- Compatibility surfaces to continue treating conservatively: `modules.processing` top-level helpers, `StableDiffusionProcessing` fields/methods, sampler re-exports from `modules.sd_samplers`, `modules.sd_samplers_compvis`/`VanillaStableDiffusionSampler`, scheduler labels/aliases, `CFGDenoiser` callback parameter behavior, and extra-network registry/alias functions.

### Validation log
- `PYTHONPYCACHEPREFIX=/tmp/gb10-a1111-pycompile-pass4 python3 -m py_compile modules/processing.py modules/processing_scripts/comments.py modules/processing_scripts/refiner.py modules/processing_scripts/sampler.py modules/processing_scripts/seed.py modules/sd_samplers.py modules/sd_samplers_common.py modules/sd_samplers_compvis.py modules/sd_samplers_extra.py modules/sd_samplers_kdiffusion.py modules/sd_samplers_lcm.py modules/sd_samplers_timesteps.py modules/sd_samplers_timesteps_impl.py modules/sd_samplers_cfg_denoiser.py modules/sd_schedulers.py modules/extra_networks.py modules/api/api.py modules/ui.py modules/img2img.py modules/shared_state.py modules/images.py modules/textual_inversion/textual_inversion.py scripts/img2imgalt.py scripts/loopback.py scripts/prompt_matrix.py scripts/sd_upscale.py scripts/xyz_grid.py extensions/sd-webui-incantations/scripts/dynamic_thresholding.py extensions/sd-webui-incantations/scripts/pag.py extensions/openclaw-multi-sampler/scripts/openclaw_multi_sampler.py extensions/openclaw-denoise-ramp/scripts/openclaw_denoise_ramp.py` - passed.
- `git diff --check` - passed.

### Next unchecked scope
- Continue with image/postprocessing/output/UI-adjacent generation surfaces: `modules/images.py`, `modules/extras.py`, `modules/postprocessing.py`, `modules/generation_parameters_copypaste.py`, `modules/infotext_utils.py`, `modules/ui_common.py`, `modules/ui_components.py`, `modules/ui.py` sections not already audited, and built-in postprocessing scripts/extensions.


## Pass 5 - image/postprocessing/output/UI-adjacent generation surfaces (2026-06-20)

### Checked scope
- `modules/images.py`: font/grid helpers, grid split/combine/annotation helpers, resize/sanitize/sampler-scheduler filename helpers, `FilenameGenerator` and replacement methods, sequence numbering, image save/metadata write paths, info readers, image data reader, and image fix/flatten helpers.
- `modules/extras.py`: PNG info extraction, checkpoint merge config/metadata helpers, tensor half conversion, metadata merge, and model merger handler.
- `modules/postprocessing.py`: caption merge helper, `run_postprocessing`, web UI wrapper, and legacy/API `run_extras` adapter.
- `modules/scripts_postprocessing.py`: `PostprocessedImageSharedInfo`, `PostprocessedImage`, base `ScriptPostprocessing` API, `wrap_call`, and `ScriptPostprocessingRunner` setup/order/run/argument/image-change paths.
- `modules/scripts_auto_postprocessing.py`: main-UI postprocessing script adapter and auto-preprocessing script data factory.
- `modules/infotext_utils.py`: legacy `modules.generation_parameters_copypaste` alias, paste binding classes, image-from-URL/text loading, paste field/button registration, image/dimension sender, old hires-fix restore, inpaint infotext converters, generation parameter parser, override settings helpers, and paste connection path.
- `modules/ui_common.py`: generation-info update, plaintext HTML conversion, log CSV migration, UI save/download/zip path, output panel construction, refresh button, and dialog setup.
- `modules/ui_components.py`: form-compatible Gradio component wrappers, dropdown wrappers, and `InputAccordion` lifecycle/reset behavior.
- `modules/ui_postprocessing.py`: extras/postprocessing UI construction and paste/image-change wiring.
- Remaining relevant `modules/ui.py` generation/output sections: txt2img/img2img output panel usage, infotext paste fields, extras/pnginfo tab construction, compatibility aliases `plaintext_to_html` and `create_output_panel`, and pnginfo paste button registration.
- Built-in postprocessing scripts: `scripts/postprocessing_upscale.py`, `scripts/postprocessing_codeformer.py`, `scripts/postprocessing_gfpgan.py`, and `extensions-builtin/postprocessing-for-training/scripts/postprocessing_caption.py`, `postprocessing_create_flipped_copies.py`, `postprocessing_focal_crop.py`, `postprocessing_split_oversized.py`, `postprocessing_autosized_crop.py`.
- Adjacent callers/tests checked for reachability/duplication: `modules/api/api.py`, `modules/txt2img.py`, `modules/img2img.py`, `modules/shared_items.py`, `modules/ui_settings.py`, `modules/ui_extra_networks_user_metadata.py`, `modules/processing.py`, scripts using image save/output helpers, Lora infotext paste callbacks, and focused image/postprocessing/infotext tests.

### Findings and fix decision
- No safe source deletion or deduplication was found in this bounded slice.
- `modules/generation_parameters_copypaste.py` is intentionally absent in this checkout. The compatibility surface is provided by `modules/infotext_utils.py` installing `sys.modules["modules.generation_parameters_copypaste"] = sys.modules[__name__]`, so no file restoration or removal was needed.
- `modules/ui.py` keeps compatibility aliases (`plaintext_to_html`, `create_output_panel`) for older imports while delegating to `ui_common`; these wrappers are tiny but public and first-party code still imports `plaintext_to_html` through `modules.ui`.
- `modules/postprocessing.run_postprocessing_webui()` is a UI/call-queue signature adapter that discards the `id_task` value supplied by the WebUI submit path; `run_extras()` is still the API extras adapter used by `modules/api/api.py` and tested by `test/test_postprocessing_api_defaults.py`.
- `modules/scripts_postprocessing.ScriptPostprocessing` methods and postprocessing script `ui`/`process`/`process_firstpass` hooks are extension-facing plugin APIs. Empty base methods and similar per-script UI shapes were preserved.
- `ScriptPostprocessingForMainUI` and `ScriptPostprocessingRunner` both adapt postprocessing scripts, but for different surfaces: always-visible generation-tab processing versus Extras/Postprocess tab processing. Their argument/control handling is not a removable duplicate.
- `images.save_image_with_geninfo()` and `images.save_image()` overlap around metadata writing, but `save_image()` owns filename policy, callbacks, replacement/downscale/text side effects, and delegates the actual format-specific metadata write to `save_image_with_geninfo()`.
- `images.read_info_from_image()` and `extras.run_pnginfo()` are complementary, not duplicate parsers: the former extracts generation metadata from image formats; the latter formats that metadata for the PNG Info UI/API path.
- `ui_common.save_files()` is not duplicate save logic despite calling `images.save_image()`: it reconstructs selected gallery outputs from browser data, writes the UI save log, and optionally zips the selected/generated files.
- Infotext parsing/paste code contains several similar mapping loops (`create_override_settings_dict`, `get_override_settings`, paste-field conversion), but they serve different output contracts and rely on dynamic `OptionInfo.infotext`, legacy mappings, script callbacks, and Gradio update values.
- Built-in postprocessing scripts have repeated `InputAccordion`/visibility/blend patterns (GFPGAN and CodeFormer especially), but they call different model APIs and expose distinct settings/infotext fields. A shared helper would be cosmetic and extension-risky.
- A broad text dump initially suggested duplicate lines in `images.get_next_sequence_number()` and `ui_common.create_output_panel()`, but targeted line-numbered inspection of the committed tree showed only one assignment in each location and `git diff` stayed clean. No source change was made.

### Static/dynamic audit map notes
- Postprocessing chain: `ui_postprocessing.create_ui()` builds Extras inputs and `scripts.scripts_postproc.setup_ui()` controls -> submit calls `postprocessing.run_postprocessing_webui()` -> `run_postprocessing()` loads/fixes images, runs `scripts.scripts_postproc.run()`, writes postprocessing infotext/PNG info/captions, saves via `images.save_image()`, and returns output gallery/log HTML.
- API extras chain: `modules/api/api.py` calls `postprocessing.run_extras()` for single and batch extras requests; `run_extras()` maps legacy API fields to postprocessing script arguments and delegates to `run_postprocessing()`.
- Infotext paste chain: UI tabs call `infotext_utils.add_paste_fields()` and `register_paste_params_button()`; final `connect_paste_params_buttons()` wires image transfer, text parsing, override settings, script callbacks, and tab switching.
- Output save chain: processing/scripts/API save generated images through `images.save_image()`; UI gallery save/download goes through `ui_common.save_files()` and then back into `images.save_image()` with parsed infotext metadata.
- Dynamic/public surfaces to continue treating conservatively: image save/read helpers and callbacks, filename pattern replacement names, `modules.generation_parameters_copypaste` alias, `modules.ui` compatibility aliases, postprocessing script base classes/hooks, paste field/button registries, output panel attributes, and Gradio component wrapper subclasses.

### Validation log
- `PYTHONPYCACHEPREFIX=/tmp/gb10-a1111-pycompile-pass5 python3 -m py_compile modules/images.py modules/extras.py modules/postprocessing.py modules/scripts_postprocessing.py modules/scripts_auto_postprocessing.py modules/generation_parameters_copypaste.py modules/infotext_utils.py modules/ui_common.py modules/ui_components.py modules/ui_postprocessing.py modules/ui.py scripts/postprocessing_upscale.py scripts/postprocessing_codeformer.py scripts/postprocessing_gfpgan.py extensions-builtin/postprocessing-for-training/scripts/postprocessing_caption.py extensions-builtin/postprocessing-for-training/scripts/postprocessing_create_flipped_copies.py extensions-builtin/postprocessing-for-training/scripts/postprocessing_focal_crop.py extensions-builtin/postprocessing-for-training/scripts/postprocessing_split_oversized.py extensions-builtin/postprocessing-for-training/scripts/postprocessing_autosized_crop.py` - failed as expected because `modules/generation_parameters_copypaste.py` is intentionally absent and supplied as a runtime alias by `modules/infotext_utils.py`.
- `PYTHONPYCACHEPREFIX=/tmp/gb10-a1111-pycompile-pass5 python3 -m py_compile modules/images.py modules/extras.py modules/postprocessing.py modules/scripts_postprocessing.py modules/scripts_auto_postprocessing.py modules/infotext_utils.py modules/ui_common.py modules/ui_components.py modules/ui_postprocessing.py modules/ui.py scripts/postprocessing_upscale.py scripts/postprocessing_codeformer.py scripts/postprocessing_gfpgan.py extensions-builtin/postprocessing-for-training/scripts/postprocessing_caption.py extensions-builtin/postprocessing-for-training/scripts/postprocessing_create_flipped_copies.py extensions-builtin/postprocessing-for-training/scripts/postprocessing_focal_crop.py extensions-builtin/postprocessing-for-training/scripts/postprocessing_split_oversized.py extensions-builtin/postprocessing-for-training/scripts/postprocessing_autosized_crop.py` - passed.
- `python3 -m pytest -q test/test_images_save.py test/test_postprocessing_script_args.py test/test_postprocessing_api_defaults.py test/test_infotext_api_mappings.py tests/test_save_serialization_contract.py tests/test_postprocessing_caption_contract.py` - passed, 27 tests, 1 existing pytest config warning about unknown `base_url`.
- `git diff --check` - passed.

### Next unchecked scope
- Continue with API/admin/output-adjacent surfaces not covered function-by-function in this pass: `modules/api/api.py` generation/extras/pnginfo/metadata endpoints, `modules/api/models.py`, `modules/api/api_parser.py`, `modules/progress.py`, `modules/call_queue.py`, `modules/ui_extensions.py`, `modules/ui_settings.py`, `modules/ui_loadsave.py`, `modules/ui_tempdir.py`, `modules/ui_html_extensions.py`, and remaining administrative UI/API compatibility helpers.


## Pass 6 - API/admin/output-adjacent surfaces (2026-06-20)

### Checked scope
- `modules/api/api.py`: OpenClaw precision-map helpers and cache key construction, script arg/default helpers, infotext API conversion, sampler/upscaler/image encode/decode helpers, middleware, every `Api` route registration and endpoint method, task queue cleanup, config/options/model listing endpoints, training/create endpoints, extension listing, and server stop/restart helpers.
- `modules/api/models.py`: dynamic processing API model generator, API request/response models, generated options/flags models, list item models, script/extension metadata models, and API field defaults/descriptions.
- `modules/api/api_parser.py`: requested file is absent in this checkout; API argument/model generation currently lives in `modules/api/models.py` and endpoint conversion helpers in `modules/api/api.py`.
- `modules/progress.py`: task lifecycle queues, internal progress API models/routes, live-preview serialization, pending/completed task reporting, result recording, and restore-progress helper.
- `modules/call_queue.py`: queued GPU/UI wrapper stack, progress result recording, state cleanup, memmon/profiling/performance HTML append paths, and queue lock export.
- `modules/ui_extensions.py`: extension access/apply/update/config-state helpers, extension table rendering, install/index filtering/search callbacks, git metadata preload, and extension admin UI construction.
- `modules/ui_settings.py`: settings component creation/value refresh, settings apply/single-setting apply, settings UI construction, quicksettings, reload-script/checkpoint/sysinfo actions, and settings search.
- `modules/ui_loadsave.py`: legacy choice normalization, component/default tracking, UI config read/write, change iteration, review/apply UI, and setup wiring.
- `modules/ui_tempdir.py`: temp file registration/checking, PIL temp-file override, temp-dir option handling, cleanup, and temp-path classifier.
- `modules/ui_html_extensions.py`: script/css injection helpers and Gradio template response override.
- Adjacent administrative compatibility call sites checked: `modules/ui.py`, `modules/ui_toprow.py`, `modules/ui_common.py`, `modules/ui_postprocessing.py`, `modules/ui_checkpoint_merger.py`, `modules/infotext_utils.py`, `extensions-builtin/extra-options-section/scripts/extra_options_section.py`, and focused API/UI contract tests.

### Findings and fixes
- Deduplicated the identical extension-index refresh/search callback bodies in `modules/ui_extensions.py` by adding `refresh_available_extensions_filtered()` and keeping the existing Gradio-facing `refresh_available_extensions_for_tags()` and `search_extensions()` signatures as thin wrappers. This removes a real duplicate while preserving callback input ordering and public function names.
- The only other exact duplicate function-body finding was `fastapi_exception_handler()` and `http_exception_handler()` in `modules/api/api.py`; both are separate FastAPI-decorated handlers already delegating to the shared local `handle_exception()`, so no additional source change was useful.
- `modules/api/api_parser.py` is not present. No stale import or missing file reference was found in the checked scope.
- API request parameter conversion has similar txt2img/img2img blocks, but the branches diverge on init image/mask handling, output paths, timing payload, and include-init-image behavior. No safe shared helper was introduced.
- `setUpscalers()`, `api_infotext_value_for_field()`, `processed_js_with_image_paths()`, and `ScriptArgsList` are covered by focused tests or active API generation paths and were preserved.
- `modules/progress.py` and `modules/call_queue.py` intentionally share task lifecycle concepts but serve different layers: `/internal/progress`/restore-progress polling versus queued UI/API execution wrappers. The queue/result helpers are still reached from `modules/ui.py`, API generation endpoints, and UI top-row restore buttons.
- `modules/ui_tempdir.py` keeps legacy Gradio temp-file compatibility branches (`temp_file_sets` and `temp_dirs`) because the UI temp override and infotext image loading still call them, and upstream Gradio compatibility differs by version.
- `modules/ui_settings.py`, `modules/ui_loadsave.py`, and `modules/ui_html_extensions.py` are UI construction/compatibility surfaces with dynamic callbacks and settings metadata. No dead option serialization/default handling helper was isolated enough to remove.

### Static/dynamic audit map notes
- API generation chain: FastAPI route -> `Api.text2imgapi()`/`img2imgapi()` -> infotext/script arg conversion -> task queued in `modules.progress` -> `queue_lock` guarded processing -> `processed_js_with_image_paths()` response metadata.
- UI queue chain: Gradio callbacks wrap through `call_queue.wrap_ui_gpu_call()` or `wrap_ui_call_no_job()` -> `progress.start_task()`/`record_results()` -> top-row `progress.restore_progress()` can recover recent UI results.
- Admin extension chain: `ui_extensions.create_ui()` wires Installed/Available/Install/Backup tabs; tag changes and search now share `refresh_available_extensions_filtered()` while retaining their distinct callback signatures.
- Compatibility surfaces to continue treating conservatively: API model field names/aliases, `/sdapi/v1/*` endpoint names, internal `/internal/progress` payloads, `modules.ui` re-exported queue/settings helpers, Gradio temp-dir monkeypatches, UI defaults config paths, and extension admin helper names.

### Validation log
- `PYTHONPYCACHEPREFIX=/tmp/gb10-a1111-pycompile-pass6 python3 -m py_compile modules/api/api.py modules/api/models.py modules/progress.py modules/call_queue.py modules/ui_extensions.py modules/ui_settings.py modules/ui_loadsave.py modules/ui_tempdir.py modules/ui_html_extensions.py modules/ui.py modules/ui_toprow.py modules/ui_common.py modules/ui_postprocessing.py modules/ui_checkpoint_merger.py modules/infotext_utils.py extensions-builtin/extra-options-section/scripts/extra_options_section.py` - passed.
- Initial focused pytest command used the wrong `tests/` prefix for legacy `test/` files and failed at collection with `file or directory not found: tests/test_infotext_api_mappings.py`; rerun used the correct mixed paths.
- `python3 -m pytest -q tests/test_ui_extensions_contract.py tests/test_ui_loadsave_contract.py test/test_infotext_api_mappings.py test/test_postprocessing_api_defaults.py tests/test_save_serialization_contract.py tests/test_api_progress_contract.py` - passed: 21 passed, 1 warning (`PytestConfigWarning: Unknown config option: base_url`).
- `git diff --check` - passed.

### Next unchecked scope
- Continue with remaining administrative/model metadata and training-adjacent surfaces not yet audited in detail: `modules/sysinfo.py`, `modules/config_states.py`, `modules/ui_checkpoint_merger.py`, `modules/hypernetworks/*`, `modules/textual_inversion/*`, `modules/deepbooru.py`, `modules/interrogate.py` if present, and adjacent API/admin compatibility helpers.


## Pass 7 - administrative/model metadata and training-adjacent surfaces (2026-06-20)

### Checked scope
- `modules/sysinfo.py`: checksum generation/verification, environment/argv redaction, CPU/RAM/package/torch collection, git status helpers, extension fallback inventory, and config loading fallback.
- `modules/config_states.py`: saved config-state listing, webui git snapshot, extension snapshot, combined config serialization, and webui/extension restore helpers.
- `modules/ui_checkpoint_merger.py`: interpolation description callback, model merger wrapper, checkpoint merger UI construction, metadata read callback, and queued merge callback wiring.
- `modules/hypernetworks/hypernetwork.py`: `HypernetworkModule`, dropout parsing, `Hypernetwork` load/save/hash/lifecycle methods, hypernetwork registry/load/apply helpers, attention forward compatibility hook, condition stacking, loss statistics, create/train/save helpers.
- `modules/hypernetworks/ui.py`: UI adapter functions for create/train hypernetwork.
- `modules/textual_inversion/textual_inversion.py`: template listing, `Embedding`, embedding directory watcher, `EmbeddingDatabase`, embedding creation/loading/hash/image-embedding paths, loss/tensorboard helpers, training validation, train/save embedding helpers.
- `modules/textual_inversion/dataset.py`: dataset entries, `PersonalizedBase`, grouped sampler, data loader, batch collate/loaders for deterministic and random latent sampling.
- `modules/textual_inversion/autocrop.py`: focal crop pipeline, face/corner/entropy point selection, orientation helpers, OpenCV model download/cache, point/settings classes.
- `modules/textual_inversion/image_embedding.py`: embedding JSON encoder/decoder, base64 conversion, image-side embedding encode/decode helpers, caption overlay helper and deprecated `textfont` compatibility argument.
- `modules/textual_inversion/learn_schedule.py`: schedule iterator and optimizer learning-rate scheduler.
- `modules/textual_inversion/saving_settings.py`: shared textual-inversion/hypernetwork training settings serialization.
- `modules/textual_inversion/ui.py`: UI adapter functions for create/train embedding.
- `modules/deepbooru.py`: `DeepDanbooru` model load/start/stop/tag/tag_multi path and module singleton.
- `modules/interrogate.py`: category discovery/download, BLIP/CLIP loader lifecycle, ranking, caption generation, and full interrogate path.
- Adjacent compatibility/admin callers checked: `modules/api/api.py` create/train/interrogate endpoints, `modules/api/models.py` interrogate model request field, `modules/ui.py` training/interrogate/model-merger/sysinfo wiring, `modules/ui_extensions.py` config-state backup/restore/table wiring, `modules/ui_settings.py` sysinfo verification UI, `modules/shared_init.py`, `modules/shared_options.py`, `scripts/loopback.py`, `extensions/openclaw-clear-cond-cache/scripts/openclaw_clear_cond_cache.py`, and focused contract tests under `tests/` and legacy `test/`.

### Findings and fix decision
- No safe source deletion or deduplication was found in this bounded slice, so this pass is a ledger-only checkpoint.
- `modules/sysinfo.py` and `modules/config_states.py` both serialize git/config/extension metadata, but for different contracts: sysinfo produces a signed diagnostic bundle with redacted argv/environment, packages, exceptions, startup timing, and fallback extension inventory; config states produce restorable webui/extension snapshots for the Extensions backup/restore UI and startup restore path. The similar repo fields were not merged.
- `sysinfo.get_info_from_repo_path()` and `config_states.get_webui_config()` overlap on git metadata concepts but use different inputs, fallback behavior, and consumers. A shared helper would either pull GitPython into sysinfo fallback paths or lose the existing diagnostic behavior, so it was not safe.
- `ui_checkpoint_merger.modelmerger()` is a thin UI error-handling wrapper around `extras.run_modelmerger()` rather than a dead duplicate. It refreshes checkpoint dropdowns after failures and is the queued Gradio callback for the Model Merger tab.
- Hypernetwork and textual-inversion training loops share dataset, learning-rate, validation, loss logging, tensorboard, preview-image, and save-settings helpers where the existing code already made that safe. The remaining apparent duplication is workflow-specific: hypernetwork trains module weights and restores RNG/optimization state, while textual inversion trains embedding vectors and optionally embeds saved TI data into generated preview images.
- `modules/textual_inversion/dataset.py` has tiny exact duplicate method shape in `LearnScheduleIterator.__iter__()` and `BatchLoaderRandom.pin_memory()`, both single-line API methods with different class contracts. Collapsing them would add indirection without removing meaningful logic.
- `BatchLoader`/`BatchLoaderRandom` and `collate_wrapper`/`collate_wrapper_random` are not dead duplicates: random latent sampling must avoid pinning freshly sampled random latents while deterministic/once sampling pins stacked latent samples when configured.
- `caption_image_overlay(textfont=None)` keeps an unused/deprecated argument intentionally; it emits a deprecation warning and preserves older callers.
- `interrogate` and `deepbooru` both expose image-to-text tagging, but they load different models, options, ranking behavior, and UI/API selections (`clip` versus `deepdanbooru`). No shared helper was useful.
- Several top-level helpers have only internal textual references because they are Gradio/API/option callback surfaces (`interrogate.category_types`, create/train adapters, sysinfo check/download, config state callbacks). They were preserved as public/dynamic compatibility hooks.

### Static/dynamic audit map notes
- Sysinfo chain: launch `--dump-sysinfo` or `/internal/sysinfo*` calls `sysinfo.get()`; Settings Sysinfo tab calls `sysinfo.check()` to validate uploaded diagnostic bundles; extra-network metadata uses `sysinfo.pretty_bytes()` for file-size display.
- Config-state chain: Extensions backup tab calls `config_states.get_config()`, `list_config_states()`, and restore helpers; startup `initialize_util.restore_config_state_file()` can restore extension state from a saved config file.
- Training chain: UI/API create/train endpoints call textual-inversion and hypernetwork create/train helpers; both share `PersonalizedBase`, `PersonalizedDataLoader`, `LearnRateScheduler`, `validate_train_inputs()`, loss CSV/tensorboard helpers, and settings serialization.
- Interrogate chain: shared initialization constructs `shared.interrogator`; UI top-row and API `/sdapi/v1/interrogate` dispatch to either CLIP/BLIP `shared.interrogator.interrogate()` or `deepbooru.model.tag()`; loopback can append either result during img2img iteration.
- Compatibility surfaces to continue treating conservatively: sysinfo JSON/checksum field names, config-state JSON keys, model merger component IDs/callback signatures, training API argument names, embedding PNG metadata keys, image-embedding sidecar encoding, hypernetwork `.pt`/`.optim` schema, CLIP category filenames/options, and DeepBooru tag formatting options.

### Validation log
- `PYTHONPYCACHEPREFIX=/tmp/gb10-a1111-pycompile-pass7 python3 -m py_compile modules/sysinfo.py modules/config_states.py modules/ui_checkpoint_merger.py modules/deepbooru.py modules/interrogate.py modules/hypernetworks/hypernetwork.py modules/hypernetworks/ui.py modules/textual_inversion/textual_inversion.py modules/textual_inversion/dataset.py modules/textual_inversion/image_embedding.py modules/textual_inversion/ui.py modules/textual_inversion/saving_settings.py modules/textual_inversion/learn_schedule.py modules/textual_inversion/autocrop.py modules/api/api.py modules/api/models.py modules/ui.py modules/ui_extensions.py modules/ui_settings.py modules/shared_init.py modules/shared_options.py scripts/loopback.py extensions/openclaw-clear-cond-cache/scripts/openclaw_clear_cond_cache.py extensions-builtin/postprocessing-for-training/scripts/postprocessing_focal_crop.py` - passed.
- `python3 -m pytest -q tests/test_hypernetwork_creation_contract.py tests/test_textual_inversion_preview_save_contract.py tests/test_textual_inversion_autocrop_contract.py tests/test_textual_inversion_preview_save_contract.py test/test_extras.py tests/test_ui_extensions_contract.py tests/test_api_progress_contract.py` - failed: 7 passed, 3 errors, 1 existing pytest config warning. The errors were all from legacy live-server `test/test_extras.py` needing unavailable `base_url`.
- `python3 -m pytest -q tests/test_hypernetwork_creation_contract.py tests/test_textual_inversion_preview_save_contract.py tests/test_textual_inversion_autocrop_contract.py tests/test_ui_extensions_contract.py tests/test_api_progress_contract.py tests/test_textual_inversion_preview_save_contract.py` - passed: 7 passed, 1 existing pytest config warning (`Unknown config option: base_url`). `tests/test_textual_inversion_preview_save_contract.py` was listed twice by mistake; pytest collected it once.
- `git diff --check` - passed.

### Next unchecked scope
- Continue with remaining model/UI/metadata surfaces not yet audited function-by-function: `modules/ui_extra_networks*.py`, `modules/ui_prompt_styles.py`, `modules/styles.py`, `modules/prompt_parser.py`, `modules/sd_emphasis.py`, `modules/sd_hijack_clip.py`, `modules/sd_hijack_clip_old.py`, `modules/sd_hijack_open_clip.py`, `modules/sd_hijack_unet.py`, `modules/openclaw_*`, and remaining built-in extension/script compatibility helpers.

## Pass 8 - prompt/style/extra-network/CLIP/OpenClaw-specific surfaces (2026-06-20)

### Checked scope
- `modules/ui_extra_networks.py`: preview extension allow-list helpers, extra-network route endpoints, metadata/card JSON endpoints, tree construction, card/tree/dirs HTML builders, search/sort/path helpers, page registration/default page setup, Gradio UI construction, gallery selection wiring, and path parent checks.
- `modules/ui_extra_networks_checkpoints.py`, `modules/ui_extra_networks_checkpoints_user_metadata.py`, `modules/ui_extra_networks_hypernets.py`, `modules/ui_extra_networks_textual_inversion.py`, and `modules/ui_extra_networks_user_metadata.py`: built-in extra-network page classes, item factories/listing, user metadata editor save/load/card/preview handlers, checkpoint VAE metadata helpers, and preview path filtering.
- `modules/ui_prompt_styles.py` and `modules/styles.py`: style selection/save/delete/materialize/refresh callbacks, style CSV load/save, prompt merge/extract helpers, style path tracking, and prompt-style UI wiring.
- `modules/prompt_parser.py`: schedule parser, composable conditioning classes, multicond prompt splitting, condition reconstruction/stacking, and attention/emphasis parser.
- `modules/sd_emphasis.py`: emphasis mode classes, option lookup, option description generation, and CLIP multiplier application hooks.
- `modules/sd_hijack_clip.py`, `modules/sd_hijack_clip_old.py`, and `modules/sd_hijack_open_clip.py`: CLIP/OpenCLIP tokenizer wrappers, prompt chunk/fix handling, textual-inversion embedding insertion, new and old emphasis paths, pooled-output handling, embedding init helpers, and SDXL/OpenCLIP variants.
- `modules/sd_hijack_unet.py`: UNet dtype/upcast monkeypatches, timestep embedding patching, SpatialTransformer patching, OpenCLIP GELU patching, ddpm/ddpm_edit first-stage/apply-model wrappers, and SGM wrapper compatibility hooks.
- `modules/openclaw_cuda_graphs.py` and `modules/openclaw_generation_diagnostics.py`: CUDA graph status/control/cache-key/bypass/capture/replay helpers, SEG key/bypass logic, diagnostic JSON sanitization, CUDA graph counter summaries, request summaries, before/after sample capture, and last-diagnostics endpoint helper.
- Adjacent callers/tests checked for reachability and dynamic contracts: `modules/ui.py`, `modules/ui_toprow.py`, `modules/initialize.py`, `modules/api/api.py`, `modules/processing.py`, `modules/sd_hijack.py`, `modules/sd_samplers_cfg_denoiser.py`, `modules/shared_init.py`, `modules/shared_options.py`, `javascript/extraNetworks.js`, `modules/models/sd3/sd3_cond.py`, focused prompt/style/extra-network/OpenClaw tests, and route/API references.

### Findings and fixes
- Deduplicated two stale nested `CondFunc` registrations in `modules/sd_hijack_unet.py`: the older conditional `ldm.models.diffusion.ddpm.LatentDiffusion.apply_model` wrapper and older conditional `ldm.modules.diffusionmodules.openaimodel.timestep_embedding` cast wrapper. Newer unconditional wrappers below them already route through `apply_model()` and `timestep_embedding_cast_result()` for the same ldm targets, while preserving dtype/upcast decisions and the SGM wrapper path. Removing the older inner wrappers avoids double wrapping/casting without changing the active public hook names.
- `modules/sd_hijack_unet.py` still keeps the custom on-device `timestep_embedding` patch, `SpatialTransformer.forward` patch, conditional GroupNorm/GEGLU/OpenCLIP GELU patches, first-stage VAE dtype wrappers, unconditional ldm/sgm `apply_model` wrappers, and ldm/sgm timestep result-cast wrappers.
- No safe deletion was made in extra-network UI routes or JavaScript-facing helpers. `add_pages_to_demo()` has only route-local references in this grep slice, but `/sd_extra_networks/*` endpoints are consumed by `javascript/extraNetworks.js` and the function is a route-registration compatibility surface, so it was preserved.
- Extra-network metadata/card helpers (`fetch_cover_images()`, `get_metadata()`, `get_single_card()`, `link_preview()`, `find_embedded_preview()`, metadata editor save/preview helpers) share path/metadata concepts but serve different request/UI contracts. The dynamic Gradio and JavaScript card-refresh behavior made further dedupe unsafe.
- Prompt parsing and emphasis logic intentionally remains split: `prompt_parser.get_learned_conditioning_prompt_schedules()` handles scheduled/alternate prompt syntax, `prompt_parser.parse_prompt_attention()` handles attention weights/BREAK chunks, and `sd_emphasis` applies configured multiplier behavior after transformer encoding. The old CLIP emphasis implementation is still reachable through `opts.use_old_emphasis_implementation` and was preserved.
- Style helpers contain small top-level callback functions that are directly wired into Gradio events. They are not dead despite low textual fanout.
- `sd_hijack_clip_old.py` is a compatibility path for the `use_old_emphasis_implementation` option and is imported dynamically by `FrozenCLIPEmbedderWithCustomWordsBase.forward()`.
- OpenCLIP wrapper classes duplicate some tokenizer/id setup intentionally because they wrap different upstream embedder return contracts (`encode_with_transformer()` tensor versus dict with pooled output). No safe shared base helper was introduced.
- OpenClaw CUDA graph and generation diagnostic helpers are active through `/sdapi/v1/openclaw/cuda-graphs`, `/sdapi/v1/openclaw/generation-diagnostics`, `sd_samplers_cfg_denoiser`, and `processing`. Their private helpers are covered by focused tests or are cache-key/bypass internals and were preserved.

### Static/dynamic audit map notes
- Extra-network UI chain: `initialize.initialize()` registers built-in pages -> `ui_extra_networks.create_ui()` builds txt2img/img2img extra-network tabs -> JavaScript calls `/sd_extra_networks/metadata`, `/sd_extra_networks/get-single-card`, `/sd_extra_networks/thumb`, and `/sd_extra_networks/cover-images` for metadata, refreshed cards, previews, and embedded cover images.
- Style chain: `ui_toprow.Toprow` creates `UiPromptStyles`; style callbacks read/write `shared.prompt_styles`, which is initialized by `shared_init.initialize()` from `styles.StyleDatabase`.
- Prompt/conditioning chain: processing calls `prompt_parser.get_multicond_learned_conditioning()` and reconstruction helpers through sampler denoiser paths; CLIP hijack wrappers call `parse_prompt_attention()` and `sd_emphasis.get_current_option()` during text encoding.
- CLIP hijack chain: `sd_hijack.py` swaps applicable conditioner embedders to CLIP/OpenCLIP custom-word wrappers; textual inversion embedding fixes and emphasis metadata are recorded through the shared hijack state.
- OpenClaw CUDA graph chain: API toggles `openclaw_cuda_graphs.set_enabled()`/`status()`; `sd_samplers_cfg_denoiser` routes denoiser calls through `openclaw_cuda_graphs.run()`; `processing` records before/after diagnostics and API exposes the last diagnostic snapshot.
- Compatibility surfaces to continue treating conservatively: `/sd_extra_networks/*` endpoints, extra-network card template keys/data attributes, Gradio callback signatures, style CSV schema, prompt schedule/attention syntax, `use_old_emphasis_implementation`, CLIP/OpenCLIP wrapper class names, CondFunc target strings, OpenClaw `/sdapi/v1/openclaw/*` routes, CUDA graph status field names, and diagnostic JSON keys.

### Validation log
- `PYTHONPYCACHEPREFIX=/tmp/gb10-a1111-pycompile-pass8 python3 -m py_compile modules/ui_extra_networks.py modules/ui_extra_networks_checkpoints.py modules/ui_extra_networks_checkpoints_user_metadata.py modules/ui_extra_networks_hypernets.py modules/ui_extra_networks_textual_inversion.py modules/ui_extra_networks_user_metadata.py modules/ui_prompt_styles.py modules/styles.py modules/prompt_parser.py modules/sd_emphasis.py modules/sd_hijack_clip.py modules/sd_hijack_clip_old.py modules/sd_hijack_open_clip.py modules/sd_hijack_unet.py modules/openclaw_cuda_graphs.py modules/openclaw_generation_diagnostics.py` - passed before the combined pytest command reached test collection.
- `python3 -m pytest -q test/test_styles.py test/test_openclaw_cuda_graphs.py tests/test_extra_networks_path_contract.py tests/test_extra_networks_metadata_contract.py` - failed during collection because the system Python environment on GB10 does not have `torch` installed for `test/test_openclaw_cuda_graphs.py`; no tests ran after that collection error.
- `python3 -m pytest -q test/test_styles.py tests/test_extra_networks_path_contract.py tests/test_extra_networks_metadata_contract.py` - passed: 6 passed, 1 existing pytest config warning (`Unknown config option: base_url`).
- `git diff --check` - passed.

### Next unchecked scope
- More slices are still needed for the remaining built-in extension/script compatibility surfaces and lower-level model/runtime helpers not yet audited function-by-function, especially `extensions-builtin/*/scripts/*.py`, `scripts/*.py`, remaining `modules/sd_*` loaders/samplers/optimizations not covered in prompt/CLIP scope, and any adjacent compatibility helpers discovered from those entry points.


## Pass 9 - built-in extension/script compatibility surfaces (2026-06-20)

### Checked scope
- Built-in extension script entry points: `extensions-builtin/LDSR/scripts/ldsr_model.py`, `extensions-builtin/Lora/scripts/lora_script.py`, `extensions-builtin/ScuNET/scripts/scunet_model.py`, `extensions-builtin/SwinIR/scripts/swinir_model.py`, `extensions-builtin/canvas-zoom-and-pan/scripts/hotkey_config.py`, `extensions-builtin/extra-options-section/scripts/extra_options_section.py`, `extensions-builtin/hypertile/scripts/hypertile_script.py`, `extensions-builtin/postprocessing-for-training/scripts/postprocessing_autosized_crop.py`, `extensions-builtin/postprocessing-for-training/scripts/postprocessing_caption.py`, `extensions-builtin/postprocessing-for-training/scripts/postprocessing_create_flipped_copies.py`, `extensions-builtin/postprocessing-for-training/scripts/postprocessing_focal_crop.py`, `extensions-builtin/postprocessing-for-training/scripts/postprocessing_split_oversized.py`, and `extensions-builtin/soft-inpainting/scripts/soft_inpainting.py`.
- Core script entry points: `scripts/custom_code.py`, `scripts/img2imgalt.py`, `scripts/loopback.py`, `scripts/mxfp8_diagnostics_api.py`, `scripts/outpainting_mk_2.py`, `scripts/poor_mans_outpainting.py`, `scripts/postprocessing_codeformer.py`, `scripts/postprocessing_gfpgan.py`, `scripts/postprocessing_upscale.py`, `scripts/prompt_matrix.py`, `scripts/prompts_from_file.py`, `scripts/sd_upscale.py`, and `scripts/xyz_grid.py`.
- Directly adjacent helpers reached from those entry points where needed for reachability/duplication decisions: Lora registration/API/paste hooks and sibling helper modules, postprocessing script contracts, upscaler option registrations, Hypertile XYZ-axis registration, soft-inpainting mask/kernel helpers, xyz-grid axis helper classes/functions, and img2img/outpainting processing wrappers.

### Findings and fixes
- Deduplicated the two identical nested tuple-to-NumPy helper functions named `vec()` in `extensions-builtin/soft-inpainting/scripts/soft_inpainting.py` by extracting one private module-local helper. Both `weighted_histogram_filter()` and `get_gaussian_kernel()` now share it without changing call sites or public script behavior.
- The remaining exact duplicate body in this slice is `show()` returning `scripts.AlwaysVisible` in `extra-options-section` and `hypertile`. These are separate `scripts.Script` hook methods on separate built-in extensions, so they were preserved as compatibility surface rather than abstracted.
- `hotkey_config.py` is an intentionally empty script-side compatibility placeholder for the canvas zoom/pan built-in; no dead callable was present.
- Lora script callbacks (`on_model_loaded`, `on_script_unloaded`, `on_before_ui`, `on_app_started`, and infotext paste hooks) have low textual fanout but are public callback/API registration surfaces and were preserved. The Lora API JSON helper is active through `/sdapi/v1/loras`.
- Built-in upscaler scripts share option-registration and load/upscale shapes, but their model discovery, device selection, caching, architecture expectations, URL defaults, and tile semantics differ enough that no safe shared helper was introduced.
- Postprocessing scripts intentionally share the `ScriptPostprocessing` UI/process pattern. The face restoration scripts have similar visibility blending, but they call different restoration backends and expose different generation metadata, so no cross-script helper was introduced.
- Img2img alternative, loopback, SD upscale, prompt matrix, prompts-from-file, outpainting, and XYZ grid helpers are user-visible script contracts with dynamic Gradio/infotext/API behavior. No dead Script methods or stale argument parsers were isolated enough to remove.
- `scripts/mxfp8_diagnostics_api.py` is a callback-registered OpenClaw API surface and was preserved despite having only registration-style reachability.

### Static/dynamic audit map notes
- Script loader chain: `modules.scripts` discovers these `scripts.Script` and `scripts_postprocessing.ScriptPostprocessing` subclasses dynamically, so `title()`, `show()`, `ui()`, `run()`, `process()`, `process_firstpass()`, and `image_changed()` methods are treated as entry-point hooks even when grep fanout is low.
- Built-in callback chain: extension scripts register UI settings, before-UI hooks, app routes, model-loaded hooks, infotext paste hooks, and unload hooks through `modules.script_callbacks`; these callback functions are not dead by textual search alone.
- Postprocessing chain: core extras and postprocessing-for-training scripts share target dimensions through `pp.shared.target_width`/`target_height`; crop/split/focal-crop scripts depend on `postprocessing_upscale` first-pass target propagation.
- XYZ/Hypertile chain: `hypertile_script.add_axis_options()` locates the dynamically loaded `xyz_grid.py` script module and extends `axis_options`, so `xyz_grid` helper functions/classes remain public extension points.
- Compatibility surfaces to continue treating conservatively: Script method signatures and return ordering, postprocessing argument dict keys, infotext field names, `/sdapi/v1/loras` and `/sdapi/v1/refresh-loras`, Gradio element IDs, XYZ axis labels/value parsing, outpainting image/mask tiling behavior, and OpenClaw MXFP8 diagnostic route fields.

### Validation log
- `PYTHONPYCACHEPREFIX=/tmp/gb10-a1111-pycompile-pass9 python3 -m py_compile extensions-builtin/LDSR/scripts/ldsr_model.py extensions-builtin/Lora/scripts/lora_script.py extensions-builtin/ScuNET/scripts/scunet_model.py extensions-builtin/SwinIR/scripts/swinir_model.py extensions-builtin/canvas-zoom-and-pan/scripts/hotkey_config.py extensions-builtin/extra-options-section/scripts/extra_options_section.py extensions-builtin/hypertile/scripts/hypertile_script.py extensions-builtin/postprocessing-for-training/scripts/postprocessing_autosized_crop.py extensions-builtin/postprocessing-for-training/scripts/postprocessing_caption.py extensions-builtin/postprocessing-for-training/scripts/postprocessing_create_flipped_copies.py extensions-builtin/postprocessing-for-training/scripts/postprocessing_focal_crop.py extensions-builtin/postprocessing-for-training/scripts/postprocessing_split_oversized.py extensions-builtin/soft-inpainting/scripts/soft_inpainting.py scripts/custom_code.py scripts/img2imgalt.py scripts/loopback.py scripts/mxfp8_diagnostics_api.py scripts/outpainting_mk_2.py scripts/poor_mans_outpainting.py scripts/postprocessing_codeformer.py scripts/postprocessing_gfpgan.py scripts/postprocessing_upscale.py scripts/prompt_matrix.py scripts/prompts_from_file.py scripts/sd_upscale.py scripts/xyz_grid.py` - passed.
- `python3 -m pytest -q tests/test_postprocessing_caption_contract.py tests/test_save_serialization_contract.py test/test_postprocessing_script_args.py test/test_infotext_api_mappings.py` - passed: 16 passed, 1 existing pytest config warning (`Unknown config option: base_url`).
- `git diff --check` - passed.
- No JavaScript files were touched in this pass, so `node --check` was not applicable.

### Next unchecked scope
- More slices are still needed. Recommended next slice: lower-level model/runtime helper surfaces not yet audited function-by-function, especially remaining `modules/sd_*` loaders/samplers/optimizations outside the pass 8 prompt/CLIP/UNet scope, `modules/sd_models*.py`, `modules/sd_samplers*.py`, `modules/sd_vae*.py`, `modules/processing*.py` adjacent helpers not already covered, and any remaining `modules/openclaw_*` runtime compatibility helpers discovered from those paths.


## Pass 10 - OpenClaw runtime compatibility helpers and adjacent low-level runtime helpers (2026-06-20)

### Checked scope
- `modules/openclaw_cuda_graphs.py`: environment/cache-size readers, status/set/clear API helpers, tensor/structure signature generation, static tensor cloning/copying, cache eviction, model/LoRA/attention/SEG cache-key construction, SEG and mask bypass logic, bypass stat recording, failed-key tracking, capture/replay/fallback path, and exception recording.
- `modules/openclaw_generation_diagnostics.py`: JSON-safe conversion, CUDA graph status retrieval/summarization, graph-key hashing, counter delta calculation, OpenClaw extra-parameter filtering, request summary construction, before/after sample capture, per-processing history storage, and last-diagnostics API helper.
- Direct adjacent runtime helpers inspected where they determine OpenClaw behavior: `modules/sd_hijack_optimizations.py` SDPA backend normalization/status/set helpers and SDP/no-mem attention wrappers, `modules/sd_samplers_cfg_denoiser.py` denoiser `run_inner_model()` bridge into CUDA graphs, `modules/api/api.py` OpenClaw API route registration plus SDPA/CUDA-graph/precision-map handlers, precision-map construction/cache helpers, and `modules/processing.py` diagnostic capture call sites plus OpenClaw cache stat serialization.
- Focused reachability/duplication checks: grep fanout for OpenClaw routes and backend helpers, AST function listing over the checked files, exact function-body duplicate scan across OpenClaw and adjacent runtime files, and focused inspection of `test/test_openclaw_cuda_graphs.py`.

### Findings and fix decision
- No safe source deletion or deduplication was found in this bounded slice; this pass is a ledger-only checkpoint.
- `openclaw_cuda_graphs` private helpers have low direct fanout but are active internals of `run()` and API status/control. `_cache_key()` deliberately includes checkpoint identity, LoRA source signature, tensor/condition structure, current SDPA backend, and SEG graph state; removing any field would risk stale CUDA graph replay across runtime mutations.
- `_clone_static()`/`_copy_into_static()` and `processing._clone_cache_value()` look conceptually similar, but they serve different contracts: CUDA graph static storage copies live tensor inputs into capture buffers without Python object cache semantics, while processing cache cloning preserves img2img init-cache payloads. No shared helper was introduced.
- `_json_safe()` in diagnostics and API/FastAPI JSON encoding overlap only at a high level. Diagnostics must sanitize arbitrary processing/runtime objects before storing them on `Processed` and serving `/sdapi/v1/openclaw/generation-diagnostics`, so it was preserved.
- `sd_hijack_optimizations.attention_backend_status()` and `set_attention_backend()` are thin compatibility aliases over the newer SDPA backend helpers. They are not referenced by first-party code in this slice, but they are public runtime/API compatibility surfaces and should remain unless a later API deprecation pass removes them deliberately.
- `get_xformers_flash_attention_op()` is a compatibility stub returning `None` after the xformers path was removed/disabled. Its name is import-facing and harmless; removing it would be more likely to break old extensions than to simplify runtime behavior.
- `scaled_dot_product_no_mem_attention_forward()` and `sdp_no_mem_attnblock_forward()` duplicate the ordinary SDP path except for the forced `flash,math` backend override. This is a real policy fork used by the `sdp-no-mem` optimizer, not dead code.
- `Api.get_sdpa_backend()`/`set_sdpa_backend()`, `get_cuda_graphs()`/`set_cuda_graphs()`, `get_precision_map()`, and `get_openclaw_generation_diagnostics()` are queue-locked where needed or intentionally read-only. Precision-map building stays under `queue_lock` to avoid racing backend/model swaps.
- `build_precision_map()` has broad local helper surface, but it reports distinct model, VAE, dtype, TorchAO quantization, LoRA target, MHA target, and skip-reason fields. The cache key is intentionally conservative and includes selected MXFP8/NVFP4 coverage and LoRA state.
- Exact duplicate scan found only already-audited compatibility wrappers: FastAPI exception handlers in `modules/api/api.py`, abstract/placeholder methods (`SdOptimization.apply`, `StableDiffusionProcessing.sd_model`, `StableDiffusionProcessing.init`), and abstract properties/methods (`CFGDenoiser.inner_model`, `StableDiffusionProcessing.sample`). These were preserved.

### Static/dynamic audit map notes
- CUDA graph chain: `CFGDenoiser.run_inner_model()` calls `openclaw_cuda_graphs.run()`; `run()` gates on enabled state, mask/SEG bypass policy, cache size, CUDA availability, grad state, failed keys, and cache hit/miss before replaying or capturing a graph.
- Diagnostics chain: `processing.process_images_inner()` calls `openclaw_generation_diagnostics.before_sample()`/`after_sample()` around each sample; the after hook stores diagnostics on the processing object, appends history, updates the module-global last snapshot, and `Processed.js()` serializes the fields.
- API chain: `Api.__init__()` registers `/sdapi/v1/openclaw/sdpa-backend`, `/sdapi/v1/openclaw/cuda-graphs`, `/sdapi/v1/openclaw/generation-diagnostics`, `/sdapi/v1/openclaw/precision-map`, and legacy `/sdapi/v1/precision-map`; runtime-sensitive setters/builders execute under `self.queue_lock`.
- Compatibility surfaces to continue treating conservatively: OpenClaw `/sdapi/v1/openclaw/*` response field names, legacy `/sdapi/v1/precision-map`, SDPA backend aliases/choice labels, `attention_backend_*` aliases, `get_xformers_flash_attention_op()`, CUDA graph stat keys, bypass reason strings, diagnostic JSON keys, precision-map layer/summary schema, and TorchAO/MXFP8/NVFP4 skip-reason helper names.

### Validation log
- `PYTHONPYCACHEPREFIX=/tmp/gb10-a1111-pycompile-pass10 python3 -m py_compile modules/openclaw_cuda_graphs.py modules/openclaw_generation_diagnostics.py modules/sd_hijack_optimizations.py modules/sd_samplers_cfg_denoiser.py modules/api/api.py modules/processing.py` - passed.
- `python3 -m pytest -q test/test_openclaw_cuda_graphs.py test/test_openclaw_cache_invalidation.py` - failed during collection because the GB10 system Python environment does not have `torch` or `numpy` installed; no tests ran.
- `git diff --check` - passed.

### Next unchecked scope
- More slices are still needed. Recommended next slice: remaining lower-level model/runtime loader and precision helpers not yet audited function-by-function, especially `modules/sd_models.py`, `modules/sd_models_*.py`, `modules/sd_disable_initialization.py`, `modules/sd_vae*.py`, `modules/sd_unet.py`, `modules/lowvram.py`, `modules/devices.py`, `modules/model_quant.py` if present, and adjacent TorchAO/MXFP8/NVFP4/attention/runtime-cache helpers reached from those files.

## Pass 11 - lower-level model/runtime loader and precision helpers (2026-06-20)

### Checked scope
- `modules/sd_models.py`: checkpoint discovery/aliasing, metadata/state-dict reads, config selection handoff, model type detection, SDXL CLIP key remapping, model weight load/reload/unload, checkpoint/model caches, FP8/MXFP8/NVFP4 selection and application, TorchAO quantized Linear restore/send/trash helpers, alpha schedule overrides, token merging, and `SdModelData` lifecycle.
- `modules/sd_models_config.py`, `modules/sd_models_types.py`, and `modules/sd_models_xl.py`: config guessing, v-parameterization probe, near-checkpoint config override, SDXL/SGM compatibility monkeypatches, GeneralConditioner adapter helpers, and model typing-only surface.
- `modules/sd_disable_initialization.py`: shared replacement/restore context helper, initialization suppression, meta-device initialization, and meta-state-dict loading wrappers.
- `modules/sd_vae.py`, `modules/sd_vae_approx.py`, and `modules/sd_vae_taesd.py`: VAE discovery/resolution/cache/base-restore/reload paths, cheap VAE approximation, approximate decoder cache, TAESD encoder/decoder construction/download/cache paths.
- `modules/sd_unet.py`: dynamic UNet option discovery, activation/deactivation, current UNet bridge, and patched forward dispatch.
- `modules/lowvram.py`: low/med-vram enablement, module CPU/GPU shuttling hooks, first-stage encode/decode wrapping, and live-preview state query.
- `modules/devices.py`: backend selection, device strings, cache cleanup, TF32 enablement, dtype globals, conditional casts, manual autocast wrapper, autocast/without-autocast, NaN detection, warmup calculation, and fp16 forcing helper.
- Adjacent precision/runtime-cache helpers reached from these files: `modules/mxfp8_diagnostics.py`, `modules/api/api.py` precision-map callers, `modules/processing.py` VAE/cache stats and dtype infotext, `extensions-builtin/Lora/networks.py` MXFP8/NVFP4 LoRA merge paths, `modules/sd_hijack.py` UNet forward patch bridge, `modules/sd_samplers_common.py` VAE preview decode/encode paths, and option onchange hooks in `modules/initialize_util.py`/`modules/shared_options.py`.
- `modules/model_quant.py` was checked by path search and is not present in this checkout.

### Findings and fixes
- Removed the duplicate `DisableInitialization.replace()` override in `modules/sd_disable_initialization.py`. Its body was identical to `ReplaceHelper.replace()` and did not customize behavior, while `DisableInitialization`, `InitializeOnMeta`, and `LoadStateDictOnMeta` all share the same replacement stack contract through the base helper.
- No additional safe source deletion or abstraction was found in this bounded slice.
- The remaining exact duplicate bodies after the fix are context-manager `__exit__()` methods that call `self.restore()` and intentionally separate placeholder/hook methods (`do_nothing()`, `SdUnet.activate()`, `SdUnet.deactivate()`). Collapsing them would add indirection or change public subclass hook shape without reducing meaningful logic.
- `check_mxfp8()` and `check_nvfp4()` plus their region/coverage/skip/filter/application helpers are intentionally parallel. They use separate option names, config modules, tensor classes, cache modules, LoRA backup attributes, skip reason text, diagnostics fields, and reload policy checks. A shared selector/apply abstraction would risk blurring MXFP8/NVFP4-specific contracts and was not safe for this audit pass.
- `apply_mxfp8_weight_quantization()` and `apply_nvfp4_weight_quantization()` duplicate broad traversal/count/cache shape, but the quantization configs, validation functions, cache keys, model attributes, and downstream LoRA merge paths are distinct. Preserved as policy-parallel code.
- `sd_vae_approx.download_model()` and `sd_vae_taesd.download_model()` have similar download wrappers but emit different model names/locations and live in different decoder stacks. No useful shared helper was introduced.
- VAE base-cache helpers (`store_base_vae()`, `restore_base_vae()`, `delete_base_vae()`) and checkpoint/model caches in `sd_models.py` look related but carry different lifecycle state: VAE swapping restores a checkpoint's original first-stage weights, while SD model caches manage whole checkpoint state dicts and loaded model instances. Preserved.
- Low-vram hooks remain reachable from model send/reload, VAE reload, interrogation, sampler preview decode, and processing condition setup. The first-stage encode/decode wrappers are necessary because the first-stage model does not route those calls through `forward()`.
- `sd_unet.original_forward` is kept as a compatibility/debug bridge set by `sd_hijack.py` after patching LDM/SGM UNet forwards. The `SdUnet.activate()`/`deactivate()` no-op methods are public extension override hooks and were preserved.
- Device dtype helpers overlap conceptually with model precision helpers but operate at different layers: `modules/devices.py` owns global execution/autocast dtype policy, while `sd_models.py` owns checkpoint weight storage/quantization policy and reload safety.

### Static/dynamic audit map notes
- Model load chain: `select_checkpoint()` -> `get_checkpoint_state_dict()`/`read_state_dict()` -> `sd_models_config.find_checkpoint_config()` -> `DisableInitialization`/`InitializeOnMeta` construction -> `LoadStateDictOnMeta`/`load_model_weights()` -> dtype/quantization/VAE policy -> `send_model_to_device()` -> hijack and callbacks.
- Reload chain: option changes and API/UI checkpoint selection call `reload_model_weights()`; TorchAO MXFP8/NVFP4 reloads intentionally force fresh uncached loads when policy/mode/coverage can make generic state-dict reload unsafe.
- VAE chain: settings, checkpoint metadata, nearby files, XYZ grid, checkpoint metadata editor, and processing overrides call `reload_vae_weights()`; processing records VAE name/hash and sampler previews use approximate/TAESD decode paths.
- UNet chain: startup lists extension-provided UNet options, `sd_hijack.py` patches LDM/SGM UNet forwards through `create_unet_forward()`, and processing reapplies selected UNet after reloads.
- Low-vram chain: model send/reload and processing/interrogate/VAE paths call low-vram helpers to move tracked modules between CPU/GPU without treating those hook helpers as dead by grep fanout.
- Compatibility surfaces to continue treating conservatively: checkpoint alias typo `checkpoint_alisases`, `model_hash` import alias, `SdModelData` properties, `sd_unet.original_forward`, `SdUnet` hook method names, SDXL monkeypatch function names, VAE option string values (`Automatic`, `auto`, `None`), model/VAE cache globals, TorchAO quantization stats field names, MXFP8/NVFP4 skip reason strings, precision-map response fields, and device dtype globals.

### Validation log
- `PYTHONPYCACHEPREFIX=/tmp/gb10-a1111-pycompile-pass11 python3 -m py_compile modules/sd_models.py modules/sd_models_config.py modules/sd_models_types.py modules/sd_models_xl.py modules/sd_disable_initialization.py modules/sd_vae.py modules/sd_vae_approx.py modules/sd_vae_taesd.py modules/sd_unet.py modules/lowvram.py modules/devices.py modules/mxfp8_diagnostics.py extensions-builtin/Lora/networks.py modules/api/api.py modules/processing.py modules/sd_hijack.py modules/sd_samplers_common.py` - passed.
- `python3 -m pytest -q test/test_openclaw_device_dtypes.py test/test_openclaw_cache_invalidation.py test/test_openclaw_cuda_graphs.py` - failed during collection because the GB10 system Python environment does not have `torch` or `numpy` installed; no tests ran. Existing pytest config warning remains: `Unknown config option: base_url`.
- `git diff --check` - passed.

### Next unchecked scope
- More slices are still needed. Recommended next slice: sampler/runtime execution surfaces not fully audited function-by-function yet, especially `modules/sd_samplers*.py`, `modules/processing*.py` portions not already covered by cache/diagnostics/model-loader passes, `modules/rng.py`, `modules/masking.py`, sampler-specific script/API compatibility helpers, and adjacent optimization/runtime-cache helpers reached from sampler execution.

## Pass 12 - sampler/runtime execution surfaces redux (2026-06-20)

### Checked scope
- Sampler registry and compatibility helpers: `modules/sd_samplers.py`, sampler/scheduler infotext normalization, visible sampler maps, hires sampler/scheduler fallback helpers, and invalid sampler/scheduler autocorrection.
- Sampler runtime base and implementations: `modules/sd_samplers_common.py`, `modules/sd_samplers_kdiffusion.py`, `modules/sd_samplers_timesteps.py`, `modules/sd_samplers_timesteps_impl.py`, `modules/sd_samplers_lcm.py`, `modules/sd_samplers_extra.py`, and `modules/sd_samplers_cfg_denoiser.py`.
- Randomness and mask helpers: `modules/rng.py` and `modules/masking.py`, including subseed interpolation, resized seed noise insertion, per-batch RNG progression, crop-region variants, crop expansion, and masked fill.
- Processing sampler execution portions not fully covered by cache/diagnostics/model-loader passes: `create_random_tensors()`, decode/uint8 conversion helpers, seed fixing, txt2img first/hires pass sampling, img2img inpaint mask setup, latent mask resizing use, init latent/noise setup, and sampler `sample()`/`sample_img2img()` calls.
- Adjacent sampler/script/API compatibility callers checked by grep/static inspection: `modules/processing_scripts/sampler.py`, `modules/api/api.py`, `modules/img2img.py`, `scripts/img2imgalt.py`, `scripts/loopback.py`, `scripts/sd_upscale.py`, `scripts/xyz_grid.py`, `extensions/openclaw-multi-sampler/scripts/openclaw_multi_sampler.py`, and `extensions/openclaw-denoise-ramp/scripts/openclaw_denoise_ramp.py`.
- Focused duplicate/reachability checks: AST function listing across sampler/runtime files, exact function-body duplicate scan across `modules/sd_samplers*.py`, `modules/processing.py`, `modules/rng.py`, and `modules/masking.py`, and grep fanout for sampler compatibility aliases, RNG helpers, mask helpers, and image conversion helpers.

### Findings and fixes
- Deduplicated the hires-fix lowres tensor-to-PIL conversion path in `StableDiffusionProcessingTxt2Img.sample_hr_pass()`. The loop now uses the existing `samples_to_uint8_images(lowres_samples)` helper instead of repeating the same clamped CHW tensor -> HWC uint8 conversion already used by the main processing output path.
- No dead sampler construction/config helper was removed. `find_sampler_config()`, `create_sampler()`, `set_samplers()`, visible sampler helpers, infotext scheduler conversion, and invalid sampler autocorrection are active through UI/API/infotext/script paths.
- `modules.sd_samplers` re-exports `samples_to_image_grid` and `sample_to_image`, `modules.sd_samplers_timesteps` installs `sys.modules['modules.sd_samplers_compvis']`, and `VanillaStableDiffusionSampler` remains a deliberate extension compatibility alias. These were preserved.
- `modules/masking.get_crop_region()` duplicates most of `get_crop_region_v2()` by design but keeps the older blank-mask behavior for extension compatibility. Current img2img inpaint full-res code uses `get_crop_region_v2()` to avoid invalid blank-mask regions, so both helpers were preserved.
- RNG helpers are all reachable through processing, device monkeypatches, sampler noise hijacking, Brownian SDE deterministic sampling, or subseed/seed-resize behavior. `create_random_tensors(..., p=None)` keeps the legacy parameter shape even though this fork no longer uses `p` internally.
- K-diffusion, timestep, and LCM sampler methods still share lifecycle concepts but diverge on sigma/timestep schedule construction, img2img noise injection, callback parameters, solver-specific kwargs, and denoiser adapters. No safe shared sampler execution helper was introduced.
- The exact duplicate-body scan after the fix reports only intentional framework/API shapes: two CompVis timestep denoiser `__init__()` methods, abstract/placeholder sampler and processing methods, and the `sd_model` setter/abstract `init()` pass bodies. These were preserved.

### Static/dynamic audit map notes
- Sampler execution chain: `process_images_inner()` seeds `p.rng`, activates scripts/extra networks, builds conds, then calls `p.sample()`; txt2img/img2img create sampler instances through `sd_samplers.create_sampler()` and dispatch into `sample()` or `sample_img2img()`.
- Noise chain: `rng.ImageRNG` owns first/noise progression, subseed interpolation, seed resize insertion, and eta noise seed delta generator replacement; `TorchHijack.randn_like()` and torchsde Brownian patch route sampler noise back through this deterministic path where required.
- Mask chain: img2img converts UI/API masks with `create_binary_mask()`, uses `masking.get_crop_region_v2()`/`expand_crop_region()` for inpaint full-res, falls back to img2img on blank masks, fills masked image regions through `masking.fill()`, and blends latent masks in CFG and final img2img output paths.
- Compatibility surfaces to continue treating conservatively: sampler/scheduler names and aliases, infotext field names (`Sampler`, `Schedule type`, `Hires sampler`, `Hires schedule type`), `modules.sd_samplers_compvis` and `VanillaStableDiffusionSampler`, `samples_to_image_grid`/`sample_to_image` re-exports, `create_random_tensors()` signature, `get_crop_region()` blank-mask legacy behavior, CFG denoiser callback parameter ordering, script `process_before_every_sampling()` and mask-blend hooks, and OpenClaw multi-sampler/denoise-ramp sampler monkeypatch contracts.

### Validation log
- `PYTHONPYCACHEPREFIX=/tmp/gb10-a1111-pycompile-pass12 python3 -m py_compile modules/sd_samplers.py modules/sd_samplers_common.py modules/sd_samplers_compvis.py modules/sd_samplers_extra.py modules/sd_samplers_kdiffusion.py modules/sd_samplers_lcm.py modules/sd_samplers_timesteps.py modules/sd_samplers_timesteps_impl.py modules/sd_samplers_cfg_denoiser.py modules/processing.py modules/rng.py modules/masking.py modules/processing_scripts/sampler.py modules/api/api.py modules/img2img.py scripts/img2imgalt.py scripts/loopback.py scripts/sd_upscale.py scripts/xyz_grid.py extensions/openclaw-multi-sampler/scripts/openclaw_multi_sampler.py extensions/openclaw-denoise-ramp/scripts/openclaw_denoise_ramp.py` - passed.
- `python3 -m pytest -q test/test_rng.py test/test_cfg_denoiser_callbacks.py test/test_openclaw_device_dtypes.py test/test_openclaw_cache_invalidation.py` - failed during collection because the GB10 system Python environment does not have `torch` or `numpy` installed; no tests ran. Existing pytest config warning remains: `Unknown config option: base_url`.
- `git diff --check` - passed.

### Next unchecked scope
- More slices are still needed. Recommended next slice: remaining lower-level runtime and optimization surfaces outside sampler/model-loader coverage, especially `modules/cache.py`, `modules/hashes.py`, `modules/modelloader.py`, `modules/upscaler.py`, `modules/realesrgan_model.py` if present, `modules/gfpgan_model.py`, `modules/codeformer_model.py`, `modules/interrogate.py`, remaining `modules/textual_inversion/*` training/runtime helpers not already covered, and adjacent cache/hash/upscaler compatibility helpers reached from image generation and postprocessing.

## Pass 13 - cache/hash/upscaler/face/interrogate/textual-inversion helpers (2026-06-20)

### Checked scope
- Cache and hash helpers: `modules/cache.py` and `modules/hashes.py`, including diskcache creation/migration, file metadata cache invalidation, full SHA256/addnet safetensors hashing, legacy partial hash compatibility, and cache aliases.
- Model discovery/upscaler helpers: `modules/modelloader.py`, `modules/upscaler.py`, `modules/realesrgan_model.py`, and adjacent first-party/built-in upscaler call sites in LDSR, ScuNET, SwinIR, DAT, HAT, ESRGAN, API model listing, and shared item refresh helpers.
- Face restoration helpers: `modules/face_restoration.py`, `modules/face_restoration_utils.py`, `modules/gfpgan_model.py`, `modules/codeformer_model.py`, postprocessing GFPGAN/CodeFormer scripts, API face-restorer listing, generation face-restoration call path, and focused face-restorer tests.
- Interrogate helpers: `modules/interrogate.py`, shared interrogator initialization, UI/API/interrogate batch paths, loopback interrogate path, and postprocessing caption integration.
- Textual inversion runtime/training helpers: `modules/textual_inversion/autocrop.py`, `dataset.py`, `image_embedding.py`, `learn_schedule.py`, `saving_settings.py`, `textual_inversion.py`, and `ui.py`, plus adjacent hypernetwork training reuse, Lora embedded-TI load path, OpenClaw clear-cond-cache template listing, postprocessing focal crop, UI/API create/train embedding paths, and focused textual-inversion contract tests.
- Focused duplicate/reachability checks: `git grep` fanout for target helper names, AST function listing for the slice, exact nontrivial function-body duplicate scan across the target modules, and targeted line-number inspection where broad dumps appeared to show duplicate lines.

### Findings and fix decision
- No safe source deletion or deduplication was found in this bounded slice; this pass is a ledger-only checkpoint.
- Cache and hash helpers overlap around mtime/size validation, but they store different value schemas and public cache sections: generic `cached_data_for_file()` stores arbitrary `value` payloads, full hash caches store `sha256`, addnet hashes use a separate section, and partial hashes intentionally preserve the old collision-prone compatibility hash. Recent size-based invalidation behavior is covered by cache tests and was preserved.
- `modules.cache.dump_cache()` is intentionally a no-op compatibility function after the diskcache migration. It is still aliased by `modules.hashes` and called after cache writes, so removing it would risk old extension imports for no runtime benefit.
- `modelloader.load_models()`, `Upscaler.find_models()`, and individual upscaler constructors share model discovery concepts but serve layered contracts: generic path/download discovery, base-upscaler command-path forwarding, and backend-specific model filtering/URL/local-candidate handling. No safe shared helper was introduced.
- `modelloader.load_upscalers()` dynamic `_model.py` import and `Upscaler.__subclasses__()` scan remain necessary for extension/built-in upscaler discovery. The reverse duplicate-class filter protects module reloads and was preserved.
- `realesrgan_model.get_realesrgan_models(None)` is live through API and shared option listing even when no upscaler instance exists; its `UpscalerData(..., upscaler=None)` behavior was preserved.
- GFPGAN and CodeFormer setup/model classes have parallel shapes, but their model URLs, device choices, facexlib patching needs, inference signatures, postprocessing entry points, and global compatibility variables differ. A shared setup helper would add indirection without removing meaningful duplicate logic.
- `face_restoration_utils` conversion/helper functions are all reached through `CommonFaceRestoration.restore_with_helper()`. `send_model_to()` looks like a simple device helper but also moves cached facexlib detector/parser modules, so it was preserved.
- Interrogate BLIP/CLIP preprocessing is not duplicated elsewhere in the slice. `create_fake_fairscale()` is a legacy import shim required before BLIP import; category download/list helpers are wired to options/UI refresh and runtime interrogation.
- Textual inversion training helpers are live through API/UI or hypernetwork reuse. `validate_train_inputs()`, tensorboard helpers, `save_settings_to_file()`, `LearnRateScheduler`, and dataset loader classes are shared between embedding and hypernetwork training paths or directly used by embedding training.
- Textual inversion image-embedding helpers (`embedding_to_b64`, `embedding_from_b64`, XOR/style block helpers, overlay helpers, embedded PNG extraction) are reachable from embedding load/save-preview paths and preserve the embedded-image format; no safe simplification was made.
- `BatchLoaderRandom.__init__()` delegates only to `BatchLoader.__init__()` and `pin_memory()` is intentionally different for random latent sampling. The tiny subclass shape is part of DataLoader collate behavior and was preserved.
- A broad overlapping dump briefly appeared to show duplicate `if self.skipped_embeddings:` lines in `EmbeddingDatabase.load_textual_inversion_embeddings()`, but targeted `nl -ba` inspection of the committed source showed only one line. No source edit was made.
- Exact duplicate-body scan across the pass-13 target modules found no nontrivial duplicate function bodies.

### Static/dynamic audit map notes
- Cache/hash chain: checkpoint/Lora/hypernetwork/embedding metadata callers use `cache.cached_data_for_file()` or hash-specific caches; cache invalidation is mtime plus size-aware for current entries while legacy entries without size intentionally refresh or tolerate compatibility depending on helper.
- Upscaler chain: startup calls `modelloader.load_upscalers()`, dynamically imports `_model.py` modules, instantiates `Upscaler` subclasses, and publishes sorted `shared.sd_upscalers`; postprocessing and API paths select `UpscalerData` entries and call backend-specific `do_upscale()`/`load_model()`.
- Face restoration chain: startup setup registers CodeFormer/GFPGAN restorers into `shared.face_restorers`; generation uses `face_restoration.restore_faces()` based on `opts.face_restoration_model`; postprocessing GFPGAN/CodeFormer scripts call backend-specific globals for direct extras processing.
- Interrogate chain: `shared_init.initialize()` creates `shared.interrogator`; UI/API/postprocessing/loopback call `InterrogateModels.interrogate()`, which lazy-loads BLIP and CLIP, ranks configured category files, and unloads models according to options.
- Textual inversion chain: startup lists templates, `sd_hijack` owns `EmbeddingDatabase`, generation reloads embeddings unless disabled, API/UI create/train embedding call into `textual_inversion.py`, and hypernetwork training reuses dataset/scheduler/settings/tensorboard/loss helpers.
- Compatibility surfaces to continue treating conservatively: cache section names and entry schemas, `modules.cache.dump_cache`, hash title prefixes, `partial_hash_from_cache()` return behavior, `modelloader.load_file_from_url` re-export, upscaler class names/model option names, `shared.sd_upscalers` ordering, face-restorer names/globals/API schema, interrogate category filenames/options, textual inversion embedding file/image formats, training UI/API signatures, and hypernetwork reuse of textual-inversion helpers.

### Validation log
- `PYTHONPYCACHEPREFIX=/tmp/gb10-a1111-pycompile-pass13 python3 -m py_compile modules/cache.py modules/hashes.py modules/modelloader.py modules/upscaler.py modules/realesrgan_model.py modules/gfpgan_model.py modules/codeformer_model.py modules/face_restoration.py modules/face_restoration_utils.py modules/interrogate.py modules/textual_inversion/autocrop.py modules/textual_inversion/dataset.py modules/textual_inversion/image_embedding.py modules/textual_inversion/learn_schedule.py modules/textual_inversion/saving_settings.py modules/textual_inversion/textual_inversion.py modules/textual_inversion/ui.py extensions-builtin/LDSR/scripts/ldsr_model.py extensions-builtin/ScuNET/scripts/scunet_model.py extensions-builtin/SwinIR/scripts/swinir_model.py extensions-builtin/postprocessing-for-training/scripts/postprocessing_focal_crop.py scripts/postprocessing_gfpgan.py scripts/postprocessing_codeformer.py modules/api/api.py modules/shared_init.py modules/shared_items.py modules/processing.py modules/hypernetworks/hypernetwork.py` - passed.
- `python3 -m pytest -q test/test_openclaw_cache_invalidation.py test/test_face_restorers.py tests/test_textual_inversion_autocrop_contract.py tests/test_textual_inversion_preview_save_contract.py` - failed during collection because the GB10 system Python environment does not have `numpy` installed; no tests ran. Existing pytest config warning remains: `Unknown config option: base_url`.
- `git diff --check` - passed.

### Next unchecked scope
- More slices are still needed. Recommended next slice: remaining API/server/UI compatibility and utility surfaces not yet audited function-by-function, especially `modules/api/*.py` request/response helpers, `modules/server.py`, `modules/progress.py`, `modules/call_queue.py`, `modules/ui_loadsave.py`, `modules/ui_tempdir.py`, `modules/ui_extensions.py`, `modules/sysinfo.py`, `modules/util.py`, `modules/safe.py`, `modules/txt2img.py`, `modules/img2img.py`, and adjacent route/task/progress compatibility helpers reached from API and headless UI startup.

## Pass 14 - API/server/UI utility compatibility surfaces (2026-06-20)

### Checked scope
- API route and request/response helpers: `modules/api/api.py` and `modules/api/models.py`, including precision-map diagnostics helpers, script-argument/default helpers, infotext application, sampler/scheduler validation, base64/URL image decode and encode helpers, extras request normalization, middleware, OpenClaw runtime-control endpoints, txt2img/img2img API task wrapping, progress/interrogate/options/model-listing/training/memory/extension routes, dynamic Pydantic model generation, and API response schema classes.
- Progress and queue helpers: `modules/progress.py` and `modules/call_queue.py`, including task id creation, pending/current/finished task bookkeeping, internal progress/live-preview responses, restore-progress result caching, FIFO queue wrappers, UI exception/result wrapping, and memory/stat reporting.
- UI default/temp/extension helpers: `modules/ui_loadsave.py`, `modules/ui_tempdir.py`, and `modules/ui_extensions.py`, including UI config persistence, component/default mapping, legacy Gradio temp-file registration, temp-dir checks, extension enable/update/install/backup/restore tables, available-extension filtering, URL normalization, and config-state UI callbacks.
- Utility/security/sysinfo helpers: `modules/sysinfo.py`, `modules/util.py`, and `modules/safe.py`, including sysinfo checksum/environment/package/git collection, filesystem walking/listing/cache helpers, topological sort, open-folder/download/hash helpers, restricted unpickle validation, and `torch.load` compatibility wrapping.
- UI generation entry points and adjacent route/task compatibility helpers: `modules/txt2img.py`, `modules/img2img.py`, plus targeted call-site checks in `modules/initialize_util.py`, `modules/ui.py`, `modules/ui_component_patches.py`, `modules/ui_common.py`, `modules/ui_settings.py`, `modules/ui_checkpoint_merger.py`, `modules/ui_postprocessing.py`, and `modules/infotext_utils.py`.
- `modules/server.py` was included in the requested scope but does not exist in this checkout.
- Focused duplicate/reachability checks: AST function/class listing for the slice, exact nontrivial function-body duplicate scan across target modules, grep fanout for route/task/temp/UI utility names, targeted zero-reference review for callback-bound helpers, and line-number verification for a suspected duplicate assignment artifact in `build_precision_map()`.

### Findings and fix decision
- No safe source deletion or deduplication was found in this bounded slice; this pass is a ledger-only checkpoint.
- API helpers with no external grep references are bound through FastAPI route registration, `Api.__init__()` setup, or internal method calls. Precision-map helper functions are intentionally private to `build_precision_map()` and the `/sdapi/v1/openclaw/precision-map` plus `/sdapi/v1/precision-map` compatibility routes.
- `ScriptArgsList` is a tiny list subclass used to attach `openclaw_script_args_to_overrides` during API always-on script processing. It was preserved because plain lists cannot carry that metadata cleanly and downstream processing reads the attribute.
- `script_default_ui_values()` is covered by focused tests and remains needed for finalized script controls during headless/API startup. `api_infotext_value_for_field()` preserves bool/string/generic-update coercion for API infotext paste compatibility.
- txt2img/img2img API paths and UI paths duplicate some generation setup concepts, but they differ on request models, base64 image decoding/encoding, send/save image flags, task ids, script arg handling, timings, UI gallery return values, and progress restoration contracts. No safe shared generation wrapper was introduced.
- `/internal/progress` in `modules/progress.py` and `/sdapi/v1/progress` in `modules/api/api.py` are separate compatibility surfaces: the former is the UI restore/live-preview task channel, while the latter is the public API progress schema. Their similar progress math was preserved.
- `modules/call_queue.py` wrappers intentionally layer FIFO locking, UI state cleanup, exception-to-HTML conversion, optional memory stats, and GPU task bookkeeping. The small wrappers are not dead; they are imported by UI modules and option onchange hooks.
- `modules/ui_tempdir.cleanup_tmpdr()` is unreferenced in this tree and appears typo-named, but it is a public helper in a legacy Gradio tempdir compatibility module. Because no call site proves a safe replacement/removal and external scripts may import public module helpers, it was preserved.
- `modules/ui_extensions.py` functions with no external grep references are callback-bound inside `create_ui()` or called by other helpers in the same module. Extension URL normalization and available-extension filtering are live through install/search/tag UI callbacks and were preserved.
- `modules/util.py` and `modules/safe.py` contain public compatibility helpers/re-exports used broadly or monkeypatched globally (`modules.modelloader.load_file_from_url`, `torch.load = safe.load`). Apparent local-only helpers such as `MassFileListerCachedDir`, `RestrictedUnpickler`, and `compare_sha256()` are constructor/internal helper pieces for active public APIs.
- A broad large-file read appeared to show a duplicate `bias_info = _precision_tensor_info(bias)` assignment in `build_precision_map()`, but targeted `nl -ba modules/api/api.py` inspection showed only one committed line. No source edit was made.
- Exact duplicate-body scan across the pass-14 target modules found no nontrivial duplicate function bodies.

### Static/dynamic audit map notes
- API startup chain: `initialize_util`/web API startup constructs `Api`, installs middleware, registers `/sdapi/v1/*` and OpenClaw compatibility routes through `add_api_route()`, initializes script runners/UI defaults when needed, and applies OpenClaw runtime defaults from environment.
- API generation chain: txt2img/img2img API methods create task ids, apply infotext, resolve selectable/always-on scripts, normalize sampler/scheduler fields, enqueue/start/finish progress tasks under `queue_lock`, run scripts or `process_images()`, encode images conditionally, and include OpenClaw cache/timing diagnostics in response info.
- UI generation chain: Gradio callbacks in `modules/ui.py` wrap `modules.txt2img.txt2img`/`txt2img_upscale` and `modules.img2img.img2img` through `call_queue.wrap_ui_gpu_call()`, which updates `modules.progress` task/result state for restore-progress buttons.
- Temp-file chain: `modules/ui_component_patches.py` installs `ui_tempdir.save_pil_to_file()` over Gradio's PIL temp writer; `register_tmp_file()` keeps legacy Gradio temp file/dir allow-lists current; `infotext_utils` uses `check_tmp_file()` to decide whether UI temp image paths are acceptable.
- Extension UI chain: extension management callbacks are all created inside `ui_extensions.create_ui()`; install/update/backup/restore helpers mutate config or extension state only after `check_access()` and usually request restart/stop through `modules.restart`.
- Compatibility surfaces to continue treating conservatively: public `/sdapi/v1/*` schemas and route names, `/sdapi/v1/precision-map` alias, `/internal/progress`/`/internal/pending-tasks`, task id string shape `task(type-ID)`, `ScriptArgsList.openclaw_script_args_to_overrides`, `force_task_id`, `send_images`/`save_images`, `include_init_images`, script infotext/default handling, UI restore-progress tuple shapes, Gradio temp-file attributes `temp_file_sets`/`temp_dirs`, UI defaults config key format, extension-management JS callback contracts, `modules.modelloader.load_file_from_url` re-export behavior, and global `torch.load` wrapping by `modules.safe`.

### Validation log
- `PYTHONPYCACHEPREFIX=/tmp/gb10-a1111-pycompile-pass14 python3 -m py_compile modules/api/api.py modules/api/models.py modules/progress.py modules/call_queue.py modules/ui_loadsave.py modules/ui_tempdir.py modules/ui_extensions.py modules/sysinfo.py modules/util.py modules/safe.py modules/txt2img.py modules/img2img.py modules/initialize_util.py modules/ui.py modules/ui_component_patches.py modules/ui_common.py modules/ui_settings.py modules/ui_checkpoint_merger.py modules/ui_postprocessing.py modules/infotext_utils.py` - passed.
- `python3 -m pytest -q test/test_api_script_defaults.py tests/test_ui_loadsave_contract.py tests/test_ui_extensions_contract.py tests/test_api_progress_contract.py test/test_postprocessing_api_defaults.py` - passed: 13 passed, with the existing pytest config warning `Unknown config option: base_url`.
- Initial focused pytest command used the wrong path `tests/test_postprocessing_api_defaults.py` and failed with `ERROR: file or directory not found`; rerun with the existing `test/test_postprocessing_api_defaults.py` path passed.
- `git diff --check` - passed.

### Next unchecked scope
- More slices are still needed. Recommended next slice: remaining UI/common/output/postprocessing path utilities not yet audited function-by-function, especially `modules/ui_common.py`, `modules/ui_components.py`, `modules/ui_toprow.py`, `modules/ui_settings.py`, `modules/ui_postprocessing.py`, `modules/ui_checkpoint_merger.py`, `modules/ui_extra_networks*.py`, `modules/postprocessing.py`, `modules/images.py` portions outside prior save/cache passes, and adjacent output/gallery/download compatibility helpers reached from UI/API postprocessing.

## Pass 15 - UI/common/output/postprocessing utilities (2026-06-20)

### Checked scope
- UI output/common helpers: `modules/ui_common.py`, including generation-info refresh, plaintext HTML conversion, save/log CSV handling, output panel construction, gallery save/download/open-folder callbacks, send-to-tab paste bindings, refresh-button helper, and modal dialog setup.
- Form/toprow/settings/checkpoint UI helpers: `modules/ui_components.py`, `modules/ui_toprow.py`, `modules/ui_settings.py`, `modules/ui_postprocessing.py`, and `modules/ui_checkpoint_merger.py`, including Gradio form wrapper subclasses, `InputAccordion`, prompt/toprow construction, settings component/save callbacks, extras UI task wiring, and checkpoint merger metadata/model callbacks.
- Extra networks UI helpers: `modules/ui_extra_networks.py`, `modules/ui_extra_networks_checkpoints.py`, `modules/ui_extra_networks_checkpoints_user_metadata.py`, `modules/ui_extra_networks_hypernets.py`, `modules/ui_extra_networks_textual_inversion.py`, and `modules/ui_extra_networks_user_metadata.py`, including preview extension filtering, tree/card HTML generation, FastAPI thumbnail/metadata/card routes, extra-network page registration/order, preview save compatibility helpers, page subclasses, and user metadata editors.
- Postprocessing/output helpers: `modules/postprocessing.py` and remaining relevant portions of `modules/images.py`, including extras image enumeration, caption merge/write behavior, postprocessing script runner invocation, legacy API `run_extras()`, grid/tiling/annotation helpers, filename generation tokens, image save/geninfo/read helpers, prompt-image upload parsing, and transparency/EXIF normalization.
- Adjacent compatibility call sites and tests: UI/API extras routes, `modules/extras.py`, `modules/img2img.py`, extra-network JavaScript/HTML route consumers, save serialization tests, postprocessing API/default tests, extra-network path/metadata tests, image save tests, and upscaler tiling tests.
- Focused duplicate/reachability checks: AST function/class listing for every target module, `rg` fanout for each target helper name across Python/JS/HTML/tests, exact nontrivial function-body duplicate scan across target modules, targeted line-number reads for suspected public/dynamic helpers, and focused review of duplicate-looking path-parent helpers.

### Findings and fix decision
- No safe source deletion or deduplication was found in this bounded slice; this pass is a ledger-only checkpoint.
- `ui_common.save_files()` and `images.save_image()` overlap around filename/geninfo save behavior, but `save_files()` is a gallery/download/log orchestration callback that decodes already-rendered UI gallery entries, preserves selected-gallery indices, writes the save log CSV, and optionally packages a zip. It is covered by save serialization contract tests and was preserved.
- `OutputPanel` fields and hidden buttons are callback-bound through Gradio/JavaScript (`selected_gallery_index()`, send-to-tab paste bindings, restore/download/open-folder controls). Several fields have low static fanout by design; removing them would break UI callback contracts.
- `ui_components` wrapper classes share tiny `get_block_name()` implementations, but each maps a distinct Gradio component type into form-compatible CSS/block behavior. The duplication is deliberate and clearer than a factory/metaclass for public UI component classes used by scripts/extensions.
- `InputAccordion` combines a hidden checkbox with a rendered accordion and script-unload reset state. Its `extra()`, context-manager, and reset helpers are externally used by postprocessing/training scripts or callback lifecycle and were preserved.
- `ui_toprow.Toprow` methods are split by classic/compact rendering order. Methods with only intra-class call sites are Gradio construction steps or later consumed by `modules/ui.py`/extras output panel layout; no dead methods were found.
- `ui_settings` local closures (`fun`, `call_func_and_return_text`, `reload_scripts`, `check_file`, `calculate_all_checkpoint_hash_fn`, quicksettings handlers) are callback-bound inside `create_ui()`/`add_functionality()`. `search()` is a public method but currently not wired in the visible slice; because settings UI internals and extension/UI reload code may bind it dynamically, and it is harmless, it was preserved.
- `ui_checkpoint_merger.modelmerger()` and `update_interp_description()` are live Gradio callbacks. Their wrapping of `extras.run_modelmerger()` preserves dropdown refresh/error behavior and checkpoint UI compatibility.
- Extra-network route helpers (`fetch_file`, `fetch_cover_images`, `get_metadata`, `get_single_card`) have little Python fanout because they are registered as FastAPI routes and consumed by JS/HTML. Their path/extension checks, cover-image metadata handling, and single-card refresh behavior are covered or contract-checked and were preserved.
- `ExtraNetworksPage` base methods that look empty or abstract (`refresh`, `create_item`, `list_items`, `allowed_directories_for_previews`, `create_user_metadata_editor`) are extension override hooks. Page subclasses for textual inversion, hypernetworks, checkpoints, and external Lora pages rely on that shape.
- `ui_extra_networks.path_is_parent()` duplicates the same concept as `sd_models.path_is_parent()`, but both are module-local public helpers with independent tests/call sites and avoid a dependency between checkpoint metadata and extra-network UI routing. No shared helper was introduced.
- Extra-network preview-save exists in two forms: `ui_extra_networks.setup_ui().save_preview()` for older direct preview-save compatibility and `UserMetadataEditor.save_preview()` for the current metadata editor. They differ in target state/update outputs and lister/card refresh behavior; both callback paths were preserved.
- `postprocessing.run_postprocessing_webui()` is a thin task-id adapter for `call_queue.wrap_ui_gpu_call()` and `run_extras()` is the legacy API adapter that maps old extras request fields into the postprocessing script runner. Both are covered by focused tests and were preserved.
- `images.py` contains many broad public helpers used by processing, postprocessing, img2img upload parsing, extras, callbacks, filename patterns, and tests. Remaining save/read/cache-adjacent helpers outside previous save/cache passes did not reveal safe dead code. Filename token helpers are reachable through `FilenameGenerator.replacements` dynamic pattern names even when static grep is sparse.
- Exact duplicate-body scan across pass-15 target modules found no nontrivial duplicate function bodies.

### Static/dynamic audit map notes
- UI output chain: `modules/ui.py`/`ui_postprocessing.py` construct `OutputPanel`; hidden Gradio buttons and JS feed gallery index/state into `update_generation_info()`, `save_files()`, open-folder, send-to-tab, and extras output callbacks.
- Extras chain: extras UI calls `postprocessing.run_postprocessing_webui()` through the GPU queue; API extras calls `postprocessing.run_extras()`; both converge on `run_postprocessing()`, which enumerates input modes, runs postprocessing scripts, saves images/captions through `modules.images`, and returns UI/API compatible output tuples.
- Extra networks chain: startup registers default/extension pages, `add_pages_to_demo()` installs thumbnail/metadata/card routes, `create_ui()` builds per-tab HTML/editor controls, JS refreshes single cards through `get_single_card()`, and preview saves update card HTML plus the lister cache.
- Image output chain: generation/postprocessing/extras/img2img call `save_image()`, `save_image_with_geninfo()`, `read_info_from_image()`, `image_data()`, grid/tiling helpers, and filename-token expansion through direct calls or dynamic filename patterns/callbacks.
- Compatibility surfaces to continue treating conservatively: Gradio component subclasses and `get_block_name()` values, `InputAccordion` checkbox/accordion id shape, hidden button elem IDs, output panel tuple shapes, save/download file return shape, CSV log field order, `/sd_extra_networks/*` route names/query parameters, extra-network item dict keys and HTML template variables, preview filename conventions (`.preview.{samples_format}` vs `.{samples_format}`), user metadata JSON sidecars, `run_extras()` old API signature, `run_postprocessing_webui(id_task, ...)`, image filename pattern token names, and public image geninfo read/write behavior.

### Validation log
- `PYTHONPYCACHEPREFIX=/tmp/gb10-a1111-pycompile-pass15 python3 -m py_compile modules/ui_common.py modules/ui_components.py modules/ui_toprow.py modules/ui_settings.py modules/ui_postprocessing.py modules/ui_checkpoint_merger.py modules/ui_extra_networks.py modules/ui_extra_networks_checkpoints.py modules/ui_extra_networks_checkpoints_user_metadata.py modules/ui_extra_networks_hypernets.py modules/ui_extra_networks_textual_inversion.py modules/ui_extra_networks_user_metadata.py modules/postprocessing.py modules/images.py modules/extras.py modules/img2img.py modules/ui.py` - passed.
- `python3 -m pytest -q tests/test_save_serialization_contract.py tests/test_extra_networks_path_contract.py tests/test_extra_networks_metadata_contract.py tests/test_postprocessing_caption_contract.py test/test_postprocessing_api_defaults.py test/test_images_save.py tests/test_upscaler_tiling_contract.py` - passed: 22 passed, 1 skipped, with the existing pytest config warning `Unknown config option: base_url`.
- `git diff --check` - passed.

### Next unchecked scope
- More slices are still needed. Recommended next slice: script/postprocessing extension framework and always-on script compatibility surfaces not yet audited function-by-function, especially `modules/scripts.py`, `modules/scripts_postprocessing.py`, `modules/processing_scripts/*.py`, built-in postprocessing scripts under `scripts/postprocessing_*.py` and `extensions-builtin/postprocessing-for-training/scripts/*.py`, and adjacent UI/API script argument compatibility helpers reached from txt2img/img2img/extras.

## Pass 16 - script/postprocessing extension framework and always-on script compatibility surfaces (2026-06-20)

### Checked scope
- Script framework and runner surfaces: `modules/scripts.py`, including `Script` hook methods, callback argument wrapper classes, script discovery/dependency ordering, extension script loading, selectable/always-on runner UI construction, callback ordering, script timing collection, body-only reload, named-argument mutation helper, and txt2img/img2img runner globals.
- Postprocessing framework surfaces: `modules/scripts_postprocessing.py`, including `PostprocessedImage`, shared target-size/caption/extra-image state, suffix/copy helpers, `ScriptPostprocessing` extension hooks, element-id compatibility helpers, postprocessing script ordering/filtering, UI/default argument construction, multi-image postprocessing loop, and image-change callbacks.
- Main-UI postprocessing bridge: `modules/scripts_auto_postprocessing.py`, including `ScriptPostprocessingForMainUI` and automatic always-on script data creation from `postprocessing_enable_in_main_ui`.
- Built-in always-on processing scripts: `modules/processing_scripts/seed.py`, `sampler.py`, `refiner.py`, and `comments.py`, including seed/subseed reuse JS wiring, sampler/scheduler fields, refiner checkpoint infotext mapping, prompt-comment stripping, and token-counter callback integration.
- Built-in postprocessing scripts: `scripts/postprocessing_upscale.py`, `postprocessing_gfpgan.py`, and `postprocessing_codeformer.py`, including upscale cache/size-limit helpers, two-upscaler blend behavior, simple main-UI upscale adapter, face-restoration visibility blending, and postprocessing info fields.
- Built-in training postprocessing extension scripts: `extensions-builtin/postprocessing-for-training/scripts/postprocessing_caption.py`, `postprocessing_create_flipped_copies.py`, `postprocessing_focal_crop.py`, `postprocessing_split_oversized.py`, and `postprocessing_autosized_crop.py`.
- Adjacent compatibility call sites and tests: `modules/txt2img.py`, `modules/img2img.py`, `modules/processing.py`, `modules/api/api.py`, `modules/postprocessing.py`, `modules/shared_items.py`, `modules/ui.py`, `modules/ui_postprocessing.py`, `modules/ui_common.py`, `modules/ui_settings.py`, `modules/ui_html_extensions.py`, `modules/script_callbacks.py`, first-party/built-in extensions that override script hooks, and focused script/postprocessing API tests.
- Focused duplicate/reachability checks: AST function/class listing for all target modules, exact nontrivial function-body duplicate scan across the pass-16 target files, grep fanout for low-reference helper names and compatibility aliases, and targeted inspection of dynamic Gradio/FastAPI/API/script-runner paths where static references are intentionally sparse.

### Findings and fix decision
- No safe source deletion or deduplication was found in this bounded slice; this pass is a ledger-only checkpoint.
- `Script` base hook methods with pass bodies are the public extension ABI. Several built-in and installed extensions override `before_process`, `process_batch`, `postprocess_batch`, `on_mask_blend`, `post_sample`, `postprocess_maskoverlay`, and related hooks, so the pass-body methods and runner dispatch methods were preserved.
- Callback wrapper classes (`MaskBlendArgs`, `PostSampleArgs`, `PostprocessImageArgs`, `PostProcessMaskOverlayArgs`, `PostprocessBatchListArgs`) are active compatibility payloads passed from sampler/processing paths to script hooks. Their small constructors are intentionally simple and were preserved.
- `setup_scrips()` is typo-named but live through `StableDiffusionProcessing.setup_scripts()`. Renaming or adding a cosmetic alias was avoided because the existing misspelled method is part of the current internal/public compatibility surface.
- `reload_scripts = load_scripts`, `reload_script_body_only()`, `basedir()`, `ScriptFile`, `ScriptClassData`, and `list_files_with_name()` all have active UI, extension, or compatibility callers. They were preserved.
- `ScriptRunner._script_args_for()` and `openclaw_script_args_to_overrides` are required by API always-on script requests that pass shorter explicit argument lists than the backing default script-args vector. This compatibility path is covered by prior API/default tests and was preserved.
- Script runner dispatch methods are repetitive by hook name, but each carries distinct callback category/sort behavior, error context, hook method call, kwargs shape, and timing key. A generic dispatcher would add risk to extension hooks without deleting meaningful code.
- `ScriptPostprocessing` hook methods, `elem_id()`/`elem_id_suffix()`, `tab_name`, `extra_only`, `main_ui_only`, and `order` are extension-facing contracts. The element-id suffix helper is especially needed for the main-UI bridge to avoid txt2img/img2img ID collisions.
- `PostprocessedImage.create_copy()` and `get_suffix()` are live through focal-crop/split/flip postprocessing and output filename suffix handling. Shared target size/caption state is intentionally carried across extra images.
- `scripts_in_preferred_order()` and `create_args_for_run()` are live through extras UI/API paths and focused tests. Their order/filter/default behavior preserves postprocessing operation order, disabled extras scripts, and API default values.
- `ScriptPostprocessingForMainUI` duplicates some extras script execution shape by design, but it adapts postprocessing scripts into always-on txt2img/img2img hooks with tab-specific IDs and generation-param propagation. It was preserved.
- Built-in processing scripts (`Seed`, `Sampler`, `Refiner`, `Comments`) are callback/UI-bound and provide canonical API/infotext fields. `connect_reuse_seed()` and `before_token_counter()` have dynamic Gradio/script-callback reachability and were preserved.
- `scripts/postprocessing_upscale.py` contains typo-preserved helper `limit_size_by_one_dimention()` and no-op-looking assignments such as `upscaler_1_name = upscaler_1_name`; these are harmless but not worth source churn. The full/simple upscale classes share code intentionally through inheritance, with the simple variant main-UI-only.
- GFPGAN and CodeFormer postprocessing scripts have parallel visibility-blend logic, but each calls different backend globals with different required parameters and records different infotext keys. No shared helper was introduced.
- Training postprocessing scripts are small extension plugins. Local helpers (`split_pic`, `center_crop`, `multicrop_pic`) are script-local and used by their owning plugins; extracting them would create coupling without reducing duplicated bodies.
- Exact duplicate-body scan across the pass-16 target modules found no nontrivial duplicate function bodies.

### Static/dynamic audit map notes
- Script load chain: `load_scripts()` discovers root, extension, and built-in processing scripts; applies extension metadata dependency ordering; dynamically imports script modules; registers subclasses of `Script` and `ScriptPostprocessing`; and initializes txt2img/img2img/postprocessing runners.
- UI script chain: `modules/ui.py` initializes txt2img/img2img runners, creates built-in/always-on/selectable controls, records API metadata/defaults, installs component callbacks, and routes generation callbacks through `modules/txt2img.py` and `modules/img2img.py`.
- API script chain: `modules/api/api.py` initializes default script args from runner metadata, applies infotext/script/default overrides, handles selectable and always-on script requests, preserves OpenClaw shortened-args overrides, then runs selectable scripts or normal processing with script setup/timing.
- Processing hook chain: `StableDiffusionProcessing` calls `setup_scrips()` after scripts and args are assigned; processing/sampler code dispatches before/process/batch/post/sample/mask/composite hooks through `ScriptRunner`, passing compatibility payload classes where needed.
- Postprocessing chain: extras UI/API create postprocessing args through `scripts_postproc.create_args_for_run()`, call `postprocessing.run_postprocessing()`, instantiate `PostprocessedImage`, run all enabled postprocessing scripts in first-pass/process order, propagate extra images/captions/info, and save output with suffixes.
- Main-UI postprocessing chain: settings choose postprocessing script names for txt2img/img2img; `scripts_auto_postprocessing.create_auto_preprocessing_script_data()` wraps those postprocessing scripts as always-on `Script` instances; the bridge calls `ScriptPostprocessing.process()` during `postprocess_image` and propagates image/info back into generation results.
- Compatibility surfaces to continue treating conservatively: script hook method names/signatures, callback payload class fields, `AlwaysVisible`, `ScriptBuiltinUI`, `setup_scrips` spelling, script arg vector layout and index 0 selectable-script convention, `openclaw_script_args_to_overrides`, `ScriptInfo` arg metadata fields, script dependency metadata keys (`Requires`, `Before`, `After`), `reload_scripts` alias, `basedir()`, `list_files_with_name()`, postprocessing script names/order/filter options, `extra_only`/`main_ui_only`, `elem_id_suffix()` output, `PostprocessedImage` fields and suffix behavior, and built-in script UI elem IDs/infotext field names.

### Validation log
- `PYTHONPYCACHEPREFIX=/tmp/gb10-a1111-pycompile-pass16 python3 -m py_compile modules/scripts.py modules/scripts_postprocessing.py modules/scripts_auto_postprocessing.py modules/processing_scripts/seed.py modules/processing_scripts/sampler.py modules/processing_scripts/refiner.py modules/processing_scripts/comments.py scripts/postprocessing_gfpgan.py scripts/postprocessing_codeformer.py scripts/postprocessing_upscale.py extensions-builtin/postprocessing-for-training/scripts/postprocessing_caption.py extensions-builtin/postprocessing-for-training/scripts/postprocessing_create_flipped_copies.py extensions-builtin/postprocessing-for-training/scripts/postprocessing_focal_crop.py extensions-builtin/postprocessing-for-training/scripts/postprocessing_split_oversized.py extensions-builtin/postprocessing-for-training/scripts/postprocessing_autosized_crop.py modules/txt2img.py modules/img2img.py modules/processing.py modules/api/api.py modules/postprocessing.py modules/shared_items.py modules/ui.py modules/ui_postprocessing.py modules/ui_common.py modules/ui_settings.py modules/ui_html_extensions.py modules/script_callbacks.py` - passed.
- `python3 -m pytest -q test/test_postprocessing_script_args.py test/test_postprocessing_api_defaults.py test/test_api_script_defaults.py test/test_img2img.py tests/test_postprocessing_caption_contract.py` - failed because `test/test_img2img.py` requires the unavailable `base_url` fixture/server in this headless environment; before that fixture failure, 15 tests passed and the existing pytest config warning `Unknown config option: base_url` appeared.
- `python3 -m pytest -q test/test_postprocessing_script_args.py test/test_postprocessing_api_defaults.py test/test_api_script_defaults.py tests/test_postprocessing_caption_contract.py` - passed: 15 passed, with the existing pytest config warning `Unknown config option: base_url`.
- `git diff --check` - passed.

### Next unchecked scope
- More slices are still needed. Recommended next slice: extension management/loading and callback infrastructure not yet audited function-by-function, especially `modules/extensions.py`, `modules/extensions_metadata.py`, `modules/script_callbacks.py`, `modules/script_loading.py`, `modules/scripts_auto_postprocessing.py` only as needed for callback-loader overlap, `modules/ui_extensions.py` loader-adjacent portions not already covered, and adjacent extension metadata/config-state tests.


## Pass 17 - extension management/loading and callback infrastructure (2026-06-20)

### Checked scope
- Extension discovery and metadata surfaces: `modules/extensions.py`, including `active()`, `CallbackOrderInfo`, `ExtensionMetadata`, extension canonical-name handling, metadata list parsing, script requirement resolution, callback order instructions, `Extension` git/status/cache helpers, extension file listing, extension enumeration, duplicate canonical-name protection, requirement checks, and `find_extension()` path resolution.
- Callback infrastructure: `modules/script_callbacks.py`, including all callback parameter payload classes, registration, extension-derived callback naming, duplicate callback-name disambiguation, metadata/user callback sorting, cached ordered-callback lookup, callback enumeration/clearing, every callback dispatch wrapper, and removal helpers.
- Script loading/preload bridge: `modules/script_loading.py`, including dynamic module loading, `loaded_scripts` tracking, and extension `preload.py` execution through command-line parser setup.
- Main-UI postprocessing bridge overlap: `modules/scripts_auto_postprocessing.py`, revisited only for loader/callback overlap with pass 16, including `ScriptPostprocessingForMainUI` and automatic postprocessing script data creation.
- Loader-adjacent extension UI/config-state helpers in `modules/ui_extensions.py`, especially installed-extension table/status rendering, update/apply/install callbacks, config backup/restore/status table helpers, available-extension URL normalization/filter/install helpers, git metadata preload thread, and `create_ui()` callback wiring.
- Adjacent tests and dynamic call sites: `tests/test_extensions_metadata_contract.py`, `tests/test_ui_extensions_contract.py`, extension callback users under installed first-party extensions, `modules/scripts.py`, `modules/shared_cmd_options.py`, `modules/shared_items.py`, `modules/config_states.py`, `modules/api/api.py`, `modules/sysinfo.py`, and JavaScript extension installation hooks.
- Focused duplicate/reachability checks: AST function/class listing for target files, exact nontrivial function-body duplicate scan across the pass-17 target modules, `rg` fanout for low-reference helpers and callback APIs, targeted inspection of callback-order metadata and Gradio/JS-bound UI helpers, and verification that `modules/extensions_metadata.py` does not exist in this checkout because extension metadata is implemented inside `modules/extensions.py`.

### Findings and fix decision
- No safe source deletion or deduplication was found in this bounded slice; this pass is a ledger-only checkpoint.
- There is no separate `modules/extensions_metadata.py` file in this checkout. The metadata implementation is `ExtensionMetadata` inside `modules/extensions.py`, with focused contract coverage in `tests/test_extensions_metadata_contract.py`.
- `ExtensionMetadata.parse_list()`, `get_script_requirements()`, and `list_callback_order_instructions()` are live metadata ABI for extension dependency/order files. Script ordering in `modules/scripts.py` and callback ordering in `modules/script_callbacks.py` both consume this metadata, including fallback from metadata alternatives separated by `|`.
- `Extension.to_dict()`/`from_dict()` and `read_info_from_repo()` are small cache shims, but they preserve the `extensions-git` cache schema and avoid repeated GitPython access while extension tables, API routes, sysinfo, and config-state helpers read extension status. They were preserved.
- `Extension.do_read_info_from_repo()`, `check_updates()`, and `fetch_and_reset_hard()` have similar GitPython shapes but different semantics: cached status read, dry-run update check, and destructive update/reset. No shared helper was introduced.
- `Extension.list_files()` is live through script discovery in `modules/scripts.py`; it intentionally returns `ScriptFile` objects keyed by extension root and subdir. `find_extension()` is live through callback registration to map callback source files back to extension canonical names.
- `remove_current_script_callbacks()` has no static call site in this tree, but it is a public extension lifecycle helper in `modules.script_callbacks`. Because extension code can call it dynamically during reload/unload and the function removes both raw and ordered callback entries, it was preserved.
- `remove_callbacks_for_function()` is actively used by installed guidance extensions to unregister dynamic CFG callbacks, so callback removal APIs are not dead.
- `on_before_reload()`/`app_reload_callback()` currently have sparse static fanout, but they are public callback ABI and pair with the `callbacks_on_reload` registry. Removing them would break extension reload hooks for no maintenance benefit.
- Callback dispatch wrappers are repetitive by category, but each preserves a public registration name, payload/signature, reverse-order behavior where applicable (`script_unloaded`, `before_ui`), return aggregation where applicable (`ui_tabs`, optimizer/unet lists), and category-specific error context. A generic dispatcher would add risk to extension ABI without deleting meaningful duplication.
- `script_loading.load_module()` overlaps conceptually with script body reload in `modules/scripts.py`, but `load_module()` owns importlib execution and `loaded_scripts` tracking used by installed extensions; reload logic in `modules/scripts.py` selectively re-executes already-loaded modules. They were preserved as separate layers.
- `script_loading.preload_extensions()` is live before normal shared options initialization through `modules/shared_cmd_options.py`, where extension `preload.py` can extend the CLI parser. It must remain lightweight and independent of full extension loading.
- `ScriptPostprocessingForMainUI` and `create_auto_preprocessing_script_data()` were already covered in pass 16; this pass rechecked callback-loader overlap and found no new safe dedupe. The bridge remains necessary to adapt postprocessing scripts into always-on txt2img/img2img script classes.
- `ui_extensions.py` loader-adjacent helpers with low external fanout are Gradio callback-bound inside `create_ui()` or JS-bound through `javascript/extensions.js`. `normalize_git_url()`/`get_extension_dirname_from_url()` intentionally support install duplicate checks and index/UI install flows. Config-state table helpers overlap with extension table rendering but compare against backup/current state and are covered by escaping contract tests.
- Exact duplicate-body scan across the pass-17 target modules found no nontrivial duplicate function bodies.

### Static/dynamic audit map notes
- Extension load chain: startup calls `extensions.list_extensions()`, which scans built-in and user extension directories, reads `metadata.ini`, keys `loaded_extensions` by canonical metadata name, builds path-to-extension lookup, records requirements, and reports missing/disabled dependencies without deleting extension records.
- Script discovery chain: `modules/scripts.py` asks active extensions for script files, reads extension metadata requirements/before/after fields, topologically orders scripts, dynamically imports them through `script_loading.load_module()`, and registers script classes and callbacks.
- Callback registration chain: extension/base code calls `script_callbacks.on_*()` or `add_callback()`, source filename maps to an extension through `extensions.find_extension()`, callback names include canonical extension/script/category/name, metadata and user settings sort callbacks, and dispatch wrappers invoke callbacks with category-specific payloads.
- Preload chain: command-option setup calls `script_loading.preload_extensions()` for user and built-in extension directories before full startup so `preload.py` files can add parser options without requiring normal extension discovery.
- Extension UI chain: `ui_extensions.create_ui()` binds installed/update/install/backup/restore callbacks, extension tables call `Extension.read_info_from_repo()`, update/apply routes call `check_updates()`/`fetch_and_reset_hard()`, and index installation uses JS to pass URLs into hidden Gradio controls.
- Compatibility surfaces to continue treating conservatively: metadata file name and sections (`metadata.ini`, `[Extension]`, `callbacks/*`, script relative sections), canonical extension names and alternative requirements with `|`, `loaded_extensions` keying by metadata name, `extensions.extension_paths`, `Extension.cached_fields`, `extensions-git` cache schema, public callback `on_*` function names/signatures, callback category names, callback-name format `extension/script/category[/name]`, ordered callback cache behavior, `loaded_scripts` module map, extension `preload.py` parser hook, Gradio element ids in extension UI, extension index JSON keys, and config-state backup JSON shape.

### Validation log
- `PYTHONPYCACHEPREFIX=/tmp/gb10-a1111-pycompile-pass17 python3 -m py_compile modules/extensions.py modules/script_callbacks.py modules/script_loading.py modules/scripts_auto_postprocessing.py modules/ui_extensions.py modules/scripts.py modules/shared_cmd_options.py modules/shared_items.py modules/config_states.py modules/api/api.py modules/sysinfo.py` - passed.
- `python3 -m pytest -q tests/test_extensions_metadata_contract.py tests/test_ui_extensions_contract.py test/test_api_script_defaults.py test/test_postprocessing_script_args.py` - passed: 11 passed, with the existing pytest config warning `Unknown config option: base_url`.
- `git diff --check` - passed.

### Next unchecked scope
- More slices are still needed. Recommended next slice: remaining initialization/startup/runtime control and launch-adjacent surfaces not yet audited function-by-function, especially `modules/initialize.py`, `modules/initialize_util.py`, `modules/shared_cmd_options.py`, `modules/shared_options.py`, `modules/shared.py`, `modules/shared_init.py`, `modules/options.py` if present, `modules/restart.py`, `modules/errors.py`, `modules/devices.py` startup-facing helpers not already covered by precision/device dtype passes, `launch.py`, `modules/launch_utils.py`, `modules/timer.py`, and adjacent startup/config tests.
## Pass 18 - initialization/startup/runtime control and launch-adjacent surfaces (2026-06-20)

### Checked scope
- Startup orchestration: `modules/initialize.py`, including import ordering, version checks, full initialization, reload initialization, model-load thread setup, extension/script/upscaler/VAE/textual-inversion/extra-network refresh, optimizer/unet callback registration, and startup timer records.
- Startup utility helpers: `modules/initialize_util.py`, including server-name resolution, torch and PyTorch Lightning compatibility shims, asyncio event-loop policy setup, extension config-state restore, TLS validation, legacy Gradio auth credential parsing, stack dumping, SIGINT handling, options onchange wiring, middleware setup, and CORS configuration.
- Command/options bootstrap: `modules/shared_cmd_options.py`, `modules/shared_options.py`, `modules/shared.py`, `modules/shared_init.py`, and `modules/options.py`, including extension preload before parser finalization, non-local/extension-access flags, option template/category registration, hidden/restricted options, shared compatibility aliases/globals, controlled shared initialization, dtype/device selection, config load/migrations, option freeze/restriction enforcement, API restriction flags, option JSON dump, dynamic option addition/reorder/casting, and settings category metadata.
- Runtime stop/error/device helpers: `modules/restart.py`, `modules/errors.py`, and startup-facing portions of `modules/devices.py`, including restart sentinel behavior, process stop, exception recording/report formatting, one-shot display helper, version warning, optimal-device selection, CUDA/MPS/XPU/NPU cache cleanup, TF32 enablement, dtype cast helpers, autocast/manual-cast wrappers, NaN checks, first-calculation warmup, and forced fp16 GroupNorm replacement.
- Launch surfaces: `launch.py`, `modules/launch_utils.py`, and `modules/timer.py`, including launch compatibility re-exports, uv hook entry, sysinfo dump path, environment preparation, git/version helpers, package/install checks, repository clone/pull helpers, extension installers, requirements parsing, test-server argument mutation, API/headless start, startup timer subcategories, summaries, dumps, and reset.
- Adjacent startup/config tests and dynamic call sites: `webui.py`, `modules/cmd_args.py`, `modules/api/api.py` stop/restart and extension routes, `modules/ui_extensions.py`, `modules/sysinfo.py`, `modules/config_states.py`, `modules/ui.py`, `modules/processing.py`, first-party extensions that call `shared.opts.add_option()` or device helpers, `tests/test_errors_contract.py`, `tests/test_launch_shell_contract.py`, `test/test_openclaw_device_dtypes.py`, `tests/test_ui_extensions_contract.py`, and API/script/default tests touching options and startup wiring.
- Focused duplicate/reachability checks: AST function/class listing for every target file, exact nontrivial function-body duplicate scan across the pass-18 target modules, grep fanout for sparse-reference helpers and launch re-exports, targeted reads for webui startup/auth/middleware paths, options method call sites, restart/API extension routes, error helper usage, and device helper/runtime dtype call sites.

### Findings and fix decision
- No safe source deletion or deduplication was found in this bounded slice; this pass is a ledger-only checkpoint.
- `initialize.imports()`, `initialize.initialize()`, and `initialize.initialize_rest()` look broad but encode import-order side effects, shared initialization, extension/script discovery, model-list refreshes, and reload behavior. The nested `load_model()` helper is thread-targeted startup work and must retain access to the surrounding optimizer/model setup state.
- `initialize_util.fix_pytorch_lightning()` is called both before heavy imports in `webui.py` and during full initialization; the duplicate call is intentional idempotent protection for legacy imports and extension compatibility. `fix_torch_version()` and `fix_asyncio_event_loop_policy()` are one-time startup shims with sparse static fanout by design.
- `get_gradio_auth_creds()` has no active call site in the current GB10 headless-only web UI path, but `--gradio-auth` and `--gradio-auth-path` remain accepted legacy CLI options and the helper is public in `modules.initialize_util`. Because the browser UI was removed from this fork but compatibility arguments remain, removing the parser helper would be a risky public-surface deletion without meaningful maintenance benefit.
- `setup_middleware()` and `configure_cors_middleware()` are live through `webui.api_only()` and must preserve the reset/build sequence for FastAPI middleware changes. `server_name()` is live through API launch and preserves legacy `--listen`/`--server-name` semantics.
- `shared_cmd_options.py` performs extension preloading before final argument parsing so extension `preload.py` files can extend the parser; the module-level assignments to `webui_is_non_local` and `disable_extension_access` are startup policy state consumed by extension/UI/API paths.
- `shared_options.py` is mostly declarative option template data. Duplicate-looking `OptionInfo(...).info()/needs_reload_ui()` chains are settings metadata, not code duplication. Hidden options such as `restore_config_state_file`, old migration keys, and directory restrictions are live through config-state restore, UI/API settings, and compatibility migrations.
- `modules/shared.py` contains many module-level globals and aliases with uneven static fanout. Low-reference fields such as `batch_cond_uncond`, `cmd_opts`, `opts`, `OptionInfo`, path aliases, and utility aliases are compatibility exports for installed scripts/extensions and older internal imports; only strongly-dead removals would be safe here.
- `shared_init.initialize()` centralizes mutable shared state setup after command-line parsing. The dtype and device code overlaps conceptually with `modules/devices.py`, but the split is intentional: `shared_init` chooses startup configuration, while `devices` owns runtime helpers and context managers.
- `Options` methods in `modules/options.py` are all live or public settings API: `set()` is used by API/settings flows, `dumpjson()` feeds settings/UI/API metadata, `add_option()` is used by extensions, `reorder()` is used during UI construction, and `cast_value()` is used when applying textual/API setting values. `OptionHTML` is live for explanatory settings rows.
- `restart.is_restartable()`, `restart_program()`, and `stop_program()` are live through extension UI and API stop/restart routes. The restart sentinel file and immediate process exit are intentionally tiny process-control helpers.
- In `modules/errors.py`, `display_once()` has sparse local fanout, but the module is a public error-reporting surface used by extensions and dynamic startup code. `record_exception()` intentionally stores recent exceptions for sysinfo, and the simple formatting helpers feed that sysinfo/API diagnostic surface.
- `modules/devices.py` startup-facing helpers with sparse references are runtime hardware compatibility hooks. `enable_tf32()` is invoked at import through `errors.run()`, manual-cast/autocast helpers are selected by dtype/device state, and NPU/XPU/MPS/CUDA helpers must remain separate because they gate backend-specific cleanup and context behavior.
- `launch.py` re-exports many `launch_utils` helpers for legacy imports (`import launch; launch.run_pip`, `launch.commit_hash`, `launch.git_tag`, etc.). Even where current internal fanout is small, extensions and scripts historically import these names from `launch`, so the shim was preserved.
- `launch_utils` has several small wrappers around `run()`/git/pip behavior. They differ by error handling, live output, environment, autofix behavior, extension installer context, requirements parsing, and test-server mutation, so no shared abstraction would safely remove meaningful duplication.
- `TimerSubcategory.__exit__()` contains a typo-preserved local name `elapsed_for_subcategroy`, but it is harmless and purely local. No cosmetic rename was made.
- Exact duplicate-body scan across pass-18 target modules found no nontrivial duplicate function bodies.

### Static/dynamic audit map notes
- Launch chain: `launch.py` imports `launch_utils`, optionally patches pip/subprocess through `uv_hook`, prepares environment unless skipped, mutates test-server args when requested, then imports `webui` through `launch_utils.start()` and enters API-only mode when `--nowebui` is set.
- Webui startup chain: `webui.py` records launcher time, installs PyTorch Lightning compatibility before heavy imports, calls `initialize.imports()`, checks versions, and in API-only mode runs full initialization, middleware setup, API construction, startup callbacks, and Uvicorn launch.
- Shared/options chain: `shared_cmd_options.py` preloads extension parser hooks, parses CLI args, computes remote-access extension policy; `shared_init.initialize()` loads option templates/config, chooses devices/dtypes, creates state/styles/interrogate/progress/mem-monitor singletons, and writes those fields back into `modules.shared`.
- Reload/startup rest chain: `initialize_rest()` refreshes samplers/extensions/config-state/model lists/localizations/scripts/upscalers/VAE/textual inversion/optimizer and extra-network registries; optionally reloads UI modules; and starts deferred model loading unless skipped.
- Settings chain: `Options.__setattr__()` enforces frozen/restricted settings, `Options.set()` applies API/UI changes plus callbacks, `configure_opts_onchange()` wires heavyweight reload callbacks through the queue, and `dumpjson()` exposes settings/comments/categories to UI/API consumers.
- Runtime control chain: API and extension UI stop/restart routes call `modules.restart`; exception helpers collect recent failures for sysinfo; device helpers initialize TF32 at import, provide backend-specific device selection/cleanup, and wrap inference/training call sites in precision-aware context managers.
- Compatibility surfaces to continue treating conservatively: `launch.py` re-export names, `modules.shared` globals/aliases, accepted legacy CLI args even when browser UI is removed, `OptionInfo` chain methods and option keys, hidden config migration keys, `Options` method names, `errors` public helpers, restart sentinel path `tmp/restart`, startup timer record/category names, `startup_timer` singleton, and backend-specific device helper names/signatures.

### Validation log
- `PYTHONPYCACHEPREFIX=/tmp/gb10-a1111-pycompile-pass18 python3 -m py_compile modules/initialize.py modules/initialize_util.py modules/shared_cmd_options.py modules/shared_options.py modules/shared.py modules/shared_init.py modules/options.py modules/restart.py modules/errors.py modules/devices.py launch.py modules/launch_utils.py modules/timer.py webui.py modules/cmd_args.py modules/api/api.py modules/ui_extensions.py modules/sysinfo.py modules/config_states.py modules/ui.py modules/processing.py` - passed.
- `python3 -m pytest -q tests/test_errors_contract.py tests/test_launch_shell_contract.py test/test_openclaw_device_dtypes.py tests/test_ui_extensions_contract.py test/test_api_script_defaults.py test/test_postprocessing_script_args.py` - failed during collection because system Python on GB10 has no `torch` module available (`ModuleNotFoundError: No module named torch`).
- `python3 -m pytest -q tests/test_errors_contract.py tests/test_launch_shell_contract.py tests/test_ui_extensions_contract.py test/test_api_script_defaults.py test/test_postprocessing_script_args.py` - passed: 10 passed, with the existing pytest config warning `Unknown config option: base_url`.
- `git diff --check` - passed after removing an extra blank line at EOF in the ledger.

### Next unchecked scope
- More slices are still needed. Recommended next slice: remaining model/checkpoint/VAE/hijack/extra-network registration and model-list startup surfaces not yet deeply audited function-by-function after the startup pass, especially `modules/sd_models.py` startup/list/reload helpers not already covered by cache passes, `modules/sd_vae.py`, `modules/sd_unet.py`, `modules/sd_hijack*.py`, `modules/sd_hijack_optimizations.py`, `modules/sd_samplers*.py` registration/listing portions not already covered by sampler/device passes, `modules/modelloader.py`, `modules/upscaler.py`, `modules/shared_items.py`, and adjacent model-list/metadata tests.

## Pass 19 - model/checkpoint/VAE/hijack/upscaler/sampler registration and model-list startup surfaces (2026-06-20)

### Checked scope
- Checkpoint model-list and selection surfaces: `modules/sd_models.py`, including `CheckpointInfo` ID/title registration, `checkpoint_alisases` typo compatibility alias, `setup_model()`, `checkpoint_tiles()`, `list_models()`, `get_closet_checkpoint_match()` typo-preserved lookup, `select_checkpoint()`, `SdModelData` lazy model property backing, model reuse/reload entry points, and adjacent checkpoint path contract tests.
- VAE registration/list/reload surfaces: `modules/sd_vae.py`, including loaded/base VAE state helpers, `refresh_vae_list()`, near-checkpoint and metadata resolution helpers, `VaeResolution`, `load_vae_dict()`, `load_vae()`, `_load_vae_dict()`, `clear_loaded_vae()`, and `reload_vae_weights()`.
- UNet and hijack registration surfaces: `modules/sd_unet.py`, `modules/sd_hijack.py`, `modules/sd_hijack_checkpoint.py`, `modules/sd_hijack_clip.py`, `modules/sd_hijack_clip_old.py`, `modules/sd_hijack_open_clip.py`, `modules/sd_hijack_unet.py`, `modules/sd_hijack_ip2p.py`, `modules/sd_hijack_xlmr.py`, and `modules/sd_hijack_utils.py`, including optimizer and UNet callback lists, model hijack/undo/redo, weighted-forward/circular-conv patch helpers, textual-inversion embedding wrappers, CLIP/OpenCLIP/XLMR custom-word wrappers, DDPM edit hijacks, checkpointing hijack toggles, and conditional monkey-patch helper.
- Attention optimizer registration/listing and runtime-select surfaces: `modules/sd_hijack_optimizations.py`, including `SdOptimization` subclasses, `list_optimizers()`, cross-attention implementations, SDP backend availability/status/setter helpers, and attnblock forward replacements.
- Sampler registration/listing portions: `modules/sd_samplers.py`, `modules/sd_samplers_common.py`, `modules/sd_samplers_compvis.py`, `modules/sd_samplers_kdiffusion.py`, `modules/sd_samplers_lcm.py`, `modules/sd_samplers_timesteps.py`, and `modules/sd_samplers_extra.py`, including sampler data lists, `set_samplers()`, visible sampler helpers, infotext sampler/scheduler mapping helpers, refiner application, KDiffusion/LCM/timestep sampler construction, and restart sampler registration.
- Model loader/upscaler/shared settings list surfaces: `modules/modelloader.py`, `modules/upscaler.py`, and `modules/shared_items.py`, including generic model discovery, upscaler class discovery/de-duplication, spandrel extra-architecture one-shot registration, upscaler base/data classes, built-in upscalers, settings dropdown item providers, refresh wrappers, callback ordering settings, and `modules.shared.sd_model` lazy-property bridge.
- Adjacent call sites and tests: `modules/initialize.py`, `modules/ui.py`, `modules/ui_settings.py`, `modules/ui_checkpoint_merger.py`, `modules/ui_extra_networks_checkpoints.py`, `modules/ui_extra_networks_textual_inversion.py`, `modules/processing.py`, `modules/processing_scripts/refiner.py`, `modules/processing_scripts/sampler.py`, `modules/extras.py`, `modules/img2img.py`, `scripts/xyz_grid.py`, `scripts/prompts_from_file.py`, first-party upscaler extensions, `tests/test_sd_models_checkpoint_info_contract.py`, `tests/test_upscaler_tiling_contract.py`, `test/test_openclaw_device_dtypes.py`, `test/test_openclaw_multi_sampler.py`, and `test/test_openclaw_cache_invalidation.py`.
- Focused duplicate/reachability checks: AST function/class listing for all target modules, exact nontrivial function-body duplicate scan across the pass-19 target files before and after remediation, grep fanout for sparse-reference registration helpers and compatibility aliases, and targeted reads of model-list/upscaler/VAE/sampler/hijack startup call chains.

### Findings and fixes
- Removed duplicated OpenCLIP tokenizer wrapper setup by adding `FrozenOpenCLIPEmbedderWithCustomWordsBase` in `modules/sd_hijack_open_clip.py`. The public `FrozenOpenCLIPEmbedderWithCustomWords` and `FrozenOpenCLIPEmbedder2WithCustomWords` class names remain intact, while their distinct transformer and embedding-init behavior stays in the concrete classes.
- Removed duplicated timestep denoiser constructor bodies by adding `CompVisTimestepsDenoiserBase` in `modules/sd_samplers_timesteps.py`. The non-v and v-prediction denoiser classes still expose the same names and forward behavior, and only share their `inner_model` initialization.
- No other safe source deletion or deduplication was found in this bounded slice. The post-fix exact duplicate-body scan across target modules found no remaining nontrivial duplicate function bodies.

### Preserved compatibility/dead-code decisions
- `checkpoint_alisases` and `get_closet_checkpoint_match()` are typo-preserved compatibility surfaces with active internal and script callers. They were not renamed or replaced.
- `path_is_parent()` and `replace_key()` have narrow fanout but are covered by checkpoint path/title registration contracts and protect model root handling and dictionary order during hash/title updates.
- `CheckpointInfo.register()`, `calculate_shorthash()`, `list_models()`, `checkpoint_tiles()`, and `select_checkpoint()` are startup/UI/API-facing model-list contracts. Alias registration intentionally includes hashes, sha256, model names, titles, short titles, and old partial-hash title forms.
- VAE helpers with sparse fanout are state-machine pieces for command-line VAE override, metadata VAE choice, near-checkpoint discovery, base-VAE restoration, LRU VAE cache, and settings onchange reloads. `_load_vae_dict()` remains private but useful as the single dtype-load point.
- `sd_unet` option classes and `SdUnet.activate()`/`deactivate()` pass-body methods are extension callback contracts. `original_forward` remains as a documented temporary compatibility field even though current code does not read it.
- Hijack functions such as `fix_checkpoint()`, circular-conv patching, `register_buffer()`, weighted-forward helpers, and `undo_hijack()` are monkey-patch or model-lifecycle hooks with intentionally low direct fanout. Removing them would risk model, training, or extension compatibility for little cleanup value.
- `sd_hijack_optimizations.py` contains many parallel attention implementations and backend status/setter helpers. They are deliberately separate because they patch different attention classes or encode different backend constraints and user-selectable modes.
- Sampler registration helpers are settings/API/infotext-facing. `set_samplers()` runs both at import and startup refresh, visible sampler helpers feed UI choices, and HR sampler/scheduler getters preserve paste/infotext compatibility.
- `modelloader.load_file_from_url` is a compatibility re-export. Upcaler discovery by `Upscaler.__subclasses__()` and reversed duplicate filtering is reload-sensitive and should remain explicit.
- `shared_items.py` wrappers look thin but are settings dropdown/refresh callback providers, with imports deferred to avoid startup cycles; the `Shared` module-class swap is the lazy `shared.sd_model` bridge and was preserved.

### Static/dynamic audit map notes
- Startup registration chain: `initialize.initialize()` calls `sd_models.setup_model()`, `sd_samplers.set_samplers()`, `sd_models.list_models()`, `modelloader.load_upscalers()`, `sd_vae.refresh_vae_list()`, registers optimizer/UNet callbacks, then calls `sd_hijack.list_optimizers()` and `sd_unet.list_unets()`.
- Checkpoint list chain: `sd_models.list_models()` clears global lists, optionally exposes the `--ckpt` path, discovers checkpoint files through `modelloader.load_models()`, creates `CheckpointInfo`, and registers every title/hash/name alias for UI, API, infotext, refiner, grid, and img2img lookup paths.
- VAE chain: startup refreshes `vae_dict`; settings, user metadata, nearby checkpoint names, and command-line `--vae-path` resolve through `resolve_vae()`; reload temporarily undoes model hijacks, loads/restores VAE weights, reapplies hijacks, and emits `model_loaded_callback()`.
- Hijack/UNet chain: model load and reload call `sd_hijack.model_hijack.hijack()`, which wraps supported text encoders, applies attention optimizations, weighted-forward and DDPM edit patches, and loads textual-inversion embeddings; reload and VAE swaps undo/reapply those patches as needed.
- Upscaler chain: `modelloader.load_upscalers()` imports `*_model.py` modules for subclass registration, instantiates the latest copy of each `Upscaler` subclass after reload, stores `UpscalerData` entries in `shared.sd_upscalers`, and keeps `None`/Lanczos/Nearest first.
- Compatibility surfaces to continue treating conservatively: checkpoint alias keys and typo names, VAE option strings (`Automatic`, `auto`, `None`), `SdUnetOption`/`SdUnet` class API, hijack wrapper class names, optimizer labels and callback object fields, sampler names/aliases/options and infotext keys, `load_file_from_url` re-export, `Upscaler`/`UpscalerData` attributes, shared item provider function names, and `modules.shared.sd_model` lazy property behavior.

### Validation log
- `PYTHONPYCACHEPREFIX=/tmp/gb10-a1111-pycompile-pass19 python3 -m py_compile modules/sd_models.py modules/sd_vae.py modules/sd_unet.py modules/sd_hijack.py modules/sd_hijack_optimizations.py modules/sd_hijack_checkpoint.py modules/sd_hijack_clip.py modules/sd_hijack_clip_old.py modules/sd_hijack_open_clip.py modules/sd_hijack_unet.py modules/sd_hijack_ip2p.py modules/sd_hijack_xlmr.py modules/sd_hijack_utils.py modules/sd_samplers.py modules/sd_samplers_common.py modules/sd_samplers_compvis.py modules/sd_samplers_kdiffusion.py modules/sd_samplers_lcm.py modules/sd_samplers_timesteps.py modules/sd_samplers_extra.py modules/modelloader.py modules/upscaler.py modules/shared_items.py tests/test_sd_models_checkpoint_info_contract.py tests/test_upscaler_tiling_contract.py test/test_openclaw_device_dtypes.py test/test_openclaw_multi_sampler.py test/test_openclaw_cache_invalidation.py` - passed.
- `python3 -m pytest -q tests/test_sd_models_checkpoint_info_contract.py tests/test_upscaler_tiling_contract.py test/test_openclaw_device_dtypes.py test/test_openclaw_multi_sampler.py test/test_openclaw_cache_invalidation.py` - failed during collection because system Python on GB10 has no `torch` module and no `diskcache` module available (`ModuleNotFoundError: No module named 'torch'`; `ModuleNotFoundError: No module named 'diskcache'`).
- `python3 -m pytest -q tests/test_sd_models_checkpoint_info_contract.py tests/test_upscaler_tiling_contract.py test/test_openclaw_multi_sampler.py` - passed: 7 passed, 1 skipped, with the existing pytest config warning `Unknown config option: base_url`.
- `git diff --check` - passed after code and ledger updates.

### Next unchecked scope
- More slices are still needed. Recommended next slice: extra-network and textual-inversion registry/list/page surfaces not fully covered by this model-list slice, especially `modules/extra_networks.py`, `modules/ui_extra_networks.py`, `modules/ui_extra_networks_*.py`, `modules/textual_inversion/*.py`, `extensions-builtin/Lora/*` registration/listing surfaces as needed, and adjacent extra-network metadata/list tests.

## Pass 20 - extra-network and textual-inversion registry/list/page surfaces (2026-06-20)

### Checked scope
- Extra-network prompt registry/runtime: `modules/extra_networks.py`, including registry/alias reset and registration, default hypernetwork registration, `ExtraNetworkParams`, lookup/activation/deactivation ordering, prompt parsing, batch extra-network equality checks, and sidecar user metadata loading.
- Extra-network UI base/list/card/tree/API surfaces: `modules/ui_extra_networks.py`, including allowed preview extension helpers, page registration and allowed-directory refresh, thumb/cover/metadata/single-card API routes, JS/HTML quoting helpers, card HTML assembly, tree/dirs/card pane rendering, sort keys, preview/embedded-preview/description lookup, page ordering, UI tab refresh wiring, path parent checks, and legacy save-preview callback.
- Built-in extra-network page subclasses: `modules/ui_extra_networks_textual_inversion.py`, `modules/ui_extra_networks_hypernets.py`, `modules/ui_extra_networks_checkpoints.py`, `modules/ui_extra_networks_user_metadata.py`, and `modules/ui_extra_networks_checkpoints_user_metadata.py`, including per-page refresh/create/list/allowed-directory methods, user metadata editor save/preview flows, checkpoint VAE metadata editing, and metadata table rendering.
- Textual-inversion registry/database/training helpers: `modules/textual_inversion/textual_inversion.py`, `modules/textual_inversion/ui.py`, `modules/textual_inversion/image_embedding.py`, `modules/textual_inversion/dataset.py`, `modules/textual_inversion/autocrop.py`, `modules/textual_inversion/learn_schedule.py`, and `modules/textual_inversion/saving_settings.py`, including template listing, `Embedding`, embedding directory mtime tracking, `EmbeddingDatabase` directory registration/load/cache/find behavior, embedding creation/load/save, tensorboard/loss helpers, training validation and preview-save branches, image-embedded embedding encode/decode, dataset/batch loader classes, autocrop point selection, and save-settings logging.
- Lora registration/listing surfaces: `extensions-builtin/Lora/scripts/lora_script.py`, `extensions-builtin/Lora/lora.py`, `extensions-builtin/Lora/networks.py` registration/listing portions, `extensions-builtin/Lora/network.py`, `extensions-builtin/Lora/extra_networks_lora.py`, `extensions-builtin/Lora/ui_extra_networks_lora.py`, and `extensions-builtin/Lora/ui_edit_user_metadata.py`, including before-UI page/extra-network registration, `lyco` alias registration, API JSON/list/refresh routes, compatibility `lora.py` aliases, network file discovery/list refresh, aliases/hash lookup, AddNet infotext conversion, Lora card filtering and user metadata prompt/negative prompt generation.
- Adjacent tests and contracts: `tests/test_extra_networks_path_contract.py`, `tests/test_extra_networks_metadata_contract.py`, `tests/test_textual_inversion_preview_save_contract.py`, `tests/test_textual_inversion_autocrop_contract.py`, plus grep/AST fanout through UI/API/processing/hijack/Lora call sites.
- Focused duplicate/reachability checks: AST function/class listing for target files, exact nontrivial function-body duplicate scan across the pass-20 target set, targeted reads of page item builders and metadata editors, and grep fanout for registry aliases, list refresh functions, compatibility aliases, and embedding database methods.

### Findings and fixes
- Deduplicated the common extra-network page listing loop by adding `ExtraNetworksPage.list_items_from_names()` in `modules/ui_extra_networks.py`. The helper preserves the existing snapshot-before-iteration behavior, enumeration index assignment, `create_item()` filtering, and non-`None` yield behavior.
- Replaced the identical `list_items()` loops in textual inversion, hypernetwork, checkpoint, and Lora pages with `yield from self.list_items_from_names(...)`, while keeping each page's registry source explicit.
- No safe dead public registry/list alias deletion was found in this bounded slice.
- Exact duplicate-body scan across pass-20 targets still reports the known `network_mxfp8_wanted_names()` / `network_nvfp4_wanted_names()` pair in `extensions-builtin/Lora/networks.py`; those are quantization-runtime helpers rather than registration/listing surfaces and should be handled only in a quantized-weight runtime slice with matching tests.

### Preserved compatibility/dead-code decisions
- `extra_network_registry`, `extra_network_aliases`, `register_extra_network_alias()`, and the `lyco` alias remain public extension/prompt ABI. Sparse static fanout is expected because extensions register extra networks dynamically during `before_ui` callbacks.
- `ExtraNetworkParams.named` and `.positional` are preserved even when current first-party networks mostly read positional items; prompt argument parsing is a public extra-network contract.
- `get_single_card()` and `setup_ui().save_preview()` have compatibility-sensitive JS/API callers and preserve the single-card metadata refresh and legacy gallery preview-save behavior.
- Page `create_item()` methods intentionally remain separate: checkpoint items use checkpoint metadata and `selectCheckpoint()`, textual inversion uses embedding names and `.preview.*` local previews, hypernetworks use cached hypernet hashes and `.preview.*`, and Lora items apply user metadata, activation text, negative prompt, embedded safetensors previews, and model-version filtering.
- `allowed_directories_for_previews()` implementations remain per page because each exposes different filesystem roots to the thumbnail/preview API allowlist.
- Textual-inversion image embedding cache behavior, directory mtime reload gate, skipped-embedding shape handling, and `register_embedding_by_name(None, ...)` unregister path were preserved as live runtime behavior used by hijack and bundled-Lora embedding code.
- `extensions-builtin/Lora/lora.py` remains a compatibility alias module for legacy imports (`available_loras`, `loaded_loras`, etc.). The Lora API route JSON helper and refresh route remain public API surfaces.
- Lora `available_network_aliases`, `forbidden_network_aliases`, `available_network_hash_lookup`, and AddNet infotext migration logic remain compatibility surfaces for prompt paste/API/UI behavior.

### Static/dynamic audit map notes
- Extra-network startup chain: `initialize.initialize_rest()` clears/registers core extra-network registries, default pages are registered in `ui_extra_networks.register_default_pages()`, and the built-in Lora script registers its page plus `lora`/`lyco` extra-network handlers during `before_ui`.
- Prompt runtime chain: processing parses prompts through `extra_networks.parse_prompts()`, resolves names and aliases with `lookup_extra_networks()`, activates mentioned networks first, activates unmentioned registered networks with empty params, then deactivates in matching registry-aware order.
- UI page chain: `ui_extra_networks.create_ui()` snapshots registered pages in preferred order, creates empty initial panes, refreshes page data on hidden refresh buttons, and lazily builds page HTML through each page's `list_items()` and `create_item()` methods.
- Metadata/preview chain: pages cache metadata in `create_html()`, sidecar JSON metadata can override descriptions, preview fetches are restricted to registered page directories and allowed preview extensions, Lora safetensors cover images are served through the cover-image API, and single-card refresh re-reads metadata without the MassFileLister cache.
- Textual inversion chain: startup/model hijack registers embedding dirs, `EmbeddingDatabase.load_textual_inversion_embeddings()` reloads only when embedding directories change unless forced, page refresh forces reload, CLIP hijack finds embeddings through `find_embedding_at_position()`, and Lora bundled embeddings use `create_embedding_from_data()` plus database registration/unregistration.
- Lora list chain: `networks.list_available_networks()` clears available network maps, scans Lora and legacy LyCORIS directories, filters quantized model-cache paths, builds alias/hash lookup maps, and the Lora page/API read those maps for UI cards and `/sdapi/v1/loras`.
- Compatibility surfaces to continue treating conservatively: extra-network registry names and aliases, extra-network prompt tag syntax, `ExtraNetwork` method signatures, page names/tab ids/API route paths, thumbnail allowlist behavior, card HTML template argument names, metadata sidecar JSON keys, textual inversion file formats and PNG metadata keys, embedding database public fields, `lora.py` aliases, Lora API response keys, `lyco` alias, AddNet infotext field names, and Lora user metadata keys (`activation text`, `preferred weight`, `negative text`, `sd version`).

### Validation log
- `PYTHONPYCACHEPREFIX=/tmp/gb10-a1111-pycompile-pass20 python3 -m py_compile modules/extra_networks.py modules/ui_extra_networks.py modules/ui_extra_networks_textual_inversion.py modules/ui_extra_networks_checkpoints.py modules/ui_extra_networks_hypernets.py modules/ui_extra_networks_user_metadata.py modules/ui_extra_networks_checkpoints_user_metadata.py modules/textual_inversion/textual_inversion.py modules/textual_inversion/ui.py modules/textual_inversion/image_embedding.py modules/textual_inversion/dataset.py modules/textual_inversion/autocrop.py modules/textual_inversion/learn_schedule.py modules/textual_inversion/saving_settings.py extensions-builtin/Lora/lora.py extensions-builtin/Lora/networks.py extensions-builtin/Lora/network.py extensions-builtin/Lora/extra_networks_lora.py extensions-builtin/Lora/ui_extra_networks_lora.py extensions-builtin/Lora/ui_edit_user_metadata.py extensions-builtin/Lora/scripts/lora_script.py` - passed.
- `python3 -m pytest -q tests/test_extra_networks_path_contract.py tests/test_extra_networks_metadata_contract.py tests/test_textual_inversion_preview_save_contract.py tests/test_textual_inversion_autocrop_contract.py` - passed: 7 passed, with the existing pytest config warning `Unknown config option: base_url`.
- `git diff --check` - passed.

### Next unchecked scope
- More slices are still needed. Recommended next slice: quantized/merged Lora runtime helper duplication and remaining Lora network application surfaces not covered by this registration/listing pass, especially the paired MXFP8/NVFP4 helper families in `extensions-builtin/Lora/networks.py`, related network module operation helpers, and focused tests or compile-only gates that do not require unavailable GB10 system-Python `torch` runtime imports.


## Pass 21 - quantized/merged Lora runtime helper duplication and remaining Lora network application surfaces (2026-06-20)

### Checked scope
- Quantized LoRA active-config runtime in `extensions-builtin/Lora/networks.py`, including MXFP8/NVFP4 wanted-name signatures, active config signatures, model-prepared guards, model-unprepared markers, snapshot/restore helpers, split q/k/v LoRA operation discovery, merged-LoRA application paths, Linear forward guards, load-state reset hooks, network file scanning/listing exclusions for quant cache paths, AddNet infotext paste conversion, and global Lora runtime maps.
- Remaining LoRA network module operation helpers: `extensions-builtin/Lora/network.py`, `network_lora.py`, `network_glora.py`, `network_hada.py`, `network_ia3.py`, `network_lokr.py`, `network_full.py`, `network_norm.py`, and `network_oft.py`, including `NetworkOnDisk`, `Network`, `ModuleType`, `NetworkModule`, `finalize_updown()`, DoRA weight decomposition, LoRA/LoCon/LoHa/LoKr/OFT/IA3/Norm/Full module creation and `calc_updown()` implementations, and the direct functional `forward()` path for standard LoRA.
- LoRA activation/API/listing surfaces not deeply changed in pass 20: `extensions-builtin/Lora/extra_networks_lora.py`, `extensions-builtin/Lora/scripts/lora_script.py`, and `extensions-builtin/Lora/ui_extra_networks_lora.py`, including eager quantized LoRA preparation failure handling, `/sdapi/v1/loras`, `/sdapi/v1/refresh-loras`, `lyco` alias registration, infotext hash replacement, Lora page filtering, and user metadata prompt/negative prompt behavior.
- Adjacent quantized-model references in `modules/sd_models.py`, `modules/mxfp8_config.py`, `modules/nvfp4_config.py`, `modules/mxfp8_model_cache.py`, `modules/nvfp4_model_cache.py`, and `modules/api/api.py` were inspected by grep/fanout to confirm the LoRA runtime helpers remain live through quantized model load/reload, diagnostics/API stats, and extra-network activation.
- Focused duplicate/reachability checks: AST function/class listing for LoRA target files, exact nontrivial function-body duplicate scan across `extensions-builtin/Lora/networks.py` and `extensions-builtin/Lora/network*.py`, targeted reads of the quantized helper families and every first-party network module class, and grep fanout for MXFP8/NVFP4 helper names and LoRA runtime/list/API hooks.

### Findings and fixes
- Deduplicated the identical MXFP8/NVFP4 active LoRA wanted-name tuple builder by adding shared `network_wanted_names()` while preserving the public `network_mxfp8_wanted_names()` and `network_nvfp4_wanted_names()` wrappers.
- Deduplicated quantized Linear state snapshot/restore mechanics by adding `network_quant_snapshot_state()`, `network_quant_restore_attr()`, and `network_quant_restore_state()`. MXFP8/NVFP4 wrapper functions still pass their own merged-LoRA attribute names, so runtime-visible attributes and transaction behavior remain unchanged.
- Deduplicated split q/k/v LoRA operation discovery by adding `network_quant_lora_ops_for_layer()`. MXFP8/NVFP4 operation helpers remain as thin wrappers, preserving their names for local call sites and possible diagnostics.
- No safe dead-code deletion was found in the remaining LoRA runtime/listing/API surfaces. The exact duplicate-body scan across the post-fix LoRA target set reported no remaining nontrivial duplicate function bodies.

### Preserved compatibility/dead-code decisions
- The MXFP8 and NVFP4 prepare transactions remain separate because their device flags, managed-module attributes, cache/config names, validation functions, stats keys, error attributes, tensor-type checks, and user-visible messages differ. Collapsing those whole transactions was deferred as too risky without torch/TorchAO runtime validation.
- `network_mxfp8_*` and `network_nvfp4_*` wrapper names were preserved even when they now delegate to shared helpers because they are runtime/debug-facing helper surfaces and make quantized failure reports easier to map to the active precision mode.
- `network_apply_mxfp8_merged_lora()` and `network_apply_nvfp4_merged_lora()` still duplicate much of the merge/quantize body. They differ by base-weight/base-bias/merged flags, config getters, validators, operation-kind error text, debug text, and restore functions; factoring them would require torch/TorchAO behavioral tests that are unavailable under system Python.
- Network module classes in `network_lora.py`, `network_hada.py`, `network_lokr.py`, `network_oft.py`, `network_glora.py`, `network_ia3.py`, `network_norm.py`, and `network_full.py` have superficially similar `to(orig_weight.device)` and `finalize_updown()` flows, but each encodes a distinct LoRA/LyCORIS/OFT/IA3 format and tensor-shape rule. No safe shared abstraction was introduced there.
- LoRA API/listing and metadata helpers remain live public surfaces: `/sdapi/v1/loras`, `/sdapi/v1/refresh-loras`, `lora.py` compatibility aliases, `lyco` prompt alias, AddNet infotext migration, hash replacement, alias/hash lookup maps, `lora_in_memory_limit` onchange behavior, user metadata keys, and page version filtering.

### Static/dynamic audit map notes
- Quantized LoRA activation chain: `ExtraNetworkLora.activate()` parses prompt params, calls `networks.load_networks()`, then eagerly calls `prepare_mxfp8_active_config()` and `prepare_nvfp4_active_config()` so quantized Linear forward paths do not perform per-step LoRA merging.
- MXFP8/NVFP4 forward chain: `network_Linear_forward()` detects `network_mxfp8_base_weight` or `network_nvfp4_base_weight`, verifies the model-level active config is prepared, triggers one model-level prepare as a guard if needed, then delegates to the original Linear forward. It intentionally never falls back to normal per-layer `network_apply_weights()` for quantized managed layers.
- Quantized merge chain: each prepare function rebuilds managed Linear modules from CPU BF16 base backups, applies direct or SD3 QkvLinear q/k/v LoRA deltas, quantizes with the relevant TorchAO config, records per-mode stats on `shared.sd_model`, and restores all snapshots on failure before surfacing a fatal preparation error through the extra-network path.
- Normal LoRA chain: non-quantized Linear/Conv2d/Norm/MHA paths either use the functional forward path when `lora_functional` is enabled or mutate/restores weights via `network_apply_weights()` and backup fields keyed by `network_current_names`.
- Compatibility surfaces to continue treating conservatively: quantized helper attribute names (`network_mxfp8_*`, `network_nvfp4_*`), model stats/error keys, `network_current_names`, `network_weights_backup`, `network_bias_backup`, module class names and `calc_updown()` signatures, `NetworkOnDisk` metadata/hash behavior, LoRA API response keys, prompt syntax, AddNet infotext keys, and `lyco` alias behavior.

### Validation log
- `PYTHONPYCACHEPREFIX=/tmp/gb10-a1111-pycompile-pass21 python3 -m py_compile extensions-builtin/Lora/networks.py extensions-builtin/Lora/network.py extensions-builtin/Lora/network_lora.py extensions-builtin/Lora/network_glora.py extensions-builtin/Lora/network_hada.py extensions-builtin/Lora/network_ia3.py extensions-builtin/Lora/network_lokr.py extensions-builtin/Lora/network_full.py extensions-builtin/Lora/network_norm.py extensions-builtin/Lora/network_oft.py extensions-builtin/Lora/extra_networks_lora.py extensions-builtin/Lora/scripts/lora_script.py extensions-builtin/Lora/ui_extra_networks_lora.py modules/sd_models.py modules/mxfp8_config.py modules/nvfp4_config.py modules/mxfp8_model_cache.py modules/nvfp4_model_cache.py modules/api/api.py` - passed.
- `python3 -m pytest -q test/test_openclaw_quant_cache.py` - failed during collection because GB10 system Python has no `torch` module available (`ModuleNotFoundError: No module named 'torch'`), with the existing pytest config warning `Unknown config option: base_url`.
- `git diff --check` - passed.

### Next unchecked scope
- More slices are still needed. Recommended next slice: quantized-model load/reload/cache and precision diagnostics duplication outside the LoRA extension, especially the paired MXFP8/NVFP4 helper families in `modules/sd_models.py`, `modules/mxfp8_model_cache.py`, `modules/nvfp4_model_cache.py`, `modules/mxfp8_config.py`, `modules/nvfp4_config.py`, `modules/mxfp8_diagnostics.py`, and adjacent API precision stats paths. Keep this conservative unless a torch/TorchAO runtime gate is available.

## Pass 22 - MXFP8/NVFP4 model-cache and precision diagnostics duplication outside Lora (2026-06-20)

### Checked scope
- Quantized weight selectors and load/reload helpers in `modules/sd_models.py`: `check_mxfp8()`, `check_nvfp4()`, TorchAO mutual exclusion/policy signature helpers, selected Linear coverage helpers, Linear region/policy/technical skip helpers, MXFP8/NVFP4 filters, and `apply_mxfp8_weight_quantization()` / `apply_nvfp4_weight_quantization()`.
- Paired model-cache modules: `modules/mxfp8_model_cache.py` and `modules/nvfp4_model_cache.py`, including cache path detection, source/cache metadata hashing, sidecar validation, safe-global registration, weights-only cache loading, eligible Linear iteration, cache assignment, and cache save/sidecar write paths.
- Config modules: `modules/mxfp8_config.py` and `modules/nvfp4_config.py`, including config factory functions, coverage constants, technical skip rules, and config validators.
- Diagnostics/API precision surfaces: `modules/mxfp8_diagnostics.py`, `scripts/mxfp8_diagnostics_api.py`, and `modules/api/api.py` precision-map helpers, including precision signatures, tensor info, source/kind classification, skip-reason reporting, quantization stats, and LoRA target reporting.
- Adjacent call sites: `modules/shared_options.py`, `modules/initialize_util.py`, `modules/processing.py`, and `extensions-builtin/Lora/networks.py` references to precision coverage, quant cache exclusion, active quantized runtime state, and precision infotext.
- Focused duplicate/reachability checks: grep fanout for MXFP8/NVFP4 helpers, AST function duplicate scan across the pass-22 target files before and after remediation, targeted reads of cache/config/model/API diagnostics call chains, and confirmation that no `modules/nvfp4_diagnostics.py` counterpart currently exists.

### Findings and fixes
- Deduplicated the nontrivial shared MXFP8/NVFP4 model-cache mechanics by adding `modules/torchao_model_cache.py`. The shared helper now owns safetensors detection, source/cache hashing, tensor and bias metadata comparison, device matching, cache path construction, sidecar load/match/write, A1111-safe `torch.load` bypass handling, eligible Linear iteration, cache validation/assignment, and cache payload/sidecar saving.
- Slimmed `modules/mxfp8_model_cache.py` and `modules/nvfp4_model_cache.py` into backend wrappers that retain cache identity, sidecar suffix, label text, TorchAO safe-global registration, quantized tensor predicates, public cache path predicates, and `load_into_model()` / `save_from_model()` entry points.
- Preserved the old private helper names in both backend modules as aliases/wrappers to the shared helper where existing focused tests and possible local diagnostics touch them directly (`_tensor_meta`, `_cached_bias_matches`, `_parameter_on_device`, `_cache_path_for`, `_sidecar_matches`, etc.).
- No safe dead-code deletion was found in the precision selectors, config validators, API precision-map helpers, or MXFP8 diagnostics route. Those surfaces are live through settings onchange reloads, infotext, diagnostics API routes, LoRA activation guards, and runtime model reload policy checks.

### Preserved compatibility/dead-code decisions
- `check_mxfp8()` and `check_nvfp4()` intentionally remain separate public helpers because callers and reload diagnostics key off specific mode names/options, even though their storage-option shape mirrors `check_fp8()`.
- MXFP8/NVFP4 selected coverage, region, policy skip, technical skip, and apply-quantization functions still duplicate some structure in `modules/sd_models.py`. Collapsing those whole flows would mix distinct config modules, tensor subclasses, validation calls, base-backup attribute names, stats keys, device flags, cache modules, and user-visible error text; this was deferred as TorchAO/runtime-sensitive without an available torch runtime gate.
- `mxfp8_config.py` and `nvfp4_config.py` retain separate config factories and validators because the TorchAO config classes and required kernel/scaling constraints differ. The shared coverage constants are duplicated but small and option-facing.
- `modules/mxfp8_diagnostics.py` remains MXFP8-specific. It probes MXTensor internals, MX scaling modes, native/emulated kernel preference, and MXFP8 integration state; there is no first-party NVFP4 diagnostics counterpart in the repo to dedupe against.
- API precision-map helpers intentionally report both `mxfp8_*` and `nvfp4_*` fields separately for stable JSON shape and diagnostics readability.
- Tiny backend wrapper duplicates left after remediation are compatibility wrappers around the shared cache helper, not independent logic.

### Static/dynamic audit map notes
- Model load chain: `load_model()` calls `check_weight_quantization_mutual_exclusion()`, applies MXFP8 and NVFP4 quantization after model dtype setup, records `openclaw_torchao_quant_policy_signature`, and avoids checkpoint cache retention for TorchAO-mutated models.
- Quantized cache chain: `apply_*_weight_quantization()` computes eligible Linear modules from policy and technical skip reasons, records BF16 base backups, tries `*_model_cache.load_into_model()`, otherwise quantizes with TorchAO and writes a sidecar-validated cache through `*_model_cache.save_from_model()`.
- Reload chain: `reload_model_weights()` detects mode/coverage/forced reload changes, invalidates cached checkpoint state for TorchAO paths, and restores quantized Linears to BF16 before generic reload/device movement paths.
- Precision-map chain: `/sdapi/v1/openclaw/precision-map` and `/sdapi/v1/precision-map` run under the model queue lock, cache by model/device/options/LoRA signature, inspect layer tensor types, include MXFP8/NVFP4 skip reasons and base-backup flags, and summarize quantization plus LoRA targeting.
- Diagnostics chain: `scripts/mxfp8_diagnostics_api.py` exposes get/run API routes that call `modules.mxfp8_diagnostics`; diagnostics can save a last-result JSON under the data path and run a background probe guarded by a module lock.
- Compatibility surfaces to continue treating conservatively: `mxfp8_model_cache` / `nvfp4_model_cache` public functions and private helper names used by tests, sidecar JSON suffixes and payload keys, quantization stats keys, skip-reason strings, option keys and coverage strings, API precision-map response fields, diagnostics route paths, and TorchAO safe-global loading behavior.

### Validation log
- `PYTHONPYCACHEPREFIX=/tmp/gb10-a1111-pycompile-pass22 python3 -m py_compile modules/sd_models.py modules/torchao_model_cache.py modules/mxfp8_model_cache.py modules/nvfp4_model_cache.py modules/mxfp8_config.py modules/nvfp4_config.py modules/mxfp8_diagnostics.py modules/api/api.py scripts/mxfp8_diagnostics_api.py test/test_openclaw_quant_cache.py` - passed.
- `python3 -m pytest -q test/test_openclaw_quant_cache.py` - failed during collection because GB10 system Python has no `torch` module available (`ModuleNotFoundError: No module named torch`), with the existing pytest config warning `Unknown config option: base_url`.
- Local interpreter check found no repo `venv/bin/python` or `.venv/bin/python`; `python3` also lacks `torch`, so Torch/TorchAO runtime validation was unavailable in this slice.
- Exact duplicate-body scan across pass-22 target files after remediation reported only tiny backend cache wrapper functions in `modules/mxfp8_model_cache.py` and `modules/nvfp4_model_cache.py`; no remaining nontrivial duplicate helper bodies were reported at the 500-character threshold.
- `git diff --check` - passed.

### Next unchecked scope
- More slices are still needed. Recommended next slice: shared generation/cache invalidation surfaces outside precision-specific caches, especially `modules/processing.py`, `modules/cache.py`, `modules/openclaw_generation_diagnostics.py`, img2img/txt2img cache-stat plumbing, infotext cache fields, and adjacent tests around cond-cache and init-image cache behavior. Keep precision runtime consolidation deferred until a torch/TorchAO-capable validation environment is available.

## Pass 23 - shared generation/cache invalidation surfaces outside precision-specific caches (2026-06-20)

### Checked scope
- Generation cache/stat plumbing in `modules/processing.py`, including `_image_cache_fingerprint()`, `_array_cache_fingerprint()`, `_clone_cache_value()`, conditional-conditioning cache keys/stat updates, `StableDiffusionProcessing.cached_c` / `cached_uc`, txt2img HR cond caches, img2img init-cache key/restore/store/status helpers, and `create_infotext()` cache/conditioning-related fields.
- File metadata cache helpers in `modules/cache.py`, including legacy JSON conversion, subsection cache creation, and `cached_data_for_file()` mtime/size invalidation behavior.
- Generation diagnostics in `modules/openclaw_generation_diagnostics.py`, including CUDA graph status summarization, request summaries, extra-generation-param filtering, per-sample diagnostics capture, and last-diagnostics storage.
- UI/API entry-point plumbing in `modules/txt2img.py` and `modules/img2img.py`, especially whether cache/stat fields are duplicated or dead across Processed JSON, infotext, and gallery-generation responses.
- Clear-cond-cache extension and tests: `extensions/openclaw-clear-cond-cache/scripts/openclaw_clear_cond_cache.py` and `test/test_openclaw_cache_invalidation.py`, covering granular cache clears, token estimation, endpoint queue locking, img2img init-cache key invalidation, file-cache invalidation, and hash-cache size checks.
- Focused duplicate/reachability checks: grep fanout for cache/stat/infotext/diagnostic/init-image symbols; targeted reads of cond-cache, img2img init-cache, diagnostics, and clear-cond-cache call chains; AST duplicate-body scan across the pass-23 target set.

### Findings and fixes
- Deduplicated shared generation-cache counter mechanics in `modules/processing.py` by adding `_cache_stats()`, `_reset_cache_stats()`, `_record_cache_stats_hit()`, and `_record_cache_stats_miss()`.
- Reused those helpers for conditional-conditioning cache stats and img2img init-cache stats while preserving the existing public/stat keys (`hits`, `misses`, `compute_seconds`, `last_hit`, `cached`, and `bypass_reason`) and the per-request `openclaw_*_cache_stats` snapshots.
- No safe dead cache-stat field deletion was found. `openclaw_cond_cache_stats`, `openclaw_img2img_init_cache_stats`, and generation diagnostics fields are copied into `Processed.js()`/API-style outputs and are useful runtime diagnostics, even when not all are embedded into image infotext.
- No safe stale infotext cache field was found. `Cache FP16 weight for LoRA`, precision weight fields, and `Conditional mask weight` remain generation-replay or diagnostics-facing fields tied to active runtime options and conditioning behavior.
- No additional duplicate init-image/conditioning reset flow was removed. The clear-cond-cache extension intentionally resets class-level caches by target, while img2img init-cache helpers maintain status and payload cloning around generation lifecycle.

### Preserved compatibility/dead-code decisions
- `dump_cache()` in `modules/cache.py` remains a no-op compatibility shim for callers from the old JSON cache era; deleting it would be low value and may break extensions that still call it after writing cache entries.
- `cached_img2img_init_stats` remains class-level state because the persistent init cache is class-level; per-processing instances snapshot it into `openclaw_img2img_init_cache_stats` for result JSON and diagnostics.
- `cached_c`, `cached_uc`, `cached_hr_c`, and `cached_hr_uc` stay as two-slot mutable list caches for legacy cond-cache behavior and the openclaw clear-cond-cache endpoint. Their sparse direct fanout is expected because callers mutate slot zero/one in place.
- Img2img init-cache bypass for masked requests was preserved. Although the key includes image-mask and latent-mask fingerprints, `_img2img_init_cache_bypass_reason()` bypasses masked requests before key creation to avoid reusing stateful inpaint/mask setup across requests.
- `openclaw_generation_diagnostics.py` remains separate from cache-stat construction. It summarizes sample/runtime diagnostics and only carries selected extra params, while cache stats live on processing/result objects.
- Public JSON field names in `Processed.js()` and cache-clear API response keys were not renamed.

### Static/dynamic audit map notes
- Cond-cache chain: `setup_conds()` calls `get_conds_with_caching()` for negative and positive prompts; the cache key includes prompts, schedules, checkpoint info, active LoRA cond signature, crop/size, FP8 settings, and emphasis mode, and records hit/miss/compute seconds on the processing instance.
- Txt2img HR cache chain: `StableDiffusionProcessingTxt2Img` maintains separate class-level HR cond caches, calls the same cond-cache helper during HR conditioning, and the clear-cond-cache extension can reset those HR slots independently.
- Img2img init-cache chain: `StableDiffusionProcessingImg2Img.init()` computes a key from init images, model/VAE identity, sampler/model conditioning mode, request geometry, inpaint settings, VAE/background/dtype/device options, restores cached latents/conditioning when available, or stores cloned payload state after computing init latents.
- File-cache chain: `cached_data_for_file()` invalidates cached metadata when either mtime or size differs, including legacy entries without size. Existing tests cover backward mtime movement and legacy size-missing entries.
- Diagnostics chain: `process_images_inner()` captures CUDA graph status before/after each sample through `openclaw_generation_diagnostics`, stores per-processing history, and `Processed.js()` exposes the latest diagnostics/history plus cache stats.
- Compatibility surfaces to continue treating conservatively: `dump_cache()`, cache subsection names, class-level cond-cache slot shapes, clear-cond-cache target names/API routes, `Processed.js()` openclaw field names, infotext precision/cache field labels, and img2img init-cache status keys.

### Validation log
- `PYTHONPYCACHEPREFIX=/tmp/gb10-a1111-pycompile-pass23 python3 -m py_compile modules/processing.py modules/cache.py modules/openclaw_generation_diagnostics.py modules/txt2img.py modules/img2img.py extensions/openclaw-clear-cond-cache/scripts/openclaw_clear_cond_cache.py test/test_openclaw_cache_invalidation.py` - passed.
- Exact AST duplicate-body scan across the pass-23 target set at the 500-character threshold reported no duplicate nontrivial function bodies after remediation.
- `python3 -m pytest -q test/test_openclaw_cache_invalidation.py` - failed during collection because GB10 system Python has no `numpy` module available (`ModuleNotFoundError: No module named 'numpy'`), with the existing pytest config warning `Unknown config option: base_url`.
- `git diff --check` - passed after trimming the ledger EOF blank line.

### Next unchecked scope
- More slices are still needed. Recommended next slice: remaining generation/result serialization and API/reporting surfaces adjacent to processing caches, especially `modules/api/api.py` txt2img/img2img response models and infotext overrides, `modules/images.py` metadata save/read helpers, `modules/infotext_utils.py` parse/paste mappings, `modules/generation_parameters_copypaste.py`, `modules/ui_common.py` save/download paths, and adjacent serialization/infotext tests.


## Pass 24 - generation/result serialization and API/reporting surfaces adjacent to processing caches (2026-06-20)

### Checked scope
- API generation/result serialization in `modules/api/api.py` and `modules/api/models.py`, including txt2img/img2img request/response models, `text2imgapi()`, `img2imgapi()`, `apply_infotext()`, `api_infotext_value_for_field()`, `processed_js_with_image_paths()`, `encode_pil_to_base64()`, `decode_base64_to_image()`, and `pnginfoapi()`.
- Image metadata save/read helpers in `modules/images.py`, including `save_image_with_geninfo()`, `save_image()`, EXIF/PNG/GIF metadata writing, `read_info_from_image()`, `image_data()`, and the `already_saved_as` path used by API response reporting.
- Infotext parse/paste compatibility in `modules/infotext_utils.py`, including the `modules.generation_parameters_copypaste` alias, paste field registration, image-from-URL/file parsing, inpaint label conversion helpers, `parse_generation_parameters()`, override-setting mapping helpers, and paste connection logic.
- UI save/download paths in `modules/ui_common.py`, including gallery generation-info switching, `save_files()`, CSV log update, selected-image vs save-all indexing, zip archive creation, and output panel save/download wiring.
- Adjacent tests/contracts: `test/test_infotext_api_mappings.py`, `tests/test_save_serialization_contract.py`, `test/test_images_save.py`, and `tests/test_processing_auxiliary_infotext_alignment.py`.
- Focused duplicate/reachability checks: grep fanout for response models, infotext aliases, metadata read/write helpers, save/download functions, and exact AST duplicate-body scan across the targeted API/images/infotext/UI files and adjacent tests.

### Findings and fixes
- Deduplicated generation-info EXIF user-comment serialization by adding `images.geninfo_to_exif_bytes()` and reusing it from `save_image_with_geninfo()` for JPEG/WebP/AVIF metadata and from API `encode_pil_to_base64()` for JPEG/WebP base64 responses.
- Removed the now-unneeded direct `piexif` imports from `modules/api/api.py`; EXIF encoding details are centralized in `modules/images.py`, where image metadata read/write helpers already live.
- No safe dead response-model, infotext, paste, or UI save/download deletion was found. The sparse-looking functions are public API/UI/extension compatibility surfaces or are covered by focused contract tests.
- Exact duplicate-body scan across the pass-24 target set reported no remaining nontrivial duplicate function bodies at the 500-character threshold after remediation.

### Preserved compatibility/dead-code decisions
- `TextToImageResponse`, `ImageToImageResponse`, `PNGInfoResponse`, `send_images`, `save_images`, `include_init_images`, `force_task_id`, and `infotext` request fields remain stable API schema/behavior surfaces.
- `processed_js_with_image_paths()` remains an API reporting wrapper because it augments `Processed.js()` with saved image paths and OpenClaw timing/cache diagnostics without changing the base UI JSON contract.
- `apply_infotext()` and `api_infotext_value_for_field()` remain API-specific glue around shared paste mappings because they must coerce Pydantic field types and fill script args from Gradio components.
- `modules.generation_parameters_copypaste` remains an alias to `modules.infotext_utils` for old extension imports; there is intentionally no real `modules/generation_parameters_copypaste.py` file.
- `infotext_to_setting_name_mapping` remains an empty compatibility extension hook for older mapping-style overrides; current first-party settings use `OptionInfo(..., infotext=...)`.
- Inpaint label conversion helpers remain separate named functions because tests and paste/API mappings rely on exact label-to-value behavior for mask mode, masked content, and inpaint area.
- `save_files()` retains its UI-specific gallery/data-url handling, selected-index behavior, CSV append, and zip creation rather than sharing API response serialization code; the UI path works from already-rendered gallery payloads and generation-info JSON, not processing objects.
- `read_info_from_image()` preserves PNG `parameters`, EXIF `UserComment`, GIF `comment`, and NovelAI compatibility parsing.

### Static/dynamic audit map notes
- API generation chain: txt2img/img2img requests may apply infotext first, resolve sampler/scheduler aliases, strip API-only fields before constructing processing objects, run under the API queue lock, optionally serialize images through `encode_pil_to_base64()`, and return `Processed.js()` augmented by OpenClaw path/cache/timing fields.
- API image metadata chain: PNG base64 responses copy string metadata from `image.info`; JPEG/WebP responses now call `images.geninfo_to_exif_bytes()` with the image `parameters` string; decoded PNG-info requests call `images.read_info_from_image()` then parse/publish infotext callbacks.
- UI save chain: output panel buttons pass generation-info JSON and gallery file data to `save_files()`, which reconstructs a lightweight processing-like object for filename patterns, parses per-image infotext, calls `images.save_image()`, updates CSV when enabled, and optionally zips saved files for the download component.
- Infotext paste chain: UI registration stores per-tab paste fields, compatibility aliases update `modules.ui.*_paste_fields`, paste buttons can send images/dimensions or text fields, and API infotext application reuses the same field definitions with Pydantic type coercion.
- Compatibility surfaces to continue treating conservatively: API response/request field names, `/sdapi/v1/png-info` response shape, `image.info['parameters']`, EXIF UserComment behavior, `modules.generation_parameters_copypaste`, paste field tuple shape, `ParamBinding` attributes, `infotext_to_setting_name_mapping`, UI `generation_info` JSON shape, save/download Gradio component behavior, and CSV log field order.

### Validation log
- `PYTHONPYCACHEPREFIX=/tmp/gb10-a1111-pycompile-pass24 python3 -m py_compile modules/api/api.py modules/api/models.py modules/images.py modules/infotext_utils.py modules/ui_common.py test/test_infotext_api_mappings.py tests/test_save_serialization_contract.py test/test_images_save.py tests/test_processing_auxiliary_infotext_alignment.py` - passed.
- `python3 -m pytest -q test/test_infotext_api_mappings.py tests/test_save_serialization_contract.py test/test_images_save.py tests/test_processing_auxiliary_infotext_alignment.py` - passed: 21 passed, with the existing pytest config warning `Unknown config option: base_url`.
- Exact AST duplicate-body scan across the pass-24 target files and adjacent tests reported no duplicate nontrivial function bodies after remediation.
- `git diff --check` - passed.

### Next unchecked scope
- More slices are still needed. Recommended next slice: remaining progress/reporting and API utility surfaces around task state, progress previews, extras/postprocessing serialization, interrogate/png-info/reporting routes, and adjacent tests, especially the rest of `modules/api/api.py`, `modules/api/models.py`, `modules/progress.py`, `modules/ui.py` progress helpers, `modules/postprocessing.py`, and API progress/extras tests.

## Pass 25 - progress/reporting and API utility surfaces (2026-06-20)

### Checked scope
- Remaining progress/task reporting surfaces in `modules/api/api.py`, `modules/api/models.py`, and `modules/progress.py`, including `/sdapi/v1/progress`, `/internal/progress`, `/internal/pending-tasks`, task queue state helpers, current-task reporting, ETA/progress calculation, and live-preview serialization.
- Extras/postprocessing API and UI wrappers in `modules/api/api.py`, `modules/api/models.py`, `modules/postprocessing.py`, and `modules/ui_postprocessing.py` fanout via grep, including `setUpscalers()`, `decode_extras_batch_images()`, `extras_single_image_api()`, `extras_batch_images_api()`, `run_postprocessing_webui()`, and `run_extras()`.
- Interrogate and PNG-info/reporting routes in `modules/api/api.py`, `modules/ui.py`, and `modules/extras.py`, including API `pnginfoapi()`, API `interrogateapi()`, UI `process_interrogate()`, `interrogate()`, `interrogate_deepbooru()`, and `run_pnginfo()`.
- Adjacent API/extras/progress tests and contracts in `test/test_postprocessing_api_defaults.py`, `test/test_extras.py`, and `tests/test_postprocessing_caption_contract.py`.
- Focused duplicate/reachability checks: grep fanout for progress/task/extras/interrogate/png-info symbols, targeted reads of route/helper call chains, and AST duplicate-body scan across the pass-25 target files and adjacent tests.

### Findings and fixes
- Fixed stale API progress task reporting by changing `/sdapi/v1/progress` to read `modules.progress.current_task` through the module object. The previous `from modules.progress import current_task` captured the initial `None` and did not reflect later `start_task()` / `finish_task()` global reassignments.
- Deduplicated the progress fraction and ETA calculation shared by `/internal/progress` and `/sdapi/v1/progress` into `progress.calculate_progress_and_eta()`. The public API keeps its existing `base_progress=0.01` behavior while the internal progress endpoint keeps its previous zero-progress/`None` ETA behavior.
- Added a focused regression test confirming API progress reports the live task id from the progress module reference.
- No safe dead-code deletion was found in extras, postprocessing, interrogate, or PNG-info route helpers. Sparse-looking wrappers are live UI/API compatibility surfaces with distinct response shapes and Gradio/API call contracts.

### Preserved compatibility/dead-code decisions
- `modules.progress.ProgressRequest` / `ProgressResponse` remain separate from `modules.api.models.ProgressRequest` / `ProgressResponse` because `/internal/progress` is a Gradio live-preview/task-state protocol while `/sdapi/v1/progress` is the public API schema with a state snapshot and optional current image.
- `run_postprocessing_webui(id_task, *args, **kwargs)` remains a UI queue wrapper even though it ignores `id_task`; `ui_postprocessing.py` calls it through `call_queue.wrap_ui_gpu_call()` and the signature is part of the queued UI contract.
- API extras endpoints continue using `run_extras()` rather than calling `run_postprocessing()` directly because `run_extras()` maps legacy API fields to postprocessing script args and preserves `upscale_first` ordering.
- API and UI PNG-info helpers stay separate: API `pnginfoapi()` returns raw/parsed JSON and fires infotext callbacks; UI `modules.extras.run_pnginfo()` returns rendered HTML plus hidden generation text for paste buttons.
- API and UI interrogate helpers stay separate because the API returns a JSON caption under queue lock, while UI helpers return Gradio update-compatible values and support batch directory modes.
- Live preview serialization in `/internal/progress` is intentionally not shared with API image serialization because it returns `data:image/...` URIs, honors `opts.live_previews_image_format`, and uses preview-specific PNG compression shortcuts.

### Static/dynamic audit map notes
- API task chain: txt2img/img2img API calls create or accept a task id, add it to `pending_tasks`, call `start_task()` inside the queue lock, call `finish_task()` in `finally`, and now `/sdapi/v1/progress` reports the live `progress_module.current_task` rather than an imported snapshot.
- Internal progress chain: JavaScript calls `/internal/progress` with a task id and last preview id; the endpoint reports active/queued/completed flags, queue position text, ETA/progress, and only sends a live preview when the preview id changes.
- Extras API chain: API request models normalize `upscaler_1`/`upscaler_2` to postprocessing names, decode images, run under the API queue lock with `save_output=False`, and serialize either a single optional image or a batch list plus HTML info.
- Postprocessing chain: UI extras call `run_postprocessing_webui()` through the queued GPU wrapper, then `run_postprocessing()` enumerates upload/directory/single-image inputs, runs postprocessing scripts, optionally saves output/captions, assigns current image for previews, and returns gallery images plus HTML info/log.
- PNG-info/interrogate chain: UI routes feed Gradio components and paste fields; API routes decode base64 input and return stable JSON response models. These chains overlap conceptually but not enough to safely collapse without changing public/UI behavior.
- Compatibility surfaces to continue treating conservatively: `/internal/progress` response shape, `/sdapi/v1/progress` response shape, task id strings, `force_task_id`, live-preview data URI format, extras API field names, `run_postprocessing_webui()` signature, API/HTML PNG-info outputs, and interrogate model names (`clip`, `deepdanbooru`).

### Validation log
- `PYTHONPYCACHEPREFIX=/tmp/gb10-a1111-pycompile-pass25 python3 -m py_compile modules/api/api.py modules/api/models.py modules/progress.py modules/ui.py modules/postprocessing.py modules/extras.py test/test_postprocessing_api_defaults.py test/test_extras.py tests/test_postprocessing_caption_contract.py` - passed.
- `python3 -m pytest -q test/test_postprocessing_api_defaults.py::test_api_progress_reports_live_current_task_reference test/test_postprocessing_api_defaults.py::test_api_extras_always_returns_images_despite_directory_gallery_toggle test/test_postprocessing_api_defaults.py::test_extras_batch_decode_skips_corrupt_images_but_keeps_valid_items test/test_postprocessing_api_defaults.py::test_extras_single_response_allows_no_output_from_skipped_or_interrupted_run tests/test_postprocessing_caption_contract.py` - passed: 6 passed, with the existing pytest config warning `Unknown config option: base_url`.
- Broader `python3 -m pytest -q test/test_postprocessing_api_defaults.py tests/test_postprocessing_caption_contract.py` was attempted and failed in GB10 system Python before/while collecting existing tests because optional/runtime dependencies are unavailable there (`ModuleNotFoundError: No module named fastapi` and extension preload errors from missing `torch`).
- Exact AST duplicate-body scan across the pass-25 target set reported only two tiny duplicate test fixture `__init__` methods in `test/test_postprocessing_api_defaults.py`; no duplicate nontrivial production function bodies were reported at the 500-character threshold.
- `git diff --check` - passed.

### Next unchecked scope
- More slices are still needed. Recommended next slice: low-level model/listing and refresh API surfaces plus remaining utility/reporting endpoints, especially `modules/api/api.py` getters after progress (`get_config`, `set_config`, sampler/scheduler/upscaler/model/embedding/extension/memory/train/create endpoints), `modules/api/models.py` item/response schemas, `modules/sd_models.py` listing helpers, `modules/sd_vae.py`, `modules/shared_items.py`, and adjacent API model/listing tests.

## Pass 26 - model/listing and refresh API surfaces (2026-06-20)

### Checked scope
- Low-level public API getters and utility/reporting endpoints in `modules/api/api.py` after progress: `get_config()`, `set_config()`, `get_cmd_flags()`, sampler/scheduler/upscaler/latent-upscale/model/VAE/hypernetwork/face-restorer/RealESRGAN/prompt-style/embedding getters, refresh endpoints, create/train embedding and hypernetwork endpoints, `get_memory()`, and `get_extensions_list()`.
- API item/response schemas in `modules/api/models.py`, especially sampler/scheduler/upscaler/model/VAE/hypernetwork/face-restorer/RealESRGAN/prompt-style/embedding/memory/extension schemas.
- Model/listing helpers in `modules/sd_models.py`, `modules/sd_vae.py`, and `modules/shared_items.py`, including `checkpoint_tiles()`, `list_models()`, `refresh_vae_list()`, `sd_vae_items()`, `refresh_checkpoints()`, `list_samplers()`, and `reload_hypernetworks()`.
- Adjacent API/model listing tests and contracts: `tests/test_api_progress_contract.py`, `tests/test_sd_models_checkpoint_info_contract.py`, and new `tests/test_api_extension_item_contract.py`.
- Focused duplicate/reachability checks: route registration grep for listing endpoints and schemas, shared-items fanout through option refresh callbacks/UI refresh buttons, extension metadata source inspection, and exact AST duplicate-body scan for `modules/api/api.py` and `modules/shared_items.py`.

### Findings and fixes
- Fixed stale `/sdapi/v1/extensions` response schema metadata types. `modules.extensions.Extension` initializes `branch` and `commit_date` as `None`, and `read_info_from_repo()` can leave them unset for detached/problematic repositories while `get_extensions_list()` still includes remote-backed extensions. `models.ExtensionItem` now declares `branch: Optional[str]` and `commit_date: Optional[int]`, matching the actual GitPython metadata (`committed_date` is an integer timestamp).
- Added `tests/test_api_extension_item_contract.py` to lock the extension metadata nullable contract against the extension source defaults.
- Updated the adjacent progress API contract test to match pass-25 behavior: `/sdapi/v1/progress` now reports `progress_module.current_task`, not the stale imported `current_task` snapshot.
- No safe dead-code deletion or serializer collapse was found in the model/listing endpoints. The compact list comprehensions are public API response serializers with stable field names, while similarly named `shared_items` functions are live UI option refresh/list callbacks.

### Preserved compatibility/dead-code decisions
- API listing item schemas (`SamplerItem`, `SchedulerItem`, `UpscalerItem`, `SDModelItem`, `SDVaeItem`, `EmbeddingItem`, `MemoryResponse`, and related item models) remain separate explicit public schemas rather than being replaced with generic dicts or shared UI helpers.
- `shared_items.refresh_vae_list()` and `shared_items.refresh_checkpoints()` remain live wrappers because `shared.py` exposes them as stable refresh callbacks and `shared_options.py` wires them into UI option refresh behavior.
- `shared_items.sd_vae_items()`, `list_checkpoint_tiles()`, `list_samplers()`, and `reload_hypernetworks()` remain live because option choices, extra-network refreshes, training UI dropdowns, and sampler visibility settings call them through `shared`/`shared_items` indirection.
- API refresh endpoints remain thin wrappers under `self.queue_lock`; collapsing them into direct route lambdas or shared UI callbacks would obscure the public API queue boundary.
- `get_sd_models()` continues returning checkpoint config via `find_checkpoint_config_near_filename()` in the API serializer; `sd_models.checkpoint_tiles()` remains UI dropdown title serialization and is not a duplicate API helper.
- `get_sd_vaes()` remains API-specific `{model_name, filename}` serialization; `shared_items.sd_vae_items()` intentionally prepends `Automatic` and `None` for UI option choices.

### Static/dynamic audit map notes
- API listing chain: route registration in `Api.__init__()` binds each `/sdapi/v1/...` listing endpoint to a method with a matching response model in `modules/api/models.py`; response keys are public API compatibility surface.
- Checkpoint listing chain: `shared_items.refresh_checkpoints()` calls `sd_models.list_models()`, `shared_items.list_checkpoint_tiles()` calls `sd_models.checkpoint_tiles()`, and API `get_sd_models()` serializes richer checkpoint records directly from `sd_models.checkpoints_list.values()`.
- VAE listing chain: `sd_vae.refresh_vae_list()` owns filesystem discovery into `sd_vae.vae_dict`; UI choice helpers add `Automatic`/`None`, while API `get_sd_vaes()` exposes discovered VAE names and filenames only.
- Extension listing chain: API `get_extensions_list()` calls `extensions.list_extensions()`, then `read_info_from_repo()` for each extension, and returns only entries with `remote is not None`; branch and commit date may still be unset depending on repository state.
- Compatibility surfaces to continue treating conservatively: `/sdapi/v1/options`, `/sdapi/v1/cmd-flags`, all `/sdapi/v1/*` listing field names, extension metadata field names, training/create endpoint response strings, queue locking on refresh routes, and UI option refresh wrappers in `shared_items.py`.

### Validation log
- `PYTHONPYCACHEPREFIX=$(mktemp -d) python3 -m py_compile modules/api/api.py modules/api/models.py modules/sd_models.py modules/sd_vae.py modules/shared_items.py tests/test_api_extension_item_contract.py tests/test_api_progress_contract.py tests/test_sd_models_checkpoint_info_contract.py` - passed.
- `pytest -q tests/test_api_extension_item_contract.py tests/test_api_progress_contract.py tests/test_sd_models_checkpoint_info_contract.py` - passed: 4 passed, with the existing pytest config warning `Unknown config option: base_url`.
- Exact AST duplicate-body scan across `modules/api/api.py` and `modules/shared_items.py` reported `0 exact duplicate function bodies` for both files.
- `git diff --check` - passed.

### Next unchecked scope
- More slices are still needed. Recommended next slice: API process/control and server lifecycle utility surfaces around auth/middleware, route registration wrappers, `/sdapi/v1/interrupt`, `/sdapi/v1/skip`, `/sdapi/v1/unload-checkpoint`, `/sdapi/v1/reload-checkpoint`, launch/kill/restart/stop handlers, URL/base64 helpers, and remaining low-level API utility functions in `modules/api/api.py` plus adjacent tests.


## Pass 27 - API process/control and server lifecycle utility surfaces (2026-06-20)

### Checked scope
- `modules/api/api.py`: auth credential parsing and `auth`, `api_middleware` request timing/error handling, `add_api_route` auth wrapper, `/sdapi/v1/interrupt`, `/sdapi/v1/skip`, `/sdapi/v1/unload-checkpoint`, `/sdapi/v1/reload-checkpoint`, `/sdapi/v1/server-kill`, `/sdapi/v1/server-restart`, `/sdapi/v1/server-stop`, `launch`, URL/base64 helpers `verify_url`, `decode_base64_to_image`, `decode_extras_batch_images`, `encode_pil_to_base64`, low-level API utility helpers adjacent to process/control routes, and route-registration context around these endpoints.
- `modules/restart.py`: `is_restartable`, `restart_program`, `stop_program`.
- Lifecycle overlap checked in `modules/shared_state.py` (`server_command`, `wait_for_server_command`, `request_restart`, `interrupt`, `skip`), `webui.py` API launch path, `modules/initialize_util.py` middleware setup overlap, and `modules/progress.py` internal route registration shape.
- Adjacent tests checked/run as feasible: `tests/test_api_extension_item_contract.py`, `tests/test_api_progress_contract.py`, and relevant API helper tests in `test/test_postprocessing_api_defaults.py`.
- `modules/server.py` was requested for overlap but is not present in this checkout; lifecycle behavior is carried by `webui.py`, `modules/restart.py`, and `modules/shared_state.py`.

### Findings and fixes
- Removed one exact duplicate exception-handler wrapper in `api_middleware()`. `fastapi_exception_handler()` and `http_exception_handler()` both returned `handle_exception(request, e)`; a single `api_exception_handler()` is now registered for both `Exception` and `HTTPException` with stacked FastAPI decorators.
- No safe deletion was found for `Api.add_api_route()`: it centralizes Basic Auth dependency injection for every API route and preserves the unauthenticated path when `--api-auth` is unset.
- `/interrupt` and `/skip` handlers intentionally map to different `shared.state` flags and log messages; their small bodies are not duplicate behavior.
- `/unload-checkpoint` and `/reload-checkpoint` are intentionally distinct model-memory controls (`unload_model_weights()` versus `send_model_to_device(shared.sd_model)`) and remain public API hooks.
- `/server-kill`, `/server-restart`, and `/server-stop` are gated by `--api-server-stop` and encode different lifecycle semantics: process exit, restart-file-plus-exit when restartable, and graceful API loop command via `shared.state.server_command`. No route wrapper was dead.
- `restart.restart_program()` and `restart.stop_program()` are live through API server-control endpoints and UI extension apply/restart flows. `is_restartable()` is live through API/UI control decisions and mirrors the `SD_WEBUI_RESTART` launcher contract.
- URL/base64 helpers were preserved: `decode_base64_to_image()` is used by img2img, extras, png-info, and interrogate APIs; `decode_extras_batch_images()` intentionally skips only corrupt batch images; `encode_pil_to_base64()` centralizes response encoding and metadata preservation; `verify_url()` enforces the local-resource request guard.
- `api_middleware()` and `initialize_util.setup_middleware()` both add middleware, but they cover different layers: API timing/logging/error normalization versus CORS/GZip/proxy/static-asset setup. No dedupe was safe.

### Static/dynamic audit map notes
- API server launch chain: `launch_utils.start()` or `webui.__main__` -> `webui.api_only()` -> `initialize.initialize()` -> FastAPI app -> `initialize_util.setup_middleware(app)` -> `Api(app, queue_lock)` -> `api_middleware(app)` and `Api.add_api_route()` registration -> `Api.launch()`/`uvicorn.run()`.
- Control route chain: API route -> `shared.state.interrupt()`/`skip()` or `sd_models` checkpoint memory helpers or `modules.restart` process helpers. Graceful server stop is signaled through `shared.state.server_command` for the server loop path.
- Dynamic/public surfaces to continue treating conservatively: API route method names, `Api.add_api_route`, auth/middleware behavior, image URL/base64 helpers, server-control endpoints gated by `--api-server-stop`, and restart helpers used by UI/extensions.

### Validation log
- `PYTHONPYCACHEPREFIX=/tmp/gb10-a1111-pycompile-pass27 python3 -m py_compile modules/api/api.py modules/restart.py modules/shared_state.py webui.py modules/initialize_util.py modules/progress.py` - passed.
- `python3 -m pytest tests/test_api_extension_item_contract.py tests/test_api_progress_contract.py -q` - passed: 2 passed, 1 warning about unknown `base_url` config.
- `python3 -m pytest tests/test_api_extension_item_contract.py tests/test_api_progress_contract.py test/test_postprocessing_api_defaults.py -q` - partial: 9 passed before `test_postprocessing_runner_order_override_preserves_script_defaults` failed at import because this bare Python environment lacks `fastapi`; extension preload also reported missing `torch`. This appears environment-related rather than caused by the handler dedupe.
- Exact AST duplicate-body scan over `modules/api/api.py`, `modules/restart.py`, `modules/shared_state.py`, `webui.py`, `modules/initialize_util.py`, and `modules/progress.py` - clean: no exact duplicate function bodies in the pass-27 scan set.
- `git diff --check` - passed.

### Next unchecked scope
- Continue with API/data-listing and metadata helper surfaces not yet fully audited in this API pass: `/sdapi/v1/options`, `/cmd-flags`, samplers/schedulers/upscalers/latent-upscale-modes, model/VAE/hypernetwork/face-restorer/realesrgan/prompt-style/embedding list helpers, refresh/create/train endpoints, memory reporting, and adjacent `modules/api/models.py` response contracts.


## Pass 28 - API data-listing and metadata helper surfaces (2026-06-20)

### Checked scope
- `modules/api/api.py` public API data/metadata routes and helpers: `/sdapi/v1/options`, `/sdapi/v1/cmd-flags`, sampler/scheduler/upscaler/latent-upscale/model/VAE/hypernetwork/face-restorer/RealESRGAN/prompt-style/embedding list helpers, refresh endpoints, create/train embedding and hypernetwork endpoints, and `/sdapi/v1/memory`.
- `modules/api/models.py` response contracts for the same route set: `OptionsModel`, `FlagsModel`, `SamplerItem`, `SchedulerItem`, `UpscalerItem`, `LatentUpscalerModeItem`, `SDModelItem`, `SDVaeItem`, `HypernetworkItem`, `FaceRestorerItem`, `RealesrganItem`, `PromptStyleItem`, `EmbeddingItem`, `EmbeddingsResponse`, `CreateResponse`, `TrainResponse`, and `MemoryResponse`.
- Adjacent source fanout for runtime value shapes and live hooks: `modules/sd_schedulers.py`, `modules/upscaler.py`, `modules/sd_models.py`, `modules/sd_vae.py`, `modules/shared_items.py`, `modules/realesrgan_model.py`, `modules/face_restoration.py`, `modules/hypernetworks/hypernetwork.py`, `modules/textual_inversion/textual_inversion.py`, `modules/shared_state.py`, and existing focused API/listing tests.
- Focused duplicate checks: route registration grep, response model fanout grep, training/create string/body inspection, and exact AST duplicate-body scan over `modules/api/api.py` and `modules/api/models.py`.

### Findings and fixes
- Fixed stale create endpoint response classes: `create_embedding()` and `create_hypernetwork()` now return `models.CreateResponse` on assertion-error paths, matching their registered `response_model=models.CreateResponse`. The serialized field shape remains the same (`info`) for API compatibility.
- Fixed copied hypernetwork training response text in `train_hypernetwork()`. It now reports `train hypernetwork complete` / `train hypernetwork error` instead of embedding messages.
- Removed a duplicated `shared.state.end()` call in `train_hypernetwork()`. The method now mirrors `train_embedding()` by restoring model/optimization state in the inner `finally` and ending the shared job once in the outer `finally`; this avoids double-ending the same shared state job.
- Added `tests/test_api_training_contract.py` to lock the create response-class contract and the hypernetwork train response/state-end behavior.
- No safe deletion or generic serializer collapse was found in the list helpers. Their compact comprehensions are public `/sdapi/v1/*` compatibility serializers with route-specific field names and response models.

### Preserved compatibility/dead-code decisions
- `get_config()` and `OptionsModel` remain dynamic over `shared.opts.data` / `opts.data_labels`; even sparse option keys are part of the public settings API.
- `get_cmd_flags()` and `FlagsModel` remain parser-derived compatibility surfaces; command-line flag defaults/types are intentionally exposed as runtime metadata.
- Sampler, scheduler, upscaler, latent upscale, checkpoint, VAE, hypernetwork, face-restorer, RealESRGAN, prompt-style, and embedding serializers remain explicit instead of being collapsed into generic dict serialization because their field names are public API contracts and their backing registries have different shapes.
- API refresh endpoints remain thin queue-locked wrappers around embedding, checkpoint, and VAE refresh hooks. They are intentionally separate from UI option refresh callbacks.
- `/sdapi/v1/memory` keeps RAM and CUDA logic in one route-local helper because it reports two different best-effort runtime sources and the public response is a permissive `dict` contract.
- `CreateResponse` and `TrainResponse` remain separate model classes despite identical fields because they document different route families in the generated API schema.

### Static/dynamic audit map notes
- Listing route chain: `Api.__init__()` registers each `/sdapi/v1/...` listing route with a response model in `modules/api/models.py`; each method serializes a distinct runtime registry into stable public field names.
- Create/train chain: API routes begin a shared state job, call textual inversion or hypernetwork create/train implementation, restore optimization/model placement where needed, and return a single `info` string through the create/train response schema.
- Refresh chain: API refresh routes run under `self.queue_lock` and dispatch to embedding DB reload, checkpoint refresh, or VAE refresh, preserving the same queue boundary as other runtime-mutating API calls.
- Memory chain: `/sdapi/v1/memory` gathers process RSS through `psutil` when available and CUDA allocator stats through `torch.cuda` when available, returning `error` dictionaries on unavailable platforms instead of failing the whole route.
- Compatibility surfaces to continue treating conservatively: all `/sdapi/v1/*` listing field names, dynamic options/flags schemas, create/train `info` response strings, queue locks on refresh endpoints, and permissive memory error dictionaries.

### Validation log
- `PYTHONPYCACHEPREFIX=/tmp/gb10-a1111-pycompile-pass28 python3 -m py_compile modules/api/api.py modules/api/models.py tests/test_api_training_contract.py` - passed.
- `python3 -m pytest -q tests/test_api_training_contract.py tests/test_api_extension_item_contract.py tests/test_api_progress_contract.py` - passed: 4 passed, with the existing pytest config warning `Unknown config option: base_url`.
- Exact AST duplicate-body scan over `modules/api/api.py` and `modules/api/models.py` reported `0 duplicate nontrivial function body groups` for both files.
- `git diff --check` - passed.

### Next unchecked scope
- More slices are still needed. Recommended next slice: remaining API script/extension metadata and OpenClaw-specific runtime endpoints, especially `/sdapi/v1/scripts`, `/sdapi/v1/script-info`, `/sdapi/v1/extensions`, `/sdapi/v1/openclaw/sdpa-backend`, `/sdapi/v1/openclaw/cuda-graphs`, `/sdapi/v1/openclaw/generation-diagnostics`, `/sdapi/v1/openclaw/precision-map` and `/sdapi/v1/precision-map`, plus adjacent script metadata models and OpenClaw diagnostic helper modules.
## Pass 29 - API script/extension metadata and OpenClaw runtime endpoints (2026-06-20)

### Checked scope
- `modules/api/api.py` route registration and handlers for `/sdapi/v1/scripts`, `/sdapi/v1/script-info`, `/sdapi/v1/extensions`, `/sdapi/v1/openclaw/sdpa-backend`, `/sdapi/v1/openclaw/cuda-graphs`, `/sdapi/v1/openclaw/generation-diagnostics`, `/sdapi/v1/openclaw/precision-map`, and legacy alias `/sdapi/v1/precision-map`.
- `modules/api/api.py` precision-map helpers: `_precision_lora_signature()`, `_precision_selected_coverage()`, `_precision_model_signature()`, `_precision_tensor_info()`, `_precision_name_from_info()`, `_precision_module_source()`, `_precision_layer_kind()`, `_precision_skip_reason()`, and `build_precision_map()`.
- `modules/api/models.py` script and extension metadata schemas: `ScriptsList`, `ScriptArg`, `ScriptInfo`, and `ExtensionItem`.
- Adjacent implementation fanout in `modules/scripts.py` (`create_script_ui_inner()`, script API info construction, and OpenClaw script-arg override handling), `modules/openclaw_generation_diagnostics.py`, `modules/openclaw_cuda_graphs.py`, and `modules/sd_hijack_optimizations.py` SDPA backend helpers.
- Existing focused tests/contracts checked where runnable: `tests/test_api_extension_item_contract.py`, `tests/test_api_progress_contract.py`, and `test/test_openclaw_cuda_graphs.py` compilation/import boundary.

### Findings and fixes
- No safe dead-code deletion or duplicate route implementation was found in this slice.
- Preserved both precision-map routes. `/sdapi/v1/openclaw/precision-map` is the OpenClaw namespaced route, while `/sdapi/v1/precision-map` is a compatibility alias bound to the same implementation; removing either could break existing runtime tooling.
- Preserved the separate OpenClaw runtime endpoint wrappers. The SDPA backend, CUDA graph status/toggle, generation diagnostics, and precision map handlers are small, but each crosses a different runtime subsystem and exposes a distinct public JSON contract.
- Preserved script metadata models and builder. `modules/scripts.py` builds `ScriptArg`/`ScriptInfo` from live Gradio/headless controls after script UI creation; collapsing this into `/scripts` would lose the richer `/script-info` argument metadata contract.
- Preserved extension metadata serialization in `get_extensions_list()`. It intentionally filters to remote-backed extensions and now aligns with the nullable `ExtensionItem` fields fixed in pass 26.
- Confirmed apparent duplicate lines seen during broad terminal output were display artifacts by re-reading exact source slices with line numbers; no source duplicate was present.

### Preserved compatibility/dead-code decisions
- `/sdapi/v1/scripts` remains a compact list of script titles by txt2img/img2img runner, while `/sdapi/v1/script-info` remains the richer per-script argument schema populated during UI bootstrap.
- `ScriptArg.value`, `minimum`, `maximum`, `step`, and `choices` remain permissive `Any`/optional fields because Gradio components and extension controls can expose heterogeneous value types.
- `ScriptInfo.name` remains the lowercased `script.name` value generated by `create_script_ui_inner()`, not a freshly computed title, because API script selection and metadata need the same normalized script identity.
- `get_precision_map()` keeps the queue lock around `build_precision_map()` because it walks `shared.sd_model` and should not race model reload or generation.
- `build_precision_map()` remains in `modules/api/api.py` for now despite its size because it is API-specific response assembly over live model/LoRA/device state; moving it without behavior changes would be cosmetic churn for this audit slice.
- `openclaw_generation_diagnostics.last_generation_diagnostics()` returns a deep copy of the last sample diagnostics, while `Processed.js()` separately carries per-result diagnostics/history. The endpoint and generation response fields are overlapping observability surfaces, not duplicate response builders.
- `openclaw_cuda_graphs.status()` and generation diagnostics intentionally summarize CUDA graph state differently: one exposes live counters/cache state, the other stores before/after/delta snapshots for the last generation.

### Static/dynamic audit map notes
- Script metadata chain: `Api.__init__()` ensures script runners are initialized, `scripts.ScriptRunner.create_script_ui_inner()` builds `script.api_info` from finalized controls, `/sdapi/v1/scripts` lists normalized script names, and `/sdapi/v1/script-info` returns the prebuilt argument metadata objects.
- Extension metadata chain: `/sdapi/v1/extensions` refreshes `extensions.extensions`, calls `read_info_from_repo()` per extension, filters entries without a remote, and returns public metadata fields through `ExtensionItem`.
- OpenClaw runtime control chain: API route -> thin handler -> subsystem status/toggle (`sd_hijack_optimizations`, `openclaw_cuda_graphs`, `openclaw_generation_diagnostics`) with environment defaults applied once during `Api.__init__()`.
- Precision-map chain: API route -> queue lock -> `build_precision_map()` -> live model modules, device dtype flags, quantization stats, LoRA target metadata, and skip-reason helpers from `sd_models`.
- Compatibility surfaces to continue treating conservatively: both precision-map paths, OpenClaw runtime JSON field names, script names and script-info argument field names, extension metadata field names, and queue locking around model-walking diagnostics.

### Validation log
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pycompile-pass29.XXXXXX) python3 -m py_compile modules/api/api.py modules/api/models.py modules/openclaw_generation_diagnostics.py modules/openclaw_cuda_graphs.py modules/sd_hijack_optimizations.py modules/scripts.py tests/test_api_extension_item_contract.py tests/test_api_progress_contract.py test/test_openclaw_cuda_graphs.py` - passed.
- `python3 -m pytest -q tests/test_api_extension_item_contract.py tests/test_api_progress_contract.py` - passed: 2 passed, with the existing pytest config warning `Unknown config option: base_url`.
- `python3 -m pytest -q test/test_openclaw_cuda_graphs.py tests/test_api_extension_item_contract.py tests/test_api_progress_contract.py` - blocked during collection because the GB10 system Python used for this audit does not have `torch` installed (`ModuleNotFoundError: No module named 'torch'`); this matches prior bare-environment limitations and is not caused by pass 29 changes.
- Exact AST duplicate-body scan across `modules/api/api.py`, `modules/api/models.py`, `modules/openclaw_generation_diagnostics.py`, `modules/openclaw_cuda_graphs.py`, `modules/sd_hijack_optimizations.py`, and `modules/scripts.py` reported `0 duplicate nontrivial function body groups` in every file.
- `git diff --check` - passed.

### Next unchecked scope
- More slices are still needed. Recommended next slice: remaining script runner and OpenClaw script-argument runtime surfaces outside the API route layer, especially `modules/scripts.py` callback/timing helpers, `ScriptArgsList`/`openclaw_script_args_to_overrides` fanout, OpenClaw denoise-ramp extension integration, and adjacent script lifecycle tests.

## Pass 30 - script runner and OpenClaw script-argument runtime surfaces (2026-06-20)

### Checked scope
- `modules/scripts.py` script lifecycle/runtime helpers: `wrap_call()`, `ScriptRunner.create_script_ui()` / `create_script_ui_inner()`, `setup_ui_for_section()`, `prepare_ui()`, `setup_ui()`, `run()`, callback ordering helpers, `_script_timing_name()`, `_script_args_for()`, `_record_script_timing()`, all timed lifecycle dispatchers (`before_process`, `process`, `process_before_every_sampling`, `before_process_batch`, `after_extra_networks_activate`, `process_batch`, `postprocess`, `postprocess_batch`, `postprocess_batch_list`, `post_sample`, `on_mask_blend`, `postprocess_image`, `postprocess_maskoverlay`, `postprocess_image_after_composite`, `before_hr`, `setup_scrips`), component callbacks, `script()`, `reload_sources()`, `set_named_arg()`, and compatibility alias `reload_scripts`.
- OpenClaw API script-argument fanout in `modules/api/api.py`: `ScriptArgsList`, `script_default_ui_values()`, `init_default_script_args()`, `persist_openclaw_denoise_ramp_args()`, `init_script_args()`, infotext script-arg handoff, and txt2img/img2img propagation of `openclaw_script_args_to_overrides` into processing objects.
- OpenClaw denoise-ramp integration: `extensions/openclaw-denoise-ramp/scripts/openclaw_denoise_ramp.py`, `extensions/openclaw-denoise-ramp/tests/test_openclaw_denoise_ramp.py`, and cross-extension reuse in `extensions/openclaw-multi-sampler/scripts/openclaw_multi_sampler.py` / `tests/test_openclaw_multi_sampler.py`.
- Adjacent script lifecycle fanout: `modules/processing.py` `setup_scripts()` call into the existing `setup_scrips()` compatibility spelling, UI script lookup uses in `modules/ui.py`, direct txt2img/img2img script arg assignment, and extension always-on hook implementations.
- Focused duplicate/reachability checks: grep fanout for script timing and argument override fields, denoise-ramp helper loading, lifecycle hook definitions, `setup_scrips` callers, and exact AST duplicate-body scan over the pass-30 target files.

### Findings and fixes
- Consolidated duplicated timed lifecycle wrapper logic in `modules/scripts.py`. The repeated per-hook pattern for script-arg lookup, `time.perf_counter()` timing, `_record_script_timing()`, and error reporting is now centralized in private `ScriptRunner._run_timed_script_hook()`.
- Kept all public `ScriptRunner` lifecycle method names and signatures intact. Each hook still calls the same script method, with the same positional object arguments, script UI args, keyword args, callback ordering, timing accumulation, and exception swallowing/reporting behavior.
- Corrected two stale copy/paste error-report labels as part of the consolidation: `on_mask_blend` and `postprocess_maskoverlay` now report their actual hook names via the shared helper instead of stale `post_sample` / `postprocess_image` labels.
- No safe deletion was found for `ScriptArgsList` or `openclaw_script_args_to_overrides`. `ScriptArgsList` is the API-side list subclass that carries the override metadata while still behaving as a plain script-args list/tuple source for A1111 processing, and `_script_args_for()` is the runtime consumer that prevents truncating always-on extension payloads whose controls exceed the default-args bootstrap range.
- No safe deletion or dedupe was found in the denoise-ramp plumbing. API persistence keeps the hidden always-on ramp delta sticky across requests, while multi-sampler's `_load_denoise_ramp_func()` intentionally reuses the ramp helper only if the denoise-ramp extension was already loaded, avoiding fallback imports with sampler monkeypatch side effects.

### Preserved compatibility/dead-code decisions
- `setup_scrips()` remains misspelled because `modules.processing.StableDiffusionProcessing.setup_scripts()` calls that existing method; renaming would require an alias and would be compatibility churn for this audit slice.
- `reload_scripts = load_scripts` remains a compatibility alias for external/plugin callers even though local references are indirect and sparse.
- `Script.describe()` remains as the upstream/public script API stub marked unused; deleting it could break third-party script subclasses or callers that reflect over the base script interface.
- Component-specific callbacks (`on_before_component`, `on_after_component`, `before_component`, `after_component`) remain separate from timed generation hooks because they run during UI construction and use `OnComponent` callback lists plus all-script dispatch instead of always-on generation script args.
- `set_named_arg()` remains a public helper for tuple/list script-arg mutation by script name and elem id; its fuzzy matching behavior is not duplicated by the API `init_script_args()` path.
- Denoise-ramp API persistence remains title-based for the hidden always-on script because it deliberately targets the extension's public script title and avoids importing or depending on the extension module from API bootstrap.

### Static/dynamic audit map notes
- UI script chain: `load_scripts()` discovers script classes, `ScriptRunner.initialize_scripts()` instantiates and categorizes visible/always-on scripts, `setup_ui()` builds controls and API metadata, and component callbacks remain UI-construction hooks.
- Generation script chain: `StableDiffusionProcessing.script_args` setter triggers `setup_scripts()` once scripts and args are present; processing then calls the appropriate `ScriptRunner` lifecycle methods, which dispatch ordered always-on callbacks through `_run_timed_script_hook()` and record OpenClaw script timing metadata on the processing object.
- API script-arg chain: API defaults are copied into `ScriptArgsList`, selectable script args and always-on args are overlaid, over-length always-on payloads set `openclaw_script_args_to_overrides`, txt2img/img2img attach those overrides to `p`, and `_script_args_for()` expands the slice end only for that target script id.
- Denoise-ramp chain: API always-on args can persist the hidden delta default, the denoise-ramp script sets `p.openclaw_denoise_step_delta`, patched k-diffusion img2img sampling marks ramp context and calls `ramp_sigmas_for_img2img()`, and multi-sampler reuses that helper only when the extension module is already loaded.
- Compatibility surfaces to continue treating conservatively: all `Script` base hook names, `ScriptRunner` public hook names, `setup_scrips` spelling, script arg vector positions, `openclaw_script_args_to_overrides`, `openclaw_script_timings` JSON shape, denoise-ramp hidden always-on title/arg position, and script loader extension boundaries.

### Validation log
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pycompile-pass30.XXXXXX) python3 -m py_compile modules/scripts.py modules/api/api.py extensions/openclaw-denoise-ramp/scripts/openclaw_denoise_ramp.py extensions/openclaw-multi-sampler/scripts/openclaw_multi_sampler.py extensions/openclaw-denoise-ramp/tests/test_openclaw_denoise_ramp.py extensions/openclaw-multi-sampler/tests/test_openclaw_multi_sampler.py` - passed.
- `python3 -m pytest -q extensions/openclaw-denoise-ramp/tests/test_openclaw_denoise_ramp.py extensions/openclaw-multi-sampler/tests/test_openclaw_multi_sampler.py` - blocked during collection in the GB10 bare system Python because `torch` is not installed (`ModuleNotFoundError: No module named 'torch'`), matching the prior pass-29 environment limitation.
- Exact AST duplicate-body scan across `modules/scripts.py`, `modules/api/api.py`, `extensions/openclaw-denoise-ramp/scripts/openclaw_denoise_ramp.py`, and `extensions/openclaw-multi-sampler/scripts/openclaw_multi_sampler.py` reported `0 duplicate nontrivial function body groups` in every file.
- `git diff --check` - passed.

### Next unchecked scope
- More slices are still needed. Recommended next slice: postprocessing script runner and script-argument surfaces outside the main generation runner, especially `modules/scripts_postprocessing.py`, postprocessing script arg dict/list conversion, extras API/UI postprocessing fanout, postprocessing tests, and any duplicated timing/error-wrapper patterns between postprocessing and generation script runners.


## Pass 31 - postprocessing script runner and extras fanout (2026-06-20)

### Checked scope
- `modules/scripts_postprocessing.py` function/class surface: `PostprocessedImageSharedInfo`, `PostprocessedImage.__init__()`, `get_suffix()`, `create_copy()`, `ScriptPostprocessing` public stubs/element-id helpers, postprocessing `wrap_call()`, and `ScriptPostprocessingRunner` initialization, UI creation, ordering/filtering, run, arg-vector creation, and `image_changed()` dispatch.
- `modules/postprocessing.py`: `combine_caption()`, `run_postprocessing()` including upload/batch-directory/single-image fanout, output saving/caption merge, `run_postprocessing_webui()`, and legacy/API `run_extras()` argument dict-to-list bridge.
- Extras API surface in `modules/api/api.py`: `setUpscalers()`, `decode_extras_batch_images()`, `Api.extras_single_image_api()`, and `Api.extras_batch_images_api()`.
- Focused postprocessing tests/contracts: `test/test_postprocessing_script_args.py`, `test/test_postprocessing_api_defaults.py`, `tests/test_postprocessing_caption_contract.py`, and live API smoke coverage in `test/test_extras.py`.
- Adjacent generation runner comparison: `modules/scripts.py` `wrap_call()`, `ScriptRunner._script_args_for()`, `_record_script_timing()`, and `_run_timed_script_hook()` from pass 30, checked only to decide whether postprocessing timing/error wrappers could be safely shared.

### Findings and fixes
- Consolidated duplicated extras API queue/run fanout in `modules/api/api.py`. New private `Api._run_extras()` owns the common locked `postprocessing.run_extras(..., input_dir="", output_dir="", save_output=False, **reqDict)` call used by both single-image and batch-image extras endpoints.
- Updated `extras_single_image_api()` and `extras_batch_images_api()` to perform only endpoint-specific decoding/response shaping around the shared helper.
- Updated the AST-based focused test loader in `test/test_postprocessing_api_defaults.py` so extracted extras endpoint methods include `_run_extras()` and continue testing the real helper path.
- No safe dead-code deletion was found in the postprocessing runner. The base `ScriptPostprocessing` stubs and element-id helpers are public extension API; `PostprocessedImage` fields are mutated by extension scripts; `run_postprocessing_webui()` is the UI task wrapper that preserves the queued task signature.
- No safe shared timing/error wrapper was added between generation and postprocessing runners. Generation script lifecycle dispatch uses positional `p.script_args`, records `openclaw_script_timings`, and reports tracebacks with `errors.report`; postprocessing scripts use named UI-control dictionaries, direct `process_firstpass()`/`process()` extension calls, `shared.state.job`, and a UI-facing `errors.display()` wrapper for `ui()` construction.

### Preserved compatibility/dead-code decisions
- Preserved `ScriptPostprocessing.ui()`, `process()`, `process_firstpass()`, and `image_changed()` no-op stubs as subclass extension hooks.
- Preserved `ScriptPostprocessing.extra_only`, `main_ui_only`, `order`, `group`, `args_from`, `args_to`, `controls`, and `tab_name` fields because they are populated/read across UI setup, ordering, filtering, and extension-facing element IDs.
- Preserved dict-to-list conversion in `ScriptPostprocessingRunner.create_args_for_run()` as distinct from generation API script args: postprocessing API extras passes named per-script dictionaries, fills UI defaults from controls, and then runs the same positional vector the UI runner expects.
- Preserved the two-phase postprocessing script run (`process_firstpass()` over all selected scripts, then `process()` over primary plus extra images) because extra image fanout and `disable_processing` semantics are postprocessing-specific.
- Preserved `setUpscalers()` despite its legacy camelCase name because it is the current request-normalization bridge from API model field names (`upscaler_1`/`upscaler_2`) to `run_extras()` parameters (`extras_upscaler_1`/`extras_upscaler_2`) and enforces API image return behavior.
- Preserved `decode_extras_batch_images()` corrupt-image skipping behavior because focused tests cover the tolerant batch contract.

### Static/dynamic audit map notes
- Extras API chain: endpoint request model -> `setUpscalers()` normalization -> endpoint-specific image decode -> `_run_extras()` queue-locked call -> `postprocessing.run_extras()` legacy dict assembly -> `scripts.scripts_postproc.create_args_for_run()` -> `run_postprocessing()` -> `scripts.scripts_postproc.run()`.
- Postprocessing args chain: `setup_ui()` assigns each script `args_from`/`args_to` and ordered controls; `create_args_for_run()` creates a sparse default-filled list from current control values plus explicit named overrides; `run()` slices by each script's arg range and converts back to a named `process_args` dict for extension hooks.
- Extra-image chain: each script can append PIL images or `PostprocessedImage` instances to `single_image.extra_images`; runner normalizes PIL extras with `create_copy()`, appends them to the current all-images list, and clears consumed extras before assigning `pp.extra_images` after all scripts finish.
- Output chain: `run_postprocessing()` reads/generates infotext, saves primary plus extra images when requested, merges caption sidecars via `combine_caption()`, and returns gallery outputs only when mode/toggle requires it.
- Compatibility surfaces to continue treating conservatively: extras endpoint JSON field names, `setUpscalers()` name, `run_extras()` signature, `run_postprocessing_webui(id_task, *args, **kwargs)`, postprocessing script hook names, arg vector positions, `PostprocessedImage` mutable fields, and postprocessing operation/disable option names.

### Validation log
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pycompile-pass31.XXXXXX) python3 -m py_compile modules/scripts_postprocessing.py modules/postprocessing.py modules/api/api.py test/test_postprocessing_script_args.py test/test_postprocessing_api_defaults.py tests/test_postprocessing_caption_contract.py` - passed.
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pytest-pass31.XXXXXX) python3 -m pytest -q test/test_postprocessing_script_args.py test/test_postprocessing_api_defaults.py tests/test_postprocessing_caption_contract.py` - passed: 14 passed, with the existing pytest config warning `Unknown config option: base_url`.
- Exact AST duplicate-body scan across `modules/scripts_postprocessing.py`, `modules/postprocessing.py`, `modules/api/api.py`, and `modules/scripts.py` reported `0 duplicate nontrivial function body groups` in every file after the extras API helper extraction.
- `git diff --check` - passed.

### Next unchecked scope
- More slices are still needed. Recommended next slice: image saving/metadata sidecar and infotext duplication surfaces adjacent to postprocessing outputs, especially `modules/images.py` save/info helpers, caption/sidecar call sites, PNG info propagation in generation vs extras, and focused image-save tests/contracts.


## Pass 32 - image save metadata sidecars and extras infotext propagation (2026-06-20)

### Checked scope
- `modules/images.py` image save/info surface: `save_image_with_geninfo()`, `geninfo_to_exif_bytes()`, `save_image()`, nested `_atomically_save_image()`, `read_info_from_image()`, `image_data()`, and adjacent filename/save sequencing behavior relevant to metadata sidecars.
- `modules/postprocessing.py` output surface: `combine_caption()`, `run_postprocessing()` image loading, inherited PNG info extraction, extras infotext assembly, returned-image info propagation, `images.save_image()` call, caption sidecar read/merge/write, and `run_extras()` save-output bridge.
- Infotext-adjacent helpers/callers: `modules/infotext_utils.py` parse/paste helpers at a reachability level, `modules/extras.py` PNG info display, `modules/ui_common.py` gallery save-to-files path, `modules/ui_extra_networks.py` and `modules/ui_extra_networks_user_metadata.py` preview metadata preservation.
- Focused tests/contracts: `test/test_images_save.py`, `tests/test_postprocessing_caption_contract.py`, `tests/test_save_serialization_contract.py`, `test/test_extras.py`, and postprocessing API default stubs that pin extras caption/pnginfo interactions.
- Focused grep/AST checks for `save_image_with_geninfo`, `geninfo_to_exif_bytes`, `save_image`, `read_info_from_image`, `image_data`, `.txt` sidecar writes, caption sidecars, `existing_pnginfo`, `pnginfo_section_name`, and exact duplicate nontrivial function bodies across the save/metadata caller set.

### Findings and fixes
- No safe source-code remediation was found in this slice; this is a ledger-only checkpoint.
- `save_image_with_geninfo()` remains active as the shared low-level embedder for normal saves plus extra-network preview replacement. Its format branches are not dead: PNG preserves text chunks, JPEG/WebP use EXIF insertion after save, AVIF passes EXIF bytes at save time, GIF uses `comment`, and the fallback covers other registered Pillow extensions.
- `geninfo_to_exif_bytes()` remains active via image saving and API encode paths. Pass 24 already consolidated the EXIF byte construction; this slice intentionally did not rework it.
- `save_image()` remains the correct high-level owner of filename generation, callback mutation, atomic temp-save/replace policy, 4chan downscale export, `already_saved_as`, and optional `.txt` infotext sidecar output. The `.txt` sidecar write is not duplicated with extras caption output because it records generation/extras infotext when `opts.save_txt` is enabled, while postprocessing captions are extension-generated captions merged under `postprocessing_existing_caption_action`.
- `read_info_from_image()` and `image_data()` are reachable through PNG info display, img2img infotext import, API png-info, extra-network metadata/preview paths, and text/image upload parsing. No safe deletion was identified.
- Postprocessing PNG-info propagation looks repetitive but has separate effects: `existing_pnginfo["parameters"] = parameters` preserves source generation parameters after `read_info_from_image()` pops them; `pp.image.info["postprocessing"] = infotext` feeds returned gallery image metadata; `images.save_image(... pnginfo_section_name="extras", existing_info=existing_pnginfo)` embeds the extras infotext into saved files. Collapsing these without a broader contract change would risk API/UI metadata behavior.
- Caption sidecar merge logic is currently localized to postprocessing output and covered by `combine_caption()` contract tests; extracting a one-call wrapper would be cosmetic churn rather than a dead-code/duplication fix.

### Preserved compatibility/dead-code decisions
- Preserved `existing_info` mutation behavior in `save_image()` and `save_image_with_geninfo()` because callback/plugin callers receive and may mutate the same PNG info dictionary before save.
- Preserved `pnginfo_section_name` flexibility because generation uses `parameters`, extras uses `extras`, and low-level preview callers rely on the default section.
- Preserved `save_txt` sidecar semantics and returned `txt_fullfn` because UI save-to-files appends the `.txt` artifact to downloads and focused image-save tests assert the path follows callback-rewritten/truncated/exported filenames.
- Preserved postprocessing's `postprocessing` image-info key separately from saved `extras` PNG section. The first is for returned PIL/gallery metadata; the second is the saved-file infotext section.
- Preserved preview save paths in extra networks even though there are two similar preview-save closures; one is explicitly marked backwards-compatible UI glue and both perform path/permission/UI refresh work around the shared `save_image_with_geninfo()` helper.
- Preserved `image_data()`'s fallback text decode path because upload/paste surfaces can pass either an image byte stream with embedded metadata or a short plain-text infotext payload.

### Static/dynamic audit map notes
- Generation save chain: processing/UI save caller -> `images.save_image()` -> before-save callback can rewrite image/filename/pnginfo -> `_atomically_save_image()` -> `save_image_with_geninfo()` -> optional `.txt` sidecar -> image-saved callback.
- Extras save chain: `run_postprocessing()` reads inherited source metadata -> scripts mutate `PostprocessedImage.info` and optional `caption` -> returned PIL image gets `postprocessing` metadata when PNG info is enabled -> `save_image()` embeds inherited source metadata plus `extras` infotext -> optional generation/extras `.txt` sidecar -> optional caption sidecar merge overwrites/keeps/prepends/appends according to user option.
- PNG-info read chain: image upload/API/preview callers -> `read_info_from_image()` extracts canonical generation parameters from PNG `parameters`, EXIF user comment, GIF comment, or NovelAI fields and returns remaining non-ignored metadata for display or preservation.
- Compatibility surfaces to continue treating conservatively: `ImageSaveParams` callback mutation of filename/pnginfo, `already_saved_as`, returned `txt_fullfn`, `parameters`/`extras`/`postprocessing` metadata keys, `postprocessing_existing_caption_action` values, extra-network preview save callbacks, and API `/sdapi/v1/png-info` output shape.

### Validation log
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pycompile-pass32.XXXXXX) python3 -m py_compile modules/images.py modules/postprocessing.py modules/infotext_utils.py modules/extras.py modules/ui_common.py modules/ui_extra_networks.py modules/ui_extra_networks_user_metadata.py test/test_images_save.py tests/test_postprocessing_caption_contract.py tests/test_save_serialization_contract.py test/test_postprocessing_api_defaults.py` - passed.
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pytest-pass32.XXXXXX) python3 -m pytest -q test/test_images_save.py tests/test_postprocessing_caption_contract.py tests/test_save_serialization_contract.py test/test_postprocessing_api_defaults.py` - blocked by the GB10 bare system Python dependency set after 16 passes: `test_postprocessing_runner_order_override_preserves_script_defaults` imports the full `modules.scripts_postprocessing` stack and failed on `ModuleNotFoundError: No module named fastapi`; extension preload also reported the existing `torch` absence.
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pytest-pass32-focused.XXXXXX) python3 -m pytest -q test/test_images_save.py tests/test_postprocessing_caption_contract.py tests/test_save_serialization_contract.py` - passed: 9 passed, with the existing pytest config warning `Unknown config option: base_url`.
- Exact AST duplicate-body scan across `modules/images.py`, `modules/postprocessing.py`, `modules/infotext_utils.py`, `modules/ui_common.py`, `modules/extras.py`, `modules/ui_extra_networks.py`, and `modules/ui_extra_networks_user_metadata.py` reported `0 duplicate nontrivial function body groups` in every file.
- `git diff --check` - passed.

### Next unchecked scope
- More slices are still needed. Recommended next slice: PNG-info/API infotext import/export and paste-parameter surfaces outside raw saving, especially `modules/api/api.py` png-info/encode/decode helpers, `modules/generation_parameters_copypaste.py`, img2img infotext import, and any duplicated image/base64/infotext parsing contracts between API and UI paths.

## Pass 33 - PNG-info API and paste-parameter import surfaces (2026-06-20)

### Checked scope
- `modules/api/api.py`: `api_infotext_value_for_field()`, `verify_url()`, `decode_base64_to_image()`, `decode_extras_batch_images()`, `encode_pil_to_base64()`, `Api.apply_infotext()`, `Api.txt2imgapi()`/`img2imgapi()` infotext application, extras image decode reuse, `/sdapi/v1/png-info` `pnginfoapi()`, and `interrogateapi()` image decode.
- `modules/infotext_utils.py` as the active implementation and compatibility alias for old `modules.generation_parameters_copypaste`: `ParamBinding`, `PasteField`, paste-field registration/reset, `image_from_url_text()`, `create_buttons()`, compatibility `bind_buttons()`, `register_paste_params_button()`, `connect_paste_params_buttons()`, `send_image_and_dimensions()`, indexed inpaint infotext helpers, `parse_generation_parameters()`, `create_override_settings_dict()`, `get_override_settings()`, and `connect_paste()`.
- UI PNG-info and paste callers: `modules/ui.py` PNG Info tab send-to buttons, txt2img/img2img paste registration, `modules/ui_common.py` gallery save/paste registration and `update_generation_info()`, `modules/extras.py` `run_pnginfo()`, `modules/txt2img.py` first-pass image infotext path, and `modules/img2img.py` batch `use_png_info` import path.
- Compatibility and reachability checks for `modules.generation_parameters_copypaste`, `image_from_url_text()`, `decode_base64_to_image()`, `encode_pil_to_base64()`, `create_buttons()`, `bind_buttons()`, `register_paste_params_button()`, `connect_paste()`, and all parse-generation-parameters call sites.
- Focused duplicate-body scan across `modules`, `test`, and `tests`, filtered for duplicates touching the API/infotext/UI PNG-info scope.

### Findings and fixes
- No safe source-code remediation was found in this slice; this is a ledger-only checkpoint.
- The apparent duplicate `image_from_url_text(x)` call in `send_image_and_dimensions()` was a display artifact from an earlier combined `sed` output. A numbered source read and blame confirmed the working file contains only one decode call.
- `decode_base64_to_image()` and `image_from_url_text()` overlap conceptually but are not safe consolidation targets. The API helper accepts HTTP/HTTPS URLs behind API request policy, generic `data:image/*` payloads, and raw base64, and reports `HTTPException` details used by extras batch tolerant skipping. The UI helper accepts Gradio file/list payloads, validates temporary-file paths through `ui_tempdir.check_tmp_file()`, handles gallery payload shape, and returns PIL images directly for UI callbacks.
- `encode_pil_to_base64()` and `images.save_image_with_geninfo()`/`geninfo_to_exif_bytes()` share metadata-writing concepts, but API response encoding is an in-memory response contract keyed by `opts.samples_format`, while save paths own filenames, callbacks, atomic writes, sidecars, and `already_saved_as`. Only the shared EXIF byte construction is already centralized in `images.geninfo_to_exif_bytes()`.
- API `apply_infotext()` and UI `connect_paste()` intentionally share `infotext_utils.parse_generation_parameters()` and paste-field metadata but differ in output contracts: API mutates unset pydantic request fields, carries override settings, and maps script controls by `script_runner.inputs`; UI returns Gradio updates, supports prompt-history fallback, and triggers recalculate JavaScript.
- `/sdapi/v1/png-info` and UI `run_pnginfo()` intentionally differ. Both read metadata through `images.read_info_from_image()`, but the API returns structured `info`, `items`, and parsed `parameters` with `infotext_pasted_callback()`, while the UI returns HTML display plus hidden generation text for Send-to buttons.

### Preserved compatibility/dead-code decisions
- Preserved `sys.modules['modules.generation_parameters_copypaste'] = sys.modules[__name__]` because old extensions may still import the historical module name even though the file no longer exists.
- Preserved `bind_buttons()` despite no first-party call sites because it is explicitly documented as the old compatibility wrapper around `register_paste_params_button()`.
- Preserved `create_buttons()` and `register_paste_params_button()` because current PNG Info, txt2img/img2img, output-panel, and extras UI surfaces use them to assemble Send-to/paste actions.
- Preserved `paste_fields` compatibility writes to `modules.ui.txt2img_paste_fields` and `modules.ui.img2img_paste_fields` for extension/public surface compatibility.
- Preserved indexed inpaint infotext helpers because img2img paste fields use them to map textual infotext labels into UI/API enum/boolean values.
- Preserved `create_override_settings_dict()` and `get_override_settings()` as distinct helpers: the former converts UI multiselect strings back into processing override dictionaries, while the latter derives non-default override candidates from parsed infotext for UI/API paste behavior.

### Static/dynamic audit map notes
- API infotext chain: request `infotext` -> `Api.apply_infotext()` -> `infotext_utils.parse_generation_parameters()` -> paste fields registered during UI setup -> unset API request fields and override settings -> optional script arg extraction by component identity.
- UI paste chain: `add_paste_fields()` registers tab fields -> `register_paste_params_button()` records source/destination bindings -> `connect_paste_params_buttons()` wires image transfer, text parsing, tab switching, and output-panel field copy -> `connect_paste()` returns Gradio updates and override-setting dropdown choices.
- PNG-info chain: uploaded/API image -> `images.read_info_from_image()` -> UI `extras.run_pnginfo()` for HTML/hidden infotext or API `pnginfoapi()` for structured JSON and parsed parameters.
- Img2img batch PNG-info import chain: source image or parallel `png_info_dir` image -> `images.read_info_from_image()` -> `parse_generation_parameters()` -> whitelisted `png_info_props` -> prompt/negative prompt/seed/CFG/sampler/steps/checkpoint override mutation.
- Compatibility surfaces to continue treating conservatively: `modules.generation_parameters_copypaste` alias, paste-field tuple/PasteField shape, tab names (`txt2img`, `img2img`, `inpaint`, `extras`), API `/sdapi/v1/png-info` response fields, `decode_base64_to_image()` error details, Gradio gallery/file payload handling, and `infotext_pasted_callback()` invocation points.

### Validation log
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pycompile-pass33.XXXXXX) python3 -m py_compile modules/api/api.py modules/infotext_utils.py modules/extras.py modules/img2img.py modules/txt2img.py modules/ui.py modules/ui_common.py modules/ui_postprocessing.py modules/ui_extra_networks.py modules/ui_extra_networks_user_metadata.py modules/api/models.py` - passed.
- Exact AST duplicate-body scan across `modules`, `test`, and `tests`, filtered for duplicates touching `modules/api/api.py`, `modules/infotext_utils.py`, `modules/img2img.py`, `modules/ui.py`, `modules/ui_common.py`, and `modules/extras.py`, printed no duplicate nontrivial function body groups.
- `git diff --check` - passed.

### Next unchecked scope
- More slices are still needed. Recommended next slice: infotext parsing/backcompat internals and option-setting override maps, especially deeper `modules/infotext_utils.py` parsing branches after the paste/API entrypoints, `modules/infotext_versions.py`, prompt/style extraction interactions, and focused parser/backcompat tests or fixtures.

## Pass 34 - infotext parser backcompat internals and option override maps (2026-06-20)

### Checked scope
- `modules/infotext_utils.py`: parser internals after UI/API paste entrypoints, including prompt/negative-prompt line splitting, final-parameter regex handling, quoted value and image-size expansion, style extraction and hires prompt/negative prompt style interaction, old hires-fix restoration, defaulted inpaint/hires/scheduler/RNG/VAE/FP8/MXFP8/refiner fields, prompt-emphasis backcompat default, `infotext_versions.backcompat()` handoff, skip-field pruning, override-settings helpers, and `connect_paste()` override-setting fanout.
- `modules/infotext_versions.py`: version parsing and all current backcompat gates (`Old prompt editing timelines`, `Pad conds v0`, `Downcast alphas_cumprod`, `Refiner switch by sampling steps`) plus `v180_hr_styles` use from the parser.
- Prompt/style extraction interactions: `shared.prompt_styles.extract_styles_from_prompt()` call sites, `Hires prompt` / `Hires negative prompt` version gate, `Styles array` paste field behavior in `modules/ui.py`, and API/UI paste consumers.
- Option override mapping surfaces: dynamic `OptionInfo(..., infotext=...)` labels in `modules/shared_options.py`, legacy `infotext_to_setting_name_mapping`, `create_override_settings_dict()`, `get_override_settings()`, and the extra-options extension's legacy-map inversion.
- Focused tests/contracts: `test/test_infotext_api_mappings.py` and adjacent parser/infotext references in `tests/test_save_serialization_contract.py`, `modules/txt2img.py`, `modules/img2img.py`, `modules/api/api.py`, `modules/shared_items.py`, and `modules/processing_scripts/seed.py`.

### Findings and fix decision
- Consolidated the duplicated dynamic-plus-legacy infotext option mapping construction into `infotext_setting_name_mapping()` and reused it from both `create_override_settings_dict()` and `get_override_settings()`.
- Added a focused contract test proving the shared mapping keeps both current `OptionInfo.infotext` entries and the legacy `infotext_to_setting_name_mapping` extension hook reachable.
- No safe dead backcompat conversion was found. The version thresholds in `modules/infotext_versions.py` still map historical infotexts into current processing settings, and the `v180_hr_styles` threshold remains live in hires prompt/style extraction.
- No safe parser branch deletion was found. The apparently default-heavy branches in `parse_generation_parameters()` are user-data-facing replay compatibility for old infotexts, omitted fields, and UI/API paste behavior.
- The empty `infotext_to_setting_name_mapping` list remains an intentional legacy extension hook. It is still read by the built-in extra-options extension and now also by the shared helper used by both override paths.

### Static/dynamic audit map notes
- Parser chain: UI/API/text upload infotext -> `parse_generation_parameters()` -> prompt/style extraction and historical defaults -> `infotext_versions.backcompat()` -> skip-field pruning -> UI paste/API request mutation/override settings.
- Override chain: UI override multiselect strings use `create_override_settings_dict()` while parsed infotext paste/API override suggestions use `get_override_settings()`; both now share the same setting-label map while preserving their different output contracts.
- Compatibility surfaces to continue treating conservatively: infotext field labels, missing-field defaults, version threshold constants, `modules.generation_parameters_copypaste` alias, `infotext_to_setting_name_mapping`, `OptionInfo.infotext`, style extraction option modes, and `Hires prompt` / `Hires negative prompt` backcompat behavior.

### Validation log
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pycompile-pass34.XXXXXX) python3 -m py_compile modules/infotext_utils.py modules/infotext_versions.py test/test_infotext_api_mappings.py modules/shared_options.py modules/shared_items.py modules/api/api.py modules/txt2img.py modules/img2img.py modules/ui.py` - passed.
- `python3 -m pytest -q test/test_infotext_api_mappings.py` - passed: 8 passed, with the existing pytest config warning `Unknown config option: base_url`.
- Exact AST duplicate-body scan across `modules/infotext_utils.py`, `modules/infotext_versions.py`, and `test/test_infotext_api_mappings.py` reported `0 duplicate nontrivial function body groups` in every file.
- `git diff --check` - passed.

### Next unchecked scope
- More slices are still needed. Recommended next slice: parser/backcompat consumers just outside infotext_utils, especially `modules/shared_items.py` infotext-name enumeration, `modules/shared_options.py` `OptionInfo.infotext` coverage and duplicates, UI settings/extra-options setting surfacing, and focused checks for stale or duplicate infotext labels across `modules/ui.py`, processing scripts, and built-in extensions.

## Pass 35 - infotext parser/backcompat consumers outside infotext_utils (2026-06-20)

### Checked scope
- `modules/shared_items.py`: `get_infotext_names()` enumeration of `OptionInfo.infotext` labels and registered paste-field names for the `infotext_skip_pasting` settings dropdown.
- `modules/shared_options.py`: current `OptionInfo(..., infotext=...)` labels and settings UI option coverage, especially labels used by override settings and extra-options paste fields.
- `modules/ui.py`: txt2img/img2img paste-field construction, override-settings dropdown surfacing, PNG-info send-to bindings, script infotext field splicing, and inpaint backcompat paste helpers.
- `modules/processing_scripts/seed.py`, `modules/processing_scripts/sampler.py`, and `modules/processing_scripts/refiner.py`: built-in processing script `infotext_fields` and labels surfaced through `scripts.scripts_*` into paste behavior.
- `modules/scripts.py`: script-runner accumulation of `infotext_fields`/`paste_field_names`, selectable script visibility paste fields, and public script field compatibility surfaces.
- Built-in extensions with infotext/backcompat consumers: `extensions-builtin/extra-options-section/scripts/extra_options_section.py`, `extensions-builtin/hypertile/scripts/hypertile_script.py`, `extensions-builtin/soft-inpainting/scripts/soft_inpainting.py`, and Lora infotext callback references.

### Findings and fixes
- Fixed a stale duplicate mapping consumer in `extensions-builtin/extra-options-section/scripts/extra_options_section.py`. The extra-options script still inverted `infotext_utils.infotext_to_setting_name_mapping` directly, but that list is now only the legacy backcompat hook and is empty in-tree. It now inverts `infotext_utils.infotext_setting_name_mapping()`, so extra-options can surface paste fields for current `OptionInfo.infotext` settings while still preserving legacy extension-added mappings.
- No duplicate `OptionInfo.infotext` labels were found in the scanned core and built-in option providers. The exact AST scan covered `modules/shared_options.py` and `extensions-builtin/hypertile/scripts/hypertile_script.py` and reported 51 labels, 0 duplicates.
- No safe stale wrapper removal was found in `modules/shared_items.get_infotext_names()`. It intentionally combines current settings labels and currently registered paste fields so `infotext_skip_pasting` can show both option-backed labels and UI/script labels such as seed, sampler, inpaint, refiner, and extension fields.
- No safe consolidation was found for txt2img/img2img paste-field lists in `modules/ui.py`. They share common labels, but the tab-specific fields differ by hires-fix, img2img-only image CFG/inpaint fields, source image handling, API names, and registered target tabs (`txt2img`, `img2img`, `inpaint`).
- Preserved processing-script and built-in extension `infotext_fields`/`paste_field_names` surfaces because they are the public contract by which scripts participate in paste/send-to behavior.

### Static/dynamic audit map notes
- Settings label chain: `OptionInfo.infotext` and legacy `infotext_to_setting_name_mapping` -> `infotext_setting_name_mapping()` -> override settings and extra-options paste field lookup.
- Skip-pasting choices chain: `shared_items.get_infotext_names()` -> current `shared.opts.data_labels` plus registered `infotext_utils.paste_fields` -> `shared_options.infotext_skip_pasting` dropdown choices.
- UI paste chain outside infotext_utils: `modules/ui.py` tab paste lists and processing/built-in script `infotext_fields` -> `parameters_copypaste.add_paste_fields()` -> paste/send-to behavior and override dropdown fanout.
- Extension compatibility chain: script classes expose `infotext_fields`/`paste_field_names`; `modules/scripts.py` aggregates them and `modules/ui_common.py` passes names for output-panel Send-to buttons.

### Validation log
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pycompile-pass35.XXXXXX) python3 -m py_compile modules/shared_items.py modules/shared_options.py modules/ui.py modules/processing_scripts/seed.py modules/processing_scripts/sampler.py modules/processing_scripts/refiner.py modules/scripts.py modules/infotext_utils.py extensions-builtin/extra-options-section/scripts/extra_options_section.py extensions-builtin/hypertile/scripts/hypertile_script.py extensions-builtin/soft-inpainting/scripts/soft_inpainting.py test/test_infotext_api_mappings.py` - passed.
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pytest-pass35.XXXXXX) python3 -m pytest -q test/test_infotext_api_mappings.py` - passed: 8 passed, with the existing pytest config warning `Unknown config option: base_url`.
- Exact AST `OptionInfo.infotext` duplicate-label scan across `modules/shared_options.py` and `extensions-builtin/hypertile/scripts/hypertile_script.py` reported `OptionInfo infotext labels scanned: 51; duplicates: 0`.
- Exact AST duplicate-body scan across `modules/shared_items.py`, `modules/shared_options.py`, `modules/ui.py`, `modules/processing_scripts/seed.py`, `modules/processing_scripts/sampler.py`, `modules/processing_scripts/refiner.py`, `modules/scripts.py`, `extensions-builtin/extra-options-section/scripts/extra_options_section.py`, `extensions-builtin/hypertile/scripts/hypertile_script.py`, and `extensions-builtin/soft-inpainting/scripts/soft_inpainting.py` reported `duplicate nontrivial function body groups: 0`.
- `git diff --check` - passed.

### Next unchecked scope
- More slices are still needed. Recommended next slice: script/paste-field API consumers and infotext callbacks beyond the core UI path, especially `modules/ui_common.py` output-panel Send-to behavior, `modules/script_callbacks.py` infotext callback registration/callers, Lora `infotext_pasted` compatibility handling, and any duplicated paste-field filtering or script-arg mapping between UI/API callback paths.


## Pass 36 - script/paste-field API consumers and infotext callbacks beyond core UI (2026-06-20)

### Checked scope
- `modules/ui_common.py`: `create_output_panel()` Send-to button construction, output-panel `ParamBinding` registration, `paste_field_names` derivation from script runners, gallery save inputs, and `update_generation_info()` infotext selection behavior.
- `modules/infotext_utils.py`: `ParamBinding`, compatibility `bind_buttons()`, `register_paste_params_button()`, `connect_paste_params_buttons()`, source-tab paste-field copy wiring, text paste `connect_paste()`, and the `modules.generation_parameters_copypaste` alias.
- `modules/script_callbacks.py`: `callbacks_infotext_pasted`, `infotext_pasted_callback()`, `on_infotext_pasted()`, ordered callback preservation, and caller reachability.
- `modules/api/api.py`: API infotext application into script args via `apply_infotext()`, default script-arg initialization, always-on script-arg override/persistence handling, and `/sdapi/v1/png-info` callback invocation.
- Lora callback consumers: `extensions-builtin/Lora/scripts/lora_script.py` Lora hash alias replacement callback and `extensions-builtin/Lora/networks.py` AddNet compatibility callback.
- Adjacent tests: existing `test/test_infotext_api_mappings.py` plus new focused coverage for source-tab paste binding field filtering.

### Findings and fixes
- Consolidated duplicated source/destination paste-field filtering in `connect_paste_params_buttons()` into one local `paste_fields_with_names()` helper. This removes the duplicated inline list-comprehension logic used by output-panel Send-to source-tab copy bindings while keeping the public `ParamBinding`/paste-field data shape unchanged.
- Added `test/test_infotext_paste_bindings.py` to lock the source-tab Send-to filtering contract: prompt, seed, and steps are copied when allowed, while unrelated fields are excluded from both inputs and outputs through the same helper path.
- No safe dead callback removal was found. `on_infotext_pasted()`/`infotext_pasted_callback()` remain public extension hooks and are reached from both UI paste (`connect_paste()`) and API PNG-info parsing (`pnginfoapi()`).
- No safe Lora infotext alias/backcompat removal was found. `lora_script.infotext_pasted()` rewrites `<lora:alias:...>` tokens from `Lora hashes`, while `networks.infotext_pasted()` converts historical AddNet parameters unless the AddNet extension already exposes corresponding infotext fields.
- No safe API/UI script-arg mapping consolidation was found. UI paste returns Gradio updates and recalculate JS; API paste mutates unset pydantic request fields, override settings, and script args by component identity, so their shared contract remains the paste-field metadata and parsed infotext dictionary rather than a single caller helper.

### Static/dynamic audit map notes
- Output-panel Send-to chain: `ui_common.create_output_panel()` builds tab buttons -> resolves script runner `paste_field_names` for txt2img/img2img output panels -> registers `ParamBinding` with `source_tabname="txt2img"` for txt2img output copy and gallery image source for all send targets -> `infotext_utils.connect_paste_params_buttons()` wires image copy, source-tab field copy, and tab switch JS.
- Infotext callback chain: UI text paste -> `parse_generation_parameters()` -> `script_callbacks.infotext_pasted_callback()` -> registered Lora/AddNet/extension callbacks can mutate parsed params before field application. API PNG-info uses the same callback after image metadata parse so callback consumers see non-UI imports too.
- Lora callback chain: `extensions-builtin/Lora/scripts/lora_script.py` registers both `networks.infotext_pasted` and its local hash-alias callback; the former preserves AddNet-era infotext compatibility, while the latter resolves short hashes to current on-disk aliases.
- Compatibility surfaces to continue treating conservatively: `on_infotext_pasted()`, `callbacks_infotext_pasted`, callback ordering/user priority, `ParamBinding` constructor parameters, paste-field tuple/PasteField shape, `modules.generation_parameters_copypaste` alias, Lora `Lora hashes`, historical AddNet infotext keys, and script runner `infotext_fields`/`paste_field_names`.

### Validation log
- `PYTHONPYCACHEPREFIX=/tmp/a1111-pyc-s36 pytest -q test/test_infotext_paste_bindings.py test/test_infotext_api_mappings.py` - passed: 9 passed, with the existing pytest config warning `Unknown config option: base_url`.
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pycompile-pass36.XXXXXX) python3 -m py_compile modules/ui_common.py modules/infotext_utils.py modules/script_callbacks.py modules/api/api.py extensions-builtin/Lora/scripts/lora_script.py extensions-builtin/Lora/networks.py test/test_infotext_paste_bindings.py test/test_infotext_api_mappings.py` - passed.
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pytest-pass36.XXXXXX) python3 -m pytest -q test/test_infotext_paste_bindings.py test/test_infotext_api_mappings.py` - passed: 9 passed, with the existing pytest config warning `Unknown config option: base_url`.
- Exact AST duplicate-body scan across `modules/ui_common.py`, `modules/infotext_utils.py`, `modules/script_callbacks.py`, `modules/api/api.py`, `extensions-builtin/Lora/scripts/lora_script.py`, `extensions-builtin/Lora/networks.py`, `test/test_infotext_paste_bindings.py`, and `test/test_infotext_api_mappings.py` reported `duplicate nontrivial function body groups: 0`.
- `git diff --check` - passed.

### Next unchecked scope
- More slices are still needed. Recommended next slice: callback and paste-field consumers adjacent to script lifecycle/runtime behavior, especially `modules/scripts.py` script field registration/exposure, processing script infotext-field producers after paste consumers, and extension callback ordering/priority surfaces outside `infotext_pasted`.

## Pass 37 - script lifecycle callback ordering and paste-field producers (2026-06-20)

### Checked scope
- `modules/scripts.py`: `Script` public field/callback registration surfaces, selectable-script UI visibility paste fields, script runner aggregation of `infotext_fields`/`paste_field_names`, script-specific ordered callback construction/cache, before/after component elem-id callback collection, runtime script hook dispatch, and reload/body-only compatibility alias.
- `modules/script_callbacks.py`: global callback naming, extension metadata ordering, user priority sorting, ordered callback cache, callback enumeration for settings UI, public `on_*` registration helpers, removal helpers, reversed unload/UI callbacks, and all callback dispatch functions outside `infotext_pasted`.
- `modules/shared_items.py`: callback priority option enumeration for global callbacks and script callbacks, including unsorted callback list creation for settings UI.
- Built-in script field producers: `modules/processing_scripts/seed.py`, `modules/processing_scripts/sampler.py`, `modules/processing_scripts/refiner.py`, `scripts/xyz_grid.py`, `scripts/img2imgalt.py`, `scripts/sd_upscale.py`, `scripts/loopback.py`, `scripts/prompt_matrix.py`, and `scripts/prompts_from_file.py`.
- Adjacent tests: callback/metadata and infotext-result alignment tests in `tests/test_extensions_metadata_contract.py`, `tests/test_processing_auxiliary_infotext_alignment.py`, `tests/test_save_serialization_contract.py`, plus existing infotext paste/API mapping tests from earlier passes.

### Findings and fixes
- Consolidated duplicated component elem-id callback registration in `modules/scripts.py`. `Script.on_before_component()` and `Script.on_after_component()` now share `_add_component_callback()` while preserving the public attributes, tuple shape, and registration order.
- Consolidated duplicated before/after elem-id callback collection in `ScriptRunner.apply_on_before_component_callbacks()` with one local `register_callbacks()` helper using the same `elem_id -> [(callback, script)]` structure and same per-script order.
- No safe script field registration/exposure removal was found. `infotext_fields` and `paste_field_names` remain public script/extension contracts consumed by UI paste fields, output-panel Send-to buttons, API script-arg paste, and the settings skip-pasting list.
- No safe processing-script infotext producer deletion was found. Seed/sampler/refiner producers map parsed infotext into live UI/API fields, XYZ grid writes grid-specific script parameters, and the auxiliary scripts maintain live per-result infotext alignment covered by tests.
- No safe callback compatibility alias removal was found. `reload_scripts = load_scripts`, `topological_sort = util.topological_sort`, public `on_*` registration helpers, and callback maps are extension-facing compatibility surfaces.
- No safe consolidation of global and script callback ordering helpers was made. Both use `script_callbacks.sort_callbacks()`, but global callbacks cache by category while script callbacks must build per-runner categories from current script objects and expose unsorted lists for settings UI without mutating the runtime cache.

### Static/dynamic audit map notes
- Script field chain: script `ui()` assigns `infotext_fields`/`paste_field_names` -> `ScriptRunner.create_script_ui_inner()` aggregates them -> `modules/ui.py` and `modules/ui_common.py` expose them to paste/send-to paths -> `infotext_utils` applies parsed fields or copies source fields.
- Selectable script chain: `ScriptRunner.setup_ui()` creates the script dropdown -> adds `Script` paste field and selectable group visibility fields -> parsed `Script` infotext reopens the selected script group during paste.
- Script callback ordering chain: script method overrides -> `ScriptRunner.create_ordered_callbacks_list()` wraps them as `ScriptCallback` entries with `script_<method>` categories -> `script_callbacks.sort_callbacks()` applies extension metadata and user priority -> runtime `ordered_scripts()` dispatches methods with script args and timing.
- Global callback ordering chain: public `script_callbacks.on_*()` registration -> `callback_map` -> `ordered_callbacks()` cache/sort -> dispatch functions; settings UI reads unsorted ordered lists with `enable_user_sort=False` for priority option choices.
- Compatibility surfaces to continue treating conservatively: `Script.infotext_fields`, `Script.paste_field_names`, `Script.on_before_component_elem_id`, `Script.on_after_component_elem_id`, `reload_scripts`, `topological_sort`, callback category names, `callbacks_*` map keys, callback priority option names, and reversed unload/before-UI dispatch order.

### Validation log
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pycompile-pass37.XXXXXX) python3 -m py_compile modules/scripts.py modules/script_callbacks.py modules/shared_items.py modules/processing_scripts/seed.py modules/processing_scripts/sampler.py modules/processing_scripts/refiner.py scripts/xyz_grid.py scripts/img2imgalt.py scripts/sd_upscale.py scripts/loopback.py scripts/prompt_matrix.py scripts/prompts_from_file.py tests/test_extensions_metadata_contract.py tests/test_processing_auxiliary_infotext_alignment.py tests/test_save_serialization_contract.py test/test_infotext_api_mappings.py test/test_infotext_paste_bindings.py` - passed.
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pytest-pass37.XXXXXX) python3 -m pytest -q tests/test_extensions_metadata_contract.py tests/test_processing_auxiliary_infotext_alignment.py tests/test_save_serialization_contract.py test/test_infotext_api_mappings.py test/test_infotext_paste_bindings.py` - passed: 22 passed, with the existing pytest config warning `Unknown config option: base_url`.
- Exact AST duplicate-body scan across `modules/scripts.py`, `modules/script_callbacks.py`, `modules/shared_items.py`, `modules/processing_scripts/seed.py`, `modules/processing_scripts/sampler.py`, `modules/processing_scripts/refiner.py`, `scripts/xyz_grid.py`, `scripts/img2imgalt.py`, `scripts/sd_upscale.py`, `scripts/loopback.py`, `scripts/prompt_matrix.py`, and `scripts/prompts_from_file.py` reported `duplicate nontrivial function body groups: 0`.
- `git diff --check` - passed.

### Next unchecked scope
- More slices are still needed. Recommended next slice: script runtime helpers and API script-arg consumers beyond callback ordering, especially `modules/api/api.py` script argument initialization/defaulting, `modules/scripts.py` `set_named_arg()` and `init_default_script_args()` interactions, always-on script API persistence, and tests around script arg override/infotext paste behavior.

## Pass 38 - API script-arg defaulting and runtime helper consumers (2026-06-20)

### Checked scope
- `modules/api/api.py`: API bootstrap default vectors for txt2img/img2img scripts, `script_default_ui_values()`, `Api.init_default_script_args()`, `Api.init_script_args()`, infotext-provided script arg overlays, always-on script payload validation and sparse-vector extension, OpenClaw Denoise Ramp API persistence, and txt2img/img2img propagation of `openclaw_script_args_to_overrides` into processing objects.
- `modules/scripts.py`: runtime script-arg dispatch through `_script_args_for()` and `_run_timed_script_hook()`, selectable script `run()` arg slicing, `set_named_arg()` tuple/list behavior, script API metadata from `create_script_ui_inner()`, and always-on/selectable script lookup helpers.
- `modules/api/models.py`: txt2img/img2img `script_args` and `alwayson_scripts` request fields plus `ScriptArg`/`ScriptInfo` response models.
- Focused tests: `test/test_api_script_defaults.py` and `test/test_infotext_api_mappings.py` around script default extraction, API infotext conversion, and newly covered sparse script-arg setting.

### Findings and fixes
- Consolidated duplicated sparse script-arg assignment in `modules/api/api.py` into private `_set_script_arg()`. The helper preserves existing list mutation semantics while centralizing the shared extend-with-`None` behavior used by always-on API payloads, OpenClaw Denoise Ramp default persistence, and infotext script-arg overlays.
- Added focused tests in `test/test_api_script_defaults.py` covering both in-range script-arg replacement and extension of sparse API vectors.
- No safe removal of `ScriptRunner.set_named_arg()` was found. It has no in-repo callers in this slice, but it is a public runtime helper on the script runner object, handles both tuple and list script args, and can be used by extensions or dynamic/plugin code by script/control elem_id.
- No safe removal of `openclaw_script_args_to_overrides` was found. The override map is needed when an always-on request sends more args than the script runner's captured `args_to`, and `modules/scripts.py` consumes it in `_script_args_for()` for all timed always-on hook dispatch.
- No safe unification of API script defaulting with UI/script metadata creation was made. API bootstrap must build a non-UI `script_args` vector with position `0` reserved for selectable script index, while UI metadata registration exposes controls, paste fields, and API model descriptions.
- No safe deletion of OpenClaw Denoise Ramp default persistence was found. The persistence intentionally updates the API default vector for that always-on script after explicit API requests, preserving runtime/API behavior across subsequent calls.

### Static/dynamic audit map notes
- API default chain: API constructor ensures script runners are initialized -> `init_default_script_args()` sizes a vector to the max script `args_to` -> position `0` stores selectable script index -> default values come from finalized controls when available via `script_default_ui_values()`.
- Request overlay chain: infotext paste fills `infotext_script_args` by component identity -> `init_script_args()` overlays those indexes, then selectable script request args, then always-on script args -> processing receives `p.script_args` and optional `p.openclaw_script_args_to_overrides`.
- Always-on extension chain: request payload args beyond captured `args_to` extend the vector and record an override keyed by `id(script)` -> `ScriptRunner._script_args_for()` uses the override to avoid truncating hook args.
- Compatibility surfaces to continue treating conservatively: request `script_args` list shape, `alwayson_scripts` dict shape, script arg position `0`, `ScriptArg`/`ScriptInfo` fields, `ScriptRunner.set_named_arg()`, and OpenClaw Denoise Ramp API persistence.

### Validation log
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pycompile-pass38.XXXXXX) python3 -m py_compile modules/api/api.py modules/api/models.py modules/scripts.py test/test_api_script_defaults.py test/test_infotext_api_mappings.py` - passed.
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pytest-pass38.XXXXXX) python3 -m pytest -q test/test_api_script_defaults.py test/test_infotext_api_mappings.py` - passed: 12 passed, with the existing pytest config warning `Unknown config option: base_url`.
- Exact AST duplicate-body scan across `modules/api/api.py`, `modules/api/models.py`, `modules/scripts.py`, `test/test_api_script_defaults.py`, and `test/test_infotext_api_mappings.py` reported one pre-existing duplicate group in `modules/scripts.py` (`postprocess_image()` and `postprocess_maskoverlay()`), outside this slice's script-arg/defaulting scope.
- `git diff --check` - passed.

### Next unchecked scope
- More slices are still needed. Recommended next slice: adjacent runtime script dispatch wrappers and postprocess helper duplication, especially `modules/scripts.py` postprocess hook wrappers (`postprocess_image`, `postprocess_maskoverlay`, image-after-composite, batch/list/sample/mask hooks), their argument object types, and extension compatibility constraints around hook method names.

## Pass 39 - runtime script postprocess hook wrappers (2026-06-20)

### Checked scope
- `modules/scripts.py`: callback payload classes (`MaskBlendArgs`, `PostSampleArgs`, `PostprocessImageArgs`, `PostProcessMaskOverlayArgs`, `PostprocessBatchListArgs`), `Script` base postprocess hook stubs, `ScriptRunner` timed dispatch helpers, `postprocess_batch()`, `postprocess_batch_list()`, `post_sample()`, `on_mask_blend()`, `postprocess_image()`, `postprocess_maskoverlay()`, and `postprocess_image_after_composite()`.
- `modules/processing.py`: generation call sites for post-sample, batch tensor/list hooks, per-image postprocess hook, mask-overlay hook, after-composite hook, and inpainting mask-blend payload construction.
- `modules/sd_samplers_cfg_denoiser.py`: per-step/final mask-blend payload construction.
- `modules/scripts_auto_postprocessing.py`: main-UI postprocessing bridge override of `postprocess_image()`.
- Installed extension samples overriding adjacent hooks, especially `extensions/sd-webui-incantations/scripts/*` `postprocess_batch()` wrappers.

### Findings and fixes
- Consolidated the duplicated runtime dispatch bodies for `ScriptRunner.postprocess_image()`, `postprocess_maskoverlay()`, and `postprocess_image_after_composite()` into private `_run_postprocess_arg_hook()`. The public method names/signatures remain intact, and the helper still dispatches through `_run_timed_script_hook()` so script args, timing, and hook-specific error labels are preserved.
- No safe removal was found for callback payload object types. They are live mutable compatibility payloads constructed by `processing.py` and `sd_samplers_cfg_denoiser.py` and then passed to extension hooks that may mutate images, masks, overlays, samples, or batch lists.
- No safe removal was found for `Script` base hook stubs. They are the extension ABI used by `ScriptRunner.create_ordered_callbacks_list()` to detect overrides; removing or aliasing them would risk dynamic extension compatibility.
- No safe consolidation was made for `postprocess_batch()`, `postprocess_batch_list()`, `post_sample()`, or `on_mask_blend()`. They have different call shapes (`images=` keyword, positional payloads, extra `**kwargs`) and preserving exact hook argument behavior is more important than introducing a broader generic wrapper.
- No safe consolidation was made with `ScriptPostprocessingForMainUI.postprocess_image()`. That method adapts extras/postprocessing scripts into generation-tab always-on hooks and mutates `PostprocessImageArgs.image` plus generation info, so it remains a distinct bridge.

### Static/dynamic audit map notes
- Per-image postprocess chain: `processing.py` creates `PostprocessImageArgs(image)` -> `ScriptRunner.postprocess_image()` -> extension hook may replace `pp.image` -> processing continues with the possibly replaced image.
- Mask overlay chain: `processing.py` creates `PostProcessMaskOverlayArgs(index, mask_for_overlay, overlay_image)` -> `ScriptRunner.postprocess_maskoverlay()` -> extension hook may replace `mask_for_overlay`/`overlay_image` before color correction and overlay composition.
- After-composite chain: after `apply_overlay()`, `processing.py` creates a new `PostprocessImageArgs(image)` -> `postprocess_image_after_composite()` -> extension hook operates on the final full image.
- Batch/list/sample/mask chains: `postprocess_batch()` receives the 4D tensor as `images=` plus batch kwargs; `postprocess_batch_list()` receives mutable `PostprocessBatchListArgs.images`; `post_sample()` receives latent/sample payload before VAE decode; `on_mask_blend()` receives denoiser/sigma-aware per-step and final blend payloads.
- Compatibility surfaces to continue treating conservatively: script hook method names/signatures, payload class names/fields/mutability, hook-specific timing keys/error labels, `ScriptRunner.callback_names`, `ScriptPostprocessingForMainUI.postprocess_image()`, and extension override detection by comparing subclass methods with `Script` base methods.

### Validation log
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pycompile-pass39.XXXXXX) python3 -m py_compile modules/scripts.py modules/processing.py modules/scripts_auto_postprocessing.py` - passed.
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pytest-pass39.XXXXXX) python3 -m pytest -q test/test_api_script_defaults.py test/test_infotext_api_mappings.py` - passed: 12 passed, with the existing pytest config warning `Unknown config option: base_url`.
- Exact AST duplicate-body scan across `modules/scripts.py`, `modules/processing.py`, and `modules/scripts_auto_postprocessing.py` reported `duplicate nontrivial function body groups: 0`.
- `git diff --check` - passed.
- Ad hoc runtime smoke under system `python3` was attempted but not used as a gate because direct app import failed on missing non-test runtime dependencies (`torch`, then `fastapi`); no repo `venv/bin/python` was present for rerun in this shell context.

### Next unchecked scope
- More slices are still needed. Recommended next slice: adjacent processing per-image output/mask save flow after script hooks, especially duplicated mask/overlay save-return branches in `modules/processing.py`, image/info mutation around `postprocess_image_after_composite()`, and safe helper boundaries for mask return/save behavior without changing output ordering or infotext metadata.

## Pass 40 - processing per-image output and mask return/save flow (2026-06-20)

### Checked scope
- `modules/processing.py`: `process_images_inner()` per-image output loop after `postprocess_image()`, `postprocess_maskoverlay()`, color correction, `apply_overlay()`, and `postprocess_image_after_composite()`; final sample save, PNG infotext mutation, output image/infotext list append order, returned/saved mask branch, returned/saved mask-composite branch, grid prepend behavior, and `Processed` construction.
- `modules/processing.py`: adjacent img2img inpainting setup around `image_mask`, `mask_for_overlay`, `overlay_images`, full-res crop fallback, latent mask setup, and `MaskBlendArgs` payload construction in `StableDiffusionProcessingImg2Img.sample()`.
- Adjacent contracts: `tests/test_processing_auxiliary_infotext_alignment.py` and `tests/test_image_mask_fix_contract.py`.

### Findings and fixes
- Consolidated duplicated result-list append behavior in the per-image output loop with a local `append_output_image()` helper scoped after `text = infotext(i)`. The main image, returned mask, and returned mask composite now share the same `infotexts.append(text)` plus `output_images.append(...)` path without recomputing infotext or changing return ordering.
- Updated the auxiliary infotext alignment source-contract test so it asserts the helper owns aligned append behavior and both returned auxiliary branches call it.
- No safe consolidation was made for the mask and mask-composite save branches themselves. They intentionally differ in image construction, option flags (`return_mask`/`save_mask` versus `return_mask_composite`/`save_mask_composite`), and suffixes (`-mask` versus `-mask-composite`).
- No safe changes were made to `postprocess_image_after_composite()` placement, `image.info["parameters"]` mutation, save timing, or grid prepend logic. These are behavior-sensitive output/API/UI contracts.
- No safe removal was found in the adjacent inpainting mask setup. `image_mask`, `latent_mask`, `mask_for_overlay`, `overlay_images`, and `paste_to` feed different downstream conditioning, overlay, cache, and return/save behavior.

### Static/dynamic audit map notes
- Per-image output order remains: save final sample if enabled -> compute one `text = infotext(i)` -> append final image/infotext -> write PNG `parameters` metadata on the final image when enabled -> optionally save/return mask -> optionally save/return mask composite.
- Returned mask and mask composite still reuse the already-computed sample `text`, keeping `Processed.images` and `Processed.infotexts` lengths aligned for auxiliary images.
- Mask composite construction still uses `original_denoised_image` from `apply_overlay()` and a resized `mask_for_overlay`; plain mask construction still uses `mask_for_overlay.convert(RGB)`.
- Inpainting setup remains deliberately separate from return/save behavior: setup prepares conditioning/cache/overlay state, while the per-image loop decides returned/saved auxiliary images after script hooks can mutate mask/overlay payloads.
- Compatibility surfaces to continue treating conservatively: `opts.return_mask`, `opts.save_mask`, `opts.return_mask_composite`, `opts.save_mask_composite`, saved suffixes, `image.info["parameters"]`, `Processed.index_of_first_image`, `mask_for_overlay`, `overlay_images`, `paste_to`, and script hook timing around compositing.

### Validation log
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pycompile-pass40.XXXXXX) python3 -m py_compile modules/processing.py tests/test_processing_auxiliary_infotext_alignment.py` - passed.
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pytest-pass40.XXXXXX) python3 -m pytest -q tests/test_processing_auxiliary_infotext_alignment.py tests/test_image_mask_fix_contract.py` - passed: 8 passed, with the existing pytest config warning `Unknown config option: base_url`.
- Exact AST duplicate-body scan across `modules/processing.py`, `tests/test_processing_auxiliary_infotext_alignment.py`, and `tests/test_image_mask_fix_contract.py` still reports four pre-existing nontrivial branch-shape duplicate groups in `modules/processing.py` at lines `(167, 184)`, `(1601, 1693)`, `(2031, 1585, 1494)`, and `(1395, 1407)`; none are in the remediated per-image mask return/save append path.
- `git diff --check` - passed.

### Next unchecked scope
- More slices are still needed. Recommended next slice: nearby `modules/processing.py` branch-shape duplication outside the just-checked output loop, especially txt2img/img2img image conditioning branches, hires resize target calculations, VAE encoder metadata branches, and hires/img2img init mask/latent setup where behavior-sensitive duplication may or may not be safely extractable.


## Pass 41 - processing conditioning, VAE metadata, and branch-shape audit (2026-06-20)

### Checked scope
- `modules/processing.py`: `txt2img_image_conditioning()`, `StableDiffusionProcessing.txt2img_image_conditioning()`, `StableDiffusionProcessing.img2img_image_conditioning()`, hires resize target calculations in `StableDiffusionProcessingTxt2Img.calculate_target_resolution()`, firstpass/hires VAE encode paths in `sample()` and `sample_hr_pass()`, hires conditioning activation in `sample_hr_pass()` and `setup_conds()`, and img2img init mask/latent/cache setup in `StableDiffusionProcessingImg2Img.init()`.
- Adjacent tests/contracts: `tests/test_processing_auxiliary_infotext_alignment.py`, `test/test_openclaw_cache_invalidation.py`, and `tests/test_image_mask_fix_contract.py`.

### Findings and fixes
- Consolidated duplicated txt2img full-mask inpainting conditioning construction into `_full_masked_image_conditioning()`. The hybrid/concat and SDXL-inpaint branches still choose the same branches as before, and the helper preserves the all-0.5 image tensor, VAE approximation selection, fake full mask padding, and output dtype conversion.
- Consolidated repeated `VAE Encoder` metadata assignment into `StableDiffusionProcessing.add_vae_encoder_generation_param()`, used by firstpass image VAE encode, hires image-resize encode, and img2img init encode paths without changing when the metadata is emitted.
- Added a source-contract test for the two helper boundaries so future duplicate-remediation passes preserve the sensitive tensor construction and metadata-call counts.
- No safe extraction was made for hires resize target calculations. The duplicated width/height ratio assignments are small arithmetic branches coupled to `hr_resize_x`, `hr_resize_y`, `target_w`, `target_h`, and truncate calculations; extracting them would add indirection with little risk reduction.
- No safe extraction was made for hires/img2img init mask and latent setup. `image_mask`, `latent_mask`, `mask_for_overlay`, `overlay_images`, `paste_to`, cache keys, inpainting fill metadata, and script-visible mask behavior share branch shapes but feed distinct downstream contracts.
- No dead branch variables were proven. Public/dynamic model fields and extension-observable generation params remain conservative.

### Static/dynamic audit map notes
- Txt2img conditioning branch map remains: hybrid/concat -> full masked inpaint conditioning; crossattn-adm -> UnCLIP zero ADM conditioning; SDXL inpaint under non-hybrid/non-UnCLIP -> full masked inpaint conditioning; otherwise -> dummy 5-channel zero conditioning.
- VAE Encoder metadata remains tied to `opts.sd_vae_encode_method != 'Full'` and is recorded before every image-to-latent encode path that uses `approximation_indexes.get(opts.sd_vae_encode_method)` in this scope.
- Hires resize target branch map remains deliberately local: no explicit resize uses `hr_scale`; one zero dimension preserves source aspect ratio; two explicit dimensions choose the source-ratio-constrained side and compute latent truncation.
- Img2img init mask setup remains deliberately local because overlay composition, latent fill modes, cache restore/store payloads, color correction, and mask return/save behavior all depend on the exact state assembled there.

### Validation log
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pycompile-pass41.XXXXXX) python3 -m py_compile modules/processing.py tests/test_processing_auxiliary_infotext_alignment.py` - passed.
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pytest-pass41-source.XXXXXX) python3 -m pytest -q tests/test_processing_auxiliary_infotext_alignment.py` - passed: 8 passed, with the existing pytest config warning `Unknown config option: base_url`.
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pytest-pass41.XXXXXX) python3 -m pytest -q tests/test_processing_auxiliary_infotext_alignment.py test/test_openclaw_cache_invalidation.py tests/test_image_mask_fix_contract.py` - blocked during collection because this shell's `/usr/bin/python3` lacks `numpy` (`ModuleNotFoundError: No module named 'numpy'`); no repo `venv`/`.venv` was present on GB10 for rerun.
- Exact AST duplicate-body scan across `modules/processing.py` and `tests/test_processing_auxiliary_infotext_alignment.py` reported `duplicate nontrivial function body groups: 0`.
- `git diff --check` - passed.

### Next unchecked scope
- More slices are still needed. Recommended next slice: continue through adjacent `modules/processing.py` generation-state/cache helpers outside the branch-shape groups, especially cache key/clone/restore/store helper duplication, `Processed` metadata assembly, and remaining conditioning/cache lifecycle boundaries where public extension hooks should stay conservative.

## Pass 42 - processing generation-state cache helpers and Processed metadata assembly (2026-06-20)

### Checked scope
- `modules/processing.py`: generic cache helpers (`_clone_cache_value`, `_cache_stats`, `_reset_cache_stats`, hit/miss recorders), conditioning cache lifecycle in `get_conds_with_caching()`/`setup_conds()`, img2img init-cache status, bypass/hit/miss stat propagation, cache bypass reasons, cache key construction, restore/store payload cloning, init-cache lifecycle in `StableDiffusionProcessingImg2Img.init()`/`close()`, `Processed.__init__()`, `Processed.js()`, and `create_infotext()` generation parameter assembly around cache-visible metadata.
- Adjacent consumers/contracts: API `processed_js_with_image_paths()` OpenClaw stat propagation, txt2img/img2img `processed.js()` callers, `test/test_openclaw_cache_invalidation.py`, and `tests/test_processing_auxiliary_infotext_alignment.py` source contracts.

### Findings and fixes
- Consolidated duplicated img2img init-cache payload attribute bookkeeping into `_IMG2IMG_INIT_CACHE_ATTRS`. Cache restore and store now share the same authoritative attribute tuple while preserving clone-on-read/write semantics for tensors, PIL images, numpy arrays, lists, tuples, and dicts.
- Consolidated duplicated img2img init-cache stat snapshot updates into `_snapshot_img2img_init_cache_stats()`. Bypass, hit, and miss paths still update the same `last_hit`, `cached`, `bypass_reason`, hit/miss counters, compute seconds, and per-processing-object snapshot fields as before.
- Added a focused source-contract test covering the shared payload and stat-helper boundaries so later audit slices preserve this behavior-sensitive cache state.
- No safe removal was found in `_clone_cache_value()`. Its recursive tensor/PIL/numpy/container handling is needed by cache restore/store and generation-param cloning boundaries to avoid mutable cached payload aliasing.
- No safe removal or broad extraction was made for img2img init-cache key construction. The tuple is intentionally explicit and behavior-sensitive, covering image/mask fingerprints, model/VAE/sampler identity, resize/mask/inpaint options, dtype/device, background color, and effective inpainting mask weight.
- No safe consolidation was made between conditioning cache and img2img init cache lifecycles. They share small stats primitives, but their invalidation keys, payloads, bypass rules, and public clear/status behavior are different.
- No safe Processed metadata field removal was found. `Processed.__init__()`, `Processed.js()`, API `processed_js_with_image_paths()`, txt2img/img2img UI consumers, and OpenClaw diagnostics/stat fields intentionally expose overlapping but not identical metadata surfaces; `openclaw_cond_cache_stats` is added by the API wrapper while `openclaw_img2img_init_cache_stats` also remains in `Processed.js()` for UI/API compatibility.

### Static/dynamic audit map notes
- Img2img init-cache lifecycle remains: `init()` builds request metadata and `cache_extra_generation_params` -> `_img2img_init_cache_key()` may bypass and snapshot stats -> `_restore_img2img_init_cache()` clones cached payload back to the processing object and records hit -> cold path computes init latent/conditioning -> `_store_img2img_init_cache()` clones payload into the class cache and records miss -> `close()` clears when persistent caching is disabled.
- Cache payload attributes now have one shared list: `init_latent`, `image_conditioning`, `mask`, `nmask`, `mask_for_overlay`, `overlay_images`, `color_corrections`, and `paste_to`; non-attribute cache metadata remains `is_using_inpainting_conditioning` and cloned `extra_generation_params`.
- Conditioning cache lifecycle remains separate: `cached_params()` keys prompt schedules/model/options/LoRA signatures, `get_conds_with_caching()` checks one or more cond caches, records simple per-processing stats, and recomputes under autocast on miss.
- Processed metadata assembly remains conservative: constructor copies processing state into serializable/public fields; `Processed.js()` emits the historic UI JSON surface; API response wrapping adds image paths and OpenClaw timing/cache fields.
- Compatibility surfaces to continue treating conservatively: `StableDiffusionProcessing.cached_*` class cache shapes, `clear_img2img_init_cache()`, `img2img_init_cache_status()`, `openclaw_*_cache_stats` field names, `Processed.js()` JSON keys, `processed_js_with_image_paths()` extras, generation param names, and explicit cache key tuple ordering.

### Validation log
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pycompile-pass42.XXXXXX) python3 -m py_compile modules/processing.py tests/test_processing_auxiliary_infotext_alignment.py` - passed.
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pytest-pass42-source.XXXXXX) python3 -m pytest -q tests/test_processing_auxiliary_infotext_alignment.py` - passed: 9 passed, with the existing pytest config warning `Unknown config option: base_url`.
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pytest-pass42-cache.XXXXXX) python3 -m pytest -q test/test_openclaw_cache_invalidation.py` - blocked during collection because this shell's `/usr/bin/python3` lacks `numpy` (`ModuleNotFoundError: No module named 'numpy'`); no repo `venv`/`.venv` was present for rerun in this shell context.
- Exact AST duplicate-body scan across `modules/processing.py` and `tests/test_processing_auxiliary_infotext_alignment.py` reported `duplicate nontrivial function body groups: 0` for both files.
- `git diff --check` - passed.

### Next unchecked scope
- More slices are still needed. Recommended next slice: continue through adjacent `modules/processing.py` process loop state assembly after cache helpers, especially prompt/seed/batch list setup, comments/infotext construction around `Processed(...)`, grid and `index_of_first_image` behavior, and callback-visible generation diagnostics/history propagation.

## Pass 43 - processing process-loop state assembly and Processed output boundaries (2026-06-20)

### Checked scope
- `modules/processing.py`: `process_images_inner()` prompt/seed/subseed list initialization, per-batch prompt/negative/seed/subseed slicing, script callback-visible prompt reset before `postprocess_batch_list()`, params.txt infotext construction through `Processed(p, [])`, model hijack comment propagation, OpenClaw generation diagnostics `before_sample()`/`after_sample()` history propagation, per-image infotext append behavior, grid prepend/save behavior, `index_of_first_image`, and final `Processed(...)` assembly.
- `modules/processing.py`: `Processed.__init__()`, `Processed.js()`, and `Processed.infotext()` field/JSON boundaries as consumers of process-loop state.
- `modules/openclaw_generation_diagnostics.py`: request summary, diagnostics history append, and last-diagnostics copy behavior adjacent to the process-loop sample wrapper.
- Adjacent consumers/contracts: `modules/ui_common.py` save-selected/index handling, `modules/txt2img.py` gallery infotext replacement, `modules/img2img.py` batch-result infotext accumulation, `modules/processing_scripts/seed.py` infotext index lookup, `tests/test_processing_auxiliary_infotext_alignment.py`, and `tests/test_save_serialization_contract.py`.

### Findings and fixes
- Consolidated duplicated per-batch slice-boundary arithmetic in `process_images_inner()` into `_batch_slice_range()`. The first batch setup still assigns prompts, negative prompts, seeds, and subseeds before script callbacks; the later reset after `postprocess_batch()` still restores only prompts and negative prompts before `postprocess_batch_list()`, preserving callback-visible seed/subseed state.
- Added a focused source-contract test asserting both helper use sites and guarding that the post-`postprocess_batch()` reset does not start resetting seed/subseed lists.
- No safe removal or broad consolidation was found for prompt/seed/all-list setup. `setup_prompts()`, list seeds, scalar seeds with subseed-strength behavior, and prompt/negative prompt length checks are user/API/script-visible generation contracts.
- No safe consolidation was made for params.txt infotext generation or final `Processed(...)` construction. The empty-image `Processed(p, [])` path intentionally reuses the public infotext method after script batch processing can mutate generation params, while the final `Processed` object carries returned-image ordering, infotexts, first-image index, and public JSON fields.
- No safe changes were made to grid insertion or `index_of_first_image`. UI save-selected and save-all paths depend on the grid prefix being counted as non-sample only when `opts.return_grid` inserts it into returned images.
- No safe removal or consolidation was found for OpenClaw diagnostics/history propagation. `after_sample()` updates the current processing object, appends history, and stores a deepcopy for the diagnostics API; `Processed` mirrors those fields into the existing UI/API JSON surface.

### Static/dynamic audit map notes
- Batch-list lifecycle remains: `setup_prompts()` creates full prompt lists -> seed/subseed lists are derived from scalar/list seeds -> each iteration slices all four lists with the shared bounds -> scripts may see and mutate current batch state -> after `postprocess_batch()` only prompt lists are restored before list-style postprocessing.
- Infotext lifecycle remains: params history writes after `process_batch()` via `Processed(p, []).infotext()` -> per-image saves and returned images use the per-batch `infotext()` closure -> returned auxiliary images reuse the already-computed sample text -> fallback interrupted/no-output runs still append one `Processed(p, []).infotext()`.
- Grid lifecycle remains: grid may be created when return/save options allow it and image count is sufficient -> returned grid prepends one main-prompt infotext and sets `index_of_first_image = 1` -> saved-only grid does not alter returned image ordering or first-image index.
- Diagnostics lifecycle remains: process loop wraps `p.sample()` with `before_sample()`/`after_sample()` in `finally` -> diagnostics fields are attached before `Processed` construction -> `Processed.js()` exposes current diagnostics/history and img2img init-cache stats.
- Compatibility surfaces to continue treating conservatively: `p.prompts`, `p.negative_prompts`, `p.seeds`, `p.subseeds`, `p.all_*` list fields, params.txt timing, model hijack comments, `Processed` constructor fields, `Processed.js()` keys, `infotexts` ordering, `index_of_first_image`, returned grid placement, and OpenClaw diagnostics field names/history shape.

### Validation log
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pycompile-pass43.XXXXXX) python3 -m py_compile modules/processing.py modules/openclaw_generation_diagnostics.py tests/test_processing_auxiliary_infotext_alignment.py` - passed.
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pytest-pass43-source.XXXXXX) python3 -m pytest -q tests/test_processing_auxiliary_infotext_alignment.py` - passed: 10 passed, with the existing pytest config warning `Unknown config option: base_url`.
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pytest-pass43-save.XXXXXX) python3 -m pytest -q tests/test_save_serialization_contract.py` - passed: 3 passed, with the existing pytest config warning `Unknown config option: base_url`.
- Exact AST duplicate-body scan across `modules/processing.py`, `modules/openclaw_generation_diagnostics.py`, and `tests/test_processing_auxiliary_infotext_alignment.py` reported `duplicate nontrivial function body groups: 0`.
- `git diff --check` - passed.

### Next unchecked scope
- More slices are still needed. Recommended next slice: continue through adjacent `modules/processing.py` subclasses after the main loop, especially `StableDiffusionProcessingTxt2Img` hires setup/sample/sample_hr_pass state assembly, duplicated HR prompt/negative prompt/conditioning transitions, firstpass image handling, and script-visible hires metadata boundaries.

## Pass 44 - txt2img hires setup/sample state assembly (2026-06-20)

### Checked scope
- `modules/processing.py`: `StableDiffusionProcessingTxt2Img.__post_init__()`, `calculate_target_resolution()`, `init()`, `sample()`, `sample_hr_pass()`, `close()`, `setup_prompts()`, `calculate_hr_conds()`, `setup_conds()`, `get_conds()`, and `parse_extra_network_prompts()`.
- Adjacent firstpass/hires callers and contracts: `modules/txt2img.py` firstpass-image/upscale entry, `modules/infotext_utils.py` hires prompt/resize restore behavior, `modules/processing_scripts/comments.py` hires prompt mutation, `modules/scripts.py` `before_hr()` hook surface, `modules/sd_samplers_common.py` hires refiner branch, `modules/shared_options.py` hires options, and `extensions/openclaw-clear-cond-cache` HR cond-cache clear/status handling.

### Findings and fixes
- Consolidated duplicated firstpass image conversion in `StableDiffusionProcessingTxt2Img.sample()` into private `_firstpass_image_to_chw_array()`. The signed decoded-sample path still applies `/ 255.0 * 2.0 - 1.0`, and the latent VAE-encode path still uses unsigned `/ 255.0` before tensor/device conversion and `VAE Encoder` metadata emission.
- No safe consolidation was made for hires prompt and negative-prompt metadata callbacks. The nested `get_hr_prompt()` and `get_hr_negative_prompt()` functions are script/infotext-visible callables with different source fields and comparison arguments, and keeping them local preserves the current metadata boundary.
- No safe consolidation was made for `setup_prompts()` HR prompt/negative-prompt expansion. The two branches are mechanically similar, but they populate distinct public fields that built-in scripts can mutate before per-batch parsing.
- No safe consolidation was made for HR conditioning activation in `setup_conds()` and `sample_hr_pass()`. The branches intentionally differ on firstpass-cond reuse, lowvram early calculation, checkpoint switching, extra-network restoration, and script/refiner-visible `is_hr_pass` state.
- No safe removal was found for HR fields (`hr_c`, `hr_uc`, `all_hr_prompts`, `hr_extra_network_data`, `hr_checkpoint_info`, `latent_scale_mode`, truncate fields). They are used across generation, metadata, cache clear/status, scripts, and sampler/refiner transitions.
- The AST duplicate scan for `StableDiffusionProcessingTxt2Img` after the fix reported only repeated `devices.torch_gc()` calls in `sample_hr_pass()`, which are intentional lifecycle barriers around memory-heavy hires stages and were not collapsed.

### Static/dynamic audit map notes
- Firstpass-image hires path remains: `modules/txt2img.py` may set `p.firstpass_image` -> txt2img `sample()` skips first-pass sampling only when `enable_hr` is true -> decoded or latent samples are prepared according to `latent_scale_mode` -> optional HR checkpoint reload -> `sample_hr_pass()`.
- Hires prompt lifecycle remains: `setup_prompts()` expands/stylizes full HR prompt lists -> `parse_extra_network_prompts()` slices current HR batch and parses extra networks -> `calculate_hr_conds()` builds HR conditioning at final resolution with HR sampler total-step scheduling -> `get_conds()` exposes HR conds only during `is_hr_pass`.
- Script-visible HR metadata remains local to `init()`: `Hires prompt` and `Hires negative prompt` are callable generation-param entries that compare final HR prompt text with the first-pass prompt text at infotext creation time.
- Firstpass image handling remains conservative: signed decoded samples are only used when there is no latent upscaler, while latent-upscale mode still encodes the unsigned image tensor through the selected VAE encode approximation before `sample_hr_pass()`.
- Compatibility surfaces to continue treating conservatively: `firstpass_image`, `is_hr_pass`, `before_hr()`, HR prompt fields/lists, HR cond-cache class variables, HR checkpoint/sampler/scheduler generation params, `save_images_before_highres_fix`, output ordering, and exact hires infotext keys.

### Validation log
- `PYTHONPYCACHEPREFIX=/tmp/gb10-a1111-pycompile-pass44 python3 -m py_compile modules/processing.py modules/txt2img.py modules/infotext_utils.py modules/processing_scripts/comments.py extensions/openclaw-clear-cond-cache/scripts/openclaw_clear_cond_cache.py` - passed.
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pytest-pass44-source.XXXXXX) python3 -m pytest -q tests/test_processing_auxiliary_infotext_alignment.py` - passed: 10 passed, with the existing pytest config warning `Unknown config option: base_url`.
- Focused AST duplicate statement scan inside `StableDiffusionProcessingTxt2Img` reported only duplicated `devices.torch_gc()` calls at `sample_hr_pass()` lines 1610 and 1636 after the fix; no remaining safe duplicate firstpass-image conversion was found.
- `git diff --check` - passed.

### Next unchecked scope
- More slices are still needed. Recommended next slice: continue with `StableDiffusionProcessingImg2Img` subclass methods after the txt2img hires class, especially init-image preparation, mask/latent/cache interactions, resize-mode handling, img2img sampler setup/sample state, and duplicated image-to-latent or mask-conditioning transitions adjacent to the already-audited cache helpers.

## Pass 45 - img2img subclass init/mask/sample state assembly (2026-06-20)

### Checked scope
- `modules/processing.py`: `StableDiffusionProcessingImg2Img.__post_init__()`, `mask_blur` property/setter, img2img init-cache public/status helpers, cache bypass/key/restore/store boundaries adjacent to pass 42, `init()` init-image preparation, resize-mode/crop-region handling, mask blur/invert/full-res/overlay setup, latent-mask resize/fill transitions, `image_conditioning` construction, `close()`, `sample()`, mask latent blending, and `get_token_merging_ratio()`.
- Adjacent callers/contracts: `modules/img2img.py` UI/API construction of `StableDiffusionProcessingImg2Img`, `modules/api/api.py` img2img API path and `openclaw_img2img_init_cache_stats` propagation, sampler `sample_img2img()` call shape in `modules/sd_samplers_kdiffusion.py` and `modules/sd_samplers_timesteps.py`, mask blend callback surface in `modules/scripts.py`, `tests/test_processing_auxiliary_infotext_alignment.py`, `test/test_openclaw_cache_invalidation.py`, and `tests/test_image_mask_fix_contract.py`.

### Findings and fixes
- Consolidated duplicated PIL-image-to-CHW float32 conversion into `_image_to_chw_float32_array()`. The txt2img firstpass decoded-sample path still applies signed `[-1, 1]` scaling, the txt2img firstpass VAE encode path still uses unsigned `[0, 1]` scaling, and img2img init-image preparation still appends unsigned CHW arrays after flatten/resize/mask-fill/color-correction setup.
- Removed the now-redundant `StableDiffusionProcessingTxt2Img._firstpass_image_to_chw_array()` private method and routed both firstpass branches through the shared helper.
- Added a focused source-contract assertion for the conversion helper and expected call count in `tests/test_processing_auxiliary_infotext_alignment.py`.
- No safe extraction was made for img2img mask preparation. The similar-looking blur/invert/full-res/non-full-res branches are coupled to `mask_for_overlay`, `overlay_images`, `paste_to`, generation params, blank-mask fallback, mask return/save behavior, and inpaint crop/resize ordering.
- No safe extraction was made for latent mask/cache handling. `image_mask`, `latent_mask`, `repeat_init_latent`, seeded latent noise fill, cache bypass/key payloads, and inpainting conditioning all remain behavior-sensitive and extension/API visible.
- No dead resize-mode variables or sampler setup branches were proven. `resize_mode == 3`, crop-region handling, sampler creation before cache-key construction, `process_before_every_sampling()`, and mask blending are all still active runtime contracts.

### Static/dynamic audit map notes
- Img2img init image lifecycle remains: save optional init image -> flatten transparency with `opts.img2img_background_color` -> resize unless crop/full-res or latent-resize mode says otherwise -> compose overlays before crop -> crop/full-res resize if needed -> optional mask fill -> optional color correction -> shared CHW float32 conversion -> batch/repeat handling -> VAE encode -> optional latent resize -> optional mask latent blending/fill -> image conditioning.
- Mask lifecycle remains conservative: UI/API `mask` is moved to `image_mask` in `__post_init__`; `self.mask` later becomes latent keep-mask, while `self.nmask` becomes latent masked-region mask used by sampler denoiser and final blend callbacks.
- Cache lifecycle remains unchanged from pass 42: masked requests and latent noise fill bypass persistent img2img init cache; unmasked cache keys still include image fingerprints, resize/mask/inpaint settings, sampler conditioning key, VAE/model identity, dtype/device, background color, and effective inpainting mask weight.
- Sample lifecycle remains: RNG noise -> optional initial noise multiplier metadata -> script `process_before_every_sampling()` with init latent/noise/conds -> sampler `sample_img2img()` with cached/precomputed `image_conditioning` -> optional latent blend and `on_mask_blend()` callback -> GC.
- Compatibility surfaces to continue treating conservatively: `image_mask` versus latent `mask`, `mask_for_overlay`, `overlay_images`, `paste_to`, `extra_generation_params` strings, img2img init cache tuple ordering, `openclaw_img2img_init_cache_stats`, `scripts.MaskBlendArgs`, and sampler `image_conditioning` argument semantics.

### Validation log
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pycompile-pass45.XXXXXX) python3 -m py_compile modules/processing.py tests/test_processing_auxiliary_infotext_alignment.py` - passed.
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pytest-pass45-source.XXXXXX) python3 -m pytest -q tests/test_processing_auxiliary_infotext_alignment.py` - passed: 10 passed, with the existing pytest config warning `Unknown config option: base_url`.
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pytest-pass45-cache.XXXXXX) python3 -m pytest -q test/test_openclaw_cache_invalidation.py tests/test_image_mask_fix_contract.py` - blocked during collection because this shell's `/usr/bin/python3` lacks `numpy` (`ModuleNotFoundError: No module named 'numpy'`).
- Exact AST duplicate-body scan across `modules/processing.py` and `tests/test_processing_auxiliary_infotext_alignment.py` reported `duplicate nontrivial function body groups: 0` for both files.
- `git diff --check` - passed.

### Next unchecked scope
- More slices are still needed. Recommended next slice: continue past the processing subclasses into adjacent img2img entry/API assembly, especially `modules/img2img.py` mode-specific init image/mask selection, batch img2img setup, duplicate mask construction between UI modes, API img2img request population, and boundaries where scripts/API compatibility should stay conservative.

## Pass 46 - img2img entry/API assembly (2026-06-20)

### Checked scope
- `modules/img2img.py`: UI `img2img()` mode-specific init-image/mask selection for modes 0-5, inpaint binary mask creation, inpaint-sketch diff-mask construction, upload-mask pass-through, fixed image/mask normalization, scale-by behavior, `StableDiffusionProcessingImg2Img` request assembly, script runner invocation, and batch upload/from-dir setup through `process_batch()`.
- `modules/img2img.py`: `process_batch()` input discovery, upload-vs-directory batch handling, inpaint mask directory matching, PNG-info parameter restoration, output filename override setup, and batch result accumulation/limit behavior.
- `modules/api/api.py`: `img2imgapi()` init image presence validation, mask decode timing relative to task queue registration, infotext application, selectable/always-on script argument setup, sampler/scheduler normalization, pydantic request population, decoded init-image injection, queue/task cleanup boundaries, response timing metadata, and `include_init_images` response mutation.
- `modules/api/models.py`: generated `StableDiffusionImg2ImgProcessingAPI` fields for `init_images`, `mask`, `include_init_images`, script args, `alwayson_scripts`, `force_task_id`, and `infotext`.
- Adjacent tests/contracts: `test/test_img2img.py`, `test/conftest.py`, `tests/test_image_mask_fix_contract.py`, and API/server requirements around img2img request error cleanup.

### Findings and fixes
- Extracted duplicated UI img2img mode image/mask selection from `img2img()` into `_select_img2img_init_image_and_mask()`. This keeps the same mode-specific behavior while separating init-image/mask assembly from request population, scaling, script dispatch, and batch handling.
- Preserved inpaint mode binary mask conversion exactly at the UI mode-selection boundary. API-supplied masks still decode to PIL images in `img2imgapi()` and flow through `StableDiffusionProcessingImg2Img.__post_init__()`/processing mask preparation, so no shared UI/API mask-construction helper was introduced.
- Preserved inpaint-sketch mask construction as a UI-only behavior: it derives a diff mask from the sketch image versus original image, applies mask-alpha brightness, and composites blurred edits before processing. This is not duplicate API behavior.
- Preserved batch upload/from-dir branching. Upload mode intentionally clears output and inpaint mask directories and honors hidden-dir config for PNG-info source, while directory mode enforces `--hide-ui-dir-config` and passes user-selected input/output/mask dirs.
- No dead mode branches were removed. Modes 0-4 are live UI tabs, and mode 5 is batch mode; the default `image = None, mask = None` fallback remains conservative for unexpected mode values and existing assertion/error behavior.
- No API request-population extraction was made. `text2imgapi()` and `img2imgapi()` share sampler/save/script setup patterns, but img2img has behavior-specific early init-image validation, mask decode before queue registration, decoded init-image timing metrics, include-init response mutation, and processing-object `init_images` injection. Collapsing these would be broader than this safe slice.
- No `include_init_images` model/API compatibility cleanup was made. The explicit `args.pop('include_init_images', None)` and response-side mutation are kept because the code already documents pydantic exclude uncertainty and external API clients may depend on the current response shape.

### Static/dynamic audit map notes
- UI img2img lifecycle remains: mode selects raw image/mask -> `images.fix_image()` normalizes both -> optional scale-by reads image dimensions for non-batch requests -> `StableDiffusionProcessingImg2Img` receives `init_images=[image]` and `mask=mask` -> script runner may handle request before `process_images()` fallback.
- Batch img2img lifecycle remains: initial processing object is constructed before batch dispatch, then `process_batch()` replaces `p.init_images` for each input image, optionally sets `p.image_mask` from mask directory matching, restores PNG-info-selected fields, and accumulates/limits results.
- API img2img lifecycle remains: request validates `init_images` before queue registration -> optional mask decodes before queue registration so invalid masks do not leak pending tasks -> pydantic copy carries decoded mask and save/sampler overrides -> init images decode and are assigned to `p.init_images` inside the queue lock -> response optionally strips original init images/mask from returned parameters.
- Compatibility surfaces to continue treating conservatively: `img2img()` positional signature and mode numbers, `process_batch()` source-type behavior, inpaint-sketch mask-alpha semantics, `StableDiffusionImg2ImgProcessingAPI` generated field names/defaults, `include_init_images`, `script_name`/`script_args`, `alwayson_scripts`, `force_task_id`, and img2img API pending-task cleanup ordering.

### Validation log
- `PYTHONPYCACHEPREFIX=/tmp/gb10-a1111-pycompile-pass46 python3 -m py_compile modules/img2img.py modules/api/api.py modules/api/models.py test/test_img2img.py test/conftest.py tests/test_image_mask_fix_contract.py` - passed.
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pytest-pass46-mask.XXXXXX) python3 -m pytest -q tests/test_image_mask_fix_contract.py` - passed: 1 passed, with the existing pytest config warning `Unknown config option: base_url`.
- Exact AST duplicate-body scan across `modules/img2img.py`, `modules/api/api.py`, and `modules/api/models.py` reported no duplicate function bodies.
- `git diff --check` - passed.
- Server-backed `test/test_img2img.py` API tests were inspected but not run because they require a live WebUI/API server fixture.

### Next unchecked scope
- More slices are still needed. Recommended next slice: continue into adjacent API request assembly outside img2img, especially shared txt2img/img2img API helper boundaries (`apply_infotext()`, selectable/always-on script arg population, sampler/scheduler normalization, task queue cleanup), and extras/png-info decode helpers where duplicate request/response population may be safely reducible without breaking public API compatibility.

## Pass 47 - API request helper boundaries outside img2img (2026-06-20)

### Checked scope
- `modules/api/api.py`: `validate_sampler_name()`, `decode_base64_to_image()`, `decode_extras_batch_images()`, `Api.get_selectable_script()`, `Api.get_script()`, `Api.init_default_script_args()`, `Api.persist_openclaw_denoise_ramp_args()`, `Api.init_script_args()`, `Api.apply_infotext()`, `Api.text2imgapi()`, `Api.img2imgapi()`, `_run_extras()`, `extras_single_image_api()`, `extras_batch_images_api()`, and `pnginfoapi()`.
- `modules/api/models.py`: generated txt2img/img2img API request fields for sampler index/name compatibility, script selection, script args, `alwayson_scripts`, `send_images`, `save_images`, `force_task_id`, `infotext`, img2img `init_images`/`mask`, and `include_init_images`.
- Adjacent tests/contracts: `test/test_api_script_defaults.py`, `test/test_infotext_paste_bindings.py`, server-backed `test/test_txt2img.py`, `test/test_img2img.py`, and `test/test_extras.py` API coverage shape.

### Findings and fixes
- Extracted duplicated txt2img/img2img request-preparation logic into `Api._prepare_generation_api_request()`. The helper now owns infotext application, selectable-script lookup, sampler/scheduler normalization, pydantic request copy/save flag population, common API-only field removal, script-arg initialization, script-arg override extraction, and `send_images`/`save_images` handling.
- Kept endpoint-specific behavior at each public API entry point. `img2imgapi()` still validates missing init images before any queue registration, decodes masks before queue registration, decodes init images before queue registration, records img2img API timing fields, injects decoded init images on the processing object inside the queue lock, and applies `include_init_images` response mutation after processing.
- Preserved task queue cleanup ordering. The duplicated `add_task_to_queue()`/`start_task()`/`finish_task()`/`pending_tasks.pop()` patterns remain explicit in `text2imgapi()` and `img2imgapi()` because combining them would have crossed processing-object setup, script runner selection, timing, and init-image injection differences.
- No dead API helper wrappers were removed. `validate_sampler_name()`, `get_selectable_script()`, `get_script()`, `init_default_script_args()`, and script arg helpers remain live across public API calls and extension/script compatibility paths.
- No extras/png-info decode extraction was made. `decode_base64_to_image()` is already shared; `decode_extras_batch_images()` intentionally ignores invalid encoded images/URLs for batch extras while `pnginfoapi()` and single extras keep normal decode failure behavior, so their apparent similarity is not safely reducible.
- No `apply_infotext()` split was made. Its field population, override-setting merge, and script-argument mention capture share parsed generation parameters and paste-field bindings; extracting only one loop would not remove meaningful duplication and could obscure pydantic v1/v2 compatibility behavior.

### Static/dynamic audit map notes
- Generation API request setup now flows: endpoint pre-validation/decode where needed -> `_prepare_generation_api_request()` mutates request from infotext, normalizes sampler/scheduler, creates processing args, initializes selectable/always-on script args -> endpoint queue lock builds the right processing object and runs either selected script or `process_images()`.
- Script API contracts remain conservative: selectable scripts still occupy `script_args[0]`, always-on script payloads still extend sparse vectors and record OpenClaw override bounds, and `OpenClaw Denoise Ramp` requested args still persist back into default script args.
- Sampler/scheduler compatibility remains unchanged: callers may provide `sampler_name` or legacy `sampler_index`; normalized sampler names still clear `sampler_index`, and non-automatic schedulers still populate empty request scheduler fields.
- Compatibility surfaces to continue treating conservatively: `StableDiffusionTxt2ImgProcessingAPI`/`StableDiffusionImg2ImgProcessingAPI` generated field names/defaults, pydantic `copy()` semantics, `script_name`/`script_args`/`alwayson_scripts`, `infotext`, `force_task_id`, `send_images`, `save_images`, `include_init_images`, task queue state, response `parameters`, and `processed_js_with_image_paths()` fields.

### Validation log
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pycompile-pass47.XXXXXX) python3 -m py_compile modules/api/api.py modules/api/models.py` - passed.
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pytest-pass47-api.XXXXXX) python3 -m pytest -q test/test_api_script_defaults.py test/test_infotext_paste_bindings.py` - passed: 5 passed, with the existing pytest config warning `Unknown config option: base_url`.
- Exact AST duplicate-body scan across `modules/api/api.py` and `modules/api/models.py` reported no duplicate nontrivial function bodies.
- `git diff --check` - passed.
- Server-backed `test/test_txt2img.py`, `test/test_img2img.py`, and `test/test_extras.py` API tests were inspected but not run because they require a live WebUI/API server fixture.

### Next unchecked scope
- More slices are still needed. Recommended next slice: continue to adjacent API surfaces after generation/extras helpers, especially progress/interrogate/options/reload/refresh endpoint helpers, model/listing APIs, and any remaining response serialization or queue/status helper duplication that can be reduced without changing public endpoint fields.


## Pass 48 - API progress/options/reload/refresh/listing helpers (2026-06-20)

### Checked scope
- `modules/api/api.py`: OpenClaw runtime/default endpoints adjacent to API helpers, `get_precision_map()`, generation diagnostics, task cleanup around `text2imgapi()`/`img2imgapi()`, extras queue locking, `progressapi()`, `interrogateapi()`, `interruptapi()`, `unloadapi()`, `reloadapi()`, `skip()`, `get_config()`, `set_config()`, model/listing endpoints (`get_samplers()`, `get_schedulers()`, `get_upscalers()`, `get_latent_upscale_modes()`, `get_sd_models()`, `get_sd_vaes()`, `get_hypernetworks()`, `get_face_restorers()`, `get_realesrgan_models()`, `get_prompt_styles()`, `get_embeddings()`), refresh endpoints, memory/extensions/server-control helpers.
- `modules/api/models.py`: response contracts for progress, options/flags, sampler/scheduler/upscaler/model/VAE/hypernetwork/face-restorer/RealESRGAN/style/embedding/memory/extension items.
- Adjacent contracts/tests: existing source-level API script/infotext tests, API training contract tests, and server-backed API suites inspected as compatibility surfaces.

### Findings and fixes
- Extracted repeated queue-lock call wrappers into `Api._call_with_queue_lock()` and routed extras execution plus `refresh_embeddings()`, `refresh_checkpoints()`, and `refresh_vae()` through it. This preserves the same lock boundary and return behavior while removing repeated `with self.queue_lock:` one-call endpoint bodies.
- Extracted duplicated generation task cleanup into `Api._finish_generation_task()` and `Api._clear_pending_task_unless_finished()`. `text2imgapi()` and `img2imgapi()` still add/start/finish tasks in the same order, keep endpoint-specific processing-object setup and timing behavior local, and only remove pending tasks on unfinished exceptions.
- No safe consolidation was made for `progressapi()` and `modules/progress.progressapi()`. They look adjacent but serve different APIs and response models: `/sdapi/v1/progress` reports WebUI state/current image/current task, while `/internal/progress` reports per-task queue/active/completed/live-preview fields.
- No safe consolidation was made for `interrogateapi()` and generation/extras queue use. Interrogation decodes and RGB-converts before the lock, then branches between CLIP and deepdanbooru under the lock with public 404 behavior for unknown models.
- No dead API helper wrappers were removed. `interruptapi()`, `unloadapi()`, `reloadapi()`, `skip()`, refresh endpoints, model/listing endpoints, memory/extensions helpers, and server-control endpoints are public route handlers or gated public route handlers.
- Model/listing response builders were left explicit. Their dictionaries map distinct public field names and source attributes, so a generic serializer would add indirection without removing meaningful duplicate logic and could risk field compatibility.
- Options/config handling was left unchanged. `get_config()` preserves fallback to option metadata defaults, and `set_config()` preserves checkpoint alias validation before applying/saving API-supplied options.

### Static/dynamic audit map notes
- Queue-sensitive public endpoints remain conservative: generation/extras/interrogate/refresh/precision-map code still acquires `self.queue_lock` around shared model or mutable registry operations; invalid img2img masks/init image decoding still happens before task queue registration.
- Public response surfaces preserved: progress fields (`progress`, `eta_relative`, `state`, `current_image`, `textinfo`, `current_task`), model/listing item keys, embeddings `loaded`/`skipped`, memory `ram`/`cuda`, and extension fields are unchanged.
- Task cleanup lifecycle remains: `add_task_to_queue()` before queue-lock processing -> `shared.state.begin()` -> `start_task()` -> selected script or `process_images()` -> `_finish_generation_task()` in the inner `finally` -> pending task removal only if that finishing block did not complete.
- Compatibility surfaces to continue treating conservatively: `/sdapi/v1/progress` versus `/internal/progress`, route handler return shapes that intentionally return `{}`/`None`/`Response`, `opts.data_labels` defaults, `sd_models.checkpoint_aliases`, model/listing dictionary keys, and `api_server_stop`-gated server endpoints.

### Validation log
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pycompile-pass48.XXXXXX) python3 -m py_compile modules/api/api.py modules/api/models.py` - passed.
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pytest-pass48-api.XXXXXX) python3 -m pytest -q test/test_api_script_defaults.py test/test_infotext_paste_bindings.py tests/test_api_training_contract.py` - passed: 7 passed, with the existing pytest config warning `Unknown config option: base_url`.
- Exact AST duplicate-body scan across `modules/api/api.py` and `modules/api/models.py` reported no duplicate nontrivial function bodies.
- `git diff --check` - passed.
- Server-backed endpoint suites such as `test/test_txt2img.py`, `test/test_img2img.py`, `test/test_extras.py`, and live progress/listing endpoint checks were not run because they require a running WebUI/API server fixture.

### Next unchecked scope
- More slices are still needed. Recommended next slice: continue into training/create/memory/extensions/server-control API helpers and then out toward lower-level model/registry refresh implementations (`shared.refresh_checkpoints()`, VAE/model loader listing refresh, extension listing metadata), looking for dead wrappers or duplicate registry serialization while preserving public API response contracts.


## Pass 49 - API training/create/memory/extensions/server-control helpers (2026-06-20)

### Checked scope
- `modules/api/api.py`: create endpoints (`create_embedding()`, `create_hypernetwork()`), training endpoints (`train_embedding()`, `train_hypernetwork()`), new private response/task helpers, memory endpoint (`get_memory()`), extension listing (`get_extensions_list()`), server-control endpoints (`kill_webui()`, `restart_webui()`, `stop_webui()`), and adjacent interrupt/skip/unload/reload wrappers.
- `modules/api/models.py`: `TrainResponse`, `CreateResponse`, `MemoryResponse`, `ExtensionItem`, and adjacent embedding/listing response models.
- Adjacent contracts/tests: `tests/test_api_training_contract.py`; live server-control and memory/extension endpoints were inspected as public API surfaces but not exercised without a running WebUI/API server fixture.

### Findings and fixes
- Extracted duplicated create endpoint state/response handling into `Api._run_create_task()` plus `_create_response()`. `create_embedding()` still reloads textual inversion embeddings immediately after successful creation, and both create endpoints preserve the exact `CreateResponse.info` strings for success and `AssertionError` handling.
- Extracted duplicated train endpoint optimization/response handling into `Api._run_training_task()` plus `_train_response()`. `train_embedding()` and `train_hypernetwork()` still preserve their exact success/error text prefixes, single `shared.state.end()` lifecycle, and optimization undo/apply behavior.
- Kept hypernetwork-specific training behavior explicit through `_prepare_hypernetwork_training()` and `_restore_hypernetwork_training_devices()`, preserving `shared.loaded_hypernetworks = []` before training and device restoration of `cond_stage_model`/`first_stage_model` in the training cleanup path.
- Updated `tests/test_api_training_contract.py` so the source-level contract follows the new helper structure: create endpoints must route through `_run_create_task()`, train endpoints through `_run_training_task()`, response helper types remain separated, and the shared training helper still owns exactly one `shared.state.end()` call.
- No dead memory helper code was found. `get_memory()` is a public `/sdapi/v1/memory` route, and its RAM/torch CUDA probes intentionally tolerate platform/device errors by returning `error` dictionaries under the existing `MemoryResponse` contract.
- No safe extension metadata serializer extraction was made. `get_extensions_list()` maps the public `ExtensionItem` fields directly after `extensions.list_extensions()`/`ext.read_info_from_repo()`, skips extensions without remotes as existing behavior, and has no repeated serialization body elsewhere in the checked scope.
- No server-control wrapper removal or consolidation was made. `interruptapi()`, `skip()`, `unloadapi()`, `reloadapi()`, `kill_webui()`, `restart_webui()`, and `stop_webui()` are public route handlers or gated public route handlers with intentionally distinct return shapes (`{}`, `None`, and `Response`) that should remain compatibility-preserved.

### Static/dynamic audit map notes
- Create endpoint flow now remains: public route -> `_run_create_task()` -> `shared.state.begin()` -> concrete create function -> optional post-create reload -> `CreateResponse(info=...)` -> `shared.state.end()`.
- Training endpoint flow now remains: public route -> `_run_training_task()` -> `shared.state.begin()` -> optional hypernetwork pre-hook -> optional optimization undo -> concrete train function -> optional hypernetwork device restore -> optional optimization apply -> `TrainResponse(info=...)` -> `shared.state.end()`.
- Public response schemas checked in this slice remain unchanged: `TrainResponse.info`, `CreateResponse.info`, `MemoryResponse.ram`/`cuda`, and `ExtensionItem` fields.
- Compatibility surfaces to continue treating conservatively: `/sdapi/v1/memory` platform fallbacks, extension list remote filtering and repo metadata reads, `api_server_stop`-gated routes, and public server-control return shapes.

### Validation log
- `PYTHONPYCACHEPREFIX=$(mktemp -d) python3 -m py_compile modules/api/api.py modules/api/models.py tests/test_api_training_contract.py` - passed.
- `pytest -q tests/test_api_training_contract.py` - passed: 3 passed, with the existing pytest config warning `Unknown config option: base_url`.
- Exact AST duplicate-body scan across `modules/api/api.py`, `modules/api/models.py`, and `tests/test_api_training_contract.py` reported no duplicate nontrivial function bodies.
- `git diff --check` - passed.
- Live memory/extensions/server-control endpoint tests were not run because they require a running WebUI/API server fixture and server-control calls can stop/restart the process.

### Next unchecked scope
- More slices are still needed. Recommended next slice: continue below the API route layer into lower-level registry refresh and listing implementations, especially `shared.refresh_checkpoints()`, VAE/model loader refresh/listing paths, extension metadata refresh internals, and any duplicate registry serialization behind the API wrappers while preserving public API contracts.


## Pass 50 - Lower-level registry refresh/listing helpers (2026-06-20)

### Checked scope
- `modules/shared.py`: compatibility aliases for `list_checkpoint_tiles`, `refresh_checkpoints`, `list_samplers`, and `reload_hypernetworks`.
- `modules/shared_items.py`: `sd_vae_items()`, `refresh_vae_list()`, `list_checkpoint_tiles()`, `refresh_checkpoints()`, adjacent sampler/unet/hypernetwork list helpers.
- `modules/sd_models.py`: checkpoint registry globals, `CheckpointInfo.register()`, `CheckpointInfo.calculate_shorthash()`, `checkpoint_tiles()`, `list_models()`, `get_closet_checkpoint_match()`, and `select_checkpoint()`.
- `modules/sd_vae.py`: VAE registry globals, `get_filename()`, new `vae_search_paths()`, `refresh_vae_list()`, and registry consumers that resolve/load VAE names.
- `modules/modelloader.py`: `load_models()`, `friendly_name()`, `load_upscalers()`, and spandrel loader initialization adjacent to registry/listing behavior.
- `modules/extensions.py`: extension registry globals, `ExtensionMetadata`, `Extension.read_info_from_repo()`, `Extension.do_read_info_from_repo()`, `list_extensions()`, and `find_extension()`.
- Adjacent contracts/callers: API refresh/listing endpoints, shared options refresh hooks, UI checkpoint/VAE refresh buttons, model converter extension calls, `tests/test_extensions_metadata_contract.py`, and `tests/test_sd_models_checkpoint_info_contract.py`.

### Findings and fixes
- Extracted VAE search-path construction from `sd_vae.refresh_vae_list()` into `sd_vae.vae_search_paths()`. This separates path discovery from registry mutation while preserving the exact path order, extension filters, optional `--ckpt-dir`/`--vae-dir` gating, duplicate-name overwrite behavior, and final natural-sort registry order.
- No safe removal was made for `shared.refresh_checkpoints()` / `shared.list_checkpoint_tiles()` or the corresponding `shared_items` wrappers. They are compatibility aliases and option-refresh/UI/API surfaces; external extensions may import them even when tracked in-tree callers mostly use lower-level modules directly.
- No consolidation was made between checkpoint and VAE listing serializers. Checkpoints expose `title`/`short_title` from `CheckpointInfo` objects, while VAE listings intentionally expose names from `vae_dict` plus `Automatic`/`None` UI sentinels in selected callers.
- No safe change was made to `sd_models.list_models()`. Its registry refresh includes default checkpoint download fallback, explicit `--ckpt` handling, alias registration, metadata/hash side effects, and sd_model_checkpoint option mutation that should stay local to checkpoint loading.
- No safe change was made to `modelloader.load_models()`. It is a broad compatibility helper used by checkpoints, upscalers, GFPGAN/CodeFormer, interrogate/deepbooru, and extensions; its silent exception tolerance and URL fallback are public behavior.
- No safe extension registry extraction/removal was made. `extensions.list_extensions()` owns registry clearing, metadata canonical-name de-duplication, disabled/builtin filtering, and requirement checks; `read_info_from_repo()` separately owns cached Git metadata refresh and remains used by API/config/UI surfaces.
- `modules/extensions_metadata.py` is not present in this checkout; extension metadata behavior lives in `modules/extensions.py` with coverage in `tests/test_extensions_metadata_contract.py`.

### Static/dynamic audit map notes
- Registry refresh boundaries remain conservative: checkpoint/VAE/model-loader/extension refresh paths mutate module-level registries that are consumed by UI, API, options refresh hooks, and extensions.
- Public compatibility surfaces preserved: `shared.refresh_checkpoints`, `shared.list_checkpoint_tiles`, `shared_items.refresh_checkpoints`, `shared_items.refresh_vae_list`, `sd_models.list_models`, `sd_models.checkpoint_tiles`, `sd_vae.refresh_vae_list`, `modelloader.load_models`, and `extensions.list_extensions`.
- Exact AST duplicate-body scan across `modules/shared.py`, `modules/shared_items.py`, `modules/sd_models.py`, `modules/sd_vae.py`, `modules/modelloader.py`, and `modules/extensions.py` reported no duplicate nontrivial function bodies after the VAE extraction.

### Validation log
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pycompile-pass50.XXXXXX) python3 -m py_compile modules/shared.py modules/shared_items.py modules/sd_models.py modules/sd_vae.py modules/modelloader.py modules/extensions.py tests/test_extensions_metadata_contract.py tests/test_sd_models_checkpoint_info_contract.py` - passed.
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pytest-pass50.XXXXXX) python3 -m pytest -q tests/test_extensions_metadata_contract.py tests/test_sd_models_checkpoint_info_contract.py` - passed: 5 passed, with the existing pytest config warning `Unknown config option: base_url`.
- Exact AST duplicate-body scan across the checked registry modules reported no duplicate nontrivial function bodies.
- `git diff --check` - passed.
- Live WebUI/API registry refresh endpoints and UI refresh buttons were not exercised because they require a running WebUI/API server fixture.

### Next unchecked scope
- More slices are still needed. Recommended next slice: continue from lower-level registries into UI refresh button plumbing and option refresh contracts, especially `modules/ui_common.py`/refresh-button helpers, checkpoint/VAE dropdown update lambdas, and extra-networks checkpoint refresh surfaces, while preserving extension-visible callback and Gradio update contracts.


## Pass 51 - UI refresh button plumbing and option refresh contracts (2026-06-20)

### Checked scope
- `modules/ui_common.py`: `create_refresh_button()` label selection, callback wrapper, component attribute mutation, and Gradio update return shape for single and multiple outputs.
- `modules/ui.py`: hires checkpoint refresh button, training embedding/hypernetwork/template refresh callbacks, and adjacent settings/model-merger wiring.
- `modules/shared_items.py`: option/listing wrappers for VAE, checkpoint, UNet, sampler, and hypernetwork refresh providers.
- `modules/ui_settings.py`: `create_setting_component()` refresh wiring for normal settings and quicksettings, including `OptionInfo.refresh` and `component_args` contracts.
- Extra-network checkpoint surfaces: `modules/ui_extra_networks.py` internal refresh tab callback, `modules/ui_extra_networks_checkpoints.py`, and `modules/ui_extra_networks_checkpoints_user_metadata.py` preferred-VAE refresh/save behavior.
- Adjacent first-party refresh button callers: `modules/ui_checkpoint_merger.py`, `modules/processing_scripts/refiner.py`, `modules/ui_prompt_styles.py`, `modules/ui_extensions.py`, plus the bundled model-converter extension refresh surfaces.

### Findings and fixes
- Extracted repeated checkpoint dropdown update callback construction into `shared_items.checkpoint_dropdown_args(*prefix_items, use_short=False)` and routed first-party checkpoint refresh buttons through it where the callback shape was identical: hires checkpoint, checkpoint merger A/B/C, and refiner checkpoint.
- Extracted repeated VAE dropdown update callback construction into `shared_items.sd_vae_dropdown_args(*prefix_items)` and routed first-party VAE refresh buttons through it for checkpoint merger bake-in VAE and checkpoint metadata preferred VAE.
- Preserved sentinel differences explicitly: hires checkpoint still prefixes `Use same checkpoint` and uses short checkpoint tiles, generic checkpoint merger/refiner choices have no prefix, preferred VAE still prefixes `Automatic`/`None`, and bake-in VAE still prefixes only `None`.
- Left `create_refresh_button()` behavior unchanged. Its extension-visible callback wrapper still refreshes the backing registry first, updates component attributes, returns one `gr.update(...)` for a single output, and returns a list of updates for multi-output refresh buttons.
- Left `modules/shared_items.refresh_vae_list()` and `refresh_checkpoints()` wrappers intact. They remain public option-refresh and compatibility surfaces for settings, API/UI wiring, and extensions.
- Left extra-network internal refresh plumbing unchanged. It intentionally refreshes all stored extra-network pages, rebuilds cached page HTML, returns `ui.pages_contents`, then reapplies filters/resizers in JS for each hidden refresh button.
- Left bundled extension model-converter callbacks unchanged as extension-owned surface, even though they have similar choice lambdas.

### Static/dynamic audit map notes
- Settings refresh chain remains: `OptionInfo.refresh` -> `ui_settings.create_setting_component()` -> `ui_common.create_refresh_button()` -> option `component_args` dict -> component attribute mutation plus Gradio update.
- Checkpoint refresh chain remains: refresh button -> `sd_models.list_models()` -> `shared_items.checkpoint_dropdown_args()`/checkpoint tiles -> dropdown choices update.
- VAE refresh chain remains: refresh button -> `sd_vae.refresh_vae_list()` -> `shared_items.sd_vae_dropdown_args()`/`vae_dict` keys -> dropdown choices update.
- Compatibility surfaces to continue treating conservatively: `create_refresh_button()`, `OptionInfo.refresh`, `OptionInfo.component_args`, `shared_items` refresh wrappers, extra-network hidden refresh buttons and JS callbacks, and bundled/third-party extension refresh lambdas.

### Validation log
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pycompile-pass51.XXXXXX) python3 -m py_compile modules/ui_common.py modules/ui.py modules/shared_items.py modules/ui_settings.py modules/ui_checkpoint_merger.py modules/processing_scripts/refiner.py modules/ui_extra_networks.py modules/ui_extra_networks_checkpoints.py modules/ui_extra_networks_checkpoints_user_metadata.py` - passed.
- Focused AST duplicate-lambda scan across the checked UI refresh modules and the bundled model-converter extension reported no duplicate in-scope first-party `choices` lambdas after extraction; the remaining similar model-converter callbacks were left as extension surface.
- `git diff --check` - passed.
- Live WebUI UI refresh buttons were not exercised because they require a running Gradio/WebUI session.

### Next unchecked scope
- More slices are still needed. Recommended next slice: continue through prompt-style/UI state refresh and settings reload plumbing, especially style reload/save/materialize callbacks, config-state refresh in extensions UI, quicksettings/settings component refresh contracts, and any remaining duplicated Gradio update helpers while preserving public UI and extension callback behavior.

## Pass 52 - Prompt-style/UI state refresh and settings reload plumbing (2026-06-20)

### Checked scope
- `modules/ui_prompt_styles.py`: `select_style()`, `save_style()`, `delete_style()`, `materialize_styles()`, `refresh_styles()`, and `UiPromptStyles` edit/materialize/copy callback wiring.
- `modules/ui_settings.py`: `get_value_for_setting()`, new `get_setting_component_args()` / `refreshed_setting_component_args()`, `create_setting_component()`, quicksettings construction, `run_settings_single()`, settings reload actions, and `demo.load()` settings value refresh.
- `modules/ui_common.py`: `create_refresh_button()` Gradio update helper contract and multi-output update shape.
- `modules/ui_extensions.py`: config-state save/refresh/dropdown plumbing in the Backup/Restore UI, `save_config_state()`, new `config_state_choices()`, and adjacent config-state restore/update callbacks.
- `modules/config_states.py`: `list_config_states()`, `get_config()`, webui/extension restore helpers, and `all_config_states` refresh side effects.
- `modules/styles.py`: style database reload/save/materialize-adjacent methods (`reload()`, `save_styles()`, style apply/extract helpers) as called by the UI style editor.
- Adjacent focused tests: `tests/test_ui_loadsave_contract.py` and `tests/test_ui_extensions_contract.py`.

### Findings and fixes
- Extracted repeated settings `OptionInfo.component_args` resolution into `ui_settings.get_setting_component_args()`. `get_value_for_setting()`, `create_setting_component()`, and refresh-button callbacks now share the same callable/dict/default handling and the same exclusion of Gradio `precision` from update payloads.
- Added `ui_settings.refreshed_setting_component_args()` as the refresh-button-facing wrapper, keeping `OptionInfo.refresh`/`component_args` behavior explicit for settings and quicksettings while preserving the existing `create_refresh_button()` contract.
- Extracted config-state dropdown choice construction into `ui_extensions.config_state_choices()`. Config-state save still selects the newly saved config, and the refresh button still calls `config_states.list_config_states()` before returning updated choices.
- No dead style editor callbacks were removed. `select_style()`, `save_style()`, `delete_style()`, `materialize_styles()`, and `refresh_styles()` are all directly wired to visible style editor controls and preserve their Gradio update shapes.
- No change was made to `ui_common.create_refresh_button()`. It remains the shared extension-visible refresh helper that mutates component attributes and returns either a single `gr.update(...)` or a list of updates depending on output count.
- No config-state restore/save wrappers were removed. They are UI callbacks with side effects across config files, extension disabled state, git restore, and restart request behavior.

### Static/dynamic audit map notes
- Settings refresh chain remains: `OptionInfo.refresh` -> `ui_settings.create_setting_component()` -> `ui_common.create_refresh_button()` -> `refreshed_setting_component_args()` -> component attribute mutation plus Gradio update.
- Settings load/save refresh remains: `run_settings_single()` updates `opts`, saves `config.json`, then returns `get_value_for_setting()` plus the hidden JSON settings payload; `demo.load()` refreshes every settings component through the same value/update helper.
- Style edit chain remains: style selection updates prompt fields and Save/Delete visibility; Save/Delete persist through `shared.prompt_styles.save_styles()` and then refresh both the main style dropdown and editor selection choices; Materialize applies selected styles and clears the dropdown.
- Config-state refresh chain remains: save/list refresh `config_states.all_config_states`, dropdown choices are rebuilt with `Current` first, and restore keeps selected config names tied to the refreshed global map.
- Compatibility surfaces to continue treating conservatively: `OptionInfo.component_args`, `OptionInfo.refresh`, `ui_common.create_refresh_button()`, style dropdown Gradio update shapes, config-state dropdown value/choice behavior, extension Backup/Restore UI callbacks, and public settings/quicksettings component dictionaries.

### Validation log
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pycompile-pass52.XXXXXX) python3 -m py_compile modules/ui_settings.py modules/ui_prompt_styles.py modules/ui_common.py modules/ui_extensions.py modules/config_states.py modules/styles.py` - passed.
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pytest-pass52.XXXXXX) python3 -m pytest -q tests/test_ui_loadsave_contract.py tests/test_ui_extensions_contract.py` - passed: 3 passed, with the existing pytest config warning `Unknown config option: base_url`.
- Exact AST duplicate-body scan across `modules/ui_settings.py`, `modules/ui_prompt_styles.py`, `modules/ui_common.py`, `modules/ui_extensions.py`, `modules/config_states.py`, and `modules/styles.py` reported no duplicate nontrivial function bodies after the helper extractions.
- `git diff --check` - passed.
- Live Gradio/WebUI style editor, quicksettings, settings reload, and extension Backup/Restore buttons were not exercised because they require a running WebUI session and some actions can restart or alter extension/git state.

### Next unchecked scope
- More slices are still needed. Recommended next slice: continue into UI component construction and paste/load/save plumbing around `modules/ui_loadsave.py`, `modules/ui_components.py`, `modules/ui_toprow.py`, and infotext/paste binding helpers, looking for duplicated Gradio update shapes or dead callback wrappers while preserving API-visible paste/settings contracts.


## Pass 53 - UI component construction and paste/load/save plumbing (2026-06-20)

### Checked scope
- `modules/ui_loadsave.py`: `radio_choices()`, `UiLoadsave.__init__()`, `add_component()`, `add_block()`, `read_from_file()`, `write_to_file()`, `dump_defaults()`, `iter_changes()`, `ui_view()`, `ui_apply()`, `create_ui()`, and `setup_ui()`.
- `modules/ui_components.py`: `FormComponent`, `ToolButton`, `ResizeHandleRow`, form wrapper classes, `DropdownMulti`, `DropdownEditable`, and `InputAccordion` lifecycle/accordion ID/reset helpers.
- `modules/ui_toprow.py`: `Toprow` construction methods, prompt image import binding, submit/interrupt/skip handlers, tool-row buttons, clear-prompt JS callback, compact/classic layout split, and style UI handoff.
- `modules/ui_common.py`: `update_generation_info()`, `plaintext_to_html()`, `update_logfile()`, `save_files()`, `OutputPanel`, `create_output_panel()` output-panel button wiring, paste/send-to registration, `create_refresh_button()`, and `setup_dialog()` as needed for adjacent update shapes.
- `modules/infotext_utils.py`: `ParamBinding`, `PasteField`, paste-field registration, legacy `create_buttons()`/`bind_buttons()`, `register_paste_params_button()`, `connect_paste_params_buttons()`, `send_image_and_dimensions()`, and `connect_paste()` output update handling.
- Adjacent callers/contracts: txt2img/img2img/pnginfo paste registrations in `modules/ui.py`, script-runner `paste_field_names`, `javascript/generationParams.js` gallery generation-info triggers, `tests/test_ui_loadsave_contract.py`, and `tests/test_save_serialization_contract.py`.

### Findings and fixes
- Extracted the duplicated output-panel save/save-zip callback payload in `ui_common.create_output_panel()` into local `save_button_kwargs()`. The two buttons still use the same wrapped `save_files()` callback, the same four inputs, the same two outputs, the same `selected_gallery_index()` JavaScript argument shape, and preserve the prior `show_progress=False` difference on the normal save button.
- Preserved `UiLoadsave` component field handling. The exact type checks, `InputAccordion` open/value mapping, custom-script key prefixing, dropdown choice validation, and `component_mapping` order are UI/defaults contracts rather than dead code.
- Preserved `ui_components` wrapper classes even though several only override `get_block_name()`. They are public script/extension-facing component classes and encode distinct Gradio/form block names; a factory would add indirection without removing a real maintenance hazard.
- Preserved `Toprow` methods and hidden buttons. The classic/compact render split, prompt image upload binding, interrupt/skip placeholders, token counter buttons, restore-progress button, and paste/style controls are wired later by `modules/ui.py`, JavaScript, or output-panel layout.
- Preserved infotext paste compatibility helpers. `create_buttons()` and `bind_buttons()` are explicitly legacy extension compatibility surfaces; `ParamBinding` and `PasteField` shape API-visible paste behavior, script paste-field aggregation, image/dimension send-to behavior, override-settings dropdown updates, and tab-switch JavaScript.
- No dead paste/load/save callback wrappers were removed. Dynamic Gradio callbacks and extension/script hooks make apparently small wrappers API-visible unless proven otherwise.

### Static/dynamic audit map notes
- UI defaults chain remains: `UiLoadsave.add_block()` walks Gradio blocks -> `add_component()` records supported component fields and restores config values -> `setup_ui()` binds View/Apply to ordered `component_mapping` values.
- Output save chain remains: output panel button -> `save_button_kwargs()` -> wrapped `save_files()` -> gallery URL/image decode -> selected/all image indexing -> CSV/log/zip/save-image side effects -> `gr.File.update(value=..., visible=True)` and HTML log output.
- Paste/send-to chain remains: UI registers `PasteField` values -> output panels and PNG Info register `ParamBinding` objects -> `connect_paste_params_buttons()` wires image copy, source-tab field copy, parsed infotext paste, override settings, prompt recalculation, and tab switch JS.
- Compatibility surfaces to keep conservative: `modules.generation_parameters_copypaste` alias, `ParamBinding`, `PasteField`, `create_buttons()`, `bind_buttons()`, `registered_param_bindings`, script `paste_field_names`, `UiLoadsave` saved key paths, component elem IDs, and Gradio update return shapes.

### Validation log
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pycompile-pass53.XXXXXX) python3 -m py_compile modules/ui_loadsave.py modules/ui_components.py modules/ui_toprow.py modules/ui_common.py modules/infotext_utils.py tests/test_ui_loadsave_contract.py tests/test_save_serialization_contract.py` - passed.
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pytest-pass53.XXXXXX) python3 -m pytest -q tests/test_ui_loadsave_contract.py tests/test_save_serialization_contract.py` - passed: 4 passed, with the existing pytest config warning `Unknown config option: base_url`.
- Exact AST duplicate-body scan across `modules/ui_loadsave.py`, `modules/ui_components.py`, `modules/ui_toprow.py`, `modules/ui_common.py`, and `modules/infotext_utils.py` reported `duplicate nontrivial function body groups: 0`.
- `git diff --check` - passed.
- Live Gradio/WebUI save, paste, send-to, compact/classic toprow, and UI-defaults interactions were not exercised because they require a running WebUI session and user-visible UI state.

### Next unchecked scope
- More slices are still needed. Recommended next slice: continue through remaining UI dialog/output-adjacent and extension surfaces not covered by this pass, especially `modules/ui_postprocessing.py`, `modules/ui_checkpoint_merger.py`, `modules/ui_extra_networks*.py`, `modules/ui_html_extensions.py`, and bundled extension UI helpers, looking for duplicated callback/update payloads while preserving extension-visible Gradio contracts.

## Pass 54 - UI dialog/output-adjacent and extension surfaces (2026-06-20)

### Checked scope
- `modules/ui_postprocessing.py`: Extras tab construction, single/batch/batch-directory tab index callbacks, postprocessing submit wiring, paste registration, and postprocessing image-change callback.
- `modules/ui_checkpoint_merger.py`: interpolation description helper, model-merger exception/update path, checkpoint/VAE refresh button callbacks, metadata accordion visibility, metadata read callback, merge-result clearing, and wrapped merge callback output shape.
- `modules/ui_extra_networks.py`: preview extension helpers, page registration/API routes, thumbnail/cover-image/metadata/single-card endpoints, JS/HTML escaping helpers, `ExtraNetworksPage` card/tree/dir HTML builders, preview/description/metadata lookup helpers, page ordering, tab/refresh/load callback wiring, path parent check, and save-preview callback.
- `modules/ui_extra_networks_checkpoints.py`, `modules/ui_extra_networks_hypernets.py`, and `modules/ui_extra_networks_textual_inversion.py`: page refresh/list/create-item/allowed-preview-directory methods and prompt/search/sort/preview payloads.
- `modules/ui_extra_networks_user_metadata.py` and `modules/ui_extra_networks_checkpoints_user_metadata.py`: metadata editor component construction, file metadata/card preview rendering, metadata save/write, popup/save callback chaining, preview replacement, checkpoint preferred-VAE refresh/save/reload behavior.
- `modules/ui_html_extensions.py`: JS/CSS head injection helpers, cache-busting web paths, user CSS inclusion, theme/background style injection, and `TemplateResponse` wrapper preservation.
- Bundled extension UI helpers: `extensions/sd-webui-incantations/scripts/ui_wrapper.py` abstract/no-op extension hook surface and `extensions/sd-webui-model-converter/scripts/ui.py` tab/API UI helpers.
- Adjacent tests/contracts: `tests/test_extra_networks_path_contract.py`, `tests/test_extra_networks_metadata_contract.py`, pass-51 refresh-helper ledger notes, and direct callers in `modules/ui.py`, `modules/profiling.py`, `modules/initialize.py`, and bundled extension scripts.

### Findings and fixes
- Routed the bundled model-converter extension checkpoint refresh callback through `shared_items.checkpoint_dropdown_args()` instead of its local duplicate `{"choices": sd_models.checkpoint_tiles()}` payload.
- Routed the bundled model-converter VAE refresh callback through `shared_items.sd_vae_dropdown_args("None")` instead of its local duplicate `{"choices": ["None", *list(sd_vae.vae_dict)]}` payload.
- Preserved the model-converter initial dropdown choices, component IDs, tab tuple, FastAPI routes, request model defaults, and conversion callback payload shape.
- No first-party extra-network card/tree/page helpers were removed. The small wrappers are directly tied to API routes, JavaScript callbacks, HTML templates, Gradio hidden buttons, extension registration, preview-file safety checks, or user metadata side effects.
- No checkpoint-merger or postprocessing callback wrappers were removed. Their lambdas and update lists preserve Gradio event contracts, JavaScript submit/merge hooks, output ordering, and error-recovery dropdown refresh behavior.
- No `UIWrapper` hooks were removed from the incantations extension. The pass found they are subclass/extension lifecycle hooks used by bundled scripts and callback aggregation, not dead no-ops.

### Static/dynamic audit map notes
- Extra-network route chain remains: `add_pages_to_demo()` registers thumbnail, cover-image, metadata, and single-card endpoints; those endpoints consume `extra_pages`, `allowed_dirs`, page metadata, card HTML templates, and preview extension policy.
- Extra-network UI refresh chain remains: hidden per-page Refresh button -> refresh every stored page -> rebuild cached page HTML -> return one HTML update per page -> reapply JS filter and resize handles.
- Extra-network card/editor chain remains: card edit button and JS set hidden name -> editor `button_edit` populates metadata/preview fields -> save/replace-preview writes sidecar/preview files -> JS refreshes just the affected card.
- Checkpoint preferred-VAE metadata chain remains: editor save writes JSON metadata and reloads VAE weights only when the edited checkpoint is currently active.
- HTML extension injection chain remains: `reload_javascript()` wraps Gradio `TemplateResponse`, injects JS before `</head>`, injects CSS before `</body>`, and keeps the original template response cached in `shared.UITemplateResponseOriginal`.
- Compatibility surfaces to keep conservative: `register_page()`, `ExtraNetworksPage` subclass methods, extra-network API route query shapes, hidden extra-network refresh/save-preview buttons, metadata editor button IDs, `UIWrapper` lifecycle hooks, model-converter API routes, and Gradio update return shapes.

### Validation log
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pycompile-pass54.XXXXXX) python3 -m py_compile modules/ui_postprocessing.py modules/ui_checkpoint_merger.py modules/ui_extra_networks.py modules/ui_extra_networks_checkpoints.py modules/ui_extra_networks_checkpoints_user_metadata.py modules/ui_extra_networks_hypernets.py modules/ui_extra_networks_textual_inversion.py modules/ui_extra_networks_user_metadata.py modules/ui_html_extensions.py extensions/sd-webui-incantations/scripts/ui_wrapper.py extensions/sd-webui-model-converter/scripts/ui.py tests/test_extra_networks_path_contract.py tests/test_extra_networks_metadata_contract.py` - passed.
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pytest-pass54.XXXXXX) python3 -m pytest -q tests/test_extra_networks_path_contract.py tests/test_extra_networks_metadata_contract.py` - passed: 5 passed, with the existing pytest config warning `Unknown config option: base_url`.
- Exact AST duplicate-body scan across the checked UI/extra-network/extension files reported `duplicate nontrivial function body groups: 0`.
- `git diff --check` - passed.
- Live Gradio/WebUI extra-network card refresh, metadata editor, Extras postprocessing, model merger, model-converter tab/API, and HTML injection behavior were not exercised because they require a running WebUI session and user-visible UI/server state.

### Next unchecked scope
- More slices are still needed. Recommended next slice: continue into script/extension callback aggregation and script UI surfaces, especially `modules/scripts.py`, `modules/script_callbacks.py`, extension callback registration metadata, and bundled extension script classes, looking for duplicated lifecycle hook wrappers or stale compatibility shims while preserving third-party extension APIs.

## Pass 55 - Script/extension callback aggregation and script UI surfaces (2026-06-20)

### Checked scope
- `modules/script_callbacks.py`: callback parameter classes, `ScriptCallback`, callback naming/registration, extension callback ordering metadata, ordered callback cache, callback enumeration/removal helpers, lifecycle dispatch wrappers, and public `on_*` registration API wrappers.
- `modules/scripts.py`: `Script` base lifecycle hooks, component-specific callback shims, `ScriptBuiltinUI`, script discovery/dependency ordering, script loading, `ScriptRunner` initialization, script UI construction, selectable/always-on UI setup, callback aggregation, timed lifecycle hook wrappers, component hook dispatch, script source reload, and named script argument setter.
- Bundled extension script/UI surfaces: `extensions/sd-webui-incantations/scripts/ui_wrapper.py`, incantations callback registration/use sites, bundled script classes under `extensions-builtin/*/scripts`, OpenClaw bundled extension script classes, and `extensions/sd-webui-model-converter/scripts/ui.py` callback registrations.
- Adjacent contracts/callers: `modules/ui_component_patches.py` component hook patch points, `modules/shared_items.py` callback order settings, `modules/ui_settings.py` callback order UI, `modules/ui.py` UI tab/train/settings callback dispatch, model/image/sampler callback call sites, and extension tests with script callback stubs.

### Findings and fixes
- Extracted the repeated public callback registration payload into `script_callbacks.add_callback_for_category()`. Public `on_*` functions keep their existing names, signatures, docstrings, categories, callback-map keys, generated callback names, and `name=` behavior while sharing one internal helper for the repeated `callback_map[...]`/`category=` wiring.
- Verified the `on_after_component()` public compatibility wrapper is still live and left its single docstring/API surface intact.
- Preserved the lifecycle dispatch wrappers in `script_callbacks.py`. Although many have similar try/report loops, each wrapper has API-visible ordering direction, argument shape, return aggregation behavior, and error label semantics.
- Preserved `Script` base no-op hooks and `ScriptRunner` lifecycle wrappers. These are third-party extension subclass surfaces and timed dispatch points, not dead code.
- Preserved component-specific `Script.on_before_component()`/`on_after_component()` and `ScriptRunner.apply_on_before_component_callbacks()` behavior. The seed processing script still relies on elem-id-specific post-component hooks, and global component callbacks remain separate through `script_callbacks.on_before_component()`/`on_after_component()`.
- Preserved bundled extension `UIWrapper` hooks and extension script classes. They are subclass/UI lifecycle surfaces or registered extension callbacks, even when their default implementations are no-ops.

### Static/dynamic audit map notes
- Callback registration chain remains: public `script_callbacks.on_*()` wrapper -> `add_callback_for_category()` -> `add_callback()` -> extension/file/category/name-derived `ScriptCallback` entry in `callback_map`.
- Callback ordering chain remains: `ordered_callbacks()` -> `sort_callbacks()` -> extension metadata `list_callback_order_instructions()` dependency graph -> optional `shared.opts.prioritized_callbacks_<category>` user ordering -> cached category list.
- Script UI chain remains: `ScriptRunner.initialize_scripts()` instantiates visible scripts -> `setup_ui()`/`setup_ui_for_section()` creates groups/dropdown -> `create_script_ui()` records controls, API arg metadata, infotext fields, and paste field names -> processing lifecycle methods slice `p.script_args` through recorded ranges.
- Script lifecycle chain remains: processing call sites invoke `ScriptRunner` hook methods -> ordered script callback list -> `_run_timed_script_hook()` -> script method with script UI args and timing capture.
- Compatibility surfaces to keep conservative: all public `script_callbacks.on_*` wrappers, `callback_map` key names, callback generated name format, `ScriptCallback`, `Script` base hooks, `ScriptRunner` hook method names, `reload_scripts` alias, `Script.elem_id()`, component hook callback object shape, and bundled extension `UIWrapper` method names.

### Validation log
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pycompile-pass55.XXXXXX) python3 -m py_compile modules/script_callbacks.py modules/scripts.py extensions/sd-webui-incantations/scripts/ui_wrapper.py extensions/sd-webui-model-converter/scripts/ui.py` - passed.
- Stubbed import contract for `modules/script_callbacks.py` confirmed `on_ui_tabs(callback, name="contract")` still appends one callback to `callback_map["callbacks_ui_tabs"]`, preserves callback identity, produces a generated name ending in `/ui_tabs/contract`, and exposes the category through `enumerate_callbacks()`.
- Direct system-Python import of `modules.script_callbacks` without stubs is blocked in this shell by missing runtime dependencies (`fastapi`, then `torch` through preload/shared imports). No dependencies were installed for this audit slice.
- Exact AST duplicate-body scan across `modules/script_callbacks.py`, `modules/scripts.py`, `extensions/sd-webui-incantations/scripts/ui_wrapper.py`, and `extensions/sd-webui-model-converter/scripts/ui.py` reported only the intentionally empty public base hook stubs `Script.postprocess_image()` and `Script.postprocess_maskoverlay()`; they were preserved as subclass API contracts.
- `git diff --check` - passed.
- Live Gradio/WebUI script dropdown, extension callback ordering UI, component hook dispatch, and bundled extension script interactions were not exercised because they require a running WebUI session and extension/runtime state.

### Next unchecked scope
- More slices are still needed. Recommended next slice: continue into postprocessing script runner and bundled postprocessing extension classes, especially `modules/scripts_postprocessing.py`, `modules/scripts_auto_postprocessing.py`, `modules/processing_scripts/*`, and `extensions-builtin/postprocessing-for-training/scripts/*`, looking for duplicated postprocessing UI/run wrappers or stale compatibility shims while preserving extension-visible postprocessing APIs.

## Pass 56 - Postprocessing script runner and bundled postprocessing extension classes (2026-06-20)

### Checked scope
- `modules/scripts_postprocessing.py`: `PostprocessedImageSharedInfo`, `PostprocessedImage.__init__()`, `get_suffix()`, `create_copy()`, `ScriptPostprocessing` base methods and elem-id helpers, `wrap_call()`, `ScriptPostprocessingRunner.initialize_scripts()`, `create_script_ui()`, `scripts_in_preferred_order()`, `setup_ui()`, `run()`, `create_args_for_run()`, and `image_changed()`.
- `modules/scripts_auto_postprocessing.py`: `ScriptPostprocessingForMainUI` title/show/ui/postprocess bridge and `create_auto_preprocessing_script_data()` main-UI wrapper factory.
- `modules/processing_scripts/seed.py`, `sampler.py`, `refiner.py`, and `comments.py`: built-in always-visible UI scripts, seed reuse callback binding, sampler/refiner setup methods, prompt comment stripping, token-counter callback registration, and option registration.
- `extensions-builtin/postprocessing-for-training/scripts/postprocessing_autosized_crop.py`, `postprocessing_caption.py`, `postprocessing_create_flipped_copies.py`, `postprocessing_focal_crop.py`, and `postprocessing_split_oversized.py`: bundled postprocessing UI dictionaries, enable gates, helper functions, extra-image/caption semantics, and crop/split processing methods.
- Adjacent callers/contracts: `modules/postprocessing.py` extras/API dispatch, `modules/ui_postprocessing.py` Extras submit and image-change callback, `modules/scripts.py` postprocessing script discovery/main-UI injection, `modules/shared_items.py` postprocessing option lists, `modules/shared_options.py` postprocessing options, `modules/api/api.py` legacy extras API call, and `tests/test_postprocessing_caption_contract.py`.

### Findings and fixes
- No safe source-code remediation was found in this slice; committed a ledger-only checkpoint after function-by-function inspection.
- Preserved `ScriptPostprocessing` base no-op methods, public class names, method signatures, `extra_only`/`main_ui_only`, `tab_name`, and elem-id helpers as third-party extension APIs.
- Preserved `ScriptPostprocessingRunner` UI/run argument slicing. The `args_from`/`args_to` ranges, dictionary conversion from ordered controls, first-pass hook, extra-image expansion, `disable_processing`, `scripts_order`, disabled-in-extras filtering, and lazy UI bootstrap are part of Extras/API execution behavior.
- Preserved `ScriptPostprocessingForMainUI` and `create_auto_preprocessing_script_data()`. They are live through `modules/scripts.py` when `postprocessing_enable_in_main_ui` is configured, and the wrapper keeps postprocessing classes usable from txt2img/img2img `postprocess_image()` hooks.
- Preserved `modules/postprocessing.run_postprocessing_webui()` and `run_extras()` adjacent wrappers as live Gradio/API compatibility surfaces; `run_extras()` is still called from `modules/api/api.py`.
- Preserved repeated `show() -> scripts.AlwaysVisible` methods in the main processing scripts. The exact duplicate scan flagged these, but they are intentional always-on UI declarations rather than dead code.
- Preserved bundled postprocessing extension helper functions (`center_crop()`, `multicrop_pic()`, `split_pic()`) and UI dictionaries. Their code is short, operation-specific, and directly consumed by the script `process()` methods; abstracting the repeated `InputAccordion`/`enable` pattern would add extension-visible indirection without removing a proven maintenance hazard.

### Static/dynamic audit map notes
- Extras chain remains: `ui_postprocessing.create_ui()` builds postprocessing script controls through `scripts.scripts_postproc.setup_ui()` -> Extras submit calls `postprocessing.run_postprocessing_webui()` -> `run_postprocessing()` creates a `PostprocessedImage` -> `ScriptPostprocessingRunner.run()` executes first-pass and per-image process hooks -> `modules/postprocessing.py` saves images/captions and returns gallery/html outputs.
- Legacy API chain remains: API extras endpoint -> `postprocessing.run_extras()` -> `scripts.scripts_postproc.create_args_for_run()` maps legacy API fields into postprocessing script args -> `run_postprocessing()` with explicit script order.
- Main-UI postprocessing chain remains: `modules/scripts.py` loads `scripts_auto_postprocessing.create_auto_preprocessing_script_data()` -> selected postprocessing class is wrapped in `ScriptPostprocessingForMainUI` -> generation `postprocess_image()` constructs a temporary `PostprocessedImage`, runs the postprocessing script, copies info into `p.extra_generation_params`, and replaces `script_pp.image`.
- Bundled training-postprocessing chain remains: split/focal crop can set `pp.extra_images` to `PostprocessedImage` copies; create-flipped-copies appends raw PIL images that the runner wraps via `create_copy()`; caption updates `pp.caption`; auto-sized crop replaces `pp.image` only when a valid crop is found.
- Compatibility surfaces to keep conservative: `PostprocessedImage`, `PostprocessedImageSharedInfo`, `PostprocessedImage.extra_images`, `caption`, `nametags`, `disable_processing`, `ScriptPostprocessing` method names/signatures, `ScriptPostprocessingRunner.create_args_for_run()`, `run_extras()`, postprocessing option names, bundled postprocessing script class names, and UI control dictionary keys.

### Validation log
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pycompile-pass56.XXXXXX) python3 -m py_compile modules/scripts_postprocessing.py modules/scripts_auto_postprocessing.py modules/postprocessing.py modules/ui_postprocessing.py modules/processing_scripts/seed.py modules/processing_scripts/sampler.py modules/processing_scripts/refiner.py modules/processing_scripts/comments.py extensions-builtin/postprocessing-for-training/scripts/postprocessing_autosized_crop.py extensions-builtin/postprocessing-for-training/scripts/postprocessing_caption.py extensions-builtin/postprocessing-for-training/scripts/postprocessing_create_flipped_copies.py extensions-builtin/postprocessing-for-training/scripts/postprocessing_focal_crop.py extensions-builtin/postprocessing-for-training/scripts/postprocessing_split_oversized.py tests/test_postprocessing_caption_contract.py` - passed.
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pytest-pass56.XXXXXX) python3 -m pytest -q tests/test_postprocessing_caption_contract.py` - passed: 2 passed, with the existing pytest config warning `Unknown config option: base_url`.
- Exact AST duplicate-body scan across the checked postprocessing runner/classes reported only intentionally empty stubs (`ScriptPostprocessing.image_changed()` and `ScriptRefiner.__init__()`) plus repeated `show() -> scripts.AlwaysVisible` methods on always-on script surfaces; preserved as public hook/UI behavior.
- `git diff --check` - passed.
- Live Gradio/WebUI Extras, txt2img/img2img postprocessing hooks, actual upscaler/caption/deepbooru/BLIP/focal-crop runtime behavior, and browser UI interactions were not exercised because they require a running WebUI session and user-visible/runtime model state.

### Next unchecked scope
- More slices are still needed. Recommended next slice: continue into core postprocessing execution and image save/output plumbing around `modules/postprocessing.py`, `modules/images.py`, `modules/ui_postprocessing.py`, and related extras API/image metadata helpers, looking for duplicated save/caption/infotext/output wrappers while preserving API-visible extras behavior and image metadata semantics.


## Pass 57 - Core postprocessing save/caption/infotext output plumbing (2026-06-20)

### Checked scope
- `modules/postprocessing.py`: `combine_caption()`, postprocessing input collection for single/batch/directory modes, image metadata read-through, `PostprocessedImage` execution, infotext construction, PNG info propagation, current-image assignment, `images.save_image()` call shape, caption sidecar handling, output gallery assembly, `run_postprocessing_webui()`, and legacy API `run_extras()` wrapper.
- `modules/images.py`: `save_image_with_geninfo()`, `geninfo_to_exif_bytes()`, `save_image()` naming/save/txt-sidecar/callback behavior, `read_info_from_image()`, and `image_data()` metadata decode helper.
- `modules/ui_postprocessing.py`: Extras tab mode selection, script input setup, submit callback inputs/outputs, paste-field registration, and postprocessing image-change callback.
- Extras API/image metadata helpers: `modules/api/api.py` `setUpscalers()`, `decode_extras_batch_images()`, `_run_extras()`, `extras_single_image_api()`, `extras_batch_images_api()`, `encode_pil_to_base64()`, and `pnginfoapi()`; `modules/extras.py` `run_pnginfo()`.
- Adjacent contracts/tests: `tests/test_postprocessing_caption_contract.py`, `test/test_postprocessing_api_defaults.py`, `test/test_images_save.py`, `tests/test_save_serialization_contract.py`, and prior pass notes for postprocessing runner/classes and UI output-save contracts.

### Findings and fixes
- Extracted the embedded caption sidecar read/combine/write block from `run_postprocessing()` into `save_caption_sidecar()`, reusing the already-tested `combine_caption()` behavior while keeping the output loop's image save order, caption sidecar filename, existing-caption action semantics, and empty-caption skip behavior unchanged.
- Added focused contract coverage for `save_caption_sidecar()` to prove existing caption append behavior and the no-file behavior for an empty combined caption.
- Preserved `run_postprocessing_webui()` as the live Gradio queue wrapper and `run_extras()` as the legacy extras API bridge.
- Preserved API-visible extras behavior: API extras still force `show_extras_results=True`, call `run_extras(..., save_output=False)`, encode returned PIL images only, and keep single-image no-output responses as `image=None`.
- Preserved `images.save_image()` txt sidecar and callback behavior. Its `.txt` generation writes generation infotext when `opts.save_txt` is enabled; postprocessing caption sidecars are separate caption text files intentionally written from `pp.caption` after the image path is finalized.
- No safe removal was found for image metadata helpers or output wrappers. `save_image_with_geninfo()`, `read_info_from_image()`, `image_data()`, `run_pnginfo()`, and `pnginfoapi()` are still directly wired to image save, PNG Info, upload metadata, API encoding, and UI paths.

### Static/dynamic audit map notes
- Extras execution chain remains: `ui_postprocessing.create_ui()` -> `call_queue.wrap_ui_gpu_call(postprocessing.run_postprocessing_webui)` -> `run_postprocessing()` -> postprocessing scripts -> `images.save_image()` -> optional `save_caption_sidecar()` -> gallery/html outputs.
- Infotext propagation remains: source image metadata from `images.read_info_from_image()` is copied into `existing_pnginfo`; per-output `pp.info` is rendered into the `extras` PNG section and `postprocessing` image info when PNG info is enabled; API responses encode returned PIL image metadata through `encode_pil_to_base64()`.
- Save path behavior remains split by purpose: `images.save_image()` owns naming, atomic image writes, optional generation `.txt` sidecar, callbacks, 4chan downscale export, and replacement policy; `save_caption_sidecar()` owns only postprocessing caption text generated by postprocessing scripts.
- Compatibility surfaces to keep conservative: `run_postprocessing()` positional inputs and return tuple, `run_postprocessing_webui(id_task, *args, **kwargs)`, `run_extras()` legacy signature, `images.save_image()` return `(fullfn, txt_fullfn)`, PNG info section names, API extras response shapes, caption action option names, and output image ordering.

### Validation log
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pycompile-pass57.XXXXXX) python3 -m py_compile modules/postprocessing.py modules/images.py modules/ui_postprocessing.py modules/api/api.py modules/extras.py tests/test_postprocessing_caption_contract.py test/test_postprocessing_api_defaults.py test/test_images_save.py` - passed.
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pytest-pass57-caption.XXXXXX) python3 -m pytest -q tests/test_postprocessing_caption_contract.py test/test_images_save.py` - passed: 8 passed, with the existing pytest config warning `Unknown config option: base_url`.
- Broader focused pytest command `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pytest-pass57.XXXXXX) python3 -m pytest -q tests/test_postprocessing_caption_contract.py test/test_postprocessing_api_defaults.py test/test_images_save.py` was blocked by unrelated/pre-existing failures in this shell: direct `modules.scripts_postprocessing` import requires missing `fastapi`/`torch`, and `test_extras_single_response_allows_no_output_from_skipped_or_interrupted_run` fails because its fake API class lacks `_call_with_queue_lock()` for the current `_run_extras()` implementation.
- Exact AST duplicate-body scan across `modules/postprocessing.py`, `modules/images.py`, `modules/ui_postprocessing.py`, `modules/api/api.py`, and `modules/extras.py` reported no duplicate nontrivial function body groups.
- `git diff --check` - passed.
- Live WebUI Extras/API image processing, PNG Info UI, and browser interactions were not exercised because they require a running WebUI session and runtime/UI state.

### Next unchecked scope
- More slices are still needed. Recommended next slice: continue into generation/save plumbing adjacent to postprocessing, especially `modules/processing.py`, `modules/ui_common.py` save/download paths, scripts that call `images.save_image()` (`scripts/loopback.py`, `scripts/outpainting_mk_2.py`, `scripts/sd_upscale.py`, `scripts/xyz_grid.py`), and API processed-image response helpers, looking for duplicated save/infotext/list assembly while preserving callback contracts and user-visible output ordering.

## Pass 58 - Generation/save plumbing adjacent to postprocessing (2026-06-20)

### Checked scope
- `modules/processing.py`: `process_images_inner()` generation output loop, pre-face-restoration and pre-color-correction save branches, final sample save, output image/infotext list assembly, returned mask/mask-composite handling, grid return/save handling, highres-fix intermediate save, and img2img init-image save.
- `modules/ui_common.py`: `update_generation_info()`, `update_logfile()`, `save_files()`, output-panel save/download button wiring, selected-image save behavior, CSV log handling, zip creation, and gallery/open-folder paths.
- Script save callers: `scripts/loopback.py`, `scripts/outpainting_mk_2.py`, `scripts/sd_upscale.py`, and `scripts/xyz_grid.py` grid/sample save calls, processed image/infotext list assembly, and returned grid ordering.
- API processed-image response helpers: `modules/api/api.py` `processed_js_with_image_paths()`, txt2img/img2img response assembly, `encode_pil_to_base64()`, extras batch response encoding, and PNG info response helper; adjacent response models in `modules/api/models.py` were checked for field contracts.

### Findings and fixes
- Removed a small duplicated per-image infotext recomputation in `process_images_inner()`. The final sample save, returned image metadata/list entry, saved mask, and saved mask-composite now reuse the same `text = infotext(i)` value for that output image.
- Preserved pre-face-restoration and pre-color-correction save calls with their inline `infotext(i)` calls because those saves intentionally happen before later image transformations and callbacks; moving or sharing their infotext across the later output block could change callback-visible timing.
- Preserved `ui_common.save_files()` as a UI/download-specific path. It decodes gallery image data, reconstructs a lightweight processing object for filename pattern expansion, updates optional CSV logs, handles selected-image save semantics, and builds zip downloads; it is similar to generation save plumbing but not a dead wrapper around `images.save_image()`.
- Preserved script-level save calls in loopback, outpainting, SD upscale, and XYZ grid. Each script has different output ordering, seed/infotext alignment, grid/subgrid behavior, and `Processed` assembly semantics; no safe shared helper was found without risking user-visible ordering or extension/script contracts.
- Preserved API response helpers and duplicated-looking txt2img/img2img response mapping. `processed_js_with_image_paths()` enriches the `Processed.js()` JSON with saved paths and OpenClaw diagnostics, while each endpoint must keep its response model fields, `send_images` behavior, and img2img timing metadata.

### Static/dynamic audit map notes
- Generation output chain remains: decoded samples -> script postprocess callbacks -> optional overlay/color correction -> final `text = infotext(i)` -> optional `images.save_image()` -> output image/infotext append -> optional returned/saved mask and mask-composite images.
- Grid chain remains: returned grid inserts its infotext/image at index 0 and shifts `index_of_first_image`; saved grid still uses main prompt/seed and main-grid infotext through the existing `images.save_image()` call.
- UI save/download chain remains: gallery save button -> `save_files()` -> `image_from_url_text()` -> `images.save_image()` under `outdir_save` -> optional CSV row and zip archive -> Gradio file update plus saved filename HTML.
- API response chain remains: endpoint processing -> optional base64 encoding of `processed.images` -> `processed_js_with_image_paths()` JSON in `info` with image path/diagnostic fields -> pydantic response model.
- Compatibility surfaces to keep conservative: `Processed.images` ordering, `Processed.infotexts` alignment, `index_of_first_image`, `already_saved_as` path propagation, `save_files()` Gradio output shape, CSV column order, script save side effects, API `images`/`parameters`/`info` fields, and image metadata preserved by `encode_pil_to_base64()`.

### Validation log
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pycompile-pass58.XXXXXX) python3 -m py_compile modules/processing.py modules/ui_common.py modules/api/api.py scripts/loopback.py scripts/outpainting_mk_2.py scripts/sd_upscale.py scripts/xyz_grid.py` - passed.
- Exact AST duplicate-body scan across `modules/processing.py`, `modules/ui_common.py`, `modules/api/api.py`, `scripts/loopback.py`, `scripts/outpainting_mk_2.py`, `scripts/sd_upscale.py`, and `scripts/xyz_grid.py` reported no duplicate nontrivial function body groups.
- `git diff --check` - passed.
- Focused runtime generation/UI/API save-path tests were not run because they require a running WebUI/API session and model/runtime state; this slice used static inspection plus bytecode compilation.

### Next unchecked scope
- More slices are still needed. Recommended next slice: continue into core image save implementation and filename/path helpers in `modules/images.py`, especially `FilenameGenerator`, filename sanitization, save path/subdirectory selection, existing-info metadata propagation, and text sidecar/callback branches, looking for duplicate output path handling while preserving public save callback contracts and filename-pattern behavior.

## Pass 59 - Core image save implementation and filename/path helpers (2026-06-20)

### Checked scope
- `modules/images.py`: `sanitize_filename_part()`, scheduler/sampler filename helper strings, `FilenameGenerator` replacement map and methods (`get_vae_filename()`, `hasprompt()`, `prompt_no_style()`, `prompt_words()`, `datetime()`, `image_hash()`, `string_hash()`, and `apply()`), `get_next_sequence_number()`, `save_image_with_geninfo()`, `geninfo_to_exif_bytes()`, `save_image()`, `read_info_from_image()`, `image_data()`, and adjacent image read/fix helpers.
- Save-path branches in `save_image()`: `save_to_dirs` and directory pattern expansion, forced filenames, numbered filenames, replacement suffix behavior, callback-rewritten filenames/directories, max filename component truncation, atomic temp-file writes, 4chan downscale export path updates, `already_saved_as`, `.txt` infotext sidecars, and save callbacks.
- Adjacent contracts/tests: `test/test_images_save.py`, `tests/test_save_serialization_contract.py`, and callers of image metadata helpers in `modules/extras.py`, `modules/img2img.py`, `modules/ui_toprow.py`, `modules/ui_extra_networks.py`, `modules/ui_extra_networks_user_metadata.py`, `modules/api/api.py`, and `modules/postprocessing.py`.

### Findings and fixes
- Removed duplicated EXIF metadata gating in `save_image_with_geninfo()`. JPEG/WebP and AVIF save branches now share `enabled_geninfo_exif_bytes()`, preserving the existing behavior that EXIF bytes are generated only when PNG-info metadata is enabled and generation info is present.
- Preserved JPEG/WebP `piexif.insert()` timing after image save and AVIF `exif=` argument semantics. The helper only centralizes the common condition and conversion.
- Preserved `FilenameGenerator` pattern behavior and public replacement names. Even helper-looking methods such as `hasprompt()`, `prompt_no_style()`, `datetime()`, and hash helpers are dynamically reached through filename patterns and must remain conservative.
- Preserved filename sanitization and sequence/path logic. `sanitize_filename_part()` is the shared policy for filename pattern expansions; `get_next_sequence_number()` intentionally scans existing names for legacy numbering behavior; `save_to_dirs` uses the directory filename pattern while sample filenames use sample filename patterns.
- Preserved `save_image()` callback contracts. `before_image_saved_callback()` may rewrite `params.image`, `params.filename`, and `params.pnginfo`; the post-callback directory creation, filename truncation, final `params.filename`, and `image_saved_callback()` behavior are covered by focused tests and were not rearranged.
- Preserved `.txt` sidecar behavior. Generation infotext sidecars are intentionally written after the final image path is known, including duplicate suffixes, callback filename rewrites, max-name truncation, and optional 4chan JPG export path changes.
- Preserved image metadata helpers. `save_image_with_geninfo()`, `read_info_from_image()`, `image_data()`, and `geninfo_to_exif_bytes()` remain directly used by PNG Info, Extras/Postprocessing, img2img image-info import, extra-network previews, API encoding, and top-row upload metadata paths.

### Static/dynamic audit map notes
- Filename pattern chain remains: caller supplies `p`/seed/prompt/image -> `FilenameGenerator.apply()` resolves bracket patterns dynamically -> each replacement sanitizes only the fragment it owns -> `save_image()` applies directory/sample patterns and numbering.
- Save chain remains: path/name selection -> `ImageSaveParams` before-save callback -> post-callback directory creation and filesystem max-name truncation -> atomic temp image write -> optional downscaled JPG export -> `already_saved_as` and optional `.txt` sidecar -> saved callback.
- Metadata chain remains: `existing_info` is merged with current `info` before callbacks; callbacks receive/may mutate `params.pnginfo`; image writers embed PNG text chunks, JPEG/WebP EXIF, AVIF EXIF, or GIF comments according to extension behavior.
- Compatibility surfaces to keep conservative: `FilenameGenerator.replacements` keys and argument parsing, `save_image()` signature/return tuple, callback-visible `ImageSaveParams`, filename numbering and replacement suffix policy, `.txt` sidecar paths, PNG section names, `read_info_from_image()` NovelAI conversion, and `image_data()` upload return shape.

### Validation log
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pycompile-pass59.XXXXXX) python3 -m py_compile modules/images.py test/test_images_save.py tests/test_save_serialization_contract.py` - passed.
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pytest-pass59.XXXXXX) python3 -m pytest -q test/test_images_save.py tests/test_save_serialization_contract.py` - passed: 7 passed, with the existing pytest config warning `Unknown config option: base_url`.
- Exact AST duplicate-body scan across `modules/images.py` reported no duplicate nontrivial function body groups.
- Targeted literal duplicate scan found the prior repeated `opts.enable_pnginfo and geninfo is not None` branch reduced to the single shared helper.
- `git diff --check` - passed.
- Live WebUI image generation/save callbacks, browser upload metadata, and real model/API runtime paths were not exercised because they require a running WebUI/API session and runtime state.

### Next unchecked scope
- More slices are still needed. Recommended next slice: continue into lower/adjacent image utility code and metadata consumers not deeply covered here, especially PNG Info/extras upload paths (`modules/extras.py`, `modules/img2img.py`, `modules/ui_toprow.py`, `modules/ui_extra_networks*.py`, `modules/api/api.py` image encode/decode metadata branches), looking for duplicated metadata extraction/encoding and dead image-info compatibility helpers while preserving UI/API response contracts.


## Pass 60 - Adjacent metadata consumers and PNG Info/extras upload/API paths (2026-06-20)

### Checked scope
- `modules/extras.py`: `run_pnginfo()`, image metadata extraction/display, and model-merger metadata helpers adjacent to the PNG Info tab.
- `modules/img2img.py`: batch image upload/directory handling, `use_png_info` metadata import, alternate PNG-info directory behavior, override-setting/model-hash handling, and init image/mask selection.
- `modules/ui_toprow.py`: prompt-image upload metadata import through `modules.images.image_data()` and prompt file-change wiring.
- `modules/ui_extra_networks.py`: extra-network preview save compatibility handler, metadata/cover-image endpoints, preview directory allow checks, and gallery preview metadata extraction.
- `modules/ui_extra_networks_user_metadata.py` plus checkpoint/hypernetwork/textual inversion adjacent modules: user metadata editor preview replacement, local preview update, and page-specific metadata editor contracts.
- `modules/api/api.py`: `decode_base64_to_image()`, `decode_extras_batch_images()`, `encode_pil_to_base64()`, extras single/batch upload paths, `pnginfoapi()`, `apply_infotext()`, and processed-image API response helpers.
- Adjacent contracts/tests: `tests/test_extra_networks_path_contract.py`, `tests/test_extra_networks_metadata_contract.py`, `tests/test_api_extension_item_contract.py`, and prior pass notes for `modules/images.py` metadata helpers.

### Findings and fixes
- Extracted duplicated extra-network gallery preview metadata extraction into `ui_extra_networks.read_gallery_image_metadata()`. The legacy save-preview handler and the user-metadata preview replacement handler now share the same index clamping, Gradio gallery payload decode, and `read_info_from_image()` call.
- Removed the now-unused `infotext_utils` import from `modules/ui_extra_networks_user_metadata.py`.
- Preserved the extra-network preview compatibility wrapper and user-metadata editor method as separate UI handlers. They intentionally have different empty-gallery messages, output shapes, allow-directory checks, lister refresh behavior, card refresh behavior, and Gradio callback wiring.
- Preserved `extras.run_pnginfo()` and `api.pnginfoapi()` as separate UI/API contract surfaces. Both read metadata from an image, but the UI path renders plaintext-safe HTML and returns the legacy three-value Gradio tuple, while the API path parses parameters, fires `infotext_pasted_callback()`, and returns the pydantic response shape with `info`, `items`, and `parameters`.
- Preserved API image encode/decode helpers. `decode_base64_to_image()` owns URL/data/base64 request validation and HTTP errors; `decode_extras_batch_images()` intentionally skips only invalid batch entries; `encode_pil_to_base64()` preserves API-selected output format and metadata propagation for PNG/JPEG/WebP responses.
- Preserved `images.image_data()` for prompt-image upload compatibility. It accepts raw file bytes from the hidden prompt upload control and falls back to UTF-8 text decode; this is distinct from API/base64 and Gradio gallery payload decoding.
- Preserved img2img batch PNG-info import logic. It has batch-specific defaults, alternate metadata image directory support, selected-property filtering, prompt/negative-prompt append semantics, and checkpoint override fallback behavior that should not be merged with PNG Info/API parsing helpers.

### Static/dynamic audit map notes
- Prompt-image upload chain remains: hidden top-row file input -> `modules.images.image_data()` -> prompt textbox text plus prompt file reset.
- PNG Info UI chain remains: image input -> `extras.run_pnginfo()` -> `images.read_info_from_image()` -> HTML infotext blocks plus geninfo text output.
- API PNG-info chain remains: base64/URL image decode -> `images.read_info_from_image()` -> generation-parameter parse -> paste callback -> `PNGInfoResponse`.
- Extra-network preview save chain now shares only gallery image/metadata extraction: selected gallery index -> `read_gallery_image_metadata()` -> caller-specific permission/update behavior -> `save_image_with_geninfo()`.
- Compatibility surfaces to keep conservative: Gradio callback input/output tuple shapes, JavaScript selected-gallery index wiring, preview allow-directory assertions, API error details/status codes, API response field names, `infotext_pasted_callback()` side effects, and img2img batch PNG-info property semantics.

### Validation log
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pycompile-pass60.XXXXXX) python3 -m py_compile modules/extras.py modules/img2img.py modules/ui_toprow.py modules/ui_extra_networks.py modules/ui_extra_networks_user_metadata.py modules/ui_extra_networks_checkpoints.py modules/ui_extra_networks_checkpoints_user_metadata.py modules/ui_extra_networks_hypernets.py modules/ui_extra_networks_textual_inversion.py modules/api/api.py` - passed.
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pytest-pass60.XXXXXX) python3 -m pytest -q tests/test_extra_networks_path_contract.py tests/test_extra_networks_metadata_contract.py tests/test_api_extension_item_contract.py` - passed: 6 passed, with the existing pytest config warning `Unknown config option: base_url`.
- Exact AST duplicate-body scan across `modules/extras.py`, `modules/img2img.py`, `modules/ui_toprow.py`, `modules/ui_extra_networks.py`, `modules/ui_extra_networks_user_metadata.py`, `modules/ui_extra_networks_checkpoints.py`, `modules/ui_extra_networks_checkpoints_user_metadata.py`, `modules/ui_extra_networks_hypernets.py`, `modules/ui_extra_networks_textual_inversion.py`, and `modules/api/api.py` reported no duplicate nontrivial function body groups.
- `git diff --check` - passed.
- Live browser/UI upload, paste/send-to, and API image upload calls were not exercised because they require a running WebUI/API session and runtime state.

### Next unchecked scope
- More slices are still needed. Recommended next slice: continue into infotext/paste/send-to plumbing around `modules/infotext_utils.py`, `modules/ui.py`, `modules/ui_common.py`, `modules/generation_parameters_copypaste.py`, and txt2img/img2img paste-field registration, looking for duplicated paste-field mapping, stale compatibility wrappers, and dead image-info helpers while preserving extension-visible paste/send-to behavior and UI/API infotext contracts.



## Pass 61 - Infotext paste/send-to UI plumbing (2026-06-20)

### Checked scope
- `modules/infotext_utils.py`: compatibility alias for `modules.generation_parameters_copypaste`, `ParamBinding`, `PasteField`, `reset()`, `image_from_url_text()`, `add_paste_fields()`, `create_buttons()`, legacy `bind_buttons()`, `register_paste_params_button()`, `connect_paste_params_buttons()`, `send_image_and_dimensions()`, infotext parse/backcompat defaults, inpaint infotext conversion helpers, override-setting mapping helpers, and `connect_paste()`.
- `modules/ui_common.py`: `update_generation_info()`, `save_files()` image decode/import use, `OutputPanel`, `create_output_panel()`, send-to button construction, gallery selected-infotext update, save/save-zip wiring, paste field name propagation, and output-panel paste-button registration.
- `modules/ui.py`: `create_output_panel()` wrapper, `image_from_url_text()` import/use, txt2img/img2img `PasteField` registration, PNG Info send-to registration, and final `connect_paste_params_buttons()` call.
- Adjacent API/tests/contracts: `modules/api/api.py` `api_infotext_value_for_field()` and script arg infotext handling; `test/test_infotext_paste_bindings.py`, `test/test_infotext_api_mappings.py`, and `tests/test_save_serialization_contract.py`.
- Compatibility-module check: there is no physical `modules/generation_parameters_copypaste.py`; the old import path is intentionally served by `sys.modules['modules.generation_parameters_copypaste'] = sys.modules[__name__]` in `modules/infotext_utils.py`.

### Findings and fixes
- No safe source-code removals or consolidations were made in this slice.
- Preserved the `modules.generation_parameters_copypaste` alias. It is a stale-looking name but remains an extension compatibility surface; removing or replacing it with a real wrapper file would change import behavior for extensions that import the old module after `infotext_utils` is loaded.
- Preserved `bind_buttons()` and `create_buttons()` as compatibility/UI helper surfaces. `bind_buttons()` is explicitly marked old compatibility and still translates legacy callers into `ParamBinding`; `create_buttons()` is still used by the PNG Info tab to build its send-to buttons.
- Preserved `modules.ui.create_output_panel()` as a wrapper around `ui_common.create_output_panel()`. It is thin, but it keeps the historical `modules.ui` API surface while the implementation lives in `ui_common`.
- Preserved duplicated-looking txt2img/img2img `PasteField` lists. They share prompt/style/size fields, but the component instances, high-res-only txt2img fields, img2img/inpaint mask fields, script infotext fields, API names, and extension-visible `modules.ui.txt2img_paste_fields`/`img2img_paste_fields` assignments make a shared constructor riskier than the duplication it would remove.
- Preserved send-to button registration in `ui_common.create_output_panel()` and PNG Info registration in `ui.py`. They look similar but differ in source components, source tab copying, generated-info source text, image transfer behavior, and output-panel tab semantics.
- Preserved `image_from_url_text()` and `send_image_and_dimensions()`. They are still used by UI save/download, gallery send-to, PNG Info image send-to, and tests; their list/file/base64 handling is distinct from API base64 decode helpers.
- Preserved `api_infotext_value_for_field()` as API-specific conversion logic. It mirrors some UI paste coercion concerns, but it consumes `PasteField` metadata for pydantic/script defaults and must unwrap Gradio update dictionaries without depending on live Gradio components.

### Static/dynamic audit map notes
- UI paste chain remains: tab UI registers `PasteField` lists with `add_paste_fields()` -> top-row/PNG Info/output-panel buttons register `ParamBinding`s -> `connect_paste_params_buttons()` wires image transfer, infotext parse, same-name source-tab copy, and final tab switch callbacks in that order.
- Gallery send-to chain remains: output-panel gallery -> `extract_image_from_gallery` JS -> `image_from_url_text()` or `send_image_and_dimensions()` -> destination init image plus optional width/height updates when `shared.opts.send_size` permits.
- Source-tab send-to chain remains: txt2img output panel can copy a filtered allowlist of fields to img2img/inpaint/extras using matching infotext names; script-provided `paste_field_names` extend the built-in prompt/steps/seed allowlist.
- PNG Info chain remains: image upload -> `extras.run_pnginfo()` fills hidden generation-info textbox -> PNG Info send-to buttons use that textbox as source text and the image component as source image.
- Compatibility surfaces to keep conservative: old module alias, legacy `bind_buttons()` behavior, `PasteField` tuple shape plus `.api/.component/.label/.function`, global `paste_fields` and `registered_param_bindings`, extension-visible `modules.ui.txt2img_paste_fields`/`img2img_paste_fields`, callback ordering, generated JS function names, Gradio input/output tuple shapes, `infotext_pasted_callback()` side effects, and `shared.opts.send_seed/send_size` behavior.

### Validation log
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pycompile-pass61.XXXXXX) python3 -m py_compile modules/infotext_utils.py modules/ui.py modules/ui_common.py modules/api/api.py test/test_infotext_paste_bindings.py test/test_infotext_api_mappings.py tests/test_save_serialization_contract.py` - passed.
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pytest-pass61.XXXXXX) python3 -m pytest -q test/test_infotext_paste_bindings.py test/test_infotext_api_mappings.py tests/test_save_serialization_contract.py` - passed: 12 passed, with the existing pytest config warning `Unknown config option: base_url`.
- Exact AST duplicate-body scan across `modules/infotext_utils.py`, `modules/ui.py`, `modules/ui_common.py`, `modules/api/api.py`, `test/test_infotext_paste_bindings.py`, and `test/test_infotext_api_mappings.py` reported no duplicate nontrivial function body groups.
- Targeted reference scans confirmed paste/send-to registration is concentrated in `modules/infotext_utils.py`, `modules/ui_common.py`, `modules/ui.py`, and focused infotext tests; no separate `modules/generation_parameters_copypaste.py` file exists.
- `git diff --check` - passed.
- Live browser send-to/paste interactions were not exercised because they require a running WebUI session and browser/UI runtime state.

### Next unchecked scope
- More slices are still needed. Recommended next slice: continue into script and extension infotext/paste integration surfaces, especially `modules/scripts.py` script `infotext_fields`/`paste_field_names`, sampler/seed script UI controls that feed paste fields, and any extension-facing script callback contracts, looking for duplicate script paste metadata while preserving script API and extension hook behavior.


## Pass 62 - Script and extension infotext/paste integration surfaces (2026-06-20)

### Checked scope
- `modules/scripts.py`: `Script.infotext_fields`, `Script.paste_field_names`, component callback registration helpers, `ScriptRunner.initialize_scripts()`, `apply_on_before_component_callbacks()`, `create_script_ui_inner()`, `setup_ui()`, `before_component()`, and `after_component()`.
- `modules/processing_scripts/sampler.py`: sampler/steps/scheduler UI controls and `PasteField` metadata consumed by txt2img/img2img paste and API infotext mapping.
- `modules/processing_scripts/seed.py`: seed/subseed controls, variation visibility paste metadata, seed-resize paste metadata, and generation-info reuse button hookup via `on_after_component()`.
- `modules/processing_scripts/refiner.py`: refiner accordion/checkpoint/switch-at paste metadata and checkpoint lookup behavior.
- `scripts/xyz_grid.py`: selectable script infotext metadata for X/Y/Z axis type/value/dropdown restoration.
- `modules/script_callbacks.py`: extension-facing `on_before_component()`, `on_after_component()`, and `on_infotext_pasted()` registration/callback contracts adjacent to script paste integration.
- Adjacent wiring already touched by script metadata: `modules/ui.py` txt2img/img2img `PasteField` list expansion, `modules/ui_common.py` output-panel `paste_field_names` propagation, and `modules/infotext_utils.py` source-tab field-name filtering.
- Focused contracts/tests: `test/test_infotext_paste_bindings.py` and `test/test_infotext_api_mappings.py`.

### Findings and fixes
- No safe source-code removals or consolidations were made in this slice.
- Preserved `Script.infotext_fields` and `Script.paste_field_names` as public script API surfaces. Built-in and extension scripts populate them from `ui()`, `ScriptRunner.create_script_ui_inner()` aggregates them into tab-level paste/API metadata, and output-panel send-to buttons consume script `paste_field_names` for same-name field copying.
- Preserved sampler, seed, and refiner `PasteField` declarations. The entries look small and repeated across txt2img/img2img because each built-in script instance owns tab-specific controls; sharing or moving them into static metadata would break component identity, API names, callable paste conversion, or UI visibility updates.
- Preserved seed `on_after_component()` generation-info wiring. The callbacks target `generation_info_{tabname}` after output-panel construction and connect reuse-seed/reuse-subseed buttons to parsed generation infotext; this is active UI behavior, not dead callback plumbing.
- Preserved xyz-grid `infotext_fields`. The tuple restores selected script UI controls from infotext when the selectable script is active, including dropdown/list conversion for axis values; it is distinct from built-in always-visible sampler/seed/refiner metadata.
- Preserved extension-facing component callbacks in both `modules/scripts.py` and `modules/script_callbacks.py`. They are separate contract surfaces: per-script elem-id callbacks receive `OnComponent`, while global script callbacks receive raw component/kwargs through ordered callback lists.
- Preserved `modules.ui` paste-field expansion and `ui_common` script field-name propagation. The former registers destination paste/API fields; the latter chooses which script fields can be copied by output-panel send-to buttons. They are paired but not redundant.

### Static/dynamic audit map notes
- Script paste metadata chain remains: script `ui()` builds live Gradio components -> script sets `infotext_fields`/`paste_field_names` -> `ScriptRunner.create_script_ui_inner()` appends them to the tab runner -> `modules/ui.py` expands runner `infotext_fields` into tab paste fields -> `modules/infotext_utils.add_paste_fields()` stores them and exposes legacy `modules.ui.*_paste_fields` -> UI/API paste consumers resolve values by labels/functions/API names.
- Send-to script field chain remains: output panel chooses `scripts_txt2img.paste_field_names` or `scripts_img2img.paste_field_names` -> `ParamBinding.paste_field_names` extends the built-in allowlist -> `connect_paste_params_buttons()` copies same-name source/destination fields.
- Component callback chain remains: scripts may enqueue elem-id callbacks during `show()`/`ui()` -> `apply_on_before_component_callbacks()` registers and clears pending entries -> component patch hooks call per-script callbacks around component construction while global `script_callbacks` hooks preserve extension-wide behavior.
- Compatibility surfaces to keep conservative: `Script` attributes, `PasteField` tuple/API metadata, `OnComponent`, global component callbacks, callback ordering/user sort behavior, script group visibility restoration, `generation_info_{tabname}` elem IDs, and `modules.ui.txt2img_paste_fields`/`img2img_paste_fields` legacy exposure.

### Validation log
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pycompile-pass62.XXXXXX) python3 -m py_compile modules/scripts.py modules/processing_scripts/sampler.py modules/processing_scripts/seed.py modules/processing_scripts/refiner.py scripts/xyz_grid.py modules/script_callbacks.py modules/ui.py modules/ui_common.py modules/infotext_utils.py test/test_infotext_paste_bindings.py test/test_infotext_api_mappings.py` - passed.
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pytest-pass62.XXXXXX) python3 -m pytest -q test/test_infotext_paste_bindings.py test/test_infotext_api_mappings.py` - passed: 9 passed, with the existing pytest config warning `Unknown config option: base_url`.
- Exact AST duplicate-body scan across `modules/scripts.py`, `modules/processing_scripts/sampler.py`, `modules/processing_scripts/seed.py`, `modules/processing_scripts/refiner.py`, `scripts/xyz_grid.py`, `modules/script_callbacks.py`, `modules/ui.py`, `modules/ui_common.py`, and `modules/infotext_utils.py` reported no duplicate nontrivial function body groups.
- Targeted reference scans confirmed active paste/script metadata declarations are limited to the built-in processing scripts, xyz-grid selectable script, `modules/scripts.py` aggregation, `modules/ui.py` destination paste-field registration, and `modules/ui_common.py` send-to field-name propagation.
- `git diff --check` - passed.
- Live browser paste/send-to/reuse-seed interactions were not exercised because they require a running WebUI session and browser/UI runtime state.

### Next unchecked scope
- More slices are still needed. Recommended next slice: continue into generation-parameter production and script infotext emission paths, especially where `Processed.infotexts`, `processing.create_infotext()`, script extra generation params, sampler/scheduler/refiner/seed metadata emission, and API processed responses assemble infotext, looking for duplicate metadata formatting or stale script-parameter compatibility while preserving PNG/API/UI infotext contracts.


## Pass 63 - Generation-parameter production and script infotext emission paths (2026-06-20)

### Checked scope
- `modules/processing.py`: `Processed.__init__()`, `Processed.js()`, `Processed.infotext()`, `create_infotext()`, `process_images()` infotext accumulation, params.txt emission, per-image/grid infotext construction, mask/mask-composite infotext alignment, and `extra_generation_params` injection points in txt2img/img2img setup.
- `modules/sd_samplers_common.py`: `is_sampler_using_eta_noise_seed_delta()`, sampler `initialize()`, eta/sigma metadata emission, `add_infotext()`, and refiner metadata emission in `apply_refiner()`.
- `modules/sd_samplers_kdiffusion.py` and `modules/sd_samplers_timesteps.py`: scheduler/discard/extra-noise metadata emission, sample/img2img sample infotext hooks, and sampler argument reflection branches.
- Built-in script metadata emitters: `modules/processing_scripts/sampler.py`, `modules/processing_scripts/seed.py`, `modules/processing_scripts/refiner.py`, `scripts/img2imgalt.py`, `scripts/loopback.py`, `scripts/sd_upscale.py`, `scripts/prompts_from_file.py`, and `scripts/xyz_grid.py` grid/subgrid infotext creation.
- API processed response contracts: `modules/api/api.py` `processed_js_with_image_paths()`, txt2img/img2img API response assembly, and `modules/api/models.py` response field contracts.
- Adjacent focused contracts/tests: `tests/test_processing_auxiliary_infotext_alignment.py`, `tests/test_save_serialization_contract.py`, `test/test_openclaw_multi_sampler.py`, `test/test_cfg_denoiser_callbacks.py`, `test/test_infotext_api_mappings.py`, and `test/test_infotext_paste_bindings.py`.

### Findings and fixes
- No safe source-code removals or consolidations were made in this slice.
- Preserved `Processed.infotext()` even though it is a thin wrapper around `create_infotext()`. It is still used for params.txt fallback and empty-result fallback paths, and keeping it preserves the `Processed` helper surface without changing ordering or default comment behavior.
- Preserved `create_infotext()` as the single central formatting path for emitted generation metadata. It owns ordered field emission, `infotext_utils.quote()` formatting, prompt/negative-prompt layout, list/callable extra-param expansion, option-controlled model/VAE/version/user fields, and script/extension-visible `p.extra_generation_params` behavior.
- Preserved the apparent duplicate sampler `self.add_infotext(p)` calls at the end of K-diffusion and timestep `sample()`/`sample_img2img()`. They sit on distinct sampler execution paths and are the shared point where pad-condition metadata is emitted after sampler initialization/callback state is known.
- Preserved separate scheduler/eta/sigma/refiner extra-param emitters. They look like scattered metadata writes, but each is tied to runtime sampler/refiner decisions, option default comparisons, hires-vs-base pass handling, or model-swap state that is not available to `create_infotext()` without broad behavioral changes.
- Preserved script-specific extra generation params in img2img alternative, loopback, SD upscale, prompts-from-file, and xyz-grid. These scripts intentionally mutate `p.extra_generation_params` or `Processed.infotexts` to preserve per-script grid/sample infotext contracts, initial-info propagation, and subgrid/main-grid axis labels.
- Preserved API response assembly through `processed_js_with_image_paths(processed)`. The API intentionally returns `Processed.js()` plus image paths and OpenClaw diagnostics as a JSON string in `info`; changing it to parsed generation parameters or per-image infotext-only data would break existing API clients.

### Static/dynamic audit map notes
- Emission chain remains: processing setup/script hooks populate `p.extra_generation_params` -> sampler/refiner/runtime branches add actual runtime metadata -> `create_infotext()` formats ordered fields and expands list/callable extras -> `process_images()` stores one infotext per returned image plus optional grid -> `Processed.js()` exposes `infotexts` and raw `extra_generation_params` to UI/API consumers.
- Grid/auxiliary image chain remains: each returned sample/mask/mask-composite gets the same sample infotext for alignment; optional returned grids insert a main-prompt infotext at index 0 and shift `index_of_first_image`; scripts such as xyz-grid and SD upscale override or replace those infotexts where their grid/sample layout differs from core generation.
- API chain remains: txt2img/img2img API calls process images under the queue lock -> optional base64 encoding preserves image metadata -> response `info` is `Processed.js()` augmented with saved image paths and OpenClaw timing/cache diagnostics.
- Compatibility surfaces to keep conservative: field ordering in `create_infotext()`, prompt/negative-prompt line format, `k if k == v` flag formatting, `infotext_utils.quote()` use, list/callable extra-param expansion semantics, `p.extra_generation_params` mutability for scripts/extensions, `Processed.js()` field names, `Processed.infotexts` length/index alignment, and API response `info` as a JSON string.

### Validation log
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pycompile-pass63.XXXXXX) python3 -m py_compile modules/processing.py modules/sd_samplers_common.py modules/sd_samplers_kdiffusion.py modules/sd_samplers_timesteps.py modules/api/api.py modules/api/models.py modules/processing_scripts/sampler.py modules/processing_scripts/seed.py modules/processing_scripts/refiner.py scripts/img2imgalt.py scripts/loopback.py scripts/sd_upscale.py scripts/prompts_from_file.py scripts/xyz_grid.py tests/test_processing_auxiliary_infotext_alignment.py tests/test_save_serialization_contract.py test/test_openclaw_multi_sampler.py test/test_cfg_denoiser_callbacks.py test/test_infotext_api_mappings.py test/test_infotext_paste_bindings.py` - passed.
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pytest-pass63-subset.XXXXXX) python3 -m pytest -q tests/test_processing_auxiliary_infotext_alignment.py tests/test_save_serialization_contract.py test/test_infotext_api_mappings.py test/test_infotext_paste_bindings.py` - passed: 22 passed, with the existing pytest config warning `Unknown config option: base_url`.
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pytest-pass63.XXXXXX) python3 -m pytest -q tests/test_processing_auxiliary_infotext_alignment.py tests/test_save_serialization_contract.py test/test_openclaw_multi_sampler.py test/test_cfg_denoiser_callbacks.py test/test_infotext_api_mappings.py test/test_infotext_paste_bindings.py` - blocked during collection by missing runtime dependency: `ModuleNotFoundError: No module named 'torch'` in `test/test_cfg_denoiser_callbacks.py`.
- Exact AST duplicate-body scan across the checked processing, sampler, API, and script files reported no duplicate nontrivial function body groups.
- Targeted reference scans confirmed `create_infotext()` remains the only central generation-parameter formatter, while sampler/script writes feed `p.extra_generation_params` rather than formatting metadata text themselves.
- `git diff --check` - passed.
- Live WebUI/API generation and PNG save/readback were not exercised because they require a running WebUI/API session and model runtime state.

### Next unchecked scope
- More slices are still needed. Recommended next slice: continue into generation-parameter parsing/backcompat/versioning and override-settings ingestion, especially `modules/infotext_utils.py` `parse_generation_parameters()`, `modules/infotext_versions.py`, txt2img/img2img infotext override paths, and API `infotext` request handling, looking for duplicated stale compatibility parsing while preserving old infotext import and extension callback behavior.


## Pass 64 - Generation-parameter parsing/backcompat/versioning and override ingestion (2026-06-20)

### Checked scope
- `modules/infotext_utils.py`: `quote()`, `unquote()`, `parse_generation_parameters()`, `restore_old_hires_fix_params()`, inpaint label conversion helpers, `infotext_to_setting_name_mapping`, `infotext_setting_name_mapping()`, `create_override_settings_dict()`, `get_override_settings()`, and `connect_paste()` override dropdown ingestion.
- `modules/infotext_versions.py`: version constants, `parse_version()`, and `backcompat()` compatibility switches for old prompt editing, old pad conds, downcast alpha behavior, and old refiner switch semantics.
- UI request paths: `modules/txt2img.py` `txt2img_create_processing()` and `txt2img_upscale()`, plus `modules/img2img.py` `process_batch()` PNG-info import and `img2img()` override-setting ingestion.
- API request paths: `modules/api/api.py` `api_infotext_value_for_field()`, `Api.apply_infotext()`, `_prepare_generation_api_request()` callers, and `pnginfoapi()` parsing/callback behavior.
- Adjacent option/test contracts: `modules/shared_options.py` `OptionInfo(..., infotext=...)` mappings and backcompat options, `test/test_infotext_api_mappings.py`, `test/test_infotext_paste_bindings.py`, `test/test_txt2img.py`, `test/test_img2img.py`, `tests/test_save_serialization_contract.py`, and `tests/test_processing_auxiliary_infotext_alignment.py`.

### Findings and fixes
- No safe source-code removals or consolidations were made in this slice.
- Preserved `parse_generation_parameters()` as the central parser/backcompat path. It is still shared by UI paste, PNG Info API, txt2img upscale seed reuse, img2img batch PNG-info import, save/update helpers, and API request infotext ingestion; its default filling and `infotext_versions.backcompat()` call are compatibility behavior rather than dead code.
- Preserved `infotext_versions.py` as a separate versioning helper. The version constants and backcompat switches are small but active through `parse_generation_parameters()` and map directly to `OptionInfo(..., infotext=...)` settings that reproduce old generation behavior.
- Preserved both `create_override_settings_dict()` and `get_override_settings()`. The UI generation path consumes explicit dropdown text and should cast all selected overrides; paste/API paths derive candidate overrides from parsed infotext, skip already-handled paste fields, ignore unchanged current values, and respect `disable_weights_auto_swap` for checkpoint import.
- Preserved API-specific infotext request handling in `Api.apply_infotext()`. It intentionally fills only unset request fields, preserves explicit `request.override_settings`, unwraps Gradio update dictionaries for script fields, records mentioned script args, and does not fire extension paste callbacks for generation requests. `pnginfoapi()` remains the callback-firing parse surface for PNG-info inspection.
- Preserved img2img batch PNG-info import logic. It imports only user-selected properties, appends prompt text rather than replacing it, handles alternate metadata-image directories, and applies checkpoint override fallback separately from UI/API override-setting dropdown ingestion.
- Preserved old/default parser fill-ins for CLIP skip, hires fields, inpaint labels, RNG, scheduler sigma fields, VAE encode/decode, FP8/MXFP8 fields, emphasis, and refiner switch mode. These keys feed paste fields, API mappings, option overrides, and old infotext reproduction; removing them would change imports of older or partial infotext.
- Left `re_hypernet_hash` untouched despite no in-repo references. It is an internal-looking leftover, but `modules.infotext_utils` has explicit old-module compatibility exposure and extension import history, so removing a module-level regex was not strong enough to satisfy the safe-removal bar for this audit.

### Static/dynamic audit map notes
- Parser chain remains: raw infotext -> prompt/negative prompt split -> comma-separated `key: value` parse with quoted value support and size fanout -> style extraction -> default/backcompat fill-ins -> `infotext_versions.backcompat()` -> user skip-field removal.
- UI paste chain remains: parsed params -> per-component paste fields -> callable conversion helpers for legacy labels -> optional override dropdown populated by `get_override_settings(params, skip_fields=already_handled_fields)`.
- UI generation override chain remains: override dropdown text pairs -> `create_override_settings_dict()` -> processing `override_settings` for txt2img/img2img, independent of raw infotext parsing.
- API generation infotext chain remains: request `infotext` -> parser -> unset pydantic request fields/script args only -> missing `override_settings` initialized -> derived option overrides added without replacing explicit request overrides.
- PNG Info API chain remains: image metadata -> parser -> `infotext_pasted_callback()` -> response `info`, `items`, and parsed `parameters`.
- Compatibility surfaces to keep conservative: old infotext field names and defaults, `Version` parsing of prerelease/build strings, `auto_backcompat`, `infotext_skip_pasting`, `infotext_styles`, `OptionInfo.infotext` mappings, legacy `infotext_to_setting_name_mapping`, extension `infotext_pasted_callback()` side effects, API request field precedence, and Gradio update dictionary unwrapping.

### Validation log
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pycompile-pass64.XXXXXX) python3 -m py_compile modules/infotext_utils.py modules/infotext_versions.py modules/txt2img.py modules/img2img.py modules/api/api.py modules/shared_options.py test/test_infotext_api_mappings.py test/test_infotext_paste_bindings.py test/test_txt2img.py test/test_img2img.py tests/test_save_serialization_contract.py tests/test_processing_auxiliary_infotext_alignment.py` - passed.
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pytest-pass64-focused.XXXXXX) python3 -m pytest -q test/test_infotext_api_mappings.py test/test_infotext_paste_bindings.py tests/test_save_serialization_contract.py tests/test_processing_auxiliary_infotext_alignment.py` - passed: 22 passed, with the existing pytest config warning `Unknown config option: base_url`.
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pytest-pass64.XXXXXX) python3 -m pytest -q test/test_infotext_api_mappings.py test/test_infotext_paste_bindings.py test/test_txt2img.py test/test_img2img.py tests/test_save_serialization_contract.py tests/test_processing_auxiliary_infotext_alignment.py` - partially ran but blocked on live API fixture setup: 22 passed, 19 setup errors because `base_url` fixture was not available for `test/test_txt2img.py` and `test/test_img2img.py`; existing warning `Unknown config option: base_url` was also emitted.
- Exact AST duplicate-body scan across `modules/infotext_utils.py`, `modules/infotext_versions.py`, `modules/txt2img.py`, `modules/img2img.py`, `modules/api/api.py`, `test/test_infotext_api_mappings.py`, `test/test_infotext_paste_bindings.py`, `test/test_txt2img.py`, and `test/test_img2img.py` reported no duplicate nontrivial function body groups.
- Targeted reference scans confirmed parser/version/override-setting use is concentrated in infotext utilities, txt2img/img2img UI generation paths, API infotext handling, PNG-info parsing, and focused infotext tests.
- `git diff --check` - passed.
- Live WebUI paste/API generation/PNG-info requests were not exercised because they require a running WebUI/API session and model runtime state.

### Next unchecked scope
- More slices are still needed. Recommended next slice: continue into infotext option inventory and option-name discovery surfaces, especially `modules/shared_items.py` `get_infotext_names()`, `modules/shared_options.py` infotext-bearing options, settings UI skip-pasting choices, and any duplicated option-to-infotext registries, looking for stale option mappings while preserving old import names and settings/API compatibility.


## Pass 65 - Infotext option inventory and option-name discovery surfaces (2026-06-20)

### Checked scope
- `modules/shared_items.py`: `get_infotext_names()` settings-option label inventory, paste-field label inventory, ordering/dedup behavior, and settings UI choice supplier for skipped pasted fields.
- `modules/shared_options.py`: `OptionInfo(..., infotext=...)` labels, `infotext_skip_pasting`, `infotext_styles`, `disable_weights_auto_swap`, backcompat infotext-bearing options, sampler/optimization infotext-bearing options, and adjacent settings/UI option choices.
- `modules/infotext_utils.py`: `infotext_to_setting_name_mapping`, `infotext_setting_name_mapping()`, `parse_generation_parameters()` skip-field removal, `get_override_settings()` skip-field handling, and override dropdown already-handled field filtering.
- Paste-field and script label producers adjacent to option-name discovery: `modules/ui.py` txt2img/img2img paste fields, `modules/scripts.py` script metadata aggregation, `modules/processing_scripts/sampler.py`, `modules/processing_scripts/seed.py`, `modules/processing_scripts/refiner.py`, `scripts/xyz_grid.py`, and built-in extension option/script infotext declarations found by targeted scans.
- Focused tests/contracts: `test/test_infotext_api_mappings.py`, `test/test_infotext_paste_bindings.py`, and prior pass contracts around paste-field/API mapping behavior.

### Findings and fixes
- Consolidated `shared_items.get_infotext_names()` to use `infotext_utils.infotext_setting_name_mapping()` for settings-derived labels instead of directly walking `shared.opts.data_labels`. This removes a duplicated option-to-infotext registry walk and makes the skip-pasting settings dropdown include labels added through the legacy `infotext_to_setting_name_mapping` compatibility list as well as modern `OptionInfo(..., infotext=...)` labels.
- Added a focused AST-based regression test proving `get_infotext_names()` combines canonical option/legacy setting mappings with string paste-field labels while deduplicating labels already present in both sources.
- Preserved `infotext_to_setting_name_mapping` despite being empty in-tree. It remains a documented extension/backcompat mutation point, is consumed by `infotext_setting_name_mapping()`, and now feeds the skip-pasting choice inventory too.
- Preserved the settings UI `infotext_skip_pasting` option and its dynamic `shared_items.get_infotext_names()` choices. It is active parser behavior: `parse_generation_parameters()` removes user-selected labels after compatibility/default fill-ins.
- Preserved duplicate-looking txt2img/img2img paste labels such as `Prompt`, `Negative prompt`, `CFG scale`, `Size-1`, `Size-2`, `Batch size`, and `Denoising strength`. They are duplicated across tab-specific live components by design and are used for tab paste/API mapping and send-to same-name copying; they are not a safe static registry consolidation.
- Preserved extension infotext-bearing option registrations, especially Hypertile and Extra Options Section integration. These are dynamic settings/script surfaces and must remain extension-visible.

### Static/dynamic audit map notes
- Skip-pasting choice chain now remains: settings UI calls `shared_items.get_infotext_names()` -> canonical `infotext_setting_name_mapping()` supplies modern `OptionInfo.infotext` labels plus legacy mapping entries -> registered tab paste fields add live UI/script labels -> `infotext_skip_pasting` stores selected labels -> `parse_generation_parameters()` removes those keys from parsed infotext.
- Override-setting chain remains: `infotext_setting_name_mapping()` is the single settings label registry for UI dropdown override parsing, paste/API override detection, Extra Options Section script paste metadata, and now skip-pasting choice inventory.
- Paste-field label chain remains: tab registration exposes only string labels to the skip-pasting dropdown; callable paste functions are intentionally excluded because they do not map to a stable pasted infotext field name.
- Compatibility surfaces to keep conservative: `OptionInfo.infotext`, empty-but-mutable `infotext_to_setting_name_mapping`, `infotext_setting_name_mapping()` ordering, `paste_fields` tuple shape, `infotext_skip_pasting` stored values, old infotext labels/defaults, and extension-added settings/script infotext fields.

### Validation log
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pycompile-pass65.XXXXXX) python3 -m py_compile modules/shared_items.py modules/shared_options.py modules/infotext_utils.py modules/ui.py modules/ui_common.py modules/scripts.py modules/processing_scripts/sampler.py modules/processing_scripts/seed.py modules/processing_scripts/refiner.py test/test_infotext_api_mappings.py test/test_infotext_paste_bindings.py` - passed.
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pytest-pass65.XXXXXX) python3 -m pytest -q test/test_infotext_api_mappings.py test/test_infotext_paste_bindings.py` - passed: 10 passed, with the existing pytest config warning `Unknown config option: base_url`.
- Exact AST duplicate-body scan across `modules/shared_items.py`, `modules/shared_options.py`, `modules/infotext_utils.py`, `test/test_infotext_api_mappings.py`, and `test/test_infotext_paste_bindings.py` reported no duplicate nontrivial function body groups.
- Targeted AST/static scans found 42 unique in-tree `OptionInfo(..., infotext=...)` labels in `modules/shared_options.py` and no duplicate labels there; duplicate string `PasteField` labels were limited to expected txt2img/img2img tab-paired fields.
- `git diff --check` - passed.
- Live settings UI dropdown behavior was not exercised because it requires a running WebUI session; behavior is covered by the focused inventory unit test and py_compile.

### Next unchecked scope
- More slices are still needed. Recommended next slice: continue into infotext/style/prompt parsing and prompt-style extraction surfaces, especially `modules/infotext_utils.py` style extraction in `parse_generation_parameters()`, `modules/styles.py`, prompt/style UI controls, and any duplicated style-name/prompt parsing helpers, while preserving `infotext_styles` behavior and old prompt/style import contracts.

## Pass 66 - Infotext/style/prompt parsing and prompt-style extraction surfaces (2026-06-20)

### Checked scope
- `modules/styles.py`: `PromptStyle`, `merge_prompts()`, module-level `apply_styles_to_prompt()`, `extract_style_text_from_prompt()`, `extract_original_prompts()`, `StyleDatabase.__init__()`, `reload()`, `load_from_csv()`, `get_style_paths()`, style prompt getters, positive/negative style application, `save_styles()`, and `extract_styles_from_prompt()`.
- `modules/infotext_utils.py`: the style extraction branch in `parse_generation_parameters()`, `infotext_styles` mode behavior, hires prompt/negative prompt style comparison, `Styles array` emission, and old `modules.generation_parameters_copypaste` import alias preservation.
- Prompt/style UI and API controls: `modules/ui_prompt_styles.py`, `modules/ui_toprow.py`, `modules/ui.py` style dropdown paste fields/token-counter integration, `/sdapi/v1/prompt-styles` response assembly in `modules/api/api.py`, `modules/api/models.py` `PromptStyleItem`, and `scripts/xyz_grid.py` Styles axis application.
- Focused contract tests and adjacent references: `test/test_styles.py`, `test/test_infotext_api_mappings.py`, and `test/test_infotext_paste_bindings.py`.

### Findings and fixes
- No safe source-code removals or consolidations were made in this slice; this is a ledger-only checkpoint commit.
- Preserved `modules.styles` module-level helpers even where `StyleDatabase` wraps them. `merge_prompts()`, `apply_styles_to_prompt()`, `extract_style_text_from_prompt()`, and `extract_original_prompts()` are small but form a direct public/importable prompt-style helper surface, and focused tests already import `extract_original_prompts()` directly.
- Preserved the split between style merge/extract logic and `infotext_utils.parse_generation_parameters()`. `modules.styles` owns prompt-template matching and style database order; `parse_generation_parameters()` owns old infotext parsing, `infotext_styles` policy, hires prompt comparison, `Styles array` paste emission, and default/backcompat field filling.
- Preserved the duplicate-looking positive/negative prompt application calls in processing/UI/token-counter paths. Positive and negative prompts intentionally use separate style text fields and component identities, while token counters optionally apply the same active styles before prompt parsing.
- Preserved `ui_prompt_styles` save/delete/materialize/refresh helper functions. They are callback entry points bound into Gradio component events and operate on live `shared.prompt_styles` state; consolidating them into methods or anonymous callbacks would not remove real duplication and would risk UI callback/API-name behavior.
- Preserved old style symbols in `modules.ui` despite sparse static use. They are module-level UI symbol exports from the older prompt/style surface, and this slice's compatibility bar explicitly calls for preserving old prompt/style import contracts.
- Preserved `scripts/xyz_grid.py` `apply_styles()` as a script-specific style-name parser. It appends comma-separated XYZ axis style names into `p.styles`; this is separate from prompt-text style extraction and from the multiselect style UI value path.
- Preserved `/sdapi/v1/prompt-styles` list assembly and `PromptStyleItem` shape. The route exposes the style CSV/database contract to API clients, so replacing it with internal namedtuple serialization or removing field-specific response logic would be externally visible.

### Static/dynamic audit map notes
- Runtime generation style chain remains: UI/API/script request stores selected style names in `p.styles` -> `StableDiffusionProcessing.setup_prompts()` and hires setup apply positive/negative style prompts through `shared.prompt_styles` -> generated infotext emits already-styled prompt text plus optional style inference on paste.
- Infotext paste style chain remains: raw infotext prompt/negative prompt -> `parse_generation_parameters()` -> `shared.prompt_styles.extract_styles_from_prompt()` for base prompt and, when version-compatible, hires prompt -> only if base/hires extracted style names match, prompt text is stripped and `Styles array` may be emitted according to `infotext_styles`.
- Prompt-style UI chain remains: `Toprow.create_styles_ui()` creates `UiPromptStyles` -> edit dialog callbacks select/save/delete/refresh database entries -> materialize buttons write styled prompt text into prompt boxes and clear the style dropdown -> style dropdown changes can recalculate token counters.
- Compatibility surfaces to keep conservative: style CSV column names and old `name,text` import support, `PromptStyle` tuple field order, `None` style sentinel, style list divider entries/path marker `do_not_save`, `{prompt}` replacement semantics, style extraction order/reversal, `infotext_styles` modes, `Styles array` paste key/API name `styles`, old `modules.generation_parameters_copypaste` alias, and old module-level style/UI helper imports.

### Validation log
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pycompile-pass66.XXXXXX) python3 -m py_compile modules/styles.py modules/ui_prompt_styles.py modules/infotext_utils.py modules/ui.py modules/ui_toprow.py modules/api/api.py modules/api/models.py scripts/xyz_grid.py test/test_styles.py` - passed.
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pytest-pass66.XXXXXX) python3 -m pytest -q test/test_styles.py test/test_infotext_api_mappings.py test/test_infotext_paste_bindings.py` - passed: 11 passed, with the existing pytest config warning `Unknown config option: base_url`.
- Exact AST duplicate-body scan across `modules/styles.py`, `modules/ui_prompt_styles.py`, `modules/infotext_utils.py`, `modules/ui.py`, `modules/ui_toprow.py`, `modules/api/api.py`, `modules/api/models.py`, `scripts/xyz_grid.py`, and `test/test_styles.py` reported no duplicate nontrivial function body groups.
- Targeted reference scans confirmed style extraction/application is concentrated in `modules.styles`, `parse_generation_parameters()`, the prompt-style UI, txt2img/img2img paste fields, token counters, the prompt-styles API route, and the XYZ Styles axis.
- `git diff --check` - passed.
- Live WebUI prompt-style dialog/materialize behavior and API `/sdapi/v1/prompt-styles` calls were not exercised because they require a running WebUI/API session and style runtime state.

### Next unchecked scope
- More slices are still needed. Recommended next slice: continue into prompt parsing and conditioning syntax surfaces not covered by style extraction, especially `modules/prompt_parser.py` schedule/AND/attention parsing, `modules/sd_emphasis.py`, CLIP token chunk reconstruction paths adjacent to prompt parsing, and tests around scheduled prompts/emphasis, while preserving prompt syntax compatibility and old emphasis behavior.


## Pass 67 - Prompt parsing and conditioning syntax surfaces (2026-06-20)

### Checked scope
- `modules/prompt_parser.py`: `schedule_parser`, `get_learned_conditioning_prompt_schedules()` nested `collect_steps()`/`at_step()`/`get_schedule()` helpers, `SdConditioning`, `get_learned_conditioning()`, `get_multicond_prompt_list()`, `ComposableScheduledPromptConditioning`, `MulticondLearnedConditioning`, `DictWithShape`, `reconstruct_cond_batch()`, `stack_conds()`, `reconstruct_multicond_batch()`, `re_attention`, `re_break`, and `parse_prompt_attention()`.
- `modules/sd_emphasis.py`: `Emphasis`, `EmphasisNone`, `EmphasisIgnore`, `EmphasisOriginal`, `EmphasisOriginalNoNorm`, `get_current_option()`, `get_options_descriptions()`, and `options`.
- Adjacent CLIP/token chunk reconstruction paths: `modules/sd_hijack_clip.py` `PromptChunk`, `TextConditionalModel.empty_chunk()`, `get_target_prompt_token_count()`, `tokenize_line()`, `process_texts()`, `forward()`, `process_tokens()`, old emphasis dispatch in `FrozenCLIPEmbedderWithCustomWordsBase.forward()`, and tokenizer subclasses; `modules/sd_hijack_clip_old.py` old emphasis/token remaking path; `modules/sd_hijack_open_clip.py`; `modules/sd_hijack_xlmr.py`; and SD3 prompt parsing adjacency in `modules/models/sd3/sd3_cond.py`.
- Runtime call surfaces and tests/fixtures: `modules/sd_samplers_cfg_denoiser.py` conditioning reconstruction calls, `modules/processing.py` schedule cache comparison and conditioning cache keys, `modules/ui.py` token-count schedule path, `modules/infotext_utils.py` emphasis detection, `test/test_txt2img.py`, `test/test_openclaw_cache_invalidation.py`, and `test/test_cfg_denoiser_callbacks.py`.

### Findings and fixes
- No safe source-code removals or consolidations were made in this slice; this is a ledger-only checkpoint commit.
- Preserved the Lark prompt schedule parser and nested schedule helpers. Schedule grammar, fractional/integer step conversion, hires old/new scheduling behavior, alternates, escaped literals, and fallback-on-parse-error semantics are tightly coupled to long-standing prompt editing compatibility.
- Preserved `parse_prompt_attention()` separately from schedule parsing. It handles emphasis weights, escaped brackets, `BREAK` chunk separators, unbalanced bracket behavior, and run merging; it is used by CLIP chunking, SD3 T5 tokenization, and infotext emphasis detection rather than being duplicate schedule syntax parsing.
- Preserved `get_multicond_prompt_list()` and the `AND` regex/weight split. It feeds composable conditioning and prompt token counting, and its word-boundary `AND` behavior plus text deduplication are prompt syntax compatibility surfaces.
- Preserved both `reconstruct_cond_batch()` and `reconstruct_multicond_batch()` despite a shared target-schedule selection loop. The duplication is small, while the surrounding output assembly differs substantially for plain scheduled conditioning versus composable weighted multicond conditioning and dict/tensor stacking; extracting a helper would add indirection without removing risky behavioral duplication.
- Preserved all emphasis option classes. `EmphasisNone`, `EmphasisIgnore`, `EmphasisOriginal`, and `EmphasisOriginalNoNorm` are user-visible settings choices and feed infotext emission; old and no-norm modes are compatibility/reproduction surfaces.
- Preserved `modules/sd_hijack_clip_old.py`. It is selected by `opts.use_old_emphasis_implementation` for old-seed reproduction and intentionally differs from current CLIP chunking by using tokenizer parenthesis token multipliers and truncation comments.
- Preserved CLIP/OpenCLIP/XLMR/SD3 tokenization adjacency. The exact duplicate scan found only two trivial SD3 tokenizer passthrough methods (`Sd3ClipLG.tokenize()` and `Sd3T5.tokenize()`); they sit on different tokenizer classes and are not worth consolidating in this compatibility slice.
- Preserved `DictWithShape`. It supplies a `.shape` compatibility property for dict conditioning batches consumed by sampler code expecting tensor-like shape access.

### Static/dynamic audit map notes
- Schedule chain remains: raw prompts -> `get_learned_conditioning_prompt_schedules()` -> Lark schedule parse/alternate expansion -> model conditioning for each scheduled text -> `ScheduledPromptConditioning` schedule entries -> sampler-step reconstruction.
- Composable conditioning chain remains: raw prompts -> `get_multicond_prompt_list()` `AND` split with optional weights -> flat prompt schedule conditioning -> per-prompt `ComposableScheduledPromptConditioning` entries -> `reconstruct_multicond_batch()` returns composable indexes/weights and stacked tensors or dict conditioning.
- Attention/chunk chain remains: prompt text -> `parse_prompt_attention()` unless emphasis disabled -> tokenizer-specific `tokenize()` -> `PromptChunk` packing with `BREAK`, comma backtracking, textual inversion fixes, and empty chunk padding -> transformer encode -> selected `sd_emphasis` class mutates output.
- Compatibility surfaces to keep conservative: prompt editing brackets and escapes, alternates, fractional and integer scheduling with hires offset behavior, old scheduling comparison/cache behavior, word-boundary `AND`, `BREAK`, unbalanced emphasis brackets, comma backtracking, textual inversion chunk fixes, old emphasis implementation switch, emphasis option names/descriptions, dict conditioning `.shape`, and extension-visible prompt parser/emphasis module imports.

### Validation log
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pycompile-pass67.XXXXXX) python3 -m py_compile modules/prompt_parser.py modules/sd_emphasis.py modules/sd_hijack_clip.py modules/sd_hijack_clip_old.py modules/sd_hijack_open_clip.py modules/sd_hijack_xlmr.py modules/models/sd3/sd3_cond.py modules/sd_samplers_cfg_denoiser.py modules/infotext_utils.py modules/ui.py test/test_txt2img.py test/test_openclaw_cache_invalidation.py test/test_cfg_denoiser_callbacks.py` - passed.
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-doctest-pass67.XXXXXX) python3 modules/prompt_parser.py` - blocked before doctest execution by missing bare-environment dependency: `ModuleNotFoundError: No module named lark`.
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pytest-pass67.XXXXXX) python3 -m pytest -q test/test_openclaw_cache_invalidation.py test/test_cfg_denoiser_callbacks.py test/test_txt2img.py` - blocked during collection by missing bare-environment dependencies: `ModuleNotFoundError: No module named numpy` and `ModuleNotFoundError: No module named torch`; existing warning `Unknown config option: base_url` was also emitted.
- Exact AST duplicate-body scan across `modules/prompt_parser.py`, `modules/sd_emphasis.py`, `modules/sd_hijack_clip.py`, `modules/sd_hijack_clip_old.py`, `modules/sd_hijack_open_clip.py`, `modules/sd_hijack_xlmr.py`, `modules/models/sd3/sd3_cond.py`, `modules/sd_samplers_cfg_denoiser.py`, `modules/infotext_utils.py`, and `modules/ui.py` reported only the two trivial SD3 tokenizer passthrough methods noted above; no actionable duplicate nontrivial function body groups were found.
- Targeted reference scans confirmed prompt scheduling, emphasis parsing, multicond splitting, CLIP chunking, conditioning reconstruction, infotext emphasis detection, and old emphasis dispatch are all still connected to runtime or compatibility surfaces.
- `git diff --check` - passed.
- Live generation, token counter UI behavior, and model-backed conditioning tensor comparisons were not exercised because they require a fully provisioned WebUI/model runtime environment.

### Next unchecked scope
- More slices are still needed. Recommended next slice: continue into prompt/token counting and textual inversion chunk integration surfaces, especially `modules/sd_hijack.py` token counting helpers, textual inversion embedding DB/chunk fix paths, `modules/textual_inversion/*`, and prompt token counter UI/API paths, looking for stale tokenizer/embedding compatibility branches while preserving extension-visible embedding behavior and old token count contracts.

## Pass 68 - Prompt/token counting and textual inversion chunk integration surfaces (2026-06-20)

### Checked scope
- `modules/sd_hijack.py`: `StableDiffusionModelHijack.get_prompt_lengths()`, textual-inversion hijack setup/undo for SD/SDXL/OpenCLIP/XLMR, `EmbeddingsWithFixes`, `TextualInversionEmbeddings`, and model-level `fixes` handoff behavior.
- `modules/textual_inversion/textual_inversion.py`: template listing, `Embedding`, `DirWithTextualInversionEmbeddings`, `EmbeddingDatabase` directory/load/register/unregister/lookup/cache behavior, embedding image reads, data-format compatibility in `create_embedding_from_data()`, training image chunk save path, and `save_embedding()` state restoration.
- Prompt token counter UI/API paths: `modules/ui.py` `update_token_counter()` and `update_negative_prompt_token_counter()`, txt2img/img2img token button wiring, `modules/ui_toprow.py` hidden token counter components, `javascript/token-counters.js`, `modules/script_callbacks.py` before-token-counter callback API, and `extensions/openclaw-clear-cond-cache/scripts/openclaw_clear_cond_cache.py` `estimate_token_count()`.
- Adjacent embedding/token chunk contracts: `modules/sd_hijack_clip.py` `PromptChunkFix` handoff and `process_texts()`, `modules/sd_models_xl.py` GeneralConditioner token-count passthroughs, `modules/models/sd3/sd3_cond.py` SD3 token count wrappers, `modules/models/sd3/other_impls.py` SD3 `TextualInversionEmbeddings` use, `modules/ui_extra_networks_textual_inversion.py`, `extensions-builtin/Lora/networks.py` bundled-embedding registration/unregistration, and focused textual-inversion/cache tests.

### Findings and fixes
- No safe source-code removals or consolidations were made in this slice; this is a ledger-only checkpoint commit.
- Preserved `StableDiffusionModelHijack.get_prompt_lengths()` and its `hasattr(self.clip, 'get_token_count')` fallback split. SD3 exposes `get_token_count()`, while SD/SDXL-era CLIP wrappers and SDXL `GeneralConditioner` expose `process_texts()`/`get_target_prompt_token_count()`; the wrapper is also consumed directly by UI token counters and extension token-estimation code.
- Preserved the apparent duplicate token-count passthroughs in `modules/sd_models_xl.py` and `modules/models/sd3/sd3_cond.py`. They adapt different conditioner/model-family shapes while maintaining the same UI/API token-count contract.
- Preserved `EmbeddingsWithFixes` and `TextualInversionEmbeddings` sharing a forward implementation. `EmbeddingsWithFixes` wraps existing token embedding modules during SD/SDXL/OpenCLIP/XLMR hijack, while `TextualInversionEmbeddings` is constructed as the token embedding for SD3 CLIP models; the shared call path is intentional and not safely collapsible without changing owner/module semantics.
- Preserved `EmbeddingDatabase.register_embedding_by_name(None, ...)` and `find_embedding_at_position()`. Lora bundled embeddings unregister through the `None` path, while CLIP chunk reconstruction resolves multi-token textual-inversion matches through the sorted `ids_lookup` table at runtime.
- Preserved textual-inversion embedding format branches for `.pt`/`.bin`, `.safetensors`, image-embedded PNG/WebP/JXL/AVIF, SDXL `clip_g`/`clip_l`, diffuser concepts, and torch 1.11/1.12 parameter compatibility. These are import/runtime compatibility surfaces, not stale tokenizer branches.
- Preserved prompt token counter plumbing across Python callbacks and JavaScript hidden buttons. The positive/negative and txt2img/img2img event wiring is duplicated by live component identity; consolidating it would not remove dead behavior and could break Gradio callback names, JS debounce hooks, or extension `before_token_counter` callbacks.
- Preserved extension-side old/current token count compatibility in `openclaw_clear_cond_cache.estimate_token_count()`. Its Forge branch still calls `model_hijack.get_prompt_lengths(prompt, cond_stage_model)`, while the non-Forge branch uses the current one-argument A1111 contract; this remains an extension-visible compatibility hedge.

### Static/dynamic audit map notes
- Token counter chain remains: prompt textbox/style state -> optional `before_token_counter` callback mutation -> optional style application -> extra-network stripping and prompt schedule expansion -> maximum scheduled prompt token count through `model_hijack.get_prompt_lengths()` -> hidden Gradio token counter HTML, triggered by JS debounce/buttons.
- Textual inversion chunk chain remains: embedding DB load/register builds `ids_lookup` by tokenizer ids -> CLIP/OpenCLIP/XLMR token chunking calls `find_embedding_at_position()` -> `PromptChunkFix` markers are stored in `model_hijack.fixes` per chunk -> wrapped token embedding modules splice embedding vectors in `EmbeddingsWithFixes.forward()`.
- Textual inversion load/cache chain remains: model load or forced refresh calls `load_textual_inversion_embeddings()` -> directory mtime gate avoids unnecessary reloads -> file/image readers support legacy and current embedding containers -> shape mismatch goes to `skipped_embeddings` for API/UI visibility -> optional image-embedding cache avoids repeated non-embedding image reads.
- Compatibility surfaces to keep conservative: `get_prompt_lengths()` return shape and old extension probes, `BeforeTokenCounterParams`, token counter HTML shape, hidden token button ids, `process_texts()` token-count return convention, `get_target_prompt_token_count()`, `EmbeddingDatabase` dictionaries and mutable registration methods, textual-inversion image metadata keys/chunks, SDXL dual-vector keys, Lora bundled embedding registration, old torch `_parameters` compatibility, and SD3 textual inversion embedding construction.

### Validation log
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pycompile-pass68.XXXXXX) python3 -m py_compile modules/sd_hijack.py modules/textual_inversion/textual_inversion.py modules/textual_inversion/image_embedding.py modules/textual_inversion/ui.py modules/textual_inversion/dataset.py modules/textual_inversion/learn_schedule.py modules/textual_inversion/saving_settings.py modules/ui.py modules/ui_toprow.py modules/script_callbacks.py modules/sd_hijack_clip.py modules/sd_models_xl.py modules/models/sd3/sd3_cond.py modules/models/sd3/other_impls.py modules/ui_extra_networks_textual_inversion.py modules/api/api.py extensions/openclaw-clear-cond-cache/scripts/openclaw_clear_cond_cache.py tests/test_textual_inversion_preview_save_contract.py tests/test_textual_inversion_autocrop_contract.py test/test_openclaw_cache_invalidation.py` - passed.
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pytest-pass68.XXXXXX) python3 -m pytest -q tests/test_textual_inversion_preview_save_contract.py tests/test_textual_inversion_autocrop_contract.py test/test_openclaw_cache_invalidation.py` - blocked during collection by missing bare-environment dependencies: `ModuleNotFoundError: No module named numpy`; existing warning `Unknown config option: base_url` was also emitted.
- Exact AST duplicate-body scan across `modules/sd_hijack.py`, `modules/textual_inversion/textual_inversion.py`, `modules/textual_inversion/image_embedding.py`, `modules/textual_inversion/ui.py`, `modules/ui.py`, `modules/ui_toprow.py`, `modules/script_callbacks.py`, `modules/sd_hijack_clip.py`, `modules/sd_models_xl.py`, `modules/models/sd3/sd3_cond.py`, `modules/models/sd3/other_impls.py`, `modules/ui_extra_networks_textual_inversion.py`, `modules/api/api.py`, and `extensions/openclaw-clear-cond-cache/scripts/openclaw_clear_cond_cache.py` reported only the two trivial SD3 tokenizer passthrough methods preserved in pass 67; no actionable duplicate nontrivial function body groups were found.
- Targeted reference scans confirmed `sd_hijack_clip_old.py` remains selected by `opts.use_old_emphasis_implementation`, `TextualInversionEmbeddings` remains live through SD3 CLIP embeddings, `register_embedding_by_name(None, ...)` remains live through bundled Lora embedding unload, and `get_prompt_lengths()` remains consumed by UI and extension token-count paths.
- `git diff --check` - passed.
- Live WebUI token-counter clicks, API embedding endpoints, and model-backed textual-inversion loading/training were not exercised because they require a running WebUI/model runtime and provisioned ML dependencies.

### Next unchecked scope
- More slices are still needed. Recommended next slice: continue into textual-inversion training helpers and hypernetwork training reuse surfaces, especially shared training validation/logging/tensorboard helpers in `modules/textual_inversion/textual_inversion.py` versus `modules/hypernetworks/hypernetwork.py`, embedding save/preview image contracts, and adjacent training UI/API task wrappers, while preserving training output formats and old embedding/hypernetwork compatibility.

## Pass 69 - Textual-inversion training helpers and hypernetwork training reuse surfaces (2026-06-20)

### Checked scope
- `modules/textual_inversion/textual_inversion.py`: shared training helpers `write_loss()`, `tensorboard_setup()`, `tensorboard_add()`, `tensorboard_add_scaler()`, `tensorboard_add_image()`, `validate_train_inputs()`, textual-inversion `train_embedding()` save/log/preview/image-embedding path, and `save_embedding()` compatibility state restoration.
- `modules/hypernetworks/hypernetwork.py`: `train_hypernetwork()` validation/logging/tensorboard reuse, hypernetwork checkpoint/optimizer save path, preview generation path, RNG/device restoration, and `save_hypernetwork()` compatibility state restoration.
- Training UI/API wrappers: `modules/textual_inversion/ui.py`, `modules/hypernetworks/ui.py`, `modules/api/api.py` create/train task wrappers, hypernetwork pre/post training hooks, lowvram checks, optimization toggles, and response contracts.
- Focused contracts/tests: `tests/test_textual_inversion_preview_save_contract.py`, `tests/test_api_training_contract.py`, and `tests/test_hypernetwork_creation_contract.py`.

### Findings and fixes
- Consolidated the duplicate txt2img-preview parameter assignment block used by both textual-inversion and hypernetwork training into `textual_inversion.apply_txt2img_preview_params()`.
- Updated `train_embedding()` and `train_hypernetwork()` to call the shared helper while preserving their separate preview-generation contracts: embedding training still uses `do_not_reload_embeddings=True`, optional embedded-image PNG chunks, and tensorboard image writes; hypernetwork training still disables extra networks, preserves/restores RNG state, switches hypernetwork train/eval state, and manages cond/first-stage devices.
- Added a focused AST contract to keep both training loops on the shared preview helper and prevent direct duplicate `p.sampler_name` assignment from reappearing in those loops.
- Preserved `write_loss()` and tensorboard helpers in `textual_inversion.py` as the existing shared training helper surface used by hypernetwork training; no separate duplicate logging/tensorboard helpers remained to remove.
- Preserved `validate_train_inputs()` despite generic wording quirks such as the literal `Save {name}` assertion text. The helper is shared by embedding and hypernetwork training and its exact assertion strings can surface through UI/API error paths.
- Preserved `save_embedding()` and `save_hypernetwork()` as separate compatibility save wrappers. They restore different model metadata and optimizer/cache fields, and textual-inversion save additionally controls cached checksum invalidation and optimizer state through the embedding object.
- Preserved training UI/API wrappers. Their duplicate-looking shape is callback/task-contract plumbing with distinct lowvram text, optimization/device cleanup, API route names, progress task names, and train response messages.

### Static/dynamic audit map notes
- Shared training validation/logging chain remains: UI/API wrapper -> `train_embedding()` or `train_hypernetwork()` -> `validate_train_inputs()` -> dataset construction -> `write_loss()` CSV output and optional tensorboard scalar/image helpers.
- Preview chain now shares only txt2img parameter transfer: preview UI/API values -> `apply_txt2img_preview_params()` -> model-specific preview processing in each training loop -> image save/tensorboard/update-current-image behavior remains local to each model type.
- Embedding save chain remains: periodic/final `save_embedding()` sets checkpoint metadata, name, optimizer state, and cached-checksum behavior -> `.pt` and optional `.optim` output -> optional PNG chunk embedding uses the last periodic embedding file.
- Hypernetwork save chain remains: periodic/final training writes optimizer metadata only when enabled, dereferences optimizer state after save, and preserves old hypernetwork metadata branches.
- Compatibility surfaces to keep conservative: `textual_inversion_loss.csv` and `hypernetwork_loss.csv` field order, tensorboard tag names, preview image filenames/prompts, embedded PNG chunk keys, old embedding/hypernetwork checkpoint metadata fields, optimizer `.optim` sidecars, UI wrapper return strings, API task names/messages, and progress `shared.state` updates.

### Validation log
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pycompile-pass69.XXXXXX) python3 -m py_compile modules/textual_inversion/textual_inversion.py modules/hypernetworks/hypernetwork.py modules/textual_inversion/ui.py modules/hypernetworks/ui.py modules/api/api.py modules/textual_inversion/dataset.py modules/textual_inversion/learn_schedule.py modules/textual_inversion/saving_settings.py tests/test_textual_inversion_preview_save_contract.py tests/test_api_training_contract.py tests/test_hypernetwork_creation_contract.py` - passed.
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pytest-pass69.XXXXXX) python3 -m pytest -q tests/test_textual_inversion_preview_save_contract.py tests/test_api_training_contract.py tests/test_hypernetwork_creation_contract.py` - passed: 6 passed, with the existing pytest config warning `Unknown config option: base_url`.
- Exact AST duplicate-body scan across `modules/textual_inversion/textual_inversion.py`, `modules/hypernetworks/hypernetwork.py`, `modules/textual_inversion/ui.py`, `modules/hypernetworks/ui.py`, `modules/api/api.py`, and the focused training contract tests reported no duplicate nontrivial function body groups.
- `git diff --check` - passed.
- Live WebUI/API training and model-backed preview generation were not exercised because they require a running WebUI/model training runtime.

### Next unchecked scope
- More slices are still needed. Recommended next slice: continue into textual-inversion dataset/preprocessing/autocrop training-data helpers, especially `modules/textual_inversion/dataset.py`, `modules/textual_inversion/preprocess.py`, `modules/textual_inversion/autocrop.py`, preprocessing UI/API wrappers, caption/tag handling, and focused autocrop/preprocess contracts, while preserving generated dataset formats and old preprocessing behavior.

## Pass 70 - Textual-inversion dataset/preprocessing/autocrop training-data helpers (2026-06-20)

### Checked scope
- `modules/textual_inversion/dataset.py`: `DatasetEntry`, `PersonalizedBase.__init__()`, `create_text()`, dataset length/item access, latent sampling mode behavior, alpha-channel loss weighting, caption `.txt` fallback/name regex handling, tag shuffle/dropout handling, `GroupedBatchSampler`, `PersonalizedDataLoader`, `BatchLoader`, `BatchLoaderRandom`, and collate wrapper selection.
- `modules/textual_inversion/autocrop.py`: crop/focal-point selection, face/corner/entropy point collection, image entropy, point centroid/weighted average helpers, orientation helpers, OpenCV face-model cache helper, `PointOfInterest`, and `Settings`.
- Preprocessing/training-data wrappers and contracts: `extensions-builtin/postprocessing-for-training/scripts/postprocessing_focal_crop.py` as the active focal-crop preprocessing UI script, `modules/ui.py` training tab dataset/template/tag controls, `modules/textual_inversion/ui.py` create/train wrappers, `modules/api/api.py` embedding/hypernetwork training task wrappers, `tests/test_textual_inversion_autocrop_contract.py`, and `tests/test_textual_inversion_preview_save_contract.py`.

### Findings and fixes
- Removed the redundant `BatchLoaderRandom.__init__()` override from `modules/textual_inversion/dataset.py`. It only called `super().__init__(data)` and added no state or behavior, so normal inheritance preserves construction while leaving the random-latent-specific `pin_memory()` no-op override intact.
- Confirmed there is no `modules/textual_inversion/preprocess.py` in this branch. The active preprocessing/focal-crop path for training data is the built-in `postprocessing-for-training` script calling `autocrop.Settings()`, `download_and_cache_models()`, and `crop_image()`.
- Preserved dataset caption/tag handling. `.txt` sidecar captions, filename fallback stripping of leading numeric prefixes, optional filename regex/join settings, comma-based tag shuffling, and tag dropout directly affect generated training prompts and old dataset semantics.
- Preserved dataset latent/weight branches. `once`, `deterministic`, and `random` latent sampling change cache/re-sampling behavior, while alpha-channel loss weighting depends on latent sample dimensions even for random sampling.
- Preserved autocrop helper structure despite small orientation wrappers. The focal-crop extension imports the module-level `Settings`, model downloader, and `crop_image()` contract; internal point-selection helpers are coupled to annotated debug output and OpenCV fallback behavior.
- Preserved UI/API training wrappers. Their duplicate-looking create/train task shapes are public route/callback contracts with distinct task names, post-create refresh hooks, and hypernetwork device cleanup.

### Static/dynamic audit map notes
- Dataset prompt chain remains: image path -> optional sidecar `.txt` caption or filename-derived caption -> optional regex/join filename word extraction -> template `[filewords]`/`[name]` substitution -> optional per-access tag dropout/shuffle -> embedding/hypernetwork conditioning.
- Dataset batch chain remains: `PersonalizedBase` groups images by size -> `GroupedBatchSampler` emits same-size batches plus weighted remainder batches -> `PersonalizedDataLoader` chooses fixed or random collate wrapper -> random mode inherits `BatchLoader` construction and skips pinning mutated per-item latent samples.
- Focal preprocessing chain remains: postprocessing script UI options -> optional YuNet model download -> `autocrop.Settings` -> `crop_image()` chooses scale/crop around weighted corner, entropy, and face focal points -> optional annotated extra image output.
- Compatibility surfaces to keep conservative: training dataset sidecar `.txt` format, filename regex/join options, comma tag semantics, `training_image_repeats_per_epoch`, latent sampling names, alpha-channel weight normalization, `BatchLoader` attribute names, autocrop annotation/debug image ordering, OpenCV model paths/URLs, Gradio training component IDs, API task route names, and extension-visible postprocessing script behavior.

### Validation log
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pycompile-pass70.XXXXXX) python3 -m py_compile modules/textual_inversion/dataset.py modules/textual_inversion/autocrop.py modules/textual_inversion/ui.py modules/ui.py modules/api/api.py extensions-builtin/postprocessing-for-training/scripts/postprocessing_focal_crop.py tests/test_textual_inversion_autocrop_contract.py tests/test_textual_inversion_preview_save_contract.py` - passed.
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pytest-pass70.XXXXXX) python3 -m pytest -q tests/test_textual_inversion_autocrop_contract.py tests/test_textual_inversion_preview_save_contract.py` - passed: 3 passed, with the existing pytest config warning `Unknown config option: base_url`.
- Exact AST duplicate-body scan across `modules/textual_inversion/dataset.py`, `modules/textual_inversion/autocrop.py`, `modules/textual_inversion/ui.py`, `modules/ui.py`, `modules/api/api.py`, `extensions-builtin/postprocessing-for-training/scripts/postprocessing_focal_crop.py`, and the focused textual-inversion contract tests reported no duplicate nontrivial function body groups after the `BatchLoaderRandom.__init__()` removal.
- Targeted reference scans confirmed autocrop is active through `extensions-builtin/postprocessing-for-training/scripts/postprocessing_focal_crop.py`, dataset loading remains shared by embedding and hypernetwork training, and training dataset tag/caption controls remain wired through the Train tab and API training paths.
- `git diff --check` - passed.
- Live WebUI postprocessing/preprocess UI, API training routes, and model-backed dataset construction were not exercised because they require a running WebUI/model training runtime and real image datasets.

### Next unchecked scope
- More slices are still needed. Recommended next slice: continue into postprocessing and extras training-data preparation surfaces, especially `modules/postprocessing.py`, `modules/ui_postprocessing.py`, `extensions-builtin/postprocessing-for-training/scripts/postprocessing_focal_crop.py`, split/resize/save postprocessing scripts, and API extras wrappers, looking for duplicate image-save/resize/split helper logic while preserving generated training-data output filenames and postprocessing script contracts.

## Pass 71 - Postprocessing and extras training-data preparation surfaces (2026-06-20)

### Checked scope
- `modules/postprocessing.py`: `combine_caption()`, `save_caption_sidecar()`, `run_postprocessing()` including batch/single/directory image enumeration, existing PNG info propagation, postprocessing-script execution, extra-image suffix handling, forced output filename handling, caption sidecar save path, WebUI wrapper `run_postprocessing_webui()`, and legacy API wrapper `run_extras()`.
- `modules/ui_postprocessing.py`: extras tab mode selection, batch-directory controls, `show_extras_results` wiring, postprocessing script input wiring, paste-field registration, and image-change callback.
- `modules/scripts_postprocessing.py`: `PostprocessedImageSharedInfo`, `PostprocessedImage.get_suffix()`, `PostprocessedImage.create_copy()`, runner UI/control bounds, preferred-order filtering, postprocessing first-pass/process execution over extra images, and API argument creation.
- `extensions-builtin/postprocessing-for-training/scripts/postprocessing_autosized_crop.py`: `center_crop()`, `multicrop_pic()`, and auto-sized crop script UI/process contract.
- `extensions-builtin/postprocessing-for-training/scripts/postprocessing_split_oversized.py`: `split_pic()` and oversized split script UI/process contract.
- `extensions-builtin/postprocessing-for-training/scripts/postprocessing_focal_crop.py`: focal-crop UI/process contract, autocrop model fallback, shared target-size use, debug extra-image output, and `disable_processing` handling.
- Adjacent training-data scripts and wrappers: caption generation, flipped-copy creation, `scripts/postprocessing_upscale.py` target-size producers, API extras request normalization/decoding/response wrappers in `modules/api/api.py`, API request/response models in `modules/api/models.py`, and focused caption/postprocessing contracts in `tests/test_postprocessing_caption_contract.py`.

### Findings and fixes
- No safe source-code remediation was made in this slice.
- Preserved `PostprocessedImage.get_suffix()` and forced filename handling. Extra images from split, flip, and focal debug paths converge through this one suffix allocator, and `opts.use_original_name_batch` plus `forced_filename=basename + suffix` preserve generated training-data filenames for batch outputs.
- Preserved `split_pic()`, `center_crop()`, and `multicrop_pic()` as separate helpers. They are not duplicate implementations: split resize/crop creates sequential overlapping tiles, auto-sized crop chooses a centered target dimension under area/error constraints, and focal crop delegates weighted point selection to textual-inversion autocrop.
- Preserved focal-crop debug copies with `disable_processing=True`; this prevents later postprocessing scripts from treating debug annotation images as normal training outputs while still allowing the runner to save them with the `focal-crop-debug` suffix.
- Preserved API extras wrappers and `run_extras()` despite legacy shape. External API fields still use `upscaler_1`/`upscaler_2`, are normalized to postprocessing script argument names, always force `show_extras_results=True`, and route through the queue lock with `save_output=False`.
- Preserved `run_postprocessing_webui(id_task, ...)` as UI queue compatibility glue; the task id is consumed by the call-queue wrapper contract even though the local function forwards only postprocessing arguments.
- Preserved caption and sidecar behavior. Caption append/prepend/keep semantics and sidecar output paths are covered by focused tests and affect generated training-data captions.

### Static/dynamic audit map notes
- Extras UI chain remains: tab selection -> numeric `extras_mode` -> `run_postprocessing_webui()` -> `run_postprocessing()` -> `scripts.scripts_postproc.run()` -> `images.save_image()`/caption sidecar output.
- API extras chain remains: API request model -> `setUpscalers()` legacy field normalization -> image decode or batch decode -> `_run_extras()` queue lock -> `run_extras()` old handler -> postprocessing runner -> base64 response without disk save.
- Training-data postprocessing chain remains: Upscale first-pass sets `pp.shared.target_width/height` -> split/focal scripts consume those target dimensions -> extra images are converted to `PostprocessedImage` copies by the runner -> suffix allocation keeps output filenames unique and stable.
- Compatibility surfaces to keep conservative: extras tab component IDs, API route names and request field names, `show_extras_results` API override, `PostprocessedImage` attributes/extensions contract, postprocessing operation order/disable options, original-name batch output behavior, caption sidecar merge action strings, split tile ordering, focal-crop debug suffix, and saved PNG `postprocessing` metadata.

### Validation log
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pycompile-pass71.XXXXXX) python3 -m py_compile modules/postprocessing.py modules/ui_postprocessing.py modules/scripts_postprocessing.py modules/api/api.py modules/api/models.py scripts/postprocessing_upscale.py scripts/postprocessing_gfpgan.py scripts/postprocessing_codeformer.py extensions-builtin/postprocessing-for-training/scripts/postprocessing_autosized_crop.py extensions-builtin/postprocessing-for-training/scripts/postprocessing_caption.py extensions-builtin/postprocessing-for-training/scripts/postprocessing_create_flipped_copies.py extensions-builtin/postprocessing-for-training/scripts/postprocessing_focal_crop.py extensions-builtin/postprocessing-for-training/scripts/postprocessing_split_oversized.py tests/test_postprocessing_caption_contract.py` - passed.
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pytest-pass71.XXXXXX) python3 -m pytest -q tests/test_postprocessing_caption_contract.py` - passed: 4 passed, with the existing pytest config warning `Unknown config option: base_url`.
- Exact AST duplicate-body scan across `modules/postprocessing.py`, `modules/ui_postprocessing.py`, `modules/scripts_postprocessing.py`, `modules/api/api.py`, `modules/api/models.py`, `scripts/postprocessing_*.py`, and `extensions-builtin/postprocessing-for-training/scripts/*.py` reported no actionable duplicate function body groups in the checked postprocessing scope. Broader `modules/*.py` hits were unrelated pre-existing pairs outside this slice (`ui_extra_networks.path_is_parent`/`sd_models.path_is_parent`, two `scripts.py` callback wrappers, and XLMR module pairs).
- Targeted reference scans confirmed split/autosized/focal helpers are active through postprocessing script classes, target dimensions are produced by `scripts/postprocessing_upscale.py`, API extras calls still route through `postprocessing.run_extras()`, and caption sidecar helpers remain covered by focused tests.
- `git diff --check` - passed.
- Live WebUI extras tab clicks, API extras requests, model-backed upscaling/restoration, BLIP/Deepbooru captioning, and focal-crop OpenCV model paths were not exercised because they require a running WebUI/model runtime and provisioned ML assets.

### Next unchecked scope
- More slices are still needed. Recommended next slice: continue into postprocessing script framework and main-UI postprocessing bridges, especially `modules/scripts_postprocessing.py`, `modules/scripts_auto_postprocessing.py`, `scripts/postprocessing_upscale.py`, `scripts/postprocessing_gfpgan.py`, `scripts/postprocessing_codeformer.py`, and main UI always-on postprocessing integration, looking for stale compatibility wrappers or duplicate upscaler/face-restoration argument mapping while preserving extension-visible postprocessing contracts.

## Pass 72 - Postprocessing script framework and main-UI postprocessing bridges (2026-06-20)

### Checked scope
- `modules/scripts_postprocessing.py`: framework classes, extension-visible element-id helpers, runner initialization/UI/control bounds, preferred-order filtering, first-pass/process execution, API argument default synthesis, image-change fanout, and late import cycle guard.
- `modules/scripts_auto_postprocessing.py`: `ScriptPostprocessingForMainUI` always-on script adapter, main UI tab-name suffixing, postprocess image argument mapping, generation-params bridge, and `create_auto_preprocessing_script_data()` option/name filtering.
- Bundled postprocessing scripts: `scripts/postprocessing_upscale.py`, `scripts/postprocessing_gfpgan.py`, and `scripts/postprocessing_codeformer.py`, including upscaler argument normalization, first-pass target-size production, cache invalidation, face-restoration visibility blending, and info metadata.
- Main extras/UI bridges and contracts: `modules/ui_postprocessing.py`, relevant `modules/ui.py` extras/main UI integration points, `modules/shared_items.py` postprocessing option list filters, `modules/shared_options.py` postprocessing settings, `modules/postprocessing.py` legacy API extras mapping, `test/test_postprocessing_script_args.py`, and `test/test_postprocessing_api_defaults.py`.

### Findings and fixes
- Consolidated the duplicate restored-image blend/resize/mode-conversion block from `scripts/postprocessing_gfpgan.py` and `scripts/postprocessing_codeformer.py` into `scripts_postprocessing.blend_with_original()`.
- Updated both bundled face-restoration scripts to call the shared helper while preserving output image semantics and metadata keys: `GFPGAN visibility`, `CodeFormer visibility`, and `CodeFormer weight`.
- Added `test_blend_with_original_resizes_and_converts_processed_image()` to pin the helper's resize, mode conversion, alpha preservation, and blend behavior.
- Removed two inert self-assignments in `ScriptPostprocessingUpscale.process()` (`upscaler_1_name = upscaler_1_name` and `upscaler_2_name = upscaler_2_name`). The existing `"None"` normalization, upscaler lookup/assertion behavior, two-upscaler blend, and info metadata remain unchanged.
- Preserved `ScriptPostprocessingForMainUI` despite its thin adapter shape. It is the bridge that gives postprocessing scripts unique txt2img/img2img element IDs, maps always-on script values by script UI control order, copies updated images back into the processing script object, and exposes postprocessing info through `p.extra_generation_params`.
- Preserved `create_auto_preprocessing_script_data()` and postprocessing option filters. They are option-driven extension-visible wiring, not dead wrappers; `extra_only` and `main_ui_only` must keep their current filtering semantics for extras/main UI lists.
- Preserved `run_postprocessing_webui(id_task, ...)` and `run_extras()` legacy mapping as UI/API compatibility bridges already audited in pass 71.

### Static/dynamic audit map notes
- Main-UI bridge chain remains: options `postprocessing_enable_in_main_ui` -> `create_auto_preprocessing_script_data()` -> always-on `ScriptPostprocessingForMainUI` -> script `ui()` control order -> `postprocess_image()` -> `PostprocessedImage` wrapper -> image and generation-param transfer back into txt2img/img2img processing.
- Extras/API bridge chain remains: extras UI or legacy API -> `run_postprocessing()`/`run_extras()` -> `scripts.scripts_postproc.create_args_for_run()` -> runner control slices -> script first-pass/process -> postprocessing metadata and saved/output images.
- Bundled script contracts to keep conservative: script class names and `name` strings, `ui()` return keys, process signatures/defaults, `elem_id`/`elem_id_suffix` behavior, upscaler `"None"` normalization, `limit_size_by_one_dimention()` spelling, cache clearing on image change, and face-restoration metadata keys.
- Exact duplicate scan after remediation found no remaining duplicated GFPGAN/CodeFormer visibility blend block and no remaining `upscaler_[12]_name = upscaler_[12]_name` no-op assignments in the checked files.

### Validation log
- `PYTHONPYCACHEPREFIX=$(mktemp -d) python3 -m py_compile modules/scripts_postprocessing.py modules/scripts_auto_postprocessing.py scripts/postprocessing_upscale.py scripts/postprocessing_gfpgan.py scripts/postprocessing_codeformer.py modules/ui_postprocessing.py` - passed.
- `python3 -m pytest test/test_postprocessing_script_args.py -q` - passed: 5 passed, with the existing pytest config warning `Unknown config option: base_url`.
- `python3 -m pytest test/test_postprocessing_script_args.py test/test_postprocessing_api_defaults.py -q` - partially blocked by an unrelated pre-existing focused-test harness failure in `test_extras_single_response_allows_no_output_from_skipped_or_interrupted_run`: the AST-loaded fake `Api` class lacks `_call_with_queue_lock`; 12 tests passed before the failure.
- `rg -n "if .*visibility < 1\\.0|Image\\.blend\\(pp\\.image, res|upscaler_[12]_name = upscaler_[12]_name|blend_with_original" modules/scripts_postprocessing.py scripts/postprocessing_gfpgan.py scripts/postprocessing_codeformer.py scripts/postprocessing_upscale.py` - confirmed only the shared helper and its two call sites remain.
- `git diff --check` - passed.
- Live WebUI txt2img/img2img always-on postprocessing, extras tab clicks, API extras requests, and model-backed upscaling/restoration were not exercised because they require a running WebUI/model runtime and provisioned ML assets.

### Next unchecked scope
- More slices are still needed. Recommended next slice: continue into API extras/postprocessing request and response bridges around `modules/api/api.py`, `modules/api/models.py`, `modules/postprocessing.py`, and focused API tests, especially the `_call_with_queue_lock` AST-test harness gap surfaced during this pass, while preserving public API request fields, queue-lock behavior, and legacy extras response semantics.

## Pass 73 - API extras/postprocessing request and response bridges (2026-06-20)

### Checked scope
- `modules/api/api.py`: `setUpscalers()`, `decode_extras_batch_images()`, `Api._call_with_queue_lock()`, `Api._run_extras()`, `Api.extras_single_image_api()`, `Api.extras_batch_images_api()`, adjacent route registration, and queue-lock helper call sites.
- `modules/api/models.py`: extras base request fields, legacy single/batch request models, shared `ExtraBaseResponse`, single-image null response field, and batch image response schema.
- `modules/postprocessing.py`: `run_postprocessing()`, `run_postprocessing_webui()`, and `run_extras()` as the API-facing legacy adapter into the postprocessing runner.
- Focused contracts: `test/test_postprocessing_api_defaults.py`, `test/test_postprocessing_script_args.py`, and `tests/test_api_progress_contract.py` around extras request normalization, batch decode tolerance, no-output extras response semantics, postprocessing argument defaults, and queue-lock-related API helpers.

### Findings and fixes
- Fixed the focused AST-loaded extras endpoint harness in `test/test_postprocessing_api_defaults.py` by including `_call_with_queue_lock` when constructing the fake `Api` class. This removes the stale test-only gap surfaced in pass 72 and keeps the test exercising the current production `_run_extras()` queue-lock bridge.
- No production source-code remediation was made. The inspected API extras bridge is intentionally thin and public-contract-sensitive: request models expose legacy `upscaler_1`/`upscaler_2` fields, `setUpscalers()` maps them to postprocessing script names and forces API image return semantics, `_run_extras()` preserves queue-lock behavior and `save_output=False`, and endpoint response wrappers preserve `html_info`, batch image lists, and single-image `None` output for skipped/interrupted runs.
- Preserved `ExtraBaseResponse` and separate single/batch response classes. They share `html_info`, but their image payload shapes differ and are route schema contracts.
- Preserved `decode_extras_batch_images()` tolerance of corrupt batch items. It intentionally skips per-image decode errors while keeping valid items, matching the focused contract.
- Preserved `run_extras()` despite its legacy signature. It remains the compatibility adapter from API extras request fields into `scripts.scripts_postproc.create_args_for_run()` and `run_postprocessing()` order semantics.

### Static/dynamic audit map notes
- API extras request chain remains: FastAPI route -> request model -> `setUpscalers()` legacy field normalization -> single image decode or tolerant batch decode -> `_run_extras()` queue lock -> `postprocessing.run_extras(..., save_output=False)` -> base64 response images plus `html_info`.
- Single-image response semantics remain: if postprocessing produces no output because the run is skipped or interrupted, the API returns `image=None` rather than attempting to encode a missing image.
- Batch response semantics remain: successful decoded inputs are postprocessed and encoded from `result[0]`; corrupt batch entries are skipped before postprocessing.
- Compatibility surfaces to keep conservative: route paths, request field names, Pydantic model names and field defaults, `show_extras_results` API override, queue-lock wrapping, `html_info`, response image/null shapes, and no-disk-save behavior.

### Validation log
- `PYTHONPYCACHEPREFIX=$(mktemp -d) python3 -m py_compile modules/api/api.py modules/api/models.py modules/postprocessing.py test/test_postprocessing_api_defaults.py` - passed.
- `python3 -m pytest test/test_postprocessing_api_defaults.py test/test_postprocessing_script_args.py tests/test_api_progress_contract.py -q` - passed: 14 passed, with the existing pytest config warning `Unknown config option: base_url`.
- `rg -n "def setUpscalers|def decode_extras_batch_images|def _run_extras|def extras_single_image_api|def extras_batch_images_api|class ExtrasBaseRequest|class ExtraBaseResponse|class ExtrasSingleImage|class ExtrasBatchImages|def run_extras|def run_postprocessing" modules/api/api.py modules/api/models.py modules/postprocessing.py test/test_postprocessing_api_defaults.py` - confirmed the bridge symbols remain single, explicit contract points with no duplicate replacement candidate in this scope.
- `git diff --check` - passed.
- Live API extras requests, live WebUI extras tab behavior, and model-backed upscaling/restoration were not exercised because they require a running WebUI/model runtime and provisioned ML assets.

### Next unchecked scope
- More slices are still needed. Recommended next slice: continue into API PNG info, interrogate, metadata, model-refresh, and administrative compatibility endpoints around `modules/api/api.py`, `modules/api/models.py`, `modules/extras.py`, `modules/interrogate.py`, `modules/deepbooru.py`, and focused API contract tests, looking for dead response shims or duplicate queue-lock/admin wrappers while preserving public route schemas and extension-facing behavior.

## Pass 74 - API PNG info, interrogate, metadata, model-refresh, and administrative compatibility endpoints (2026-06-20)

### Checked scope
- `modules/api/api.py`: helper image decode/encode context around `decode_base64_to_image()`, `decode_extras_batch_images()`, `encode_pil_to_base64()`, routes for `/sdapi/v1/png-info`, `/sdapi/v1/interrogate`, `/sdapi/v1/options`, `/sdapi/v1/refresh-embeddings`, `/sdapi/v1/refresh-checkpoints`, `/sdapi/v1/refresh-vae`, `/sdapi/v1/unload-checkpoint`, and `/sdapi/v1/reload-checkpoint`, plus `pnginfoapi()`, `interrogateapi()`, `get_config()`, `set_config()`, `unloadapi()`, `reloadapi()`, and refresh wrappers.
- `modules/api/models.py`: `PNGInfoRequest`, `PNGInfoResponse`, `InterrogateRequest`, `InterrogateResponse`, dynamic `OptionsModel`, and adjacent progress/extras response shapes for compatibility context.
- `modules/extras.py`: `run_pnginfo()`, `stringify_safetensors_metadata()`, `read_metadata()`, and model-merger metadata flow as the UI/metadata counterpart to API PNG info and checkpoint metadata surfaces.
- `modules/interrogate.py` and `modules/deepbooru.py`: CLIP interrogator and DeepDanbooru routing/load/unload/tag behavior reached by the API interrogate endpoint.
- Focused API contracts: `test/test_api_info_interrogate_admin_contract.py`, `test/test_postprocessing_api_defaults.py`, `tests/test_api_progress_contract.py`, and older live API smoke tests in `test/test_extras.py` for PNG info/interrogate route coverage.

### Findings and fixes
- Consolidated the duplicate manual queue-lock wrapper in `Api.interrogateapi()` into the existing `Api._call_with_queue_lock()` helper. The endpoint still decodes the image before locking, converts it to RGB, dispatches `clip` to `shared.interrogator.interrogate()`, dispatches `deepdanbooru` to `deepbooru.model.tag()`, and raises the same `HTTPException(404, "Model not found")` for unknown models.
- Added `test/test_api_info_interrogate_admin_contract.py` to pin the interrogate endpoint's use of `_call_with_queue_lock()` and preserve CLIP/DeepDanbooru dispatch through the helper without importing the full WebUI runtime.
- Preserved `pnginfoapi()` separately from `extras.run_pnginfo()`. They look related but serve different contracts: the API route returns raw infotext, parsed generation parameters, and image metadata items, while the UI helper returns HTML display output and includes a synthetic `parameters` item for display.
- Preserved `PNGInfoResponse` and `InterrogateResponse` schemas. Their field names and default/null behavior are public `/sdapi/v1/*` route contracts and are used by existing API clients and smoke tests.
- Preserved `get_config()`/`set_config()` and the dynamic `OptionsModel` despite duplication with shared option metadata. The route returns API-visible option values and validates checkpoint aliases before applying `opts.set(..., is_api=True)` and saving the config file.
- Preserved `refresh_embeddings()`, `refresh_checkpoints()`, and `refresh_vae()` as thin route wrappers. They are public administrative compatibility endpoints and now consistently use `_call_with_queue_lock()` for mutable model-list refresh operations.
- Preserved `unloadapi()` and `reloadapi()` as compatibility wrappers for existing admin endpoints. They deliberately return `{}` and call the specific SD model weight/device helpers; changing their locking/return shape would alter live admin semantics.
- Preserved separate image decode/encode helpers. `decode_base64_to_image()` handles URL/data-URI/base64 request inputs, `decode_extras_batch_images()` intentionally skips corrupt extras batch items, and `encode_pil_to_base64()` preserves response format/metadata behavior for multiple API routes.

### Static/dynamic audit map notes
- PNG info chain remains: API request model -> `decode_base64_to_image(req.image.strip())` -> `images.read_info_from_image()` -> `infotext_utils.parse_generation_parameters()` -> `script_callbacks.infotext_pasted_callback()` -> `PNGInfoResponse(info, items, parameters)`.
- Interrogate chain now remains: API request model -> decode/convert RGB -> `_call_with_queue_lock(interrogate_image)` -> CLIP or DeepDanbooru model dispatch -> `InterrogateResponse(caption=...)`.
- Admin refresh chain remains: route -> thin wrapper -> `_call_with_queue_lock()` for embedding/checkpoint/VAE list refresh. These wrappers intentionally expose no response body beyond FastAPI's empty/null compatibility behavior.
- Options chain remains: dynamic options schema is built from `opts.data_labels`; GET returns current option data with defaults; POST validates `sd_model_checkpoint` aliases and persists via `shared.opts.save()`.
- Compatibility surfaces to keep conservative: route paths, request/response model class names, `PNGInfoResponse` fields (`info`, `items`, `parameters`), `InterrogateRequest.model` names (`clip`, `deepdanbooru`), HTTP status/detail strings for missing images/models, `OptionsModel` dynamic field defaults, admin endpoint empty responses, and model-list refresh side effects.

### Validation log
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pycompile-pass74.XXXXXX) python3 -m py_compile modules/api/api.py modules/api/models.py modules/extras.py modules/interrogate.py modules/deepbooru.py test/test_api_info_interrogate_admin_contract.py test/test_postprocessing_api_defaults.py tests/test_api_progress_contract.py` - passed.
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pytest-pass74-focused.XXXXXX) python3 -m pytest -q test/test_api_info_interrogate_admin_contract.py` - passed: 2 passed, with the existing pytest config warning `Unknown config option: base_url`.
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pytest-pass74-progress.XXXXXX) python3 -m pytest -q tests/test_api_progress_contract.py` - passed: 1 passed, with the existing pytest config warning `Unknown config option: base_url`.
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pytest-pass74.XXXXXX) python3 -m pytest -q test/test_api_info_interrogate_admin_contract.py test/test_postprocessing_api_defaults.py tests/test_api_progress_contract.py` - partially blocked in the bare environment: 9 tests passed, then `test/test_postprocessing_api_defaults.py::test_postprocessing_runner_order_override_preserves_script_defaults` failed during import with `ModuleNotFoundError: No module named 'fastapi'`; extension preload stderr also reported missing `torch`. The new pass-74 focused tests passed before this dependency failure.
- Exact AST duplicate-body scan across `modules/api/api.py`, `modules/api/models.py`, `modules/extras.py`, `modules/interrogate.py`, `modules/deepbooru.py`, `test/test_api_info_interrogate_admin_contract.py`, `test/test_postprocessing_api_defaults.py`, and `tests/test_api_progress_contract.py` found no actionable production duplicate function bodies; the only reported duplicate group was two pre-existing test-only fake `__init__` methods in `test/test_postprocessing_api_defaults.py`.
- `rg -n "with self\.queue_lock|_call_with_queue_lock|def (pnginfoapi|interrogateapi|unloadapi|reloadapi|refresh_embeddings|refresh_checkpoints|refresh_vae)|class (PNGInfo|Interrogate)" modules/api/api.py modules/api/models.py test/test_api_info_interrogate_admin_contract.py` - confirmed `interrogateapi()` now uses `_call_with_queue_lock()`, while remaining direct `with self.queue_lock` uses are precision-map/generation internals outside this slice's endpoint-remediation target.
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pycompile-pass74b.XXXXXX) python3 -m py_compile modules/api/api.py modules/api/models.py modules/extras.py modules/interrogate.py modules/deepbooru.py test/test_api_info_interrogate_admin_contract.py tests/test_api_progress_contract.py && git diff --check` - passed.
- Live `/sdapi/v1/png-info`, `/sdapi/v1/interrogate`, `/sdapi/v1/options`, refresh, unload, and reload requests were not exercised because they require a running WebUI/model runtime and provisioned ML dependencies.

### Next unchecked scope
- More slices are still needed. Recommended next slice: continue into API model-list and extension/script listing surfaces, especially `get_sd_models()`, `get_sd_vaes()`, `get_hypernetworks()`, `get_face_restorers()`, `get_realesrgan_models()`, `get_prompt_styles()`, `get_embeddings()`, `get_scripts_list()`, `get_script_info()`, `get_extensions_list()`, their `modules/api/models.py` item schemas, and focused API list-contract tests, looking for stale item fields or duplicate list-shaping helpers while preserving public route schemas and extension-facing behavior.


## Pass 75 - API model-list and extension/script listing surfaces (2026-06-20)

### Checked scope
- `modules/api/api.py`: `get_sd_models()`, `get_sd_vaes()`, `get_hypernetworks()`, `get_face_restorers()`, `get_realesrgan_models()`, `get_prompt_styles()`, `get_embeddings()`, `get_scripts_list()`, `get_script_info()`, and `get_extensions_list()` plus route registration for the corresponding `/sdapi/v1/*` listing endpoints.
- `modules/api/models.py`: `SDModelItem`, `SDVaeItem`, `HypernetworkItem`, `FaceRestorerItem`, `RealesrganItem`, `PromptStyleItem`, `EmbeddingItem`, `EmbeddingsResponse`, `ScriptsList`, `ScriptArg`, `ScriptInfo`, and `ExtensionItem` response schemas.
- Adjacent list data producers and compatibility surfaces: `modules/sd_models.py`, `modules/sd_vae.py`, `modules/styles.py`, `modules/textual_inversion/textual_inversion.py`, `modules/scripts.py`, `modules/extensions.py`, and `modules/ui_extensions.py` extension metadata preload behavior.
- Focused contracts: new `tests/test_api_listing_contract.py`, existing `tests/test_api_extension_item_contract.py`, `tests/test_extensions_metadata_contract.py`, and live smoke list references in `test/test_utils.py`.

### Findings and fixes
- Consolidated the duplicate txt2img/img2img script collection access used by `Api.get_scripts_list()` and `Api.get_script_info()` into a private `Api._script_lists()` helper. The public `/sdapi/v1/scripts` shape still filters `None` script names and returns `models.ScriptsList(txt2img=..., img2img=...)`; `/sdapi/v1/script-info` still returns non-`None` `script.api_info` objects in txt2img-then-img2img order.
- Added `tests/test_api_listing_contract.py` to pin the shared script-list source behavior without importing the full WebUI runtime.
- Preserved model-list response fields in `SDModelItem`, `SDVaeItem`, `HypernetworkItem`, `FaceRestorerItem`, `RealesrganItem`, `PromptStyleItem`, and `EmbeddingsResponse`. These fields map directly to public route schemas and live runtime metadata such as checkpoint config sidecars, VAE filenames, embedding training metadata, face restorer command directories, and RealESRGAN scales.
- Preserved `get_prompt_styles()` tuple indexing. It is tied to the existing prompt style storage shape and public API field names; replacing it with a guessed named abstraction would be compatibility churn.
- Preserved `get_extensions_list()` filtering to extensions with a non-`None` remote. `ExtensionItem.remote` and `commit_hash` remain required schema fields, while `branch` and `commit_date` are optional for missing git metadata; broadening the endpoint to non-git/builtin extensions would change the route contract rather than remove dead code.
- Preserved `ScriptArg`/`ScriptInfo` schemas and `script.api_info` generation in `modules/scripts.py`. These are extension-facing API metadata contracts, including empty hook methods in `Script` that are intentionally overridden by scripts/extensions.

### Static/dynamic audit map notes
- Script listing chain remains: route -> `Api._script_lists()` -> `scripts.scripts_txt2img.scripts` / `scripts.scripts_img2img.scripts` -> `/scripts` names or `/script-info` `api_info` payloads.
- Model listing chain remains: route -> current shared/global registry (`sd_models.checkpoints_list`, `sd_vae.vae_dict`, `shared.hypernetworks`, `shared.face_restorers`, RealESRGAN model loader, `shared.prompt_styles.styles`, embedding DB dictionaries) -> public item schemas.
- Extension listing chain remains: route -> `extensions.list_extensions()` -> per-extension `read_info_from_repo()` -> include only repo-backed extensions with a remote -> `ExtensionItem` git metadata response.
- Compatibility surfaces to keep conservative: route paths, item field names, `hash` short-hash field, optional `sha256`/config/branch/commit-date fields, script order and filtering semantics, `api_info` object shape, extension remote filtering, and embedding dictionary keys by embedding name.

### Validation log
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pycompile-pass75.XXXXXX) python3 -m py_compile modules/api/api.py modules/api/models.py modules/extensions.py modules/scripts.py modules/sd_models.py modules/sd_vae.py modules/styles.py modules/textual_inversion/textual_inversion.py tests/test_api_listing_contract.py tests/test_api_extension_item_contract.py tests/test_extensions_metadata_contract.py` - passed.
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pytest-pass75.XXXXXX) python3 -m pytest -q tests/test_api_listing_contract.py tests/test_api_extension_item_contract.py tests/test_extensions_metadata_contract.py` - passed: 5 passed, with the existing pytest config warning `Unknown config option: base_url`.
- Exact AST duplicate-body scan across `modules/api/api.py`, `modules/api/models.py`, `modules/extensions.py`, `modules/scripts.py`, `modules/sd_models.py`, `modules/sd_vae.py`, `modules/styles.py`, `modules/textual_inversion/textual_inversion.py`, and focused list/extension tests reported one pre-existing duplicate empty hook pair in `modules/scripts.py` (`postprocess_image()` / `postprocess_maskoverlay()`), preserved as extension-visible script hooks outside this slice's remediation target.
- `git diff --check` - passed.
- Live `/sdapi/v1/sd-models`, `/sdapi/v1/sd-vae`, `/sdapi/v1/hypernetworks`, `/sdapi/v1/face-restorers`, `/sdapi/v1/realesrgan-models`, `/sdapi/v1/prompt-styles`, `/sdapi/v1/embeddings`, `/sdapi/v1/scripts`, `/sdapi/v1/script-info`, and `/sdapi/v1/extensions` requests were not exercised because they require a running WebUI/model runtime and provisioned extension/model state.

### Next unchecked scope
- More slices are still needed. Recommended next slice: continue into API memory/progress/queue/status and task-control surfaces, especially `get_memory()`, `progressapi()`, `skip()`, `interrupt()`, task-id/state handling, queue-lock/precision-map interactions, `MemoryResponse`/`ProgressResponse`, and focused API progress/task tests, looking for dead response fields or duplicate status-shaping helpers while preserving public API schemas and live task semantics.

## Pass 76 - API memory/progress/queue/status and task-control surfaces (2026-06-20)

### Checked scope
- `modules/api/api.py`: route registration for `/sdapi/v1/progress`, `/sdapi/v1/interrupt`, `/sdapi/v1/skip`, `/sdapi/v1/memory`, `/sdapi/v1/openclaw/precision-map`, and `/sdapi/v1/precision-map`; `get_precision_map()`, `_call_with_queue_lock()`, `_clear_pending_task_unless_finished()`, `_finish_generation_task()`, `text2imgapi()`, `img2imgapi()`, `progressapi()`, `interruptapi()`, `skip()`, and `get_memory()`.
- `modules/api/models.py`: `ProgressRequest`, `ProgressResponse`, and `MemoryResponse` public API schemas.
- `modules/progress.py`: `current_task`, `pending_tasks`, `finished_tasks`, `create_task_id()`, `add_task_to_queue()`, `start_task()`, `finish_task()`, `calculate_progress_and_eta()`, `/internal/pending-tasks`, `/internal/progress`, queue status shaping, live-preview encoding, and `restore_progress()`.
- Focused contracts and live-test references: `tests/test_api_progress_contract.py`, `test/test_postprocessing_api_defaults.py::test_api_progress_reports_live_current_task_reference`, and `test/test_img2img.py` forced-task cleanup checks.

### Findings and fixes
- No safe production dead-code or duplication remediation was made in this slice. The apparent overlap between public `/sdapi/v1/progress` and internal `/internal/progress` is intentional: the public API returns the compatibility `ProgressResponse` with `progress`, `eta_relative`, `state`, optional raw base64 `current_image`, `textinfo`, and `current_task`; the internal endpoint returns task-centric queue state (`active`, `queued`, `completed`), data-URI live previews keyed by `id_live_preview`, and queue-position text for UI restore/polling.
- Preserved `ProgressResponse.current_task`. Existing focused contracts pin that `Api.progressapi()` exposes `progress_module.current_task`, and removing it would break active task-id visibility for API clients.
- Preserved separate base64 handling in `Api.progressapi()` and `modules.progress.progressapi()`. The public API uses `encode_pil_to_base64()` and honors `skip_current_image`; internal progress intentionally follows `opts.live_previews_enable`, `opts.live_previews_image_format`, preview ID deduplication, and `data:image/...;base64,...` response formatting.
- Preserved `interruptapi()` and `skip()` as separate public task-control routes. They dispatch to different shared-state operations (`interrupt()` vs `skip()`), and the skip endpoint's empty/implicit response shape is a public compatibility surface.
- Preserved `get_memory()` and `MemoryResponse` dictionary fields. RAM and CUDA payload subkeys are intentionally runtime-dependent and error-tolerant, so replacing them with stricter structured models would risk breaking clients and cross-platform behavior.
- Preserved task-id handling in `text2imgapi()` and `img2imgapi()`: `force_task_id` support, `add_task_to_queue()`, `start_task()`, `finish_task()`, bounded `finished_tasks`, and pending cleanup through `_clear_pending_task_unless_finished()` are live semantics used by internal progress polling and forced-task regression tests.
- Preserved `get_precision_map()`'s direct `queue_lock` use instead of folding it into task-control helpers. Precision-map generation walks `shared.sd_model` and must stay serialized with generation/model reload windows; it is not a stale task-control wrapper.

### Static/dynamic audit map notes
- Public progress chain remains: `/sdapi/v1/progress` -> `Api.progressapi()` -> `progress_module.calculate_progress_and_eta()` -> `shared.state.set_current_image()` when needed -> `models.ProgressResponse(...)` with `current_task=progress_module.current_task`.
- Internal task progress chain remains: txt2img/img2img forced/generated task id -> `add_task_to_queue()` -> `start_task()` under `queue_lock` -> `finish_task()`/`shared.state.end()`/`shared.total_tqdm.clear()` -> `/internal/progress` and `/internal/pending-tasks` observe `current_task`, `pending_tasks`, and `finished_tasks`.
- Memory chain remains: `/sdapi/v1/memory` -> `psutil.Process(...).memory_info()` plus calculated RAM total -> optional CUDA `torch.cuda.mem_get_info()` and `torch.cuda.memory_stats(shared.device)` -> `models.MemoryResponse(ram=..., cuda=...)`.
- Compatibility surfaces to keep conservative: route paths, response field names, `skip_current_image`, `current_task`, task-id prefixes and `force_task_id`, pending/finished task semantics, live preview data-URI format, `interrupt`/`skip` empty response behavior, memory dict subkeys, and precision-map queue-lock serialization.

### Validation log
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pycompile-pass76.XXXXXX) python3 -m py_compile modules/api/api.py modules/api/models.py modules/progress.py test/test_postprocessing_api_defaults.py tests/test_api_progress_contract.py test/test_img2img.py` - passed.
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pytest-pass76.XXXXXX) python3 -m pytest -q tests/test_api_progress_contract.py test/test_postprocessing_api_defaults.py::test_api_progress_reports_live_current_task_reference` - passed: 2 passed, with the existing pytest config warning `Unknown config option: base_url`.
- Exact AST duplicate-body scan across `modules/api/api.py`, `modules/api/models.py`, `modules/progress.py`, `test/test_postprocessing_api_defaults.py`, `tests/test_api_progress_contract.py`, and `test/test_img2img.py` found no actionable production duplicate function bodies. Reported duplicates were test-only fake methods in `test/test_postprocessing_api_defaults.py` and one trivial test fake method overlap (`set_current_image()`/`nextjob()`).
- `rg -n "pending_tasks|finished_tasks|current_task|force_task_id|start_task\\(|finish_task\\(|restore_progress\\(|/internal/progress|/internal/pending-tasks" modules test tests -S` - confirmed task state producers/consumers are limited and live in generation, internal progress, and focused task cleanup tests.
- `git diff --check` - passed.
- Live `/sdapi/v1/progress`, `/sdapi/v1/memory`, `/sdapi/v1/interrupt`, `/sdapi/v1/skip`, `/internal/progress`, and `/internal/pending-tasks` requests were not exercised because this bounded slice did not start a WebUI/model runtime.

### Next unchecked scope
- More slices are still needed. Recommended next slice: continue into API create/train embedding and hypernetwork task surfaces, especially `create_embedding()`, `create_hypernetwork()`, `train_embedding()`, `train_hypernetwork()`, `_run_create_task()`, `_run_training_task()`, training device restore/prepare helpers, `CreateResponse`/`TrainResponse`, and focused training/create tests, looking for stale wrappers or duplicate task lifecycle handling while preserving public response strings and training side effects.

## Pass 77 - API create/train embedding and hypernetwork task surfaces (2026-06-20)

### Checked scope
- `modules/api/api.py`: route registration for `/sdapi/v1/create/embedding`, `/sdapi/v1/create/hypernetwork`, `/sdapi/v1/train/embedding`, and `/sdapi/v1/train/hypernetwork`; `_create_response()`, `_train_response()`, `_run_create_task()`, `_run_training_task()`, `create_embedding()`, `create_hypernetwork()`, `_prepare_hypernetwork_training()`, `_restore_hypernetwork_training_devices()`, `train_embedding()`, and `train_hypernetwork()`.
- `modules/api/models.py`: `CreateResponse` and `TrainResponse` public API schemas.
- Adjacent runtime/UI producers: `modules/textual_inversion/textual_inversion.py`, `modules/textual_inversion/ui.py`, `modules/hypernetworks/hypernetwork.py`, and `modules/hypernetworks/ui.py` create/train entry points and device/optimization side effects.
- Focused contracts: `tests/test_api_training_contract.py`, `tests/test_hypernetwork_creation_contract.py`, and `tests/test_textual_inversion_preview_save_contract.py`.

### Findings and fixes
- No safe production dead-code or duplication remediation was made in this slice. The apparent private wrappers are live route handlers or callback helpers for the public create/train endpoints and remain covered by focused AST contracts from the earlier helper extraction.
- Preserved `_create_response()` and `_train_response()` even though they are tiny. They intentionally keep `CreateResponse` and `TrainResponse` schema construction separate; collapsing them into a generic response helper would trade explicit API schema boundaries for cosmetic deduplication.
- Preserved `_run_create_task()` and `_run_training_task()` as separate lifecycle helpers. Create tasks only begin/end shared state, call the creator, optionally refresh embeddings, and translate `AssertionError` into the historical create error string. Training tasks additionally handle before/after hooks, xattention optimization toggling, inner train exceptions, and the historical `"... error: {error}"` success-path response format.
- Preserved `create_embedding()`/`create_hypernetwork()`/`train_embedding()`/`train_hypernetwork()` wrappers and their literal response prefixes. These strings are public API compatibility surfaces and the wrappers bind distinct imported train/create functions, job names, and side-effect hooks.
- Preserved `_prepare_hypernetwork_training()` and `_restore_hypernetwork_training_devices()`. They are single-use but intentional callback seams for hypernetwork-only side effects: clearing `shared.loaded_hypernetworks` before training and restoring cond/first-stage models to `devices.device` afterward. Inlining would not remove real dead code and would make the generic training helper less clear.
- Preserved similar-looking UI training/create lifecycle code in `modules/textual_inversion/ui.py` and `modules/hypernetworks/ui.py`. The UI paths return Gradio update tuples/human HTML messages and re-raise exceptions, while the API paths return public response models and compatibility strings; sharing them would risk route and UI behavior churn.
- Preserved `CreateResponse.info` and `TrainResponse.info` schemas. They are directly referenced by route `response_model`s and encode different public endpoint families despite having the same field shape.

### Static/dynamic audit map notes
- Create embedding chain remains: `/sdapi/v1/create/embedding` -> `Api.create_embedding()` -> `_run_create_task("create_embedding", textual_inversion.create_embedding, ..., after_create=embedding_db.load_textual_inversion_embeddings)` -> `CreateResponse(info="create embedding filename: ..." | "create embedding error: ...")`.
- Create hypernetwork chain remains: `/sdapi/v1/create/hypernetwork` -> `Api.create_hypernetwork()` -> `_run_create_task("create_hypernetwork", hypernetwork.create_hypernetwork, ...)` -> `CreateResponse(info="create hypernetwork filename: ..." | "create hypernetwork error: ...")`.
- Train embedding chain remains: `/sdapi/v1/train/embedding` -> `Api.train_embedding()` -> `_run_training_task("train_embedding", textual_inversion.train_embedding, ...)` with optional xattention optimization undo/apply -> `TrainResponse(info="train embedding complete: filename: ... error: ..." | "train embedding error: ...")`.
- Train hypernetwork chain remains: `/sdapi/v1/train/hypernetwork` -> `Api.train_hypernetwork()` -> `_prepare_hypernetwork_training()` -> `hypernetwork.train_hypernetwork()` -> `_restore_hypernetwork_training_devices()` and optional optimization restore -> `TrainResponse(info="train hypernetwork complete: filename: ... error: ..." | "train hypernetwork error: ...")`.
- Compatibility surfaces to keep conservative: route paths, response model class names, `info` field name, exact response prefixes, inclusion of `error: None` on successful training, job names passed to `shared.state.begin()`, embedding refresh after create, hypernetwork loaded-list clearing, device restore to `devices.device`, and optimization undo/apply gating by `shared.opts.training_xattention_optimizations`.

### Validation log
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pycompile-pass77.XXXXXX) python3 -m py_compile modules/api/api.py modules/api/models.py modules/textual_inversion/textual_inversion.py modules/textual_inversion/ui.py modules/hypernetworks/hypernetwork.py modules/hypernetworks/ui.py tests/test_api_training_contract.py tests/test_hypernetwork_creation_contract.py tests/test_textual_inversion_preview_save_contract.py` - passed.
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pytest-pass77.XXXXXX) python3 -m pytest -q tests/test_api_training_contract.py tests/test_hypernetwork_creation_contract.py tests/test_textual_inversion_preview_save_contract.py` - passed: 6 passed, with the existing pytest config warning `Unknown config option: base_url`.
- Exact AST duplicate-body scan across the checked API, model, training/UI, and focused test files reported only the pre-existing `TextToImageResponse`/`ImageToImageResponse` response-model twins in `modules/api/models.py`, outside this slice's create/train target.
- `rg -n "shared\\.state\\.(begin|end)|sd_hijack\\.(undo_optimizations|apply_optimizations)|loaded_hypernetworks|cond_stage_model\\.to|first_stage_model\\.to|create .*filename|train .*complete|error:" modules/api modules/textual_inversion modules/hypernetworks tests test -S` - confirmed the API create/train lifecycle overlaps are limited and intentionally differ from UI train/create wrappers.
- `git diff --check` - passed.
- Live create/train embedding and hypernetwork API requests were not exercised because they require a running WebUI/model runtime, training assets, and provisioned ML dependencies.

### Next unchecked scope
- More slices are still needed. Recommended next slice: continue into API options/configuration and command-control surfaces, especially `get_config()`, `set_config()`, command-line flag exposure, `/sdapi/v1/options`, `/sdapi/v1/cmd-flags`, `/sdapi/v1/samplers`, `/sdapi/v1/schedulers`, `/sdapi/v1/upscalers`, related response schemas/list helpers, and focused option/list tests, looking for stale fields or duplicate list-shaping helpers while preserving public route schemas and runtime side effects.

## Pass 78 - API options/configuration and command/list surfaces (2026-06-20)

### Checked scope
- `modules/api/api.py`: route registration and handlers for `/sdapi/v1/options`, `/sdapi/v1/cmd-flags`, `/sdapi/v1/samplers`, `/sdapi/v1/schedulers`, and `/sdapi/v1/upscalers`; `get_config()`, `set_config()`, `get_cmd_flags()`, `get_samplers()`, `get_schedulers()`, and `get_upscalers()`.
- `modules/api/models.py`: dynamic `OptionsModel`, dynamic `FlagsModel`, `SamplerItem`, `SchedulerItem`, and `UpscalerItem` response schemas.
- Adjacent option/list producers and compatibility surfaces: `modules/options.py`, `modules/shared.py`, `modules/shared_options.py`, `modules/sd_samplers.py`, `modules/sd_schedulers.py`, and `modules/modelloader.py` upscaler loading/list state.
- Focused tests and smoke references: `test/test_utils.py`, `tests/test_api_listing_contract.py`, and `tests/test_api_progress_contract.py` where they exercise API list/progress contract helpers without requiring a live WebUI server.

### Findings and fixes
- Removed one dead local assignment in `modules/api/models.py`: `value = opts.data.get(key)` inside the `OptionsModel` dynamic field loop was never read and had no side effects. The dynamic options schema still uses `opts.data_labels`, each option's default, label text, and `opts.typemap` exactly as before.
- Preserved `get_config()`'s explicit current-value/default fallback behavior. Although it parallels the dynamic `OptionsModel` field generation, the route returns runtime values while the schema describes available option fields, so folding them together would risk changing settings compatibility.
- Preserved `set_config()`'s direct `sd_model_checkpoint` alias guard followed by `shared.opts.set(..., is_api=True)` and `shared.opts.save(...)`. This is live validation/persistence behavior rather than duplicate generic option-setting code.
- Preserved `get_cmd_flags()` returning `vars(shared.cmd_opts)`. It intentionally reports parsed runtime command options; the `FlagsModel` builder reads the parser's option actions for documentation/default schema and is not a replacement for live command flag values.
- Preserved the separate sampler, scheduler, and upscaler list comprehensions. They read from different runtime registries (`sd_samplers.all_samplers`, `sd_schedulers.schedulers`, and `shared.sd_upscalers`) and expose distinct public item fields; a generic list-shaping helper would add abstraction without removing real duplication.
- Preserved public response schema field names and route paths for options, cmd-flags, samplers, schedulers, and upscalers, including sampler aliases/options, scheduler `default_rho`/`need_inner_model`, and upscaler `model_url=None` compatibility.

### Static/dynamic audit map notes
- Options GET chain remains: `/sdapi/v1/options` -> `Api.get_config()` -> `shared.opts.data` with defaults from `shared.opts.data_labels` -> dynamic `OptionsModel` response schema.
- Options POST chain remains: `/sdapi/v1/options` -> `Api.set_config()` -> checkpoint alias validation through `sd_models.checkpoint_aliases` -> `shared.opts.set(..., is_api=True)` for every submitted key -> `shared.opts.save(shared.config_filename)`.
- Command flag chain remains: `/sdapi/v1/cmd-flags` -> `Api.get_cmd_flags()` -> `vars(shared.cmd_opts)` with `FlagsModel` generated from `shared.parser` option actions.
- List chains remain: `/sdapi/v1/samplers` -> `sd_samplers.all_samplers`; `/sdapi/v1/schedulers` -> `sd_schedulers.schedulers`; `/sdapi/v1/upscalers` -> `shared.sd_upscalers` entries populated by `modelloader.load_upscalers()` and startup fallback.
- Compatibility surfaces to keep conservative: option keys/labels/defaults/aliases, `is_api=True` option-set behavior, config save side effects, command-flag response values, sampler tuple indexing, scheduler labels/aliases/default rho/inner-model flag, upscaler model metadata, and existing live-smoke endpoint list in `test/test_utils.py`.

### Validation log
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pycompile-pass78.XXXXXX) python3 -m py_compile modules/api/api.py modules/api/models.py modules/options.py modules/shared.py modules/shared_options.py modules/sd_samplers.py modules/sd_schedulers.py modules/modelloader.py test/test_utils.py tests/test_api_listing_contract.py tests/test_api_progress_contract.py` - passed.
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pytest-pass78.XXXXXX) python3 -m pytest -q tests/test_api_listing_contract.py tests/test_api_progress_contract.py` - passed: 2 passed, with the existing pytest config warning `Unknown config option: base_url`.
- Exact AST duplicate-body scan across `modules/api/api.py`, `modules/api/models.py`, `modules/options.py`, `modules/shared.py`, `modules/shared_options.py`, `modules/sd_samplers.py`, `modules/sd_schedulers.py`, `modules/modelloader.py`, `test/test_utils.py`, `tests/test_api_listing_contract.py`, and `tests/test_api_progress_contract.py` found no duplicate function bodies.
- `git diff --check` - passed.
- Live `/sdapi/v1/options`, `/sdapi/v1/cmd-flags`, `/sdapi/v1/samplers`, `/sdapi/v1/schedulers`, and `/sdapi/v1/upscalers` requests were not exercised because this bounded slice did not start a WebUI/model runtime.

### Next unchecked scope
- More slices are still needed. Recommended next slice: continue into API latent upscale modes and remaining static metadata/list endpoints adjacent to the current surface, especially `get_latent_upscale_modes()`, `/sdapi/v1/latent-upscale-modes`, any remaining response item schemas not covered by passes 75 and 78, and focused endpoint smoke/list tests, looking for stale fields or duplicate list-shaping helpers while preserving public route schemas and runtime compatibility.

## Pass 79 - API latent upscale modes and adjacent static list endpoints (2026-06-20)

### Checked scope
- `modules/api/api.py`: route registration and handlers for `/sdapi/v1/samplers`, `/sdapi/v1/schedulers`, `/sdapi/v1/upscalers`, `/sdapi/v1/latent-upscale-modes`, and adjacent static metadata/list handlers; focused inspection of `get_samplers()`, `get_schedulers()`, `get_upscalers()`, `get_latent_upscale_modes()`, `get_sd_models()`, `get_sd_vaes()`, `get_hypernetworks()`, `get_face_restorers()`, `get_realesrgan_models()`, `get_prompt_styles()`, and `get_embeddings()` boundaries from the nearby list surface.
- `modules/api/models.py`: `SamplerItem`, `SchedulerItem`, `UpscalerItem`, `LatentUpscalerModeItem`, and adjacent static list response item schemas previously mapped in passes 75 and 78.
- `modules/shared.py`: `latent_upscale_modes`, `latent_upscale_default_mode`, `sd_upscalers`, and UI/processing consumers for latent hires upscale mode names.
- Focused tests and smoke references: `tests/test_api_listing_contract.py` and `test/test_utils.py` static API list smoke parameterization.

### Findings and fixes
- Removed one small duplicate list-materialization step in `Api.get_latent_upscale_modes()`: iterating `(shared.latent_upscale_modes or {})` preserves dict key order and the existing empty fallback while avoiding the stale-looking `[*(...)]` copy used only for iteration.
- Added a focused AST-isolated contract test for `get_latent_upscale_modes()` to pin that the endpoint returns only public `{"name": ...}` items in the same order as `shared.latent_upscale_modes` keys. This protects the public route schema while keeping internal interpolation details private.
- Added `/sdapi/v1/schedulers` and `/sdapi/v1/latent-upscale-modes` to the existing live API smoke URL list beside samplers and upscalers. These are adjacent public static list endpoints and were missing from the smoke surface.
- Preserved `LatentUpscalerModeItem` as a separate response model even though it has one field. The route is a public API schema, and merging it with unrelated one-field item schemas would be cosmetic rather than a safe dead-code removal.
- Preserved `shared.latent_upscale_modes` values and UI/processing consumers. The API intentionally exposes only mode names; the values are live runtime interpolation settings consumed by hires-fix processing.
- Preserved separate sampler, scheduler, upscaler, latent-mode, model, VAE, hypernetwork, face-restorer, RealESRGAN, prompt-style, and embedding list helpers. They read from distinct registries and expose distinct public item shapes, so generic static-list wrapping would add abstraction without eliminating real duplicate behavior.

### Static/dynamic audit map notes
- Latent upscale API chain remains: `/sdapi/v1/latent-upscale-modes` -> `Api.get_latent_upscale_modes()` -> keys from `shared.latent_upscale_modes` -> `models.LatentUpscalerModeItem(name=...)` response validation.
- Latent upscale runtime chain remains: UI hires-fix dropdown choices from `[*shared.latent_upscale_modes, *[x.name for x in shared.sd_upscalers]]` -> `StableDiffusionProcessing` lookup in `shared.latent_upscale_modes` -> `torch.nn.functional.interpolate(..., mode=..., antialias=...)` for latent hires upscale modes.
- Adjacent list smoke chain now includes command flags, samplers, schedulers, upscalers, latent upscale modes, models, hypernetworks, face restorers, RealESRGAN models, prompt styles, and embeddings.
- Compatibility surfaces to keep conservative: `/sdapi/v1/latent-upscale-modes` path, response item field name `name`, mode-name ordering, empty fallback if `shared.latent_upscale_modes` is absent/falsey, hidden internal interpolation config values, and all adjacent static list response field names.

### Validation log
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pycompile-pass79b.XXXXXX) python3 -m py_compile modules/api/api.py modules/api/models.py modules/shared.py test/test_utils.py tests/test_api_listing_contract.py` - passed.
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pytest-pass79b.XXXXXX) python3 -m pytest -q tests/test_api_listing_contract.py` - passed: 2 passed, with the existing pytest config warning `Unknown config option: base_url`.
- Exact AST duplicate-body scan across `modules/api/api.py`, `modules/api/models.py`, `modules/shared.py`, `test/test_utils.py`, and `tests/test_api_listing_contract.py` found no duplicate function bodies.
- `git diff --check` - passed after removing one trailing blank line in the new listing contract test.
- Live `/sdapi/v1/schedulers` and `/sdapi/v1/latent-upscale-modes` smoke requests were not exercised because this bounded slice did not start a WebUI/model runtime.

### Next unchecked scope
- More slices are still needed. Recommended next slice: continue into API refresh/checkpoint reload and server-control endpoints, especially `refresh_embeddings()`, `refresh_checkpoints()`, `refresh_vae()`, `unloadapi()`, `reloadapi()`, `kill_webui()`, `restart_webui()`, `stop_webui()`, route gating by `api_server_stop`, and adjacent response/side-effect tests, looking for stale wrappers or duplicate command-control logic while preserving public route behavior and runtime side effects.

## Pass 80 - API refresh/checkpoint reload and server-control endpoints (2026-06-20)

### Checked scope
- `modules/api/api.py`: route registration for `/sdapi/v1/refresh-embeddings`, `/sdapi/v1/refresh-checkpoints`, `/sdapi/v1/refresh-vae`, `/sdapi/v1/unload-checkpoint`, `/sdapi/v1/reload-checkpoint`, and route-gated `/sdapi/v1/server-kill`, `/sdapi/v1/server-restart`, `/sdapi/v1/server-stop`; `refresh_embeddings()`, `refresh_checkpoints()`, `refresh_vae()`, `unloadapi()`, `reloadapi()`, `kill_webui()`, `restart_webui()`, `stop_webui()`, `add_api_route()`, and the `api_server_stop` gate in `Api.__init__()`.
- Adjacent side-effect helpers: `modules/shared_items.py` refresh wrappers, `modules/sd_models.py` checkpoint unload/reload/send-to-device surface, `modules/restart.py` restartability and stop/restart helpers, and API queue-lock serialization via `_call_with_queue_lock()`.
- Focused tests: new `tests/test_api_server_control_contract.py` plus adjacent listing contract smoke for route/list regression context.

### Findings and fixes
- Renamed the bound endpoint parameter in `Api.stop_webui()` from `request` to `self`. The method never accepted or used a FastAPI request object; the old name was stale and obscured that this is a normal instance route handler. Runtime behavior is unchanged: it still sets `shared.state.server_command = "stop"` and returns `Response("Stopping.")`.
- Added focused AST-isolated contracts for this API surface. The tests pin queue-lock use and side-effect targets for the three refresh endpoints, empty `{}` responses for checkpoint unload/reload, route gating by `shared.cmd_opts.api_server_stop`, and server-control side effects/status behavior without starting WebUI or invoking real process exits.
- Preserved `refresh_embeddings()`, `refresh_checkpoints()`, and `refresh_vae()` as separate route handlers despite similar queue-lock wrappers. They bind distinct public route paths and distinct runtime refresh targets, and folding them into a generic dispatch table would not remove dead code.
- Preserved `unloadapi()` and `reloadapi()` as separate public checkpoint command endpoints. They intentionally differ: unload delegates to `sd_models.unload_model_weights()` while reload sends the current `shared.sd_model` back to device through `sd_models.send_model_to_device()`.
- Preserved `kill_webui()`, `restart_webui()`, and `stop_webui()` as distinct server-control commands behind `--api-server-stop`. Their side effects and response semantics differ, and `kill_webui()`/restartable `restart_webui()` intentionally reach process-control helpers that tests must stub rather than execute.
- Preserved the `restart_webui()` 501 response even after a restartable restart command. This is existing public status behavior and changing it would be API churn outside a dead-code/duplication cleanup.

### Static/dynamic audit map notes
- Refresh chains remain: `/sdapi/v1/refresh-embeddings` -> queue lock -> `sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings(force_reload=True)`; `/sdapi/v1/refresh-checkpoints` -> queue lock -> `shared.refresh_checkpoints()` -> `shared_items.refresh_checkpoints()` -> `sd_models.list_models()`; `/sdapi/v1/refresh-vae` -> queue lock -> `shared_items.refresh_vae_list()` -> `sd_vae.refresh_vae_list()`.
- Checkpoint command chains remain: `/sdapi/v1/unload-checkpoint` -> `sd_models.unload_model_weights()` -> `{}`; `/sdapi/v1/reload-checkpoint` -> `sd_models.send_model_to_device(shared.sd_model)` -> `{}`.
- Server-control chains remain gated by `shared.cmd_opts.api_server_stop`: `/sdapi/v1/server-kill` -> `restart.stop_program()`; `/sdapi/v1/server-restart` -> `restart.restart_program()` only when `restart.is_restartable()` is true, then `Response(status_code=501)`; `/sdapi/v1/server-stop` -> `shared.state.server_command = "stop"` and `Response("Stopping.")`.
- Compatibility surfaces to keep conservative: route paths, POST methods, route absence when `--api-server-stop` is not enabled, queue-lock serialization for refresh endpoints, force-reload embedding refresh flag, empty dict checkpoint command responses, server-stop response body, restart 501 status, and process-control side effects.

### Validation log
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pycompile-pass80.XXXXXX) python3 -m py_compile modules/api/api.py tests/test_api_server_control_contract.py` - passed.
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pycompile-pass80.XXXXXX) python3 -m py_compile modules/api/api.py modules/shared_items.py modules/sd_models.py modules/restart.py tests/test_api_server_control_contract.py tests/test_api_listing_contract.py` - passed.
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pytest-pass80.XXXXXX) python3 -m pytest -q tests/test_api_server_control_contract.py` - passed: 4 passed, with the existing pytest config warning `Unknown config option: base_url`.
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pytest-pass80.XXXXXX) python3 -m pytest -q tests/test_api_server_control_contract.py tests/test_api_listing_contract.py` - passed: 6 passed, with the existing pytest config warning `Unknown config option: base_url`.
- Exact AST duplicate-body scan across `modules/api/api.py`, `modules/shared_items.py`, `modules/sd_models.py`, `modules/restart.py`, `tests/test_api_server_control_contract.py`, and `tests/test_api_listing_contract.py` found no duplicate function bodies.
- `git diff --check` - passed.
- Live refresh/checkpoint/server-control API requests were not exercised because this bounded slice did not start a WebUI/model runtime and must not invoke real process-stop/restart helpers.

### Next unchecked scope
- More slices are still needed. Recommended next slice: continue into API scripts/extensions metadata surfaces, especially `get_scripts_list()`, `get_script_info()`, `get_extensions_list()`, extension enabled/path metadata, and adjacent script API info helpers, looking for stale wrappers or duplicate list shaping while preserving public route schemas and extension/script compatibility.

## Pass 81 - API scripts/extensions metadata surfaces (2026-06-20)

### Checked scope
- `modules/api/api.py`: route registration and handlers for `/sdapi/v1/scripts`, `/sdapi/v1/script-info`, and `/sdapi/v1/extensions`; focused inspection of `_script_lists()`, `get_scripts_list()`, `get_script_info()`, and `get_extensions_list()`.
- `modules/api/models.py`: `ScriptsList`, `ScriptArg`, `ScriptInfo`, and `ExtensionItem` response schemas.
- `modules/scripts.py`: script `api_info` construction in `ScriptRunner.create_script_ui_inner()`, script name filtering, always-on/img2img flags, control argument metadata, and adjacent script callback wrappers surfaced by the duplicate scan.
- `modules/extensions.py`: extension discovery, enabled state, path metadata, repo metadata cache fields, `read_info_from_repo()`, remote filtering, and `Extension.path`/`extension_paths` behavior.
- Focused tests: `tests/test_api_listing_contract.py`, `tests/test_api_extension_item_contract.py`, and `tests/test_extensions_metadata_contract.py`.

### Findings and fixes
- Added focused contract coverage for `get_extensions_list()` in `tests/test_api_listing_contract.py`. The new AST-isolated test pins the existing public API shape: refreshes the extension registry, calls `read_info_from_repo()` for each discovered extension, includes only remote-backed extensions, preserves `enabled=False`, permits nullable `branch`/`commit_date`, and does not expose internal extension `path` metadata.
- Preserved `_script_lists()` as the shared script registry seam for `get_scripts_list()` and `get_script_info()`. Pass 75 already consolidated the duplicated txt2img/img2img source lookup, and the current functions now differ only in public response shape: selectable script names versus prebuilt `ScriptInfo` records.
- Preserved `ScriptRunner.create_script_ui_inner()` script info assembly. It is the live owner of script UI/control metadata and captures labels, defaults, ranges, choices, always-on state, and img2img state while controls are built; rebuilding this in the API endpoint would duplicate dynamic UI behavior and risk extension/script compatibility.
- Preserved `get_extensions_list()` direct field mapping instead of extracting a serializer. No repeated extension API serializer body was found in this slice, and the route remote-only filter plus public fields are compatibility behavior rather than dead code.
- Preserved absence of `path` from `ExtensionItem`. `Extension.path` and `extensions.extension_paths` are live internal/runtime metadata for extension discovery and `find_extension()` lookup, but exposing filesystem paths through `/sdapi/v1/extensions` would be public schema expansion, not dead-code remediation.
- Preserved `Extension.cached_fields` without `enabled` or `path`. The cache is intentionally repo metadata only; `enabled` comes from current options during `list_extensions()`, and `path` is runtime discovery state.

### Static/dynamic audit map notes
- Script list chains remain: `/sdapi/v1/scripts` -> `Api.get_scripts_list()` -> `_script_lists()` -> `scripts.scripts_txt2img.scripts`/`scripts.scripts_img2img.scripts` -> non-`None` script names; `/sdapi/v1/script-info` -> same sources -> non-`None` `script.api_info` records built during UI/script setup.
- Script info build chain remains: `ScriptRunner.create_script_ui_inner()` -> `script.ui(...)` controls -> `ScriptArg` label/value/minimum/maximum/step/choices -> `ScriptInfo(name, is_img2img, is_alwayson, args)`.
- Extension listing chain remains: `/sdapi/v1/extensions` -> `extensions.list_extensions()` -> per-extension `read_info_from_repo()` -> include only entries with `remote is not None` -> public `ExtensionItem` fields `name`, `remote`, `branch`, `commit_hash`, `commit_date`, `version`, and `enabled`.
- Compatibility surfaces to keep conservative: route paths, response field names, script ordering, filtering of `None` script names/api_info, script control choice normalization, extension remote-only filtering, nullable git branch/date metadata, extension enabled state, and non-exposure of internal extension filesystem paths.

### Validation log
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pycompile-pass81.XXXXXX) python3 -m py_compile modules/api/api.py modules/api/models.py modules/extensions.py modules/scripts.py tests/test_api_listing_contract.py tests/test_api_extension_item_contract.py tests/test_extensions_metadata_contract.py` - passed.
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pytest-pass81.XXXXXX) python3 -m pytest -q tests/test_api_listing_contract.py tests/test_api_extension_item_contract.py tests/test_extensions_metadata_contract.py` - passed: 7 passed, with the existing pytest config warning `Unknown config option: base_url`.
- Exact AST duplicate-body scan across `modules/api/api.py`, `modules/api/models.py`, `modules/extensions.py`, `modules/scripts.py`, `tests/test_api_listing_contract.py`, `tests/test_api_extension_item_contract.py`, and `tests/test_extensions_metadata_contract.py` found no actionable API metadata duplicates. The only duplicate-body hit was the pre-existing adjacent `modules/scripts.py` callback pair `postprocess_image()`/`postprocess_maskoverlay()`, outside this API metadata surface.
- `git diff --check` - passed.
- Live `/sdapi/v1/scripts`, `/sdapi/v1/script-info`, and `/sdapi/v1/extensions` requests were not exercised because this bounded slice did not start a WebUI/model runtime.

### Next unchecked scope
- More slices are still needed. Recommended next slice: continue into API memory and diagnostics/runtime metadata endpoints adjacent to this region, especially `get_memory()`, CUDA/RAM response shaping, generation diagnostics/CUDA graph status toggles, and any nearby low-level API status helpers, looking for stale response fields or duplicate stats shaping while preserving public route schemas and runtime side effects.

## Pass 82 - API memory and diagnostics/runtime metadata endpoints (2026-06-20)

### Checked scope
- `modules/api/api.py`: route registration and handlers for `/sdapi/v1/memory`, `/sdapi/v1/openclaw/cuda-graphs`, `/sdapi/v1/openclaw/generation-diagnostics`, and adjacent runtime-default helpers `_env_flag()`, `apply_openclaw_runtime_defaults()`, `get_cuda_graphs()`, `set_cuda_graphs()`, and `get_openclaw_generation_diagnostics()`.
- `modules/api/models.py`: `MemoryResponse` public schema for RAM/CUDA dictionaries.
- `modules/openclaw_generation_diagnostics.py`: `_cuda_graph_status()`, `_summarize_graph_key()`, `_summarize_cuda_graph_status()`, `_counter_delta()`, `_interesting_extra_params()`, `_request_summary()`, `before_sample()`, `after_sample()`, and `last_generation_diagnostics()`.
- `modules/openclaw_cuda_graphs.py`: status/toggle surface used by the API and generation diagnostics, including `status()`, `set_enabled()`, `clear()`, cache/stat counters, bypass reasons, and graph-key summary inputs.
- Focused tests: `tests/test_api_server_control_contract.py` and `test/test_openclaw_cuda_graphs.py` coverage context.

### Findings and fixes
- Added focused AST-isolated API contract coverage for this runtime metadata surface in `tests/test_api_server_control_contract.py`. The tests pin delegation and public fallback behavior for CUDA graph status/toggle endpoints, generation diagnostics empty fallback, `OPENCLAW_SDPA_BACKEND` and `OPENCLAW_CUDA_GRAPHS` runtime-default application, `_env_flag()` parsing, and the `/sdapi/v1/memory` RAM/CUDA response dictionary shape.
- No production dead code or safe duplicate stats helper was removed in this slice. `get_memory()` has compact inline RAM/CUDA shaping, but no second equivalent API helper was found in the checked scope; extracting serializers would be cosmetic and could change exception boundaries or public error fields.
- Preserved `MemoryResponse.ram` and `MemoryResponse.cuda` as untyped dictionaries. The current endpoint intentionally returns nested legacy field groups and error dictionaries; narrowing or restructuring the schema would be public API churn.
- Preserved `get_cuda_graphs()` and `set_cuda_graphs()` as thin route wrappers. They bind a public OpenClaw route pair and preserve side effects on the CUDA graph runtime cache/stat state; merging them into diagnostics or defaults would obscure route behavior without eliminating duplicate logic.
- Preserved `get_openclaw_generation_diagnostics()` returning `{}` when no generation has been captured. That empty-object fallback is a compatibility behavior distinct from the internal `None` sentinel returned by `last_generation_diagnostics()`.
- Preserved generation diagnostics CUDA graph summarization helpers. They intentionally hide raw graph keys behind a hash/shape summary, preserve counter deltas and bypass reasons, and avoid leaking large runtime key details through the diagnostics endpoint.

### Static/dynamic audit map notes
- Memory chain remains: `/sdapi/v1/memory` -> `Api.get_memory()` -> process RSS/percent RAM estimate and `torch.cuda.mem_get_info()`/`torch.cuda.memory_stats(shared.device)` -> `models.MemoryResponse(ram=..., cuda=...)`.
- CUDA graph status chain remains: `/sdapi/v1/openclaw/cuda-graphs` GET -> `openclaw_cuda_graphs.status()`; POST -> `openclaw_cuda_graphs.set_enabled(enabled, clear=clear)` with dict-shaped request coercion.
- Runtime default chain remains: `Api.__init__()` -> `apply_openclaw_runtime_defaults()` -> `OPENCLAW_SDPA_BACKEND` into `sd_hijack_optimizations.set_sdpa_backend()` and optional `OPENCLAW_CUDA_GRAPHS` into `openclaw_cuda_graphs.set_enabled(..., clear=True)`, with error reporting instead of startup failure.
- Generation diagnostics chain remains: processing sample hooks -> `openclaw_generation_diagnostics.before_sample()`/`after_sample()` -> `last_generation_diagnostics()` deep copy -> `/sdapi/v1/openclaw/generation-diagnostics`, with CUDA graph status summarized before/after and deltaed.
- Compatibility surfaces to keep conservative: route paths and methods, MemoryResponse field names, RAM/CUDA nested keys, error dictionary fallback shape, CUDA graph status keys/counter meanings, POST request coercion for `enabled` and `clear`, environment flag accepted values, startup error reporting, diagnostics empty fallback, hashed graph-key summary, and process/runtime side effects.

### Validation log
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pycompile-pass82.XXXXXX) python3 -m py_compile modules/api/api.py modules/api/models.py modules/openclaw_generation_diagnostics.py modules/openclaw_cuda_graphs.py tests/test_api_server_control_contract.py test/test_openclaw_cuda_graphs.py` - passed.
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pytest-pass82b.XXXXXX) python3 -m pytest -q tests/test_api_server_control_contract.py` - passed: 7 passed, with the existing pytest config warning `Unknown config option: base_url`.
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pytest-pass82.XXXXXX) python3 -m pytest -q tests/test_api_server_control_contract.py test/test_openclaw_cuda_graphs.py` - blocked during collection of `test/test_openclaw_cuda_graphs.py`: `ModuleNotFoundError: No module named torch` in the system Python environment. The API contract half of that command collected and passed separately as noted above.
- Exact AST duplicate-body scan across `modules/api/api.py`, `modules/api/models.py`, `modules/openclaw_generation_diagnostics.py`, `modules/openclaw_cuda_graphs.py`, `tests/test_api_server_control_contract.py`, and `test/test_openclaw_cuda_graphs.py` found no duplicate function bodies.
- `git diff --check` - passed.
- Live `/sdapi/v1/memory`, `/sdapi/v1/openclaw/cuda-graphs`, and `/sdapi/v1/openclaw/generation-diagnostics` requests were not exercised because this bounded slice did not start a WebUI/model runtime.

### Next unchecked scope
- More slices are still needed. Recommended next slice: continue into API training/create response helpers adjacent to the same class region, especially `_create_response()`, `_run_training_task()`, `create_embedding()`, `create_hypernetwork()`, `train_embedding()`, `train_hypernetwork()`, and `tests/test_api_training_contract.py`, looking for stale wrapper names or duplicate task response shaping while preserving public response bodies and training side effects.

## Pass 83 - API training/create response helper names and task response shaping (2026-06-20)

### Checked scope
- `modules/api/api.py`: route registrations for create/train embedding and hypernetwork endpoints; `_create_response()`, `_train_response()`, `_run_create_task()`, `_run_training_task()`, `create_embedding()`, `create_hypernetwork()`, `_prepare_hypernetwork_training()`, `_restore_hypernetwork_training_devices()`, `train_embedding()`, and `train_hypernetwork()`.
- `modules/api/models.py`: `CreateResponse` and `TrainResponse` response schemas.
- `tests/test_api_training_contract.py`: source-level contract coverage for create/train helper selection, hypernetwork response strings, and single training state-end lifecycle.
- Adjacent checked references: API route response-model fanout, focused training/create response strings, and related hypernetwork/textual-inversion create/train tests from prior passes.

### Findings and fixes
- No production dead code or safe duplicate-response consolidation was found in this bounded slice.
- Preserved `_create_response()` and `_train_response()` as intentionally separate schema constructors. They are tiny wrappers, but their names make create/train response-model boundaries explicit and are pinned by focused contract coverage; a generic `info` response helper would be cosmetic and would weaken the source-level guard against response-class mixups.
- Preserved `_run_create_task()` and `_run_training_task()` as separate lifecycle helpers. The create helper owns begin/end state, creator invocation, optional post-create refresh, and `AssertionError` create-string handling; the training helper additionally owns before/after training hooks, xattention optimization toggling, broad training exception capture, and the historical `error: {error}` success-path suffix.
- Preserved `create_embedding()` post-create embedding reload, hypernetwork pre/post training hooks, and exact public response prefixes. No stale wrapper name or dead helper parameter was identified.
- Left `tests/test_api_training_contract.py` unchanged. Its helper-name tests are still useful for this public API seam and do not duplicate lower-level runtime training tests.

### Static/dynamic audit map notes
- Create embedding chain remains: `/sdapi/v1/create/embedding` -> `Api.create_embedding()` -> `_run_create_task("create_embedding", textual_inversion.create_embedding, ..., after_create=embedding_db.load_textual_inversion_embeddings)` -> `CreateResponse(info="create embedding filename: ..." | "create embedding error: ...")`.
- Create hypernetwork chain remains: `/sdapi/v1/create/hypernetwork` -> `Api.create_hypernetwork()` -> `_run_create_task("create_hypernetwork", hypernetwork.create_hypernetwork, ...)` -> `CreateResponse(info="create hypernetwork filename: ..." | "create hypernetwork error: ...")`.
- Training chains remain: `/sdapi/v1/train/embedding` and `/sdapi/v1/train/hypernetwork` -> `_run_training_task(...)` -> optional optimization undo/apply and model-specific hooks -> `TrainResponse(info="train ... complete: filename: ... error: ..." | "train ... error: ...")` with one `shared.state.end()` in the helper.
- Compatibility surfaces to keep conservative: route paths, response model class names, `info` field name, exact response prefixes/suffixes, create `AssertionError` handling, inclusion of `error: None` on successful training, job names passed to `shared.state.begin()`, post-create embedding reload, hypernetwork loaded-list clearing, device restoration, and `training_xattention_optimizations` gating.

### Validation log
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pycompile-pass83.XXXXXX) python3 -m py_compile modules/api/api.py modules/api/models.py tests/test_api_training_contract.py tests/test_hypernetwork_creation_contract.py tests/test_textual_inversion_preview_save_contract.py` - passed.
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pytest-pass83.XXXXXX) python3 -m pytest -q tests/test_api_training_contract.py tests/test_hypernetwork_creation_contract.py tests/test_textual_inversion_preview_save_contract.py` - passed: 6 passed, with the existing pytest config warning `Unknown config option: base_url`.
- Exact AST duplicate-body scan across `modules/api/api.py`, `modules/api/models.py`, and `tests/test_api_training_contract.py` found only the pre-existing adjacent `TextToImageResponse`/`ImageToImageResponse` response-model twins in `modules/api/models.py`, outside this create/train response-helper target.
- `git diff --check` - passed.
- Live create/train embedding and hypernetwork API requests were not exercised because they require a running WebUI/model runtime, training assets, and provisioned ML dependencies.

### Next unchecked scope
- More slices are still needed. Recommended next slice: continue into remaining API response-model twins and generated model surfaces near `modules/api/models.py`, especially `TextToImageResponse`/`ImageToImageResponse`, extras response models, and adjacent route response construction, looking for safe schema-helper consolidation while preserving public response fields and OpenAPI schemas.

## Pass 84 - API generation and extras response-model twins (2026-06-20)

### Checked scope
- `modules/api/models.py`: `TextToImageResponse`, `ImageToImageResponse`, `ExtrasBaseRequest`, `ExtraBaseResponse`, `ExtrasSingleImageRequest`, `ExtrasSingleImageResponse`, `FileData`, `ExtrasBatchImagesRequest`, and `ExtrasBatchImagesResponse`.
- `modules/api/api.py`: route registrations for `/sdapi/v1/txt2img`, `/sdapi/v1/img2img`, `/sdapi/v1/extra-single-image`, and `/sdapi/v1/extra-batch-images`; response construction in `text2imgapi()`, `img2imgapi()`, `_run_extras()`, `extras_single_image_api()`, and `extras_batch_images_api()`.
- Focused adjacent tests/references: `tests/test_api_server_control_contract.py` and `test/test_postprocessing_api_defaults.py` references to API response models and extras route construction.

### Findings and fixes
- Consolidated the pre-existing adjacent `TextToImageResponse`/`ImageToImageResponse` field duplication behind a private `_ImageGenerationResponse` base class. The public response class names, route `response_model` bindings, field names, field metadata, and response constructor call sites are preserved.
- Preserved distinct `TextToImageResponse` and `ImageToImageResponse` classes instead of aliasing or merging them. They are public FastAPI response-model identities and should continue to produce distinct OpenAPI component names and route contracts.
- No safe extras response-model deletion or consolidation was found. `ExtrasSingleImageResponse` and `ExtrasBatchImagesResponse` already share `ExtraBaseResponse.html_info`, and their payload fields differ intentionally (`image: Optional[str]` versus `images: list[str]`).
- No safe route response-construction helper was extracted. `text2imgapi()` and `img2imgapi()` differ in processing setup, init image/mask handling, response timing metadata, and init-image redaction; extras single/batch differ in input decoding and response payload shape. Additional helpers would be cosmetic or risk changing public API behavior.
- Generated request model surfaces (`StableDiffusionTxt2ImgProcessingAPI` and `StableDiffusionImg2ImgProcessingAPI`) were inspected near this scope and left unchanged; their overlap mirrors distinct processing classes and public request schemas.

### Static/dynamic audit map notes
- Txt2img response chain remains: `/sdapi/v1/txt2img` -> `Api.text2imgapi()` -> `TextToImageResponse(images=b64images, parameters=vars(txt2imgreq), info=processed_js_with_image_paths(processed))`.
- Img2img response chain remains: `/sdapi/v1/img2img` -> `Api.img2imgapi()` -> optional init-image/mask redaction and OpenClaw timing metadata -> `ImageToImageResponse(images=b64images, parameters=vars(img2imgreq), info=processed_js_with_image_paths(...))`.
- Extras single response chain remains: `/sdapi/v1/extra-single-image` -> decode one image -> `_run_extras(extras_mode=0, ...)` -> `ExtrasSingleImageResponse(image=..., html_info=...)`.
- Extras batch response chain remains: `/sdapi/v1/extra-batch-images` -> decode `imageList` folder -> `_run_extras(extras_mode=1, ...)` -> `ExtrasBatchImagesResponse(images=..., html_info=...)`.
- Compatibility surfaces to keep conservative: route paths/methods, response model class names, response field names and metadata, OpenAPI component identity, `parameters=vars(...)` behavior, img2img init-image redaction when `include_init_images` is false, OpenClaw timing metadata in img2img `info`, and extras single/batch payload cardinality.

### Validation log
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pycompile-pass84.XXXXXX) python3 -m py_compile modules/api/models.py modules/api/api.py` - passed.
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pytest-pass84-api.XXXXXX) python3 -m pytest -q tests/test_api_server_control_contract.py` - passed: 7 passed, with the existing pytest config warning `Unknown config option: base_url`.
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pytest-pass84.XXXXXX) python3 -m pytest -q tests/test_api_server_control_contract.py test/test_postprocessing_api_defaults.py` - partially blocked by missing runtime dependencies in the system Python environment: API control tests passed, then `test/test_postprocessing_api_defaults.py::test_postprocessing_runner_order_override_preserves_script_defaults` failed on `ModuleNotFoundError: No module named fastapi`; extension preload stderr also reported missing `torch`.
- Exact AST duplicate class/function scan across `modules/api/models.py` and `modules/api/api.py` now reports only intentional empty subclass bodies for `TextToImageResponse`/`ImageToImageResponse` plus the pre-existing `ScriptArgsList` shape match. The substantive duplicate field definitions were removed while retaining distinct response classes.
- `git diff --check` - passed.
- Live txt2img/img2img/extras API requests and full OpenAPI schema generation were not exercised because this bounded slice did not start a WebUI/model runtime and the system Python environment lacks FastAPI/Pydantic/Torch dependencies.

### Next unchecked scope
- More slices are still needed. Recommended next slice: continue down `modules/api/models.py` after the extras response models into `PNGInfoRequest`/`PNGInfoResponse`, `ProgressRequest`/`ProgressResponse`, `InterrogateRequest`/`InterrogateResponse`, and adjacent API handlers (`pnginfoapi()`, `progressapi()`, `interrogateapi()`), looking for stale response fields, duplicate request/response field declarations, or safe helper consolidation while preserving public OpenAPI schemas and runtime side effects.

## Pass 85 - API PNG info/progress/interrogate request and response models (2026-06-20)

### Checked scope
- `modules/api/models.py`: `PNGInfoRequest`, `PNGInfoResponse`, `ProgressRequest`, `ProgressResponse`, `InterrogateRequest`, `InterrogateResponse`, and adjacent single-image request model overlap with `ExtrasSingleImageRequest`.
- `modules/api/api.py`: route registration and handlers for `/sdapi/v1/png-info`, `/sdapi/v1/progress`, and `/sdapi/v1/interrogate`, including `pnginfoapi()`, `progressapi()`, and `interrogateapi()`.
- Focused adjacent tests: `test/test_api_info_interrogate_admin_contract.py`, `tests/test_api_progress_contract.py`, and the progress/extras source-level coverage in `test/test_postprocessing_api_defaults.py`.

### Findings and fixes
- Consolidated the duplicated base64 `image` request field used by `ExtrasSingleImageRequest` and `InterrogateRequest` into a private `_Base64ImageRequest` base model. Public request class names are preserved, the `image` field default/title/description are preserved, and `InterrogateRequest.model` remains the only interrogate-specific field.
- Added focused source-contract coverage in `test/test_api_info_interrogate_admin_contract.py` to pin the shared base64 image field and both public request model inheritance points.
- Preserved `PNGInfoRequest.image` as a separate required field. It has a different public contract from the optional/default-empty base64 image request used by extras/interrogate.
- Preserved `PNGInfoResponse.info/items/parameters` as the public response shape. The empty-image fallback in `pnginfoapi()` is conservative legacy code; `decode_base64_to_image()` normally raises `HTTPException` for invalid input, and broadening/removing response fields would be public API churn.
- Preserved `ProgressRequest.skip_current_image` and every `ProgressResponse` field, including `current_task`. These fields directly control current-image serialization and expose live task/progress state; no stale response field was proven dead.
- Preserved `interrogateapi()` model dispatch and queue-lock helper usage. The handler still supports `clip` and `deepdanbooru` paths and raises the existing 404 for unknown models.

### Static/dynamic audit map notes
- PNG info chain remains: `/sdapi/v1/png-info` -> `Api.pnginfoapi()` -> `decode_base64_to_image(req.image.strip())` -> `images.read_info_from_image()` -> `infotext_utils.parse_generation_parameters()` -> `script_callbacks.infotext_pasted_callback()` -> `PNGInfoResponse(info=..., items=..., parameters=...)`.
- Progress chain remains: `/sdapi/v1/progress` GET query params -> `ProgressRequest(skip_current_image=False)` -> `progress_module.calculate_progress_and_eta()` -> optional `shared.state.set_current_image()` and base64 current image encoding -> `ProgressResponse(progress, eta_relative, state, current_image, textinfo, current_task)`.
- Interrogate chain remains: `/sdapi/v1/interrogate` -> `InterrogateRequest(image, model="clip")` -> decode and RGB conversion -> queued `shared.interrogator.interrogate()` or `deepbooru.model.tag()` -> `InterrogateResponse(caption=...)`.
- Compatibility surfaces to keep conservative: route paths/methods, request class names, inherited `image` field metadata/defaults, PNG info required image contract, progress query default and skip-current-image behavior, progress state/current_task/textinfo fields, queue locking, CLIP/deepdanbooru dispatch, and image decode/interrogate side effects.

### Validation log
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pycompile-pass85.XXXXXX) python3 -m py_compile modules/api/models.py modules/api/api.py test/test_api_info_interrogate_admin_contract.py tests/test_api_progress_contract.py test/test_postprocessing_api_defaults.py` - passed.
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pytest-pass85.XXXXXX) python3 -m pytest -q test/test_api_info_interrogate_admin_contract.py tests/test_api_progress_contract.py` - passed: 4 passed, with the existing pytest config warning `Unknown config option: base_url`.
- Runtime Pydantic schema introspection was not available in the system Python environment because importing `modules.api.models` failed with `ModuleNotFoundError: No module named pydantic`; validation used py_compile and source-level contract tests instead.
- Exact AST duplicate class/function scan across `modules/api/models.py`, `modules/api/api.py`, `test/test_api_info_interrogate_admin_contract.py`, and `tests/test_api_progress_contract.py` now reports only intentional empty public wrapper classes (`TextToImageResponse`, `ImageToImageResponse`, `ExtrasSingleImageRequest`) plus the pre-existing `ScriptArgsList` shape match.
- `git diff --check` - passed.
- Live `/sdapi/v1/png-info`, `/sdapi/v1/progress`, and `/sdapi/v1/interrogate` requests were not exercised because this bounded slice did not start a WebUI/model runtime.

### Next unchecked scope
- More slices are still needed. Recommended next slice: continue down `modules/api/models.py` after `TrainResponse`/`CreateResponse` into flags/options/sampler/upscaler/model-list and embedding response models plus adjacent API handlers (`get_config()`, `set_config()`, `get_cmd_flags()`, list endpoints, `get_embeddings()`), looking for stale public fields, duplicate list-item serializers, or safe helper consolidation while preserving OpenAPI schemas and runtime refresh/config side effects.

## Pass 86 - API flags/options/list models and embeddings metadata (2026-06-20)

### Checked scope
- `modules/api/models.py`: dynamic `OptionsModel` generation, `FlagsModel` generation, `SamplerItem`, `SchedulerItem`, `UpscalerItem`, `LatentUpscalerModeItem`, `SDModelItem`, `SDVaeItem`, `HypernetworkItem`, `EmbeddingItem`, and `EmbeddingsResponse`.
- `modules/api/api.py`: route registration and handlers for `/sdapi/v1/options`, `/sdapi/v1/cmd-flags`, `/sdapi/v1/samplers`, `/sdapi/v1/schedulers`, `/sdapi/v1/upscalers`, `/sdapi/v1/latent-upscale-modes`, `/sdapi/v1/sd-models`, `/sdapi/v1/sd-vae`, `/sdapi/v1/hypernetworks`, `/sdapi/v1/embeddings`, and adjacent refresh handlers.
- `modules/shared_items.py` and `modules/shared.py`: sampler and shared metadata list helper references relevant to API response construction.
- Focused adjacent tests: `tests/test_api_listing_contract.py` and `tests/test_api_server_control_contract.py` route binding stubs.

### Findings and fixes
- Removed an unreachable fallback branch in dynamic `OptionsModel` generation. The loop iterates `opts.data_labels.items()`, and `Options`/UI callers treat every label entry as an `OptionInfo`-like object with `default` and `label`; the `metadata is None` fallback could not produce a meaningful public option schema and duplicated the same field update path with less metadata.
- Added focused AST-isolated contract coverage in `tests/test_api_listing_contract.py` for config/flags sources, sampler/scheduler/upscaler/model/VAE/hypernetwork list shapes, and `EmbeddingsResponse` loaded/skipped map shaping.
- Preserved public list response models and field names. `SamplerItem`, `SchedulerItem`, `UpscalerItem`, `SDModelItem`, `SDVaeItem`, `HypernetworkItem`, `EmbeddingItem`, and `EmbeddingsResponse` each map to distinct public endpoint payloads; merging them would collapse OpenAPI component identities or change route-specific schemas.
- Preserved dynamic `FlagsModel` generation from parser option actions. The apparent repeated `dest` writes are argparse alias behavior and should remain compatible with public command-flag schema generation.
- Preserved `get_config()` runtime shaping, including keys present in `shared.opts.data` but missing from `data_labels`, because it reflects live config state and migration/plugin dynamics. `set_config()` and refresh handlers were left unchanged to preserve validation and queue-lock side effects.
- No safe shared serializer was extracted for metadata list endpoints. Each constructor is short, route-local, and binds different backing objects/field names; a generic helper would be mostly cosmetic and could obscure public schema compatibility.

### Static/dynamic audit map notes
- Options chain remains: `opts.data_labels` at import time -> `OptionsModel = create_model("Options", ...)`; runtime `/sdapi/v1/options` GET returns values from `shared.opts.data` with metadata defaults only for known labels, and POST uses `shared.opts.set(..., is_api=True)` plus config save.
- Flags chain remains: argparse parser option actions -> `FlagsModel = create_model("Flags", ...)`; runtime `/sdapi/v1/cmd-flags` returns `vars(shared.cmd_opts)`.
- List endpoints remain direct metadata projections from `sd_samplers.all_samplers`, `sd_schedulers.schedulers`, `shared.sd_upscalers`, `shared.latent_upscale_modes`, `sd_models.checkpoints_list`, `sd_vae.vae_dict`, and `shared.hypernetworks`.
- Embeddings chain remains: `/sdapi/v1/embeddings` -> `sd_hijack.model_hijack.embedding_db` -> `loaded` and `skipped` dictionaries keyed by embedding name with `step`, `sd_checkpoint`, `sd_checkpoint_name`, `shape`, and `vectors`.
- Compatibility surfaces to keep conservative: route paths/methods, dynamic `OptionsModel`/`FlagsModel` component names, option and flag field names, live `get_config()` key behavior, argparse alias handling, all list item field names, embeddings loaded/skipped map names, and refresh route side effects.

### Validation log
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pycompile-pass86.XXXXXX) python3 -m py_compile modules/api/models.py modules/api/api.py tests/test_api_listing_contract.py` - passed.
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pytest-pass86.XXXXXX) python3 -m pytest -q tests/test_api_listing_contract.py` - passed: 6 passed, with the existing pytest config warning `Unknown config option: base_url`.
- Exact AST duplicate class/function scan across `modules/api/models.py`, `modules/api/api.py`, and `tests/test_api_listing_contract.py` reports only intentional empty public wrapper classes from earlier passes: `TextToImageResponse`, `ImageToImageResponse`, and `ExtrasSingleImageRequest`.
- `git diff --check` - passed.
- Live `/sdapi/v1/options`, `/sdapi/v1/cmd-flags`, metadata list, and embeddings requests were not exercised because this bounded slice did not start a WebUI/model runtime.

### Next unchecked scope
- More slices are still needed. Recommended next slice: continue immediately after embeddings into refresh endpoints and the next API metadata families in `modules/api/api.py`/`modules/api/models.py`, especially face restorers, RealESRGAN models, prompt styles, script list/script info, extension item listing, and their existing listing contract tests, looking for stale fields, duplicate serializers, or safe helper consolidation while preserving public OpenAPI schemas and refresh/list side effects.

## Pass 87 - API refresh and adjacent metadata listing families (2026-06-20)

### Scope checked
- `modules/api/api.py`: route registration and handlers for `/sdapi/v1/face-restorers`, `/sdapi/v1/realesrgan-models`, `/sdapi/v1/prompt-styles`, `/sdapi/v1/embeddings`, `/sdapi/v1/refresh-embeddings`, `/sdapi/v1/refresh-checkpoints`, `/sdapi/v1/refresh-vae`, `/sdapi/v1/scripts`, `/sdapi/v1/script-info`, and `/sdapi/v1/extensions`; focused inspection of `get_face_restorers()`, `get_realesrgan_models()`, `get_prompt_styles()`, `get_embeddings()`, refresh handlers, `_script_lists()`, `get_scripts_list()`, `get_script_info()`, and `get_extensions_list()`.
- `modules/api/models.py`: response models for `FaceRestorerItem`, `RealesrganItem`, `PromptStyleItem`, `EmbeddingItem`, `EmbeddingsResponse`, `ScriptsList`, `ScriptArg`, `ScriptInfo`, and `ExtensionItem`.
- Adjacent contract tests: `tests/test_api_listing_contract.py`, `tests/test_api_server_control_contract.py`, `tests/test_api_extension_item_contract.py`, and `tests/test_extensions_metadata_contract.py`.

### Findings / fixes
- Added focused contract coverage for face-restorer, RealESRGAN, and prompt-style listing shapes in `tests/test_api_listing_contract.py`. The new AST-isolated test pins nullable `cmd_dir`, nullable RealESRGAN `path`, scale preservation, prompt-style ordering from the style registry, and nullable `negative_prompt`.
- Preserved the current face-restorer, RealESRGAN, prompt-style, embedding, script-list, script-info, and extension-list serializers. Each endpoint projects a different live registry and public item shape, so a generic serializer/helper would add abstraction without removing actionable duplication.
- Preserved `get_realesrgan_models(None)` even though the argument is unused by the list factory for current static model data; the imported helper's signature accepts an upscaler instance for runtime construction, and changing the API call shape would not remove dead code.
- Preserved refresh handlers as small queue-locked side-effect wrappers. They intentionally dispatch to separate registries: textual inversion embeddings, checkpoint discovery, and VAE discovery.
- Preserved script and extension listing behavior already covered in pass 81: shared script source lookup through `_script_lists()`, filtering of `None` names/api-info records, extension registry refresh, per-extension git metadata refresh, remote-only filtering, and non-exposure of extension filesystem paths.

### Static/dynamic audit map notes
- Face-restorer chain remains: `/sdapi/v1/face-restorers` -> `shared.face_restorers` -> per-restorer `name()` plus optional `cmd_dir`.
- RealESRGAN chain remains: `/sdapi/v1/realesrgan-models` -> `modules.realesrgan_model.get_realesrgan_models(None)` -> public `name`, `path` from `data_path`, and `scale`.
- Prompt-style chain remains: `/sdapi/v1/prompt-styles` -> `shared.prompt_styles.styles` insertion order -> tuple fields `name`, `prompt`, and `negative_prompt`.
- Refresh chain remains: `/sdapi/v1/refresh-embeddings` -> queued `embedding_db.load_textual_inversion_embeddings(force_reload=True)`; `/sdapi/v1/refresh-checkpoints` -> queued `shared.refresh_checkpoints`; `/sdapi/v1/refresh-vae` -> queued `shared_items.refresh_vae_list`.
- Compatibility surfaces to keep conservative: route paths/methods, response field names, nullable metadata fields, registry/list ordering, script and extension filtering semantics, OpenAPI component names, and refresh endpoint side effects.

### Validation log
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pycompile-pass87.XXXXXX) python3 -m py_compile modules/api/models.py modules/api/api.py tests/test_api_listing_contract.py` - passed.
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pytest-pass87.XXXXXX) python3 -m pytest -q tests/test_api_listing_contract.py` - passed: 7 passed, with the existing pytest config warning `Unknown config option: base_url`.
- Exact AST duplicate class/function scan across `modules/api/models.py`, `modules/api/api.py`, and `tests/test_api_listing_contract.py` reports only intentional empty wrapper/marker classes: `TextToImageResponse`, `ImageToImageResponse`, `ExtrasSingleImageRequest`, `ScriptArgsList`, and the test-local `Extension` marker used by the extension listing contract.
- `git diff --check` - passed.
- Live `/sdapi/v1/face-restorers`, `/sdapi/v1/realesrgan-models`, `/sdapi/v1/prompt-styles`, refresh, scripts, script-info, and extensions requests were not exercised because this bounded slice did not start a WebUI/model runtime.

### Next unchecked scope
- More slices are still needed. Recommended next slice: continue past the static/listing API metadata family into the create/train and memory endpoint models/handlers in `modules/api/api.py`/`modules/api/models.py`, especially `CreateResponse`, `TrainResponse`, `_run_create_task()`, `_run_training_task()`, `create_embedding()`, `create_hypernetwork()`, `train_embedding()`, `train_hypernetwork()`, `_prepare_hypernetwork_training()`, `_restore_hypernetwork_training_devices()`, and `get_memory()`, looking for stale response helpers, duplicate task wrappers, or unreachable training/memory branches while preserving task state, refresh side effects, and public response strings.

## Pass 88 - API create/train task helpers and memory response surface (2026-06-20)

### Scope checked
- `modules/api/api.py`: `_create_response()`, `_train_response()`, `_run_create_task()`, `_run_training_task()`, `create_embedding()`, `create_hypernetwork()`, `_prepare_hypernetwork_training()`, `_restore_hypernetwork_training_devices()`, `train_embedding()`, `train_hypernetwork()`, and `get_memory()`.
- `modules/api/models.py`: `CreateResponse`, `TrainResponse`, and `MemoryResponse` public schemas.
- Focused adjacent tests: `tests/test_api_training_contract.py` and `tests/test_api_server_control_contract.py` memory/task-response coverage.

### Findings / fixes
- No safe production-code deletion or consolidation was found in this slice. The create/train helpers already represent the useful deduplication boundary: one lifecycle wrapper for create endpoints, one lifecycle wrapper for training endpoints, and separate response constructors to preserve public schema-family intent.
- Preserved `_create_response()` and `_train_response()` despite identical implementation shape. `CreateResponse` and `TrainResponse` are distinct public FastAPI response models with different endpoint families and generated schema names; collapsing them into a generic helper would be cosmetic and would weaken contract readability.
- Preserved `_run_create_task()` and `_run_training_task()` as separate wrappers. Create tasks have post-create refresh side effects and assertion-error handling; training tasks have optimization undo/apply, train-function exception capture, optional before/after hooks, and one shared state end path. A merged wrapper would add branching around live task semantics without removing dead code.
- Preserved hypernetwork training hooks. `_prepare_hypernetwork_training()` intentionally clears loaded hypernetworks before training, and `_restore_hypernetwork_training_devices()` restores model submodules to `devices.device` after training regardless of training success inside the helper.
- Preserved `get_memory()` and `MemoryResponse` loose dictionary fields. The endpoint intentionally tolerates RAM/CUDA probe failures with `error` dictionaries and exposes runtime-dependent nested CUDA stats; stricter models or branch removal would risk public API compatibility.

### Static/dynamic audit map notes
- Create embedding chain remains: `/sdapi/v1/create/embedding` -> `Api.create_embedding()` -> `_run_create_task("create_embedding", textual_inversion.create_embedding, ..., after_create=embedding_db.load_textual_inversion_embeddings)` -> `CreateResponse(info="create embedding filename: ..." | "create embedding error: ...")` -> `shared.state.end()`.
- Create hypernetwork chain remains: `/sdapi/v1/create/hypernetwork` -> `Api.create_hypernetwork()` -> `_run_create_task("create_hypernetwork", hypernetwork.create_hypernetwork, ...)` -> `CreateResponse(info="create hypernetwork filename: ..." | "create hypernetwork error: ...")`.
- Training chains remain: `/sdapi/v1/train/embedding` and `/sdapi/v1/train/hypernetwork` -> `_run_training_task(...)` -> optional optimization undo/apply and optional hypernetwork prepare/restore hooks -> `TrainResponse(info="train ... complete: filename: ... error: ..." | "train ... error: ...")` -> `shared.state.end()`.
- Memory chain remains: `/sdapi/v1/memory` -> process RSS/percent RAM calculation plus optional `torch.cuda.mem_get_info()` and `torch.cuda.memory_stats(shared.device)` -> `MemoryResponse(ram=..., cuda=...)`, with error dictionaries for failed probes.
- Compatibility surfaces to keep conservative: route paths/methods, `CreateResponse.info`, `TrainResponse.info`, `MemoryResponse.ram`/`cuda`, exact public response strings, embedding refresh after create, training optimization restore, hypernetwork device restore, and memory fallback error shapes.

### Validation log
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pycompile-pass88.XXXXXX) python3 -m py_compile modules/api/api.py modules/api/models.py tests/test_api_training_contract.py tests/test_api_server_control_contract.py` - passed.
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pytest-pass88-training.XXXXXX) python3 -m pytest -q tests/test_api_training_contract.py tests/test_api_server_control_contract.py -k "training or memory"` - passed: 4 passed, 6 deselected, with the existing pytest config warning `Unknown config option: base_url`.
- Exact AST duplicate class/function scan across `modules/api/api.py`, `modules/api/models.py`, `tests/test_api_training_contract.py`, and `tests/test_api_server_control_contract.py` reports only intentional empty wrapper/marker classes from earlier passes: `ScriptArgsList`, `TextToImageResponse`, `ImageToImageResponse`, and `ExtrasSingleImageRequest`.
- `git diff --check` - passed.
- Live create/train/memory API requests were not exercised because this bounded slice did not start a WebUI/model runtime or perform training operations.

### Next unchecked scope
- More slices are still needed. Recommended next slice: continue after memory into extension/script-control and remaining API utility response surfaces, especially `get_extensions_list()`, script-info/list coverage boundaries not already covered by pass 81/87, launch-info/help-style responses if present, and any trailing `modules/api/models.py` response classes after `MemoryResponse`, looking for stale public fields or duplicated serializers while preserving OpenAPI schema identities and registry refresh side effects.

## Pass 89 - API script/extension utility response tail (2026-06-20)

### Scope checked
- `modules/api/models.py`: trailing response models after `MemoryResponse`: `ScriptsList`, `ScriptArg`, `ScriptInfo`, and `ExtensionItem`.
- `modules/api/api.py`: route registrations for `/sdapi/v1/scripts`, `/sdapi/v1/script-info`, and `/sdapi/v1/extensions`; `_script_lists()`, `get_scripts_list()`, `get_script_info()`, and `get_extensions_list()`.
- Adjacent route/control boundaries around `/sdapi/v1/memory`, unload/reload checkpoint routes, server stop route registration, and `Api.launch()`; searched for launch-info/help-style API response surfaces and found no separate launch-info/help-style response classes or routes in this fork.
- Focused contract tests: `tests/test_api_listing_contract.py` script-list/script-info and extension-list coverage, plus `tests/test_api_server_control_contract.py` route registration coverage.

### Findings / fixes
- No safe production-code deletion or consolidation was found in this slice. The remaining post-memory model classes are all live public schema/registry surfaces: `ScriptsList` is the `/sdapi/v1/scripts` response model, `ScriptArg` is nested in `ScriptInfo`, `ScriptInfo` is produced by the script registry for `/sdapi/v1/script-info`, and `ExtensionItem` is the `/sdapi/v1/extensions` response model.
- Preserved `_script_lists()` as the small shared boundary between `/sdapi/v1/scripts` and `/sdapi/v1/script-info`. It is already the useful deduplication point and keeps both endpoints tied to the same txt2img/img2img runner sources.
- Preserved separate `get_scripts_list()` and `get_script_info()` shaping. The first filters `None` script names and returns a `ScriptsList` model; the second preserves script registry `api_info` objects and filters `None` API records. Combining them would add branching around two public endpoint shapes without removing dead code.
- Preserved `get_extensions_list()` as an explicit route-local serializer. Its `extensions.list_extensions()` refresh, per-extension `read_info_from_repo()` calls, remote-only filtering, and omission of filesystem paths are observable compatibility and privacy boundaries already pinned by tests.
- Preserved route response model identities and field names for `ScriptsList`, `ScriptArg`, `ScriptInfo`, and `ExtensionItem`; no stale public fields had strong enough evidence for removal.

### Static/dynamic audit map notes
- Script list chain remains: `/sdapi/v1/scripts` -> `_script_lists()` -> `scripts.scripts_txt2img.scripts` and `scripts.scripts_img2img.scripts` -> filter non-`None` `script.name` -> `ScriptsList(txt2img=..., img2img=...)`.
- Script info chain remains: `/sdapi/v1/script-info` -> `_script_lists()` -> preserve non-`None` `script.api_info` records generated in `modules/scripts.py` as `models.ScriptInfo` with nested `ScriptArg` controls.
- Extension list chain remains: `/sdapi/v1/extensions` -> `extensions.list_extensions()` -> iterate `extensions.extensions` -> `ext.read_info_from_repo()` for every extension -> include only remote-backed extensions with `name`, `remote`, `branch`, `commit_hash`, `commit_date`, `version`, and `enabled`.
- Launch/help-style check: this fork exposes dynamic command flags through `FlagsModel` and `/sdapi/v1/cmd-flags`; no distinct launch-info/help-style response model exists after `MemoryResponse` or in adjacent API routes.
- Compatibility surfaces to keep conservative: route paths/methods, OpenAPI component names, script ordering, `None` filtering semantics, registry side effects, extension remote-only filtering, per-extension git metadata refresh, and non-exposure of extension filesystem paths.

### Validation log
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pycompile-pass89.XXXXXX) python3 -m py_compile modules/api/api.py modules/api/models.py tests/test_api_listing_contract.py tests/test_api_server_control_contract.py` - passed.
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pytest-pass89.XXXXXX) python3 -m pytest -q tests/test_api_listing_contract.py tests/test_api_server_control_contract.py -k "script or extension or route"` - passed: 3 passed, 11 deselected, with the existing pytest config warning `Unknown config option: base_url`.
- Exact AST duplicate class/function scan across `modules/api/models.py`, `modules/api/api.py`, `tests/test_api_listing_contract.py`, and `tests/test_api_server_control_contract.py` reports only intentional empty wrapper/marker classes from earlier passes: `TextToImageResponse`, `ImageToImageResponse`, `ExtrasSingleImageRequest`, `ScriptArgsList`, and the test-local `Extension` marker used by the extension listing contract.
- `git diff --check` - passed.
- Live `/sdapi/v1/scripts`, `/sdapi/v1/script-info`, and `/sdapi/v1/extensions` requests were not exercised because this bounded slice did not start a WebUI/model runtime or refresh live extension repositories outside the unit contract.

### Next unchecked scope
- More slices are still needed. Recommended next slice: continue past the API utility tail into adjacent server-control/runtime utility handlers after `Api.launch()`, especially `kill_webui()`, `restart_webui()`, `stop_webui()`, unload/reload checkpoint handlers, and any remaining OpenClaw runtime utility endpoints in `modules/api/api.py`, looking for stale wrappers or duplicated control responses while preserving route gating, restart/stop side effects, queue/runtime locking, and public API behavior.


## Pass 90 - API server-control and OpenClaw runtime utility handlers (2026-06-20)

### Scope checked
- `modules/api/api.py`: route registration after `Api.__init__()` for `/sdapi/v1/unload-checkpoint`, `/sdapi/v1/reload-checkpoint`, gated `/sdapi/v1/server-kill`, `/sdapi/v1/server-restart`, `/sdapi/v1/server-stop`, and OpenClaw runtime utility routes `/sdapi/v1/openclaw/sdpa-backend`, `/sdapi/v1/openclaw/cuda-graphs`, `/sdapi/v1/openclaw/generation-diagnostics`, `/sdapi/v1/openclaw/precision-map`, and `/sdapi/v1/precision-map`.
- `modules/api/api.py`: `apply_openclaw_runtime_defaults()`, `_env_flag()`, `get_sdpa_backend()`, `set_sdpa_backend()`, `get_cuda_graphs()`, `set_cuda_graphs()`, `get_openclaw_generation_diagnostics()`, `get_precision_map()`, `_call_with_queue_lock()`, `unloadapi()`, `reloadapi()`, `kill_webui()`, `restart_webui()`, and `stop_webui()`.
- Adjacent side-effect targets: `modules/restart.py`, `modules/sd_models.py` `unload_model_weights()` / `reload_model_weights()` references, and `modules/ui_extensions.py` restart/stop behavior.
- Focused contract tests: `tests/test_api_server_control_contract.py` route gating, checkpoint reload response shape, server-control side effects, and OpenClaw runtime metadata/default behavior.

### Findings / fixes
- No safe production-code deletion or consolidation was found in this slice. The scoped functions are short but live public/API or runtime-control surfaces with intentional side effects.
- Preserved `unloadapi()` and `reloadapi()` despite identical empty response shape. They call different model-side effects (`sd_models.unload_model_weights()` and `sd_models.send_model_to_device(shared.sd_model)`) and the public empty-object response is already pinned by focused tests.
- Preserved gated server-control handlers as explicit methods. `kill_webui()` intentionally returns `None` after `restart.stop_program()`; `restart_webui()` preserves the historical `501` response path after the restartability check; `stop_webui()` sets `shared.state.server_command = "stop"` and returns `Response("Stopping.")`. Collapsing those into a shared response helper would be cosmetic and risk public behavior around process exit/restart side effects.
- Preserved `/sdapi/v1/server-*` route gating behind `shared.cmd_opts.api_server_stop`; exposing or reshaping those routes would be a security/compatibility behavior change.
- Preserved OpenClaw runtime utility endpoints. `get_precision_map()` intentionally uses `queue_lock` directly because it walks `shared.sd_model`; `_call_with_queue_lock()` remains the shared utility for other queued side effects; SDPA/CUDA graph setters delegate to separate runtime modules and keep request/fallback semantics covered by tests.
- No duplicate AST bodies were found across `modules/api/api.py` and `tests/test_api_server_control_contract.py` in the scoped scan.

### Static/dynamic audit map notes
- Checkpoint utility chain remains: `/sdapi/v1/unload-checkpoint` -> `Api.unloadapi()` -> `sd_models.unload_model_weights()` -> `{}`; `/sdapi/v1/reload-checkpoint` -> `Api.reloadapi()` -> `sd_models.send_model_to_device(shared.sd_model)` -> `{}`.
- Server-control chain remains gated: `--api-server-stop` / `shared.cmd_opts.api_server_stop` controls registration of `/sdapi/v1/server-kill`, `/sdapi/v1/server-restart`, and `/sdapi/v1/server-stop`.
- Server side effects remain: kill exits via `restart.stop_program()`, restart calls `restart.restart_program()` only when `restart.is_restartable()` is true then returns `Response(status_code=501)` if control reaches the response, and stop sets `shared.state.server_command` for the server loop.
- OpenClaw runtime chains remain: `/sdapi/v1/openclaw/sdpa-backend` delegates to `sd_hijack_optimizations`; `/sdapi/v1/openclaw/cuda-graphs` delegates to `modules.openclaw_cuda_graphs`; generation diagnostics reads `modules.openclaw_generation_diagnostics`; precision-map aliases both use `queue_lock` before `build_precision_map()`.
- Compatibility surfaces to keep conservative: route paths/methods, disabled-by-default server-control gating, server-control response statuses/content, checkpoint empty-object responses, process exit/restart behavior, environment default parsing, queue/runtime locking, precision-map alias route, and OpenClaw endpoint request/fallback dictionaries.

### Validation log
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pycompile-pass90.XXXXXX) python3 -m py_compile modules/api/api.py tests/test_api_server_control_contract.py modules/restart.py modules/ui_extensions.py` - passed.
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pytest-pass90.XXXXXX) python3 -m pytest -q tests/test_api_server_control_contract.py` - passed: 7 passed, with the existing pytest config warning `Unknown config option: base_url`.
- Exact AST duplicate class/function scan across `modules/api/api.py` and `tests/test_api_server_control_contract.py` produced no duplicate bodies.
- `git diff --check` - passed.
- Live server-control and checkpoint API requests were not exercised because this bounded slice did not start or stop a WebUI runtime.

### Next unchecked scope
- More slices are still needed. Recommended next slice: leave the audited API utility tail and move into adjacent startup/runtime glue outside `modules/api/api.py`, especially `modules/restart.py`, `modules/initialize_util.py` queued onchange registration, and the UI/API checkpoint unload/reload callers, looking for stale wrappers or duplicated restart/reload utility paths while preserving process-control semantics and model reload locking.

## Pass 91 - Startup/runtime restart and model reload glue (2026-06-20)

### Scope checked
- `modules/restart.py`: `is_restartable()`, `restart_program()`, and `stop_program()` process-control helpers.
- `modules/initialize_util.py`: `restore_config_state_file()` and `configure_opts_onchange()` queued option-change registration for checkpoint, VAE, temp-dir, theme, attention optimization, FP8/MXFP8/NVFP4, and weight-cache settings.
- `modules/sd_models.py`: `reload_model_weights()` TorchAO/forced-reload branches and `unload_model_weights()` references/side effects.
- `modules/ui_settings.py`: Actions-tab checkpoint unload/reload and Reload UI callers.
- `modules/ui_extensions.py`: `apply_and_restart()`, `restore_config_state()`, Installed-tab apply label, and restart/stop behavior after extension changes.
- Adjacent state/control surfaces: `modules/shared_state.py` `request_restart()`, `server_command`, and compatibility `need_restart`; route callers in `modules/api/api.py` were reference-checked from pass 90 context.

### Findings / fixes
- No safe production-code deletion or helper consolidation was found in this slice. The scoped wrappers are short, but they are live public, UI, process-control, or startup/runtime side-effect surfaces.
- Preserved `modules/restart.py` as the central process-control primitive. `restart_program()` creates `tmp/restart` before exiting so launch wrappers can restart; `stop_program()` intentionally exits immediately via `os._exit(0)`; `is_restartable()` reflects the `SD_WEBUI_RESTART` launcher contract.
- Preserved `configure_opts_onchange()` registrations. The queued checkpoint/VAE/runtime reload callbacks are option side effects, not dead registration; the quantization-related callbacks intentionally force fresh reload paths for MXFP8/NVFP4/weight-cache changes while preserving model reload locking through `wrap_queued_call()`.
- Preserved separate UI and API checkpoint unload/reload callers. UI settings return user-visible timing text and reload from RAM to VRAM through `send_model_to_device(shared.sd_model)`; API handlers preserve empty-object route responses; `unload_model_weights()` remains the shared unload side effect.
- Preserved both restart paths in `ui_extensions.py`: extension apply/update must hard restart or quit through `modules.restart` after writing extension options, while config-state restore requests an in-process UI restart through `shared.state.request_restart()` after setting `restore_config_state_file`/webui config state.
- Preserved `shared.state.request_restart()`/`server_command` behavior. UI reload and API stop/restart flows depend on distinct commands and compatibility `need_restart` access; no duplicate process-control helper could be removed without changing external behavior.

### Static/dynamic audit map notes
- Extension apply chain remains: Installed tab Apply -> `apply_and_restart()` -> validate extension access/list payloads -> optional update backup/fetch -> save extension options -> `restart.restart_program()` when restartable, otherwise `restart.stop_program()`.
- Config restore chain remains: Extensions config-state restore -> set `shared.opts.restore_config_state_file` for extension restore and/or restore webui config -> `shared.state.request_restart()` -> startup `initialize_rest()` later calls `initialize_util.restore_config_state_file()` to apply extension state.
- Settings checkpoint chain remains: Actions tab unload -> `sd_models.unload_model_weights()` -> timing text; Actions tab reload -> `sd_models.send_model_to_device(shared.sd_model)` -> timing text.
- Option onchange chain remains: startup `initialize.initialize()` -> `initialize_util.configure_opts_onchange()` -> `Options.onchange()` callbacks -> `wrap_queued_call()` -> model/VAE/hijack reload side effects under `queue_lock`.
- Compatibility surfaces to keep conservative: launcher `SD_WEBUI_RESTART` contract, `tmp/restart` marker, immediate process exit semantics, UI text/response shapes, API empty-object checkpoint responses, Gradio JS restart hooks, option onchange `call=False` behavior, queue locking, TorchAO forced-reload behavior, and extension/config restore side effects.

### Validation log
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pycompile-pass91.XXXXXX) python3 -m py_compile modules/restart.py modules/initialize_util.py modules/ui_extensions.py modules/sd_models.py modules/ui_settings.py modules/shared_state.py tests/test_api_server_control_contract.py tests/test_ui_extensions_contract.py` - passed.
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pytest-pass91.XXXXXX) python3 -m pytest -q tests/test_api_server_control_contract.py tests/test_ui_extensions_contract.py` - passed: 9 passed, with the existing pytest config warning `Unknown config option: base_url`.
- Exact AST duplicate function scan across `modules/restart.py`, `modules/initialize_util.py`, `modules/ui_extensions.py`, `modules/sd_models.py`, `modules/ui_settings.py`, and `modules/shared_state.py` produced no duplicate function bodies.
- Reference scans confirmed live callers for restart/stop helpers, queued onchange registration, checkpoint unload/reload, extension apply/restart, config restore restart, and API/UI server-command surfaces.
- `git diff --check` - passed after the ledger-only checkpoint.
- Live process restart/stop, UI reload, extension update/apply, and model checkpoint reload/unload actions were not exercised because this bounded slice did not start, stop, or mutate a WebUI runtime.

### Next unchecked scope
- More slices are still needed. Recommended next slice: continue from startup/runtime glue into UI startup/app reload orchestration around `webui.py`, `modules/ui.py` reload/restart hooks, and `modules/shared_state.py` command consumption, looking for stale restart compatibility shims or duplicated UI reload loop helpers while preserving launcher behavior, Gradio unload/reload hooks, server command semantics, and extension/config restore side effects.
## Pass 92 - UI startup/app reload orchestration and command shims (2026-06-20)

### Scope checked
- webui.py: module startup import path, create_api(), api_only(), browser-UI guard webui(), and __main__ dispatch.
- modules/ui.py: top-level headless UI/runtime imports, create_ui() startup side effects, Gradio block construction return surface, demo.ui_loadsave, and setup_ui_api() internal route mounting.
- modules/shared_state.py: server_command property/signal, wait_for_server_command(), compatibility need_restart, and request_restart().
- Adjacent reload/callback surfaces: modules/script_callbacks.py app_reload_callback(), on_before_reload(), script_unloaded_callback(), on_before_ui(), and app_started_callback(); modules/initialize.py initialize_rest(reload_script_modules=...); modules/launch_utils.py startup dispatch; modules/headless_ui.py fallback Blocks methods; UI extension/config restore callers in modules/ui_extensions.py and UI settings reload caller in modules/ui_settings.py.

### Findings / fixes
- No safe production-code deletion or consolidation was found in this slice. This fork has already removed the browser UI runtime loop from webui.py: non---nowebui startup now raises a clear SystemExit, while --nowebui --api uses api_only() without constructing a Gradio app.
- Preserved modules/ui.py create_ui() and setup_ui_api() despite the disabled browser-UI entry point. The module remains imported during initialization, exports helper symbols consumed by txt2img/img2img/extension modules, and still carries public/internal UI construction APIs that extension code can import dynamically.
- Preserved script_callbacks.app_reload_callback() and on_before_reload(). No tracked in-repo caller remains after browser UI removal, but these are public extension callback registration/dispatch surfaces; deleting them would risk extension compatibility without reducing active runtime complexity.
- Preserved shared_state.server_command, wait_for_server_command(), and need_restart. The current tracked runtime no longer consumes commands in a webui reload loop, but API stop still writes server_command = "stop", UI settings/config restore still call request_restart(), and need_restart is an explicit compatibility shim. Removing or narrowing these would change public runtime state semantics.
- Preserved extension/config restore side effects: ui_extensions.restore_config_state() sets restore config state and requests an in-process restart; startup initialize_rest() still consumes restore_config_state_file() during initialization.
- No duplicated UI reload loop helper remains to consolidate. The active code has one API-only launch path and no Gradio unload/reload loop; remaining duplicate AST bodies are intentionally inert headless UI fallback methods or compatibility no-op methods.

### Static/dynamic audit map notes
- API-only startup chain remains: modules.launch_utils.start() -> webui.api_only() when --nowebui is present -> initialize.initialize() -> FastAPI app/middleware/API setup -> script_callbacks.before_ui_callback() -> script_callbacks.app_started_callback(None, app) -> Api.launch().
- Browser UI chain remains intentionally disabled: modules.launch_utils.start() without --nowebui -> webui.webui() -> SystemExit("The browser UI has been removed from this GB10 fork...").
- UI helper import chain remains: initialize.imports() imports modules.ui; other modules import modules.ui symbols such as plaintext_to_html, paste symbols, wrap_ui_call, wrap_ui_call_no_job, and create_refresh_button.
- Reload/config chain remains compatibility-only/currently non-looped: UI settings Reload UI and extension config restore call shared.state.request_restart(); request_restart() interrupts generation and sets server_command = "restart"; no tracked loop consumes it in the current headless-only runtime.
- Callback chain remains: extensions may register on_before_reload(), but no tracked active runtime caller invokes app_reload_callback() after browser UI removal; before_ui_callback() and app_started_callback(None, app) are still invoked by api_only().
- Compatibility surfaces to keep conservative: disabled browser-UI error text, --nowebui API launch behavior, before_ui/app_started callback timing, public callback registration names, server_command/need_restart properties, extension config restore request side effects, modules.ui public helper imports, and headless UI fallback no-op methods.

### Validation log
- PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pycompile-pass92.XXXXXX) python3 -m py_compile webui.py modules/ui.py modules/shared_state.py modules/script_callbacks.py modules/initialize.py modules/launch_utils.py modules/headless_ui.py tests/test_api_server_control_contract.py tests/test_ui_extensions_contract.py - passed.
- PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pytest-pass92.XXXXXX) python3 -m pytest -q tests/test_api_server_control_contract.py tests/test_ui_extensions_contract.py - passed: 9 passed, with the existing pytest config warning Unknown config option: base_url.
- Exact no-ignore reference scans confirmed no tracked active caller for wait_for_server_command() or app_reload_callback(), while server_command writes, request_restart() callers, before_ui/app_started callbacks, and modules.ui public helper imports remain live.
- Exact AST duplicate function scan across webui.py, modules/ui.py, modules/shared_state.py, modules/script_callbacks.py, modules/initialize.py, modules/launch_utils.py, and modules/headless_ui.py reported only intentional headless UI fallback/no-op duplicates: _FallbackComponent fluent methods plus Blocks.queue, inert load/from_hub, and inert setup_progressbar/Blocks.close/dump shapes.
- git diff --check - passed after replacing the ledger entry.
- Live UI reload/restart, extension config restore, and browser-UI startup were not exercised because this bounded slice did not start, stop, or mutate a WebUI runtime; the browser UI entry point is intentionally disabled in this fork.

### Next unchecked scope
- More slices are still needed. Recommended next slice: move from UI startup/reload orchestration into the headless UI compatibility layer and UI helper exports, especially modules/headless_ui.py, top-level modules/ui.py compatibility no-ops/aliases, and callers that import UI symbols without constructing the browser UI, looking for safely removable fallback methods or duplicate compatibility helpers while preserving extension import compatibility and API/headless startup behavior.

## Pass 93 - Headless UI compatibility layer and UI helper exports (2026-06-20)

### Scope checked
- `modules/headless_ui.py`: inert Gradio replacement module, `_FallbackUpdate`, `_FallbackComponent`, fluent event methods, component subclasses, `Blocks`, `Request`, `update()`, `Warning()`, dynamic `__getattr__()`, `components`/`blocks`/`routes` module shims, and `sys.modules` compatibility registration for `gradio*` imports.
- `modules/ui.py`: top-level `headless_ui` import/aliasing, UI helper exports (`plaintext_to_html`, `create_setting_component`, symbols), `gr_show()`, gallery/resolution helpers, `connect_clear_prompt()`, token counter helpers, `setup_progressbar()`, `apply_setting()`, `create_output_panel()`, `ordered_ui_categories()`, `create_override_settings_dropdown()`, `create_ui()`, `versions_html()`, and `setup_ui_api()`.
- Callers importing UI symbols without constructing the browser UI: `modules/txt2img.py`, `modules/img2img.py`, `modules/ui_toprow.py`, `modules/processing_scripts/seed.py`, `scripts/postprocessing_upscale.py`, `modules/ui_common.py`, `modules/ui_settings.py`, `modules/ui_extensions.py`, `modules/infotext_utils.py`, bundled scripts, and bundled extensions that import `modules.headless_ui as gr`.
- Compatibility import surfaces: tracked direct `modules.headless_ui` imports, dynamic/legacy `gradio`, `gradio.components`, `gradio.blocks`, and `gradio.routes` module names, and component/update/staticmethod access patterns such as `gr.Dropdown.update()` / `gr.Button.update()`.

### Findings / fixes
- No safe production-code deletion or consolidation was found in this slice. The scoped code is mostly compatibility scaffolding, but it is actively used to let API/headless startup collect script defaults and metadata without launching the browser UI.
- Preserved `_FallbackComponent` fluent methods (`style`, `then`, `success`, `click`, `change`, `submit`, `release`, `select`, `upload`, `clear`, `load`, and `render`) despite identical no-op bodies. Current core UI setup, settings, extra-network metadata editing, extension pages, and bundled scripts call a broad subset of these methods while constructing API metadata; collapsing or removing individual methods would risk extension/script import compatibility for negligible runtime gain.
- Preserved `Blocks.queue()`, `Blocks.close()`, and `Blocks.get_config_file()`. They mirror old Gradio lifecycle/config hooks and are reached by UI construction, callbacks, tests, or extension-facing compatibility expectations; `Blocks.launch()` remains the deliberate failure guard if a browser UI launch is attempted in the headless-only build.
- Preserved dynamic component/class shims and `sys.modules.setdefault("gradio*", ...)` registrations. Reference scans showed tracked scripts/extensions importing `modules.headless_ui as gr`, while third-party extensions may still import historical `gradio` modules directly during API script-control declaration.
- Preserved top-level `modules.ui` helper exports and thin aliases. First-party code still imports `plaintext_to_html`, prompt/tool symbols, and wrapping helpers through `modules.ui`; the browser UI entry point is disabled, but `modules.ui` remains an import compatibility module and the owner of script-control metadata construction.
- Preserved `setup_progressbar()` as an inert compatibility hook. It is duplicate-shaped with other no-op shims, but its name is public on `modules.ui`; deleting it would be a compatibility break and would not reduce active headless runtime complexity.
- No safe helper consolidation was made. The exact duplicate scan only reported intentional inert fallback/no-op bodies in `modules/headless_ui.py`, `modules/ui.py`, and already-audited `modules/ui_components.py` component-name helpers.

### Static/dynamic audit map notes
- Headless script-control chain remains: startup/API imports script modules -> scripts define UI controls with `modules.headless_ui as gr` -> inert components preserve `value`, `label`, `elem_id`, `choices`, visibility, and event methods -> `modules/scripts.py` extracts defaults/API metadata without requiring real Gradio or a browser UI.
- Legacy Gradio import chain remains: `modules/headless_ui.py` registers itself as `gradio` plus `gradio.components`, `gradio.blocks`, and `gradio.routes`; dynamic `__getattr__()` supplies `themes`, `utils`, `components`, `blocks`, `routes`, and `__version__` compatibility values.
- UI helper import chain remains: `modules/txt2img.py` and `modules/img2img.py` import `plaintext_to_html` from `modules.ui`; `modules/ui_toprow.py`, `modules/processing_scripts/seed.py`, and `scripts/postprocessing_upscale.py` import old prompt/tool symbols from `modules.ui`; settings/extensions code still reaches `modules.ui.wrap_ui_call*` imports from earlier top-level re-exports.
- API/headless startup still imports `modules.ui` during initialization, but `webui.webui()` remains the explicit disabled-browser-UI path from pass 92. `create_ui()` is therefore preserved as compatibility/API metadata construction surface rather than treated as active browser launch code.
- Compatibility surfaces to keep conservative: exported component class names, static `Component.update()` behavior, `gr.update()` object type and dictionary keys, fluent event method names, context-manager behavior, `gradio*` module aliases, route/template placeholders, theme/utils placeholders, `modules.ui` symbol exports, token counter HTML shape, settings/apply side effects, and browser-UI launch failure semantics.

### Validation log
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pycompile-pass93.XXXXXX) python3 -m py_compile modules/headless_ui.py modules/ui.py modules/ui_common.py modules/ui_components.py modules/ui_toprow.py modules/ui_settings.py modules/ui_extensions.py modules/infotext_utils.py modules/txt2img.py modules/img2img.py scripts/postprocessing_upscale.py tests/test_api_server_control_contract.py tests/test_ui_extensions_contract.py test/test_infotext_paste_bindings.py test/test_postprocessing_script_args.py` - passed.
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pytest-pass93.XXXXXX) python3 -m pytest -q tests/test_api_server_control_contract.py tests/test_ui_extensions_contract.py test/test_infotext_paste_bindings.py test/test_postprocessing_script_args.py` - passed: 15 passed, with the existing pytest config warning `Unknown config option: base_url`.
- Reference scans confirmed tracked use of `modules.headless_ui as gr` across core modules, bundled scripts, bundled extensions, and tests; tracked `gr.*` attribute usage includes component classes, `Blocks`, `Request`, `Warning`, `Info`, `update`, `themes`, `components`, `blocks`, and `routes`.
- Reference scans confirmed tracked `modules.ui` helper imports for `plaintext_to_html`, prompt/tool symbols, and UI construction/extension/settings surfaces; direct grep did not find removable top-level `modules.ui` helper exports that were both unreferenced and non-public.
- Exact AST duplicate function scan across `modules/headless_ui.py`, `modules/ui.py`, `modules/ui_common.py`, `modules/ui_components.py`, `modules/ui_toprow.py`, `modules/ui_settings.py`, and `modules/infotext_utils.py` reported only intentional compatibility/no-op duplicates: `_FallbackComponent` fluent methods plus `Blocks.queue`, inert `Blocks.close`/theme `dump`/`modules.ui.setup_progressbar`, inert theme `load`/`from_hub`, and component `get_block_name()` overrides in `modules/ui_components.py`.
- `git diff --check` - passed after the ledger-only checkpoint.
- Live browser UI launch, third-party extension imports outside the checked tree, and full API runtime startup were not exercised in this bounded slice.

### Next unchecked scope
- More slices are still needed. Recommended next slice: continue from headless UI compatibility into remaining UI component wrapper/export surfaces, especially `modules/ui_components.py`, `modules/ui_common.py` helper/button/dialog wrappers, and UI HTML/static helper modules (`modules/ui_html_extensions.py`, `modules/ui_loadsave.py`, `modules/ui_tempdir.py`) looking for stale wrapper methods or duplicate compatibility helpers while preserving extension-facing component subclasses, JavaScript hook IDs, and API/headless metadata construction.

## Pass 94 - UI component wrapper/export surfaces (2026-06-20)

### Scope checked
- `modules/ui_components.py`: `FormComponent`, `ToolButton`, `ResizeHandleRow`, `FormRow`, `FormColumn`, `FormGroup`, `FormHTML`, `FormColorPicker`, `DropdownMulti`, `DropdownEditable`, and `InputAccordion` including `extra()`, context-manager forwarding, `get_block_name()` overrides, duplicate accordion-ID reset, and JavaScript hook wiring.
- `modules/ui_common.py`: `update_generation_info()`, `plaintext_to_html()`, `update_logfile()`, `save_files()`, `OutputPanel`, `create_output_panel()`, `create_refresh_button()`, and `setup_dialog()` helper/button/dialog wrappers.
- `modules/ui_html_extensions.py`: `webpath()`, `javascript_html()`, `css_html()`, `reload_javascript()`, and `UITemplateResponseOriginal` preservation.
- `modules/ui_loadsave.py`: `radio_choices()` and `UiLoadsave` component registration, defaults dump/review/apply, saved-value coercion, `InputAccordion` handling, and tab/default mapping behavior.
- `modules/ui_tempdir.py`: `Savedfile`, `register_tmp_file()`, `check_tmp_file()`, `save_pil_to_file()`, `install_ui_tempdir_override()`, `on_tmpdir_changed()`, `cleanup_tmpdr()`, and `is_ui_temp_path()`.
- Adjacent callers/reference surfaces: `modules/ui.py`, `modules/ui_settings.py`, `modules/ui_checkpoint_merger.py`, `modules/ui_postprocessing.py`, `modules/ui_toprow.py`, `modules/shared_options.py`, `modules/shared_items.py`, `modules/processing_scripts/*`, bundled `scripts/*`, bundled `extensions-builtin/*`, `modules/infotext_utils.py`, `modules/postprocessing.py`, `modules/extras.py`, `modules/profiling.py`, and focused tests for save serialization/loadsave contracts.

### Findings / fixes
- No safe production-code deletion or consolidation was found in this slice. The scoped code is wrapper-heavy, but the wrappers remain live component/export/compatibility surfaces for headless metadata construction, settings defaults, bundled scripts, bundled extensions, and likely third-party extensions.
- Preserved all `modules/ui_components.py` subclasses. Reference scans confirmed live tracked use of `ToolButton`, `ResizeHandleRow`, `FormRow`, `FormColumn`, `FormGroup`, `FormHTML`, `FormColorPicker`, `DropdownMulti`, `DropdownEditable`, and `InputAccordion`; several are imported through `modules.ui_components` attributes rather than direct imports. The repeated `get_block_name()` bodies are intentional Gradio/headless block-name compatibility overrides, not removable dead code.
- Preserved `InputAccordion` ID/reset, hidden-checkbox synchronization, `extra()` slot, and context-manager forwarding. Bundled postprocessing scripts and soft-inpainting use it, and `UiLoadsave` has special handling for the accordion/open value plus `custom_script_source` propagation.
- Preserved `ui_common` output-panel, save, refresh, and dialog helpers. `create_output_panel()` still owns output/gallery DOM IDs and JavaScript hook IDs; `save_files()` behavior is covered by focused save serialization tests; `create_refresh_button()` and `setup_dialog()` have live settings/metadata/style dialog callers.
- Preserved `ui_html_extensions` HTML/CSS injection helpers and template-response wrapper. They remain called from `modules/ui.py` and `modules/ui_settings.py`, and the exact script/style ordering plus `UITemplateResponseOriginal` guard are public UI compatibility behavior even in this headless-focused fork.
- Preserved `UiLoadsave` and `radio_choices()`. Component default persistence is live during UI metadata construction, with tests covering HTML escaping; helper consolidation would risk saved `ui-config.json` semantics for legacy tuple/list choices, dropdown validation, tab selection, and `InputAccordion` open/value mapping.
- Preserved `ui_tempdir` helpers. The legacy `temp_file_sets` and `temp_dirs` branches are version-compatibility paths used by `save_pil_to_file()`, infotext paste path validation, temp-dir option onchange registration, and output-folder selection logic; cleanup semantics and `GRADIO_TEMP_DIR` detection were left unchanged.

### Static/dynamic audit map notes
- Component wrapper chain remains: UI/script code constructs `modules.ui_components` subclasses -> headless/Gradio block name and expected-parent compatibility are preserved -> script/default metadata extraction continues without browser UI launch.
- Output-panel chain remains: txt2img/img2img/extras UI construction -> `ui_common.create_output_panel()` -> stable gallery/download/html DOM IDs, JS selected-gallery hooks, paste-parameter button registration, and save/download helpers.
- Refresh/dialog chain remains: settings/checkpoint/style/metadata controls -> `ui_common.create_refresh_button()` / `setup_dialog()` -> Gradio event wiring and JS popup hooks.
- HTML injection chain remains: `reload_javascript()` wraps `gr.routes.templates.TemplateResponse` once -> injects localization, core `script.js`, extension JS/MJS, theme setter, referrer meta, stylesheets, user CSS, and theme background style.
- UI defaults chain remains: `modules/ui.py` creates `UiLoadsave(cmd_opts.ui_config_file)` -> `add_block()` / `add_component()` collect component defaults -> `dump_defaults()`, `ui_view()`, and `ui_apply()` preserve saved UI configuration behavior.
- Temp-file chain remains: `install_ui_tempdir_override()` overrides `gr.components.IOComponent.pil_to_temp_file` -> `save_pil_to_file()` preserves PNG metadata and temp registration -> `infotext_utils` validates pasted file paths with `check_tmp_file()` -> output folder helper avoids treating UI temp paths as regular output dirs unless configured.
- Compatibility surfaces to keep conservative: component class names, `get_block_name()` strings, `get_expected_parent()` monkey patch, DOM/elem IDs, JavaScript hook names (`inputAccordionChecked`, `popupId`, `closePopup`, `selected_gallery_index`), saved UI config keys/values, CSV log migration shape, Gradio temp-file registration behavior, temporary-file cleanup scope, and HTML/JS/CSS injection order.

### Validation log
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pycompile-pass94.XXXXXX) python3 -m py_compile modules/ui_components.py modules/ui_common.py modules/ui_html_extensions.py modules/ui_loadsave.py modules/ui_tempdir.py modules/ui.py modules/ui_settings.py modules/ui_toprow.py modules/ui_checkpoint_merger.py modules/ui_postprocessing.py modules/infotext_utils.py modules/postprocessing.py modules/extras.py tests/test_ui_loadsave_contract.py tests/test_save_serialization_contract.py` - passed.
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pytest-pass94.XXXXXX) python3 -m pytest -q tests/test_ui_loadsave_contract.py tests/test_save_serialization_contract.py` - passed: 4 passed, with the existing pytest config warning `Unknown config option: base_url`.
- Reference scans confirmed live tracked callers/imports for all scoped component subclasses except some are only reached as `ui_components.*` option/component factories, which is still a live extension/settings export surface.
- Reference scans confirmed live callers for `create_output_panel()`, `create_refresh_button()`, `setup_dialog()`, `reload_javascript()`, `UiLoadsave`, and the tempdir registration/check/path helpers.
- Exact AST duplicate function scan across `modules/ui_components.py`, `modules/ui_common.py`, `modules/ui_html_extensions.py`, `modules/ui_loadsave.py`, and `modules/ui_tempdir.py` reported only intentional compatibility duplicates: `FormRow.get_block_name()`/`ResizeHandleRow.get_block_name()` both returning `row`, and `DropdownMulti.get_block_name()`/`DropdownEditable.get_block_name()` both returning `dropdown`.
- `git diff --check` - passed after the ledger-only checkpoint.
- Live browser UI launch, third-party extension imports outside the checked tree, real file save/download, and temp-dir cleanup execution were not exercised in this bounded slice.

### Next unchecked scope
- More slices are still needed. Recommended next slice: continue from UI wrapper/export surfaces into remaining UI-adjacent configuration and settings helper surfaces, especially `modules/ui_settings.py`, `modules/shared_options.py`, `modules/shared_items.py`, and option-component factory paths, looking for stale settings wrappers or duplicate option helper logic while preserving saved config keys, option metadata, refresh hooks, and extension-facing option APIs.

## Pass 95 - UI-adjacent settings and option helper surfaces (2026-06-20)

### Scope checked
- `modules/ui_settings.py`: `get_setting_component_args()`, `get_value_for_setting()`, removed `refreshed_setting_component_args()`, `create_setting_component()`, `UiSettings.run_settings()`, `run_settings_single()`, `register_settings()`, `create_ui()`, `add_quicksettings()`, `add_functionality()`, and `search()`.
- `modules/shared_options.py`: option template registration, category registration, restricted option key list, component factories, refresh hooks, API-restricted settings, UI reordering/infotext/postprocessing settings, and saved config key metadata.
- `modules/shared_items.py`: model/upscaler/VAE/UNet/checkpoint/sampler listing helpers, dropdown-args helpers, refresh/list wrappers, hypernetwork reload, infotext-name collection, UI reorder categories, and callback priority option generation.
- Option component factory/caller paths: `modules/options.py` `OptionInfo`/`OptionHTML`/`options_section()`, `modules/ui_common.py` `create_refresh_button()`, `modules/ui.py` `apply_setting()`/refresh-button callers, `extensions-builtin/extra-options-section/scripts/extra_options_section.py`, API refresh routes, and focused tests covering API refresh/control and infotext option names.

### Findings / fixes
- Removed the private `modules.ui_settings.refreshed_setting_component_args()` wrapper. It only delegated to `get_setting_component_args()` and had no tracked callers outside `modules/ui_settings.py`; settings refresh buttons now call the existing helper directly through their lazy lambda, preserving dynamic `component_args` recomputation after refresh.
- Preserved `get_setting_component_args()`, including its `precision` filtering. It is shared by settings value updates, component construction, and refresh-button argument recomputation, and preserves Gradio/headless component behavior for numeric settings.
- Preserved `get_value_for_setting()` and `create_setting_component()` as extension-facing settings helpers. The bundled extra-options extension imports both, and third-party scripts may use the same public `modules.ui_settings` surface.
- Preserved `shared_options.options_templates`, `restricted_opts`, category registration, saved option keys, option metadata, refresh hooks, onchange hooks, and API restriction flags. Deleting or consolidating these would alter saved config compatibility, generated API config behavior, or settings freeze/restrict semantics.
- Preserved `shared_items` list/refresh helpers and dropdown-args helpers despite similar shapes. Reference scans confirmed live tracked callers, and these helpers are public compatibility surfaces re-exported through `modules.shared` or used by bundled UI/extension code.
- No additional safe duplicate helper consolidation was found. The exact duplicate scan over this slice reported no duplicate function bodies after the wrapper removal.

### Static/dynamic audit map notes
- Settings component chain remains: `OptionInfo.component_args` may be a dict or callable -> `get_setting_component_args()` evaluates it lazily and strips unsupported `precision` -> `create_setting_component()` constructs the correct headless/Gradio component -> `create_refresh_button()` recomputes choices/args after the refresh method runs.
- Quicksettings and extra-options chain remains: normal settings tabs skip quicksettings into `dummy_component`; `add_quicksettings()` creates them later; `extensions-builtin/extra-options-section` uses `create_setting_component()` and `get_value_for_setting()` for embedded option controls.
- Settings save chain remains: `run_settings()` validates each submitted value against `opts.data_labels` defaults before `opts.set()` and save; `run_settings_single()` preserves value restoration and `opts.dumpjson()` output for the hidden generated API/config text.
- Shared option metadata chain remains: `shared_options.options_templates` builds stable option keys and metadata; `options_section()` assigns settings sections/categories; `Options` enforces frozen/restricted settings and API restriction behavior elsewhere.
- Shared item helper chain remains: checkpoint/VAE/UNet/hypernetwork/sampler/postprocessing helpers feed option component choices, API refresh routes, UI refresh buttons, metadata dialogs, and `modules.shared` compatibility re-exports.
- Compatibility surfaces kept conservative: saved config keys, option section/category IDs, `OptionInfo` constructor fields and fluent metadata methods, refresh/onchange hooks, `restrict_api`, generated `opts.dumpjson()` behavior, `modules.shared` helper aliases, extension imports from `modules.ui_settings` and `modules.shared_items`, and dynamic callable component args.

### Validation log
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pycompile-pass95.XXXXXX) python3 -m py_compile modules/ui_settings.py modules/shared_options.py modules/shared_items.py modules/options.py modules/ui_common.py modules/ui.py extensions-builtin/extra-options-section/scripts/extra_options_section.py test/test_infotext_api_mappings.py tests/test_api_server_control_contract.py` - passed.
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pytest-pass95.XXXXXX) python3 -m pytest -q test/test_infotext_api_mappings.py tests/test_api_server_control_contract.py` - passed: 16 passed, with the existing pytest config warning `Unknown config option: base_url`.
- Reference scan for `refreshed_setting_component_args`, `get_setting_component_args`, `get_value_for_setting`, and `create_setting_component()` confirmed the removed wrapper has no remaining references, while the public settings helpers remain live in core settings and the bundled extra-options extension.
- Reference scans confirmed live tracked callers for shared item dropdown/list/refresh helpers, `callbacks_order_settings()`, `get_infotext_names()`, `ui_reorder_categories()`, and postprocessing script option helpers.
- Exact AST duplicate function scan across `modules/ui_settings.py`, `modules/shared_options.py`, `modules/shared_items.py`, `modules/options.py`, and `modules/ui_common.py` reported no duplicate function bodies after the helper removal.
- `git diff --check` - passed after the production and ledger edits.
- Live browser UI, API runtime startup, third-party extension imports outside the checked tree, and actual model/metadata refresh button clicks were not exercised in this bounded slice.

### Next unchecked scope
- More slices are still needed. Recommended next slice: continue from settings helper surfaces into script/default metadata extraction and option override plumbing, especially `modules/scripts.py`, `modules/scripts_auto_postprocessing.py`, `modules/scripts_postprocessing.py`, `modules/infotext_utils.py`, and override-settings UI/API paths, looking for stale script-control/default wrappers or duplicate override mapping logic while preserving extension callback/script APIs and infotext/API compatibility.

## Pass 96 - Script/default metadata extraction and option override plumbing (2026-06-20)

### Scope checked
- `modules/scripts.py`: `Script` public hook/default methods, `ScriptRunner.initialize_scripts()`, `create_script_ui()`, `create_script_ui_inner()`, script control metadata extraction, API `ScriptInfo` construction, script arg slicing/timing helpers, `set_named_arg()`, and reload compatibility alias.
- `modules/scripts_auto_postprocessing.py`: `ScriptPostprocessingForMainUI`, `ui()`, `postprocess_image()`, and `create_auto_preprocessing_script_data()` bridge from postprocessing scripts into txt2img/img2img always-on scripts.
- `modules/scripts_postprocessing.py`: `ScriptPostprocessing`, `ScriptPostprocessingRunner.create_script_ui()`, `scripts_in_preferred_order()`, `setup_ui()`, `run()`, and `create_args_for_run()` default-preserving postprocessing arg vector construction.
- `modules/infotext_utils.py`: paste-field registration/binding, image/dimension paste, generation parameter parsing defaults, legacy infotext-to-option mapping, `create_override_settings_dict()`, `get_override_settings()`, and paste dropdown override population.
- Override/API callers and tests: `modules/ui.py` override dropdown wiring, `modules/txt2img.py`, `modules/img2img.py`, `modules/api/api.py` script defaults/infotext override application, `modules/api/models.py` script arg metadata model, `modules/processing.py` override setting application/restore, bundled scripts that mutate `p.override_settings`, and focused tests for API script defaults, API listing, infotext mappings, and postprocessing defaults.

### Findings / fixes
- Consolidated script-control default extraction into `modules.scripts.script_controls_default_values()`. Normal API default-arg initialization now uses the same helper as the script metadata/control surface instead of carrying a separate implementation in `modules/api/api.py`.
- Consolidated per-control API argument metadata extraction into `modules.scripts.script_control_api_arg()`. `ScriptRunner.create_script_ui_inner()` now delegates `ScriptArg` label/value/minimum/maximum/step/choices extraction to this helper, preserving the exact legacy choices tuple handling and generated `ScriptInfo` shape.
- Kept the `modules.api.api.script_default_ui_values()` wrapper for compatibility with existing internal tests and any external code importing that helper. In normal API runtime it delegates to `modules.scripts.script_controls_default_values()` through the module global; in AST-isolated tests without the module global it retains the old lightweight fallback to avoid importing the full runtime stack.
- Preserved all `Script` no-op hook/default methods, `describe()`, `elem_id()` helpers, `reload_scripts` alias, and `set_named_arg()`. They are extension-facing script APIs even when some bodies are default no-ops or duplicate-shaped.
- Preserved `scripts_auto_postprocessing.py` bridge logic. The lambda constructor default argument is intentional to bind each postprocessing script class; changing it risks always-on main UI postprocessing behavior.
- Preserved `scripts_postprocessing.py` default arg vector behavior. Existing tests cover omitted-key default preservation, explicit operation order, filtered scripts, and postprocessing API default ordering.
- Preserved infotext override mapping helpers and the empty legacy `infotext_to_setting_name_mapping` list. The list is intentionally mutable/public for backward compatibility, while current option mappings come from `OptionInfo(..., infotext=...)`; tests cover combining both sources.
- No override-settings behavior was changed. `create_override_settings_dict()` remains the UI multiselect-to-processing mapping for txt2img/img2img, while `get_override_settings()` remains the infotext paste/API helper that filters current values and disable-weights-auto-swap.

### Static/dynamic audit map notes
- Script metadata chain now centralizes: `ScriptRunner.create_script_ui_inner()` calls each script `ui()` once -> stores `script.controls` -> assigns `custom_script_source` -> builds `ScriptInfo.args` with `script_control_api_arg()` -> API default initialization reads defaults with `script_controls_default_values()`.
- API request chain remains: `Api.init_default_script_args()` builds default vectors from finalized script controls; `Api.init_script_args()` overlays selectable, always-on, and infotext-provided script args; `openclaw_script_args_to_overrides` still preserves extended always-on payloads.
- Infotext override chain remains: `parse_generation_parameters()` normalizes old/default generation labels -> `get_override_settings()` maps infotext labels to option names from `OptionInfo.infotext` plus legacy mutable mappings -> API/paste paths only apply non-current, non-skipped values.
- UI override dropdown chain remains: txt2img/img2img UI creates hidden multiselect dropdowns -> paste populates visible override choices through `get_override_settings()` -> generation submits selected `Label: value` strings -> `create_override_settings_dict()` casts them into `p.override_settings`.
- Postprocessing defaults chain remains: extras/API postprocessing builds per-script UI controls on demand -> `create_args_for_run()` pre-fills defaults and overlays requested script args by script name/order -> runner dispatches firstpass/process in preferred order.
- Compatibility surfaces kept conservative: script hook method names/signatures, script control `value`/`label`/`choices` metadata, `ScriptInfo` API shape, script arg vector indexes, `custom_script_source`, postprocessing script names/order filters, paste field tuple compatibility, legacy infotext mapping mutability, override settings cast/filter behavior, and disable-weights-auto-swap handling.

### Validation log
- `PYTHONPYCACHEPREFIX=$(mktemp -d) python3 -m py_compile modules/scripts.py modules/scripts_auto_postprocessing.py modules/scripts_postprocessing.py modules/infotext_utils.py modules/api/api.py modules/txt2img.py modules/img2img.py modules/processing.py` - passed.
- `pytest -q test/test_api_script_defaults.py test/test_infotext_api_mappings.py test/test_postprocessing_script_args.py test/test_postprocessing_api_defaults.py tests/test_api_listing_contract.py` - passed: 33 passed, with the existing pytest config warning `Unknown config option: base_url`.
- `git diff --check` - passed.
- Reference scans confirmed the old API helper name remains only as the compatibility wrapper plus focused tests, `ScriptArg` construction now occurs through the new script helper, and tracked override-settings callers continue through the existing UI/API/processing paths.
- Live WebUI/API server startup, third-party extension imports outside the checked tree, and real txt2img/img2img generation with override dropdown selections were not exercised in this bounded slice.

### Next unchecked scope
- More slices are still needed. Recommended next slice: continue from script/default metadata extraction into generation parameter parsing and processing override application internals, especially `modules/processing.py` option override apply/restore blocks, `modules/txt2img.py`, `modules/img2img.py`, `modules/api/api.py` generation request preparation, and bundled scripts that mutate `p.override_settings`, looking for duplicate override/default handling while preserving infotext/API compatibility and generation behavior.

## Pass 97 - Processing/generation override application internals (2026-06-20)

### Scope checked
- `modules/processing.py`: `StableDiffusionProcessing.override_settings`, `override_settings_restore_afterwards`, `__post_init__()`, `process_images()` override snapshot/apply/restore path, checkpoint override validation, VAE reload side effects, token-merging reset, and invalid sampler/scheduler compatibility fix placement.
- `modules/txt2img.py`: `txt2img_create_processing()`, UI override dropdown parsing via `create_override_settings_dict()`, highres-fix override coercions, and `txt2img_upscale()` mutation of `p.override_settings`.
- `modules/img2img.py`: `img2img()` processing construction, `process_batch()` PNG-info override propagation, batch output filename/save overrides, and per-image checkpoint override fallback/removal.
- `modules/api/api.py`: `apply_infotext()`, `_prepare_generation_api_request()`, API request copy/pop preparation, infotext-derived override merging, and script-arg overlay ordering from pass 96.
- `modules/infotext_utils.py`: `create_override_settings_dict()`, `get_override_settings()`, override dropdown paste population, current-value filtering, legacy infotext mapping, and disable-weights-auto-swap guard.
- Bundled scripts mutating `p.override_settings`: `scripts/xyz_grid.py` override axes and `scripts/prompts_from_file.py` sd-model job override handling.

### Findings / fixes
- Extracted the inline `process_images()` override lifecycle into `store_processing_override_settings()`, `apply_processing_override_settings()`, and `restore_processing_override_settings()` in `modules/processing.py`.
- Preserved existing behavior exactly: scripts still run `before_process()` before the snapshot, invalid checkpoint overrides are still removed before applying overrides, checkpoint and VAE reload side effects stay in the same order, `override_settings_restore_afterwards=False` still skips restoration, and restore still uses direct `setattr(opts, ...)` plus the existing VAE reload behavior.
- Preserved UI/API split override parsing. `create_override_settings_dict()` remains the UI multiselect parser for `Label: value` pairs, while `get_override_settings()` remains the infotext/API/paste helper that filters current values, handled fields, and disable-weights-auto-swap.
- Preserved script mutation behavior in `txt2img_upscale()`, img2img batch processing, XYZ Grid axes, and Prompts From File. These are live generation behaviors and not dead wrappers.
- No safe removal was found for `override_settings_restore_afterwards`, API `request.override_settings is None` initialization, or the legacy infotext mapping list. They are compatibility and public/dynamic behavior surfaces.

### Static/dynamic audit map notes
- Processing override chain is now explicit: scripts may mutate `p.override_settings` in `before_process()` -> `store_processing_override_settings()` snapshots current option values for override keys -> `apply_processing_override_settings()` validates/removes stale checkpoint overrides and applies settings with reload side effects -> generation runs -> token merging resets -> `restore_processing_override_settings()` restores saved option values only when `p.override_settings_restore_afterwards` is true.
- UI generation chain remains: txt2img/img2img paste dropdown choices are submitted as `Label: value` strings -> `create_override_settings_dict()` casts to option values -> `StableDiffusionProcessing.override_settings` carries them into processing.
- API generation chain remains: `apply_infotext()` fills unset request fields and merges infotext-derived overrides only when a caller did not already set the same override key -> `_prepare_generation_api_request()` copies the request into processing kwargs and strips API-only script/infotext fields.
- Script mutation chain remains: XYZ Grid and Prompts From File copy or mutate processing objects and set `p.override_settings` before `process_images()` sees them; img2img batch mutates save/checkpoint overrides per input image.
- Compatibility surfaces kept conservative: override setting key names, UI dropdown string format, API request/response shapes, infotext setting names and legacy extension mapping, `disable_weights_auto_swap`, script-visible `p.override_settings`, and restore semantics.

### Validation log
- `PYTHONPYCACHEPREFIX=$(mktemp -d) python3 -m py_compile modules/processing.py modules/txt2img.py modules/img2img.py modules/api/api.py modules/infotext_utils.py scripts/xyz_grid.py scripts/prompts_from_file.py` - passed.
- `PYTHONPYCACHEPREFIX=$(mktemp -d) python3 -m pytest -q test/test_infotext_api_mappings.py tests/test_processing_auxiliary_infotext_alignment.py` - passed: 19 passed, with the existing pytest config warning `Unknown config option: base_url`.
- `PYTHONPYCACHEPREFIX=$(mktemp -d) python3 -m pytest -q test/test_infotext_api_mappings.py tests/test_processing_auxiliary_infotext_alignment.py test/test_img2img.py` - blocked by the existing missing `base_url` fixture/live server harness in `test/test_img2img.py`; the first 19 tests passed before the 7 fixture errors.
- Exact AST duplicate function scan across `modules/processing.py`, `modules/txt2img.py`, `modules/img2img.py`, `modules/api/api.py`, `modules/infotext_utils.py`, `scripts/xyz_grid.py`, and `scripts/prompts_from_file.py` reported only trivial no-op/property duplicates: `modules/processing.py:344:sd_model`, `modules/processing.py:480:init`, `scripts/xyz_grid.py:187:do_nothing`, and `scripts/xyz_grid.py:402:__enter__`.
- Reference scan confirmed the new processing override helpers are only used by `process_images()` and tracked override-setting mutation paths remain live in UI/API/bundled scripts.
- `git diff --check` - passed.
- Live WebUI/API server startup, actual model checkpoint/VAE reloads, and real txt2img/img2img generations with override selections were not exercised in this bounded slice.

### Next unchecked scope
- More slices are still needed. Recommended next slice: continue from processing override application into generation parameter parsing and infotext serialization internals, especially `modules/processing.py` `create_infotext()`/`Processed.js()` metadata emission, `modules/infotext_utils.py` parsing/default normalization, save-path infotext consumers, and API pnginfo handling, looking for duplicate metadata/default handling while preserving old infotext labels, extension callbacks, and image metadata compatibility.


## Pass 98 - Generation parameter parsing and infotext serialization internals (2026-06-20)

### Scope checked
- `modules/processing.py`: `Processed.__init__()`, `Processed.js()`, `Processed.infotext()`, `create_infotext()`, extra-generation-param callable/list serialization, comments/negative prompt serialization, processing-time infotext creation, grid/sample infotext alignment, and override metadata interactions from pass 97.
- `modules/infotext_utils.py`: `quote()`, `unquote()`, `parse_generation_parameters()`, old hires-fix restoration, inpaint label-to-index helpers, parser default normalization, legacy `modules.generation_parameters_copypaste` alias, paste bindings, `infotext_setting_name_mapping()`, `create_override_settings_dict()`, and `get_override_settings()`.
- Save/API consumers: `modules/images.py` `save_image_with_geninfo()`/`save_image()` PNG/JPEG/WebP/AVIF/GIF metadata handling, `modules/api/api.py` `api_infotext_value_for_field()`, `apply_infotext()`, and `pnginfoapi()`, plus `modules/api/models.py` pnginfo request/response fields.
- Adjacent tests: `test/test_infotext_api_mappings.py`, `tests/test_save_serialization_contract.py`, `tests/test_processing_auxiliary_infotext_alignment.py`, and `test/test_images_save.py`.

### Findings / fixes
- Consolidated duplicate parser default insertion in `modules/infotext_utils.py` with `add_missing_default()` and `add_missing_defaults()`. This removes repeated `if key not in res: res[key] = default` branches for hires, inpaint, scheduler, VAE, quantization, and refiner defaults while preserving the exact default values and insertion order.
- Left conditional/default branches explicit where behavior depends on parsed values or compatibility semantics: missing `Clip skip`, paired `Hires resize-1`/`Hires resize-2`, missing `RNG`, `MXFP8 Linear coverage` only when MXFP8 is enabled, `Cache FP16 weight for LoRA` only when FP8 is enabled, prompt-emphasis detection, old hires-fix restoration, and `infotext_versions.backcompat()` ordering.
- Preserved old infotext labels, inpaint label mappings, the mutable empty legacy `infotext_to_setting_name_mapping` compatibility list, the `modules.generation_parameters_copypaste` alias, extension paste callbacks, API/UI paste behavior, PNGInfo request/response shape, and image metadata section handling.
- No safe deletion was found in `Processed.js()`, `create_infotext()`, save-path metadata consumers, or API pnginfo handling. These surfaces are live serialization/API contracts and remain intentionally conservative.

### Static/dynamic audit map notes
- Serialization chain remains: processing fills `p.extra_generation_params` -> `create_infotext()` merges core fields, extra params, callable/list params, version/user fields, and quoted values -> save paths store the selected infotext under the requested PNG text key or EXIF/comment metadata -> `Processed.js()` returns `infotexts` and related generation metadata to UI/API consumers.
- Parse chain remains: `parse_generation_parameters()` splits prompt/negative prompt/last parameter line -> unquotes values -> splits `WxH` values into `-1`/`-2` fields -> extracts styles -> applies default/compatibility normalization -> applies infotext version backcompat -> removes skipped fields.
- API paste/pnginfo chain remains: `apply_infotext()` fills unset request fields, casts script/API values including generic Gradio updates and bool strings, and merges override settings without overwriting caller-provided overrides; `pnginfoapi()` reads image metadata, parses parameters, invokes `infotext_pasted_callback()`, and returns `info`, `items`, and `parameters`.
- Compatibility surfaces kept conservative: infotext label text, order-sensitive generated parameter emission, quote/unquote JSON behavior, image metadata key names, legacy paste module alias, mutable legacy option mapping list, extension callbacks, and UI/API field population rules.

### Validation log
- `PYTHONPYCACHEPREFIX=$(mktemp -d) python3 -m py_compile modules/processing.py modules/infotext_utils.py modules/images.py modules/api/api.py modules/api/models.py` - passed.
- `pytest -q test/test_infotext_api_mappings.py tests/test_save_serialization_contract.py tests/test_processing_auxiliary_infotext_alignment.py test/test_images_save.py` - passed: 26 passed, with the existing pytest config warning `Unknown config option: base_url`.
- Exact parser-default reference scan now reports only the intentionally explicit branches for `Clip skip`, paired `Hires resize-1`/`Hires resize-2`, and `RNG` in `modules/infotext_utils.py`.
- Exact AST duplicate scan across `modules/processing.py`, `modules/infotext_utils.py`, `modules/images.py`, `modules/api/api.py`, and `modules/api/models.py` reported only trivial no-op/schema duplicates: `modules/processing.py:344:sd_model`, `modules/processing.py:480:init`, `modules/api/api.py:308:ScriptArgsList`, `modules/api/models.py:146:TextToImageResponse`, `modules/api/models.py:149:ImageToImageResponse`, and `modules/api/models.py:174:ExtrasSingleImageRequest`.
- `git diff --check` - passed.
- Live WebUI/API server startup, real image generation, third-party extension paste callbacks outside the checked tree, and real PNG/JPEG/WebP/AVIF metadata round-trips were not exercised in this bounded slice.

### Next unchecked scope
- More slices are still needed. Recommended next slice: continue from infotext serialization into save/download/gallery metadata consumers and UI save-file paths, especially `modules/ui_common.py`, `modules/images.py` filename/logging helpers, `modules/generation_parameters_copypaste` compatibility callers, `modules/txt2img.py`/`modules/img2img.py` gallery-index handling, and API processed-image path injection, looking for stale save metadata wrappers or duplicate gallery/infotext index handling while preserving image metadata compatibility and UI/API save behavior.

## Pass 99 - Save/download/gallery metadata consumers and UI save-file paths (2026-06-20)

### Scope checked
- `modules/ui_common.py`: `update_generation_info()`, `save_files()`, `create_output_panel()` save/download/open-folder wiring, gallery selected-index JS handoff, CSV log maintenance, zip creation, paste button registration, and old `parameters_copypaste` import alias usage.
- `modules/images.py`: `save_image_with_geninfo()`, `enabled_geninfo_exif_bytes()`, `save_image()`, filename generation, callback path rewrites, atomic save/duplicate suffix handling, 4chan JPG export path update, txt sidecar writing, and `already_saved_as` propagation.
- `modules/infotext_utils.py`: legacy `modules.generation_parameters_copypaste` module alias and `parse_generation_parameters()` compatibility callers in this save path.
- `modules/txt2img.py`: `txt2img_upscale()` gallery-index validation, selected gallery image reuse, infotext splice/update behavior, and placeholder `already_saved_as` preservation for non-upscaled images.
- `modules/img2img.py`: main `img2img()` and batch result handling where gallery/result infotext alignment flows into UI save/download behavior.
- `modules/api/api.py`: `processed_js_with_image_paths()` path injection and txt2img/img2img API response callers.
- Adjacent tests: `tests/test_save_serialization_contract.py`, `test/test_images_save.py`, and `test/test_infotext_paste_bindings.py`.

### Findings / fixes
- Fixed `modules/ui_common.py` `save_files()` batch-index metadata for no-grid galleries. The old code used `image_index - 1`, which is only correct when `index_of_first_image == 1`; no-grid results use `index_of_first_image == 0`, so the first saved sample was exposed to filename generation as batch index `-1`. The save path now derives `p.batch_index` from `image_index - p.index_of_first_image`, preserving existing grid behavior while correcting no-grid sample indexes.
- Added `tests/test_save_serialization_contract.py::test_save_files_save_all_without_grid_uses_sample_batch_indexes` to lock the no-grid save-all contract.
- Left `parameters_copypaste` import aliases and the `modules.generation_parameters_copypaste` module alias intact. They remain compatibility surfaces for old extensions/tests and are not safe dead-code removals.
- Left `processed_js_with_image_paths()` intact. Its `image_paths` list intentionally preserves one entry per returned image and carries `already_saved_as` for API consumers; focused tests already cover image/path alignment.
- Left `txt2img_upscale()` gallery placeholder `already_saved_as` handling intact. It preserves non-upscaled gallery file paths during replacement and is live UI behavior.
- No safe consolidation was found for `save_image()` filename/logging construction beyond the batch-index fix. Callback filename rewrites, atomic duplicate suffixing, metadata embedding, 4chan JPG replacement, txt sidecar paths, and image-saved callbacks are order-sensitive side-effect surfaces.

### Static/dynamic audit map notes
- UI save chain remains: browser selected gallery index -> `save_files()` chooses all images or selected non-grid image -> original gallery index selects matching infotext -> `parse_generation_parameters()` supplies seed/prompt -> `p.batch_index` is now relative to `index_of_first_image` -> `images.save_image()` writes image metadata, paths, txt sidecars, callbacks, and optional zip entries.
- API path chain remains: image save paths are attached to PIL images as `already_saved_as` in `images.save_image()` or preserved by UI upscale placeholders -> `processed_js_with_image_paths()` injects a same-length `image_paths` list into txt2img/img2img API `info` JSON.
- Gallery/infotext chain remains conservative: `update_generation_info()` uses the selected gallery index directly against `infotexts`; `txt2img_upscale()` validates gallery index, splices replacement infotexts, and updates displayed info; img2img batch limits images and infotexts together.
- Compatibility surfaces kept conservative: UI selected-index JavaScript, Gradio `File.update()` return shape, save CSV field order, image metadata keys/sections, `already_saved_as`, API response `info` JSON shape, legacy infotext module alias, extension callbacks, and path exposure behavior.

### Validation log
- `PYTHONPYCACHEPREFIX=$(mktemp -d) python3 -m py_compile modules/ui_common.py modules/images.py modules/infotext_utils.py modules/txt2img.py modules/img2img.py modules/api/api.py tests/test_save_serialization_contract.py` - passed.
- `PYTHONPYCACHEPREFIX=$(mktemp -d) python3 -m pytest -q tests/test_save_serialization_contract.py test/test_images_save.py test/test_infotext_paste_bindings.py` - passed: 9 passed, with the existing pytest config warning `Unknown config option: base_url`.
- Reference scan confirmed tracked compatibility/path/index surfaces remain limited to the expected UI/API/save/upscale callers: `parameters_copypaste`, `modules.generation_parameters_copypaste`, `index_of_first_image`, `already_saved_as`, and `processed_js_with_image_paths`.
- Exact AST duplicate function/class scan across `modules/ui_common.py`, `modules/images.py`, `modules/infotext_utils.py`, `modules/txt2img.py`, `modules/img2img.py`, and `modules/api/api.py` reported no exact duplicates.
- `git diff --check` - passed after trimming the ledger EOF blank line.
- Live WebUI/API server startup, browser save/download clicks, real image generation, real zip downloads, and real PNG/JPEG/WebP/AVIF metadata round-trips were not exercised in this bounded slice.

### Next unchecked scope
- More slices are still needed. Recommended next slice: continue from save/download/gallery consumers into temp-file/gallery serving and UI media exposure paths, especially `modules/ui_tempdir.py`, `modules/ui_common.py` open-folder/temp path handling, Gradio gallery serialization, API/base64 image encode/decode helpers, and any tests around temporary file registration/path disclosure, looking for stale wrappers or duplicate path normalization while preserving file exposure and extension compatibility.

## Pass 100 - Temp-file/gallery serving and UI media exposure paths (2026-06-20)

### Scope checked
- `modules/ui_tempdir.py`: `Savedfile`, `register_tmp_file()`, `check_tmp_file()`, `save_pil_to_file()`, `install_ui_tempdir_override()`, `on_tmpdir_changed()`, `cleanup_tmpdr()`, and `is_ui_temp_path()`.
- `modules/ui_common.py`: output-panel `open_folder()` selected-gallery path handling, temp-path filtering via `ui_tempdir.is_ui_temp_path()`, save/download gallery inputs, and paste button gallery source registration.
- `modules/ui_component_patches.py`: Gradio component patch installation and `ui_tempdir.install_ui_tempdir_override()` call site.
- `modules/infotext_utils.py`: `image_from_url_text()` file-dict handling, temp-file allow-list check, gallery paste decode path, and PNG data-URI fallback path.
- `modules/api/api.py` and `modules/api/models.py`: `decode_base64_to_image()`, `decode_extras_batch_images()`, `encode_pil_to_base64()`, `processed_js_with_image_paths()`, txt2img/img2img/extras/pnginfo/progress/interrogate base64 callers, and public base64 image request/response models.
- Adjacent consumers/tests: `modules/progress.py`, `modules/txt2img.py`, `modules/ui_extra_networks.py`, `tests/test_save_serialization_contract.py`, `test/test_api_info_interrogate_admin_contract.py`, and `tests/test_api_progress_contract.py`.

### Findings / fixes
- No code changes were made. The inspected wrappers and branches are live compatibility or security surfaces, and no safe dead-code removal or helper consolidation was found in this bounded slice.
- Preserved both `temp_file_sets` and `temp_dirs` branches in `register_tmp_file()`/`check_tmp_file()`. They cover different Gradio compatibility shapes and feed the security check in `image_from_url_text()`.
- Preserved registration of both the raw `already_saved_as` path and the `?mtime` suffixed path in `save_pil_to_file()`. Gradio gallery/file dictionaries can carry the suffixed name back into `image_from_url_text()`, so collapsing to only one registration would risk rejecting valid in-memory gallery images.
- Preserved `is_ui_temp_path()` behavior for `shared.opts.temp_dir`, `GRADIO_TEMP_DIR`, and the default system `gradio` temp directory. `ui_common.open_folder()` relies on this distinction to avoid switching the open-folder target to UI temp paths unless configured.
- Preserved separate API and infotext/gallery base64 decode helpers. API decoding accepts remote URLs, generic `data:image/*` payloads, and raises HTTP errors; infotext/gallery paste decoding accepts Gradio file dictionaries and PNG data URIs guarded by the UI temp allow-list. They are not safe duplicates.
- Preserved API encode behavior in `encode_pil_to_base64()`: it reflects `opts.samples_format`, image metadata, JPEG/WebP conversion, and public response payload contracts.

### Static/dynamic audit map notes
- UI temp-file chain remains: `ui_component_patches` installs `save_pil_to_file()` as Gradio's PIL temp writer -> existing saved images are registered with and without the mtime query suffix -> `infotext_utils.image_from_url_text()` accepts only file dictionaries found in Gradio temp allow-lists before reading from disk.
- Output-folder chain remains: selected gallery image name is stripped of any query suffix for directory selection -> `is_ui_temp_path()` prevents UI temp directories from replacing the configured output directory unless the open-dir option explicitly allows temp paths.
- API media chain remains: request image strings flow through `decode_base64_to_image()` for txt2img/img2img masks/init images, extras, pnginfo, and interrogate; response PIL images flow through `encode_pil_to_base64()` for generation, extras, and progress current-image payloads; `processed_js_with_image_paths()` separately reports saved file paths in API `info` JSON.
- Compatibility surfaces kept conservative: legacy Gradio temp allow-list attributes, mtime-suffixed gallery names, `GRADIO_TEMP_DIR`, `already_saved_as`, selected-gallery JS/index handoff, API HTTP/local-request guards, response byte/base64 shape, and extension-facing paste/gallery behavior.

### Validation log
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pycompile-pass100.XXXXXX) python3 -m py_compile modules/ui_tempdir.py modules/ui_common.py modules/ui_component_patches.py modules/infotext_utils.py modules/api/api.py modules/api/models.py modules/progress.py modules/txt2img.py modules/ui_extra_networks.py tests/test_save_serialization_contract.py test/test_api_info_interrogate_admin_contract.py tests/test_api_progress_contract.py` - passed.
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pytest-pass100.XXXXXX) python3 -m pytest -q tests/test_save_serialization_contract.py test/test_api_info_interrogate_admin_contract.py tests/test_api_progress_contract.py` - passed: 8 passed, with the existing pytest config warning `Unknown config option: base_url`.
- Exact AST duplicate function scan across `modules/ui_tempdir.py`, `modules/ui_common.py`, `modules/api/api.py`, `modules/infotext_utils.py`, `modules/ui_component_patches.py`, `tests/test_save_serialization_contract.py`, `test/test_api_info_interrogate_admin_contract.py`, and `tests/test_api_progress_contract.py` reported no duplicate function bodies.
- Reference scan confirmed the temp/media helpers remain limited to the expected UI temp override, infotext/gallery paste, output-folder, API generation/extras/pnginfo/progress/interrogate, and focused test callers.
- `git diff --check` - passed.
- Live WebUI/API server startup, browser gallery open-folder clicks, actual Gradio media serving, remote URL API image fetches, and real temp-file cleanup were not exercised in this bounded slice.

### Next unchecked scope
- More slices are still needed. Recommended next slice: continue from media exposure into remaining server/static-file and path-safety surfaces, especially `modules/server.py`, `modules/ui.py` static route setup, `modules/ui_html_extensions.py`, `modules/util.py` path/open helpers, `modules/safe.py`, and adjacent launch/API route wiring, looking for stale path-disclosure wrappers or duplicate local/remote resource guards while preserving extension compatibility and UI/API serving behavior.
## Pass 101 - Server/static-file and path-safety surfaces (2026-06-20)

### Scope checked
- `modules/server.py`: confirmed absent in this GB10 `latest` checkout; server/API wiring is in `webui.py`, `modules/api/api.py`, `modules/ui.py`, and `modules/launch_utils.py`.
- `webui.py`: `create_api()`, `api_only()`, `webui()` headless-only guard, `FastAPI()` construction, middleware setup, API callback launch flow, `root_path`/port/server-name handoff.
- `modules/api/api.py`: `api_middleware()`, API route registration in `Api.__init__()`, `Api.launch()`, HTTP exception handling, request timing/log middleware, API base64 local/remote image guards in the adjacent decode/encode path.
- `modules/ui.py`: MIME registrations, headless UI helper overrides, ngrok connection hook, `setup_ui_api()`, internal routes, and `/webui-assets` static mount setup.
- `modules/ui_html_extensions.py`: `webpath()`, `javascript_html()`, `css_html()`, `reload_javascript()`, `UITemplateResponseOriginal` preservation, script/CSS injection, extension JS/MJS/style discovery, and `file=` cache-busted resource URLs.
- `modules/util.py`: `truncate_path()`, `html_path()`, `html()`, `walk_files()`, `open_folder()`, `load_file_from_url()` path construction, and `MassFileLister` file metadata helpers.
- `modules/safe.py`: `RestrictedUnpickler`, zip member allow-list validation, `check_pt()`, `load()`, `load_with_extra()`, `Extra`, and the `torch.load` monkeypatch surface.
- Adjacent route/path tests and launch wiring: `tests/test_api_server_control_contract.py`, `test/test_safe.py`, `modules/cmd_args.py` `--gradio-allowed-path` compatibility option, `modules/initialize_util.py` TLS/server helper checks, and `modules/ui_common.py` `open_folder()` selected-gallery path handoff.

### Findings / fixes
- No source code changes were made. The inspected server/static/path-safety wrappers are live serving, compatibility, or security surfaces, and no safe dead-code removal or helper consolidation was found in this bounded slice.
- Preserved `modules/ui_html_extensions.webpath()` and its `file={util.truncate_path(...)}?mtime` shape. It is the active script/CSS/profiling resource URL path used by `reload_javascript()` and `modules/profiling.py`, and changing it would affect Gradio static file serving/cache busting.
- Preserved separate `util.truncate_path()` and path-parent logic elsewhere. `truncate_path()` is presentation/resource-URL oriented and intentionally falls back to absolute paths outside the repo; security-sensitive model/extra-network parent checks use `commonpath` locally at their trust boundaries and are not safe duplicates to merge in this slice.
- Preserved `util.open_folder()` and `ui_common.create_output_panel()`'s selected-gallery open-folder wrapper. The UI wrapper filters temp gallery paths and honors `hide_ui_dir_config`; the utility wrapper performs OS-specific folder opening and non-directory warnings.
- Preserved `modules/ui.setup_ui_api()` `/webui-assets` mount even though browser UI startup is disabled by `webui.webui()`. `setup_ui_api()` remains an extension/headless UI compatibility surface and internal route registration hook when UI construction is exercised by tests or external callers.
- Preserved `webui.create_api()` as a small launch indirection. `api_only()` and `modules.launch_utils.start()` both depend on the current module-level API construction pattern, and tests load API control code by method names.
- Preserved `modules.safe` helper layers and the `Extra` context manager. `load()`, `load_with_extra()`, `unsafe_torch_load`, and `torch.load = load` are extension-facing checkpoint load surfaces, while zip filename validation is a distinct model safety boundary.
- Preserved `--gradio-allowed-path` even though no direct current source caller was found in this headless fork. It is documented as a legacy UI compatibility CLI option with a default `data_path`, and removing it would be a public argument break rather than safe dead code.

### Static/dynamic audit map notes
- API-only launch chain remains: `launch_utils.start()` -> `webui.api_only()` -> `initialize.initialize()` -> `FastAPI()` -> `initialize_util.setup_middleware()` -> `webui.create_api()` -> `Api.__init__()` route registration/middleware -> callbacks -> `Api.launch()` -> `uvicorn.run()` with server, port, root path, keep-alive, and TLS options.
- Static UI asset chain remains: `ui.setup_ui_api()` adds internal helper routes and mounts `/webui-assets`; `ui_html_extensions.reload_javascript()` injects cache-busted `file=` URLs for core script, extension JavaScript/MJS, extension CSS, `user.css`, and theme background CSS.
- Local open-folder chain remains: output-panel JS supplies selected gallery index -> `ui_common.open_folder()` chooses output or non-temp selected image directory -> `util.open_folder()` validates existence/directory and dispatches to the platform file manager.
- Safe-load chain remains: module import stores original `torch.load` in `unsafe_torch_load` -> monkeypatches `torch.load` to `safe.load()` -> `safe.load_with_extra()` runs restricted pickle/zip validation unless disabled -> calls original torch loader with legacy `weights_only=False` default.
- Compatibility/security surfaces kept conservative: extension JS/CSS discovery, `file=` URL semantics, Gradio template response monkeypatching, `shared.UITemplateResponseOriginal`, legacy browser UI/internal routes, `--gradio-allowed-path`, OS-specific open-folder behavior, TLS/root-path launch options, API error shape, API logging headers, and extension-safe unpickle customization.

### Validation log
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pycompile-pass101.XXXXXX) python3 -m py_compile modules/ui.py modules/ui_html_extensions.py modules/util.py modules/safe.py modules/api/api.py webui.py modules/launch_utils.py tests/test_api_server_control_contract.py test/test_safe.py` - passed.
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pytest-pass101.XXXXXX) python3 -m pytest -q tests/test_api_server_control_contract.py test/test_safe.py` - passed: 8 passed, with the existing pytest config warning `Unknown config option: base_url`.
- Reference scan confirmed expected server/static/path-safety callers for `webpath`, `reload_javascript`, `truncate_path`, `open_folder`, `setup_ui_api`, `create_api`, `api_middleware`, `safe.load_with_extra`, and `unsafe_torch_load`; no tracked `modules/server.py` exists in this checkout.
- Exact AST duplicate function/class scan across `modules/ui.py`, `modules/ui_html_extensions.py`, `modules/util.py`, `modules/safe.py`, `modules/api/api.py`, `webui.py`, and `modules/launch_utils.py` reported only empty/pass-style body collisions (`setup_progressbar`, tiny fallback context-manager methods, and API stub classes), not actionable duplicate logic.
- `git diff --check` - passed before the ledger edit and will be re-run before commit.
- Live WebUI startup, browser asset fetching, real `/webui-assets` HTTP requests, real Gradio `file=` resource serving, OS file-manager launch, and malicious checkpoint loading were not exercised in this bounded slice.

### Next unchecked scope
- More slices are still needed. Recommended next slice: continue from server/static/path-safety into model/path parent-check and metadata exposure helpers, especially `modules/sd_models.py`, `modules/ui_extra_networks.py`, `modules/ui_extra_networks_user_metadata.py`, `modules/modelloader.py`, and adjacent tests, looking for duplicate parent-path checks or stale metadata path wrappers while preserving checkpoint discovery, preview serving, extension metadata, and traversal protections.

## Pass 102 - Model/path parent-check and metadata exposure helpers (2026-06-20)

### Scope checked
- `modules/sd_models.py`: `path_is_parent` checkpoint root classification, `CheckpointInfo.__init__()`, safetensors metadata read/cache path, checkpoint alias registration, `list_models()`, `read_metadata_from_safetensors()`, checkpoint load path dispatch, and adjacent checkpoint contract tests.
- `modules/ui_extra_networks.py`: preview extension allow-list helpers, preview/cover/metadata/single-card API routes, extra-network item rendering, search/local path derivation, tree/dirs/card HTML generation, preview/embedded-preview/description discovery, save-preview parent checks, and extra-network path contract tests.
- `modules/ui_extra_networks_user_metadata.py`: user metadata editor path display, metadata JSON write path, preview refresh/save behavior, and relative path handling through the extra-network parent check.
- `modules/modelloader.py`: `load_models()`, model URL/download fallback, command path vs model path discovery, extension allow/blacklist filtering, broken symlink skip, duplicate output suppression, `friendly_name()`, and upscaler model loader callers.
- Adjacent callers/tests: `modules/ui_extra_networks_checkpoints.py`, `modules/ui_extra_networks_textual_inversion.py`, `modules/ui_extra_networks_hypernets.py`, `extensions-builtin/Lora/ui_extra_networks_lora.py`, `modules/upscaler.py`, model-specific loader callers, `tests/test_sd_models_checkpoint_info_contract.py`, `tests/test_extra_networks_metadata_contract.py`, and `tests/test_extra_networks_path_contract.py`.

### Findings / fixes
- Consolidated the duplicated `commonpath` parent-check implementation from `modules/sd_models.py` and `modules/ui_extra_networks.py` into shared `modules.util.path_is_parent()`.
- Kept `sd_models.path_is_parent` and `ui_extra_networks.path_is_parent` as compatibility aliases to preserve existing internal callers, tests, and possible extension imports while removing the duplicated implementation body.
- Updated `util.truncate_path()` to use the shared helper, removing its separate inline `commonpath` check and preserving the same fallback-to-absolute behavior for paths outside the base or on incompatible path roots.
- Updated focused path contract tests to assert the shared helper/alias shape and continue proving sibling-prefix rejection without importing full WebUI modules that parse CLI arguments or require optional runtime dependencies.
- No stale preview, cover-image, metadata, user-metadata, or model-loader branches were removed. The inspected branches remain live UI/API/extension compatibility or security behavior.

### Static/dynamic audit map notes
- Checkpoint discovery chain remains: `sd_models.list_models()` calls `modelloader.load_models()` with checkpoint extensions and VAE blacklists -> `CheckpointInfo.__init__()` classifies names relative to `--ckpt-dir` or the default model root through the shared parent helper -> metadata/hash/cache keys keep the same relative-name semantics.
- Extra-network preview chain remains: pages expose their preview roots -> `register_page()` populates global allowed dirs -> `/sd_extra_networks/thumb` enforces allowed directories and preview extensions -> card/user metadata code uses `find_preview()` and `link_preview()` URLs with lister mtime cache busting.
- Embedded metadata chain remains: LoRA cards can use `find_embedded_preview()` when safetensors metadata has `ssmd_cover_images`; `/sd_extra_networks/cover-images` still filters missing/non-list/negative indexes and decodes only selected cover image payloads.
- User metadata chain remains: editor relative-path display uses `ui_extra_networks.path_is_parent` compatibility alias; save writes `<model basename>.json` beside the item and updates the page lister cache; preview replacement saves through `item["local_preview"]` and refreshes the single card.
- Model loader chain remains: command path is searched before default model path, legacy `experiments/pretrained_models` is honored, broken symlinks are skipped, extension blacklists are applied after allow-list filtering, duplicates are suppressed, and URL/download fallback only occurs when no local output exists.
- Compatibility/security surfaces kept conservative: external imports of `sd_models.path_is_parent`/`ui_extra_networks.path_is_parent`, checkpoint title/name semantics, extra-network route names and response shapes, user metadata file placement, preview extension allow-list, hidden-directory handling, embedded cover-image metadata, command-line model path discovery, and upscaler/model loader behavior.

### Validation log
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pycompile-pass102-final.XXXXXX) python3 -m py_compile modules/sd_models.py modules/ui_extra_networks.py modules/ui_extra_networks_user_metadata.py modules/modelloader.py modules/util.py modules/ui_extra_networks_checkpoints.py modules/ui_extra_networks_textual_inversion.py modules/ui_extra_networks_hypernets.py extensions-builtin/Lora/ui_extra_networks_lora.py tests/test_sd_models_checkpoint_info_contract.py tests/test_extra_networks_metadata_contract.py tests/test_extra_networks_path_contract.py` - passed.
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pytest-pass102-fix.XXXXXX) python3 -m pytest -q tests/test_sd_models_checkpoint_info_contract.py tests/test_extra_networks_metadata_contract.py tests/test_extra_networks_path_contract.py` - passed: 7 passed, with the existing pytest config warning `Unknown config option: base_url`.
- Exact AST duplicate function/class scan across the pass 102 modules/tests reported only abstract/no-op method bodies (`refresh`, `create_extra_default_items_in_left_column`, small fallback `tqdm` methods, `create_item`, `list_items`), not duplicated model/path or metadata logic.
- Reference scan confirmed the shared parent helper feeds `sd_models` checkpoint classification, extra-network relative/search/local path handling, user metadata display, and save-preview allow checks; model loader references remain expected model/upscaler callers.
- `git diff --check` - passed.
- Live WebUI/API startup, actual checkpoint discovery against full model directories, real `/sd_extra_networks/*` HTTP requests, browser preview serving, safetensors cover-image rendering, and real user metadata writes were not exercised in this bounded slice.

### Next unchecked scope
- More slices are still needed. Recommended next slice: continue from model/path metadata helpers into checkpoint/extra-network page specializations and model-listing API surfaces, especially `modules/ui_extra_networks_checkpoints.py`, `modules/ui_extra_networks_textual_inversion.py`, `modules/ui_extra_networks_hypernets.py`, `extensions-builtin/Lora/ui_extra_networks_lora.py`, `modules/api/api.py` model metadata listing methods, and adjacent response-model tests, looking for duplicate item-shape construction or stale metadata exposure wrappers while preserving extension compatibility and public API response shapes.


## Pass 103 - Checkpoint/extra-network page specializations and model-listing API surfaces (2026-06-20)

### Scope checked
- `modules/ui_extra_networks_checkpoints.py`: `ExtraNetworksPageCheckpoints.__init__()`, `refresh()`, `create_item()`, `list_items()`, preview directory allow-list, and user metadata editor creation.
- `modules/ui_extra_networks_textual_inversion.py`: `ExtraNetworksPageTextualInversion.__init__()`, embedding refresh, `create_item()`, `list_items()`, and embedding preview directory allow-list.
- `modules/ui_extra_networks_hypernets.py`: `ExtraNetworksPageHypernetworks.__init__()`, refresh, hash/search/prompt item construction, `list_items()`, and preview directory allow-list.
- `extensions-builtin/Lora/ui_extra_networks_lora.py`: `ExtraNetworksPageLora.__init__()`, network refresh, base card construction, embedded preview fallback, user metadata prompt/negative prompt handling, SD-version filtering, `list_items()`, preview directory allow-list, and metadata editor creation.
- `modules/api/api.py`: `Api.get_sd_models()`, `get_sd_vaes()`, `get_hypernetworks()`, `get_face_restorers()`, `get_realesrgan_models()`, `get_prompt_styles()`, `get_embeddings()`, and adjacent metadata-listing helper surfaces.
- Adjacent tests: `tests/test_api_listing_contract.py`, `tests/test_api_server_control_contract.py` response-model stubs/import surface, and `tests/test_extra_networks_metadata_contract.py` metadata/path behavior assertions.

### Findings / fixes
- Consolidated repeated API mapping-to-list wrappers into `Api._items_from_mapping()` and reused it for SD VAE, hypernetwork, and prompt-style listings while preserving the exact public response dictionaries and insertion-order behavior.
- Lifted embedding response conversion out of nested per-call helpers into `Api._embedding_item()` and `Api._embedding_items()`, preserving the loaded/skipped map shape and field names.
- Updated the focused AST-based API listing contract loader to include the new helper methods so the isolated method subset still exercises the real helper path.
- Preserved checkpoint, textual inversion, hypernetwork, and LoRA extra-network `create_item()` implementations. They share a broad card-item shape but differ in public prompt insertion behavior, local preview naming, metadata/user-metadata handling, embedded preview fallback, hash sources, SD-version filtering, and extension-facing editor hooks, so further consolidation would risk behavior churn.
- Preserved API listing methods whose item sources or response fields are materially distinct (`get_sd_models()`, face restorers, RealESRGAN models). No stale metadata exposure wrapper or safe dead endpoint was found in this bounded slice.

### Static/dynamic audit map notes
- Extra-network page item chains remain separate by model type: checkpoints use checkpoint aliases and disable prompt insertion; textual inversion inserts embedding names and allows negative prompts; hypernetworks build weighted `<hypernet:...>` prompts with hash-cache metadata; LoRA cards combine alias prompts, user activation/negative text, embedded previews, metadata, and SD-version compatibility filtering.
- API metadata listing chain remains: route registration binds public `/sdapi/v1/*` endpoints to response models; model/vae/hypernetwork/style helpers shape dictionaries for those response models without changing discovery sources or keys.
- Embedding listing chain remains: `sd_hijack.model_hijack.embedding_db` supplies loaded and skipped maps; response values still expose `step`, `sd_checkpoint`, `sd_checkpoint_name`, `shape`, and `vectors` only.
- Compatibility surfaces kept conservative: extra-network card keys, `onclick` checkpoint selection, prompt JavaScript quoting, local preview filenames, user metadata lookup, LoRA embedded cover metadata, LoRA SD-version filtering, API response keys, response-model bindings, import-time API method names, and extension-facing page classes.

### Validation log
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pycompile-pass103.XXXXXX) python3 -m py_compile modules/ui_extra_networks_checkpoints.py modules/ui_extra_networks_textual_inversion.py modules/ui_extra_networks_hypernets.py extensions-builtin/Lora/ui_extra_networks_lora.py modules/api/api.py tests/test_api_listing_contract.py tests/test_api_server_control_contract.py tests/test_extra_networks_metadata_contract.py` - passed.
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pytest-pass103.XXXXXX) python3 -m pytest -q tests/test_api_listing_contract.py tests/test_extra_networks_metadata_contract.py` - passed: 10 passed, with the existing pytest config warning `Unknown config option: base_url`.
- Exact AST duplicate function/class scan across the pass 103 source files reported no repeated function/class bodies after the helper extraction.
- `git diff --check` - passed before commit.
- Live WebUI/API startup, real `/sdapi/v1/*` HTTP calls, browser extra-network card rendering, actual model directory discovery, LoRA embedded preview rendering, and user metadata editing were not exercised in this bounded slice.

### Next unchecked scope
- More slices are still needed. Recommended next slice: continue from API model metadata listings into API refresh/create/train model-management endpoints and their adjacent request/response helpers, especially `refresh_checkpoints()`, `unloadapi()`, `reloadapi()`, `create_embedding()`, `create_hypernetwork()`, `train_embedding()`, `train_hypernetwork()`, `refresh_embeddings()`, `refresh_loras()`, `get_memory()`, and adjacent training/server-control contract tests, looking for stale task wrappers or duplicate create/train response handling while preserving public API behavior and runtime queue semantics.


## Pass 104 - API refresh/create/train model-management endpoints (2026-06-20)

### Scope checked
- `modules/api/api.py`: `Api.refresh_checkpoints()`, `unloadapi()`, `reloadapi()`, `create_embedding()`, `create_hypernetwork()`, `train_embedding()`, `train_hypernetwork()`, `refresh_embeddings()`, `refresh_vae()`, `get_memory()`, shared create/train helpers, response helpers, queue-lock wrapper, and adjacent route registrations.
- `extensions-builtin/Lora/scripts/lora_script.py`: extension-owned `/sdapi/v1/refresh-loras` route and `refresh_loras()` callback shape.
- Adjacent tests: `tests/test_api_server_control_contract.py` and `tests/test_api_training_contract.py` covering refresh queue locking, reload/unload response shape, memory response shape, create/train helper dispatch, hypernetwork training device hooks, and state-end behavior.

### Findings / fixes
- Consolidated repeated CUDA memory current/peak bucket shaping in `Api.get_memory()` into `Api._memory_counter_pair()`. The public `MemoryResponse` shape is unchanged; `allocated`, `reserved`, `active`, and `inactive` still expose `current` and `peak` using the same torch memory-stat keys.
- Updated the focused AST-based API server-control contract loader to include `_memory_counter_pair()` so `get_memory()` continues to execute through the real helper path in isolated tests.
- Preserved `refresh_embeddings()`, `refresh_checkpoints()`, and `refresh_vae()` as explicit queue-locked endpoint methods. They target different reload functions and their route names are public API surfaces.
- Preserved `unloadapi()` and `reloadapi()` as separate public control endpoints with empty dict responses and distinct model side effects.
- Preserved `create_embedding()`, `create_hypernetwork()`, `train_embedding()`, and `train_hypernetwork()` wrappers. The shared helpers already centralize create/train response handling; endpoint wrappers still carry distinct job names, success/error text, post-create embedding reload, and hypernetwork training device hooks.
- Preserved the LoRA extension `refresh_loras()` route outside `Api`; it is registered by the extension app-start callback and returns the extension network refresh result directly.

### Static/dynamic audit map notes
- Refresh/control chain remains: route registration -> endpoint method -> `_call_with_queue_lock()` for refresh endpoints or direct model side effect for unload/reload -> existing empty or extension-specific response shape.
- Create chain remains: route registration -> create endpoint wrapper -> `_run_create_task()` -> `shared.state.begin()` -> create function -> optional embedding DB reload -> `CreateResponse(info=...)` -> `shared.state.end()`.
- Training chain remains: route registration -> train endpoint wrapper -> `_run_training_task()` -> optional hypernetwork pre-hook -> xattention optimization toggle handling -> train function -> optional hypernetwork device restore -> `TrainResponse(info=...)` -> one `shared.state.end()`.
- Memory chain remains: psutil process RSS totals for RAM; torch CUDA availability check, `mem_get_info()`, `memory_stats(shared.device)`, bucket shaping, retry/OOM event shaping, and `MemoryResponse(ram=..., cuda=...)`.
- Compatibility surfaces kept conservative: public endpoint paths, response-model bindings, exact info string prefixes, queue-lock semantics, training state lifecycle, post-create embedding reload, hypernetwork device restore, LoRA extension callback route, and CUDA memory response keys.

### Validation log
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pycompile-pass104.XXXXXX) python3 -m py_compile modules/api/api.py tests/test_api_server_control_contract.py tests/test_api_training_contract.py extensions-builtin/Lora/scripts/lora_script.py` - passed.
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pytest-pass104.XXXXXX) python3 -m pytest -q tests/test_api_server_control_contract.py tests/test_api_training_contract.py` - passed: 10 passed, with the existing pytest config warning `Unknown config option: base_url`.
- Reference scan confirmed the targeted API methods, `_memory_counter_pair()`, and extension-owned `/sdapi/v1/refresh-loras` route are limited to the expected source/tests in this slice.
- Exact AST duplicate function/class scan across `modules/api/api.py`, `tests/test_api_server_control_contract.py`, `tests/test_api_training_contract.py`, and `extensions-builtin/Lora/scripts/lora_script.py` reported no duplicate function/class bodies.
- `git diff --check` - passed.
- Live WebUI/API startup, real HTTP calls to refresh/create/train/reload/unload/memory endpoints, actual model reloads, embedding/hypernetwork creation/training, CUDA memory probing on the running server process, and LoRA refresh through FastAPI were not exercised in this bounded slice.

### Next unchecked scope
- More slices are still needed. Recommended next slice: continue from API model-management endpoints into API script/default-argument and selectable-script request handling, especially `Api.init_default_script_args()`, `init_script_args()`, `text2imgapi()`, `img2imgapi()`, script arg override helpers, selectable always-on script handling, and adjacent API script contract tests, looking for stale compatibility branches or duplicated txt2img/img2img script argument plumbing while preserving public API request semantics and extension script behavior.


## Pass 105 - API script/default-argument and selectable-script request handling (2026-06-20)

### Scope checked
- `modules/api/api.py`: `Api.get_selectable_script()`, `get_script()`, `init_default_script_args()`, `persist_openclaw_denoise_ramp_args()`, `init_script_args()`, `apply_infotext()` script-argument mapping path, `_prepare_generation_api_request()`, `text2imgapi()`, and `img2imgapi()`.
- Adjacent API/script contract tests: `test/test_api_script_defaults.py`, `test/test_infotext_api_mappings.py`, and `tests/test_api_server_control_contract.py`.
- Related runtime script plumbing references in `modules/scripts.py`, `modules/txt2img.py`, `modules/img2img.py`, and `modules/processing.py` for selectable/always-on script argument semantics.

### Findings / fixes
- Consolidated the duplicated txt2img/img2img selectable-script execution branch into `Api._run_generation_with_scripts()`. The helper preserves the existing contract that selectable script runs receive the mutable script-args list while normal processing receives tuple-converted script args before `process_images(p)`.
- Updated `text2imgapi()` and `img2imgapi()` to call the shared helper while leaving each endpoint's task id, queue lock, `shared.state.begin()` job name, processing class, output paths, init-image decode/timing, response model, and cleanup lifecycle unchanged.
- Preserved `init_default_script_args()` as explicit default-vector bootstrap logic. Its Gradio `Blocks` context and `script_default_ui_values()` fallback remain compatibility-sensitive for extension scripts with finalized controls or UI-only defaults.
- Preserved `init_script_args()` always-on and selectable request handling. The infotext override path, OpenClaw Denoise Ramp default persistence, sparse script-arg extension, over-length always-on override map, selectable-in-always-on rejection, and non-list args validation are active compatibility paths rather than dead code.
- Preserved public request fields (`script_name`, `script_args`, `alwayson_scripts`, `infotext`) and pydantic v1/v2 compatibility branches in `apply_infotext()`.

### Static/dynamic audit map notes
- Generation request chain remains: API request -> infotext field/script-arg extraction -> selectable script lookup -> sampler/scheduler normalization -> processing constructor args -> default script-args copy plus infotext/selectable/always-on overlays -> endpoint-specific processing object -> shared script execution helper -> encoded response.
- Selectable script chain remains: `script_name` maps only against `script_runner.selectable_scripts`; position 0 stores one-based selectable index; selected script args replace only that script's range and are passed as a list to `ScriptRunner.run()`.
- Always-on script chain remains: names map against all `script_runner.scripts`; selectable scripts are rejected in always-on params; omitted `args` keeps defaults; supplied `args` must be a list and may extend the backing vector for extension scripts whose API payload reaches past captured `args_to`.
- Compatibility surfaces kept conservative: public route method names, request/response fields, default script arg vector shape, pydantic v1/v2 dump branch, infotext script-arg overlays, extension script controls, OpenClaw Denoise Ramp persistence, and the list-vs-tuple distinction between selectable script runs and ordinary processing.

### Validation log
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pycompile-pass105.XXXXXX) python3 -m py_compile modules/api/api.py test/test_api_script_defaults.py test/test_infotext_api_mappings.py tests/test_api_server_control_contract.py` - passed.
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pytest-pass105.XXXXXX) python3 -m pytest -q test/test_api_script_defaults.py test/test_infotext_api_mappings.py tests/test_api_server_control_contract.py` - passed: 20 passed, with the existing pytest config warning `Unknown config option: base_url`.
- Exact AST duplicate function/class scan across `modules/api/api.py`, `test/test_api_script_defaults.py`, `test/test_infotext_api_mappings.py`, and `tests/test_api_server_control_contract.py` reported no duplicate function/class bodies after the helper extraction.
- Reference scan confirmed the duplicate `scripts_txt2img.run()`/`scripts_img2img.run()` branch in `modules/api/api.py` was replaced by `_run_generation_with_scripts()` call sites; non-API UI generation paths in `modules/txt2img.py` and `modules/img2img.py` remain separate UI-facing code.
- `git diff --check` - passed.
- Live WebUI/API startup, real `/sdapi/v1/txt2img` or `/sdapi/v1/img2img` HTTP requests, actual selectable extension execution, always-on extension requests, and image generation were not exercised in this bounded slice.

### Next unchecked scope
- More slices are still needed. Recommended next slice: continue from API generation script handling into API image/base64 encode-decode, response shaping, and progress/interrogate/png-info utility endpoints, especially `decode_base64_to_image()`, `encode_pil_to_base64()`, `processed_js_with_image_paths()`, `Api.progressapi()`, `interrogateapi()`, `pnginfoapi()`, and adjacent API progress/info tests, looking for duplicated image serialization or stale response helpers while preserving public API formats and task/progress semantics.

## Pass 106 - API image/base64 encode-decode and utility response endpoints (2026-06-20)

### Scope checked
- `modules/api/api.py`: `decode_base64_to_image()`, `decode_extras_batch_images()`, `encode_pil_to_base64()`, `processed_js_with_image_paths()`, `Api.pnginfoapi()`, `Api.progressapi()`, `Api.interrogateapi()`, adjacent extras image encode/decode call sites, and route registrations for `/sdapi/v1/png-info`, `/sdapi/v1/progress`, and `/sdapi/v1/interrogate`.
- `modules/api/models.py`: `PNGInfoRequest`, `PNGInfoResponse`, `ProgressRequest`, `ProgressResponse`, `InterrogateRequest`, and `InterrogateResponse` response/request fields.
- `modules/progress.py`: UI progress/live-preview image serialization path, checked only as a nearby base64/progress reference point.
- Adjacent API/progress/info tests: `test/test_postprocessing_api_defaults.py`, `test/test_api_info_interrogate_admin_contract.py`, `tests/test_save_serialization_contract.py`, and `tests/test_api_progress_contract.py`.

### Findings / fixes
- No safe source-code remediation was made in this slice.
- Preserved `decode_base64_to_image()` as the single API ingress helper for URL, data-URI, and bare-base64 image inputs. The URL branch still enforces `api_enable_requests`, local-request forbidding, optional user-agent headers, and the existing `Invalid image url` / `Invalid encoded image` error split used by extras batch skip behavior.
- Preserved `encode_pil_to_base64()` as the single API image egress helper for generation, extras, and current-progress images. PNG metadata text preservation, JPEG/WEBP RGB conversion, geninfo-to-EXIF embedding, `jpeg_quality`, `webp_lossless`, invalid-format errors, and string passthrough are compatibility-sensitive public API behavior.
- Preserved `processed_js_with_image_paths()` as the generation response-info shaper. It injects saved image paths and OpenClaw cache/timing telemetry while keeping existing `processed.js()` fields aligned with `processed.images`.
- Preserved `pnginfoapi()` because PNGInfo still needs decode, geninfo fallback to empty string, metadata item return, generation-parameter parsing, and `infotext_pasted_callback()` side effects before shaping `PNGInfoResponse`.
- Preserved `progressapi()` response branches. The idle branch intentionally omits `current_task`; the active branch calculates ETA/progress, refreshes the current image, honors `skip_current_image`, serializes through `encode_pil_to_base64()`, and exposes the live `progress_module.current_task` reference.
- Preserved `interrogateapi()` model dispatch under `_call_with_queue_lock()`. The `clip` and `deepdanbooru` branches target distinct model backends, and the image-not-found/model-not-found status details are public API behavior.
- Preserved the nearby UI `modules/progress.py` live-preview base64 path as separate from API serialization: it returns `data:image/{format};base64,...` live-preview URLs and obeys live-preview-specific options, while API utility endpoints return raw base64 strings through `encode_pil_to_base64()`.

### Static/dynamic audit map notes
- Image ingress chain remains: API request field -> optional `.strip()` for PNGInfo -> `decode_base64_to_image()` -> `images.read()` -> endpoint-specific behavior.
- Image egress chain remains: processed/current/extras PIL image -> `encode_pil_to_base64()` -> response-model field; saved-path metadata is kept in `processed_js_with_image_paths()` rather than mixed into image payloads.
- Progress chain remains: idle state shortcut -> progress calculation via `progress_module.calculate_progress_and_eta()` -> current-image refresh/optional serialization -> `ProgressResponse` with `state`, `textinfo`, and active `current_task`.
- PNGInfo/interrogate chains remain endpoint-specific utility APIs with different side effects: PNGInfo parses and broadcasts pasted infotext parameters; interrogate converts decoded images to RGB and runs the requested interrogator under the runtime queue lock.
- Compatibility surfaces kept conservative: accepted base64/data-URI/HTTP(S) inputs, request-forbid options, error status/details, PNG metadata and EXIF behavior, response field names and omission/default semantics, progress task identity, response path injection, and queue-lock semantics.

### Validation log
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pycompile-pass106.XXXXXX) python3 -m py_compile modules/api/api.py modules/api/models.py modules/progress.py test/test_postprocessing_api_defaults.py test/test_api_info_interrogate_admin_contract.py tests/test_save_serialization_contract.py tests/test_api_progress_contract.py` - passed.
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pytest-pass106-focused.XXXXXX) python3 -m pytest -q test/test_postprocessing_api_defaults.py::test_api_progress_reports_live_current_task_reference test/test_postprocessing_api_defaults.py::test_extras_batch_decode_skips_corrupt_images_but_keeps_valid_items test/test_postprocessing_api_defaults.py::test_extras_single_response_allows_no_output_from_skipped_or_interrupted_run test/test_api_info_interrogate_admin_contract.py tests/test_save_serialization_contract.py::test_processed_js_with_image_paths_keeps_paths_aligned_to_images tests/test_api_progress_contract.py` - passed: 8 passed, with the existing pytest config warning `Unknown config option: base_url`.
- Broader focused test command `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pytest-pass106.XXXXXX) python3 -m pytest -q test/test_postprocessing_api_defaults.py test/test_api_info_interrogate_admin_contract.py tests/test_save_serialization_contract.py tests/test_api_progress_contract.py` was blocked by missing runtime dependencies in bare system Python: `fastapi` and `torch` import errors during the import-heavy postprocessing runner test; remaining tests in that run passed before/after the blocked failure.
- Exact reference scan confirmed API image encode/decode and utility response shaping call sites are limited to the expected API, model, progress, and focused test surfaces; the separate UI live-preview base64 path in `modules/progress.py` intentionally uses a different public format.
- Exact AST duplicate function/class scan across `modules/api/api.py`, `modules/api/models.py`, `modules/progress.py`, `test/test_postprocessing_api_defaults.py`, `test/test_api_info_interrogate_admin_contract.py`, `tests/test_save_serialization_contract.py`, and `tests/test_api_progress_contract.py` reported only pre-existing duplicated test scaffolding classes/context-manager methods, not production API utility duplication.
- `git diff --check` - passed.
- Live WebUI/API startup, real HTTP calls to `/sdapi/v1/png-info`, `/sdapi/v1/progress`, `/sdapi/v1/interrogate`, URL image fetches, actual CLIP/deepdanbooru interrogation, and generated-image metadata round trips were not exercised in this bounded slice.

### Next unchecked scope
- More slices are still needed. Recommended next slice: continue from API utility endpoints into API options/configuration, command/script metadata, and server-info endpoints, especially `get_config()`, `set_config()`, `get_cmd_flags()`, `get_samplers()`, `get_schedulers()`, `get_upscalers()`, `get_sd_models()`, `get_sd_vaes()`, `get_hypernetworks()`, `get_face_restorers()`, `get_realesrgan_models()`, `get_prompt_styles()`, `get_artist_categories()`, `get_artists()`, `get_scripts_list()`, `get_script_info()`, and adjacent API info/listing contract tests, looking for duplicated response dictionary shaping while preserving public keys and extension-visible behavior.

## Pass 107 - API options/configuration and server-info listing endpoints (2026-06-20)

### Scope checked
- `modules/api/api.py`: route registrations for `/sdapi/v1/options`, `/sdapi/v1/cmd-flags`, `/sdapi/v1/samplers`, `/sdapi/v1/schedulers`, `/sdapi/v1/upscalers`, `/sdapi/v1/sd-models`, `/sdapi/v1/sd-vae`, `/sdapi/v1/hypernetworks`, `/sdapi/v1/face-restorers`, `/sdapi/v1/realesrgan-models`, `/sdapi/v1/prompt-styles`, `/sdapi/v1/scripts`, and `/sdapi/v1/script-info`; `Api.get_config()`, `set_config()`, `get_cmd_flags()`, `get_samplers()`, `get_schedulers()`, `get_upscalers()`, `get_latent_upscale_modes()` as adjacent listing helper, `get_sd_models()`, `get_sd_vaes()`, `get_hypernetworks()`, `get_face_restorers()`, `get_realesrgan_models()`, `get_prompt_styles()`, `_script_lists()`, `get_scripts_list()`, and `get_script_info()`.
- `modules/api/models.py`: `OptionsModel`, `FlagsModel`, `SamplerItem`, `SchedulerItem`, `UpscalerItem`, `LatentUpscalerModeItem`, `SDModelItem`, `SDVaeItem`, `HypernetworkItem`, `FaceRestorerItem`, `RealesrganItem`, `PromptStyleItem`, `ScriptsList`, and `ScriptInfo` response contracts.
- Adjacent contract tests: `tests/test_api_listing_contract.py`, `tests/test_api_server_control_contract.py`, and live-style utility endpoint smoke list in `test/test_utils.py`.

### Findings / fixes
- No safe source-code remediation was made in this slice.
- Preserved `get_config()` as an explicit `shared.opts.data` walk. The `shared.opts.data_labels` fallback expression is currently redundant for keys already present in `data`, but changing it would risk API option default semantics and dynamic option exposure without a live-options regression target.
- Preserved `set_config()` because its checkpoint alias preflight, per-option `shared.opts.set(..., is_api=True)` mutation semantics, and config-file save side effect are public API behavior.
- Preserved the small listing methods rather than forcing a broader generic response-shaping abstraction. The endpoints expose different source registries and public keys, and previous helper coverage already centralizes the genuinely repeated mapping case for VAE, hypernetwork, and prompt-style mappings.
- Preserved `_script_lists()`, `get_scripts_list()`, and `get_script_info()` as the shared script listing source. They intentionally keep names and API metadata separate while using the current txt2img/img2img script runner state.
- The requested `get_artist_categories()` and `get_artists()` functions/routes do not exist in the current tree. A targeted scan only found the previous pass handoff text, so these appear to be stale upstream-surface names rather than live unchecked functions in this branch.

### Static/dynamic audit map notes
- Options chain remains: `GET /sdapi/v1/options` -> `shared.opts.data` plus current metadata presence checks -> `OptionsModel`; `POST /sdapi/v1/options` -> optional model checkpoint alias validation -> `shared.opts.set(..., is_api=True)` for every supplied key -> `shared.opts.save(shared.config_filename)`.
- Listing chains remain registry-specific: command flags from `vars(shared.cmd_opts)`, samplers from `sd_samplers.all_samplers`, schedulers from `sd_schedulers.schedulers`, upscalers from `shared.sd_upscalers`, checkpoints from `modules.sd_models.checkpoints_list`, VAEs from `modules.sd_vae.vae_dict`, hypernetworks from `shared.hypernetworks`, face restorers from `shared.face_restorers`, RealESRGAN models from `get_realesrgan_models(None)`, prompt styles from `shared.prompt_styles.styles`, and scripts from `scripts.scripts_txt2img/scripts_img2img`.
- Compatibility surfaces kept conservative: public response keys, route paths, response-model bindings, option mutation/save side effects, checkpoint alias validation, dynamic registry side effects, script API metadata objects, insertion order of registry-backed lists, and existing command flag exposure.

### Validation log
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pycompile-pass107.XXXXXX) python3 -m py_compile modules/api/api.py modules/api/models.py tests/test_api_listing_contract.py tests/test_api_server_control_contract.py test/test_utils.py` - passed.
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pytest-pass107.XXXXXX) python3 -m pytest -q tests/test_api_listing_contract.py tests/test_api_server_control_contract.py` - passed: 14 passed, with the existing pytest config warning `Unknown config option: base_url`.
- Exact artist endpoint scan `rg -n "get_artist_categories|get_artists|artist-categories|/sdapi/v1/artists|artist_categories" modules test tests .openclaw-audit/gb10-a1111-dead-code-duplication-audit.md --glob "*.py" --glob "*.md"` found only the pass 106 handoff text, confirming no live artist endpoint functions/routes in current source/tests.
- Exact AST duplicate function/class scan across `modules/api/api.py`, `modules/api/models.py`, `tests/test_api_listing_contract.py`, `tests/test_api_server_control_contract.py`, and `test/test_utils.py` reported no duplicate function/class bodies.
- `git diff --check` - passed.
- Live WebUI/API startup, real HTTP calls to the metadata listing endpoints, actual option writes, checkpoint selection, model registry refreshes, and extension script metadata discovery through a running FastAPI server were not exercised in this bounded slice.

### Next unchecked scope
- More slices are still needed. Recommended next slice: continue from API server-info/listing endpoints into API extension listing and runtime/custom OpenClaw status endpoints, especially `get_extensions_list()`, `get_sdpa_backend()`, `set_sdpa_backend()`, `get_cuda_graphs()`, `set_cuda_graphs()`, `get_openclaw_generation_diagnostics()`, `get_precision_map()`, their route registrations, and adjacent contract tests, looking for stale response shaping or duplicated status wrappers while preserving extension-visible metadata, runtime toggles, queue-lock semantics, and public OpenClaw API compatibility.

## Pass 108 - API extension listing and OpenClaw runtime status endpoints (2026-06-20)

### Scope checked
- `modules/api/api.py`: route registrations for `/sdapi/v1/extensions`, `/sdapi/v1/openclaw/sdpa-backend`, `/sdapi/v1/openclaw/cuda-graphs`, `/sdapi/v1/openclaw/generation-diagnostics`, `/sdapi/v1/openclaw/precision-map`, and `/sdapi/v1/precision-map`; `Api.get_extensions_list()`, `get_sdpa_backend()`, `set_sdpa_backend()`, `get_cuda_graphs()`, `set_cuda_graphs()`, `get_openclaw_generation_diagnostics()`, `get_precision_map()`, `apply_openclaw_runtime_defaults()`, and `_env_flag()`.
- `modules/api/models.py`: `ExtensionItem` response contract and adjacent public API model surface.
- Runtime helper modules adjacent to the route wrappers: `modules/sd_hijack_optimizations.py` SDPA backend status/set helpers, `modules/openclaw_cuda_graphs.py` status/toggle surface, and `modules/openclaw_generation_diagnostics.py` last-diagnostics fallback/delegation surface.
- Adjacent contract tests: `tests/test_api_listing_contract.py`, `tests/test_api_server_control_contract.py`, and `tests/test_api_extension_item_contract.py`.

### Findings / fixes
- No safe source-code remediation was made in this slice.
- Preserved `get_extensions_list()` as the public extension metadata endpoint. It intentionally refreshes extension discovery, calls `read_info_from_repo()` for all known extensions, and only returns remote-backed extension rows with `name`, `remote`, `branch`, `commit_hash`, `commit_date`, `version`, and `enabled`; the focused contract test covers this shape and confirms local-only extensions remain filtered out while still being read.
- Preserved the OpenClaw SDPA and CUDA graph methods as thin route wrappers. They are public route bindings over distinct runtime subsystems, and their fallback/coercion behavior is covered by focused contract tests.
- Preserved `get_openclaw_generation_diagnostics()` returning `{}` when no last generation diagnostics exist. That public empty-object fallback is distinct from the internal `None` sentinel returned by `last_generation_diagnostics()`.
- Preserved both precision-map routes and the direct `queue_lock` use in `get_precision_map()`. The implementation walks `shared.sd_model`, LoRA state, dtype flags, quantization stats, and skip-reason helpers; keeping it serialized with generation/model reload remains intentional.
- A broad earlier read appeared to show duplicate `result = {` and duplicate `if not self.default_script_arg_txt2img:` lines in `modules/api/api.py`, but targeted line-number reads and an exact scan confirmed the committed file has one occurrence of each. No edit was warranted.

### Static/dynamic audit map notes
- Extension listing chain remains: `/sdapi/v1/extensions` -> `extensions.list_extensions()` -> each extension `read_info_from_repo()` -> remote-only `ExtensionItem` dictionaries. Extension-visible git metadata fields are intentionally preserved even when values are `None`.
- Runtime status/control chain remains: `/sdapi/v1/openclaw/sdpa-backend` -> `sd_hijack_optimizations.sdpa_backend_status()` / `set_sdpa_backend()`; `/sdapi/v1/openclaw/cuda-graphs` -> `openclaw_cuda_graphs.status()` / `set_enabled(enabled, clear=clear)`; `/sdapi/v1/openclaw/generation-diagnostics` -> `openclaw_generation_diagnostics.last_generation_diagnostics() or {}`.
- Precision-map chain remains: OpenClaw namespaced route plus legacy alias -> `self.queue_lock` -> `build_precision_map()` -> model/LoRA/device/quantization inspection and cache update.
- Compatibility surfaces kept conservative: route paths/methods, extension metadata field names and remote-only filtering, SDPA backend request key `sdpa_backend`, CUDA graph request keys `enabled` and `clear`, diagnostics empty-object fallback, precision-map alias path, precision-map response fields, runtime default environment variables, and queue-lock semantics.

### Validation log
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pycompile-pass108.XXXXXX) python3 -m py_compile modules/api/api.py modules/api/models.py modules/openclaw_generation_diagnostics.py modules/openclaw_cuda_graphs.py modules/sd_hijack_optimizations.py tests/test_api_listing_contract.py tests/test_api_server_control_contract.py tests/test_api_extension_item_contract.py` - passed.
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pytest-pass108.XXXXXX) python3 -m pytest -q tests/test_api_listing_contract.py tests/test_api_server_control_contract.py tests/test_api_extension_item_contract.py` - passed: 15 passed, with the existing pytest config warning `Unknown config option: base_url`.
- Exact occurrence scan confirmed only one committed `result = {` in `build_precision_map()`, one `if not self.default_script_arg_txt2img:` guard in API initialization, the expected OpenClaw/precision-map route registrations, and the expected endpoint method definitions.
- Exact AST duplicate-body scan across `modules/api/api.py`, `modules/api/models.py`, `modules/openclaw_generation_diagnostics.py`, `modules/openclaw_cuda_graphs.py`, `modules/sd_hijack_optimizations.py`, `tests/test_api_listing_contract.py`, and `tests/test_api_server_control_contract.py` reported no duplicate nontrivial function bodies.
- `git diff --check` - passed.
- Live WebUI/API startup, real HTTP calls to the extension/OpenClaw runtime endpoints, actual extension git metadata reads against installed repositories, SDPA backend switching under a loaded model, CUDA graph toggling during generation, and live precision-map model walking were not exercised in this bounded slice.

### Next unchecked scope
- More slices are still needed. Recommended next slice: continue from API extension/runtime status endpoints into API create/train and refresh-adjacent lifecycle endpoints, especially `create_embedding()`, `create_hypernetwork()`, `train_embedding()`, `train_hypernetwork()`, `_run_create_task()`, `_run_training_task()`, `_create_response()`, `_train_response()`, route registrations for `/sdapi/v1/create/*` and `/sdapi/v1/train/*`, and adjacent API training/creation tests, looking for duplicated task wrappers or stale response helpers while preserving public response strings, queue/runtime side effects, and optimization restore semantics.


## Pass 109 - API create/train lifecycle endpoints (2026-06-20)

### Scope checked
- `modules/api/api.py`: route registrations for `/sdapi/v1/create/embedding`, `/sdapi/v1/create/hypernetwork`, `/sdapi/v1/train/embedding`, and `/sdapi/v1/train/hypernetwork`; `Api._create_response()`, `_train_response()`, `_run_create_task()`, `_run_training_task()`, `create_embedding()`, `create_hypernetwork()`, `_prepare_hypernetwork_training()`, `_restore_hypernetwork_training_devices()`, `train_embedding()`, and `train_hypernetwork()`.
- `modules/api/models.py`: `CreateResponse` and `TrainResponse` public response models.
- Adjacent training/creation contract tests: `tests/test_api_training_contract.py`, `tests/test_api_server_control_contract.py`, `tests/test_textual_inversion_preview_save_contract.py`, and `tests/test_hypernetwork_creation_contract.py`.
- Adjacent UI wrappers for comparison only: `modules/textual_inversion/ui.py` and `modules/hypernetworks/ui.py` training/create wrappers.

### Findings / fixes
- No safe source-code remediation was made in this slice.
- Preserved `_create_response()` and `_train_response()` as intentionally separate helpers. They currently wrap the same `info` field shape, but they bind distinct public response models and the focused AST contract tests assert create paths use only the create response helper while training paths use only the train response helper.
- Preserved `_run_create_task()` and `_run_training_task()` as separate lifecycle wrappers. Creation catches only `AssertionError`, reloads embeddings after embedding creation, and always ends `shared.state`; training has broader error capture, optional pre/post hooks, optimization disable/restore handling, and returns the historical `"... error: {error}"` training info string even when `error` is `None`.
- Preserved `train_embedding()` and `train_hypernetwork()` as thin public endpoint wrappers over `_run_training_task()`. The hypernetwork endpoint intentionally carries `train_hypernetwork` job identity, hypernetwork-specific public response prefixes, clears `shared.loaded_hypernetworks`, and restores first/condition stage model devices after training.
- Preserved apparent UI/API duplication. The UI wrappers in `modules/textual_inversion/ui.py` and `modules/hypernetworks/ui.py` share some optimization/device concepts, but their public outputs, exception behavior, Gradio update/html strings, and API response semantics differ enough that consolidation would risk compatibility.

### Static/dynamic audit map notes
- Create chain remains: API route -> endpoint wrapper -> `_run_create_task()` -> `shared.state.begin(job=...)` -> creation function -> optional post-create reload -> `CreateResponse(info=...)` -> `shared.state.end()`.
- Training chain remains: API route -> endpoint wrapper -> `_run_training_task()` -> `shared.state.begin(job=...)` -> optional pre-train hook -> optional `sd_hijack.undo_optimizations()` -> training function -> optional post-train hook -> optional `sd_hijack.apply_optimizations()` -> `TrainResponse(info=...)`.
- Compatibility surfaces kept conservative: public route paths/methods, response-model bindings, `info` strings, task job names, `AssertionError` create-task behavior, broad training exception capture, optimization restore semantics, hypernetwork loaded-list clearing, hypernetwork device restore, and embedding reload side effect.

### Validation log
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pycompile-pass109.XXXXXX) python3 -m py_compile modules/api/api.py modules/api/models.py tests/test_api_training_contract.py tests/test_api_server_control_contract.py tests/test_textual_inversion_preview_save_contract.py tests/test_hypernetwork_creation_contract.py` - passed.
- First focused pytest attempt used a stale node id, `tests/test_api_server_control_contract.py::test_api_route_table_includes_server_control_and_openclaw_endpoints`, and was blocked by pytest collection with `not found`; this was a command selection error, not a source failure.
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pytest-pass109-focused.XXXXXX) python3 -m pytest -q tests/test_api_training_contract.py tests/test_api_server_control_contract.py::test_server_control_routes_are_gated_by_api_server_stop tests/test_textual_inversion_preview_save_contract.py tests/test_hypernetwork_creation_contract.py` - passed: 7 passed, with the existing pytest config warning `Unknown config option: base_url`.
- Exact reference scan confirmed the create/train route registrations, response helpers, task wrappers, public response string prefixes, optimization toggles, hypernetwork loaded-list clearing, and hypernetwork device restore are limited to the expected API, UI, model, and focused test surfaces.
- Exact AST duplicate-body scan across `modules/api/api.py`, `modules/api/models.py`, `tests/test_api_training_contract.py`, `tests/test_api_server_control_contract.py`, `tests/test_textual_inversion_preview_save_contract.py`, and `tests/test_hypernetwork_creation_contract.py` reported no duplicate nontrivial function/class bodies.
- `git diff --check` - passed.
- Live WebUI/API startup, real HTTP calls to `/sdapi/v1/create/*` or `/sdapi/v1/train/*`, actual embedding/hypernetwork file creation, and real model training were not exercised in this bounded slice.

### Next unchecked scope
- More slices are still needed. Recommended next slice: continue from create/train lifecycle endpoints into API memory/checkpoint/server-control lifecycle endpoints and remaining low-frequency control routes, especially `get_memory()`, `_memory_counter_pair()`, `unloadapi()`, `reloadapi()`, `kill_webui()`, `restart_webui()`, `stop_webui()`, their route registrations, and adjacent server-control/memory contract tests, looking for stale response branches or duplicate lifecycle/status wrappers while preserving public empty responses, process command side effects, queue/runtime semantics, and memory counter compatibility.

## Pass 110 - API memory/checkpoint/server-control lifecycle endpoints (2026-06-20)

### Scope checked
- `modules/api/api.py`: route registrations for `/sdapi/v1/memory`, `/sdapi/v1/unload-checkpoint`, `/sdapi/v1/reload-checkpoint`, and gated `/sdapi/v1/server-kill`, `/sdapi/v1/server-restart`, `/sdapi/v1/server-stop`; `Api.get_memory()`, `_memory_counter_pair()`, `unloadapi()`, `reloadapi()`, `kill_webui()`, `restart_webui()`, and `stop_webui()`.
- `modules/api/models.py`: `MemoryResponse` public response contract.
- Adjacent lifecycle/control modules: `modules/restart.py`, `modules/shared_state.py`, and UI checkpoint lifecycle buttons in `modules/ui_settings.py` for comparison.
- Adjacent contract tests: `tests/test_api_server_control_contract.py`, especially memory shape, checkpoint empty-response, server-control side-effect, and route-gating coverage.

### Findings / fixes
- No safe source-code remediation was made in this slice.
- Preserved `unloadapi()` and `reloadapi()` as separate endpoint handlers. They intentionally target different checkpoint lifecycle operations (`sd_models.unload_model_weights()` versus `sd_models.send_model_to_device(shared.sd_model)`) while preserving the public empty-object API responses.
- Preserved `kill_webui()`, `restart_webui()`, and `stop_webui()` as distinct server-control endpoints. `kill_webui()` immediately delegates to `restart.stop_program()` and returns no body; `restart_webui()` conditionally triggers `restart.restart_program()` but still returns the historical 501 response; `stop_webui()` sets `shared.state.server_command = "stop"` and returns `Response("Stopping.")` for the caller-visible graceful stop path.
- Preserved the `shared.cmd_opts.api_server_stop` route gate. The focused test confirms the server-control routes are absent unless explicitly enabled.
- Preserved `_memory_counter_pair()` and the explicit CUDA counter extraction in `get_memory()`. The helper centralizes the `current`/`peak` compatibility shape for four PyTorch memory-stat counter families, and the endpoint keeps separate RAM, CUDA-unavailable, and CUDA-error branches that are public response behavior.

### Static/dynamic audit map notes
- Memory chain remains: `/sdapi/v1/memory` -> `psutil.Process(os.getpid())` RSS/percent-derived RAM totals -> optional `torch.cuda.mem_get_info()` system VRAM stats -> `torch.cuda.memory_stats(shared.device)` counter families -> `models.MemoryResponse(ram=..., cuda=...)`.
- Checkpoint lifecycle chain remains: `/sdapi/v1/unload-checkpoint` -> `sd_models.unload_model_weights()` -> `{}`; `/sdapi/v1/reload-checkpoint` -> `sd_models.send_model_to_device(shared.sd_model)` -> `{}`.
- Server-control chain remains: gated route registration by `--api-server-stop`; `/server-kill` -> `restart.stop_program()`; `/server-restart` -> `restart.is_restartable()` guard, optional `restart.restart_program()`, then 501 response; `/server-stop` -> `shared.state.server_command = "stop"` and a text response.
- Compatibility surfaces kept conservative: public route paths/methods, route gating, empty response bodies, text/status response quirks, process-exit side effects, restart marker side effects, shared-state server command signaling, memory response field names, CUDA event key names, and PyTorch memory counter `current`/`peak` shape.

### Validation log
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pycompile-pass110.XXXXXX) python3 -m py_compile modules/api/api.py modules/api/models.py modules/restart.py modules/shared_state.py tests/test_api_server_control_contract.py` - passed.
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pytest-pass110.XXXXXX) python3 -m pytest -q tests/test_api_server_control_contract.py` - passed: 7 passed, with the existing pytest config warning `Unknown config option: base_url`.
- Exact reference scan across `modules/api/api.py`, `modules/restart.py`, `modules/shared_state.py`, `tests/test_api_server_control_contract.py`, and `modules/ui_settings.py` confirmed the scoped memory counters, checkpoint lifecycle functions, server-control side effects, and response quirks are limited to expected API/UI/test surfaces.
- Exact AST duplicate-body scan across `modules/api/api.py`, `modules/api/models.py`, `tests/test_api_server_control_contract.py`, `modules/restart.py`, and `modules/shared_state.py` reported no duplicate nontrivial function/class bodies.
- `git diff --check` - passed.
- Live WebUI/API startup, real HTTP calls to `/sdapi/v1/memory`, `/sdapi/v1/unload-checkpoint`, `/sdapi/v1/reload-checkpoint`, or server-control endpoints, actual process kill/restart/stop behavior, and live CUDA memory readings under a running model were not exercised in this bounded slice.

### Next unchecked scope
- More slices are still needed. Recommended next slice: continue from memory/checkpoint/server-control lifecycle endpoints into API embedding refresh and remaining low-frequency model-list refresh/lifecycle helpers, especially `refresh_embeddings()`, `refresh_checkpoints()`, `refresh_vae()`, `get_embeddings()`, embedding response helpers, their route registrations, and adjacent embedding/refresh contract tests, looking for stale queue-lock wrappers or duplicate mapping helpers while preserving refresh side effects, queue/runtime semantics, and public embedding response shape.

## Pass 111 - API embedding refresh and low-frequency model-list refresh helpers (2026-06-20)

### Scope checked
- `modules/api/api.py`: route registrations for `/sdapi/v1/embeddings`, `/sdapi/v1/refresh-embeddings`, `/sdapi/v1/refresh-checkpoints`, and `/sdapi/v1/refresh-vae`; `Api.get_embeddings()`, `_embedding_item()`, `_embedding_items()`, `_call_with_queue_lock()`, `refresh_embeddings()`, `refresh_checkpoints()`, and `refresh_vae()`.
- `modules/api/models.py`: `EmbeddingItem` and `EmbeddingsResponse` public response models.
- Adjacent refresh/listing providers and side-effect targets: `sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings(force_reload=True)`, `shared.refresh_checkpoints`, `shared_items.refresh_checkpoints`, `shared_items.refresh_vae_list`, `sd_vae.refresh_vae_list`, and embedding reload calls from UI/model-load paths.
- Adjacent contract tests: `tests/test_api_listing_contract.py` embedding response map coverage and `tests/test_api_server_control_contract.py` refresh endpoint queue-lock/side-effect coverage.

### Findings / fixes
- No safe source-code remediation was made in this slice.
- Preserved `refresh_embeddings()`, `refresh_checkpoints()`, and `refresh_vae()` as explicit queue-locked endpoint methods. They are public API route handlers with distinct refresh targets and side effects, and prior helper consolidation already routes them through `_call_with_queue_lock()` without changing lock boundaries.
- Preserved `_embedding_item()` and `_embedding_items()` as the dedicated embedding response shaping helpers. They are used by both loaded and skipped embedding maps and keep the public `step`, `sd_checkpoint`, `sd_checkpoint_name`, `shape`, and `vectors` fields centralized.
- Preserved `get_embeddings()` returning separate `loaded` and `skipped` maps from `word_embeddings` and `skipped_embeddings`. Collapsing the maps or replacing the helper with a generic mapping serializer would alter the public response shape or obscure the current explicit contract tests.
- Preserved `shared.refresh_checkpoints` and `shared_items.refresh_vae_list` indirection at the API layer. The wrappers remain compatibility/list-refresh surfaces used by shared options, UI paths, and extension-facing imports.

### Static/dynamic audit map notes
- Embedding response chain remains: `/sdapi/v1/embeddings` -> `sd_hijack.model_hijack.embedding_db` -> `word_embeddings` and `skipped_embeddings` -> `_embedding_items()` -> public `EmbeddingsResponse` maps.
- Embedding refresh chain remains: `/sdapi/v1/refresh-embeddings` -> queue lock -> `embedding_db.load_textual_inversion_embeddings(force_reload=True)`.
- Checkpoint refresh chain remains: `/sdapi/v1/refresh-checkpoints` -> queue lock -> `shared.refresh_checkpoints()` -> `shared_items.refresh_checkpoints()` -> `sd_models.list_models()`.
- VAE refresh chain remains: `/sdapi/v1/refresh-vae` -> queue lock -> `shared_items.refresh_vae_list()` -> `sd_vae.refresh_vae_list()`.
- Compatibility surfaces kept conservative: route paths/methods, queue-lock semantics, refresh side effects, wrapper imports through `shared`/`shared_items`, embedding map names, embedding item field names, and loaded-vs-skipped separation.

### Validation log
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pycompile-pass111.XXXXXX) python3 -m py_compile modules/api/api.py modules/api/models.py modules/shared.py modules/shared_items.py modules/sd_vae.py tests/test_api_listing_contract.py tests/test_api_server_control_contract.py` - passed.
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pytest-pass111.XXXXXX) python3 -m pytest -q tests/test_api_listing_contract.py tests/test_api_server_control_contract.py::test_refresh_endpoints_keep_queue_lock_and_side_effect_targets` - passed: 8 passed, with the existing pytest config warning `Unknown config option: base_url`.
- Exact reference scan confirmed the refresh route registrations, endpoint wrappers, queue-lock helper use, embedding response helpers, embedding reload force-refresh calls, and lower-level checkpoint/VAE refresh wrappers are limited to expected API/UI/model-load/shared/test surfaces.
- Exact AST duplicate-body scan across `modules/api/api.py`, `modules/api/models.py`, `modules/shared.py`, `modules/shared_items.py`, `modules/sd_vae.py`, `tests/test_api_listing_contract.py`, and `tests/test_api_server_control_contract.py` reported no duplicate nontrivial function/class bodies.
- `git diff --check` - passed.
- Live WebUI/API startup, real HTTP calls to `/sdapi/v1/embeddings` or refresh endpoints, actual checkpoint/VAE filesystem refresh under installed model directories, and live textual-inversion reload behavior under a loaded model were not exercised in this bounded slice.

### Next unchecked scope
- More slices are still needed. Recommended next slice: continue below these API wrappers into lower-level embedding/checkpoint/VAE registry refresh internals, especially `EmbeddingDatabase.load_textual_inversion_embeddings()`, `EmbeddingDatabase.load_from_dir()`, `shared_items.refresh_checkpoints()`, `sd_models.list_models()`, `sd_models.checkpoint_tiles()`, `shared_items.refresh_vae_list()`, `sd_vae.refresh_vae_list()`, and adjacent registry/cache tests, looking for stale refresh branches or duplicate registry serialization while preserving public API/UI/extension compatibility and model reload side effects.

## Pass 112 - Lower-level embedding/checkpoint/VAE registry refresh internals (2026-06-20)

### Scope checked
- `modules/textual_inversion/textual_inversion.py`: `EmbeddingDatabase.load_from_dir()` and `EmbeddingDatabase.load_textual_inversion_embeddings()`, plus adjacent embedding directory change tracking, expected-shape filtering, load-time skipped embedding handling, and UI/model-load/API reload call sites.
- `modules/shared_items.py`: `refresh_checkpoints()`, `list_checkpoint_tiles()`, `checkpoint_dropdown_args()`, `refresh_vae_list()`, `sd_vae_items()`, and `sd_vae_dropdown_args()` compatibility wrappers.
- `modules/sd_models.py`: `list_models()`, `checkpoint_tiles()`, `CheckpointInfo.register()`, alias registration, command-line checkpoint handling, VAE checkpoint blacklist filtering, and checkpoint fallback selection adjacency.
- `modules/sd_vae.py`: `vae_search_paths()`, `refresh_vae_list()`, VAE dict sorting, near-checkpoint resolution, user metadata/settings resolution, and cache adjacency.
- Adjacent contract and registry tests/references: `tests/test_api_server_control_contract.py`, `tests/test_api_listing_contract.py`, UI checkpoint/VAE refresh buttons, extra-network refresh surfaces, checkpoint merger refresh paths, and model-load embedding refresh side effects.

### Findings / fixes
- No safe source-code remediation was made in this slice.
- Preserved `EmbeddingDatabase.load_textual_inversion_embeddings()` change-detection and force-reload branches. The non-force path avoids needless reloads by checking every embedding directory; force reload is used by API/UI/model-load paths and must still clear/register/sort loaded and skipped embeddings after recalculating expected shape for the currently loaded model.
- Preserved `EmbeddingDatabase.load_from_dir()` as the directory walking boundary. Its `os.walk(..., followlinks=True)`, zero-byte skip, per-file exception isolation, and delegation to `load_from_file()` are embedding database behavior rather than duplicate generic file scanning.
- Preserved `shared_items` checkpoint/VAE refresh and dropdown helpers as compatibility surfaces. They are imported through `modules.shared`, options refresh callbacks, UI refresh buttons, API refresh handlers, and extension-facing paths; collapsing them into lower-level module calls would risk public import/UI callback compatibility.
- Preserved `sd_models.list_models()` registry rebuild semantics. It intentionally clears both `checkpoints_list` and `checkpoint_aliases`, excludes VAE sidecar files from checkpoint discovery, handles `--ckpt` specially, updates `shared.opts.data['sd_model_checkpoint']` for explicit checkpoint files, and registers all checkpoint aliases through `CheckpointInfo.register()`.
- Preserved `sd_models.checkpoint_tiles()` as the title/short-title serialization helper used by multiple dropdown surfaces with different `use_short` requirements.
- Preserved `sd_vae.refresh_vae_list()` and `vae_search_paths()` as VAE-specific discovery logic. Their search roots and extension patterns intentionally differ from checkpoint discovery, include checkpoint-adjacent sidecars and explicit VAE directories, and sort `vae_dict` by `shared.natural_sort_key` while retaining last-found path behavior for duplicate names.

### Static/dynamic audit map notes
- Embedding reload chain remains: UI/API/model-load refresh -> `EmbeddingDatabase.load_textual_inversion_embeddings(force_reload=...)` -> directory change checks unless forced -> clear loaded/skipped/id maps -> calculate expected shape -> `load_from_dir()` for every registered embedding dir -> per-dir `update()` -> stable name sort -> optional load summary print.
- Checkpoint refresh chain remains: API/options/UI wrappers -> `shared_items.refresh_checkpoints()` / `shared.refresh_checkpoints` -> `sd_models.list_models()` -> `modelloader.load_models(..., ext_blacklist=[".vae.ckpt", ".vae.safetensors"])` -> `CheckpointInfo.register()` into title map and alias map.
- Checkpoint dropdown chain remains: `shared_items.checkpoint_dropdown_args()` / `list_checkpoint_tiles()` -> `sd_models.checkpoint_tiles(use_short=...)` -> title or short-title list from ordered checkpoint registry values.
- VAE refresh chain remains: API/options/UI wrappers -> `shared_items.refresh_vae_list()` or direct `sd_vae.refresh_vae_list()` -> `vae_search_paths()` -> glob candidates -> filename-keyed `vae_dict` sorted by natural name.
- Compatibility surfaces kept conservative: public wrapper names in `shared` and `shared_items`, route/refresh callback targets, checkpoint and VAE dropdown choice shapes, checkpoint alias keys, explicit `--ckpt` option mutation, VAE sidecar discovery semantics, embedding symlink traversal, zero-byte skip behavior, skipped embedding map behavior, and model-load embedding reload side effects.

### Validation log
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pycompile-pass112.XXXXXX) python3 -m py_compile modules/textual_inversion/textual_inversion.py modules/shared_items.py modules/sd_models.py modules/sd_vae.py tests/test_api_listing_contract.py tests/test_api_server_control_contract.py` - passed.
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pytest-pass112.XXXXXX) python3 -m pytest -q tests/test_api_listing_contract.py tests/test_api_server_control_contract.py::test_refresh_endpoints_keep_queue_lock_and_side_effect_targets` - passed: 8 passed, with the existing pytest config warning `Unknown config option: base_url`.
- Exact reference scan confirmed the scoped embedding/checkpoint/VAE refresh functions are referenced by expected API, shared-options, UI, extra-network, model-load, checkpoint-merger, and focused test surfaces.
- Exact AST duplicate-body scan across `modules/textual_inversion/textual_inversion.py`, `modules/shared_items.py`, `modules/sd_models.py`, and `modules/sd_vae.py` reported no duplicate nontrivial function/class bodies.
- `git diff --check` - passed.
- Live WebUI/API startup, actual checkpoint/VAE filesystem refresh under installed model directories, real embedding file reloads under a loaded model, and model/VAE reload side effects were not exercised in this bounded slice.

### Next unchecked scope
- More slices are still needed. Recommended next slice: continue from registry refresh internals into checkpoint selection/load/cache internals, especially `sd_models.get_closet_checkpoint_match()`, `select_checkpoint()`, `get_checkpoint_state_dict()`, `load_model_weights()`, `reuse_model_from_already_loaded()`, `unload_model_weights()`, and adjacent cache/metadata tests, looking for stale cache invalidation branches or duplicate checkpoint lookup/loading paths while preserving model cache semantics, alias compatibility, and reload side effects.

## Pass 113 - Checkpoint selection/load/cache internals (2026-06-20)

### Scope checked
- `modules/sd_models.py`: `get_closet_checkpoint_match()`, `select_checkpoint()`, `get_checkpoint_state_dict()`, `load_model_weights()`, `reuse_model_from_already_loaded()`, `reload_model_weights()` adjacency, `unload_model_weights()`, `CheckpointInfo.calculate_shorthash()`, checkpoint alias/cache globals, `SdModelData.set_sd_model()`, and adjacent TorchAO checkpoint-cache invalidation branches.
- Adjacent public/API/UI/extension call sites: checkpoint override parsing in `modules/img2img.py`, `modules/processing.py`, `modules/processing_scripts/refiner.py`, `modules/ui.py`, `scripts/prompts_from_file.py`, `scripts/xyz_grid.py`, API unload/reload routes in `modules/api/api.py`, UI unload settings in `modules/ui_settings.py`, and OpenClaw clear-cond-cache wrappers around checkpoint load/reload hooks.
- Adjacent tests: `tests/test_sd_models_checkpoint_info_contract.py`, checkpoint unload/reload contract coverage in `tests/test_api_server_control_contract.py`, and extension cache-wrapper tests by reference.

### Findings / fixes
- Consolidated duplicate MXFP8/NVFP4 policy checks in `load_model_weights()` by caching `check_mxfp8(model)` and `check_nvfp4(model)` results before checkpoint state-dict cache handling. This keeps the same cache-store/clear semantics while avoiding repeated policy/device option evaluation in the same branch.
- Preserved `get_closet_checkpoint_match()` including the historical typo. It is the live public compatibility spelling used by scripts, processing, UI, and img2img parameter parsing; adding or renaming aliases would broaden the public surface without removing dead code.
- Preserved `select_checkpoint()` fallback behavior and error text. It is the central options-to-registry lookup and no-checkpoints failure surface used by model loading, textual inversion training, and lowvram processing paths.
- Preserved `get_checkpoint_state_dict()` returning cached state dicts by reference and moving cache entries to LRU tail. Existing reload/TorchAO branches depend on explicit invalidation rather than hidden copies, and changing this would alter memory/cache behavior.
- Preserved `reuse_model_from_already_loaded()` branches. They enforce checkpoint limit eviction, CPU retention, stale TorchAO policy eviction, already-loaded VAE state restoration, and model reuse/new-load behavior that is externally controlled by options.
- Preserved `unload_model_weights(sd_model=None, info=None)` signature despite the unused `info` parameter. The function is called by API/UI surfaces and extension code may rely on the permissive signature.

### Static/dynamic audit map notes
- Checkpoint selection chain remains: user/override string -> `checkpoint_aliases` / substring checksum-stripped match -> `CheckpointInfo` -> `select_checkpoint()` fallback when configured checkpoint is absent.
- State-dict cache chain remains: `get_checkpoint_state_dict()` calculates shorthash -> reads `checkpoints_loaded` LRU by `CheckpointInfo` object -> reads disk on miss -> `load_model_weights()` stores a shallow copy only when `sd_checkpoint_cache > 0` and MXFP8/NVFP4 are inactive -> TorchAO quantized paths clear the cache.
- Loaded-model cache chain remains: `reload_model_weights()` -> `reuse_model_from_already_loaded()` -> stale TorchAO policy eviction / sd_checkpoints_limit trimming / CPU or device migration / VAE state restoration -> either reuse, allocate another loaded model slot, or recycle an existing model instance.
- Compatibility surfaces kept conservative: misspelled `get_closet_checkpoint_match`, `checkpoint_alisases`, `unload_model_weights(..., info=None)`, checkpoint alias keys/hash metadata, `sd_model_checkpoint` and `sd_checkpoint_hash` writes, checkpoint reload callbacks, VAE restoration, lowvram/offload behavior, and OpenClaw cache-wrapper hook names.

### Validation log
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pycompile-pass113.XXXXXX) python3 -m py_compile modules/sd_models.py modules/sd_models_config.py modules/sd_models_types.py modules/sd_vae.py tests/test_sd_models_checkpoint_info_contract.py` - passed.
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pytest-pass113.XXXXXX) python3 -m pytest -q tests/test_sd_models_checkpoint_info_contract.py` - passed: 2 passed, with the existing pytest config warning `Unknown config option: base_url`.
- Exact reference scan confirmed scoped checkpoint selection/load/cache functions and aliases are referenced by expected scripts, processing, API/UI, extension wrapper, and focused test surfaces.
- Exact AST duplicate-body scan across `modules/sd_models.py` and `tests/test_sd_models_checkpoint_info_contract.py` reported no duplicate nontrivial function/class bodies.
- `git diff --check` - passed.
- Live WebUI/API startup, actual checkpoint file loading/reloading, CUDA memory/offload behavior, and real MXFP8/NVFP4 cache invalidation under a loaded model were not exercised in this bounded slice.

### Next unchecked scope
- More slices are still needed. Recommended next slice: continue from checkpoint load/cache internals into checkpoint config/model-instantiation internals, especially `load_model()`, `instantiate_from_config()`, `sd_models_config.find_checkpoint_config()`, `repair_config()`, `set_model_type()`, `set_model_fields()`, and adjacent model-type/config tests, looking for duplicate config inference or stale compatibility branches while preserving checkpoint reload side effects, model-type metadata, config repair behavior, and extension callbacks.


## Pass 114 - Checkpoint config/model-instantiation internals (2026-06-20)

### Scope checked
- `modules/sd_models.py`: `load_model()`, `instantiate_from_config()`, `repair_config()`, `set_model_type()`, `set_model_fields()`, `load_model_weights()` model-type/config adjacency, fresh-load callback/embedding/hijack side effects, and `reload_model_weights()` checkpoint-config reuse/reload branches around fresh TorchAO reloads and already-loaded model reuse.
- `modules/sd_models_config.py`: `find_checkpoint_config()`, `find_checkpoint_config_near_filename()`, `guess_model_config_from_state_dict()`, SD1/SD2/SDXL/SSD/SD3/inpainting/instruct-pix2pix/depth/UnCLIP/Alt-Diffusion config inference branches, and SD2 v-parameterization probing.
- Adjacent model-type/config consumers by reference: `modules/sd_vae_approx.py`, `modules/sd_vae_taesd.py`, `modules/sd_samplers.py`, `modules/sd_schedulers.py`, `modules/processing.py`, `modules/lowvram.py`, `extensions-builtin/Lora/*`, `extensions-builtin/hypertile/*`, `modules/api/api.py`, `modules/extras.py`, and `modules/sd_hijack_ip2p.py`.
- Adjacent tests/references: `tests/test_sd_models_checkpoint_info_contract.py`, `tests/test_api_listing_contract.py`, `test/test_openclaw_device_dtypes.py`, scheduler/cache tests by model-type attribute reference, and exact reference scans for config/model-type helpers.

### Findings / fixes
- No safe source-code remediation was made in this slice.
- Preserved `find_checkpoint_config()` and `find_checkpoint_config_near_filename()` separation. The near-file helper is a public/adjacent compatibility surface used by API model listing, extras, IP2P hijack config detection, and tests; full config selection must still prefer checkpoint-adjacent YAML before state-dict inference.
- Preserved `guess_model_config_from_state_dict()` branches, including SD3, SDXL inpaint/v-pred/base/refiner, SD2 inpaint/v/base, SD1 inpaint, instruct-pix2pix, depth, UnCLIP/OpenCLIP, and Alt-Diffusion variants. These branches map checkpoint formats to distinct YAML targets, and removing rare-format handling would be a compatibility regression rather than dead-code cleanup.
- Preserved the `v_pred` deletion in SDXL-v inference and the SD2 v-parameterization probe. Both are mutation/compatibility behaviors inside config inference, not duplicate config repair paths.
- Preserved `repair_config()` as the post-YAML normalization boundary. Its EMA default, fp16 UNet flag adjustment, VAE `vanilla-xformers` repair, Karlo path rewrite, and checkpoint disabling target different config subtrees and should remain explicit.
- Preserved `set_model_type()` and `set_model_fields()` as load-time metadata initialization. The boolean flags and `model_type` enum feed samplers, schedulers, VAE approximations, LoRA filtering, HyperTile, processing inpaint behavior, and extension-facing checks; `latent_channels` default remains a compatibility field for models/configs that omit it.
- Preserved the quick-then-slow `instantiate_from_config()` call pattern in `load_model()`. The first attempt is wrapped in `DisableInitialization(disable_clip=...)` plus meta initialization, while the fallback intentionally retries without `DisableInitialization` after reporting the fast-path exception.
- Preserved repeated checkpoint-config inference in `reload_model_weights()` branches. The fresh TorchAO path re-reads uncached disk state before delegating to `load_model()`, and the reusable-model path must compare the new checkpoint config against `sd_model.used_config` before deciding between in-place weight load and full model reconstruction.

### Static/dynamic audit map notes
- Initial load chain remains: selected checkpoint -> state dict -> optional near-file YAML -> inferred config -> `OmegaConf.load()` -> `repair_config()` -> fast model construction attempt -> slow construction fallback -> `load_model_weights()` -> model-to-device -> hijack -> embedding reload -> model-loaded callbacks -> empty prompt cache.
- Model metadata chain remains: `load_model_weights()` -> `set_model_type()` / `set_model_fields()` -> SDXL extension and SSD conversion -> CLIP key remap -> weight load -> `is_sdxl_inpaint` structural detection.
- Config inference chain remains: adjacent YAML wins; otherwise state-dict markers choose SD3/SDXL/refiner/depth/UnCLIP/SD2/SD1/instruct-pix2pix/Alt-Diffusion/default configs, with SD2 v-parameterization requiring a temporary UNet probe.
- Compatibility surfaces kept conservative: checkpoint-adjacent YAML discovery, API `config` field behavior, IP2P/extras config basename callers, rare checkpoint format branches, model boolean flags, `ModelType` enum values, model `used_config`, `latent_channels` default, extension callbacks, model hijack ordering, textual inversion reload after model load, and TorchAO precision/quant reload boundaries.

### Validation log
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pycompile-pass114.XXXXXX) python3 -m py_compile modules/sd_models.py modules/sd_models_config.py modules/sd_models_types.py tests/test_sd_models_checkpoint_info_contract.py tests/test_api_listing_contract.py test/test_openclaw_device_dtypes.py` - passed.
- `PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/gb10-a1111-pytest-pass114.XXXXXX) python3 -m pytest -q tests/test_sd_models_checkpoint_info_contract.py tests/test_api_listing_contract.py` - passed: 9 passed, with the existing pytest config warning `Unknown config option: base_url`.
- Exact reference scan confirmed scoped config/type helpers and model-type fields are referenced by expected API, extras, IP2P, sampler/scheduler, processing, VAE approximation, LoRA, HyperTile, lowvram, and focused test surfaces.
- Exact AST duplicate-body scan across `modules/sd_models.py` and `modules/sd_models_config.py` reported no duplicate nontrivial function/class bodies.
- `git diff --check` - passed.
- Live WebUI/API startup, actual checkpoint/model instantiation, rare checkpoint-format config inference with real files, CUDA/offload behavior, and extension callback execution under a loaded model were not exercised in this bounded slice.

### Next unchecked scope
- More slices are still needed. Recommended next slice: continue from model instantiation into post-load model mutation and conditioning helpers, especially `remap_sdxl_clip_text_model_state_dict_if_needed()`, `apply_alpha_schedule_override()`, `rescale_zero_terminal_snr_abar()`, `get_empty_cond()`, `sd_models_xl.extend_sdxl()`, `sd_models_xl.get_learned_conditioning()`, and adjacent prompt-conditioning/model-patch tests, looking for duplicate conditioning or stale SDXL compatibility branches while preserving prompt output shape, sampler behavior, extension patches, and inpaint/refiner compatibility.
