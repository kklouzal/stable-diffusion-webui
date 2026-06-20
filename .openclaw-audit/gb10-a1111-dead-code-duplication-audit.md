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
