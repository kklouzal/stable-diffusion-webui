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
