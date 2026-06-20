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
