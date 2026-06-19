# GB10 A1111 latest integration/API/UI audit ledger

Target: `/home/kklouzal/stable-diffusion-webui` on GB10, branch `latest`.

Baseline:
- 2026-06-19: `git status --short` reported only pre-existing untracked `.audit/` before this slice changed files.
- Baseline HEAD: `41ef77d8` (`Add save serialization contract tests`).
- Recent core/math lane commits present: `d0fd49e8`, `5be063dd`, `f9053a10`, `d56d348b`, `03e6c99d`, `8b31a973`; avoided duplicating those infotext/save-alignment fixes.

Audit scope:
- Integration and UI/API glue outside the core generation/math lane: `modules/api/**`, `modules/ui*.py`, `modules/call_queue.py`, `modules/script_callbacks.py`, `modules/extensions.py`, `modules/ui_extensions.py`, `javascript/**`, `scripts/**`, `extensions-builtin/**`, `launch.py`, `webui.py`.
- Priority checks: parameter coercion, seed/dimension/mask/resize propagation, stale options, callback ordering, queue/state isolation, UI/API contradictions, final-quality-affecting integration bugs.

Reviewed files/functions:
- `modules/api/api.py`: `_precision_*`, `build_precision_map`, `ScriptArgsList`, `script_default_ui_values`, `api_infotext_value_for_field`, `script_name_to_index`, `validate_sampler_name`, `setUpscalers`, `verify_url`, `decode_base64_to_image`, `decode_extras_batch_images`, `processed_js_with_image_paths`, `encode_pil_to_base64`, `api_middleware`, `Api.__init__`, runtime diagnostics routes, `get_selectable_script`, `get_scripts_list`, `get_script_info`, `get_script`, `init_default_script_args`, `persist_openclaw_denoise_ramp_args`, `init_script_args`, `apply_infotext`, `text2imgapi`, `img2imgapi`, extras/pnginfo/progress/interrogate/config/listing/training endpoints.
  - Findings: no confirmed remediation in reviewed API paths. Always-on argument extension and infotext-to-script-arg propagation are coherent; API txt2img/img2img uses queue lock and per-request processing object setup consistently.
- `modules/api/models.py`: `PydanticModelGenerator.__init__`, `field_type_generator`, `merge_class_params`, `generate_model`, txt2img/img2img request model construction, extras/progress/config/listing models.
  - Findings: no confirmed remediation. Dynamic model exclusions and API-only fields line up with processing constructors in reviewed paths.
- `modules/call_queue.py`: `wrap_queued_call`, `wrap_ui_gpu_call`, `wrap_ui_call`, `wrap_ui_call_no_job`.
  - Findings: no confirmed remediation. Queue lock, progress lifecycle, state reset, and stats wrapping are coherent in reviewed paths.
- `modules/script_callbacks.py`: callback parameter classes, `add_callback`, `sort_callbacks`, `ordered_callbacks`, `enumerate_callbacks`, `clear_callbacks`, all callback dispatcher functions, removal functions, and registration helpers.
  - Findings: no confirmed remediation. Cached ordered callbacks are invalidated by length change or updated by removal; callback exceptions are isolated.
- `modules/extensions.py`: `active`, `ExtensionMetadata.__init__`, `get_script_requirements`, `parse_list`, `list_callback_order_instructions`, `Extension.__init__`, `to_dict`, `from_dict`, `read_info_from_repo`, `do_read_info_from_repo`, `list_files`, `check_updates`, `fetch_and_reset_hard`, `list_extensions`, `find_extension`.
  - Findings: no generation-quality remediation. Noted non-scope robustness issue: constructing `Extension(..., metadata=None)` directly would reference `metadata.canonical_name`, but normal extension discovery passes metadata.
- `scripts/img2imgalt.py`: `find_noise_for_image`, `find_noise_for_image_sigma_adjustment`, `Script.run`.
  - Findings: no confirmed remediation in this slice; reviewed for sampler override, prompt/step/strength overrides, cache key, seed increment, and reconstructed-noise mixing.
- `scripts/poor_mans_outpainting.py`: `Script.run`.
  - Findings: no confirmed remediation. Target dimension rounding, mask/latent-mask construction, tile selection, seed increment, and combined image return are coherent.
- `scripts/sd_upscale.py`: `Script.run`.
  - Findings: no new remediation; recent commits already fixed per-result infotext alignment. Reviewed tile batching, seed progression, result infotexts, and save behavior.
- `scripts/outpainting_mk_2.py`: `get_matched_noise`, `Script.run`, nested `expand`.
  - Findings: no new remediation; recent commits already fixed returned grid indexing. Reviewed FFT normalizers, deterministic local noise source, directional expansion dimensions, masks, per-direction processing, and grid/sample save paths.
- `scripts/xyz_grid.py`: apply/confirm/format helpers, axis option definitions, start of `draw_xyz_grid`.
  - Findings: no new remediation; recent commits already addressed grid/sample index and infotext alignment.
- `extensions-builtin/soft-inpainting/scripts/soft_inpainting.py`: `SoftInpaintingSettings`, `processing_uses_inpainting`, `latent_blend`, `get_modified_nmask`, `apply_adaptive_masks`, `apply_masks`, `weighted_histogram_filter`, smoothstep helpers, kernel generation, `Script` UI setup.
  - Findings: no confirmed remediation in reviewed chunks. Mask conversion and overlay composition paths are coherent.
- `javascript/imageMaskFix.js`: `imageMaskResize`.
  - Finding: mask canvas wrapper was resized to the displayed image content but pinned to `left: 0; top: 0`, so letterboxed images offset user-painted inpaint masks from the actual displayed image. This can directly degrade inpainting quality.
  - Remediation: center wrapper with `(w - wW) / 2` and `(h - wH) / 2`.
  - Validation: `pytest -q tests/test_image_mask_fix_contract.py` passed.
- `javascript/hires_fix.js`: `onCalcResolutionHires`.
  - Findings: no confirmed remediation. Control inactivity follows old/new hires resize mode and explicit resize dimensions.
- `javascript/generationParams.js`: gallery listener setup, modal observer, `attachGalleryListeners`.
  - Findings: no confirmed remediation. Generation-info refresh wiring is coherent.
- `javascript/ui.js`: gallery selection helpers, tab switching helpers, submit arg construction, submit/restore progress functions, resolution paste setup.
  - Findings: no confirmed remediation in reviewed chunks. Submit arg gallery stripping and img2img tab index injection line up with UI submit flow.
- `javascript/settings.js`: settings search/show-all/category helpers.
  - Findings: no generation-quality remediation.
- `modules/ui_components.py`: form component wrappers, `InputAccordion`.
  - Findings: no confirmed remediation. Hidden checkbox/accordion state and unload reset are coherent.
- `modules/ui_toprow.py`: `Toprow` init, prompt creation, submit box, tool row, styles UI.
  - Findings: no confirmed remediation in reviewed chunks. Submit/interrupt hooks and prompt image extraction are coherent.

Commits made during this audit:
- Pending: fix and ledger/test commit for centered inpaint mask canvas.

Validation run:
- `pytest -q tests/test_image_mask_fix_contract.py` - passed, with pre-existing pytest warning about unknown `base_url` config option.

Remaining areas / continuation:
- Continue `modules/ui.py` beyond helper/top-level UI construction, especially img2img component wiring and paste-field mappings.
- Continue `modules/ui_common.py`, `modules/ui_postprocessing.py`, `modules/ui_settings.py`, `modules/ui_loadsave.py`, `modules/ui_extensions.py`, `modules/ui_extra_networks*.py`.
- Continue remaining JS files: `edit-attention.js`, `extraNetworks.js`, `token-counters.js`, `progressbar.js`, `aspectRatioOverlay.js`, `contextMenus.js`, `imageviewer*.js`, `resizeHandle.js`, extension JS.
- Continue remaining scripts and builtin extensions, especially Lora integration, Hypertile, LDSR/SwinIR/ScuNET postprocess paths, and extra-options-section.
- Run a broader focused gate after more remediations: API/script tests plus any available JS lint/static checks.
