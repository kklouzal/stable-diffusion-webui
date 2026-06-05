from __future__ import annotations

import importlib.util
import os
import threading
import time
from typing import Any

from fastapi import FastAPI, Request

from modules import call_queue, extra_networks, extras, prompt_parser, script_callbacks, sd_models
from modules.processing import StableDiffusionProcessing, StableDiffusionProcessingImg2Img, StableDiffusionProcessingTxt2Img
from modules.textual_inversion import textual_inversion

_last_cleared_at = 0.0
_compile_slots: dict[str, dict[str, Any]] = {}
_compile_desired: dict[str, bool] = {"vae": False}
_compile_status: dict[str, Any] = {"vae": False, "last_error": None}
_backend_activity_lock = threading.Lock()
_backend_activity_stack: list[dict[str, Any]] = []
_backend_activity_token = 0
_backend_hooks_installed = False
_backend_lora_hooks_installed = False
_backend_lora_batch: dict[str, Any] = {"total": 0, "index": 0}
_startup_model_load_token: int | None = None

try:
    from modules.sd_hijack import model_hijack
except (ImportError, ModuleNotFoundError):
    model_hijack = None


def estimate_token_count(text: str, steps: int) -> dict[str, Any]:
    """Estimate A1111 prompt token length using the active model tokenizer."""
    try:
        try:
            stripped_text, _ = extra_networks.parse_prompt(text or "")
            _, prompt_flat_list, _ = prompt_parser.get_multicond_prompt_list([stripped_text])
            prompt_schedules = prompt_parser.get_learned_conditioning_prompt_schedules(prompt_flat_list, steps)
        except Exception:
            prompt_schedules = [[[steps, text or ""]]]

        forge = importlib.util.find_spec("modules_forge") is not None

        prompts = [
            prompt_text
            for prompt_schedule in prompt_schedules
            for _step, prompt_text in prompt_schedule
        ] or [text or ""]

        if model_hijack is None:
            return {"ok": False, "error": "A1111 model_hijack tokenizer is unavailable", "token_count": None, "max_length": None}

        if forge:
            cond_stage_model = sd_models.model_data.sd_model.cond_stage_model
            token_count, max_length = max(
                [model_hijack.get_prompt_lengths(prompt, cond_stage_model) for prompt in prompts],
                key=lambda args: args[0],
            )
        else:
            token_count, max_length = max(
                [model_hijack.get_prompt_lengths(prompt) for prompt in prompts],
                key=lambda args: args[0],
            )

        return {"ok": True, "token_count": token_count, "max_length": max_length}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "token_count": None, "max_length": None}




def _backend_status_payload() -> dict[str, Any]:
    with _backend_activity_lock:
        if _backend_activity_stack:
            current = dict(_backend_activity_stack[-1])
            current.pop("token", None)
            return {"ok": True, **current}
    return {
        "ok": True,
        "active": False,
        "phase": None,
        "label": None,
        "detail": None,
        "progress": None,
        "current": None,
        "total": None,
        "started_at": None,
        "updated_at": None,
    }


def _push_backend_activity(phase: str, label: str, *, detail: Any = None, progress: Any = None, current: Any = None, total: Any = None) -> int:
    global _backend_activity_token
    now = time.time()
    detail_text = None if detail in (None, "") else str(detail)
    with _backend_activity_lock:
        _backend_activity_token += 1
        token = _backend_activity_token
        _backend_activity_stack.append({
            "token": token,
            "active": True,
            "phase": phase,
            "label": label,
            "detail": detail_text,
            "progress": progress,
            "current": current,
            "total": total,
            "started_at": now,
            "updated_at": now,
        })
        return token


def _update_backend_activity(token: int, *, label: Any = None, detail: Any = None, progress: Any = None, current: Any = None, total: Any = None) -> None:
    now = time.time()
    with _backend_activity_lock:
        for item in reversed(_backend_activity_stack):
            if item.get("token") == token:
                if label is not None:
                    item["label"] = str(label)
                if detail is not None:
                    item["detail"] = None if detail in (None, "") else str(detail)
                if progress is not None:
                    item["progress"] = progress
                if current is not None:
                    item["current"] = current
                if total is not None:
                    item["total"] = total
                item["updated_at"] = now
                break


def _pop_backend_activity(token: int) -> None:
    with _backend_activity_lock:
        for idx in range(len(_backend_activity_stack) - 1, -1, -1):
            if _backend_activity_stack[idx].get("token") == token:
                _backend_activity_stack.pop(idx)
                break


def _checkpoint_detail(checkpoint_info: Any) -> str | None:
    if checkpoint_info is None:
        return None
    for attr in ("title", "name", "model_name", "filename"):
        value = getattr(checkpoint_info, attr, None)
        if value:
            return os.path.basename(str(value))
    return str(checkpoint_info)


def _wrap_backend_function(module: Any, attr: str, wrapper_factory) -> None:
    original = getattr(module, attr, None)
    if original is None or getattr(original, "__openclaw_backend_status_wrapped__", False):
        return
    wrapped = wrapper_factory(original)
    wrapped.__openclaw_backend_status_wrapped__ = True
    setattr(module, attr, wrapped)


def _install_backend_status_hooks() -> None:
    global _backend_hooks_installed, _backend_lora_hooks_installed

    if not _backend_hooks_installed:
        try:
            from modules import sd_models as _sd_models
            from modules import sd_vae as _sd_vae
        except Exception:
            return

        _backend_hooks_installed = True

        def _wrap_reload_model_weights(original):
            def wrapped(sd_model=None, info=None, forced_reload=False):
                token = _push_backend_activity("model_load", "Reloading checkpoint", detail=_checkpoint_detail(info))
                try:
                    return original(sd_model=sd_model, info=info, forced_reload=forced_reload)
                finally:
                    _pop_backend_activity(token)
            return wrapped

        def _wrap_load_model(original):
            def wrapped(checkpoint_info=None, already_loaded_state_dict=None, checkpoint_config=None):
                token = _push_backend_activity("model_load", "Loading checkpoint", detail=_checkpoint_detail(checkpoint_info))
                try:
                    return original(checkpoint_info=checkpoint_info, already_loaded_state_dict=already_loaded_state_dict, checkpoint_config=checkpoint_config)
                finally:
                    _pop_backend_activity(token)
            return wrapped

        def _wrap_get_checkpoint_state_dict(original):
            def wrapped(checkpoint_info, timer):
                token = _push_backend_activity("checkpoint_read", "Reading checkpoint", detail=_checkpoint_detail(checkpoint_info))
                try:
                    return original(checkpoint_info, timer)
                finally:
                    _pop_backend_activity(token)
            return wrapped

        def _wrap_load_model_weights(original):
            def wrapped(model, checkpoint_info, state_dict, timer):
                token = _push_backend_activity("checkpoint_apply", "Applying checkpoint weights", detail=_checkpoint_detail(checkpoint_info))
                try:
                    return original(model, checkpoint_info, state_dict, timer)
                finally:
                    _pop_backend_activity(token)
            return wrapped

        def _wrap_load_vae(original):
            def wrapped(model, vae_file=None, vae_source="from unknown source"):
                detail = os.path.basename(str(vae_file)) if vae_file else str(vae_source)
                token = _push_backend_activity("vae_load", "Loading VAE", detail=detail)
                restore_compile = False
                try:
                    first_stage = getattr(model, "first_stage_model", None)
                    unwrapped = _unwrap_compiled_module(first_stage)
                    if first_stage is not None and first_stage is not unwrapped:
                        model.first_stage_model = unwrapped
                        if getattr(sd_models.model_data, "sd_model", None) is model:
                            _compile_slots.pop("vae", None)
                            _compile_status["vae"] = False
                            restore_compile = bool(_compile_desired.get("vae"))
                    result = original(model, vae_file=vae_file, vae_source=vae_source)
                    if restore_compile:
                        apply_torch_compile_settings(vae=True)
                    return result
                finally:
                    _pop_backend_activity(token)
            return wrapped

        def _wrap_instantiate_from_config(original):
            def wrapped(config, state_dict=None):
                detail = None
                try:
                    detail = config.get("target")
                except Exception:
                    detail = None
                token = _push_backend_activity("model_create", "Creating model from config", detail=detail)
                try:
                    return original(config, state_dict=state_dict)
                finally:
                    _pop_backend_activity(token)
            return wrapped

        def _wrap_send_model_to_device(original):
            def wrapped(model):
                token = _push_backend_activity("model_device", "Moving model to GPU")
                try:
                    return original(model)
                finally:
                    _pop_backend_activity(token)
            return wrapped

        def _wrap_get_empty_cond(original):
            def wrapped(sd_model):
                token = _push_backend_activity("conditioning", "Calculating empty prompt conditioning")
                try:
                    return original(sd_model)
                finally:
                    _pop_backend_activity(token)
            return wrapped

        def _wrap_mxfp8_quantization(original):
            def wrapped(model, timer, source_path=None):
                token = _push_backend_activity("quantize_base", "Preparing MXFP8 base weights", detail=os.path.basename(str(source_path)) if source_path else None)
                try:
                    return original(model, timer, source_path=source_path)
                finally:
                    _pop_backend_activity(token)
            return wrapped

        def _wrap_nvfp4_quantization(original):
            def wrapped(model, timer, source_path=None):
                token = _push_backend_activity("quantize_base", "Preparing NVFP4 base weights", detail=os.path.basename(str(source_path)) if source_path else None)
                try:
                    return original(model, timer, source_path=source_path)
                finally:
                    _pop_backend_activity(token)
            return wrapped

        _wrap_backend_function(_sd_models, "reload_model_weights", _wrap_reload_model_weights)
        _wrap_backend_function(_sd_models, "load_model", _wrap_load_model)
        _wrap_backend_function(_sd_models, "get_checkpoint_state_dict", _wrap_get_checkpoint_state_dict)
        _wrap_backend_function(_sd_models, "load_model_weights", _wrap_load_model_weights)
        _wrap_backend_function(_sd_models, "instantiate_from_config", _wrap_instantiate_from_config)
        _wrap_backend_function(_sd_models, "send_model_to_device", _wrap_send_model_to_device)
        _wrap_backend_function(_sd_models, "get_empty_cond", _wrap_get_empty_cond)
        _wrap_backend_function(_sd_models, "apply_mxfp8_weight_quantization", _wrap_mxfp8_quantization)
        _wrap_backend_function(_sd_models, "apply_nvfp4_weight_quantization", _wrap_nvfp4_quantization)
        _wrap_backend_function(_sd_vae, "load_vae", _wrap_load_vae)

    if _backend_lora_hooks_installed:
        return

    try:
        import networks as _lora_networks
    except Exception:
        return

    def _wrap_load_networks(original):
        def wrapped(names, te_multipliers=None, unet_multipliers=None, dyn_dims=None):
            names_list = [str(name) for name in (names or []) if name]
            total = len(names_list)
            _backend_lora_batch["total"] = total
            _backend_lora_batch["index"] = 0
            token = _push_backend_activity("lora_batch", "Loading/applying LoRAs", detail=f"{total} selected" if total else "No LoRAs", current=0 if total else None, total=total or None)
            try:
                return original(names, te_multipliers=te_multipliers, unet_multipliers=unet_multipliers, dyn_dims=dyn_dims)
            finally:
                _backend_lora_batch["total"] = 0
                _backend_lora_batch["index"] = 0
                _pop_backend_activity(token)
        return wrapped

    def _wrap_load_network(original):
        def wrapped(name, network_on_disk):
            total = int(_backend_lora_batch.get("total") or 0)
            _backend_lora_batch["index"] = int(_backend_lora_batch.get("index") or 0) + 1
            current = _backend_lora_batch["index"] if total else None
            detail = name or os.path.basename(getattr(network_on_disk, "filename", "") or "")
            token = _push_backend_activity("lora_load", "Loading LoRA", detail=detail, current=current, total=total or None)
            try:
                return original(name, network_on_disk)
            finally:
                _pop_backend_activity(token)
        return wrapped

    def _wrap_prepare_quant_lora(label: str, phase: str):
        def factory(original):
            def wrapped(*args, **kwargs):
                lora_count = len(getattr(_lora_networks, "loaded_networks", []) or [])
                token = _push_backend_activity(phase, label, detail=f"{lora_count} active LoRA{'s' if lora_count != 1 else ''}")
                try:
                    return original(*args, **kwargs)
                finally:
                    _pop_backend_activity(token)
            return wrapped
        return factory

    _wrap_backend_function(_lora_networks, "load_networks", _wrap_load_networks)
    _wrap_backend_function(_lora_networks, "load_network", _wrap_load_network)
    _wrap_backend_function(_lora_networks, "prepare_mxfp8_active_config", _wrap_prepare_quant_lora("Preparing MXFP8 LoRA weights", "quant_lora_prepare"))
    _wrap_backend_function(_lora_networks, "prepare_nvfp4_active_config", _wrap_prepare_quant_lora("Preparing NVFP4 LoRA weights", "quant_lora_prepare"))
    _backend_lora_hooks_installed = True


def _get_vae_module():
    return getattr(sd_models.model_data.sd_model, "first_stage_model", None)


def _set_vae_module(module: Any) -> None:
    sd_models.model_data.sd_model.first_stage_model = module


def _unwrap_compiled_module(module: Any) -> Any:
    return getattr(module, "_orig_mod", module)


def _compile_module_slot(name: str, enabled: bool, getter, setter) -> dict[str, Any]:
    import torch

    module = getter()
    if module is None:
        _compile_status[name] = False
        _compile_slots.pop(name, None)
        return {"name": name, "enabled": False, "changed": False, "error": "module not found"}

    slot = _compile_slots.get(name)
    unwrapped = _unwrap_compiled_module(module)

    if enabled:
        if not hasattr(torch, "compile"):
            _compile_status[name] = False
            return {"name": name, "enabled": False, "changed": False, "error": "torch.compile is unavailable"}

        if slot and id(module) == slot.get("compiled_id") and id(unwrapped) == slot.get("original_id"):
            _compile_status[name] = True
            return {"name": name, "enabled": True, "changed": False, "already_compiled": True}

        original = unwrapped
        original.eval()
        token = _push_backend_activity("torch_compile", f"Compiling {name.upper()}", detail="torch.compile reduce-overhead/dynamic")
        try:
            compiled = torch.compile(original, mode="reduce-overhead", fullgraph=False, dynamic=True)
            setter(compiled)
        finally:
            _pop_backend_activity(token)
        _compile_slots[name] = {
            "original": original,
            "original_id": id(original),
            "compiled_id": id(compiled),
        }
        _compile_status[name] = True
        return {"name": name, "enabled": True, "changed": True, "mode": "reduce-overhead", "dynamic": True}

    if slot and id(module) == slot.get("compiled_id"):
        token = _push_backend_activity("torch_compile", f"Restoring uncompiled {name.upper()}")
        try:
            setter(slot["original"])
        finally:
            _pop_backend_activity(token)
        _compile_slots.pop(name, None)
        _compile_status[name] = False
        return {"name": name, "enabled": False, "changed": True}

    if module is not unwrapped:
        token = _push_backend_activity("torch_compile", f"Restoring uncompiled {name.upper()}")
        try:
            setter(unwrapped)
        finally:
            _pop_backend_activity(token)
        _compile_slots.pop(name, None)
        _compile_status[name] = False
        return {"name": name, "enabled": False, "changed": True}

    _compile_slots.pop(name, None)
    _compile_status[name] = False
    return {"name": name, "enabled": False, "changed": False}


def apply_torch_compile_settings(vae: bool = False) -> dict[str, Any]:
    results = []
    _compile_status["last_error"] = None
    _compile_desired["vae"] = bool(vae)
    try:
        results.append(_compile_module_slot("vae", _compile_desired["vae"], _get_vae_module, _set_vae_module))
        return {"ok": True, "desired": dict(_compile_desired), "status": dict(_compile_status), "results": results}
    except Exception as exc:
        _compile_status["last_error"] = str(exc)
        return {"ok": False, "error": str(exc), "desired": dict(_compile_desired), "status": dict(_compile_status), "results": results}



def _model_merge_config_source_value(value: Any) -> str:
    choices = ["A, B or C", "B", "C", "Don't"]
    if isinstance(value, int) and 0 <= value < len(choices):
        return choices[value]
    text = str(value or "A, B or C").strip()
    return text if text in choices else "A, B or C"


def _run_openclaw_model_merge(payload: dict[str, Any]) -> dict[str, Any]:
    method = str(payload.get("interp_method") or "Weighted sum").strip()
    if method not in {"No interpolation", "Weighted sum", "Add difference"}:
        return {"ok": False, "error": f"Unsupported interpolation method: {method}"}

    primary = str(payload.get("primary_model_name") or "").strip()
    secondary = str(payload.get("secondary_model_name") or "").strip()
    tertiary = str(payload.get("tertiary_model_name") or "").strip()
    if not primary:
        return {"ok": False, "error": "Primary model A is required"}
    if method in {"Weighted sum", "Add difference"} and not secondary:
        return {"ok": False, "error": "Secondary model B is required for this merge method"}
    if method == "Add difference" and not tertiary:
        return {"ok": False, "error": "Tertiary model C is required for Add difference"}

    checkpoint_format = str(payload.get("checkpoint_format") or "safetensors").strip()
    if checkpoint_format not in {"ckpt", "safetensors"}:
        return {"ok": False, "error": "Checkpoint format must be ckpt or safetensors"}

    metadata_json = str(payload.get("metadata_json") or "{}").strip() or "{}"
    try:
        import json
        json.loads(metadata_json)
    except Exception as exc:
        return {"ok": False, "error": f"Metadata JSON is invalid: {exc}"}

    try:
        with call_queue.queue_lock:
            outputs = extras.run_modelmerger(
                None,
                primary,
                secondary,
                tertiary,
                method,
                float(payload.get("multiplier") if payload.get("multiplier") is not None else 0.3),
                bool(payload.get("save_as_half", False)),
                str(payload.get("custom_name") or "").strip(),
                checkpoint_format,
                _model_merge_config_source_value(payload.get("config_source")),
                str(payload.get("bake_in_vae") or "None").strip() or "None",
                str(payload.get("discard_weights") or ""),
                bool(payload.get("save_metadata", True)),
                bool(payload.get("add_merge_recipe", True)),
                bool(payload.get("copy_metadata_fields", True)),
                metadata_json,
            )
        message = str((outputs or [""])[-1] or "")
        if message.lower().startswith(("failed:", "error")):
            return {"ok": False, "error": message}
        return {"ok": True, "message": message}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _training_templates() -> dict[str, Any]:
    try:
        templates = textual_inversion.list_textual_inversion_templates()
        return {"ok": True, "templates": sorted(str(name) for name in templates)}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "templates": []}

def on_model_loaded(_: Any) -> None:
    global _startup_model_load_token
    if _startup_model_load_token is not None:
        _pop_backend_activity(_startup_model_load_token)
        _startup_model_load_token = None

    if _compile_desired["vae"]:
        token = _push_backend_activity("torch_compile", "Reapplying VAE compile after model load")
        try:
            apply_torch_compile_settings(**_compile_desired)
        finally:
            _pop_backend_activity(token)


def apply_cudnn_benchmark(enabled: bool) -> dict[str, Any]:
    try:
        import torch

        torch.backends.cudnn.benchmark = bool(enabled)
        return {
            "ok": True,
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc), "cudnn_benchmark": None}


def _normalize_cache_targets(targets: Any | None) -> set[str]:
    if targets is None:
        return {"c", "uc", "hr_c", "hr_uc", "img2img_init"}
    if isinstance(targets, str):
        targets = [targets]
    elif isinstance(targets, dict):
        targets = [key for key, enabled in targets.items() if enabled]

    normalized: set[str] = set()
    for target in targets or []:
        name = str(target).strip().lower().replace("-", "_")
        if name in {"all", "cond", "conds", "conditioning"}:
            normalized.update({"c", "uc", "hr_c", "hr_uc", "img2img_init"})
        elif name in {"prompt", "prompts", "txt2img"}:
            normalized.update({"c", "uc", "hr_c", "hr_uc"})
        elif name in {"base", "base_cond"}:
            normalized.update({"c", "uc"})
        elif name in {"hr", "hires", "hires_fix"}:
            normalized.update({"hr_c", "hr_uc"})
        elif name in {"img2img", "img2img_init", "init"}:
            normalized.add("img2img_init")
        elif name in {"c", "uc", "hr_c", "hr_uc"}:
            normalized.add(name)
    return normalized


def clear_cond_cache(targets: Any | None = None) -> dict:
    """Clear A1111 reusable generation caches."""
    global _last_cleared_at

    normalized_targets = _normalize_cache_targets(targets)
    cleared = []
    if "c" in normalized_targets:
        StableDiffusionProcessing.cached_c = [None, None]
        cleared.append("StableDiffusionProcessing.cached_c")
    if "uc" in normalized_targets:
        StableDiffusionProcessing.cached_uc = [None, None]
        cleared.append("StableDiffusionProcessing.cached_uc")
    if "hr_c" in normalized_targets:
        StableDiffusionProcessingTxt2Img.cached_hr_c = [None, None]
        cleared.append("StableDiffusionProcessingTxt2Img.cached_hr_c")
    if "hr_uc" in normalized_targets:
        StableDiffusionProcessingTxt2Img.cached_hr_uc = [None, None]
        cleared.append("StableDiffusionProcessingTxt2Img.cached_hr_uc")
    if "img2img_init" in normalized_targets:
        StableDiffusionProcessingImg2Img.clear_img2img_init_cache()
        cleared.append("StableDiffusionProcessing.cached_img2img_init")

    _last_cleared_at = time.time()
    return {
        "ok": True,
        "cleared_at": _last_cleared_at,
        "cleared": cleared,
        "targets": sorted(normalized_targets),
    }


def on_app_started(_: object, app: FastAPI) -> None:
    _install_backend_status_hooks()

    @app.post("/sdapi/v1/openclaw/clear-cond-cache")
    async def _clear_cond_cache(request: Request):
        data = {}
        try:
            data = await request.json()
        except Exception:
            pass
        targets = data.get("targets", data.get("target")) if isinstance(data, dict) else None
        return clear_cond_cache(targets)

    @app.post("/sdapi/v1/openclaw/token-count")
    async def _token_count(request: Request):
        data = await request.json()
        text = str(data.get("text") or "")
        try:
            steps = int(data.get("steps") or 20)
        except (TypeError, ValueError):
            steps = 20
        return estimate_token_count(text, steps)

    @app.post("/sdapi/v1/openclaw/token_counter")
    async def _token_counter_compat(request: Request):
        return await _token_count(request)


    @app.post("/sdapi/v1/openclaw/torch-compile")
    async def _torch_compile(request: Request):
        data = await request.json()
        target = data.get("target")
        with call_queue.queue_lock:
            if target == "vae-only":
                return apply_torch_compile_settings(vae=True)
            return apply_torch_compile_settings(vae=bool(data.get("vae") or data.get("enabled")))

    @app.get("/sdapi/v1/openclaw/torch-compile")
    async def _torch_compile_status():
        return {"ok": True, "desired": dict(_compile_desired), "status": dict(_compile_status)}

    @app.get("/sdapi/v1/openclaw/backend-status")
    async def _backend_status():
        return _backend_status_payload()

    @app.post("/sdapi/v1/openclaw/cudnn-benchmark")
    async def _cudnn_benchmark(request: Request):
        data = await request.json()
        return apply_cudnn_benchmark(bool(data.get("enabled")))

    @app.get("/sdapi/v1/openclaw/cudnn-benchmark")
    async def _cudnn_benchmark_status():
        try:
            import torch

            return {"ok": True, "cudnn_benchmark": bool(torch.backends.cudnn.benchmark)}
        except Exception as exc:
            return {"ok": False, "error": str(exc), "cudnn_benchmark": None}

    @app.post("/sdapi/v1/openclaw/model-merge")
    async def _model_merge(request: Request):
        data = await request.json()
        return _run_openclaw_model_merge(data if isinstance(data, dict) else {})

    @app.get("/sdapi/v1/openclaw/training-templates")
    async def _training_template_choices():
        return _training_templates()

    @app.get("/sdapi/v1/openclaw/cond-cache")
    async def _cond_cache_status():
        return {
            "ok": True,
            "last_cleared_at": _last_cleared_at,
            "cached": {
                "c": StableDiffusionProcessing.cached_c[0] is not None,
                "uc": StableDiffusionProcessing.cached_uc[0] is not None,
                "hr_c": StableDiffusionProcessingTxt2Img.cached_hr_c[0] is not None,
                "hr_uc": StableDiffusionProcessingTxt2Img.cached_hr_uc[0] is not None,
                "img2img_init": StableDiffusionProcessing.cached_img2img_init[0] is not None,
            },
            "img2img_init": StableDiffusionProcessingImg2Img.img2img_init_cache_status(),
        }


# Install backend activity hooks as soon as the extension script is imported.
# A1111 starts its initial model load before app_started, so waiting until the
# API is mounted misses exactly the startup/model-load work this status lane is
# meant to expose. The installer is idempotent and still called from
# on_app_started as a safety net for reload paths.
_install_backend_status_hooks()

# Keep a broad startup/model-load activity on the stack until A1111 fires the
# model_loaded callback. Some startup work is already inside original call
# frames before extension hooks can wrap them, so this fills the unavoidable
# gaps without adding measurable work to generation itself.
try:
    if not getattr(sd_models.model_data, "was_loaded_at_least_once", False):
        _startup_model_load_token = _push_backend_activity("startup_model_load", "Starting Web UI / loading initial model")
except Exception:
    _startup_model_load_token = None

script_callbacks.on_app_started(on_app_started)
script_callbacks.on_model_loaded(on_model_loaded)
