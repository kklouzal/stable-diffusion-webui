from __future__ import annotations

import os
import threading
import time
from functools import reduce
from typing import Any

from fastapi import FastAPI, Request

from modules import call_queue, extra_networks, prompt_parser, script_callbacks, sd_models, shared
from modules.processing import StableDiffusionProcessing, StableDiffusionProcessingImg2Img, StableDiffusionProcessingTxt2Img

_last_cleared_at = 0.0
_compile_slots: dict[str, dict[str, Any]] = {}
_compile_desired: dict[str, bool] = {"vae": False}
_compile_status: dict[str, Any] = {"vae": False, "last_error": None}
_backend_activity_lock = threading.Lock()
_backend_activity_stack: list[dict[str, Any]] = []
_backend_activity_token = 0
_backend_hooks_installed = False
_backend_lora_batch: dict[str, Any] = {"total": 0, "index": 0}

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

        try:
            from modules_forge import forge_version  # noqa: F401
            forge = True
        except Exception:
            forge = False

        flat_prompts = reduce(lambda list1, list2: list1 + list2, prompt_schedules, [])
        prompts = [prompt_text for _step, prompt_text in flat_prompts] or [text or ""]

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
    global _backend_hooks_installed
    if _backend_hooks_installed:
        return

    try:
        from modules import processing as _processing
        from modules import sd_models as _sd_models
        from modules.api import api as _api_module
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

    def _wrap_process_images(original):
        def wrapped(p, *args, **kwargs):
            is_img2img = isinstance(p, StableDiffusionProcessingImg2Img)
            phase = "img2img_pipeline" if is_img2img else "generation_pipeline"
            label = "Running img2img pipeline" if is_img2img else "Running generation pipeline"
            detail = getattr(p, "sd_model_checkpoint", None) or getattr(getattr(shared, "sd_model", None), "sd_checkpoint_info", None)
            token = _push_backend_activity(phase, label, detail=_checkpoint_detail(detail))
            try:
                return original(p, *args, **kwargs)
            finally:
                _pop_backend_activity(token)
        return wrapped

    def _wrap_load_vae(original):
        def wrapped(model, vae_file=None, vae_source="from unknown source"):
            detail = os.path.basename(str(vae_file)) if vae_file else str(vae_source)
            token = _push_backend_activity("vae_load", "Loading VAE", detail=detail)
            try:
                return original(model, vae_file=vae_file, vae_source=vae_source)
            finally:
                _pop_backend_activity(token)
        return wrapped

    _wrap_backend_function(_processing, "process_images", _wrap_process_images)
    _wrap_backend_function(_api_module, "process_images", _wrap_process_images)
    _wrap_backend_function(_sd_models, "reload_model_weights", _wrap_reload_model_weights)
    _wrap_backend_function(_sd_models, "load_model", _wrap_load_model)
    _wrap_backend_function(_sd_models, "get_checkpoint_state_dict", _wrap_get_checkpoint_state_dict)
    _wrap_backend_function(_sd_models, "load_model_weights", _wrap_load_model_weights)
    _wrap_backend_function(_sd_vae, "load_vae", _wrap_load_vae)

    try:
        import networks as _lora_networks
    except Exception:
        _lora_networks = None

    if _lora_networks is not None:
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

        _wrap_backend_function(_lora_networks, "load_networks", _wrap_load_networks)
        _wrap_backend_function(_lora_networks, "load_network", _wrap_load_network)


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


def on_model_loaded(_: Any) -> None:
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


def clear_cond_cache() -> dict:
    """Clear A1111 prompt-conditioning caches used by persistent_cond_cache."""
    global _last_cleared_at

    StableDiffusionProcessing.cached_c = [None, None]
    StableDiffusionProcessing.cached_uc = [None, None]
    StableDiffusionProcessingTxt2Img.cached_hr_c = [None, None]
    StableDiffusionProcessingTxt2Img.cached_hr_uc = [None, None]

    _last_cleared_at = time.time()
    return {
        "ok": True,
        "cleared_at": _last_cleared_at,
        "cleared": [
            "StableDiffusionProcessing.cached_c",
            "StableDiffusionProcessing.cached_uc",
            "StableDiffusionProcessingTxt2Img.cached_hr_c",
            "StableDiffusionProcessingTxt2Img.cached_hr_uc",
        ],
    }


def on_app_started(_: object, app: FastAPI) -> None:
    _install_backend_status_hooks()

    @app.post("/sdapi/v1/openclaw/clear-cond-cache")
    async def _clear_cond_cache():
        return clear_cond_cache()

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
            },
        }


script_callbacks.on_app_started(on_app_started)
script_callbacks.on_model_loaded(on_model_loaded)
