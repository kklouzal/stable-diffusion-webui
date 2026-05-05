from __future__ import annotations

import time
from functools import reduce
from typing import Any

from fastapi import FastAPI, Request

from modules import call_queue, extra_networks, prompt_parser, script_callbacks, sd_models, shared
from modules.processing import StableDiffusionProcessing, StableDiffusionProcessingTxt2Img

_last_cleared_at = 0.0
_compile_slots: dict[str, dict[str, Any]] = {}
_compile_desired: dict[str, bool] = {"main_model": False, "vae": False, "lora": False}
_compile_status: dict[str, Any] = {"main_model": False, "vae": False, "lora": False, "last_error": None}

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




def _get_main_model_module():
    sd_model = sd_models.model_data.sd_model
    return getattr(getattr(sd_model, "model", None), "diffusion_model", None)


def _set_main_model_module(module: Any) -> None:
    sd_models.model_data.sd_model.model.diffusion_model = module


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

        # If A1111 reloaded the checkpoint/VAE, the module identity changes. Compile the
        # current raw slot instead of reusing a stale wrapper from the previous model.
        original = unwrapped
        original.eval()
        compiled = torch.compile(original, mode="reduce-overhead", fullgraph=False, dynamic=True)
        setter(compiled)
        _compile_slots[name] = {
            "original": original,
            "original_id": id(original),
            "compiled_id": id(compiled),
        }
        _compile_status[name] = True
        return {"name": name, "enabled": True, "changed": True, "mode": "reduce-overhead", "dynamic": True}

    if slot and id(module) == slot.get("compiled_id"):
        setter(slot["original"])
        _compile_slots.pop(name, None)
        _compile_status[name] = False
        return {"name": name, "enabled": False, "changed": True}

    if module is not unwrapped:
        setter(unwrapped)
        _compile_slots.pop(name, None)
        _compile_status[name] = False
        return {"name": name, "enabled": False, "changed": True}

    _compile_slots.pop(name, None)
    _compile_status[name] = False
    return {"name": name, "enabled": False, "changed": False}


def apply_torch_compile_settings(main_model: bool = False, vae: bool = False, lora: bool = False) -> dict[str, Any]:
    results = []
    _compile_status["last_error"] = None
    _compile_desired["main_model"] = bool(main_model)
    _compile_desired["vae"] = bool(vae)
    _compile_desired["lora"] = bool(lora)
    try:
        results.append(_compile_module_slot("main_model", _compile_desired["main_model"], _get_main_model_module, _set_main_model_module))
        results.append(_compile_module_slot("vae", _compile_desired["vae"], _get_vae_module, _set_vae_module))
        # LoRA in this A1111 path is dynamically patched during prompt activation rather than a stable module slot.
        # Keep the setting/status first-class now, but only report it as requested until we add a safe per-network hook.
        _compile_status["lora"] = _compile_desired["lora"]
        results.append({"name": "lora", "enabled": _compile_desired["lora"], "changed": False, "note": "LoRA compile flag recorded; dynamic LoRA modules are not compiled yet."})
        return {"ok": True, "desired": dict(_compile_desired), "status": dict(_compile_status), "results": results}
    except Exception as exc:
        _compile_status["last_error"] = str(exc)
        return {"ok": False, "error": str(exc), "desired": dict(_compile_desired), "status": dict(_compile_status), "results": results}


def on_model_loaded(_: Any) -> None:
    if _compile_desired["main_model"] or _compile_desired["vae"]:
        apply_torch_compile_settings(**_compile_desired)


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
                return apply_torch_compile_settings(main_model=False, vae=True, lora=False)
            return apply_torch_compile_settings(
                main_model=bool(data.get("main_model")),
                vae=bool(data.get("vae")),
                lora=bool(data.get("lora")),
            )

    @app.get("/sdapi/v1/openclaw/torch-compile")
    async def _torch_compile_status():
        return {"ok": True, "desired": dict(_compile_desired), "status": dict(_compile_status)}

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
