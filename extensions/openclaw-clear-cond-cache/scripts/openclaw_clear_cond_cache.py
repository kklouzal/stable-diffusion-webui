from __future__ import annotations

import time
from functools import reduce
from typing import Any

from fastapi import FastAPI, Request

from modules import extra_networks, prompt_parser, script_callbacks, sd_models
from modules.processing import StableDiffusionProcessing, StableDiffusionProcessingTxt2Img

_last_cleared_at = 0.0

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
