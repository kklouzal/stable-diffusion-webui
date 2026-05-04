from __future__ import annotations

import time

from fastapi import FastAPI

from modules import script_callbacks
from modules.processing import StableDiffusionProcessing, StableDiffusionProcessingTxt2Img

_last_cleared_at = 0.0


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
