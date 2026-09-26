from __future__ import annotations

import traceback

from fastapi import FastAPI
from pydantic import BaseModel, Field

from modules import script_callbacks
from scripts import convert


class ConvertRequest(BaseModel):
    mode: str = "checkpoint"
    model: str = ""
    lora: str = ""
    lora_precision: str = "bf16"
    formats: list[str] = Field(default_factory=lambda: ["safetensors"])
    precision: str = "fp16"
    pruning: str = "disabled"
    custom_name: str = ""
    bake_in_vae: str = "None"
    unet: str = "convert"
    clip: str = "convert"
    vae: str = "convert"
    other: str = "convert"
    unet_precision: str = "inherit"
    clip_precision: str = "inherit"
    vae_precision: str = "inherit"
    other_precision: str = "inherit"
    fix_clip: bool = False
    force_position_id: bool = True
    delete_known_junk_data: bool = False


def on_app_started(_: object, app: FastAPI) -> None:
    @app.get("/sdapi/v1/openclaw/model-converter/options")
    def openclaw_model_converter_options():
        try:
            return {"ok": True, **convert.converter_options()}
        except Exception as exc:
            return {
                "ok": False,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "models": [],
                "vaes": [],
            }

    @app.post("/sdapi/v1/openclaw/model-converter/convert")
    def openclaw_model_converter_convert(request: ConvertRequest):
        try:
            payload = (
                request.model_dump()
                if hasattr(request, "model_dump")
                else request.dict()
            )
            result = convert.convert_single(payload)
            return {"ok": True, "result": result}
        except Exception as exc:
            return {"ok": False, "error": str(exc), "traceback": traceback.format_exc()}


script_callbacks.on_app_started(on_app_started)
