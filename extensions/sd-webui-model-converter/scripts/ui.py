from __future__ import annotations

import traceback

from modules import headless_ui as gr
from fastapi import FastAPI
from pydantic import BaseModel, Field

from modules import script_callbacks, sd_models, sd_vae
from modules.ui import create_refresh_button
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


def gr_show(visible=True):
    return {"visible": visible, "__type__": "update"}


def add_tab():
    with gr.Blocks(analytics_enabled=False) as ui:
        with gr.Row(equal_height=True):
            with gr.Column(variant="panel"):
                gr.HTML(value="<p>OpenClaw-owned converter for checkpoints and LoRAs. Outputs are saved next to the source file.</p>")
                mode = gr.Radio(choices=["checkpoint", "lora"], value="checkpoint", label="Mode")
                with gr.Row():
                    model_name = gr.Dropdown(sd_models.checkpoint_tiles(), elem_id="model_converter_model_name", label="Checkpoint")
                    create_refresh_button(model_name, sd_models.list_models, lambda: {"choices": sd_models.checkpoint_tiles()}, "refresh_checkpoint_Z")
                lora_name = gr.Dropdown([item["title"] for item in convert.list_loras()], label="LoRA")
                lora_precision = gr.Radio(choices=["fp32", "fp16", "bf16"], value="bf16", label="LoRA precision")
                custom_name = gr.Textbox(label="Custom Name (Optional)")

                with gr.Row():
                    precision = gr.Radio(choices=["fp32", "fp16", "bf16", "float8_e4m3fn", "float8_e5m2"], value="fp16", label="Precision")
                    m_type = gr.Radio(choices=["disabled", "no-ema", "ema-only"], value="disabled", label="Pruning Methods")

                with gr.Row():
                    checkpoint_formats = gr.CheckboxGroup(choices=["ckpt", "safetensors"], value=["safetensors"], label="Checkpoint Format")
                    show_extra_options = gr.Checkbox(label="Show part actions", value=False)

                with gr.Row():
                    bake_in_vae = gr.Dropdown(choices=["None"] + list(sd_vae.vae_dict), value="None", label="Bake in VAE")
                    create_refresh_button(bake_in_vae, sd_vae.refresh_vae_list, lambda: {"choices": ["None"] + list(sd_vae.vae_dict)}, "model_converter_refresh_bake_in_vae")

                with gr.Row():
                    force_position_id = gr.Checkbox(label="Force CLIP position_id to int64 before convert", value=True)
                    fix_clip = gr.Checkbox(label="Fix CLIP", value=False)
                    delete_known_junk_data = gr.Checkbox(label="Cleanup known junk", value=False)

                with gr.Row(visible=False) as extra_options:
                    specific_part_conv = ["copy", "convert", "delete"]
                    unet_conv = gr.Dropdown(specific_part_conv, value="convert", label="UNet")
                    text_encoder_conv = gr.Dropdown(specific_part_conv, value="convert", label="Text encoder / CLIP")
                    vae_conv = gr.Dropdown(specific_part_conv, value="convert", label="VAE")
                    others_conv = gr.Dropdown(specific_part_conv, value="convert", label="Other weights")

                with gr.Row(visible=False) as precision_options:
                    specific_precision = ["inherit", "fp32", "fp16", "bf16", "float8_e4m3fn", "float8_e5m2"]
                    unet_precision = gr.Dropdown(specific_precision, value="inherit", label="UNet precision")
                    clip_precision = gr.Dropdown(specific_precision, value="inherit", label="CLIP precision")
                    vae_precision = gr.Dropdown(specific_precision, value="inherit", label="VAE precision")
                    other_precision = gr.Dropdown(specific_precision, value="inherit", label="Other precision")

                model_converter_convert = gr.Button(value="Convert model", elem_id="model_converter_convert", variant="primary")

            with gr.Column(variant="panel"):
                submit_result = gr.Textbox(elem_id="model_converter_result", show_label=False)

            show_extra_options.change(fn=lambda x: (gr_show(x), gr_show(x)), inputs=[show_extra_options], outputs=[extra_options, precision_options])

            model_converter_convert.click(
                fn=lambda mode, model_name, lora_name, lora_precision, checkpoint_formats, precision, m_type, custom_name, bake_in_vae, unet_conv, text_encoder_conv, vae_conv, others_conv, unet_precision, clip_precision, vae_precision, other_precision, fix_clip, force_position_id, delete_known_junk_data: convert.convert_single({
                    "mode": mode,
                    "model": model_name,
                    "lora": lora_name,
                    "lora_precision": lora_precision,
                    "formats": checkpoint_formats,
                    "precision": precision,
                    "pruning": m_type,
                    "custom_name": custom_name,
                    "bake_in_vae": bake_in_vae,
                    "unet": unet_conv,
                    "clip": text_encoder_conv,
                    "vae": vae_conv,
                    "other": others_conv,
                    "unet_precision": unet_precision,
                    "clip_precision": clip_precision,
                    "vae_precision": vae_precision,
                    "other_precision": other_precision,
                    "fix_clip": fix_clip,
                    "force_position_id": force_position_id,
                    "delete_known_junk_data": delete_known_junk_data,
                }),
                inputs=[mode, model_name, lora_name, lora_precision, checkpoint_formats, precision, m_type, custom_name, bake_in_vae, unet_conv, text_encoder_conv, vae_conv, others_conv, unet_precision, clip_precision, vae_precision, other_precision, fix_clip, force_position_id, delete_known_junk_data],
                outputs=[submit_result],
            )

    return [(ui, "Model Converter", "model_converter")]


def on_app_started(_: object, app: FastAPI) -> None:
    @app.get("/sdapi/v1/openclaw/model-converter/options")
    def openclaw_model_converter_options():
        try:
            return {"ok": True, **convert.converter_options()}
        except Exception as exc:
            return {"ok": False, "error": str(exc), "traceback": traceback.format_exc(), "models": [], "vaes": []}

    @app.post("/sdapi/v1/openclaw/model-converter/convert")
    def openclaw_model_converter_convert(request: ConvertRequest):
        try:
            payload = request.model_dump() if hasattr(request, "model_dump") else request.dict()
            result = convert.convert_single(payload)
            return {"ok": True, "result": result}
        except Exception as exc:
            return {"ok": False, "error": str(exc), "traceback": traceback.format_exc()}


script_callbacks.on_ui_tabs(add_tab)
script_callbacks.on_app_started(on_app_started)
