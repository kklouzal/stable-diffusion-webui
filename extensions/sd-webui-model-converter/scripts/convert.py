from __future__ import annotations

import os
import traceback
from typing import Any

import safetensors.torch
import torch
from torch import Tensor

from modules import sd_models, sd_vae, shared

DTYPES_TO_FP16 = {torch.float32, torch.float64, torch.bfloat16}
DTYPES_TO_BF16 = {torch.float32, torch.float64, torch.float16}
DTYPES_TO_FLOAT8 = {torch.float32, torch.float64, torch.bfloat16, torch.float16}
PART_ACTIONS = {"copy", "convert", "delete"}
PRECISIONS = {"full", "fp32", "fp16", "bf16", "float8_e4m3fn", "float8_e5m2"}
FORMATS = {"ckpt", "safetensors"}


class MockModelInfo:
    def __init__(self, model_path: str) -> None:
        self.filepath = model_path
        self.filename = os.path.basename(model_path)
        self.model_name = os.path.splitext(self.filename)[0]


def conv_full(t: Tensor) -> Tensor:
    return t


def conv_fp16(t: Tensor) -> Tensor:
    return t.half() if t.dtype in DTYPES_TO_FP16 else t


def conv_bf16(t: Tensor) -> Tensor:
    return t.bfloat16() if t.dtype in DTYPES_TO_BF16 else t


def conv_float8_e4m3fn(t: Tensor) -> Tensor:
    return t.to(torch.float8_e4m3fn) if t.dtype in DTYPES_TO_FLOAT8 else t


def conv_float8_e5m2(t: Tensor) -> Tensor:
    return t.to(torch.float8_e5m2) if t.dtype in DTYPES_TO_FLOAT8 else t


PRECISION_FUNCS = {
    "full": conv_full,
    "fp32": conv_full,
    "fp16": conv_fp16,
    "bf16": conv_bf16,
    "float8_e4m3fn": conv_float8_e4m3fn,
    "float8_e5m2": conv_float8_e5m2,
}


def check_weight_type(k: str) -> str:
    if k.startswith("model.diffusion_model"):
        return "unet"
    if k.startswith("first_stage_model"):
        return "vae"
    if k.startswith("cond_stage_model") or k.startswith("conditioner.embedders"):
        return "clip"
    return "other"


def load_model(path: str) -> dict[str, Any]:
    if path.endswith(".safetensors"):
        loaded = safetensors.torch.load_file(path, device="cpu")
    else:
        loaded = torch.load(path, map_location="cpu")
    return loaded.get("state_dict", loaded) if isinstance(loaded, dict) else loaded


def position_id_keys(model: dict[str, Any]) -> list[str]:
    return [str(key) for key in model.keys() if str(key).endswith("position_ids")]


def standard_position_ids_like(tensor: Tensor | None = None) -> Tensor:
    length = int(tensor.shape[-1]) if isinstance(tensor, Tensor) and tensor.ndim > 0 else 77
    ids = torch.arange(length, dtype=torch.int64).unsqueeze(0)
    if isinstance(tensor, Tensor) and tensor.shape == ids.shape:
        return ids
    if isinstance(tensor, Tensor) and tensor.numel() == ids.numel():
        return ids.reshape(tensor.shape)
    return ids


def fix_model(model: dict[str, Any], fix_clip: bool = False, force_position_id: bool = False) -> dict[str, Any]:
    nai_keys = {
        "cond_stage_model.transformer.embeddings.": "cond_stage_model.transformer.text_model.embeddings.",
        "cond_stage_model.transformer.encoder.": "cond_stage_model.transformer.text_model.encoder.",
        "cond_stage_model.transformer.final_layer_norm.": "cond_stage_model.transformer.text_model.final_layer_norm.",
    }
    fallback_position_id_key = "cond_stage_model.transformer.text_model.embeddings.position_ids"
    for key in list(model.keys()):
        for prefix, replacement in nai_keys.items():
            if isinstance(key, str) and key.startswith(prefix):
                new_key = key.replace(prefix, replacement)
                model[new_key] = model[key]
                del model[key]
                print(f"[OpenClaw Model Converter] Fixed NovelAI CLIP key {key}")
                break

    keys = position_id_keys(model)

    if force_position_id:
        for key in keys:
            model[key] = model[key].to(torch.int64)

    if fix_clip:
        if keys:
            for key in keys:
                correct = standard_position_ids_like(model[key])
                now = model[key].to(torch.int64)
                if now.shape == correct.shape and torch.equal(now, correct):
                    print(f"[OpenClaw Model Converter] CLIP position_ids already look correct: {key}")
                    model[key] = now
                    continue
                model[key] = correct
                print(f"[OpenClaw Model Converter] Fixed broken CLIP position_ids: {key}")
        else:
            print("[OpenClaw Model Converter] Missing CLIP position_ids; adding standard 0..76 tensor")
            model[fallback_position_id_key] = standard_position_ids_like()
    return model


def is_sdxl_model(model: dict[str, Any]) -> bool:
    return any(str(k).startswith("conditioner.embedders") for k in model.keys())


def normalize_part_action(value: str, default: str = "convert") -> str:
    return value if value in PART_ACTIONS else default


def converter_options() -> dict[str, Any]:
    sd_vae.refresh_vae_list()
    sd_models.list_models()
    return {
        "models": [
            {"title": title, "model_name": info.model_name, "filename": info.filename}
            for title, info in sorted(sd_models.checkpoints_list.items(), key=lambda item: item[0].lower())
        ],
        "vaes": ["None", *sorted(sd_vae.vae_dict.keys(), key=str.lower)],
        "precisions": ["fp32", "fp16", "bf16", "float8_e4m3fn", "float8_e5m2"],
        "pruning_methods": ["disabled", "no-ema", "ema-only"],
        "formats": ["safetensors", "ckpt"],
        "part_actions": ["convert", "copy", "delete"],
    }


def resolve_model_info(model: str) -> MockModelInfo | None:
    sd_models.list_models()
    if info := sd_models.checkpoints_list.get(model):
        return MockModelInfo(info.filename)
    for info in sd_models.checkpoints_list.values():
        if model in {info.model_name, info.filename, os.path.basename(info.filename)}:
            return MockModelInfo(info.filename)
    if model and os.path.exists(model):
        return MockModelInfo(model)
    return None


def convert_warp(path_mode, model_name, model_path, directory, *args):
    match path_mode:
        case 0:
            if model_info := resolve_model_info(model_name):
                return do_convert(model_info, *args)
            return "Error: model not found"
        case 1:
            if os.path.exists(model_path):
                return do_convert(MockModelInfo(model_path), *args)
            return f'Error: model path "{model_path}" does not exist'
        case 2:
            if not os.path.isdir(directory):
                return f'Error: path "{directory}" does not exist or is not a directory'
            files = [f for f in os.listdir(directory) if f.endswith((".ckpt", ".safetensors"))]
            if not files:
                return "Error: no checkpoints found in directory"
            _args = list(args)
            _args[3] = ""
            for filename in files:
                do_convert(MockModelInfo(os.path.join(directory, filename)), *_args)
            return "Batch processing done"
        case _:
            return f"Error: unknown mode {path_mode}"


def do_convert(
    model_info: MockModelInfo,
    checkpoint_formats,
    precision,
    conv_type,
    custom_name,
    bake_in_vae,
    unet_conv,
    text_encoder_conv,
    vae_conv,
    others_conv,
    fix_clip,
    force_position_id,
    delete_known_junk_data,
):
    checkpoint_formats = [fmt for fmt in checkpoint_formats if fmt in FORMATS]
    if not checkpoint_formats:
        return "Error: choose at least one model save format"
    if precision not in PRECISIONS:
        return f"Error: unsupported precision {precision}"

    extra_opt = {
        "unet": normalize_part_action(unet_conv),
        "clip": normalize_part_action(text_encoder_conv),
        "vae": normalize_part_action(vae_conv),
        "other": normalize_part_action(others_conv),
    }
    shared.state.begin()
    try:
        shared.state.job = "model-convert"
        shared.state.textinfo = f"Loading {model_info.filename}..."
        print(f"[OpenClaw Model Converter] Loading {model_info.filepath}...")

        ok = {}
        state_dict = load_model(model_info.filepath)
        fix_model(state_dict, fix_clip=fix_clip, force_position_id=force_position_id)

        conv_func = PRECISION_FUNCS[precision]

        def should_preserve_int64(weight_key: str, tensor: Tensor) -> bool:
            # CLIP position_ids are token indices, not learned floating weights. They must not
            # be converted to fp16/bf16/float8. SDXL uses conditioner.embedders.* names.
            return force_position_id and str(weight_key).endswith("position_ids")

        def handle_weight(weight_key: str, tensor: Tensor) -> None:
            if not isinstance(tensor, Tensor):
                return
            action = extra_opt[check_weight_type(weight_key)]
            if action == "convert":
                if should_preserve_int64(weight_key, tensor):
                    ok[weight_key] = tensor.to(torch.int64)
                elif not torch.is_floating_point(tensor):
                    ok[weight_key] = tensor
                else:
                    ok[weight_key] = conv_func(tensor)
            elif action == "copy":
                ok[weight_key] = tensor.to(torch.int64) if should_preserve_int64(weight_key, tensor) else tensor

        print("[OpenClaw Model Converter] Converting model...")
        if conv_type == "ema-only":
            for key in state_dict:
                ema_key = "model_ema." + key[6:].replace(".", "") if len(key) >= 6 else "___"
                if ema_key in state_dict:
                    handle_weight(key, state_dict[ema_key])
                elif not key.startswith("model_ema.") or key in ["model_ema.num_updates", "model_ema.decay"]:
                    handle_weight(key, state_dict[key])
        elif conv_type == "no-ema":
            for key, value in state_dict.items():
                if "model_ema." not in key:
                    handle_weight(key, value)
        else:
            for key, value in state_dict.items():
                handle_weight(key, value)

        if delete_known_junk_data:
            known_junk_prefixes = ("embedding_manager.embedder.", "lora_te_text_model", "control_model.")
            for key in [k for k in ok if any(str(k).startswith(prefix) for prefix in known_junk_prefixes)]:
                del ok[key]

        bake_in_vae_filename = sd_vae.vae_dict.get(bake_in_vae)
        if bake_in_vae_filename is not None:
            print(f"[OpenClaw Model Converter] Baking in VAE from {bake_in_vae_filename}")
            vae_dict = sd_vae.load_vae_dict(bake_in_vae_filename, map_location="cpu")
            for key, value in vae_dict.items():
                handle_weight(key, value)
            del vae_dict

        ckpt_dir = os.path.dirname(model_info.filepath)
        save_name = custom_name.strip() if custom_name else f"{model_info.model_name}-{precision}"
        if conv_type != "disabled" and not custom_name:
            save_name += f"-{conv_type}"
        if fix_clip and not custom_name:
            save_name += "-clip-fix"

        output = ""
        for fmt in checkpoint_formats:
            ext = ".safetensors" if fmt == "safetensors" else ".ckpt"
            save_path = os.path.join(ckpt_dir, save_name + ext)
            print(f"[OpenClaw Model Converter] Saving to {save_path}...")
            if fmt == "safetensors":
                safetensors.torch.save_file(ok, save_path)
            else:
                torch.save({"state_dict": ok}, save_path)
            output += f"Checkpoint saved to {save_path}\n"
        return output.rstrip()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        shared.state.end()


def convert_single(payload: dict[str, Any]) -> str:
    model_info = resolve_model_info(str(payload.get("model") or ""))
    if not model_info:
        raise ValueError("selected model was not found")
    return do_convert(
        model_info,
        payload.get("formats") or ["safetensors"],
        payload.get("precision") or "fp16",
        payload.get("pruning") or "disabled",
        payload.get("custom_name") or "",
        payload.get("bake_in_vae") or "None",
        payload.get("unet") or "convert",
        payload.get("clip") or "convert",
        payload.get("vae") or "convert",
        payload.get("other") or "convert",
        bool(payload.get("fix_clip")),
        bool(payload.get("force_position_id", True)),
        bool(payload.get("delete_known_junk_data")),
    )
