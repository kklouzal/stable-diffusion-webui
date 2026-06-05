from __future__ import annotations

import json
import os
import time
import traceback
from collections import Counter
from typing import Any

import safetensors
import safetensors.torch
import torch
from torch import Tensor

from modules import paths, sd_models, sd_vae, shared

OPENCLAW_CONVERTER_VERSION = "2026-05-10.4"
DTYPES_TO_FP16 = {torch.float32, torch.float64, torch.bfloat16}
DTYPES_TO_BF16 = {torch.float32, torch.float64, torch.float16}
DTYPES_TO_FLOAT8 = {torch.float32, torch.float64, torch.bfloat16, torch.float16}
PART_ACTIONS = {"copy", "convert", "delete"}
PRECISIONS = {"full", "fp32", "fp16", "bf16", "float8_e4m3fn", "float8_e5m2"}
COMPONENT_PRECISIONS = {"inherit", *PRECISIONS}
FORMATS = {"ckpt", "safetensors"}
LORA_PRECISIONS = {"fp32", "fp16", "bf16"}
KNOWN_JUNK_PREFIXES = (
    "embedding_manager.embedder.",
    "lora_te_text_model",
    "lora_unet",
    "lycoris_",
    "control_model.",
    "optimizer.",
    "optimizers.",
    "lr_schedulers.",
    "callbacks.",
    "loops.",
)
KNOWN_JUNK_EXACT = {"global_step", "pytorch-lightning_version"}


class MockModelInfo:
    def __init__(self, model_path: str) -> None:
        self.filepath = model_path
        self.filename = os.path.basename(model_path)
        self.model_name = os.path.splitext(self.filename)[0]


def conv_full(t: Tensor) -> Tensor:
    return t.float() if torch.is_floating_point(t) and t.dtype != torch.float32 else t


def conv_fp32(t: Tensor) -> Tensor:
    return t.float() if torch.is_floating_point(t) and t.dtype != torch.float32 else t


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
    "fp32": conv_fp32,
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
    return [str(key) for key in model if str(key).endswith("position_ids")]


def standard_position_ids_like(tensor: Tensor | None = None) -> Tensor:
    length = (
        int(tensor.shape[-1]) if isinstance(tensor, Tensor) and tensor.ndim > 0 else 77
    )
    ids = torch.arange(length, dtype=torch.int64).unsqueeze(0)
    if isinstance(tensor, Tensor) and tensor.shape == ids.shape:
        return ids
    if isinstance(tensor, Tensor) and tensor.numel() == ids.numel():
        return ids.reshape(tensor.shape)
    return ids


def fix_model(
    model: dict[str, Any], fix_clip: bool = False, force_position_id: bool = False
) -> dict[str, Any]:
    nai_keys = {
        "cond_stage_model.transformer.embeddings.": "cond_stage_model.transformer.text_model.embeddings.",
        "cond_stage_model.transformer.encoder.": "cond_stage_model.transformer.text_model.encoder.",
        "cond_stage_model.transformer.final_layer_norm.": "cond_stage_model.transformer.text_model.final_layer_norm.",
    }
    fallback_position_id_key = (
        "cond_stage_model.transformer.text_model.embeddings.position_ids"
    )
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
                    print(
                        f"[OpenClaw Model Converter] CLIP position_ids already look correct: {key}"
                    )
                    model[key] = now
                    continue
                model[key] = correct
                print(
                    f"[OpenClaw Model Converter] Fixed broken CLIP position_ids: {key}"
                )
        else:
            print(
                "[OpenClaw Model Converter] Missing CLIP position_ids; adding standard 0..76 tensor"
            )
            model[fallback_position_id_key] = standard_position_ids_like()
    return model


def is_sdxl_model(model: dict[str, Any]) -> bool:
    return any(str(k).startswith("conditioner.embedders") for k in model)


def model_family(model: dict[str, Any]) -> str:
    if is_sdxl_model(model):
        return "SDXL-like"
    if any(str(k).startswith("cond_stage_model.model.transformer") for k in model):
        return "SD2-like"
    if any(str(k).startswith("cond_stage_model.transformer") for k in model):
        return "SD1-like"
    return "unknown"


def normalize_part_action(value: str, default: str = "convert") -> str:
    return value if value in PART_ACTIONS else default


def normalize_precision(value: str, default: str = "inherit") -> str:
    return value if value in COMPONENT_PRECISIONS else default


def is_known_junk_key(key: str) -> bool:
    return key in KNOWN_JUNK_EXACT or any(
        key.startswith(prefix) for prefix in KNOWN_JUNK_PREFIXES
    )


def is_known_lora_junk_key(key: str) -> bool:
    # In a LoRA file, lora_unet/lora_te keys are the payload, not junk. Keep cleanup to
    # obvious training/runtime residue.
    lora_safe_prefixes = (
        "optimizer.",
        "optimizers.",
        "lr_schedulers.",
        "callbacks.",
        "loops.",
        "embedding_manager.embedder.",
        "control_model.",
    )
    return key in KNOWN_JUNK_EXACT or any(
        key.startswith(prefix) for prefix in lora_safe_prefixes
    )


def dtype_name(tensor: Tensor) -> str:
    return str(tensor.dtype).replace("torch.", "")


def summarize_position_ids(model: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for key in position_id_keys(model):
        tensor = model[key]
        entry = {"key": key, "dtype": dtype_name(tensor), "shape": list(tensor.shape)}
        try:
            correct = standard_position_ids_like(tensor)
            now = tensor.to(torch.int64)
            entry["standard_0_to_n"] = bool(
                now.shape == correct.shape and torch.equal(now, correct)
            )
        except Exception as exc:
            entry["standard_0_to_n"] = False
            entry["error"] = str(exc)
        out.append(entry)
    return out


def checkpoint_doctor(
    model: dict[str, Any], model_info: MockModelInfo
) -> dict[str, Any]:
    dtype_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    shape_issues = []
    non_tensor_keys = []
    huge_tensors = []
    for key, value in model.items():
        if not isinstance(value, Tensor):
            non_tensor_keys.append(str(key))
            continue
        dtype_counts[dtype_name(value)] += 1
        family_counts[check_weight_type(str(key))] += 1
        if value.ndim == 0:
            shape_issues.append({"key": str(key), "shape": []})
        if value.numel() > 100_000_000:
            huge_tensors.append(
                {
                    "key": str(key),
                    "shape": list(value.shape),
                    "dtype": dtype_name(value),
                }
            )
    keys = set(map(str, model.keys()))
    junk_keys = [key for key in keys if is_known_junk_key(key)]
    has_unet = any(key.startswith("model.diffusion_model") for key in keys)
    has_vae = any(key.startswith("first_stage_model") for key in keys)
    has_clip = any(
        key.startswith("cond_stage_model") or key.startswith("conditioner.embedders")
        for key in keys
    )
    position_ids = summarize_position_ids(model)
    warnings = []
    if not has_unet:
        warnings.append("No UNet/model.diffusion_model keys detected")
    if not has_clip:
        warnings.append("No CLIP/text-encoder keys detected")
    if not position_ids:
        warnings.append("No CLIP position_ids key detected")
    for item in position_ids:
        if item.get("dtype") != "int64":
            warnings.append(
                f"{item['key']} is {item.get('dtype')}, expected int64 for a clean converted checkpoint"
            )
        if not item.get("standard_0_to_n"):
            warnings.append(
                f"{item['key']} does not contain standard sequential position IDs"
            )
    if junk_keys:
        warnings.append(f"{len(junk_keys)} known junk/training-residue key(s) detected")
    try:
        stat = os.stat(model_info.filepath)
        source = {
            "path": model_info.filepath,
            "filename": model_info.filename,
            "size_bytes": stat.st_size,
            "mtime": int(stat.st_mtime),
        }
    except OSError:
        source = {"path": model_info.filepath, "filename": model_info.filename}
    return {
        "source": source,
        "family": model_family(model),
        "tensor_count": sum(dtype_counts.values()),
        "non_tensor_keys": non_tensor_keys[:50],
        "dtype_counts": dict(sorted(dtype_counts.items())),
        "component_counts": dict(sorted(family_counts.items())),
        "has_unet": has_unet,
        "has_clip": has_clip,
        "has_vae": has_vae,
        "position_ids": position_ids,
        "known_junk_count": len(junk_keys),
        "known_junk_examples": sorted(junk_keys)[:25],
        "shape_issue_examples": shape_issues[:25],
        "huge_tensor_examples": huge_tensors[:10],
        "warnings": warnings,
        "content_scan": "metadata-only; NaN/Inf full tensor scan intentionally skipped during conversion for speed",
    }


def scan_and_repair_nonfinite(
    model: dict[str, Any], *, repair: bool = True
) -> dict[str, Any]:
    total_tensors = 0
    affected_tensors = 0
    total_values = 0
    nan_values = 0
    posinf_values = 0
    neginf_values = 0
    examples = []
    for key, tensor in list(model.items()):
        if not isinstance(tensor, Tensor) or not torch.is_floating_point(tensor):
            continue
        total_tensors += 1
        nan_count = int(torch.isnan(tensor).sum().item())
        posinf_count = int(torch.isposinf(tensor).sum().item())
        neginf_count = int(torch.isneginf(tensor).sum().item())
        bad_count = nan_count + posinf_count + neginf_count
        if bad_count:
            affected_tensors += 1
            nan_values += nan_count
            posinf_values += posinf_count
            neginf_values += neginf_count
            total_values += bad_count
            if len(examples) < 25:
                examples.append(
                    {
                        "key": str(key),
                        "nan": nan_count,
                        "+inf": posinf_count,
                        "-inf": neginf_count,
                        "dtype": dtype_name(tensor),
                        "shape": list(tensor.shape),
                    }
                )
            if repair:
                model[key] = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
    return {
        "scanned_float_tensors": total_tensors,
        "affected_tensors": affected_tensors,
        "total_nonfinite_values": total_values,
        "nan_values": nan_values,
        "posinf_values": posinf_values,
        "neginf_values": neginf_values,
        "repaired": bool(repair and total_values),
        "examples": examples,
    }


def safetensors_metadata(path: str) -> dict[str, str]:
    if not str(path).endswith(".safetensors"):
        return {}
    try:
        with safetensors.safe_open(path, framework="pt", device="cpu") as f:
            return {str(k): str(v) for k, v in (f.metadata() or {}).items()}
    except Exception:
        return {}


def lora_dir() -> str:
    return os.path.join(paths.models_path, "Lora")


def list_loras() -> list[dict[str, str]]:
    root = lora_dir()
    out = []
    if not os.path.isdir(root):
        return out
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if not filename.lower().endswith((".safetensors", ".ckpt", ".pt")):
                continue
            path = os.path.join(dirpath, filename)
            rel = os.path.relpath(path, root)
            name = os.path.splitext(rel)[0].replace(os.sep, "/")
            out.append(
                {"name": name, "title": name, "filename": filename, "path": path}
            )
    return sorted(out, key=lambda item: item["title"].lower())


def resolve_lora_info(name: str) -> MockModelInfo | None:
    if name and os.path.exists(name):
        return MockModelInfo(name)
    for item in list_loras():
        candidates = {
            item["name"],
            item["title"],
            item["filename"],
            item["path"],
            os.path.basename(item["path"]),
        }
        if name in candidates:
            return MockModelInfo(item["path"])
    return None


def lora_component(key: str) -> str:
    lowered = key.lower()
    if "lora_unet" in lowered or lowered.startswith("unet") or ".unet." in lowered:
        return "unet"
    if (
        "lora_te" in lowered
        or "text_encoder" in lowered
        or "clip" in lowered
        or "transformer_text_model" in lowered
    ):
        return "clip"
    return "other"


def lora_family(model: dict[str, Any], metadata: dict[str, str]) -> str:
    module = (
        metadata.get("ss_network_module")
        or metadata.get("modelspec.architecture")
        or ""
    ).lower()
    keys = " ".join(list(map(str, model.keys()))[:200]).lower()
    if "dora" in module or "dora" in keys:
        return "DoRA-like"
    if "loha" in module or "hada_" in keys or ("lora_down" in keys and "hada" in keys):
        return "LoHa/LyCORIS-like"
    if "locon" in module or ("conv" in keys and "lora_down" in keys):
        return "LoCon-like"
    if "oft" in module or "oft_" in keys:
        return "OFT-like"
    if "lora" in module or "lora_down" in keys or "lora_up" in keys:
        return "LoRA-like"
    return "unknown"


def lora_doctor(
    model: dict[str, Any],
    model_info: MockModelInfo,
    metadata: dict[str, str],
    nonfinite: dict[str, Any] | None = None,
) -> dict[str, Any]:
    dtype_counts: Counter[str] = Counter()
    component_counts: Counter[str] = Counter()
    rank_examples = []
    keys = set(map(str, model.keys()))
    up_bases = {k.removesuffix(".lora_up.weight") for k in keys if k.endswith(".lora_up.weight")}
    down_bases = {k.removesuffix(".lora_down.weight") for k in keys if k.endswith(".lora_down.weight")}
    alpha_keys = sorted(k for k in keys if k.endswith(".alpha"))
    for key, value in model.items():
        if not isinstance(value, Tensor):
            continue
        dtype_counts[dtype_name(value)] += 1
        component_counts[lora_component(str(key))] += 1
        if (
            len(rank_examples) < 25
            and str(key).endswith(".lora_down.weight")
            and value.ndim >= 2
        ):
            base = str(key).rsplit(".", 1)[0]
            rank_examples.append(
                {"base": base, "rank": int(value.shape[0]), "shape": list(value.shape)}
            )
    warnings = []
    missing_up = sorted(down_bases - up_bases)
    missing_down = sorted(up_bases - down_bases)
    if missing_up:
        warnings.append(f"{len(missing_up)} lora_down tensors missing matching lora_up")
    if missing_down:
        warnings.append(
            f"{len(missing_down)} lora_up tensors missing matching lora_down"
        )
    if nonfinite and nonfinite.get("total_nonfinite_values"):
        warnings.append(
            f"{nonfinite.get('total_nonfinite_values')} NaN/Inf value(s) detected and repaired"
        )
    return {
        "source": {"path": model_info.filepath, "filename": model_info.filename},
        "family": lora_family(model, metadata),
        "tensor_count": sum(dtype_counts.values()),
        "dtype_counts": dict(sorted(dtype_counts.items())),
        "component_counts": dict(sorted(component_counts.items())),
        "metadata_key_count": len(metadata),
        "network_module": metadata.get("ss_network_module") or "",
        "base_model_version": metadata.get("ss_base_model_version")
        or metadata.get("modelspec.architecture")
        or "",
        "rank_examples": rank_examples,
        "alpha_key_count": len(alpha_keys),
        "missing_up_examples": missing_up[:25],
        "missing_down_examples": missing_down[:25],
        "known_junk_count": len([k for k in keys if is_known_lora_junk_key(k)]),
        "nonfinite": nonfinite or {},
        "warnings": warnings,
    }


def lora_metadata(
    model_info: MockModelInfo,
    original: dict[str, str],
    *,
    precision: str,
    doctor: dict[str, Any],
    cleanup: bool,
    nonfinite: dict[str, Any],
) -> dict[str, str]:
    metadata = dict(original)
    metadata.update(
        {
            "openclaw_lora_converter_version": OPENCLAW_CONVERTER_VERSION,
            "openclaw_lora_converter_source": model_info.filename,
            "openclaw_lora_converter_precision": precision,
            "openclaw_lora_converter_family": str(doctor.get("family") or "unknown"),
            "openclaw_lora_converter_cleanup_known_junk": str(bool(cleanup)),
            "openclaw_lora_converter_nonfinite_repair": json.dumps(
                nonfinite, sort_keys=True
            ),
            "openclaw_lora_converter_created_at": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
            ),
            "openclaw_lora_converter_doctor_summary": json.dumps(
                {
                    "family": doctor.get("family"),
                    "tensor_count": doctor.get("tensor_count"),
                    "dtype_counts": doctor.get("dtype_counts"),
                    "component_counts": doctor.get("component_counts"),
                    "warnings": doctor.get("warnings"),
                },
                sort_keys=True,
            ),
        }
    )
    return {str(k): str(v) for k, v in metadata.items()}


def convert_lora(payload: dict[str, Any]) -> str:
    model_info = resolve_lora_info(
        str(payload.get("lora") or payload.get("model") or "")
    )
    if not model_info:
        raise ValueError("selected LoRA was not found")
    precision = str(payload.get("lora_precision") or payload.get("precision") or "bf16")
    if precision not in LORA_PRECISIONS:
        raise ValueError(f"unsupported LoRA precision: {precision}")
    custom_name = str(payload.get("custom_name") or "").strip()
    cleanup = bool(payload.get("delete_known_junk_data"))
    shared.state.begin()
    try:
        shared.state.job = "lora-convert"
        shared.state.textinfo = f"Loading LoRA {model_info.filename}..."
        model = load_model(model_info.filepath)
        original_metadata = safetensors_metadata(model_info.filepath)
        source_nonfinite = scan_and_repair_nonfinite(model, repair=True)
        before_doctor = lora_doctor(
            model, model_info, original_metadata, source_nonfinite
        )
        ok: dict[str, Tensor] = {}
        for key, tensor in model.items():
            if not isinstance(tensor, Tensor):
                continue
            if cleanup and is_known_lora_junk_key(str(key)):
                continue
            if torch.is_floating_point(tensor):
                ok[key] = PRECISION_FUNCS[precision](tensor)
            else:
                ok[key] = tensor
        output_nonfinite = scan_and_repair_nonfinite(ok, repair=True)
        after_doctor = lora_doctor(ok, model_info, original_metadata, output_nonfinite)
        metadata = lora_metadata(
            model_info,
            original_metadata,
            precision=precision,
            doctor=after_doctor,
            cleanup=cleanup,
            nonfinite={"source": source_nonfinite, "output": output_nonfinite},
        )
        save_name = custom_name or f"{model_info.model_name}-{precision}"
        save_path = os.path.join(
            os.path.dirname(model_info.filepath), save_name + ".safetensors"
        )
        safetensors.torch.save_file(ok, save_path, metadata=metadata)
        refresh_after_convert("lora")
        report = {
            "source_doctor": before_doctor,
            "output_doctor": after_doctor,
            "metadata_added": {
                k: v
                for k, v in metadata.items()
                if k.startswith("openclaw_lora_converter_")
            },
        }
        return (
            f"LoRA saved to {save_path}\nOpenClaw LoRA doctor report:\n"
            + json.dumps(report, indent=2, sort_keys=True)
        )
    except Exception:
        traceback.print_exc()
        raise
    finally:
        shared.state.end()


def refresh_after_convert(kind: str = "checkpoint") -> None:
    try:
        sd_models.list_models()
    except Exception as exc:
        print(
            f"[OpenClaw Model Converter] Checkpoint refresh after conversion failed: {exc}"
        )
    if kind == "lora":
        try:
            import networks  # type: ignore  # noqa: PLC0415 - LoRA extension import is only available after WebUI extension setup.

            if hasattr(networks, "list_available_networks"):
                networks.list_available_networks()
        except Exception as exc:
            print(
                f"[OpenClaw Model Converter] LoRA refresh after conversion failed: {exc}"
            )


def converter_options() -> dict[str, Any]:
    sd_vae.refresh_vae_list()
    sd_models.list_models()
    return {
        "models": [
            {"title": title, "model_name": info.model_name, "filename": info.filename}
            for title, info in sorted(
                sd_models.checkpoints_list.items(), key=lambda item: item[0].lower()
            )
        ],
        "loras": list_loras(),
        "vaes": ["None", *sorted(sd_vae.vae_dict.keys(), key=str.lower)],
        "precisions": ["fp32", "fp16", "bf16", "float8_e4m3fn", "float8_e5m2"],
        "lora_precisions": ["fp32", "fp16", "bf16"],
        "component_precisions": [
            "inherit",
            "fp32",
            "fp16",
            "bf16",
            "float8_e4m3fn",
            "float8_e5m2",
        ],
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
    if path_mode == 0:
        if model_info := resolve_model_info(model_name):
            return do_convert(model_info, *args)
        return "Error: model not found"
    if path_mode == 1:
        if os.path.exists(model_path):
            return do_convert(MockModelInfo(model_path), *args)
        return f'Error: model path "{model_path}" does not exist'
    if path_mode == 2:
        if not os.path.isdir(directory):
            return f'Error: path "{directory}" does not exist or is not a directory'
        files = [
            f for f in os.listdir(directory) if f.endswith((".ckpt", ".safetensors"))
        ]
        if not files:
            return "Error: no checkpoints found in directory"
        _args = list(args)
        _args[3] = ""
        for filename in files:
            do_convert(MockModelInfo(os.path.join(directory, filename)), *_args)
        return "Batch processing done"
    return f"Error: unknown mode {path_mode}"


def conversion_metadata(
    model_info: MockModelInfo,
    *,
    precision: str,
    component_precisions: dict[str, str],
    extra_opt: dict[str, str],
    conv_type: str,
    bake_in_vae: str,
    fix_clip: bool,
    force_position_id: bool,
    delete_known_junk_data: bool,
    doctor: dict[str, Any],
) -> dict[str, str]:
    data = {
        "format": "pt",
        "openclaw_converter_version": OPENCLAW_CONVERTER_VERSION,
        "openclaw_converter_source": model_info.filename,
        "openclaw_converter_family": str(doctor.get("family") or "unknown"),
        "openclaw_converter_precision": precision,
        "openclaw_converter_component_precisions": json.dumps(
            component_precisions, sort_keys=True
        ),
        "openclaw_converter_component_actions": json.dumps(extra_opt, sort_keys=True),
        "openclaw_converter_pruning": conv_type,
        "openclaw_converter_baked_vae": bake_in_vae or "None",
        "openclaw_converter_fix_clip_position_ids": str(bool(fix_clip)),
        "openclaw_converter_force_position_ids_int64": str(bool(force_position_id)),
        "openclaw_converter_cleanup_known_junk": str(bool(delete_known_junk_data)),
        "openclaw_converter_created_at": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        ),
        "openclaw_converter_doctor_summary": json.dumps(
            {
                "family": doctor.get("family"),
                "tensor_count": doctor.get("tensor_count"),
                "dtype_counts": doctor.get("dtype_counts"),
                "component_counts": doctor.get("component_counts"),
                "known_junk_count": doctor.get("known_junk_count"),
                "nonfinite": doctor.get("nonfinite"),
                "warnings": doctor.get("warnings"),
            },
            sort_keys=True,
        ),
    }
    return {str(k): str(v) for k, v in data.items()}


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
    unet_precision="inherit",
    clip_precision="inherit",
    vae_precision="inherit",
    other_precision="inherit",
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
    component_precisions = {
        "unet": normalize_precision(unet_precision),
        "clip": normalize_precision(clip_precision),
        "vae": normalize_precision(vae_precision),
        "other": normalize_precision(other_precision),
    }
    float8_components = {
        component
        for component in ("unet", "clip", "vae", "other")
        if (
            component_precisions[component]
            if component_precisions[component] != "inherit"
            else precision
        ).startswith("float8_")
    }
    if float8_components and (
        float8_components - {"unet"} or extra_opt.get("unet") != "convert"
    ):
        return "Error: float8 checkpoint export is experimental and currently limited to explicit UNet-only conversion; use MXFP8/NVFP4 runtime quantization for generation instead."
    shared.state.begin()
    try:
        shared.state.job = "model-convert"
        shared.state.textinfo = f"Loading {model_info.filename}..."
        print(f"[OpenClaw Model Converter] Loading {model_info.filepath}...")
        ok: dict[str, Tensor] = {}
        state_dict = load_model(model_info.filepath)
        source_nonfinite = scan_and_repair_nonfinite(state_dict, repair=True)
        before_doctor = checkpoint_doctor(state_dict, model_info)
        before_doctor["nonfinite"] = source_nonfinite
        fix_model(state_dict, fix_clip=fix_clip, force_position_id=force_position_id)

        def precision_for(weight_key: str) -> str:
            selected = component_precisions.get(
                check_weight_type(weight_key), "inherit"
            )
            return precision if selected == "inherit" else selected

        def convert_tensor(weight_key: str, tensor: Tensor) -> Tensor:
            if str(weight_key).endswith("position_ids"):
                return tensor.to(torch.int64) if force_position_id else tensor
            if not torch.is_floating_point(tensor):
                return tensor
            return PRECISION_FUNCS[precision_for(weight_key)](tensor)

        def handle_weight(weight_key: str, tensor: Tensor) -> None:
            if not isinstance(tensor, Tensor):
                return
            action = extra_opt[check_weight_type(weight_key)]
            if action == "convert":
                ok[weight_key] = convert_tensor(weight_key, tensor)
            elif action == "copy":
                ok[weight_key] = (
                    tensor.to(torch.int64)
                    if force_position_id and str(weight_key).endswith("position_ids")
                    else tensor
                )

        print("[OpenClaw Model Converter] Converting model...")
        if conv_type == "ema-only":
            for key in state_dict:
                ema_key = (
                    "model_ema." + key[6:].replace(".", "") if len(key) >= 6 else "___"
                )
                if ema_key in state_dict:
                    handle_weight(key, state_dict[ema_key])
                elif not key.startswith("model_ema.") or key in [
                    "model_ema.num_updates",
                    "model_ema.decay",
                ]:
                    handle_weight(key, state_dict[key])
        elif conv_type == "no-ema":
            for key, value in state_dict.items():
                if "model_ema." not in key:
                    handle_weight(key, value)
        else:
            for key, value in state_dict.items():
                handle_weight(key, value)
        removed_junk: list[str] = []
        if delete_known_junk_data:
            for key in [k for k in ok if is_known_junk_key(str(k))]:
                removed_junk.append(str(key))
                del ok[key]
        bake_in_vae_filename = sd_vae.vae_dict.get(bake_in_vae)
        if bake_in_vae_filename is not None:
            print(
                f"[OpenClaw Model Converter] Baking in VAE from {bake_in_vae_filename}"
            )
            vae_dict = sd_vae.load_vae_dict(bake_in_vae_filename, map_location="cpu")
            for key, value in vae_dict.items():
                handle_weight(key, value)
            del vae_dict
        output_nonfinite = scan_and_repair_nonfinite(ok, repair=True)
        after_doctor = checkpoint_doctor(ok, model_info)
        after_doctor["nonfinite"] = output_nonfinite
        metadata = conversion_metadata(
            model_info,
            precision=precision,
            component_precisions=component_precisions,
            extra_opt=extra_opt,
            conv_type=conv_type,
            bake_in_vae=bake_in_vae,
            fix_clip=fix_clip,
            force_position_id=force_position_id,
            delete_known_junk_data=delete_known_junk_data,
            doctor=after_doctor,
        )
        ckpt_dir = os.path.dirname(model_info.filepath)
        save_name = (
            custom_name.strip()
            if custom_name
            else f"{model_info.model_name}-{precision}"
        )
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
                safetensors.torch.save_file(ok, save_path, metadata=metadata)
            else:
                torch.save({"state_dict": ok, "openclaw_metadata": metadata}, save_path)
            output += f"Checkpoint saved to {save_path}\n"
        refresh_after_convert("checkpoint")
        report = {
            "source_doctor": before_doctor,
            "output_doctor": after_doctor,
            "removed_known_junk": {
                "count": len(removed_junk),
                "examples": removed_junk[:50],
            },
            "nonfinite_repair": {
                "source": source_nonfinite,
                "output": output_nonfinite,
            },
            "metadata": metadata,
        }
        output += "OpenClaw checkpoint doctor report:\n" + json.dumps(
            report, indent=2, sort_keys=True
        )
        return output.rstrip()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        shared.state.end()


def convert_single(payload: dict[str, Any]) -> str:
    if str(payload.get("mode") or "checkpoint").lower() == "lora":
        return convert_lora(payload)
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
        payload.get("unet_precision") or "inherit",
        payload.get("clip_precision") or "inherit",
        payload.get("vae_precision") or "inherit",
        payload.get("other_precision") or "inherit",
    )
