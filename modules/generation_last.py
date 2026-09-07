"""Persist one bounded, replay-oriented snapshot of the last completed generation."""

from __future__ import annotations

import datetime as dt
import json
import math
import os
import re
import threading
from enum import Enum
from pathlib import Path
from typing import Any

from modules import paths, shared


SCHEMA_VERSION = 3
_LOCK = threading.RLock()
_OMIT = object()
_MAX_DEPTH = 8
_MAX_ITEMS = 256
_MAX_STRING_LENGTH = 16_384
_MAX_SNAPSHOT_BYTES = 40 * 1024 * 1024
# Base64 expands already-compressed PNG data; ordinary 1024-square RGBA inputs
# can exceed 5 MiB. Keep lossless assets and bound aggregate retention separately.
_MAX_IMAGE_BYTES = 8 * 1024 * 1024
_MAX_IMAGE_TOTAL_BYTES = 32 * 1024 * 1024
_SENSITIVE_KEY_PARTS = ("password", "secret", "token", "credential", "authorization", "api_key", "cookie")
_CREDENTIAL_FILTER_EXEMPTIONS = {"token_merging_ratio", "token_merging_ratio_hr", "token_merging_ratio_img2img"}


def snapshot_path() -> Path:
    directory = os.environ.get("GENERATION_LAST_DIR")
    return Path(directory) / "generation-last.json" if directory else Path(paths.data_path) / "generation-last" / "generation-last.json"


def _limitation(limitations: list[str], message: str) -> None:
    if message not in limitations:
        limitations.append(message)


def _safe_json(value: Any, limitations: list[str], path: str, depth: int = 0):
    if depth > _MAX_DEPTH:
        _limitation(limitations, f"{path} exceeds the supported nesting depth and was not persisted.")
        return _OMIT
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        _limitation(limitations, f"{path} is non-finite and was not persisted.")
        return _OMIT
    if isinstance(value, str):
        if value.startswith("data:image/") or len(value) > _MAX_STRING_LENGTH:
            _limitation(limitations, f"{path} is an image or exceeds the response bound and was not persisted.")
            return _OMIT
        return value
    if isinstance(value, (list, tuple)):
        if len(value) > _MAX_ITEMS:
            _limitation(limitations, f"{path} has too many items and was not persisted.")
            return _OMIT
        result = []
        for index, item in enumerate(value):
            safe = _safe_json(item, limitations, f"{path}[{index}]", depth + 1)
            if safe is _OMIT:
                return _OMIT
            result.append(safe)
        return result
    if isinstance(value, dict):
        if len(value) > _MAX_ITEMS:
            _limitation(limitations, f"{path} has too many fields and was not persisted.")
            return _OMIT
        result = {}
        for key, item in value.items():
            key = str(key)
            if key.lower() not in _CREDENTIAL_FILTER_EXEMPTIONS and any(part in key.lower() for part in _SENSITIVE_KEY_PARTS):
                _limitation(limitations, f"{path}.{key} is credential-like and was not persisted.")
                continue
            safe = _safe_json(item, limitations, f"{path}.{key}", depth + 1)
            if safe is _OMIT:
                continue
            result[key] = safe
        return result

    _limitation(limitations, f"{path} has unsupported type {type(value).__name__} and was not persisted.")
    return _OMIT


def _value(p, name: str, default=None):
    value = getattr(p, name, default)
    return value() if callable(value) else value


def _copy_parameter(parameters: dict[str, Any], p, name: str, limitations: list[str]) -> None:
    value = _enum_values(_value(p, name))
    # A1111 uses infinity internally for the API value 0.0 (unlimited s_tmax).
    # Preserve a replayable request rather than attempting to serialize JSON's
    # unsupported Infinity token.
    if name == "s_tmax" and isinstance(value, float) and not math.isfinite(value):
        value = 0.0
    value = _safe_json(value, limitations, f"parameters.{name}")
    if value is not _OMIT:
        parameters[name] = value


def _first_prompt_value(p, name: str, all_name: str):
    value = _value(p, name)
    if isinstance(value, list):
        value = (_value(p, all_name, []) or [""])[0]
    return value


_HARNESS_OWNED_MAPPING_FIELDS = {
    "prompt", "negative_prompt", "hr_prompt", "hr_negative_prompt", "width", "height",
    "firstphase_width", "firstphase_height", "hr_scale", "hr_resize_x", "hr_resize_y",
}


def _omit_harness_owned_mappings(value: Any, limitations: list[str], path: str):
    """Remove named nested prompt/size settings without shifting positional args."""
    if isinstance(value, list):
        return [_omit_harness_owned_mappings(item, limitations, f"{path}[{index}]") for index, item in enumerate(value)]
    if not isinstance(value, dict):
        return value
    result = {}
    for key, item in value.items():
        if key.casefold() in _HARNESS_OWNED_MAPPING_FIELDS:
            _limitation(limitations, f"{path}.{key} is omitted; Harness must supply its replacement through the extension's documented argument.")
            continue
        result[key] = _omit_harness_owned_mappings(item, limitations, f"{path}.{key}")
    return result


_CONTROLNET_API_FIELDS = (
    "enabled", "input_mode", "module", "model", "weight", "resize_mode", "low_vram",
    "processor_res", "threshold_a", "threshold_b", "guidance_start", "guidance_end",
    "pixel_perfect", "control_mode", "inpaint_crop_input_image", "hr_option",
    "save_detected_map", "advanced_weighting", "pulid_mode", "union_control_type",
)


def _enum_values(value: Any):
    if isinstance(value, Enum):
        return _enum_values(value.value)
    if isinstance(value, (list, tuple)):
        return [_enum_values(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _enum_values(item) for key, item in value.items()}
    return value


def _is_controlnet_unit(value: Any) -> bool:
    return type(value).__name__ == "ControlNetUnit"


def _image_to_api_base64(value: Any, limitations: list[str], path: str, budget: dict[str, int]):
    """Encode a PIL/numpy API input as bounded PNG base64 without retaining paths."""
    if value is None:
        return None
    if isinstance(value, dict):
        value = value.get("image")
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            _limitation(limitations, f"{path} has multiple image inputs beyond the retained-image limit.")
            return _OMIT
        value = value[0]
    try:
        from PIL import Image
        import base64
        import io

        if isinstance(value, str):
            # API/ControlNet inputs can already be encoded. Decode inline data
            # only: never resolve paths or fetch URLs while capturing a snapshot.
            if value.startswith("data:"):
                prefix, separator, value = value.partition(",")
                if not separator or prefix not in ("data:image/png;base64", "data:image/jpeg;base64", "data:image/webp;base64"):
                    raise ValueError("Unsupported inline image encoding")
            if len(value) > _MAX_IMAGE_BYTES:
                raise ValueError("Encoded input exceeds image budget")
            raw = base64.b64decode(value, validate=True)
            with Image.open(io.BytesIO(raw)) as decoded:
                if decoded.width > 16384 or decoded.height > 16384 or decoded.width * decoded.height > 64 * 1024 * 1024:
                    raise ValueError("Image dimensions exceed budget")
                value = decoded.convert("RGBA")
        if not isinstance(value, Image.Image):
            value = Image.fromarray(value)
        image = value.convert("RGBA")
        with io.BytesIO() as output:
            image.save(output, format="PNG")
            encoded = base64.b64encode(output.getvalue()).decode("ascii")
    except Exception:
        _limitation(limitations, f"{path} could not be encoded as API PNG base64 data.")
        return _OMIT
    encoded_bytes = len(encoded)
    if encoded_bytes > _MAX_IMAGE_BYTES:
        _limitation(limitations, f"{path} exceeds the per-image retention limit of {_MAX_IMAGE_BYTES} bytes.")
        return _OMIT
    if budget["images"] + encoded_bytes > _MAX_IMAGE_TOTAL_BYTES:
        _limitation(limitations, f"{path} exceeds the total retained-image limit of {_MAX_IMAGE_TOTAL_BYTES} bytes.")
        return _OMIT
    budget["images"] += encoded_bytes
    return encoded


def _controlnet_unit_to_api_json(unit: Any, unit_index: int, limitations: list[str], budget: dict[str, int], *, retain_assets: bool = True) -> dict[str, Any]:
    """Serialize the API-supported portion of an effective ControlNet unit."""
    # UI runs retain ControlNetUnit objects; API script_args retain dictionaries.
    get = unit.get if isinstance(unit, dict) else lambda name, default=None: getattr(unit, name, default)
    result: dict[str, Any] = {}
    for field in _CONTROLNET_API_FIELDS:
        value = _safe_json(_enum_values(get(field)), limitations, f"alwayson_scripts.ControlNet.args[{unit_index}].{field}")
        if value is not _OMIT:
            result[field] = value

    enabled = bool(result.get("enabled", False))
    if not enabled:
        return result
    if not retain_assets:
        if result.get("input_mode") not in (None, "simple"):
            _limitation(limitations, f"ControlNet unit {unit_index + 1} requires an unsupported batch/merge input mode.")
        if get("ipadapter_input") is not None:
            _limitation(limitations, f"ControlNet unit {unit_index + 1} requires serialized IP-Adapter input, not a single replacement image.")
        return result

    unit_path = f"alwayson_scripts.ControlNet.args[{unit_index}]"
    raw_image = get("image")
    raw_mask = get("mask")
    if isinstance(raw_image, dict):
        raw_mask = raw_mask if raw_mask is not None else raw_image.get("mask")
    elif isinstance(raw_image, (list, tuple)) and len(raw_image) == 2:
        raw_image, raw_mask = raw_image
    image = _image_to_api_base64(raw_image, limitations, f"{unit_path}.image", budget)
    if image is _OMIT or image is None:
        _limitation(limitations, f"ControlNet unit {unit_index + 1} requires {unit_path}.image before replay.")
    else:
        result["image"] = image
    mask = _image_to_api_base64(raw_mask, limitations, f"{unit_path}.mask", budget)
    if mask is _OMIT:
        _limitation(limitations, f"ControlNet unit {unit_index + 1} requires {unit_path}.mask before replay.")
    elif mask is not None:
        result["mask"] = mask
    if get("effective_region_mask") is not None:
        _limitation(limitations, f"ControlNet unit {unit_index + 1} effective region mask is not persisted; supply {unit_path}.effective_region_mask before replay.")
    if get("ipadapter_input") is not None:
        _limitation(limitations, f"ControlNet unit {unit_index + 1} IP-Adapter input is not persisted; supply {unit_path}.ipadapter_input before replay.")
    if getattr(unit, "batch_images", None) or getattr(unit, "batch_image_files", None):
        _limitation(limitations, f"ControlNet unit {unit_index + 1} batch inputs are not persisted; supply its image inputs before replay.")
    return result


def _capture_script_parameters(p, parameters: dict[str, Any], limitations: list[str], budget: dict[str, int], *, retain_assets: bool = True) -> None:
    runner = getattr(p, "scripts", None)
    script_args = getattr(p, "script_args", None)
    if runner is None or not isinstance(script_args, (list, tuple)):
        if runner is not None:
            _limitation(limitations, "Extension parameters are unavailable because script arguments were not a list.")
        return

    alwayson = {}
    for script in getattr(runner, "alwayson_scripts", []) or []:
        title = script.title()
        start, end = getattr(p, "openclaw_script_arg_ranges", {}).get(id(script), (script.args_from, script.args_to))
        raw_values = list(script_args[start:end])
        if title.casefold() == "controlnet":
            values = [
                _controlnet_unit_to_api_json(value, index, limitations, budget, retain_assets=retain_assets) if _is_controlnet_unit(value) or isinstance(value, dict)
                else _safe_json(value, limitations, f"alwayson_scripts.{title}.args[{index}]")
                for index, value in enumerate(raw_values)
            ]
            if any(value is _OMIT for value in values):
                continue
        else:
            values = _safe_json(raw_values, limitations, f"alwayson_scripts.{title}.args")
        if values is _OMIT:
            continue
        alwayson[title] = {"args": _omit_harness_owned_mappings(values, limitations, f"alwayson_scripts.{title}.args")}
    if alwayson:
        parameters["alwayson_scripts"] = alwayson

    selected = script_args[0] if script_args else 0
    if isinstance(selected, int) and selected > 0:
        selectable = getattr(runner, "selectable_scripts", []) or []
        if selected > len(selectable):
            _limitation(limitations, "The selected script index is no longer available for replay.")
            return
        script = selectable[selected - 1]
        start, end = getattr(p, "openclaw_script_arg_ranges", {}).get(id(script), (script.args_from, script.args_to))
        values = _safe_json(list(script_args[start:end]), limitations, f"script.{script.title()}.args")
        if values is _OMIT:
            return
        parameters["script_name"] = script.title()
        parameters["script_args"] = _omit_harness_owned_mappings(values, limitations, f"script.{script.title()}.args")


def _checkpoint_identity(p) -> dict[str, Any]:
    model = getattr(p, "sd_model", None) or getattr(shared, "sd_model", None)
    info = getattr(model, "sd_checkpoint_info", None)
    filename = getattr(info, "filename", None)
    return {
        "title": getattr(info, "title", None),
        "name": getattr(info, "name_for_extra", None) or getattr(info, "name", None) or getattr(p, "sd_model_name", None),
        "hash": getattr(model, "sd_model_hash", None) or getattr(p, "sd_model_hash", None),
        "sha256": getattr(info, "sha256", None),
        "filename": os.path.basename(filename) if filename else None,
        "vae_name": getattr(p, "sd_vae_name", None),
        "vae_hash": getattr(p, "sd_vae_hash", None),
    }


def _controlnet_limitations(parameters: dict[str, Any], limitations: list[str]) -> None:
    for suffix in ("", "2", "3"):
        enabled = parameters.get(f"control_net_enabled{suffix}")
        if enabled and f"control_net_image{suffix}" not in parameters:
            _limitation(limitations, f"ControlNet unit {suffix or '1'} requires parameters.control_net_image{suffix} before replay.")


def _capture_img2img_assets(p: Any, parameters: dict[str, Any], limitations: list[str], budget: dict[str, int]) -> None:
    init_images = getattr(p, "init_images", None)
    if not isinstance(init_images, (list, tuple)) or not init_images:
        _limitation(limitations, "img2img replay requires init_images, but none were available from the completed generation.")
    else:
        encoded_images = []
        for index, image in enumerate(init_images):
            encoded = _image_to_api_base64(image, limitations, f"parameters.init_images[{index}]", budget)
            if encoded is _OMIT or encoded is None:
                _limitation(limitations, f"img2img replay requires parameters.init_images[{index}].")
                continue
            encoded_images.append(encoded)
        if len(encoded_images) == len(init_images):
            parameters["init_images"] = encoded_images
    # Img2img moves the API mask into image_mask during __post_init__ before
    # processing starts; prefer it so we retain the same source mask the run used.
    source_mask = getattr(p, "image_mask", None)
    if source_mask is None:
        source_mask = getattr(p, "mask", None)
    mask = _image_to_api_base64(source_mask, limitations, "parameters.mask", budget)
    if mask is _OMIT:
        _limitation(limitations, "img2img replay requires parameters.mask, but it exceeded the retention limit or could not be encoded.")
    elif mask is not None:
        parameters["mask"] = mask


def _build_parameters(p, processed, *, retain_assets: bool):
    """Capture tuning independently of optional previous-task assets."""
    limitations: list[str] = []
    generation_type = "img2img" if p.__class__.__name__.endswith("Img2Img") else "txt2img"
    parameters: dict[str, Any] = {}
    budget = {"images": 0}

    common_fields = (
        "styles", "subseed_strength", "seed_resize_from_h", "seed_resize_from_w",
        "sampler_name", "scheduler", "batch_size", "n_iter", "steps", "cfg_scale",
        "restore_faces", "tiling", "eta", "denoising_strength", "ddim_discretize", "s_min_uncond",
        "s_churn", "s_tmax", "s_tmin", "s_noise", "refiner_checkpoint", "refiner_switch_at",
        "disable_extra_networks", "resize_mode", "image_cfg_scale", "initial_noise_multiplier",
        "mask_blur", "mask_blur_x", "mask_blur_y", "inpainting_mask_invert",
        "inpaint_full_res", "inpaint_full_res_padding",
    )
    for field in common_fields:
        _copy_parameter(parameters, p, field, limitations)

    # A completed image has a resolved seed even when the request used -1.
    parameters["seed"] = int((getattr(processed, "all_seeds", None) or getattr(p, "all_seeds", None) or [-1])[0])
    parameters["subseed"] = int((getattr(processed, "all_subseeds", None) or getattr(p, "all_subseeds", None) or [-1])[0])
    parameters["send_images"] = True
    parameters["save_images"] = False

    if generation_type == "txt2img":
        for field in (
            "enable_hr", "hr_upscaler", "hr_second_pass_steps",
            "hr_checkpoint_name", "hr_sampler_name", "hr_scheduler",
        ):
            _copy_parameter(parameters, p, field, limitations)
    elif retain_assets:
        _capture_img2img_assets(p, parameters, limitations, budget)

    for suffix in ("", "2", "3"):
        for name in ("enabled", "module", "model", "weight", "resize_mode", "lowvram", "pres", "pthr_a", "pthr_b", "guidance_start", "guidance_end", "control_mode", "pixel_perfect"):
            _copy_parameter(parameters, p, f"control_net_{name}{suffix}", limitations)
        image_name = f"control_net_image{suffix}"
        if retain_assets and parameters.get(f"control_net_enabled{suffix}"):
            image = _image_to_api_base64(getattr(p, image_name, None), limitations, f"parameters.{image_name}", budget)
            if image is _OMIT or image is None:
                _limitation(limitations, f"ControlNet unit {suffix or '1'} requires parameters.{image_name} before replay.")
            else:
                parameters[image_name] = image
    if retain_assets:
        _controlnet_limitations(parameters, limitations)

    override_settings = _safe_json(dict(getattr(p, "override_settings", {}) or {}), limitations, "parameters.override_settings")
    if override_settings is _OMIT:
        override_settings = {}
    checkpoint = _checkpoint_identity(p)
    if checkpoint["title"]:
        override_settings["sd_model_checkpoint"] = checkpoint["title"]
    if checkpoint["vae_name"]:
        override_settings["sd_vae"] = checkpoint["vae_name"]
    # These are A1111 options rather than txt2img request fields, so replay them
    # through the documented override_settings channel.
    for field in ("token_merging_ratio", "token_merging_ratio_hr"):
        value = _safe_json(_value(p, field, 0), limitations, f"parameters.override_settings.{field}")
        if value is not _OMIT:
            override_settings[field] = value
    clip_skip = getattr(p, "clip_skip", getattr(shared.opts, "CLIP_stop_at_last_layers", None))
    if clip_skip is not None:
        value = _safe_json(clip_skip, limitations, "parameters.override_settings.CLIP_stop_at_last_layers")
        if value is not _OMIT:
            override_settings["CLIP_stop_at_last_layers"] = value
    parameters["override_settings"] = override_settings
    parameters["override_settings_restore_afterwards"] = bool(getattr(p, "override_settings_restore_afterwards", True))

    _capture_script_parameters(p, parameters, limitations, budget, retain_assets=retain_assets)
    return parameters, limitations, checkpoint, generation_type


_LORA_TAG = re.compile(r"<lora:([^<>:\x00-\x1f\x7f]{1,256}):([+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?)>")
_MAX_LORA_TAGS = 32
_MAX_PROMPT_CAPTURE_LENGTH = 262_144


def _capture_lora_tags(p, processed, limitations: list[str]) -> list[str]:
    """Retain network selections, never the surrounding descriptive prompt."""
    negative_prompts = getattr(processed, "all_negative_prompts", None) or getattr(p, "all_negative_prompts", None)
    if not negative_prompts:
        negative = getattr(p, "negative_prompt", "")
        negative_prompts = negative if isinstance(negative, list) else [negative]
    if not isinstance(negative_prompts, (list, tuple)) or len(negative_prompts) > _MAX_ITEMS:
        _limitation(limitations, "Effective negative prompt batch exceeds the LoRA capture bound.")
        return []
    for negative in negative_prompts:
        if not isinstance(negative, str) or len(negative) > _MAX_PROMPT_CAPTURE_LENGTH:
            _limitation(limitations, "Effective negative prompt exceeds the LoRA capture bound.")
            return []
        if re.search(r"<lora:", negative, flags=re.IGNORECASE):
            _limitation(limitations, "Negative-prompt LoRA selections cannot be transferred to a replacement positive prompt.")
            return []
    prompts = getattr(processed, "all_prompts", None) or getattr(p, "all_prompts", None)
    if not prompts:
        # setup_prompts normally supplies the style-expanded all_prompts. Without
        # it, configured styles cannot be reconstructed reliably after the run.
        if getattr(p, "styles", None):
            _limitation(limitations, "Effective style-expanded prompts are unavailable for LoRA selection capture.")
            return []
        prompt = getattr(p, "prompt", "")
        prompts = prompt if isinstance(prompt, list) else [prompt]
    if not isinstance(prompts, (list, tuple)) or len(prompts) > _MAX_ITEMS:
        _limitation(limitations, "Effective prompt batch exceeds the LoRA capture bound.")
        return []
    selections = None
    for prompt in prompts:
        if not isinstance(prompt, str) or len(prompt) > _MAX_PROMPT_CAPTURE_LENGTH:
            _limitation(limitations, "Effective prompt exceeds the LoRA capture bound.")
            return []
        tags = []
        for match in re.finditer(r"<lora:[^>]*(?:>|$)", prompt, flags=re.IGNORECASE):
            tag = match.group()
            parsed = _LORA_TAG.fullmatch(tag)
            if len(tag) > 256 or parsed is None or not math.isfinite(float(parsed[2])):
                _limitation(limitations, "A LoRA selection is not a supported finite numeric weighted tag.")
                return []
            if len(tags) >= _MAX_LORA_TAGS:
                _limitation(limitations, "LoRA selections exceed the retained-tag bound.")
                return []
            tags.append(tag)
        if selections is not None and tags != selections:
            _limitation(limitations, "Effective batch prompts have different LoRA selections; one reusable selection cannot represent the batch.")
            return []
        selections = tags
    return selections or []


def build_snapshot(p, processed, *, completed_at: str | None = None) -> dict[str, Any]:
    """Keep prior-task replay and independently reusable tuning distinct."""
    parameters, limitations, checkpoint, generation_type = _build_parameters(p, processed, retain_assets=True)
    settings, settings_limitations, _, _ = _build_parameters(p, processed, retain_assets=False)
    lora_tags = _capture_lora_tags(p, processed, settings_limitations)
    replayable = generation_type in ("txt2img", "img2img") and not limitations
    return {
        "schema_version": SCHEMA_VERSION,
        "completed_at": completed_at or dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "generation_type": generation_type,
        "replayable": replayable,
        "limitations": limitations,
        "checkpoint": checkpoint,
        "parameters": parameters,
        "settings_parameters": settings,
        "settings_replayable": not settings_limitations,
        "settings_limitations": settings_limitations,
        "lora_tags": lora_tags,
    }


def _completed_successfully(p, processed) -> bool:
    state = getattr(shared, "state", None)
    if getattr(state, "interrupted", False) or getattr(state, "stopping_generation", False):
        return False
    images = getattr(processed, "images", None)
    return bool(images)


def persist_snapshot(snapshot: dict[str, Any], path: Path | None = None) -> None:
    path = path or snapshot_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(snapshot, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    if len(payload.encode("utf-8")) > _MAX_SNAPSHOT_BYTES:
        raise ValueError(f"last-generation snapshot exceeds the {_MAX_SNAPSHOT_BYTES}-byte retention limit")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    with open(temporary, "w", encoding="utf-8") as file:
        file.write(payload)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def capture_completed_generation(p, processed) -> dict[str, Any] | None:
    """Persist only a newly completed, successful generation once per processing object."""
    if getattr(p, "_generation_last_captured", False) or not _completed_successfully(p, processed):
        return None
    with _LOCK:
        if getattr(p, "_generation_last_captured", False) or not _completed_successfully(p, processed):
            return None
        snapshot = build_snapshot(p, processed)
        persist_snapshot(snapshot)
        p._generation_last_captured = True
        return snapshot


def get_last_snapshot() -> dict[str, Any] | None:
    path = snapshot_path()
    try:
        with open(path, "r", encoding="utf-8") as file:
            snapshot = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    if not isinstance(snapshot, dict):
        return None
    if snapshot.get("schema_version") == 1:
        parameters = dict(snapshot.get("parameters") or {})
        for key in (
            "prompt", "negative_prompt", "width", "height", "firstphase_width", "firstphase_height",
            "hr_scale", "hr_resize_x", "hr_resize_y", "hr_prompt", "hr_negative_prompt",
        ):
            parameters.pop(key, None)
        limitations = list(snapshot.get("limitations") or [])
        generation_type = snapshot.get("generation_type")
        if generation_type == "img2img":
            _limitation(limitations, "This version-1 img2img snapshot has no retained init_images; run one img2img generation after upgrading.")
        return {
            "schema_version": 2,
            "completed_at": snapshot.get("completed_at"),
            "generation_type": generation_type,
            "replayable": bool(snapshot.get("replayable")) and generation_type == "txt2img" and not limitations,
            "limitations": limitations,
            "checkpoint": snapshot.get("checkpoint") or {},
            "parameters": parameters,
        }
    if snapshot.get("schema_version") not in (2, SCHEMA_VERSION):
        return None
    return snapshot
