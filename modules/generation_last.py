"""Persist one bounded, replay-oriented snapshot of the last completed generation."""

from __future__ import annotations

import datetime as dt
import json
import math
import os
import os
import threading
from pathlib import Path
from typing import Any

from modules import paths, shared


SCHEMA_VERSION = 1
_LOCK = threading.RLock()
_OMIT = object()
_MAX_DEPTH = 8
_MAX_ITEMS = 256
_MAX_STRING_LENGTH = 16_384
_SENSITIVE_KEY_PARTS = ("password", "secret", "token", "credential", "authorization", "api_key", "cookie")


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
            if any(part in key.lower() for part in _SENSITIVE_KEY_PARTS):
                _limitation(limitations, f"{path}.{key} is credential-like and was not persisted.")
                return _OMIT
            safe = _safe_json(item, limitations, f"{path}.{key}", depth + 1)
            if safe is _OMIT:
                return _OMIT
            result[key] = safe
        return result

    _limitation(limitations, f"{path} has unsupported type {type(value).__name__} and was not persisted.")
    return _OMIT


def _value(p, name: str, default=None):
    value = getattr(p, name, default)
    return value() if callable(value) else value


def _copy_parameter(parameters: dict[str, Any], p, name: str, limitations: list[str]) -> None:
    value = _value(p, name)
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


def _capture_script_parameters(p, parameters: dict[str, Any], limitations: list[str]) -> None:
    runner = getattr(p, "scripts", None)
    script_args = getattr(p, "script_args", None)
    if runner is None or not isinstance(script_args, (list, tuple)):
        if runner is not None:
            _limitation(limitations, "Extension parameters are unavailable because script arguments were not a list.")
        return

    alwayson = {}
    for script in getattr(runner, "alwayson_scripts", []) or []:
        title = script.title()
        args_to = getattr(p, "openclaw_script_args_to_overrides", {}).get(id(script), script.args_to)
        values = _safe_json(list(script_args[script.args_from:args_to]), limitations, f"alwayson_scripts.{title}.args")
        if values is _OMIT:
            continue
        alwayson[title] = {"args": values}
    if alwayson:
        parameters["alwayson_scripts"] = alwayson

    selected = script_args[0] if script_args else 0
    if isinstance(selected, int) and selected > 0:
        selectable = getattr(runner, "selectable_scripts", []) or []
        if selected > len(selectable):
            _limitation(limitations, "The selected script index is no longer available for replay.")
            return
        script = selectable[selected - 1]
        values = _safe_json(list(script_args[script.args_from:script.args_to]), limitations, f"script.{script.title()}.args")
        if values is _OMIT:
            return
        parameters["script_name"] = script.title()
        parameters["script_args"] = values


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
        if enabled:
            _limitation(limitations, f"ControlNet unit {suffix or '1'} requires an input image that is not persisted; provide control_net_image{suffix} before replay.")


def build_snapshot(p, processed, *, completed_at: str | None = None) -> dict[str, Any]:
    """Build a JSON-safe snapshot from the effective processing object."""
    limitations: list[str] = []
    generation_type = "img2img" if p.__class__.__name__.endswith("Img2Img") else "txt2img"
    parameters: dict[str, Any] = {}

    for name, all_name in (("prompt", "all_prompts"), ("negative_prompt", "all_negative_prompts")):
        value = _safe_json(_first_prompt_value(p, name, all_name), limitations, f"parameters.{name}")
        if value is not _OMIT:
            parameters[name] = value

    common_fields = (
        "styles", "subseed_strength", "seed_resize_from_h", "seed_resize_from_w",
        "sampler_name", "scheduler", "batch_size", "n_iter", "steps", "cfg_scale", "width", "height",
        "restore_faces", "tiling", "eta", "denoising_strength", "ddim_discretize", "s_min_uncond",
        "s_churn", "s_tmax", "s_tmin", "s_noise", "refiner_checkpoint", "refiner_switch_at",
        "disable_extra_networks",
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
            "enable_hr", "firstphase_width", "firstphase_height", "hr_scale", "hr_upscaler", "hr_second_pass_steps", "hr_resize_x", "hr_resize_y",
            "hr_checkpoint_name", "hr_sampler_name", "hr_scheduler", "hr_prompt", "hr_negative_prompt",
        ):
            _copy_parameter(parameters, p, field, limitations)
    else:
        _limitation(limitations, "img2img replay requires init_images; source images are intentionally not persisted in this bounded snapshot.")
        if getattr(p, "mask", None) is not None:
            _limitation(limitations, "img2img replay also requires its mask; masks are not persisted in this snapshot.")

    for suffix in ("", "2", "3"):
        for name in ("enabled", "module", "model", "weight", "resize_mode", "lowvram", "pres", "pthr_a", "pthr_b", "guidance_start", "guidance_end", "control_mode", "pixel_perfect"):
            _copy_parameter(parameters, p, f"control_net_{name}{suffix}", limitations)
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
    override_settings["token_merging_ratio"] = _value(p, "token_merging_ratio", 0)
    override_settings["token_merging_ratio_hr"] = _value(p, "token_merging_ratio_hr", 0)
    clip_skip = getattr(p, "clip_skip", getattr(shared.opts, "CLIP_stop_at_last_layers", None))
    if clip_skip is not None:
        override_settings["CLIP_stop_at_last_layers"] = clip_skip
    parameters["override_settings"] = override_settings
    parameters["override_settings_restore_afterwards"] = bool(getattr(p, "override_settings_restore_afterwards", True))

    _capture_script_parameters(p, parameters, limitations)
    replayable = generation_type == "txt2img" and not limitations
    return {
        "schema_version": SCHEMA_VERSION,
        "completed_at": completed_at or dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "generation_type": generation_type,
        "replayable": replayable,
        "limitations": limitations,
        "checkpoint": checkpoint,
        "parameters": parameters,
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
    if not isinstance(snapshot, dict) or snapshot.get("schema_version") != SCHEMA_VERSION:
        return None
    return snapshot
