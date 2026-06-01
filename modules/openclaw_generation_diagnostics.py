from __future__ import annotations

import copy
import hashlib
import time
from typing import Any

_LAST_GENERATION_DIAGNOSTICS: dict[str, Any] | None = None

_CUDA_GRAPH_COUNTERS = ("captures", "replays", "fallbacks", "bypasses", "failures")
_OPENCLAW_PARAM_PREFIXES = (
    "PAG ",
    "SEG ",
    "CFG Interval ",
    "Multi-Sampler ",
    "Dynamic thresholding",
    "Mimic ",
    "CFG mode",
    "CFG scale minimum",
    "Scheduler value",
    "Threshold percentile",
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _cuda_graph_status() -> dict[str, Any] | None:
    try:
        from modules import openclaw_cuda_graphs

        return openclaw_cuda_graphs.status()
    except Exception as exc:
        return {"available": False, "error": repr(exc)}


def _summarize_graph_key(raw_key: Any) -> dict[str, Any] | None:
    if raw_key in (None, ""):
        return None

    text = str(raw_key)
    return {
        "sha256_16": hashlib.sha256(text.encode("utf8", errors="replace")).hexdigest()[:16],
        "length": len(text),
        "has_seg": "'seg'" in text or '"seg"' in text,
        "has_lora": "lora" in text.lower(),
    }


def _summarize_cuda_graph_status(status: dict[str, Any] | None) -> dict[str, Any] | None:
    if status is None:
        return None

    summary = {
        "enabled": bool(status.get("enabled", False)),
        "allow_seg": bool(status.get("allow_seg", False)),
        "cache_size": int(status.get("cache_size") or 0),
        "max_cache_size": int(status.get("max_cache_size") or 0),
        "last_bypass_reason": status.get("last_bypass_reason"),
        "last_error": status.get("last_error"),
        "bypass_reasons": dict(status.get("bypass_reasons") or {}),
        "last_key_summary": _summarize_graph_key(status.get("last_key")),
    }
    for key in _CUDA_GRAPH_COUNTERS:
        summary[key] = int(status.get(key) or 0)
    return summary


def _counter_delta(before: dict[str, Any] | None, after: dict[str, Any] | None) -> dict[str, Any]:
    if before is None or after is None:
        return {}

    delta = {key: int(after.get(key) or 0) - int(before.get(key) or 0) for key in _CUDA_GRAPH_COUNTERS}
    before_reasons = before.get("bypass_reasons") or {}
    after_reasons = after.get("bypass_reasons") or {}
    reason_delta = {
        key: int(after_reasons.get(key) or 0) - int(before_reasons.get(key) or 0)
        for key in sorted(set(before_reasons) | set(after_reasons))
        if int(after_reasons.get(key) or 0) - int(before_reasons.get(key) or 0)
    }
    delta["bypass_reasons"] = reason_delta
    return delta


def _interesting_extra_params(p: Any) -> dict[str, Any]:
    params = getattr(p, "extra_generation_params", None) or {}
    return {
        str(key): _json_safe(value)
        for key, value in params.items()
        if any(str(key).startswith(prefix) for prefix in _OPENCLAW_PARAM_PREFIXES)
    }


def _request_summary(p: Any, batch_index: int) -> dict[str, Any]:
    seeds = getattr(p, "seeds", None) or getattr(p, "all_seeds", None) or []
    return {
        "job_timestamp": getattr(p, "job_timestamp", None),
        "batch_index": batch_index,
        "sampler_name": getattr(p, "sampler_name", None),
        "scheduler": getattr(p, "scheduler", None),
        "steps": getattr(p, "steps", None),
        "width": getattr(p, "width", None),
        "height": getattr(p, "height", None),
        "batch_size": getattr(p, "batch_size", None),
        "seed": seeds[0] if seeds else getattr(p, "seed", None),
        "sd_model_hash": getattr(p, "sd_model_hash", None),
        "sd_model_name": getattr(p, "sd_model_name", None),
        "denoising_strength": getattr(p, "denoising_strength", None),
        "is_img2img": bool(getattr(p, "init_images", None)),
        "openclaw_params": _interesting_extra_params(p),
    }


def before_sample(p: Any, batch_index: int) -> dict[str, Any]:
    return {
        "started_at": time.time(),
        "request": _request_summary(p, batch_index),
        "cuda_graphs_before": _summarize_cuda_graph_status(_cuda_graph_status()),
    }


def after_sample(p: Any, capture: dict[str, Any] | None, batch_index: int) -> dict[str, Any] | None:
    global _LAST_GENERATION_DIAGNOSTICS

    if capture is None:
        return None

    finished_at = time.time()
    after = _summarize_cuda_graph_status(_cuda_graph_status())
    diagnostics = {
        **capture,
        "finished_at": finished_at,
        "duration_seconds": round(finished_at - float(capture.get("started_at") or finished_at), 6),
        "request": _request_summary(p, batch_index),
        "cuda_graphs_after": after,
        "cuda_graphs_delta": _counter_delta(capture.get("cuda_graphs_before"), after),
    }
    diagnostics = _json_safe(diagnostics)
    p.openclaw_generation_diagnostics = diagnostics
    history = getattr(p, "openclaw_generation_diagnostics_history", None)
    if history is None:
        history = p.openclaw_generation_diagnostics_history = []
    history.append(diagnostics)
    _LAST_GENERATION_DIAGNOSTICS = copy.deepcopy(diagnostics)
    return diagnostics


def last_generation_diagnostics() -> dict[str, Any] | None:
    return copy.deepcopy(_LAST_GENERATION_DIAGNOSTICS)
