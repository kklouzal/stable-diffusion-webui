
from __future__ import annotations

import json
import math
import os
import platform
import threading
import time
from pathlib import Path
from typing import Any

import torch

_LOCK = threading.Lock()
_LAST_RESULT: dict[str, Any] | None = None
_LAST_RUNNING = False


def _data_dir() -> Path:
    try:
        from modules import paths
        base = Path(paths.data_path)
    except Exception:
        base = Path.cwd()
    path = base / "mxfp8-diagnostics"
    path.mkdir(parents=True, exist_ok=True)
    return path


def last_result_path() -> Path:
    return _data_dir() / "last-result.json"


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return str(value)
        return value
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return str(value)


def _timed_cuda(fn, warmup: int = 5, iters: int = 20) -> dict[str, Any]:
    if not torch.cuda.is_available():
        start = time.perf_counter()
        for _ in range(max(1, iters)):
            fn()
        return {"mean_ms": (time.perf_counter() - start) * 1000.0 / max(1, iters), "method": "wall"}
    for _ in range(max(0, warmup)):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(max(1, iters)):
        fn()
    end.record()
    torch.cuda.synchronize()
    return {"mean_ms": start.elapsed_time(end) / max(1, iters), "method": "cuda_event", "iters": iters, "warmup": warmup}


def feature_version_probe() -> dict[str, Any]:
    result: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": getattr(torch, "__version__", None),
        "torch_cuda": getattr(torch.version, "cuda", None),
        "cuda_available": torch.cuda.is_available(),
        "float8_e8m0fnu": hasattr(torch, "float8_e8m0fnu"),
        "float8_e4m3fn": hasattr(torch, "float8_e4m3fn"),
        "torch_scaled_mm": hasattr(torch, "_scaled_mm"),
        "torch_scaled_mm_v2": hasattr(torch, "_scaled_mm_v2"),
        "torch_nn_functional_scaled_mm": hasattr(torch.nn.functional, "scaled_mm"),
    }
    if torch.cuda.is_available():
        result.update({
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_capability": torch.cuda.get_device_capability(0),
            "device_count": torch.cuda.device_count(),
        })
    try:
        import triton
        result["triton"] = getattr(triton, "__version__", None)
    except Exception as e:
        result["triton_error"] = repr(e)
    try:
        import torchao
        result["torchao"] = getattr(torchao, "__version__", None)
        result["torchao_file"] = getattr(torchao, "__file__", None)
    except Exception as e:
        result["torchao_error"] = repr(e)
    try:
        from torch.nn.functional import ScalingType, SwizzleType
        result["ScalingType_BlockWise1x32"] = hasattr(ScalingType, "BlockWise1x32")
        result["SwizzleType_SWIZZLE_32_4_4"] = hasattr(SwizzleType, "SWIZZLE_32_4_4")
    except Exception as e:
        result["scaling_swizzle_error"] = repr(e)
    try:
        from torchao.prototype.mx_formats.inference_workflow import MXDynamicActivationMXWeightConfig
        import inspect
        result["mxdynamic_config_signature"] = str(inspect.signature(MXDynamicActivationMXWeightConfig))
    except Exception as e:
        result["mxdynamic_config_signature_error"] = repr(e)
    return result


def _make_config(kernel_preference: str = "AUTO", scaling_mode: str | None = None):
    from torchao.prototype.mx_formats.inference_workflow import MXDynamicActivationMXWeightConfig
    from torchao.quantization.quantize_.common.kernel_preference import KernelPreference
    kwargs = {
        "activation_dtype": torch.float8_e4m3fn,
        "weight_dtype": torch.float8_e4m3fn,
        "kernel_preference": getattr(KernelPreference, kernel_preference),
    }
    if scaling_mode is not None:
        from torchao.prototype.mx_formats.config import ScaleCalculationMode
        kwargs["scaling_mode"] = getattr(ScaleCalculationMode, scaling_mode)
    return MXDynamicActivationMXWeightConfig(**kwargs)


def _quantize_linear(in_features=1024, out_features=1024, batch=64, kernel_preference="AUTO", scaling_mode: str | None = None):
    from torchao.quantization import quantize_
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(1234)
    base = torch.nn.Linear(in_features, out_features, bias=False, device=device, dtype=torch.bfloat16).eval()
    q = torch.nn.Linear(in_features, out_features, bias=False, device=device, dtype=torch.bfloat16).eval()
    q.load_state_dict(base.state_dict())
    x = torch.randn(batch, in_features, device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        baseline = base(x)
    quantize_(q, _make_config(kernel_preference=kernel_preference, scaling_mode=scaling_mode), device=device)
    with torch.no_grad():
        out = q(x)
    return base, q, x, baseline, out


def _mx_weight_summary(weight: Any) -> dict[str, Any]:
    summary = {"type": type(weight).__name__, "class": f"{type(weight).__module__}.{type(weight).__name__}", "is_parameter": isinstance(weight, torch.nn.Parameter)}
    target = weight.detach() if isinstance(weight, torch.nn.Parameter) else weight
    for attr in ("qdata", "_data", "elem_dtype", "block_size", "scale", "_scale", "is_contiguous"):
        try:
            value = getattr(target, attr)
        except Exception:
            continue
        if callable(value) and attr == "is_contiguous":
            try: value = value()
            except Exception as e: value = repr(e)
        if isinstance(value, torch.Tensor):
            summary[attr] = {"dtype": str(value.dtype), "shape": tuple(value.shape), "is_contiguous": value.is_contiguous()}
            if attr == "qdata":
                try: summary["qdata_t_is_contiguous"] = value.t().is_contiguous()
                except Exception as e: summary["qdata_t_is_contiguous_error"] = repr(e)
        else:
            summary[attr] = value
    return summary


def torchao_mx_smoke_test() -> dict[str, Any]:
    result: dict[str, Any] = {}
    try:
        _, q, _, baseline, out = _quantize_linear()
        diff = (out.float() - baseline.float()).abs()
        result.update({
            "ok": True,
            "baseline_dtype": str(baseline.dtype),
            "output_dtype": str(out.dtype),
            "max_abs_error": float(diff.max().item()),
            "mean_abs_error": float(diff.mean().item()),
            "weight": _mx_weight_summary(q.weight),
        })
    except Exception as e:
        result.update({"ok": False, "error": repr(e)})
    return result


def shape_rejection_matrix() -> list[dict[str, Any]]:
    cases = [
        {"name": "baseline_1024x1024", "in_features": 1024, "out_features": 1024, "batch": 64, "expected_ok": True},
        {"name": "k_not_multiple_32", "in_features": 1000, "out_features": 1024, "batch": 64, "expected_ok": False},
        {"name": "n_not_multiple_16", "in_features": 1024, "out_features": 1030, "batch": 64, "expected_ok": False},
        {"name": "n_multiple_16_not_32", "in_features": 1024, "out_features": 1040, "batch": 64, "expected_ok": True},
        {"name": "m_not_multiple_16", "in_features": 1024, "out_features": 1024, "batch": 65, "expected_ok": True},
        {"name": "multiple_32_not_128", "in_features": 1056, "out_features": 1056, "batch": 64, "expected_ok": True},
    ]
    rows = []
    for case in cases:
        row = dict(case)
        try:
            _, q, _, baseline, out = _quantize_linear(case["in_features"], case["out_features"], case["batch"])
            diff = (out.float() - baseline.float()).abs()
            row.update({"ok": True, "output_dtype": str(out.dtype), "max_abs_error": float(diff.max().item()), "mean_abs_error": float(diff.mean().item()), "weight": _mx_weight_summary(q.weight)})
        except Exception as e:
            row.update({"ok": False, "error": repr(e)})
        row["matches_expected"] = bool(row.get("ok")) == bool(row.get("expected_ok"))
        rows.append(row)
    return rows


def rceil_vs_floor_accuracy() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for mode in ("RCEIL", "FLOOR"):
        try:
            _, q, _, baseline, out = _quantize_linear(scaling_mode=mode)
            diff = (out.float() - baseline.float()).abs()
            result[mode.lower()] = {"ok": True, "max_abs_error": float(diff.max().item()), "mean_abs_error": float(diff.mean().item()), "output_dtype": str(out.dtype), "weight": _mx_weight_summary(q.weight)}
        except Exception as e:
            result[mode.lower()] = {"ok": False, "error": repr(e)}
    return result


def native_vs_emulated_detection() -> dict[str, Any]:
    result: dict[str, Any] = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        torch.manual_seed(5678)
        base = torch.nn.Linear(2048, 2048, bias=False, device=device, dtype=torch.bfloat16).eval()
        x = torch.randn(128, 2048, device=device, dtype=torch.bfloat16)
        with torch.no_grad():
            base(x)
        result["bf16"] = _timed_cuda(lambda: base(x), warmup=5, iters=20)
    except Exception as e:
        result["bf16"] = {"error": repr(e)}
    for pref in ("AUTO", "EMULATED"):
        try:
            _, q, x, _, _ = _quantize_linear(2048, 2048, 128, kernel_preference=pref)
            result[pref.lower()] = _timed_cuda(lambda: q(x), warmup=5, iters=20)
            result[pref.lower()]["weight"] = _mx_weight_summary(q.weight)
        except Exception as e:
            result[pref.lower()] = {"error": repr(e)}
    try:
        if "auto" in result and "emulated" in result and "mean_ms" in result["auto"] and "mean_ms" in result["emulated"]:
            result["auto_vs_emulated_ratio"] = result["auto"]["mean_ms"] / max(result["emulated"]["mean_ms"], 1e-9)
    except Exception:
        pass
    return result


def sdpa_coverage_check() -> dict[str, Any]:
    import torch.nn.functional as F
    calls = []
    orig = F.scaled_dot_product_attention
    def wrapped(q, k, v, *args, **kwargs):
        calls.append({"q_dtype": str(q.dtype), "k_dtype": str(k.dtype), "v_dtype": str(v.dtype), "q_shape": tuple(q.shape), "k_shape": tuple(k.shape), "v_shape": tuple(v.shape)})
        return orig(q, k, v, *args, **kwargs)
    try:
        F.scaled_dot_product_attention = wrapped
        device = "cuda" if torch.cuda.is_available() else "cpu"
        q = torch.randn(2, 8, 64, 64, device=device, dtype=torch.bfloat16)
        k = torch.randn(2, 8, 64, 64, device=device, dtype=torch.bfloat16)
        v = torch.randn(2, 8, 64, 64, device=device, dtype=torch.bfloat16)
        out = F.scaled_dot_product_attention(q, k, v)
        return {"ok": True, "call_count": len(calls), "calls": calls, "output_dtype": str(out.dtype), "note": "Standalone SDPA call remains bf16; MXFP8 Linear quantization does not by itself replace SDPA kernels."}
    except Exception as e:
        return {"ok": False, "error": repr(e), "calls": calls}
    finally:
        F.scaled_dot_product_attention = orig


def a1111_integration_audit(max_names: int = 160) -> dict[str, Any]:
    try:
        from modules import shared, sd_models
        from torchao.prototype.mx_formats.mx_tensor import MXTensor
    except Exception as e:
        return {"ok": False, "error": repr(e)}
    model = getattr(getattr(sd_models, "model_data", None), "sd_model", None)
    if model is None:
        return {"ok": False, "error": "sd_model is not loaded"}
    rows = []
    skipped_reasons: dict[str, int] = {}
    eligible = 0
    quantized = 0
    managed_bf16_active_lora = 0
    linear_total = 0
    for fqn, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        linear_total += 1
        reason = sd_models.mxfp8_linear_skip_reason(module, fqn) if hasattr(sd_models, "mxfp8_linear_skip_reason") else None
        weight = getattr(module, "weight", None)
        shape = tuple(weight.shape) if weight is not None else None
        is_mx = isinstance(weight, MXTensor)
        is_managed_bf16 = getattr(module, "network_mxfp8_base_weight", None) is not None and not is_mx
        if is_managed_bf16:
            managed_bf16_active_lora += 1
        if reason is None:
            eligible += 1
        else:
            skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1
        if is_mx:
            quantized += 1
        if len(rows) < max_names:
            rows.append({"name": fqn, "shape": shape, "eligible": reason is None, "skip_reason": reason, "is_mxfp8": is_mx, "is_mxfp8_managed_bf16": is_managed_bf16, "weight_type": type(weight).__name__ if weight is not None else None})
    stats = getattr(model, "mxfp8_quantization_stats", None)
    prepare_stats = getattr(model, "network_mxfp8_prepare_stats", None)
    prepare_signature = getattr(model, "network_mxfp8_active_config_signature", None)
    prepare_error = getattr(model, "network_mxfp8_prepare_error", None)
    coverage = getattr(shared.opts, "mxfp8_linear_coverage", None)
    return {"ok": True, "linear_total": linear_total, "eligible_linear": eligible, "quantized_linear": quantized, "mxfp8_managed_bf16_active_lora_linear": managed_bf16_active_lora, "skipped_linear": linear_total - eligible, "skipped_reasons": skipped_reasons, "mxfp8_linear_coverage": coverage, "model_stats": stats, "prepare_stats": prepare_stats, "mxfp8_active_config_stats": prepare_stats, "prepare_signature_active": prepare_signature is not None, "prepare_error": prepare_error, "sample_layers": rows}


def _check_ok(value: Any) -> bool:
    if isinstance(value, dict):
        if value.get("skipped") is True:
            return True
        if "ok" in value:
            return bool(value.get("ok"))
        nested = [v for v in value.values() if isinstance(v, dict) and ("ok" in v or v.get("skipped") is True)]
        return all(_check_ok(v) for v in nested) if nested else "error" not in value
    if isinstance(value, list):
        if all(isinstance(v, dict) and "matches_expected" in v for v in value):
            return all(bool(v.get("matches_expected")) for v in value)
        return all(_check_ok(v) for v in value)
    return True


def _diagnostic_summary(result: dict[str, Any]) -> dict[str, Any]:
    checks = [
        ("feature_version_probe", "Feature/version probe", result.get("feature_versions")),
        ("torchao_mx_smoke_test", "TorchAO MX smoke test", result.get("torchao_mx_smoke")),
        ("shape_rejection_matrix", "Shape rejection matrix", result.get("shape_rejection_matrix")),
        ("rceil_vs_floor_accuracy", "RCEIL vs FLOOR accuracy", result.get("rceil_vs_floor_accuracy")),
        ("native_vs_emulated_detection", "Native vs emulated detection", result.get("native_vs_emulated_detection")),
        ("sdpa_coverage_check", "SDPA coverage check", result.get("sdpa_coverage_check")),
        ("a1111_integration_audit", "A1111 integration audit", result.get("a1111_integration_audit")),
    ]
    rows = []
    for key, label, value in checks:
        rows.append({"key": key, "label": label, "ok": _check_ok(value)})
    failed = [row["key"] for row in rows if not row["ok"]]
    return {"ok": not failed, "checks": rows, "failed_checks": failed}


def run_probe(include_benchmarks: bool = True, save: bool = True) -> dict[str, Any]:
    started = time.time()
    result: dict[str, Any] = {"ok": True, "started_at": started, "started_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started))}
    result["feature_versions"] = feature_version_probe()
    result["torchao_mx_smoke"] = torchao_mx_smoke_test()
    result["shape_rejection_matrix"] = shape_rejection_matrix()
    result["rceil_vs_floor_accuracy"] = rceil_vs_floor_accuracy()
    if include_benchmarks:
        result["native_vs_emulated_detection"] = native_vs_emulated_detection()
    else:
        result["native_vs_emulated_detection"] = {"skipped": True}
    result["sdpa_coverage_check"] = sdpa_coverage_check()
    result["a1111_integration_audit"] = a1111_integration_audit()
    result["diagnostic_summary"] = _diagnostic_summary(result)
    result["ok"] = bool(result["diagnostic_summary"]["ok"])
    finished = time.time()
    result["finished_at"] = finished
    result["duration_seconds"] = round(finished - started, 3)
    if save:
        data = _jsonable(result)
        tmp = last_result_path().with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf8")
        tmp.replace(last_result_path())
    return _jsonable(result)


def get_last_result() -> dict[str, Any]:
    global _LAST_RESULT
    with _LOCK:
        if _LAST_RESULT is not None:
            return {"running": _LAST_RUNNING, "result": _LAST_RESULT, "path": str(last_result_path())}
    path = last_result_path()
    if path.exists():
        try:
            return {"running": _LAST_RUNNING, "result": json.loads(path.read_text(encoding="utf8")), "path": str(path)}
        except Exception as e:
            return {"running": _LAST_RUNNING, "error": repr(e), "path": str(path)}
    return {"running": _LAST_RUNNING, "result": None, "path": str(path)}


def run_probe_background(include_benchmarks: bool = True) -> bool:
    global _LAST_RUNNING, _LAST_RESULT
    with _LOCK:
        if _LAST_RUNNING:
            return False
        _LAST_RUNNING = True
    def worker():
        global _LAST_RUNNING, _LAST_RESULT
        try:
            _LAST_RESULT = run_probe(include_benchmarks=include_benchmarks, save=True)
            print(f"MXFP8 diagnostics completed: {last_result_path()}", flush=True)
        except Exception as e:
            _LAST_RESULT = {"ok": False, "error": repr(e), "finished_at": time.time()}
            try:
                last_result_path().write_text(json.dumps(_LAST_RESULT, indent=2, sort_keys=True), encoding="utf8")
            except Exception:
                pass
            print(f"MXFP8 diagnostics failed: {e!r}", flush=True)
        finally:
            with _LOCK:
                _LAST_RUNNING = False
    threading.Thread(target=worker, name="mxfp8-diagnostics", daemon=True).start()
    return True
