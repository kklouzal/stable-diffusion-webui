from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Callable, Optional

import torch

from modules import persistent_artifact_cache

SUPPORTED_ROOT_NAMES = ("Stable-diffusion",)
ARTIFACT_SCHEMA_VERSION = 2


def runtime_compatibility() -> dict:
    try:
        import torchao
        torchao_version = getattr(torchao, "__version__", "unknown")
    except Exception:
        torchao_version = "unavailable"
    cuda = getattr(torch.version, "cuda", None)
    device = None
    driver = None
    if torch.cuda.is_available():
        index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        device = {
            "name": props.name,
            "capability": list(torch.cuda.get_device_capability(index)),
            "total_memory": props.total_memory,
        }
        try:
            driver = torch._C._cuda_getDriverVersion()
        except Exception:
            driver = None
    return {
        "python": list(sys.version_info[:3]),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchao": torchao_version,
        "cuda_runtime": cuda,
        "cuda_driver": driver,
        "device": device,
    }


def artifact_contract(config_name: str, coverage=None) -> dict:
    return {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "quantization": {"implementation": config_name, "options": {}, "coverage": sorted(coverage) if coverage is not None else None},
        "runtime": runtime_compatibility(),
    }


def contract_matches(actual: dict | None, expected: dict) -> bool:
    return isinstance(actual, dict) and actual == expected


def is_safetensors(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() == ".safetensors"


def sha256_file(filename: str) -> str:
    h = hashlib.sha256()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_identity(filename: str) -> dict:
    stat = os.stat(filename)
    return {
        "path": os.path.abspath(filename),
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
    }


def stat_source(filename: str, *, identity_trusted: bool = True) -> dict:
    return {
        **file_identity(filename),
        "sha256": sha256_file(filename),
        "identity_trusted": identity_trusted,
    }


def file_identity_matches(filename: str, metadata: dict | None) -> bool:
    if not metadata:
        return False

    identity = file_identity(filename)
    required = ("path", "device", "inode", "size", "mtime_ns", "ctime_ns")
    return all(identity.get(key) == metadata.get(key) for key in required)


def file_metadata_matches(filename: str, metadata: dict | None) -> bool:
    if not metadata or not metadata.get("sha256"):
        return False

    current = file_identity(filename)
    if file_identity_matches(filename, metadata):
        # Fast path: unchanged path/device/inode/size/mtime/ctime identity matches a previously
        # strong-hashed file. The recorded SHA-256 remains part of the trusted
        # metadata, so this path skips repeat hashing but never accepts metadata
        # that was not originally hash-backed.
        return bool(metadata.get("identity_trusted"))

    if current.get("path") != metadata.get("path") or current.get("size") != metadata.get("size"):
        return False

    if sha256_file(filename) != metadata.get("sha256"):
        return False

    metadata.update(file_identity(filename))
    metadata["identity_trusted"] = True
    return True


def tensor_meta(tensor) -> dict:
    return {
        "shape": list(getattr(tensor, "shape", []) or []),
        "dtype": str(getattr(tensor, "dtype", None)),
        "device": str(getattr(tensor, "device", None)),
        "tensor_type": type(tensor).__module__ + "." + type(tensor).__name__,
    }


def device_matches(actual, expected) -> bool:
    actual_device = torch.device(str(actual))
    expected_device = torch.device(expected)
    if expected_device.index is None:
        return actual_device.type == expected_device.type
    return actual_device == expected_device


def metadata_matches(tensor, metadata: dict | None, expected_device) -> bool:
    if metadata:
        actual = tensor_meta(tensor)
        for key in ("shape", "dtype", "tensor_type"):
            if actual.get(key) != metadata.get(key):
                return False
    return device_matches(getattr(tensor, "device", None), expected_device)


def bias_metadata_matches(tensor, metadata: dict | None, expected_device) -> bool:
    if metadata:
        actual = tensor_meta(tensor)
        for key in ("shape", "dtype"):
            if actual.get(key) != metadata.get(key):
                return False
    return device_matches(getattr(tensor, "device", None), expected_device)


def cached_bias_matches(bias, bias_meta: dict | None, module, expected_device) -> bool:
    if bias is None:
        return module.bias is None
    if module.bias is None or list(bias.shape) != list(module.bias.shape):
        return False
    return bias_metadata_matches(bias, bias_meta, expected_device)


def parameter_on_device(tensor, device: torch.device | str) -> torch.nn.Parameter:
    if device_matches(getattr(tensor, "device", None), device):
        return torch.nn.Parameter(tensor, requires_grad=False)
    return torch.nn.Parameter(tensor.to(device=device), requires_grad=False)


def is_cache_path(filename: str, cache_dir_name: str) -> bool:
    return cache_dir_name in Path(filename).parts


def cache_path_for(filename: Optional[str], cache_dir_name: str) -> Optional[str]:
    if not filename or not is_safetensors(filename):
        return None

    path = Path(filename).resolve()
    parts = path.parts
    if cache_dir_name in parts:
        return None

    root_index = None
    for root_name in SUPPORTED_ROOT_NAMES:
        try:
            root_index = parts.index(root_name)
            break
        except ValueError:
            continue

    if root_index is None:
        return None

    root = Path(*parts[:root_index + 1])
    relative = Path(*parts[root_index + 1:])
    if not relative.parts:
        return None

    return str(root / cache_dir_name / relative.with_suffix(relative.suffix + ".pt"))


def sidecar_path(cache_path: str, suffix: str) -> str:
    return cache_path + suffix


def load_sidecar(cache_path: str, suffix: str) -> Optional[dict]:
    try:
        with open(sidecar_path(cache_path, suffix), "r", encoding="utf8") as f:
            return json.load(f)
    except Exception:
        return None


def expected_cache_metadata(filename: str, cache_version: int, config_name: str, coverage=None) -> dict:
    return {
        "cache_version": cache_version,
        "config": config_name,
        "source": stat_source(filename),
        "coverage": sorted(coverage) if coverage is not None else None,
        "contract": artifact_contract(config_name, coverage),
    }


def sidecar_matches(filename: str, cache_path: str, cache_version: int, config_name: str, sidecar_suffix: str, coverage=None) -> bool:
    if not os.path.exists(cache_path):
        return False

    sidecar = load_sidecar(cache_path, sidecar_suffix)
    if not sidecar:
        return False

    sidecar_source = sidecar.get("source")
    sidecar_cache = sidecar.get("cache")
    original_source = dict(sidecar_source or {})
    original_cache = dict(sidecar_cache or {})
    source_matches = file_metadata_matches(filename, sidecar_source)
    cache_matches = file_metadata_matches(cache_path, sidecar_cache)
    if source_matches or cache_matches:
        updated = False
        if source_matches and sidecar_source != original_source:
            sidecar["source"] = sidecar_source
            updated = True
        if cache_matches and sidecar_cache != original_cache:
            sidecar["cache"] = sidecar_cache
            updated = True
        if updated:
            write_atomic_bytes(sidecar_path(cache_path, sidecar_suffix), json.dumps(sidecar, indent=2, sort_keys=True).encode("utf8"))

    expected_coverage = sorted(coverage) if coverage is not None else None
    coverage_matches = coverage is None or sidecar.get("coverage") in (None, expected_coverage)
    return (
        sidecar.get("cache_version") == cache_version
        and sidecar.get("config") == config_name
        and contract_matches(sidecar.get("contract"), artifact_contract(config_name, coverage))
        and source_matches
        and coverage_matches
        and cache_matches
    )


def write_atomic_bytes(path: str, data: bytes) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with NamedTemporaryFile("wb", delete=False, dir=os.path.dirname(path), prefix=".tmp-", suffix=".json") as f:
        tmp = f.name
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def torch_load_cache(cache_path: str, device: torch.device | str, register_safe_globals: Callable[[], None]):
    # A1111 monkey-patches torch.load with a legacy checkpoint pre-check that
    # rejects TorchAO tensor subclasses before PyTorch's weights_only safe
    # unpickler gets a chance to apply add_safe_globals(). Bypass only that
    # outer A1111 pre-check for our own sidecar-validated cache while keeping
    # weights_only=True.
    try:
        from modules import safe
        torch_load = safe.unsafe_torch_load
    except Exception:
        torch_load = torch.load

    register_safe_globals()
    return torch_load(cache_path, map_location=device, weights_only=True)


def iter_eligible_linear_modules(model, filter_fn: Callable):
    for fqn, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and filter_fn(module, fqn):
            yield fqn, module


def load_into_model(
    *,
    model,
    source_path: Optional[str],
    filter_fn: Callable,
    device: torch.device | str,
    coverage=None,
    cache_dir_name: str,
    cache_version: int,
    config_name: str,
    sidecar_suffix: str,
    label: str,
    is_quant_tensor: Callable,
    register_safe_globals: Callable[[], None],
) -> bool:
    cache_path = cache_path_for(source_path, cache_dir_name)
    if cache_path is None or source_path is None or not sidecar_matches(source_path, cache_path, cache_version, config_name, sidecar_suffix, coverage):
        return False

    print(f"Loading {label} cache for {source_path} from {cache_path}", flush=True)
    try:
        # Retention tooling treats the lease as an in-use barrier. Once the
        # payload is deserialized, removal of the disk artifact is harmless.
        with persistent_artifact_cache.active_lease(cache_path):
            payload = torch_load_cache(cache_path, device, register_safe_globals)
    except Exception as e:
        print(f"Ignoring unreadable {label} cache {cache_path}: {e}")
        return False

    if not isinstance(payload, dict):
        print(f"Ignoring unreadable {label} cache {cache_path}: payload is not a dict")
        return False

    expected_coverage = sorted(coverage) if coverage is not None else None
    payload_coverage_matches = coverage is None or payload.get("coverage") in (None, expected_coverage)
    if not (
        payload.get("cache_version") == cache_version
        and payload.get("config") == config_name
        and contract_matches(payload.get("contract"), artifact_contract(config_name, coverage))
        and file_metadata_matches(source_path, payload.get("source"))
        and payload_coverage_matches
    ):
        print(f"Ignoring stale {label} cache {cache_path}: payload metadata does not match requested source/config/coverage")
        return False

    tensors = payload.get("tensors", {})
    metadata = payload.get("metadata", {})
    eligible_modules = list(iter_eligible_linear_modules(model, filter_fn))
    print(f"Validating {label} cache for {source_path}: expected {len(eligible_modules)} Linear modules", flush=True)
    missing = []
    incompatible = []
    for fqn, module in eligible_modules:
        entry = tensors.get(fqn)
        weight = entry.get("weight") if entry is not None else None
        if not is_quant_tensor(weight):
            missing.append(fqn)
            continue
        expected_shape = list(module.weight.shape) if module.weight is not None else []
        weight_meta = metadata.get(fqn, {}).get("weight", {})
        cached_shape = list(getattr(weight, "shape", []) or weight_meta.get("shape", []))
        if cached_shape != expected_shape or not metadata_matches(weight, weight_meta, device):
            incompatible.append({"name": fqn, "expected": expected_shape, "cached": tensor_meta(weight)})
        bias = entry.get("bias")
        bias_meta = metadata.get(fqn, {}).get("bias")
        if not cached_bias_matches(bias, bias_meta, module, device):
            expected_bias = None if module.bias is None else list(module.bias.shape)
            cached_bias = None if bias is None else tensor_meta(bias)
            incompatible.append({"name": fqn + ".bias", "expected": expected_bias, "cached": cached_bias})

    if missing:
        print(f"Ignoring incomplete {label} cache {cache_path}: expected {len(eligible_modules)}, missing {len(missing)}")
        return False
    if incompatible:
        print(f"Ignoring incompatible {label} cache {cache_path}: {incompatible[:5]}")
        return False

    print(f"Assigning {label} cache for {source_path}: {len(eligible_modules)} Linear modules", flush=True)
    with torch.no_grad():
        for fqn, module in eligible_modules:
            entry = tensors[fqn]
            module._parameters["weight"] = entry["weight"]
            bias = entry.get("bias")
            if bias is not None:
                module._parameters["bias"] = parameter_on_device(bias, device)
            elif module.bias is not None:
                module._parameters["bias"] = None

    print(f"Loaded {label} cache for {source_path} from {cache_path}", flush=True)
    return True


def save_from_model(
    *,
    model,
    source_path: Optional[str],
    filter_fn: Callable,
    eligible: int,
    skipped_linear: int,
    skipped_reasons: dict,
    coverage=None,
    cache_dir_name: str,
    cache_version: int,
    config_name: str,
    sidecar_suffix: str,
    label: str,
    is_quant_tensor: Callable,
) -> Optional[str]:
    cache_path = cache_path_for(source_path, cache_dir_name)
    if cache_path is None or source_path is None or eligible == 0:
        return None

    tensors = {}
    metadata = {}
    for fqn, module in iter_eligible_linear_modules(model, filter_fn):
        if not is_quant_tensor(module.weight):
            return None
        tensors[fqn] = {
            "weight": module.weight.detach(),
            "bias": module.bias.detach() if module.bias is not None else None,
        }
        metadata[fqn] = {
            "weight": tensor_meta(module.weight),
            "bias": tensor_meta(module.bias) if module.bias is not None else None,
        }

    source_stat = stat_source(source_path)
    payload = {
        "cache_version": cache_version,
        "config": config_name,
        "source": source_stat,
        "coverage": sorted(coverage) if coverage is not None else None,
        "contract": artifact_contract(config_name, coverage),
        "eligible_linear": eligible,
        "skipped_linear": skipped_linear,
        "skipped_reasons": skipped_reasons,
        "metadata": metadata,
        "tensors": tensors,
    }

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with persistent_artifact_cache.exclusive_lock(cache_path):
        # A unique same-filesystem temporary plus fsync+replace prevents readers
        # from observing a partially serialized artifact. The sidecar is the
        # commit record: a crash before it is published leaves an artifact that
        # validation rejects. Writers are serialized per destination.
        with NamedTemporaryFile("wb", delete=False, dir=os.path.dirname(cache_path), prefix=".partial-", suffix=".pt") as stream:
            tmp_path = stream.name
        try:
            torch.save(payload, tmp_path)
            with open(tmp_path, "rb") as stream:
                os.fsync(stream.fileno())
            os.replace(tmp_path, cache_path)
        finally:
            try:
                os.unlink(tmp_path)
            except FileNotFoundError:
                pass

        sidecar = {
            "cache_version": cache_version,
            "config": config_name,
            "source": source_stat,
            "coverage": sorted(coverage) if coverage is not None else None,
            "contract": artifact_contract(config_name, coverage),
            "cache": stat_source(cache_path),
            "eligible_linear": eligible,
            "skipped_linear": skipped_linear,
            "skipped_reasons": skipped_reasons,
        }
        write_atomic_bytes(sidecar_path(cache_path, sidecar_suffix), json.dumps(sidecar, indent=2, sort_keys=True).encode("utf8"))
    quota_bytes = int(os.environ.get("OPENCLAW_TORCHAO_CACHE_MAX_BYTES", str(256 * 1024 ** 3)))
    quota_dry_run = os.environ.get("OPENCLAW_TORCHAO_CACHE_QUOTA_DRY_RUN", "0") == "1"
    quota = persistent_artifact_cache.enforce_directory_quota(
        os.path.dirname(cache_path), max_bytes=quota_bytes, dry_run=quota_dry_run
    )
    if quota["evicted"] or not quota["within_quota"]:
        print(f"{label} cache quota: {json.dumps(quota, sort_keys=True)}", flush=True)
    print(f"Created {label} cache for {source_path} -> {cache_path}")
    return cache_path
