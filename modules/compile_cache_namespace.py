from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from modules.persistent_artifact_cache import atomic_write

NAMESPACE_SCHEMA_VERSION = 1
APPLICATION_CACHE_ABI = "gb10-a1111-compile-v1"


def namespace_identity(*, app_revision: str, dependencies: dict, runtime: dict, options: dict) -> dict:
    payload = {
        "schema_version": NAMESPACE_SCHEMA_VERSION,
        "application_cache_abi": APPLICATION_CACHE_ABI,
        "app_revision": app_revision,
        "dependencies": dependencies,
        "runtime": runtime,
        "options": options,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf8")
    return {**payload, "digest": hashlib.sha256(canonical).hexdigest()}


def configure_environment(root: str | os.PathLike[str], identity: dict, environ: dict | None = None) -> dict:
    """Coordinate provider-owned caches without inspecting or duplicating them."""
    environ = os.environ if environ is None else environ
    namespace = identity["digest"][:24]
    root = Path(root)
    paths = {
        "TORCHINDUCTOR_CACHE_DIR": root / "torchinductor" / namespace,
        "TRITON_CACHE_DIR": root / "triton" / namespace,
        "CUDA_CACHE_PATH": root / "cuda" / namespace,
    }
    for key, path in paths.items():
        environ.setdefault(key, str(path))
    manifest = root / "namespaces" / (namespace + ".json")
    atomic_write(manifest, (json.dumps(identity, indent=2, sort_keys=True) + "\n").encode("utf8"))
    return {"namespace": namespace, "manifest": str(manifest), "environment": {key: environ[key] for key in paths}}


def plan_namespace_cleanup(root: str | os.PathLike[str], *, keep: set[str]) -> list[str]:
    """Return app-owned namespace directories eligible for operator cleanup.

    Deliberately does not delete anything and never targets provider global cache
    roots. A caller can present this plan for explicit dry-run/maintenance use.
    """
    selected = []
    root = Path(root)
    for provider in ("torchinductor", "triton", "cuda"):
        provider_root = root / provider
        if not provider_root.is_dir():
            continue
        for path in provider_root.iterdir():
            if path.is_dir() and path.name not in keep:
                selected.append(str(path))
    return sorted(selected)
