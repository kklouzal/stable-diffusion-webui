from __future__ import annotations

import hashlib
import json
import threading
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

SCHEMA_VERSION = 1
MAX_REASON_CODES = 32
MAX_KEY_SUMMARIES = 32
KEY_DIGEST_LENGTH = 16

EVENTS = frozenset({
    "hit", "miss", "bypass", "invalidate", "eviction", "publish", "reject",
    "owner_acquire", "owner_release", "owner_reject", "dependency_observe",
})

REASON_CODES = frozenset({
    "cache_hit", "cache_miss", "cache_disabled", "cache_cleared", "capacity",
    "dependency_changed", "entry_invalid", "capture_failed", "capture_skipped",
    "replay_failed", "unsupported", "unsafe_input", "owner_acquired",
    "owner_released", "owner_mismatch", "dependency_snapshot", "published",
    "rejected", "manual", "startup", "other",
})

DEPENDENCY_DIMENSIONS = (
    "source_bytes_epoch", "checkpoint_object_epoch", "model_movement_epoch",
    "vae_bytes_epoch", "vae_object_epoch", "conditioner_epoch",
    "textual_inversion_epoch", "tokenizer_epoch", "conditioning_hook_epoch",
    "lora_source_epoch", "lora_applied_epoch", "forward_hook_epoch",
    "precision_epoch", "device_epoch", "attention_epoch", "compile_epoch",
    "input_content_epoch", "mask_epoch", "preprocessor_transform_epoch",
    "controller_runtime_epoch", "request_owner",
)

# Static S00 contract declaration. Later slices strengthen runtime keys/epochs without
# changing the coverage schema or exposing source/request material.
_FAMILY_NAMES = {
    "E01": "Persistent source metadata/hash/listing",
    "E02": "Checkpoint state dict and loaded object pool",
    "E03": "VAE state dict/base backup",
    "E04": "VAE approximation models",
    "E05": "Conditioning c/uc/hires",
    "E06": "Per-call prompt token memo",
    "E07": "Textual inversion database/image embeddings",
    "E08": "Sigma/timestep generation profiles",
    "E09": "Sampler callable/reflection registry",
    "E10": "CFG denoiser layout memo",
    "E11": "UNet/VAE CUDA graphs",
    "E12": "LoRA parsed and applied weights",
    "E13": "TorchAO persistent artifacts",
    "E14": "Precision diagnostics snapshot",
    "E15": "Img2img initialized latent",
    "E16": "Upscaler/lazy/compiled models",
    "E17": "App compile wrappers",
    "E18": "Hypertile pure helper memo",
    "E19": "TeaCache residual session",
    "E20": "Controller preprocess payload cache",
    "E21": "Controller current input/result handoff",
    "E22": "Controller state/presets/experiments",
    "E23": "Controller status/runtime/progress",
    "E24": "Controller embedding/network metadata",
    "E25": "Controller LoRA normalization/selection/payload",
    "E26": "Controller ControlNet wire path",
    "E27": "Controller gallery and thumbnails",
    "E28": "Controller inpaint mask",
    "E29": "Controller generation retry",
    "E30": "Cache registry/coverage declaration",
    "E31": "ControlNet preprocessor/model/VAE caches",
    "E32": "Extension forward-hook lifecycle",
    "E33": "MultiDiffusion noise inversion",
    "E34": "Ultimate/Dynamic Threshold/Detail transient state",
}

_SOURCE_CACHE_IDS = {
    "E01": ("core:C01", "core:C02", "core:C03", "core:C23", "extension:file-metadata-and-hash-disk-cache"),
    "E02": ("core:C04", "core:C05", "extension:checkpoint-state-dict-memory"),
    "E03": ("core:C06", "extension:vae-state-dict-memory"),
    "E04": ("core:C07",),
    "E05": ("core:C08", "extension:core-conditioning-c-uc", "extension:hires-conditioning-c-uc"),
    "E06": ("core:C09",),
    "E07": ("core:C10",),
    "E08": ("core:C11",),
    "E09": ("core:C12", "extension:multi-sampler-function-signature-caches"),
    "E10": ("core:C13",),
    "E11": ("core:C14", "core:C15", "extension:cuda-unet-graphs", "extension:vae-decode-graphs"),
    "E12": ("core:C16", "core:C17", "extension:lora-loaded-network-cache"),
    "E13": ("core:C18",),
    "E14": ("core:C19",),
    "E15": ("core:C20", "extension:img2img-init-latent"),
    "E16": ("core:C21", "extension:upscaler-registry-model-load"),
    "E17": ("core:C22",),
    "E18": ("core:C24",),
    "E19": ("core:C25", "extension:teacache-residuals"),
    "E20": ("controller:C01", "controller:C02"),
    "E21": ("controller:C03", "controller:C04"),
    "E22": ("controller:C05", "controller:C17", "controller:C18"),
    "E23": ("controller:C06", "controller:C07", "controller:C08"),
    "E24": ("controller:C09", "controller:C16", "controller:C20"),
    "E25": ("controller:C10", "controller:C11", "controller:C12"),
    "E26": ("controller:C13",),
    "E27": ("controller:C14", "controller:C15"),
    "E28": ("controller:C19",),
    "E29": ("controller:C21",),
    "E30": ("controller:C22",),
    "E31": ("extension:controlnet-preprocessor-lru", "extension:controlnet-model-cache", "extension:controlnet-vae-tensor-cache"),
    "E32": ("extension:controlnet-unet-hook-lifecycle", "extension:multidiffusion-global-hijacks", "extension:incantations-pag-seg-hooks", "extension:clear-cond-cache-api"),
    "E33": ("extension:multidiffusion-noise-inverse-cache",),
    "E34": ("extension:ultimate-upscale-state-lifecycle", "extension:dynamic-threshold-transient-samplers", "extension:detail-daemon-step-state"),
}

_FAMILY_DEPENDENCIES = {
    "E01": (("source_bytes_epoch", "parser_schema", "source_presence"), ("request_owner", "seed", "cfg", "prompt", "device_dtype")),
    "E02": (("source_bytes_epoch", "checkpoint_object_epoch", "precision_epoch", "loader_schema"), ("prompt", "seed", "cfg", "sampler", "independent_vae")),
    "E03": (("vae_bytes_epoch", "vae_object_epoch", "precision_epoch", "device_epoch", "loader_schema"), ("prompt", "seed", "cfg", "sampler", "unpatched_lora")),
    "E04": (("source_bytes_epoch", "vae_object_epoch", "precision_epoch", "device_epoch", "implementation_revision"), ("prompt", "seed", "cfg", "sampler")),
    "E05": (("checkpoint_object_epoch", "conditioner_epoch", "textual_inversion_epoch", "tokenizer_epoch", "conditioning_hook_epoch", "lora_applied_epoch", "precision_epoch", "device_epoch", "prompt_schedule_shape"), ("seed", "cfg", "unchanged_effective_schedule", "independent_vae")),
    "E06": (("call_prompt", "tokenizer_epoch"), ("post_call_events",)),
    "E07": (("embedding_source", "tokenizer_epoch", "loader_schema", "reload_commit"), ("seed", "cfg", "sampler", "image_size")),
    "E08": (("sampler", "scheduler", "steps", "model_schedule", "device_epoch", "precision_epoch", "registry_revision"), ("seed", "prompt", "negative_prompt", "cfg", "batch", "image_mask", "ordinary_lora")),
    "E09": (("sampler_registry", "callable_identity", "forward_hook_epoch", "extension_reload"), ("prompt", "seed", "cfg", "image")),
    "E10": (("denoiser_layout", "batch", "device_epoch", "precision_epoch"), ("other_requests",)),
    "E11": (("checkpoint_object_epoch", "model_movement_epoch", "vae_object_epoch", "lora_applied_epoch", "forward_hook_epoch", "precision_epoch", "device_epoch", "attention_epoch", "compile_epoch", "tensor_layout", "unsafe_mode"), ("dynamic_seed_noise", "dynamic_prompt_tensor", "dynamic_cfg")),
    "E12": (("lora_source_epoch", "ordered_application", "checkpoint_object_epoch", "precision_epoch", "device_epoch", "request_owner"), ("prompt_without_lora_change", "seed", "cfg", "sampler")),
    "E13": (("source_bytes_epoch", "precision_epoch", "backend_schema", "payload_digest"), ("prompt", "seed", "cfg", "sampler", "request_owner")),
    "E14": (("checkpoint_object_epoch", "precision_epoch", "device_epoch", "lora_applied_epoch", "diagnostic_schema"), ("prompt", "seed", "cfg", "sampler")),
    "E15": (("input_content_epoch", "mask_epoch", "checkpoint_object_epoch", "vae_object_epoch", "precision_epoch", "device_epoch", "preprocess_revision", "dimensions_batch"), ("prompt", "negative_prompt", "seed_without_seeded_fill", "cfg", "sampler")),
    "E16": (("source_bytes_epoch", "device_epoch", "precision_epoch", "compile_epoch", "transform_revision"), ("prompt", "seed", "cfg", "sampler")),
    "E17": (("checkpoint_object_epoch", "vae_object_epoch", "forward_hook_epoch", "precision_epoch", "device_epoch", "attention_epoch", "compile_epoch"), ("prompt", "seed", "dynamic_cfg")),
    "E18": (("implementation_revision", "function_arguments"), ("model_request_source_epochs",)),
    "E19": (("request_owner", "conditioner_epoch", "checkpoint_object_epoch", "lora_applied_epoch", "forward_hook_epoch", "precision_epoch", "device_epoch", "layout_steps", "exception_end"), ("controller_metadata", "loaded_source_timestamps")),
    "E20": (("input_content_epoch", "preprocessor_transform_epoch", "geometry", "encoder_schema", "output_integrity"), ("prompt", "seed", "cfg", "lora", "independent_checkpoint")),
    "E21": (("input_content_epoch", "generation_uuid", "manifest_revision"), ("timestamp_only", "path_only")),
    "E22": (("cas_revision", "schema_revision", "generation_uuid"), ("unrelated_field_update",)),
    "E23": (("controller_runtime_epoch", "options_revision", "mutation_commit", "request_uuid", "ttl"), ("prompt_image_only",)),
    "E24": (("source_api_revision", "refresh", "schema_ttl"), ("seed", "cfg", "sampler")),
    "E25": (("lora_source_epoch", "converter_revision", "index_revision", "canonical_selection", "runtime_options"), ("format_only_spelling", "seed", "cfg")),
    "E26": (("input_content_epoch", "preprocessor_transform_epoch", "canonical_payload"), ("absent_controller_cache_protocol",)),
    "E27": (("source_bytes_epoch", "index_generation", "thumbnail_revision", "root_digest", "relative_tiebreak"), ("prompt", "seed", "cfg", "model")),
    "E28": (("mask_epoch", "input_content_epoch", "rasterizer_revision"), ("prompt", "seed", "cfg", "sampler")),
    "E29": (("request_uuid", "resolved_seed", "commit_state"), ("proven_uncommitted_retry",)),
    "E30": (("new_cache_surface", "schema_revision"), ("runtime_request_fields",)),
    "E31": (("preprocessor_transform_epoch", "stochastic_seed", "model_source_digest", "input_content_epoch", "vae_object_epoch", "precision_epoch", "device_epoch"), ("unit_weight", "guidance_range", "deterministic_seed")),
    "E32": (("request_owner", "forward_hook_epoch", "lifecycle"), ("unrelated_owner_cleanup",)),
    "E33": (("latent_digest", "checkpoint_object_epoch", "lora_applied_epoch", "vae_object_epoch", "algorithm_revision", "prompt_settings"), ("unused_seed", "ui_formatting")),
    "E34": (("request_owner", "lifecycle", "sampler_registry"), ("fresh_processing_reuse",)),
}

INSTRUMENTED_FAMILIES = frozenset({"E05", "E08", "E11", "E12"})


@dataclass
class _FamilyState:
    events: Counter[str] = field(default_factory=Counter)
    reasons: OrderedDict[str, int] = field(default_factory=OrderedDict)
    keys: OrderedDict[str, int] = field(default_factory=OrderedDict)
    dependencies: OrderedDict[str, int] = field(default_factory=OrderedDict)
    key_observations: int = 0
    dependency_observations: int = 0
    capacity: int | None = None
    current_size: int | None = None
    size_provider: Callable[[], tuple[int | None, int | None]] | None = None


class CacheTelemetryRegistry:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._families = {family_id: _FamilyState() for family_id in _FAMILY_NAMES}

    @staticmethod
    def digest(value: Any) -> str:
        payload = _opaque_bytes(value)
        return hashlib.sha256(payload).hexdigest()[:KEY_DIGEST_LENGTH]

    def register_size_provider(self, family_id: str, provider: Callable[[], tuple[int | None, int | None]]) -> None:
        with self._lock:
            self._state(family_id).size_provider = provider

    def set_size(self, family_id: str, *, current_size: int | None = None, capacity: int | None = None) -> None:
        with self._lock:
            state = self._state(family_id)
            state.current_size = _safe_nonnegative_int(current_size)
            state.capacity = _safe_nonnegative_int(capacity)

    def observe(self, family_id: str, event: str, *, reason: str | None = None, semantic_key: Any = None, count: int = 1) -> None:
        if event not in EVENTS:
            raise ValueError(f"unknown cache telemetry event: {event}")
        if count < 0:
            raise ValueError("count must be non-negative")
        reason = reason if reason in REASON_CODES else "other"
        with self._lock:
            state = self._state(family_id)
            state.events[event] += count
            _bounded_increment(state.reasons, reason, count, MAX_REASON_CODES)
            if semantic_key is not None:
                state.key_observations += count
                _bounded_increment(state.keys, self.digest(semantic_key), count, MAX_KEY_SUMMARIES)

    def observe_dependency(self, family_id: str, dimensions: Mapping[str, Any]) -> str:
        safe = {name: dimensions[name] for name in sorted(dimensions) if name in DEPENDENCY_DIMENSIONS}
        digest = self.digest(safe)
        with self._lock:
            state = self._state(family_id)
            state.events["dependency_observe"] += 1
            state.dependency_observations += 1
            _bounded_increment(state.reasons, "dependency_snapshot", 1, MAX_REASON_CODES)
            _bounded_increment(state.dependencies, digest, 1, MAX_KEY_SUMMARIES)
        return digest

    def reset(self) -> None:
        with self._lock:
            for state in self._families.values():
                state.events.clear()
                state.reasons.clear()
                state.keys.clear()
                state.dependencies.clear()
                state.key_observations = 0
                state.dependency_observations = 0

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            copied = {
                family_id: (
                    dict(state.events), dict(state.reasons), dict(state.keys),
                    dict(state.dependencies), state.key_observations,
                    state.dependency_observations, state.current_size, state.capacity,
                    state.size_provider,
                )
                for family_id, state in self._families.items()
            }

        families = []
        totals: Counter[str] = Counter()
        for family_id in sorted(_FAMILY_NAMES):
            events, reasons, keys, dependencies, key_total, dependency_total, current_size, capacity, provider = copied[family_id]
            if provider is not None:
                try:
                    supplied_size, supplied_capacity = provider()
                    current_size = _safe_nonnegative_int(supplied_size)
                    capacity = _safe_nonnegative_int(supplied_capacity)
                except Exception:
                    current_size, capacity = None, None
            totals.update(events)
            dirty, stable = _FAMILY_DEPENDENCIES[family_id]
            families.append({
                "id": family_id,
                "name": _FAMILY_NAMES[family_id],
                "instrumented": family_id in INSTRUMENTED_FAMILIES,
                "events": {event: int(events.get(event, 0)) for event in sorted(EVENTS)},
                "reason_counts": reasons,
                "semantic_key_digests": list(keys),
                "semantic_key_observations": key_total,
                "dependency_observation_digests": list(dependencies),
                "dependency_observations": dependency_total,
                "current_size": current_size,
                "capacity": capacity,
                "source_cache_ids": list(_SOURCE_CACHE_IDS[family_id]),
                "contract": {"dirty_on": list(dirty), "stable_on": list(stable)},
            })

        return {
            "schema_version": SCHEMA_VERSION,
            "read_only": True,
            "limits": {"reason_codes": MAX_REASON_CODES, "semantic_keys": MAX_KEY_SUMMARIES, "digest_hex_chars": KEY_DIGEST_LENGTH},
            "reason_code_vocabulary": sorted(REASON_CODES),
            "dependency_dimensions": list(DEPENDENCY_DIMENSIONS),
            "totals": {event: int(totals.get(event, 0)) for event in sorted(EVENTS)},
            "families": families,
        }

    def _state(self, family_id: str) -> _FamilyState:
        try:
            return self._families[family_id]
        except KeyError as exc:
            raise ValueError(f"unknown cache family: {family_id}") from exc


def _safe_nonnegative_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return max(0, int(value))
    except (TypeError, ValueError, OverflowError):
        return None


def _bounded_increment(target: OrderedDict[str, int], key: str, count: int, limit: int) -> None:
    if key in target:
        target[key] += count
        target.move_to_end(key)
        return
    if len(target) >= limit:
        target.popitem(last=False)
    target[key] = count


def _opaque_bytes(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), default=_opaque_fallback).encode("utf-8", "surrogatepass")
    except Exception:
        return b"opaque-unserializable"


def _opaque_fallback(value: Any) -> dict[str, str]:
    # Never call repr()/str() for arbitrary objects: they may contain prompts,
    # paths, filenames, model names, auth material, or user metadata.
    cls = type(value)
    return {"type": f"{cls.__module__}.{cls.__qualname__}"}


registry = CacheTelemetryRegistry()


def observe(family_id: str, event: str, *, reason: str | None = None, semantic_key: Any = None, count: int = 1) -> None:
    registry.observe(family_id, event, reason=reason, semantic_key=semantic_key, count=count)


def observe_dependency(family_id: str, dimensions: Mapping[str, Any]) -> str:
    return registry.observe_dependency(family_id, dimensions)


def register_size_provider(family_id: str, provider: Callable[[], tuple[int | None, int | None]]) -> None:
    registry.register_size_provider(family_id, provider)


def set_size(family_id: str, *, current_size: int | None = None, capacity: int | None = None) -> None:
    registry.set_size(family_id, current_size=current_size, capacity=capacity)


def snapshot() -> dict[str, Any]:
    return registry.snapshot()


def reset_for_tests() -> None:
    registry.reset()
