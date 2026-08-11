"""Successful lifecycle commit helpers for cache epochs.

Markers are private, bounded categories or opaque object tokens. They must never be
included in telemetry: only the fixed reason codes below reach the epoch registry.
Precision, attention, and compile epochs are intentionally deferred because this
repository has no single committed old-to-new transition setter for those states.
"""

from __future__ import annotations

from modules import openclaw_cache_epochs

_MODEL_LOCATION_ATTR = "_openclaw_epoch_location"
_VAE_PENDING_ATTR = "_openclaw_epoch_vae_pending"
_LOCATION_CPU = "cpu"
_LOCATION_ACCELERATOR = "accelerator"


def publish_checkpoint_commit(*, changed: bool, vae_bytes_changed: bool = False, vae_object_changed: bool = False) -> bool:
    if not changed:
        return False
    dimensions = ["checkpoint_object_epoch", "source_bytes_epoch"]
    if vae_bytes_changed:
        dimensions.append("vae_bytes_epoch")
    if vae_object_changed:
        dimensions.append("vae_object_epoch")
    for dimension in dimensions:
        openclaw_cache_epochs.bump_epoch(dimension, reason="checkpoint_commit")
    return True


def model_location_marker(model):
    marker = getattr(model, _MODEL_LOCATION_ATTR, None)
    if marker in (_LOCATION_CPU, _LOCATION_ACCELERATOR):
        return marker
    try:
        device = getattr(model, "device", None)
        if device is None:
            device = next(model.parameters()).device
        return _LOCATION_CPU if getattr(device, "type", str(device).split(":", 1)[0]) == "cpu" else _LOCATION_ACCELERATOR
    except (AttributeError, StopIteration, TypeError):
        return None


def publish_model_movement_commit(model, *, before, to_cpu: bool) -> bool:
    after = _LOCATION_CPU if to_cpu else _LOCATION_ACCELERATOR
    setattr(model, _MODEL_LOCATION_ATTR, after)
    if before is None or before == after:
        return False
    for dimension in ("model_movement_epoch", "device_epoch"):
        openclaw_cache_epochs.bump_epoch(dimension, reason="model_movement_commit")
    return True


def note_vae_commit(model, *, bytes_changed: bool, object_changed: bool, publish: bool) -> bool:
    if not bytes_changed and not object_changed:
        return False
    if not publish:
        prior = getattr(model, _VAE_PENDING_ATTR, (False, False))
        setattr(model, _VAE_PENDING_ATTR, (bool(prior[0] or bytes_changed), bool(prior[1] or object_changed)))
        return False
    dimensions = []
    if bytes_changed:
        dimensions.append("vae_bytes_epoch")
    if object_changed:
        dimensions.append("vae_object_epoch")
    for dimension in dimensions:
        openclaw_cache_epochs.bump_epoch(dimension, reason="vae_commit")
    return True


def take_pending_vae_commit(model) -> tuple[bool, bool]:
    pending = getattr(model, _VAE_PENDING_ATTR, (False, False))
    try:
        delattr(model, _VAE_PENDING_ATTR)
    except AttributeError:
        pass
    return bool(pending[0]), bool(pending[1])


def discard_pending_vae_commit(model) -> None:
    try:
        delattr(model, _VAE_PENDING_ATTR)
    except AttributeError:
        pass
