import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw

from modules import shared, shared_init

if getattr(shared, "opts", None) is None:
    shared_init.initialize()

from modules import cache as cache_module, hashes, openclaw_lifecycle_epochs, processing, sd_models, sd_vae
from modules.processing import StableDiffusionProcessing, StableDiffusionProcessingImg2Img


def _init_cache_key_processing(monkeypatch, checkpoint_hash="abcd", checkpoint_sha256="sha256", vae_hash="vae-hash"):
    checkpoint = SimpleNamespace(filename="model.safetensors", hash=checkpoint_hash, sha256=checkpoint_sha256)
    sd_model = SimpleNamespace(
        sd_checkpoint_info=checkpoint,
        cond_stage_key="concat",
        is_sdxl_inpaint=False,
    )
    monkeypatch.setattr(processing.shared, "sd_model", sd_model, raising=False)
    monkeypatch.setattr(processing.sd_vae, "get_loaded_vae_name", lambda: "vae", raising=False)
    monkeypatch.setattr(processing.sd_vae, "get_loaded_vae_hash", lambda: vae_hash, raising=False)
    monkeypatch.setattr(processing.opts, "persistent_img2img_init_cache", True, raising=False)
    monkeypatch.setattr(processing.opts, "sd_vae_encode_method", "Full", raising=False)
    monkeypatch.setattr(processing.opts, "inpainting_mask_weight", 1.0, raising=False)
    monkeypatch.setattr(processing.opts, "img2img_background_color", "#ffffff", raising=False)

    p = StableDiffusionProcessingImg2Img.__new__(StableDiffusionProcessingImg2Img)
    p.init_images = [object()]
    p.sd_model_name = "model"
    p.sd_model_hash = checkpoint_hash
    p.sampler = SimpleNamespace(conditioning_key="concat")
    p.width = 64
    p.height = 64
    p.resize_mode = 1
    p.batch_size = 1
    p._record_img2img_init_cache_bypass = lambda reason: None
    p.image_mask = None
    p.latent_mask = None
    return p


def test_img2img_init_cache_key_uses_effective_request_inpainting_mask_weight(monkeypatch):
    p = _init_cache_key_processing(monkeypatch)
    key_images = [Image.new("RGB", (8, 8))]

    p.inpainting_mask_weight = 0.25
    key_low = p._img2img_init_cache_key(key_images, True, False, False)
    p.inpainting_mask_weight = 0.75
    key_high = p._img2img_init_cache_key(key_images, True, False, False)

    assert key_low != key_high


def test_img2img_init_cache_key_changes_when_checkpoint_or_vae_reloads_under_the_same_name(monkeypatch):
    # --no-hashing: every hash is None, so a checkpoint or VAE file replaced in place and reloaded keeps every name
    # field of the key. The lifecycle epochs the reload commits are what make the next identical request miss.
    p = _init_cache_key_processing(monkeypatch, checkpoint_hash=None, checkpoint_sha256=None, vae_hash=None)
    key_images = [Image.new("RGB", (8, 8))]
    first = p._img2img_init_cache_key(key_images, True, False, False)
    assert p._img2img_init_cache_key(key_images, True, False, False) == first

    openclaw_lifecycle_epochs.publish_checkpoint_commit(changed=True)
    after_checkpoint_reload = p._img2img_init_cache_key(key_images, True, False, False)
    assert after_checkpoint_reload != first

    vae_owner = SimpleNamespace()
    openclaw_lifecycle_epochs.note_vae_commit(vae_owner, bytes_changed=True, object_changed=True, publish=True)
    assert p._img2img_init_cache_key(key_images, True, False, False) != after_checkpoint_reload


def test_img2img_init_cache_bypasses_masked_requests(monkeypatch):
    monkeypatch.setattr(processing.opts, "persistent_img2img_init_cache", True, raising=False)

    p = StableDiffusionProcessingImg2Img.__new__(StableDiffusionProcessingImg2Img)
    p.init_images = [object()]
    p.image_mask = object()
    p.latent_mask = None
    p.inpainting_fill = 0
    bypasses = []
    p._record_img2img_init_cache_bypass = bypasses.append

    key_images = [Image.new("RGB", (8, 8))]

    assert p._img2img_init_cache_key(key_images, True, False, False) is None
    assert bypasses == ["masked_request"]


class _GuardLock:
    def __init__(self):
        self.depth = 0
        self.entries = 0

    def __enter__(self):
        self.depth += 1
        self.entries += 1
        return self

    def __exit__(self, exc_type, exc, tb):
        self.depth -= 1


def test_img2img_init_cache_restore_clones_under_cache_lock(monkeypatch):
    guard = _GuardLock()
    monkeypatch.setattr(StableDiffusionProcessing, "cached_img2img_init_lock", guard, raising=False)
    StableDiffusionProcessing.cached_img2img_init = [("key",), {
        "init_latent": np.array([1], dtype=np.float32),
        "image_conditioning": None,
        "mask": None,
        "nmask": None,
        "mask_for_overlay": None,
        "overlay_images": [],
        "color_corrections": [],
        "paste_to": None,
        "is_using_inpainting_conditioning": True,
        "extra_generation_params": {"Cached": "yes"},
    }]
    StableDiffusionProcessing.cached_img2img_init_stats = processing._cache_stats(last_hit=False, cached=True, bypass_reason=None)

    original_clone = processing._clone_cache_value
    def guarded_clone(value):
        assert guard.depth > 0
        return original_clone(value)
    monkeypatch.setattr(processing, "_clone_cache_value", guarded_clone)

    p = StableDiffusionProcessingImg2Img.__new__(StableDiffusionProcessingImg2Img)
    p.extra_generation_params = {}

    assert p._restore_img2img_init_cache(("key",)) is True
    assert guard.entries >= 1
    assert p.is_using_inpainting_conditioning is True
    assert p.extra_generation_params == {"Cached": "yes"}


def test_img2img_init_cache_clear_and_status_use_cache_lock(monkeypatch):
    guard = _GuardLock()
    monkeypatch.setattr(StableDiffusionProcessing, "cached_img2img_init_lock", guard, raising=False)
    StableDiffusionProcessing.cached_img2img_init = [("key",), {"init_latent": object()}]
    StableDiffusionProcessing.cached_img2img_init_stats = processing._cache_stats(last_hit=True, cached=True, bypass_reason=None)

    StableDiffusionProcessingImg2Img.clear_img2img_init_cache()
    status = StableDiffusionProcessingImg2Img.img2img_init_cache_status()

    assert guard.entries == 2
    assert StableDiffusionProcessing.cached_img2img_init == [None, None]
    assert status["cached"] is False
    assert status["hits"] == 0
    assert status["misses"] == 0


def test_resize_latent_mask_uses_area_coverage_when_rounding():
    mask = Image.new("L", (8, 8), 0)
    ImageDraw.Draw(mask).rectangle((0, 0, 3, 3), fill=255)

    assert processing._resize_latent_mask(mask, (2, 2), round=True).tolist() == [[1.0, 0.0], [0.0, 0.0]]


def test_cached_data_for_file_invalidates_when_mtime_moves_backward(tmp_path, monkeypatch):
    cache_module.caches.clear()
    monkeypatch.setattr(cache_module, "cache_dir", str(tmp_path / "cache"))

    source = tmp_path / "metadata.txt"
    source.write_text("first", encoding="utf-8")
    calls = []

    def build_value():
        calls.append(len(calls) + 1)
        return {"value": calls[-1]}

    assert cache_module.cached_data_for_file("test-metadata", "entry", str(source), build_value) == {"value": 1}
    original_mtime = os.stat(source).st_mtime

    source.write_text("second", encoding="utf-8")
    os.utime(source, (original_mtime - 10, original_mtime - 10))

    assert cache_module.cached_data_for_file("test-metadata", "entry", str(source), build_value) == {"value": 2}


def test_cached_data_for_file_invalidates_legacy_entry_without_size(tmp_path, monkeypatch):
    cache_module.caches.clear()
    monkeypatch.setattr(cache_module, "cache_dir", str(tmp_path / "cache"))

    source = tmp_path / "metadata.txt"
    source.write_text("first", encoding="utf-8")
    stat = os.stat(source)

    existing_cache = cache_module.cache("test-metadata")
    existing_cache["entry"] = {"mtime": stat.st_mtime, "value": {"value": "stale"}}

    source.write_text("second content", encoding="utf-8")
    os.utime(source, (stat.st_mtime, stat.st_mtime))

    calls = []

    def build_value():
        calls.append(1)
        return {"value": "fresh"}

    assert cache_module.cached_data_for_file("test-metadata", "entry", str(source), build_value) == {"value": "fresh"}
    assert calls == [1]


def test_sha256_cache_rejects_size_mismatch(tmp_path, monkeypatch):
    source = tmp_path / "model.safetensors"
    source.write_bytes(b"current")
    stat = os.stat(source)
    fake_cache = {
        "model": {
            "mtime": stat.st_mtime,
            "size": stat.st_size + 1,
            "sha256": "stale",
        }
    }
    monkeypatch.setattr(hashes, "cache", lambda _subsection: fake_cache)

    assert hashes.sha256_from_cache(str(source), "model") is None


def test_checkpoint_state_dict_cache_invalidates_when_checkpoint_file_changes(tmp_path, monkeypatch):
    sd_models.checkpoints_loaded.clear()
    checkpoint_file = tmp_path / "model.ckpt"
    checkpoint_file.write_bytes(b"first")
    checkpoint_info = sd_models.CheckpointInfo(str(checkpoint_file))
    timer = SimpleNamespace(record=lambda _label: None)
    calls = []

    def read_state_dict(filename):
        calls.append(Path(filename).read_bytes())
        return {"value": len(calls)}

    monkeypatch.setattr(sd_models, "read_state_dict", read_state_dict)
    monkeypatch.setattr(checkpoint_info, "calculate_shorthash", lambda: "hash")

    first = sd_models.get_checkpoint_state_dict(checkpoint_info, timer)
    assert first == {"value": 1}

    checkpoint_file.write_bytes(b"second-content")

    second = sd_models.get_checkpoint_state_dict(checkpoint_info, timer)
    assert second == {"value": 2}
    assert calls == [b"first", b"second-content"]


def test_vae_checkpoint_cache_invalidates_when_vae_file_changes(tmp_path, monkeypatch):
    sd_vae.checkpoints_loaded.clear()
    vae_file = tmp_path / "model.vae.pt"
    vae_file.write_bytes(b"first")
    model = SimpleNamespace(
        sd_checkpoint_info=SimpleNamespace(filename="checkpoint.safetensors"),
        first_stage_model=SimpleNamespace(state_dict=lambda: {"base": "vae"}),
    )
    loaded = []

    monkeypatch.setattr(sd_vae.shared.opts, "sd_vae_checkpoint_cache", 1, raising=False)
    monkeypatch.setattr(sd_vae.shared, "weight_load_location", "cpu", raising=False)
    monkeypatch.setattr(sd_vae, "_load_vae_dict", lambda _model, vae_dict: loaded.append(dict(vae_dict)))

    def load_vae_dict(filename, map_location):
        return {"payload": Path(filename).read_bytes()}

    monkeypatch.setattr(sd_vae, "load_vae_dict", load_vae_dict)

    sd_vae.load_vae(model, str(vae_file), "test")
    vae_file.write_bytes(b"second-content")
    sd_vae.load_vae(model, str(vae_file), "test")

    assert loaded == [{"payload": b"first"}, {"payload": b"second-content"}]
