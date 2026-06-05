import asyncio
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw

from modules import cache as cache_module, hashes, processing
from modules.processing import StableDiffusionProcessing, StableDiffusionProcessingImg2Img, StableDiffusionProcessingTxt2Img


def _load_clear_cond_cache_module():
    path = Path(__file__).parents[1] / "extensions" / "openclaw-clear-cond-cache" / "scripts" / "openclaw_clear_cond_cache.py"
    spec = importlib.util.spec_from_file_location("openclaw_clear_cond_cache_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_clear_cond_cache_supports_granular_targets():
    clear_module = _load_clear_cond_cache_module()

    StableDiffusionProcessing.cached_c = [("c",), "c"]
    StableDiffusionProcessing.cached_uc = [("uc",), "uc"]
    StableDiffusionProcessingTxt2Img.cached_hr_c = [("hr_c",), "hr_c"]
    StableDiffusionProcessingTxt2Img.cached_hr_uc = [("hr_uc",), "hr_uc"]
    StableDiffusionProcessing.cached_img2img_init = [("img2img",), {"init_latent": object()}]

    result = clear_module.clear_cond_cache(["img2img_init"])

    assert result["targets"] == ["img2img_init"]
    assert result["cleared"] == ["StableDiffusionProcessing.cached_img2img_init"]
    assert StableDiffusionProcessing.cached_c[0] == ("c",)
    assert StableDiffusionProcessing.cached_uc[0] == ("uc",)
    assert StableDiffusionProcessingTxt2Img.cached_hr_c[0] == ("hr_c",)
    assert StableDiffusionProcessingTxt2Img.cached_hr_uc[0] == ("hr_uc",)
    assert StableDiffusionProcessing.cached_img2img_init[0] is None


def test_clear_cond_cache_default_preserves_legacy_clear_all_behavior():
    clear_module = _load_clear_cond_cache_module()

    StableDiffusionProcessing.cached_c = [("c",), "c"]
    StableDiffusionProcessing.cached_uc = [("uc",), "uc"]
    StableDiffusionProcessingTxt2Img.cached_hr_c = [("hr_c",), "hr_c"]
    StableDiffusionProcessingTxt2Img.cached_hr_uc = [("hr_uc",), "hr_uc"]
    StableDiffusionProcessing.cached_img2img_init = [("img2img",), {"init_latent": object()}]

    result = clear_module.clear_cond_cache()

    assert result["targets"] == ["c", "hr_c", "hr_uc", "img2img_init", "uc"]
    assert StableDiffusionProcessing.cached_c[0] is None
    assert StableDiffusionProcessing.cached_uc[0] is None
    assert StableDiffusionProcessingTxt2Img.cached_hr_c[0] is None
    assert StableDiffusionProcessingTxt2Img.cached_hr_uc[0] is None
    assert StableDiffusionProcessing.cached_img2img_init[0] is None


def test_token_count_falls_back_to_input_text_when_prompt_schedule_is_empty(monkeypatch):
    clear_module = _load_clear_cond_cache_module()

    monkeypatch.setattr(clear_module.extra_networks, "parse_prompt", lambda text: (text, []))
    monkeypatch.setattr(clear_module.prompt_parser, "get_multicond_prompt_list", lambda prompts: (None, prompts, None))
    monkeypatch.setattr(clear_module.prompt_parser, "get_learned_conditioning_prompt_schedules", lambda _prompts, _steps: [])
    monkeypatch.setattr(
        clear_module,
        "model_hijack",
        SimpleNamespace(get_prompt_lengths=lambda prompt, *_args: (len(prompt), 75)),
    )

    result = clear_module.estimate_token_count("fallback prompt", 20)

    assert result == {"ok": True, "token_count": len("fallback prompt"), "max_length": 75}


def test_token_count_uses_longest_scheduled_prompt(monkeypatch):
    clear_module = _load_clear_cond_cache_module()

    schedules = [
        [[5, "short"], [10, "medium prompt"]],
        [[20, "the longest scheduled prompt"]],
    ]
    monkeypatch.setattr(clear_module.extra_networks, "parse_prompt", lambda text: (text, []))
    monkeypatch.setattr(clear_module.prompt_parser, "get_multicond_prompt_list", lambda prompts: (None, prompts, None))
    monkeypatch.setattr(clear_module.prompt_parser, "get_learned_conditioning_prompt_schedules", lambda _prompts, _steps: schedules)
    monkeypatch.setattr(
        clear_module,
        "model_hijack",
        SimpleNamespace(get_prompt_lengths=lambda prompt, *_args: (len(prompt), 75)),
    )

    result = clear_module.estimate_token_count("ignored base prompt", 20)

    assert result == {"ok": True, "token_count": len("the longest scheduled prompt"), "max_length": 75}


def test_clear_cond_cache_endpoint_uses_queue_lock(monkeypatch):
    clear_module = _load_clear_cond_cache_module()

    class RecordingLock:
        def __init__(self):
            self.active = False
            self.entered = False
            self.exited = False

        def __enter__(self):
            self.entered = True
            self.active = True

        def __exit__(self, exc_type, exc, tb):
            self.active = False
            self.exited = True

    class FakeApp:
        def __init__(self):
            self.routes = {}

        def post(self, path):
            def decorator(fn):
                self.routes[("POST", path)] = fn
                return fn

            return decorator

        def get(self, path):
            def decorator(fn):
                self.routes[("GET", path)] = fn
                return fn

            return decorator

    class FakeRequest:
        async def json(self):
            return {"targets": ["c"]}

    lock = RecordingLock()
    calls = []

    def clear_cond_cache(targets):
        assert lock.active
        calls.append(targets)
        return {"ok": True, "targets": targets}

    monkeypatch.setattr(clear_module.call_queue, "queue_lock", lock)
    monkeypatch.setattr(clear_module, "clear_cond_cache", clear_cond_cache)

    app = FakeApp()
    clear_module.on_app_started(None, app)
    result = asyncio.run(app.routes[("POST", "/sdapi/v1/openclaw/clear-cond-cache")](FakeRequest()))

    assert result == {"ok": True, "targets": ["c"]}
    assert calls == [["c"]]
    assert lock.entered is True
    assert lock.exited is True


def test_img2img_init_cache_key_uses_effective_request_inpainting_mask_weight(monkeypatch):
    checkpoint = SimpleNamespace(filename="model.safetensors", hash="abcd", sha256="sha256")
    sd_model = SimpleNamespace(
        sd_checkpoint_info=checkpoint,
        cond_stage_key="concat",
        is_sdxl_inpaint=False,
    )
    monkeypatch.setattr(processing.shared, "sd_model", sd_model, raising=False)
    monkeypatch.setattr(processing.sd_vae, "get_loaded_vae_name", lambda: "vae", raising=False)
    monkeypatch.setattr(processing.sd_vae, "get_loaded_vae_hash", lambda: "vae-hash", raising=False)
    monkeypatch.setattr(processing.opts, "persistent_img2img_init_cache", True, raising=False)
    monkeypatch.setattr(processing.opts, "sd_vae_encode_method", "Full", raising=False)
    monkeypatch.setattr(processing.opts, "inpainting_mask_weight", 1.0, raising=False)
    monkeypatch.setattr(processing.opts, "img2img_background_color", "#ffffff", raising=False)

    p = StableDiffusionProcessingImg2Img.__new__(StableDiffusionProcessingImg2Img)
    p.init_images = [object()]
    p.inpainting_fill = 0
    p.sd_model_name = "model"
    p.sd_model_hash = "hash"
    p.sampler = SimpleNamespace(conditioning_key="concat")
    p.width = 64
    p.height = 64
    p.resize_mode = 1
    p.batch_size = 1
    p.mask_round = True
    p.inpainting_mask_invert = False
    p.inpaint_full_res = False
    p.inpaint_full_res_padding = 0
    p.mask_blur_x = 0
    p.mask_blur_y = 0
    p._record_img2img_init_cache_bypass = lambda reason: None
    p.image_mask = None
    p.latent_mask = None

    batch_images = np.zeros((1, 3, 8, 8), dtype=np.float32)

    p.inpainting_mask_weight = 0.25
    key_low = p._img2img_init_cache_key(batch_images, None, None, False, False)
    p.inpainting_mask_weight = 0.75
    key_high = p._img2img_init_cache_key(batch_images, None, None, False, False)

    assert key_low != key_high


def test_img2img_init_cache_bypasses_masked_requests(monkeypatch):
    monkeypatch.setattr(processing.opts, "persistent_img2img_init_cache", True, raising=False)

    p = StableDiffusionProcessingImg2Img.__new__(StableDiffusionProcessingImg2Img)
    p.init_images = [object()]
    p.image_mask = object()
    p.latent_mask = None
    p.inpainting_fill = 0
    bypasses = []
    p._record_img2img_init_cache_bypass = bypasses.append

    batch_images = np.zeros((1, 3, 8, 8), dtype=np.float32)

    assert p._img2img_init_cache_key(batch_images, None, None, False, False) is None
    assert bypasses == ["masked_request"]


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
