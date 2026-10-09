"""TeaCacheScript process/postprocess and UNet patch lifecycle (scripts/teacache.py); fixtures are in conftest.py."""

import types
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]


def _processing_with_unet(unet, **attrs):
    p = types.SimpleNamespace(
        sd_model=types.SimpleNamespace(model=types.SimpleNamespace(diffusion_model=unet), is_sdxl=True),
        extra_generation_params={},
        steps=20,
    )
    for key, value in attrs.items():
        setattr(p, key, value)
    return p


def test_masked_denoising_disables_cache(teacache):
    p = types.SimpleNamespace(mask=torch.zeros((1, 1)), nmask=None, image_mask=None)
    assert teacache._has_masked_denoising(p)

    p = types.SimpleNamespace(mask=None, nmask=None, image_mask=None)
    assert not teacache._has_masked_denoising(p)


def test_controlnet_owner_marker_is_an_external_unet_forward_hook(teacache):
    unet = types.SimpleNamespace(_controlnet_forward_hook_owner=object())

    assert teacache._has_external_unet_forward_hook(_processing_with_unet(unet))
    assert not teacache._has_external_unet_forward_hook(types.SimpleNamespace())


def test_process_does_not_wrap_controlnet_owned_unet(teacache):
    original_forward = lambda x, **kwargs: x
    owner = object()
    unet = types.SimpleNamespace(
        forward=original_forward,
        _controlnet_forward_hook_owner=owner,
    )

    teacache.TeaCacheScript().process(_processing_with_unet(unet), True)

    assert unet.forward is original_forward
    assert unet._controlnet_forward_hook_owner is owner
    assert not getattr(unet, "_teacache_patched", False)


def test_masked_enabled_process_does_not_patch_unet(teacache):
    unet = types.SimpleNamespace(forward=lambda *args, **kwargs: None)
    p = _processing_with_unet(unet, mask=object())

    teacache.TeaCacheScript().process(p, True, 0.25, 4, 0.35, 0.90)

    assert not getattr(unet, "_teacache_patched", False)
    assert not hasattr(unet, "_openclaw_teacache_original_forward")
    assert teacache._get_cache() is None


def test_masked_enabled_process_cleans_stale_owned_patch(teacache):
    original_forward = object()
    stale_unet = types.SimpleNamespace(
        _teacache_patched=True,
        _openclaw_teacache_original_forward=original_forward,
    )
    # The live patch TeaCache installs; restore only ever replaces TeaCache's own forward.
    stale_unet.forward = teacache.patched_forward.__get__(stale_unet)
    p = _processing_with_unet(stale_unet, image_mask=object())

    script = teacache.TeaCacheScript()
    script.patched_unet = stale_unet
    script.process(p, True, 0.25, 4, 0.35, 0.90)

    assert stale_unet.forward is original_forward
    assert stale_unet._teacache_patched is False
    assert not hasattr(stale_unet, "_openclaw_teacache_original_forward")
    assert script.patched_unet is None
    assert teacache._get_cache() is None


def test_api_generation_paths_hold_queue_lock_for_teacache_global_state():
    source = (REPO_ROOT / "modules" / "api" / "api.py").read_text()
    text2img = source[source.index("    def text2imgapi"):source.index("    def img2imgapi")]
    img2img = source[source.index("    def img2imgapi"):source.index("    def _run_extras")]

    assert "with self.queue_lock:" in text2img
    assert "processed = self._run_generation_with_scripts" in text2img
    assert text2img.index("with self.queue_lock:") < text2img.index("processed = self._run_generation_with_scripts")
    assert "with self.queue_lock:" in img2img
    assert "processed = self._run_generation_with_scripts" in img2img
    assert img2img.index("with self.queue_lock:") < img2img.index("processed = self._run_generation_with_scripts")


def test_postprocess_clears_session_even_when_current_unet_is_unpatched(teacache):
    script = teacache.TeaCacheScript()
    current_unet = types.SimpleNamespace(_teacache_patched=False)
    teacache._set_cache(teacache.TeaCacheSession(threshold=1.0, max_consecutive=0, start=0.0, end=1.0, steps=10))

    script.postprocess(_processing_with_unet(current_unet))

    assert teacache._get_cache() is None
    assert script.patched_unet is None


def test_patched_forward_exception_restores_unet_and_clears_cache(teacache):
    def original_forward(x, timesteps=None, context=None, y=None, **kwargs):
        raise RuntimeError("synthetic model failure")

    unet = types.SimpleNamespace(num_classes=None)
    unet.forward = teacache.patched_forward.__get__(unet)
    unet._teacache_patched = True
    unet._openclaw_teacache_original_forward = original_forward
    session = teacache.TeaCacheSession(threshold=1.0, max_consecutive=0, start=0.0, end=1.0, steps=10)
    session.disabled_reason = "synthetic exception path"
    teacache._set_cache(session)

    try:
        unet.forward(torch.zeros((1, 1)))
    except RuntimeError as exc:
        assert "synthetic model failure" in str(exc)
    else:
        raise AssertionError("expected synthetic model failure")

    assert unet.forward is original_forward
    assert unet._teacache_patched is False
    assert not hasattr(unet, "_openclaw_teacache_original_forward")
    assert teacache._get_cache() is None


def test_process_restores_previous_patched_unet_when_model_object_changes_before_disable(teacache):
    script = teacache.TeaCacheScript()

    def original_forward(x, timesteps=None, context=None, y=None, **kwargs):
        return x + 1

    first_unet = types.SimpleNamespace(forward=original_forward)
    script.process(_processing_with_unet(first_unet), True)
    patched_forward = first_unet.forward
    assert getattr(first_unet, "_teacache_patched", False)
    assert patched_forward is not original_forward

    second_unet = types.SimpleNamespace(forward=lambda x, **kwargs: x)
    script.process(_processing_with_unet(second_unet), False)

    assert first_unet.forward is original_forward
    assert not first_unet._teacache_patched
    assert not hasattr(first_unet, "_openclaw_teacache_original_forward")
    assert teacache._get_cache() is None


def test_process_restores_old_unet_before_patching_new_model_object(teacache):
    script = teacache.TeaCacheScript()

    def first_forward(x, timesteps=None, context=None, y=None, **kwargs):
        return x + 1

    def second_forward(x, timesteps=None, context=None, y=None, **kwargs):
        return x + 2

    first_unet = types.SimpleNamespace(forward=first_forward)
    second_unet = types.SimpleNamespace(forward=second_forward)
    script.process(_processing_with_unet(first_unet), True)
    script.process(_processing_with_unet(second_unet), True)

    assert first_unet.forward is first_forward
    assert not first_unet._teacache_patched
    assert not hasattr(first_unet, "_openclaw_teacache_original_forward")
    assert getattr(second_unet, "_teacache_patched", False)
    assert second_unet._openclaw_teacache_original_forward is second_forward


def test_patched_forward_falls_back_to_original_when_session_disabled(teacache):
    x = torch.zeros((1, 1), dtype=torch.float32)
    timesteps = torch.zeros((1,), dtype=torch.float32)
    calls = []

    def original_forward(x_arg, timesteps=None, context=None, y=None, **kwargs):
        calls.append((x_arg, timesteps, context, y, kwargs))
        return x_arg + 2

    unet = types.SimpleNamespace(
        num_classes=None,
        _openclaw_teacache_original_forward=original_forward,
    )
    teacache._set_cache(teacache.TeaCacheSession(
        threshold=1.0,
        max_consecutive=0,
        start=0.0,
        end=1.0,
        steps=10,
        disabled_reason="external UNet forward hook",
    ))

    result = teacache.patched_forward(unet, x, timesteps=timesteps)

    torch.testing.assert_close(result, x + 2)
    assert calls == [(x, timesteps, None, None, {})]


def test_patched_forward_falls_back_to_original_for_conditioning_kwargs(teacache):
    x = torch.zeros((1, 1), dtype=torch.float32)
    timesteps = torch.zeros((1,), dtype=torch.float32)
    calls = []

    def original_forward(x_arg, timesteps=None, context=None, y=None, **kwargs):
        calls.append(kwargs)
        return x_arg + 3

    unet = types.SimpleNamespace(
        num_classes=None,
        _openclaw_teacache_original_forward=original_forward,
    )
    teacache._set_cache(teacache.TeaCacheSession(threshold=1.0, max_consecutive=0, start=0.0, end=1.0, steps=10))

    result = teacache.patched_forward(unet, x, timesteps=timesteps, transformer_options={"control": True})

    torch.testing.assert_close(result, x + 3)
    assert calls == [{"transformer_options": {"control": True}}]
