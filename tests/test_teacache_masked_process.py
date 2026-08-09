import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace


def _install_stub_modules(monkeypatch):
    torch = ModuleType("torch")
    torch.Tensor = object
    monkeypatch.setitem(sys.modules, "torch", torch)

    sgm = ModuleType("sgm")
    sgm_modules = ModuleType("sgm.modules")
    diffusionmodules = ModuleType("sgm.modules.diffusionmodules")
    openaimodel = ModuleType("sgm.modules.diffusionmodules.openaimodel")
    openaimodel.timestep_embedding = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "sgm", sgm)
    monkeypatch.setitem(sys.modules, "sgm.modules", sgm_modules)
    monkeypatch.setitem(sys.modules, "sgm.modules.diffusionmodules", diffusionmodules)
    monkeypatch.setitem(sys.modules, "sgm.modules.diffusionmodules.openaimodel", openaimodel)

    modules = ModuleType("modules")
    headless_ui = ModuleType("modules.headless_ui")
    headless_ui.Row = lambda *args, **kwargs: None
    headless_ui.Slider = lambda *args, **kwargs: None
    headless_ui.Number = lambda *args, **kwargs: None
    processing = ModuleType("modules.processing")
    processing.StableDiffusionProcessing = object
    script_callbacks = ModuleType("modules.script_callbacks")
    script_callbacks.on_cfg_after_cfg = lambda callback: callback
    scripts = ModuleType("modules.scripts")
    scripts.Script = object
    scripts.AlwaysVisible = object()
    sd_samplers_common = ModuleType("modules.sd_samplers_common")
    sd_samplers_common.setup_img2img_steps = lambda p, steps=None: ((steps or getattr(p, steps, 20)), (steps or getattr(p, steps, 20)))
    sd_hijack_unet = ModuleType("modules.sd_hijack_unet")
    sd_hijack_unet.th = SimpleNamespace(cat=lambda *args, **kwargs: None)
    ui_components = ModuleType("modules.ui_components")

    class InputAccordion:
        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            return False
        def __exit__(self, *args):
            return False

    ui_components.InputAccordion = InputAccordion
    monkeypatch.setitem(sys.modules, "modules", modules)
    monkeypatch.setitem(sys.modules, "modules.headless_ui", headless_ui)
    monkeypatch.setitem(sys.modules, "modules.processing", processing)
    monkeypatch.setitem(sys.modules, "modules.script_callbacks", script_callbacks)
    monkeypatch.setitem(sys.modules, "modules.sd_samplers_common", sd_samplers_common)
    monkeypatch.setitem(sys.modules, "modules.scripts", scripts)
    monkeypatch.setitem(sys.modules, "modules.sd_hijack_unet", sd_hijack_unet)
    monkeypatch.setitem(sys.modules, "modules.ui_components", ui_components)


def _load_teacache(monkeypatch):
    _install_stub_modules(monkeypatch)
    path = Path(__file__).resolve().parents[1] / "extensions" / "sd-webui-teacache" / "scripts" / "teacache.py"
    spec = importlib.util.spec_from_file_location("teacache_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _processing_with_unet(unet, **attrs):
    p = SimpleNamespace(
        sd_model=SimpleNamespace(model=SimpleNamespace(diffusion_model=unet), is_sdxl=True),
        extra_generation_params={},
        steps=20,
    )
    for key, value in attrs.items():
        setattr(p, key, value)
    return p


def test_masked_enabled_process_does_not_patch_unet(monkeypatch):
    teacache = _load_teacache(monkeypatch)
    unet = SimpleNamespace(forward=lambda *args, **kwargs: None)
    p = _processing_with_unet(unet, mask=object())

    script = teacache.TeaCacheScript()
    script.process(p, True, 0.25, 4, 0.35, 0.90)

    assert not getattr(unet, "_teacache_patched", False)
    assert not hasattr(unet, "_openclaw_teacache_original_forward")
    assert teacache._get_cache() is None


def test_masked_enabled_process_cleans_stale_owned_patch(monkeypatch):
    teacache = _load_teacache(monkeypatch)
    original_forward = object()
    stale_unet = SimpleNamespace(
        forward="patched-forward",
        _teacache_patched=True,
        _openclaw_teacache_original_forward=original_forward,
    )
    p = _processing_with_unet(stale_unet, image_mask=object())

    script = teacache.TeaCacheScript()
    script.original_forward = original_forward
    script.patched_unet = stale_unet
    script.process(p, True, 0.25, 4, 0.35, 0.90)

    assert stale_unet.forward is original_forward
    assert stale_unet._teacache_patched is False
    assert not hasattr(stale_unet, "_openclaw_teacache_original_forward")
    assert script.original_forward is None
    assert script.patched_unet is None
    assert teacache._get_cache() is None
