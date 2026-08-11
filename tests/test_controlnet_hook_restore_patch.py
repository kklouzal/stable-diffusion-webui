import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
PATCH = ROOT / "gb10" / "patch-controlnet-hook-restore.py"
SOURCE = ROOT / "extensions" / "sd-webui-controlnet" / "scripts" / "hook.py"


def apply_patch(target, *extra):
    return subprocess.run([sys.executable, PATCH, *extra, target], check=True, capture_output=True, text=True)


def patched_hook(tmp_path):
    target = tmp_path / "hook.py"
    target.write_text(SOURCE.read_text(encoding="utf-8"), encoding="utf-8")
    apply_patch(target)
    return _load_patched_hook(target.read_text(encoding="utf-8"), tmp_path), target


def test_controlnet_hook_restore_patch_is_idempotent_and_verifiable(tmp_path):
    target = tmp_path / "hook.py"
    target.write_text(SOURCE.read_text(encoding="utf-8"), encoding="utf-8")
    first = apply_patch(target)
    patched = target.read_text(encoding="utf-8")
    second = apply_patch(target)
    checked = apply_patch(target, "--check")
    assert "ControlNet lifecycle verified" in first.stdout
    assert "ControlNet lifecycle verified" in second.stdout
    assert "ControlNet lifecycle verified" in checked.stdout
    assert target.read_text(encoding="utf-8") == patched
    assert patched.count("OPENCLAW_CONTROLNET_FORWARD_OWNER_V2") == 1
    assert "model._original_forward = model.forward" not in patched
    assert "self._forward_hook_installed" not in patched
    assert "if not outer.control_params:" in patched


def test_controlnet_wrapper_calls_bound_baseline_once_and_restores(tmp_path):
    hook, _ = patched_hook(tmp_path)
    calls = []

    def baseline_forward(x, timesteps=None, context=None, y=None, **kwargs):
        calls.append((x, timesteps, context, y, kwargs))
        return x

    model = _FakeUNet(baseline_forward)
    owner = hook.UnetHook()
    owner.hook(model, type("SD", (), {"is_sdxl": False})(), [], type("P", (), {"sample": lambda self, *a, **k: None})())
    wrapper = model.forward
    assert wrapper("x", timesteps="t", context="c") == "x"
    assert calls == [("x", "t", "c", None, {})]
    assert model._controlnet_forward_hook_baseline is baseline_forward
    assert model._controlnet_forward_hook_owner is owner._forward_hook_owner_token
    owner.restore()
    assert model.forward is baseline_forward
    assert not hasattr(model, "_controlnet_forward_hook_owner")
    assert owner.control_params is None


def test_controlnet_rehook_same_owner_is_idempotent_but_rejects_stale_owner(tmp_path):
    hook, _ = patched_hook(tmp_path)
    baseline = lambda x, timesteps=None, **kwargs: x
    model = _FakeUNet(baseline)
    sd = type("SD", (), {"is_sdxl": False})()
    process = type("P", (), {"sample": lambda self, *a, **k: None})()
    first = hook.UnetHook()
    first.hook(model, sd, [], process)
    wrapper = model.forward
    first.hook(model, sd, [], process)
    assert _same_callable(model.forward, wrapper)
    second = hook.UnetHook()
    with pytest.raises(RuntimeError, match="another live hook"):
        second.hook(model, sd, [], process)
    first.restore()
    second.hook(model, sd, [], process)
    second.restore()
    assert model.forward is baseline


def test_restore_never_clobbers_foreign_forward_or_owner_metadata(tmp_path):
    hook, _ = patched_hook(tmp_path)
    model = _FakeUNet(lambda x, **kwargs: x)
    owner = hook.UnetHook()
    owner.hook(model, type("SD", (), {"is_sdxl": False})(), [], type("P", (), {"sample": lambda self, *a, **k: None})())
    foreign = lambda x, **kwargs: "foreign"
    model.forward = foreign
    owner.restore()
    assert model.forward is foreign
    assert hasattr(model, "_controlnet_forward_hook_owner")
    # A later request fails closed rather than binding through stale ownership.
    with pytest.raises(RuntimeError, match="another live hook"):
        hook.UnetHook().hook(model, type("SD", (), {"is_sdxl": False})(), [], type("P", (), {"sample": lambda self, *a, **k: None})())

def _same_callable(left, right):
    if left is right:
        return True
    sentinel = object()
    return (
        getattr(left, "__func__", sentinel) is getattr(right, "__func__", sentinel)
        and getattr(left, "__self__", sentinel) is getattr(right, "__self__", sentinel)
    )


class _FakeUNet:
    def __init__(self, forward):
        self.forward = forward

    def children(self):
        return []


def _load_patched_hook(source, tmp_path):
    import importlib.util

    path = tmp_path / "patched_hook.py"
    path.write_text(source, encoding="utf-8")
    _install_hook_import_stubs()
    spec = importlib.util.spec_from_file_location("controlnet_patched_hook_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _install_hook_import_stubs():
    import sys
    import types

    import torch

    scripts_pkg = types.ModuleType("scripts")
    scripts_pkg.__path__ = []
    sys.modules["scripts"] = scripts_pkg

    logging_mod = types.ModuleType("scripts.logging")
    logging_mod.logger = types.SimpleNamespace(debug=lambda *a, **k: None, info=lambda *a, **k: None, error=lambda *a, **k: None)
    sys.modules["scripts.logging"] = logging_mod

    enums_mod = types.ModuleType("scripts.enums")
    enums_mod.ControlModelType = types.SimpleNamespace()
    enums_mod.AutoMachine = types.SimpleNamespace(Read="Read", Write="Write", StyleAlign="StyleAlign")
    enums_mod.HiResFixOption = types.SimpleNamespace(BOTH="BOTH")
    enums_mod.ControlNetUnionControlType = types.SimpleNamespace()
    sys.modules["scripts.enums"] = enums_mod

    ipadapter_pkg = types.ModuleType("scripts.ipadapter")
    ipadapter_model = types.ModuleType("scripts.ipadapter.ipadapter_model")
    ipadapter_model.ImageEmbed = object
    sys.modules["scripts.ipadapter"] = ipadapter_pkg
    sys.modules["scripts.ipadapter.ipadapter_model"] = ipadapter_model

    sparsectrl_mod = types.ModuleType("scripts.controlnet_sparsectrl")
    sparsectrl_mod.SparseCtrl = object
    sys.modules["scripts.controlnet_sparsectrl"] = sparsectrl_mod

    modules_pkg = types.ModuleType("modules")
    modules_pkg.__path__ = []
    sys.modules["modules"] = modules_pkg

    devices_mod = types.ModuleType("modules.devices")
    devices_mod.dtype_vae = torch.float32
    devices_mod.dtype_unet = torch.float32
    devices_mod.device = "cpu"
    devices_mod.autocast = lambda: types.SimpleNamespace(__enter__=lambda self: None, __exit__=lambda self, *exc: False)
    devices_mod.get_device_for = lambda _name: "cpu"
    devices_mod.cond_cast_unet = lambda x: x
    sys.modules["modules.devices"] = devices_mod

    lowvram_mod = types.ModuleType("modules.lowvram")
    lowvram_mod.send_everything_to_cpu = lambda: None
    sys.modules["modules.lowvram"] = lowvram_mod

    shared_mod = types.ModuleType("modules.shared")
    shared_mod.cmd_opts = types.SimpleNamespace(lowvram=False, medvram=False, medvram_sdxl=False)
    sys.modules["modules.shared"] = shared_mod

    callbacks = []
    scripts_mod = types.ModuleType("modules.scripts")
    scripts_mod.script_callbacks = types.SimpleNamespace(
        on_cfg_denoiser=lambda fn: callbacks.append(fn),
        remove_callbacks_for_function=lambda fn: callbacks.remove(fn) if fn in callbacks else None,
    )
    sys.modules["modules.scripts"] = scripts_mod

    prompt_parser_mod = types.ModuleType("modules.prompt_parser")
    prompt_parser_mod.MulticondLearnedConditioning = type("MulticondLearnedConditioning", (), {})
    prompt_parser_mod.ComposableScheduledPromptConditioning = type("ComposableScheduledPromptConditioning", (), {})
    prompt_parser_mod.ScheduledPromptConditioning = type("ScheduledPromptConditioning", (), {})
    sys.modules["modules.prompt_parser"] = prompt_parser_mod

    processing_mod = types.ModuleType("modules.processing")
    processing_mod.StableDiffusionProcessing = object
    sys.modules["modules.processing"] = processing_mod

    ldm_pkg = types.ModuleType("ldm")
    ldm_pkg.__path__ = []
    ldm_modules_pkg = types.ModuleType("ldm.modules")
    ldm_modules_pkg.__path__ = []
    ldm_diff_pkg = types.ModuleType("ldm.modules.diffusionmodules")
    ldm_diff_pkg.__path__ = []
    util_mod = types.ModuleType("ldm.modules.diffusionmodules.util")
    util_mod.timestep_embedding = lambda *args, **kwargs: None
    util_mod.make_beta_schedule = lambda *args, **kwargs: []
    openaimodel_mod = types.ModuleType("ldm.modules.diffusionmodules.openaimodel")
    openaimodel_mod.UNetModel = type("UNetModel", (), {})
    attention_mod = types.ModuleType("ldm.modules.attention")
    attention_mod.BasicTransformerBlock = type("BasicTransformerBlock", (), {})
    ldm_models_pkg = types.ModuleType("ldm.models")
    ldm_models_pkg.__path__ = []
    ldm_models_diff_pkg = types.ModuleType("ldm.models.diffusion")
    ldm_models_diff_pkg.__path__ = []
    ddpm_mod = types.ModuleType("ldm.models.diffusion.ddpm")
    ddpm_mod.extract_into_tensor = lambda *args, **kwargs: None
    sys.modules.update({
        "ldm": ldm_pkg,
        "ldm.modules": ldm_modules_pkg,
        "ldm.modules.diffusionmodules": ldm_diff_pkg,
        "ldm.modules.diffusionmodules.util": util_mod,
        "ldm.modules.diffusionmodules.openaimodel": openaimodel_mod,
        "ldm.modules.attention": attention_mod,
        "ldm.models": ldm_models_pkg,
        "ldm.models.diffusion": ldm_models_diff_pkg,
        "ldm.models.diffusion.ddpm": ddpm_mod,
    })

    sgm_pkg = types.ModuleType("sgm")
    sgm_pkg.__path__ = []
    sgm_modules_pkg = types.ModuleType("sgm.modules")
    sgm_modules_pkg.__path__ = []
    sgm_attention_mod = types.ModuleType("sgm.modules.attention")
    sgm_attention_mod.BasicTransformerBlock = attention_mod.BasicTransformerBlock
    sys.modules["sgm"] = sgm_pkg
    sys.modules["sgm.modules"] = sgm_modules_pkg
    sys.modules["sgm.modules.attention"] = sgm_attention_mod
