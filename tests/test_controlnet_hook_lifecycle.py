from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).parents[1]
SOURCE = ROOT / "extensions" / "sd-webui-controlnet" / "scripts" / "hook.py"


def load_hook(tmp_path):
    source = SOURCE.read_text(encoding="utf-8")
    return _load_hook_module(source, tmp_path)


def test_controlnet_hook_source_carries_owner_aware_lifecycle():
    source = SOURCE.read_text(encoding="utf-8")
    assert source.count("OPENCLAW_CONTROLNET_FORWARD_OWNER_V2") == 1
    assert "model._original_forward = model.forward" not in source
    assert "self._forward_hook_installed" not in source
    assert "if not outer.control_params:" in source
    for fragment in (
        'getattr(model, "_controlnet_forward_hook_owner", None) is self._forward_hook_owner_token',
        "model._controlnet_forward_hook_baseline",
        "model._controlnet_forward_hook_wrapper",
        "self._forward_hook_wrapper = None",
        "self.control_params = None",
    ):
        assert fragment in source


def test_controlnet_wrapper_calls_bound_baseline_once_and_restores(tmp_path):
    hook = load_hook(tmp_path)
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
    hook = load_hook(tmp_path)
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
    hook = load_hook(tmp_path)
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


def _leak_hook_through_failed_generation(hook, model, context):
    """img2img Script instance hooks the UNet; sampling raises (NaN check), so
    Script.postprocess, the normal restore point, never runs."""
    def failing_sample():
        # Inside p.sample the wrapper runs ControlNet's forward ...
        with pytest.raises(_ReachedControlNetForward, match="used_hint_cond"):
            model.forward(_LATENT, timesteps="t", context=context)
        raise RuntimeError("NansException: A tensor with NaNs was produced in Unet.")

    process = _Process(failing_sample)
    leaked = hook.UnetHook()
    leaked.hook(model, _SD, [_ProbeParam()], process)
    with pytest.raises(RuntimeError, match="NansException"):
        process.sample(conditioning=[], unconditional_conditioning=[])
    return leaked


def test_leaked_hook_after_failed_generation_is_inert_then_healed_by_next_request(tmp_path):
    hook = load_hook(tmp_path)
    callbacks = hook.scripts.script_callbacks.callbacks
    baseline_calls = []

    def baseline(x, timesteps=None, context=None, y=None, **kwargs):
        baseline_calls.append(x)
        return x

    model = _FakeUNet(baseline)
    context = torch.zeros(2, 3, 4)
    leaked = _leak_hook_through_failed_generation(hook, model, context)
    assert not leaked.sampling_active
    assert model._controlnet_forward_hook_owner is leaked._forward_hook_owner_token
    assert callbacks == [leaked.guidance_schedule_handler]

    # ... while the leaked hook is still installed, a later UNet call outside its
    # sampling goes straight to the baseline: no stale hint/control is applied.
    assert model.forward("later", timesteps="t", context=context) == "later"
    assert baseline_calls == ["later"]

    # Next request without ControlNet units, on the other (txt2img) Script
    # instance whose latest_network is None: controlnet_main_entry heals the UNet.
    hook.UnetHook.restore_leaked(model)
    assert model.forward is baseline
    for attr in ("_controlnet_forward_hook_owner", "_controlnet_forward_hook_wrapper",
                 "_controlnet_forward_hook_baseline", "_controlnet_forward_hook_restore"):
        assert not hasattr(model, attr)
    assert callbacks == []
    assert leaked.control_params is None
    hook.UnetHook.restore_leaked(model)  # idempotent once healed
    assert model.forward is baseline

    # Next request with ControlNet hooks cleanly and is applied inside p.sample.
    _assert_fresh_controlnet_request_works(hook, model, context, callbacks)
    assert model.forward is baseline
    assert callbacks == []


def test_controlnet_request_directly_after_failed_generation_hooks_cleanly(tmp_path):
    hook = load_hook(tmp_path)
    callbacks = hook.scripts.script_callbacks.callbacks
    baseline = lambda x, **kwargs: x
    model = _FakeUNet(baseline)
    context = torch.zeros(2, 3, 4)
    _leak_hook_through_failed_generation(hook, model, context)
    # Without healing first the new owner fails closed ...
    with pytest.raises(RuntimeError, match="another live hook"):
        hook.UnetHook().hook(model, _SD, [_ProbeParam()], _Process(lambda: None))
    # ... so controlnet_main_entry restores the leaked hook before hooking.
    hook.UnetHook.restore_leaked(model)
    _assert_fresh_controlnet_request_works(hook, model, context, callbacks)
    assert model.forward is baseline
    assert callbacks == []


def test_sampling_flag_is_reentrant_and_cleared_after_sampling(tmp_path):
    hook = load_hook(tmp_path)
    model = _FakeUNet(lambda x, **kwargs: x)
    owner = hook.UnetHook()
    states = []

    def body():
        states.append(owner.sampling_active)
        if len(states) == 1:
            process.sample()  # nested call through the ControlNet sample wrapper
            states.append(owner.sampling_active)

    process = _Process(body)
    owner.hook(model, _SD, [], process)
    assert owner.sampling_active is False
    process.sample()
    assert states == [True, True, True]
    assert owner.sampling_active is False
    owner.restore()


def _assert_fresh_controlnet_request_works(hook, model, context, callbacks):
    def sample():
        with pytest.raises(_ReachedControlNetForward, match="used_hint_cond"):
            model.forward(_LATENT, timesteps="t", context=context)

    process = _Process(sample)
    fresh = hook.UnetHook()
    fresh.hook(model, _SD, [_ProbeParam()], process)
    assert model._controlnet_forward_hook_restore == fresh.restore
    assert callbacks == [fresh.guidance_schedule_handler]
    process.sample(conditioning=[], unconditional_conditioning=[])
    fresh.restore()


class _ReachedControlNetForward(Exception):
    pass


class _ProbeParam:
    """Stand-in ControlParams: any attribute read beyond what hook() inspects
    means the wrapper entered ControlNet's forward path."""
    control_model_type = None
    control_model = None

    def __getattr__(self, name):
        raise _ReachedControlNetForward(name)


class _Process:
    def __init__(self, body):
        self._body = body

    def sample(self, *args, **kwargs):
        return self._body()


_SD = type("SD", (), {"is_sdxl": False})()
_LATENT = torch.zeros(2, 4, 8, 8)


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


def _load_hook_module(source, tmp_path):
    import importlib.util

    path = tmp_path / "hook_under_test.py"
    path.write_text(source, encoding="utf-8")
    _install_hook_import_stubs()
    spec = importlib.util.spec_from_file_location("controlnet_hook_under_test", path)
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
    logging_mod.logger = types.SimpleNamespace(debug=lambda *a, **k: None, info=lambda *a, **k: None, warning=lambda *a, **k: None, error=lambda *a, **k: None)
    sys.modules["scripts.logging"] = logging_mod

    enums_mod = types.ModuleType("scripts.enums")
    enums_mod.ControlModelType = types.SimpleNamespace(AttentionInjection="AttentionInjection")
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
        callbacks=callbacks,
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
