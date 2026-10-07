import sys
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


def _conds(pp, seed):
    gen = torch.Generator().manual_seed(seed)
    schedules = [
        pp.ScheduledPromptConditioning(end_at_step=4, cond=torch.randn(77, 8, generator=gen)),
        pp.ScheduledPromptConditioning(end_at_step=20, cond=torch.randn(77, 8, generator=gen)),
    ]
    c = pp.MulticondLearnedConditioning(shape=(1,), batch=[[pp.ComposableScheduledPromptConditioning(schedules, weight=0.7)]])
    sdxl_uc = [[pp.ScheduledPromptConditioning(end_at_step=20, cond={"crossattn": torch.randn(77, 8, generator=gen), "vector": torch.randn(16, generator=gen)})]]
    return c, sdxl_uc


def _snapshot(c, uc):
    leaves = list(c.batch[0][0].schedules) + uc[0]
    return (c.batch, c.batch[0], c.batch[0][0], c.batch[0][0].schedules, uc, uc[0], leaves,
            [leaf.cond if torch.is_tensor(leaf.cond) else leaf.cond["crossattn"] for leaf in leaves],
            [(leaf.cond if torch.is_tensor(leaf.cond) else leaf.cond["crossattn"]).clone() for leaf in leaves])


def _assert_unchanged(c, uc, snapshot):
    batch, row, composable, schedules, uc_list, uc_row, leaves, tensors, values = snapshot
    assert c.batch is batch and c.batch[0] is row and c.batch[0][0] is composable
    assert composable.schedules is schedules and uc is uc_list and uc[0] is uc_row
    current = list(schedules) + uc[0]
    assert len(current) == len(leaves) and all(x is y for x, y in zip(current, leaves))
    for leaf, tensor, value in zip(leaves, tensors, values):
        cond = leaf.cond if torch.is_tensor(leaf.cond) else leaf.cond["crossattn"]
        assert cond is tensor and torch.equal(cond, value) and cond.shape[0] == 77


def test_mark_prompt_context_returns_marked_copies_without_mutating_cache_entries(tmp_path):
    hook = load_hook(tmp_path)
    pp = hook.stubs["modules.prompt_parser"]
    c, uc = _conds(pp, 0)
    snapshot = _snapshot(c, uc)

    marked_c = hook.mark_prompt_context(c, positive=True)
    marked_uc = hook.mark_prompt_context(uc, positive=False)
    _assert_unchanged(c, uc, snapshot)

    assert type(marked_c) is pp.MulticondLearnedConditioning and marked_c is not c
    assert marked_c.shape == c.shape
    composable = marked_c.batch[0][0]
    assert type(composable) is pp.ComposableScheduledPromptConditioning and composable.weight == 0.7
    for marked, original in zip(composable.schedules, c.batch[0][0].schedules):
        assert marked.end_at_step == original.end_at_step
        assert torch.equal(marked.cond[0], torch.full((8,), 1024.0)) and torch.equal(marked.cond[1:], original.cond)
    leaf = marked_uc[0][0]
    assert torch.equal(leaf.cond["crossattn"][0], torch.full((8,), -1024.0))
    assert torch.equal(leaf.cond["crossattn"][1:], uc[0][0].cond["crossattn"])
    assert leaf.cond["vector"] is uc[0][0].cond["vector"]
    # Marking a marked copy is a no-op on its leaves.
    again = hook.mark_prompt_context(marked_c, positive=True)
    assert again.batch[0][0].schedules[0] is composable.schedules[0]


def test_controlnet_sample_marks_copies_and_keeps_cond_cache_entries_reusable(tmp_path):
    """The persistent cond cache returns the same objects to later requests,
    with or without ControlNet: marking must never leak into them (this also
    covers cached_hr_c/hr_uc, which unmark's old per-forward reset missed)."""
    hook = load_hook(tmp_path)
    pp = hook.stubs["modules.prompt_parser"]
    cached_c, cached_uc = _conds(pp, 1)
    cached_hr_c, cached_hr_uc = _conds(pp, 2)
    snapshots = _snapshot(cached_c, cached_uc), _snapshot(cached_hr_c, cached_hr_uc)
    seen = {}

    class Txt2Img:
        hr_c, hr_uc = cached_hr_c, cached_hr_uc

        def sample(self, conditioning, unconditional_conditioning, **kwargs):
            seen.update(c=conditioning, uc=unconditional_conditioning, hr_c=self.hr_c, hr_uc=self.hr_uc)

    process = Txt2Img()
    owner = hook.UnetHook()
    owner.hook(_FakeUNet(lambda x, **kwargs: x), _SD, [], process)
    process.sample(conditioning=cached_c, unconditional_conditioning=cached_uc, seeds=[1])
    owner.restore()

    _assert_unchanged(cached_c, cached_uc, snapshots[0])
    _assert_unchanged(cached_hr_c, cached_hr_uc, snapshots[1])
    for key, positive in (("c", True), ("uc", False), ("hr_c", True), ("hr_uc", False)):
        marked = seen[key]
        leaf = marked.batch[0][0].schedules[0] if positive else marked[0][0]
        cond = leaf.cond if torch.is_tensor(leaf.cond) else leaf.cond["crossattn"]
        assert cond.shape[0] == 78 and float(cond[0, 0]) == (1024.0 if positive else -1024.0)
    assert process.hr_c is seen["hr_c"] and process.hr_uc is seen["hr_uc"]

    # img2img processing has no hires conds: nothing is added to it.
    img2img = _Process(lambda: None)
    hook.UnetHook().hook(_FakeUNet(lambda x, **kwargs: x), _SD, [], img2img)
    img2img.sample(conditioning=[], unconditional_conditioning=[])
    assert not hasattr(img2img, "hr_c") and not hasattr(img2img, "hr_uc")
    assert "cached_c" not in SOURCE.read_text(encoding="utf-8")


def _two_sync_unmark_oracle(x):
    """unmark_prompt_context as it was before the single-transfer change."""
    t = x[..., 0, :]
    m = torch.mean(torch.abs(torch.abs(t) - 1024)).detach().cpu().float().numpy()
    if not float(m) < 1e-3:
        return torch.ones(size=(x.shape[0], 1, 1, 1), dtype=x.dtype, device=x.device), [], [], x
    mark = x[:, 0, :]
    context = x[:, 1:, :]
    mark = (torch.mean(torch.abs(mark - (-1024)), dim=1) > 1e-3).float()
    mark_batch = mark[:, None, None, None].to(x.dtype).to(x.device)
    mark = mark.detach().cpu().numpy().tolist()
    return mark_batch, [i for i, v in enumerate(mark) if v < 0.5], [i for i, v in enumerate(mark) if not v < 0.5], context


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_unmark_matches_two_sync_oracle_with_one_device_to_host_copy(tmp_path, monkeypatch, dtype):
    hook = load_hook(tmp_path)
    gen = torch.Generator().manual_seed(3)

    def context(marks):
        body = torch.randn(len(marks), 77, 8, generator=gen) * 3
        first = torch.stack([torch.full((8,), 1024.0 * m) if m else torch.randn(8, generator=gen) for m in marks])
        return torch.cat([first[:, None, :], body], dim=1).to(dtype)

    cases = [context([1, -1]), context([1, 1, -1, -1]), context([-1, 1, -1]), context([-1]),
             context([0, 0]), context([1, 0]), torch.randn(2, 77, 8, generator=gen).to(dtype)]
    transfers = []
    real_cpu = torch.Tensor.cpu
    for x in cases:
        expected = _two_sync_unmark_oracle(x)
        transfers.clear()
        monkeypatch.setattr(torch.Tensor, "cpu", lambda self, *a, **k: transfers.append(1) or real_cpu(self, *a, **k))
        got = hook.unmark_prompt_context(x)
        monkeypatch.setattr(torch.Tensor, "cpu", real_cpu)
        assert len(transfers) == 1
        assert got[0].dtype == expected[0].dtype and torch.equal(got[0], expected[0])
        assert got[1] == expected[1] and got[2] == expected[2]
        assert torch.equal(got[3], expected[3])


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
    """Stand-in ControlParams: any attribute read beyond what hook()/restore()
    inspect means the wrapper entered ControlNet's forward path."""
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
    # The stubs exist only for this import (the loaded module keeps references to them); afterwards every
    # stubbed package goes back to what it was so later test files see the real sgm/ldm/modules/scripts.
    stubbed = {"scripts", "modules", "ldm", "sgm"}
    saved = {key: value for key, value in sys.modules.items() if key.split(".")[0] in stubbed}
    try:
        _install_hook_import_stubs()
        spec = importlib.util.spec_from_file_location("controlnet_hook_under_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.stubs = {key: value for key, value in sys.modules.items() if key.split(".")[0] in stubbed}
    finally:
        for key in [key for key in sys.modules if key.split(".")[0] in stubbed]:
            del sys.modules[key]
        sys.modules.update(saved)
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
    plugable_ipadapter = types.ModuleType("scripts.ipadapter.plugable_ipadapter")
    plugable_ipadapter.clear_all_ip_adapter = lambda: None
    sys.modules["scripts.ipadapter"] = ipadapter_pkg
    sys.modules["scripts.ipadapter.ipadapter_model"] = ipadapter_model
    sys.modules["scripts.ipadapter.plugable_ipadapter"] = plugable_ipadapter

    lllite_mod = types.ModuleType("scripts.controlnet_lllite")
    lllite_mod.clear_all_lllite = lambda: None
    sys.modules["scripts.controlnet_lllite"] = lllite_mod

    sparsectrl_mod = types.ModuleType("scripts.controlnet_sparsectrl")
    sparsectrl_mod.SparseCtrl = object
    sys.modules["scripts.controlnet_sparsectrl"] = sparsectrl_mod

    modules_pkg = types.ModuleType("modules")
    modules_pkg.__path__ = []
    sys.modules["modules"] = modules_pkg

    import importlib.util

    row_memo_spec = importlib.util.spec_from_file_location("modules.sd_unet_row_memo", ROOT / "modules" / "sd_unet_row_memo.py")
    row_memo_mod = importlib.util.module_from_spec(row_memo_spec)
    row_memo_spec.loader.exec_module(row_memo_mod)
    modules_pkg.sd_unet_row_memo = row_memo_mod
    sys.modules["modules.sd_unet_row_memo"] = row_memo_mod

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

    # The real conditioning containers (prompt_parser only needs lark).
    import importlib.util
    spec = importlib.util.spec_from_file_location("modules.prompt_parser", ROOT / "modules" / "prompt_parser.py")
    prompt_parser_mod = importlib.util.module_from_spec(spec)
    sys.modules["modules.prompt_parser"] = prompt_parser_mod
    spec.loader.exec_module(prompt_parser_mod)

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
