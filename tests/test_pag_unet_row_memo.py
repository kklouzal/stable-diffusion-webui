"""CPU differential tests for PAG's main-pass row memo with a tiny random SDXL-shaped UNet.

The real CFGDenoiser drives the main pass, the real Incantations PAG script runs its hidden pass, the real
sgm UNetModel computes both, and optionally the real ControlNet hook with a tiny real cldm.ControlNet sits
in front of it. The oracle replays every logged main-pass UNet call whole with the PAG perturbation
enabled, which is what PAG computed before it was limited to cond rows.
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SGM_ROOT = ROOT / "repositories" / "generative-models"
LDM_ROOT = ROOT / "repositories" / "stable-diffusion-stability-ai"
INC_SCRIPTS = ROOT / "extensions" / "sd-webui-incantations" / "scripts"
CN_SCRIPTS = ROOT / "extensions" / "sd-webui-controlnet" / "scripts"

pytestmark = pytest.mark.skipif(
    not (SGM_ROOT / "sgm").is_dir() or not (LDM_ROOT / "ldm").is_dir(),
    reason="generative-models / stable-diffusion repositories are not checked out",
)

_REAL_MODULES = {}


def _module(name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _package(name, path=(), **attrs):
    module = _module(name, **attrs)
    module.__path__ = [str(p) for p in path]
    return module


@contextlib.contextmanager
def isolated_modules(replacements, prefixes):
    """Import with ``replacements`` in sys.modules and other modules under ``prefixes`` hidden.

    Afterwards every module under ``prefixes`` is dropped and the hidden ones come back. Modules outside
    ``prefixes`` imported meanwhile (lazy torch internals) stay: re-importing those is not safe.
    """
    assert all(key.split(".")[0] in prefixes for key in replacements)
    hidden = {key: value for key, value in sys.modules.items() if key.split(".")[0] in prefixes}
    for key in hidden:
        del sys.modules[key]
    sys.modules.update(replacements)
    try:
        yield
    finally:
        for key in [key for key in sys.modules if key.split(".")[0] in prefixes]:
            del sys.modules[key]
        sys.modules.update(hidden)


def real_modules():
    """Real sgm (kept installed: its modules import lazily at call time) and ldm-util modules."""
    if not _REAL_MODULES:
        for key in [key for key in sys.modules if key.split(".")[0] == "sgm"]:
            del sys.modules[key]
        sys.path.insert(0, str(SGM_ROOT))
        try:
            importlib.import_module("sgm.modules.diffusionmodules.openaimodel")
            importlib.import_module("sgm.modules.diffusionmodules.video_model")
            importlib.import_module("sgm.modules.attention")
        finally:
            sys.path.remove(str(SGM_ROOT))
        _REAL_MODULES.update({key: value for key, value in sys.modules.items() if key.split(".")[0] == "sgm"})
        with isolated_modules({}, {"ldm"}):
            sys.path.insert(0, str(LDM_ROOT))
            try:
                _REAL_MODULES["ldm.modules.diffusionmodules.util"] = importlib.import_module("ldm.modules.diffusionmodules.util")
            finally:
                sys.path.remove(str(LDM_ROOT))
    # Other tests replace sgm with stubs; put the real package back for this one.
    for key in [key for key in sys.modules if key.split(".")[0] == "sgm"]:
        del sys.modules[key]
    sys.modules.update({key: value for key, value in _REAL_MODULES.items() if key.split(".")[0] == "sgm"})
    return _REAL_MODULES


class Callbacks:
    def __init__(self):
        self.denoiser = []
        self.denoised = []

    def module(self):
        class CFGDenoiserParams:
            def __init__(self, x, image_cond, sigma, sampling_step, total_sampling_steps, text_cond, text_uncond, denoiser=None):
                self.x = x
                self.image_cond = image_cond
                self.sigma = sigma
                self.sampling_step = sampling_step
                self.total_sampling_steps = total_sampling_steps
                self.text_cond = text_cond
                self.text_uncond = text_uncond
                self.denoiser = denoiser

        class CFGDenoisedParams:
            def __init__(self, x, sampling_step, total_sampling_steps, inner_model):
                self.x = x
                self.sampling_step = sampling_step
                self.total_sampling_steps = total_sampling_steps
                self.inner_model = inner_model

        class AfterCFGCallbackParams:
            def __init__(self, x, sampling_step, total_sampling_steps):
                self.x = x
                self.sampling_step = sampling_step
                self.total_sampling_steps = total_sampling_steps

        def run(callbacks):
            def dispatch(params):
                for callback in list(callbacks):
                    callback(params)
            return dispatch

        def remove(fn):
            self.denoiser[:] = [c for c in self.denoiser if c != fn]
            self.denoised[:] = [c for c in self.denoised if c != fn]

        return _module(
            "modules.script_callbacks",
            CFGDenoiserParams=CFGDenoiserParams,
            CFGDenoisedParams=CFGDenoisedParams,
            AfterCFGCallbackParams=AfterCFGCallbackParams,
            cfg_denoiser_callback=run(self.denoiser),
            cfg_denoised_callback=run(self.denoised),
            cfg_after_cfg_callback=lambda params: None,
            on_cfg_denoiser=self.denoiser.append,
            on_cfg_denoised=self.denoised.append,
            on_before_ui=lambda fn: None,
            remove_callbacks_for_function=remove,
        )


def randomize(module, seed, std):
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for param in module.parameters():
            param.copy_(torch.randn(param.shape, generator=gen) * std)


# SDXL block layout (9 input blocks, 9 output blocks, attention from the second level) at toy width.
UNET_CONFIG = dict(
    in_channels=4,
    model_channels=32,
    out_channels=4,
    num_res_blocks=2,
    attention_resolutions=[2, 4],
    channel_mult=[1, 2, 2],
    num_head_channels=16,
    use_linear_in_transformer=True,
    transformer_depth=[0, 1, 1],
    context_dim=16,
    num_classes="sequential",
    adm_in_channels=12,
    spatial_transformer_attn_type="softmax",
)
CONTROLNET_CONFIG = dict(
    in_channels=4,
    model_channels=32,
    hint_channels=3,
    num_res_blocks=2,
    attention_resolutions=[2, 4],
    channel_mult=[1, 2, 2],
    num_head_channels=16,
    use_linear_in_transformer=True,
    transformer_depth=[0, 1, 1],
    context_dim=16,
    num_classes="sequential",
    adm_in_channels=12,
    use_fp16=False,
)


class Harness:
    """Real CFGDenoiser + real PAG script over a tiny sgm UNet, with A1111 globals stubbed."""

    def __init__(self):
        real = real_modules()
        self.openaimodel = real["sgm.modules.diffusionmodules.openaimodel"]
        self.unet = self.openaimodel.UNetModel(**UNET_CONFIG).eval()
        randomize(self.unet, seed=0, std=0.08)

        self.callbacks = Callbacks()
        self.unet_calls = []
        self.opts = types.SimpleNamespace(
            skip_early_cond=0.0,
            s_min_uncond_all=False,
            pad_cond_uncond_v0=False,
            pad_cond_uncond=False,
            batch_cond_uncond=True,
            live_preview_content="Prompt",
        )
        mapping = {}
        for name, module in self.unet.named_modules():
            if "CrossAttention" in module.__class__.__name__:
                module.network_layer_name = name.replace(".", "_")
                mapping[name] = module
        self.sd_model = types.SimpleNamespace(
            cond_stage_key="crossattn",
            model=types.SimpleNamespace(conditioning_key="crossattn", diffusion_model=self.unet),
            network_layer_mapping=mapping,
            cond_stage_model_empty_prompt=torch.zeros(1, 4, 16),
        )
        state = types.SimpleNamespace(interrupted=False, skipped=False, sampling_step=0, sampling_steps=4)
        self.shared = _module(
            "modules.shared",
            opts=self.opts,
            state=state,
            sd_model=self.sd_model,
            cmd_opts=types.SimpleNamespace(lowvram=False, medvram=False, medvram_sdxl=False),
        )
        self.script_callbacks = self.callbacks.module()

        spec = importlib.util.spec_from_file_location("modules.sd_unet_row_memo", ROOT / "modules" / "sd_unet_row_memo.py")
        self.row_memo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.row_memo)

        self.modules = {
            "modules": _package("modules", sd_unet_row_memo=self.row_memo, shared=self.shared, script_callbacks=self.script_callbacks),
            "modules.prompt_parser": _module(
                "modules.prompt_parser",
                reconstruct_multicond_batch=lambda cond, step: cond,
                reconstruct_cond_batch=lambda uncond, step: uncond,
                MulticondLearnedConditioning=type("MulticondLearnedConditioning", (), {}),
                ComposableScheduledPromptConditioning=type("ComposableScheduledPromptConditioning", (), {}),
                ScheduledPromptConditioning=type("ScheduledPromptConditioning", (), {}),
            ),
            "modules.sd_samplers_common": _module(
                "modules.sd_samplers_common",
                InterruptedException=Exception,
                apply_refiner=lambda denoiser, sigma: False,
                store_latent=lambda latent: None,
            ),
            "modules.shared": self.shared,
            "modules.script_callbacks": self.script_callbacks,
            "modules.sd_unet_row_memo": self.row_memo,
            "modules.processing": _module("modules.processing", StableDiffusionProcessing=type("StableDiffusionProcessing", (), {})),
        }
        with isolated_modules(self.modules, {"modules"}):
            spec = importlib.util.spec_from_file_location("cfg_denoiser_under_test", ROOT / "modules" / "sd_samplers_cfg_denoiser.py")
            cfg_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cfg_module)

        pag_modules = dict(self.modules)
        pag_modules.update({
            "scripts": _package("scripts", [INC_SCRIPTS]),
            "modules.scripts": _module("modules.scripts", Script=object, AlwaysVisible=object()),
            "modules.headless_ui": _module("modules.headless_ui"),
        })
        with isolated_modules(pag_modules, {"modules", "scripts"}):
            self.pag = importlib.import_module("scripts.pag")

        harness = self

        class InnerModel:
            """k-diffusion style eps wrapper around the UNet."""

            def __call__(self, x, sigma, cond):
                harness.unet_calls.append((x, sigma, cond, harness.pag_enabled()))
                c_in = (1.0 / (sigma ** 2 + 1.0) ** 0.5)[:, None, None, None]
                eps = harness.unet(x * c_in, timesteps=sigma * 100.0, context=cond["crossattn"], y=cond["vector"])
                return x - eps * sigma[:, None, None, None]

        self.inner = InnerModel()

        class Denoiser(cfg_module.CFGDenoiser):
            @property
            def inner_model(self):
                return harness.inner

            def run_inner_model(self, x, sigma, cond):
                # CFGDenoiser.run_inner_model minus the CUDA-graph dispatcher.
                return self.inner_model(x, sigma, cond=cond)

        sampler = types.SimpleNamespace(sampler_extra_args={}, last_latent=None)
        self.denoiser = Denoiser(sampler)
        self.denoiser.p = types.SimpleNamespace(extra_generation_params={}, scripts=None)
        self.denoiser.steps = 4
        self.denoiser.total_steps = 4
        self.script = None
        self.pag_params = None
        self.controlnet = None
        self.controlnet_calls = 0
        self.captured = {}

    def install_controlnet(self, weight=0.7, cfg_injection=False, global_average_pooling=False, seed=1):
        real = real_modules()
        logger = types.SimpleNamespace(debug=lambda *a, **k: None, info=lambda *a, **k: None, warning=lambda *a, **k: None, error=lambda *a, **k: None)
        ldm_util = real["ldm.modules.diffusionmodules.util"]

        def extract_into_tensor(a, t, x_shape):
            b, *_ = t.shape
            out = a.gather(-1, t)
            return out.reshape(b, *((1,) * (len(x_shape) - 1)))

        cn_modules = dict(self.modules)
        cn_modules.update({
            "scripts": _package("scripts", [CN_SCRIPTS]),
            "scripts.logging": _module("scripts.logging", logger=logger),
            "scripts.ipadapter": _package("scripts.ipadapter"),
            "scripts.ipadapter.ipadapter_model": _module("scripts.ipadapter.ipadapter_model", ImageEmbed=object),
            "scripts.controlnet_sparsectrl": _module("scripts.controlnet_sparsectrl", SparseCtrl=type("SparseCtrl", (), {})),
            "modules.devices": _module(
                "modules.devices",
                dtype_unet=torch.float32,
                dtype_vae=torch.float32,
                device=torch.device("cpu"),
                get_device_for=lambda name: torch.device("cpu"),
                cond_cast_unet=lambda x: x,
                autocast=contextlib.nullcontext,
            ),
            "modules.lowvram": _module("modules.lowvram", send_everything_to_cpu=lambda: None),
            "modules.scripts": _module("modules.scripts", script_callbacks=self.script_callbacks),
            "ldm": _package("ldm"),
            "ldm.modules": _package("ldm.modules"),
            "ldm.modules.diffusionmodules": _package("ldm.modules.diffusionmodules"),
            "ldm.modules.diffusionmodules.util": ldm_util,
            "ldm.modules.diffusionmodules.openaimodel": _module("ldm.modules.diffusionmodules.openaimodel", UNetModel=type("UNetModel", (), {})),
            "ldm.modules.attention": _module("ldm.modules.attention", BasicTransformerBlock=type("BasicTransformerBlock", (), {})),
            "ldm.models": _package("ldm.models"),
            "ldm.models.diffusion": _package("ldm.models.diffusion"),
            "ldm.models.diffusion.ddpm": _module("ldm.models.diffusion.ddpm", extract_into_tensor=extract_into_tensor),
        })
        for name in ("devices", "lowvram", "scripts"):
            setattr(cn_modules["modules"], name, cn_modules[f"modules.{name}"])
        with isolated_modules(cn_modules, {"modules", "scripts", "ldm"}):
            self.hook = importlib.import_module("scripts.hook")
            self.cldm = importlib.import_module("scripts.cldm")
            self.enums = importlib.import_module("scripts.enums")

        torch.manual_seed(seed)
        self.controlnet = self.cldm.PlugableControlModel(CONTROLNET_CONFIG).eval()
        randomize(self.controlnet, seed=seed, std=0.05)

        def count(module, args, kwargs):
            self.controlnet_calls += 1

        self.controlnet.control_model.register_forward_pre_hook(count, with_kwargs=True)
        gen = torch.Generator().manual_seed(seed + 1)
        self.control_param = self.hook.ControlParams(
            control_model=self.controlnet,
            preprocessor={"name": "depth_zoe", "threshold_a": 0.5, "threshold_b": 0.5},
            hint_cond=torch.rand(1, 3, 64, 64, generator=gen),
            weight=weight,
            guidance_stopped=False,
            start_guidance_percent=0.0,
            stop_guidance_percent=1.0,
            advanced_weighting=None,
            control_model_type=self.enums.ControlModelType.ControlNet,
            hr_hint_cond=None,
            global_average_pooling=global_average_pooling,
            soft_injection=False,
            cfg_injection=cfg_injection,
        )
        self.unet_hook = self.hook.UnetHook(lowvram=False)
        self.unet_hook.hook(
            model=self.unet,
            sd_ldm=types.SimpleNamespace(is_sdxl=True),
            control_params=[self.control_param],
            process=types.SimpleNamespace(sample=lambda *a, **k: None),
        )

    def pag_modules(self):
        return [m for m in self.sd_model.network_layer_mapping.values() if "middle_block_1_transformer_blocks_0_attn1" in m.network_layer_name]

    def pag_enabled(self):
        return any(getattr(m, "pag_enable", False) for m in self.pag_modules())

    def enable_pag(self, callbacks=True):
        self.script = self.pag.PAGExtensionScript()
        p = types.SimpleNamespace(incant_cfg_params={}, steps=4, cfg_scale=5.0, batch_size=1, extra_generation_params={})
        self.script.pag_process_batch(p, True, 3.0, 0, 150, False, "Constant", 0.0, 100.0, False)
        self.pag_params = p.incant_cfg_params["pag_params"]
        assert len(self.pag_params.crossattn_modules) == 1
        if not callbacks:
            self.script.remove_callbacks()

    def capture(self):
        self.callbacks.denoised.append(lambda params: self.captured.update(x_out=params.x.clone()))

    def run(self, x, sigma, conds_list, cond, uncond, image_cond, s_min_uncond=0.0):
        self.captured.clear()
        self.unet_calls.clear()
        with torch.no_grad():
            return self.denoiser(x, sigma, uncond, (conds_list, cond), 5.0, s_min_uncond, image_cond)

    @contextlib.contextmanager
    def pag_on(self):
        for module in self.pag_modules():
            module.pag_enable = True
        try:
            yield
        finally:
            for module in self.pag_modules():
                module.pag_enable = False


class DictWithShape(dict):
    """modules.prompt_parser.DictWithShape: SDXL conditioning with the cross-attention shape."""

    def __init__(self, x, shape=None):
        super().__init__()
        self.update(x)

    @property
    def shape(self):
        return self["crossattn"].shape


def make_inputs(batch_size, repeats, cond_tokens=4, uncond_tokens=4, seed=1, marked=False):
    gen = torch.Generator().manual_seed(seed)
    n_cond = sum(repeats)
    x = torch.randn(batch_size, 4, 8, 8, generator=gen)
    sigma = torch.full((batch_size,), 1.7)
    cond_ctx = torch.randn(n_cond, cond_tokens, 16, generator=gen)
    uncond_ctx = torch.randn(batch_size, uncond_tokens, 16, generator=gen)
    if marked:
        # ControlNet's mark_prompt_context: a leading +-1024 token row tells it cond from uncond.
        cond_ctx[:, 0, :] = 1024.0
        uncond_ctx[:, 0, :] = -1024.0
    cond = DictWithShape({"crossattn": cond_ctx, "vector": torch.randn(n_cond, 12, generator=gen)})
    uncond = DictWithShape({"crossattn": uncond_ctx, "vector": torch.randn(batch_size, 12, generator=gen)})
    conds_list = []
    index = 0
    for count in repeats:
        conds_list.append([(index + j, 1.0 / count) for j in range(count)])
        index += count
    image_cond = torch.zeros(batch_size, 1, 1, 1)
    return x, sigma, conds_list, cond, uncond, image_cond


def oracle_pag_cond_rows(harness, main_calls, n_cond):
    """Old PAG semantics: every main-pass call evaluated whole with PAG on; keep the cond rows."""
    outs = []
    with harness.pag_on(), torch.no_grad():
        for x, sigma, cond, _ in main_calls:
            outs.append(harness.inner(x, sigma, cond))
    return torch.cat(outs)[:n_cond]


LAYOUTS = {
    # name: (batch_size, repeats, cond_tokens, uncond_tokens, batch_cond_uncond)
    "batched": (1, [1], 4, 4, True),
    "batch_size_2": (2, [1, 1], 4, 4, True),
    "and_prompt_batched": (1, [2], 4, 4, True),
    "token_mismatch_split": (1, [2], 8, 4, True),
    "batching_disabled": (2, [1, 1], 4, 4, False),
    "and_prompt_batching_disabled": (2, [2, 1], 4, 4, False),
}


@pytest.mark.parametrize("controlnet", [False, True], ids=["unet", "controlnet"])
@pytest.mark.parametrize("layout", sorted(LAYOUTS))
def test_pag_cond_rows_match_old_full_pass_and_main_pass_is_unchanged(layout, controlnet):
    batch_size, repeats, cond_tokens, uncond_tokens, batch_cond_uncond = LAYOUTS[layout]
    inputs = make_inputs(batch_size, repeats, cond_tokens, uncond_tokens, marked=controlnet)
    n_cond = sum(repeats)

    baseline = Harness()
    baseline.opts.batch_cond_uncond = batch_cond_uncond
    if controlnet:
        baseline.install_controlnet()
    baseline.capture()
    baseline.run(*inputs)
    baseline_x_out = baseline.captured["x_out"]

    harness = Harness()
    harness.opts.batch_cond_uncond = batch_cond_uncond
    if controlnet:
        harness.install_controlnet()
    harness.enable_pag()
    harness.capture()
    harness.run(*inputs)

    main_calls = [call for call in harness.unet_calls if not call[3]]
    pag_calls = [call for call in harness.unet_calls if call[3]]
    # (a) the main pass is bitwise unchanged by recording.
    assert torch.equal(harness.captured["x_out"], baseline_x_out)
    # PAG mirrors the main pass's chunking, limited to cond rows, with the main call's own inputs.
    expected_rows = []
    start = 0
    for x, *_ in main_calls:
        rows = min(x.shape[0], n_cond - start)
        if rows > 0:
            expected_rows.append(rows)
        start += x.shape[0]
    assert [call[0].shape[0] for call in pag_calls] == expected_rows
    for (main_x, main_sigma, main_cond, _), (pag_x, pag_sigma, pag_cond, _) in zip(main_calls, pag_calls):
        rows = pag_x.shape[0]
        assert pag_x.data_ptr() == main_x.data_ptr() and torch.equal(pag_x, main_x[:rows])
        assert torch.equal(pag_sigma, main_sigma[:rows])
        assert pag_cond["crossattn"].data_ptr() == main_cond["crossattn"].data_ptr()
        assert torch.equal(pag_cond["vector"], main_cond["vector"][:rows])
    if controlnet:
        # The ControlNet model ran for the main-pass calls only; PAG reused their residuals.
        assert harness.controlnet_calls == len(main_calls)

    # (c) the cond rows equal the old full pass's cond rows (bitwise when a replay keeps the call's batch).
    pag_x_out = harness.pag_params.pag_x_out
    assert pag_x_out.shape[0] == n_cond
    expected = oracle_pag_cond_rows(harness, main_calls, n_cond)
    torch.testing.assert_close(pag_x_out, expected, rtol=1e-5, atol=1e-6)
    starts = [sum(call[0].shape[0] for call in main_calls[:i]) for i in range(len(main_calls))]
    if all(start + call[0].shape[0] <= n_cond or start >= n_cond for start, call in zip(starts, main_calls)):
        assert torch.equal(pag_x_out, expected)
    # PAG actually perturbs the output.
    assert not torch.allclose(pag_x_out, harness.captured["x_out"][:n_cond])


def test_pag_skip_uncond_main_pass_replays_cond_rows():
    inputs = make_inputs(1, [1])
    harness = Harness()
    harness.opts.skip_early_cond = 1.0
    harness.enable_pag()
    harness.capture()
    harness.run(*inputs)
    main_calls = [call for call in harness.unet_calls if not call[3]]
    pag_calls = [call for call in harness.unet_calls if call[3]]
    assert [call[0].shape[0] for call in main_calls] == [1]
    assert [call[0].shape[0] for call in pag_calls] == [1]
    assert torch.equal(harness.pag_params.pag_x_out, oracle_pag_cond_rows(harness, main_calls, 1))


def record_main_pass(harness, inputs, n_cond):
    """Run the main pass with a memo armed by hand (PAG callbacks off) and hand back the memo."""
    memo = harness.row_memo.arm(harness.denoiser, n_cond)
    harness.run(*inputs)
    assert harness.row_memo.disarm(harness.denoiser) is memo
    return memo


@pytest.mark.parametrize("batch_size", [1, 2])
def test_controlnet_replay_reuses_recorded_control_state_bitwise(batch_size):
    """(b) Replayed rows equal a full recompute of the same rows: exact at the call's batch, ULP-level below it."""
    inputs = make_inputs(batch_size, [1] * batch_size, marked=True)
    harness = Harness()
    harness.install_controlnet()
    harness.enable_pag(callbacks=False)
    memo = record_main_pass(harness, inputs, batch_size)
    assert harness.controlnet_calls == 1
    assert all(rec.prefix is not None for rec in memo.calls)

    def replay(full_rows, reuse):
        saved = [(rec.prefix, rec.row_subset_ok) for rec in memo.calls]
        for rec in memo.calls:
            rec.row_subset_ok = not full_rows
            if not reuse:
                rec.prefix = None
        before = harness.controlnet_calls
        with harness.pag_on(), torch.no_grad():
            out = harness.pag.pag_cond_rows_x_out(harness.inner, memo, False)
        for rec, (prefix, row_subset_ok) in zip(memo.calls, saved):
            rec.prefix, rec.row_subset_ok = prefix, row_subset_ok
        return out, harness.controlnet_calls - before

    reused_full, cn_calls = replay(full_rows=True, reuse=True)
    assert cn_calls == 0
    recomputed_full, cn_calls = replay(full_rows=True, reuse=False)
    assert cn_calls == 1
    assert torch.equal(reused_full, recomputed_full)

    reused_cond, cn_calls = replay(full_rows=False, reuse=True)
    assert cn_calls == 0
    recomputed_cond, cn_calls = replay(full_rows=False, reuse=False)
    assert cn_calls == 1
    torch.testing.assert_close(reused_cond, recomputed_cond, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(reused_cond, reused_full, rtol=1e-5, atol=1e-6)


def test_controlnet_replay_recomputes_when_inputs_changed_after_the_main_pass():
    inputs = make_inputs(1, [1], marked=True)
    harness = Harness()
    harness.install_controlnet()
    harness.enable_pag(callbacks=False)
    memo = record_main_pass(harness, inputs, 1)
    rec = memo.calls[0]
    expected = None
    with harness.pag_on(), torch.no_grad():
        rec.row_subset_ok = False
        expected = harness.pag.pag_cond_rows_x_out(harness.inner, memo, False)
        rec.x.add_(0.0)  # an in-place edit bumps the version counter: nothing recorded may be reused
        before = harness.controlnet_calls
        out = harness.pag.pag_cond_rows_x_out(harness.inner, memo, False)
    assert rec.prefix is None
    assert harness.controlnet_calls == before + 1
    assert torch.equal(out, expected)


def test_controlnet_replay_recomputes_for_foreign_conditioning():
    inputs = make_inputs(1, [1], marked=True)
    harness = Harness()
    harness.install_controlnet()
    harness.enable_pag(callbacks=False)
    memo = record_main_pass(harness, inputs, 1)
    rec = memo.calls[0]
    cond = harness.row_memo.slice_cond_rows(rec.cond, rec.rows, 1)
    cond["crossattn"] = cond["crossattn"].clone()
    before = harness.controlnet_calls
    with harness.row_memo.replaying(rec, 1), torch.no_grad():
        harness.inner(rec.x[:1], rec.sigma[:1], cond)
    assert harness.controlnet_calls == before + 1
