"""CPU differential tests for PAG's main-pass row memo with a tiny random SDXL-shaped UNet.

The real CFGDenoiser drives the main pass, the real Incantations PAG script runs its hidden pass, the real
sgm UNetModel computes both, and optionally the real ControlNet hook with a tiny real cldm.ControlNet sits
in front of it, entered through its hooked process.sample like in a generation. The oracle replays every
logged main-pass UNet call whole with the PAG perturbation enabled, which is what PAG computed before it
was limited to cond rows.

Module hygiene: stubs are installed only while the code under test is imported, and the private sgm import
this file runs on is in sys.modules only while each of its tests runs.
"""

from __future__ import annotations

import contextlib
import importlib
import sys
import types

import pytest
import torch

from test.helpers import ROOT, load_source, stub_modules
from test.helpers import module as stub

SGM_ROOT = ROOT / "repositories" / "generative-models"
LDM_ROOT = ROOT / "repositories" / "stable-diffusion-stability-ai"
INC_SCRIPTS = ROOT / "extensions" / "sd-webui-incantations" / "scripts"
CN_SCRIPTS = ROOT / "extensions" / "sd-webui-controlnet" / "scripts"

pytestmark = pytest.mark.skipif(
    not (SGM_ROOT / "sgm").is_dir() or not (LDM_ROOT / "ldm").is_dir(),
    reason="generative-models / stable-diffusion repositories are not checked out",
)

_PRIVATE = {}  # "sgm" / "ldm": {name: module} of this file's own import of the real package, while the file runs


@contextlib.contextmanager
def isolated_modules(replacements, prefixes):
    """Run the block with ``replacements`` and nothing else under the top-level packages ``prefixes``.

    The session's other modules under ``prefixes`` are hidden meanwhile, and the modules the block imports there
    (against the replacements) are dropped afterwards; then exactly the hidden and replaced names are restored.
    Modules outside ``prefixes`` imported meanwhile (lazy torch internals) stay: re-importing those is not safe.
    """
    assert all(key.split(".")[0] in prefixes for key in replacements)

    def under_prefixes():
        return {key for key in sys.modules if key.split(".")[0] in prefixes}

    with stub_modules({**dict.fromkeys(under_prefixes()), **replacements}):
        try:
            yield
        finally:
            for key in under_prefixes() - replacements.keys():
                del sys.modules[key]


def _import_private(package, root, names):
    """Import ``names`` from the checkout ``root`` with the session's ``package`` modules hidden; returns every
    ``package`` module that import created, by name, and leaves sys.modules as it was."""
    with isolated_modules({}, {package}):
        sys.path.insert(0, str(root))
        try:
            for name in names:
                importlib.import_module(name)
        finally:
            sys.path.remove(str(root))
        return {key: value for key, value in sys.modules.items() if key.split(".")[0] == package}


@pytest.fixture(scope="module", autouse=True)
def private_sgm_and_ldm():
    """A private import of the real sgm package (and ldm util) for this file only.

    Other files may have imported the real sgm, which the WebUI hijacks patch; this file imports its own unpatched
    copy with the session's hidden, and nothing of it stays in sys.modules (private_sgm installs it per test).
    """
    _PRIVATE["sgm"] = _import_private("sgm", SGM_ROOT, (
        "sgm.modules.diffusionmodules.openaimodel", "sgm.modules.diffusionmodules.video_model", "sgm.modules.attention"))
    _PRIVATE["ldm"] = _import_private("ldm", LDM_ROOT, ("ldm.modules.diffusionmodules.util", "ldm.modules.diffusionmodules.upscaling"))
    yield
    _PRIVATE.clear()


@pytest.fixture(autouse=True)
def private_sgm(private_sgm_and_ldm):
    """sgm imports lazily while its UNets run: during each test the private copy stands in for the session's sgm."""
    with isolated_modules(_PRIVATE["sgm"], {"sgm"}):
        yield


def real_modules():
    assert _PRIVATE, "private_sgm_and_ldm fixture is not active"
    return {**_PRIVATE["sgm"], **_PRIVATE["ldm"]}


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

        return stub(
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


class DiffusionWrapper(torch.nn.Module):
    """shared.sd_model.model: hypertile marks its layers on this module."""

    def __init__(self, unet):
        super().__init__()
        self.diffusion_model = unet
        self.conditioning_key = "crossattn"


class Harness:
    """Real CFGDenoiser + real PAG script over a tiny sgm UNet, with A1111 globals stubbed.

    mode "unet": the raw sgm forward (no producer); "base": the WebUI's sd_unet forward wrapper; with
    install_controlnet() the ControlNet hook sits on top.
    """

    def __init__(self, mode="base", device="cpu"):
        real = real_modules()
        self.device = torch.device(device)
        self.openaimodel = real["sgm.modules.diffusionmodules.openaimodel"]
        self.unet = self.openaimodel.UNetModel(**UNET_CONFIG).eval()
        randomize(self.unet, seed=0, std=0.08)
        self.unet.to(self.device)

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
            model=DiffusionWrapper(self.unet),
            network_layer_mapping=mapping,
            cond_stage_model_empty_prompt=torch.zeros(1, 4, 16),
        )
        state = types.SimpleNamespace(interrupted=False, skipped=False, sampling_step=0, sampling_steps=4)
        self.shared = stub(
            "modules.shared",
            opts=self.opts,
            state=state,
            sd_model=self.sd_model,
            cmd_opts=types.SimpleNamespace(lowvram=False, medvram=False, medvram_sdxl=False),
        )
        self.script_callbacks = self.callbacks.module()

        self.row_memo = load_source("modules.sd_unet_row_memo", "modules/sd_unet_row_memo.py")

        self.modules = {
            "modules": stub("modules", package=True, sd_unet_row_memo=self.row_memo, shared=self.shared, script_callbacks=self.script_callbacks),
            "modules.prompt_parser": stub(
                "modules.prompt_parser",
                reconstruct_multicond_batch=lambda cond, step: cond,
                reconstruct_cond_batch=lambda uncond, step: uncond,
                MulticondLearnedConditioning=type("MulticondLearnedConditioning", (), {}),
                ComposableScheduledPromptConditioning=type("ComposableScheduledPromptConditioning", (), {}),
                ScheduledPromptConditioning=type("ScheduledPromptConditioning", (), {}),
            ),
            "modules.sd_samplers_common": stub(
                "modules.sd_samplers_common",
                InterruptedException=Exception,
                apply_refiner=lambda denoiser, sigma: False,
                store_latent=lambda latent: None,
            ),
            "modules.shared": self.shared,
            "modules.script_callbacks": self.script_callbacks,
            "modules.sd_unet_row_memo": self.row_memo,
            "modules.processing": stub("modules.processing", StableDiffusionProcessing=type("StableDiffusionProcessing", (), {})),
        }
        self.modules["modules.devices"] = stub(
            "modules.devices",
            dtype_unet=torch.float32,
            dtype_vae=torch.float32,
            device=self.device,
            get_device_for=lambda name: self.device,
            cond_cast_unet=lambda x: x,
            autocast=contextlib.nullcontext,
        )
        self.modules["modules"].devices = self.modules["modules.devices"]
        with isolated_modules(self.modules, {"modules"}):
            cfg_module = load_source("cfg_denoiser_under_test", "modules/sd_samplers_cfg_denoiser.py")
            self.sd_unet = load_source("sd_unet_under_test", "modules/sd_unet.py")
        if mode == "base":
            # sd_hijack installs this wrapper as UNetModel.forward; bind it to this instance only.
            wrapper = self.sd_unet.create_unet_forward(self.openaimodel.UNetModel.forward)
            self.unet.forward = types.MethodType(wrapper, self.unet)
        else:
            assert mode == "unet"
        self.encoder_calls = 0

        def count_encoder(module, args):
            self.encoder_calls += 1

        self.unet.input_blocks[0].register_forward_pre_hook(count_encoder)

        pag_modules = dict(self.modules)
        pag_modules.update({
            "scripts": stub("scripts", __path__=[str(INC_SCRIPTS)]),
            "modules.scripts": stub("modules.scripts", Script=object, AlwaysVisible=object()),
            "modules.headless_ui": stub("modules.headless_ui"),
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
        self.unet_hook = None
        self.controlnet_calls = 0
        self.captured = {}

    def install_controlnet(self, weight=0.7, cfg_injection=False, global_average_pooling=False, seed=1, preprocessor="depth_zoe", hint_channels=3, reference=False):
        real = real_modules()
        logger = types.SimpleNamespace(debug=lambda *a, **k: None, info=lambda *a, **k: None, warning=lambda *a, **k: None, error=lambda *a, **k: None)
        ldm_util = real["ldm.modules.diffusionmodules.util"]

        def extract_into_tensor(a, t, x_shape):
            b, *_ = t.shape
            out = a.gather(-1, t)
            return out.reshape(b, *((1,) * (len(x_shape) - 1)))

        cn_modules = dict(self.modules)
        cn_modules.update({
            "scripts": stub("scripts", __path__=[str(CN_SCRIPTS)]),
            "scripts.logging": stub("scripts.logging", logger=logger),
            "scripts.controlnet_lllite": stub("scripts.controlnet_lllite", clear_all_lllite=lambda: None),
            "scripts.ipadapter": stub("scripts.ipadapter", package=True),
            "scripts.ipadapter.plugable_ipadapter": stub("scripts.ipadapter.plugable_ipadapter", clear_all_ip_adapter=lambda: None),
            "scripts.ipadapter.ipadapter_model": stub("scripts.ipadapter.ipadapter_model", ImageEmbed=object),
            "modules.devices": self.modules["modules.devices"],
            # The core's torch with a resizing cat; equal to torch for this harness's aligned shapes.
            "modules.sd_hijack_unet": stub("modules.sd_hijack_unet", th=torch),
            "modules.lowvram": stub("modules.lowvram", send_everything_to_cpu=lambda: None),
            "modules.scripts": stub("modules.scripts", script_callbacks=self.script_callbacks),
            # cldm keeps ControlNet weights' layout with the NHWC GroupNorm switch (real module: torch-only, off here).
            "modules.openclaw_nhwc_groupnorm": load_source("modules.openclaw_nhwc_groupnorm", "modules/openclaw_nhwc_groupnorm.py"),
            "ldm": stub("ldm", package=True),
            "ldm.modules": stub("ldm.modules", package=True),
            "ldm.modules.diffusionmodules": stub("ldm.modules.diffusionmodules", package=True),
            "ldm.modules.diffusionmodules.util": ldm_util,
            "ldm.modules.diffusionmodules.upscaling": real["ldm.modules.diffusionmodules.upscaling"],
            "ldm.modules.diffusionmodules.openaimodel": stub("ldm.modules.diffusionmodules.openaimodel", UNetModel=type("UNetModel", (), {})),
            "ldm.modules.attention": stub("ldm.modules.attention", BasicTransformerBlock=type("BasicTransformerBlock", (), {})),
            "ldm.models": stub("ldm.models", package=True),
            "ldm.models.diffusion": stub("ldm.models.diffusion", package=True),
            "ldm.models.diffusion.ddpm": stub("ldm.models.diffusion.ddpm", extract_into_tensor=extract_into_tensor),
        })
        for name in ("devices", "lowvram", "scripts", "openclaw_nhwc_groupnorm"):
            setattr(cn_modules["modules"], name, cn_modules[f"modules.{name}"])
        with isolated_modules(cn_modules, {"modules", "scripts", "ldm"}):
            self.hook = importlib.import_module("scripts.hook")
            self.cldm = importlib.import_module("scripts.cldm")
            self.enums = importlib.import_module("scripts.enums")

        torch.manual_seed(seed)
        self.controlnet = self.cldm.PlugableControlModel(CONTROLNET_CONFIG).eval()
        randomize(self.controlnet, seed=seed, std=0.05)
        self.controlnet.to(self.device)

        def count(module, args, kwargs):
            self.controlnet_calls += 1

        self.controlnet.control_model.register_forward_pre_hook(count, with_kwargs=True)
        gen = torch.Generator().manual_seed(seed + 1)
        hint = torch.rand(1, hint_channels, 64, 64, generator=gen)
        if hint_channels == 4:
            hint[:, 3] = (hint[:, 3] > 0.5).float()  # inpaint mask channel
        hint = hint.to(self.device)

        def fake_vae_latent(p, x, mask=None):
            # Deterministic stand-in for the VAE encode: one latent row per hint row.
            pixels = x[:, :3] * 2.0 - 1.0
            return torch.nn.functional.avg_pool2d(torch.cat([pixels, pixels[:, :1]], dim=1), 8)

        self.hook.UnetHook.call_vae_using_process = staticmethod(fake_vae_latent)
        self.control_param = self.hook.ControlParams(
            control_model=None if reference else self.controlnet,
            preprocessor={"name": preprocessor, "threshold_a": 0.5, "threshold_b": 0.5},
            hint_cond=hint,
            weight=weight,
            guidance_stopped=False,
            start_guidance_percent=0.0,
            stop_guidance_percent=1.0,
            advanced_weighting=None,
            control_model_type=self.enums.ControlModelType.AttentionInjection if reference else self.enums.ControlModelType.ControlNet,
            hr_hint_cond=None,
            global_average_pooling=global_average_pooling,
            soft_injection=False,
            cfg_injection=cfg_injection,
        )
        self.sd_ldm = types.SimpleNamespace(is_sdxl=True, model=self.sd_model.model)
        self.process = types.SimpleNamespace(sample=lambda *a, **k: None)
        self.unet_hook = self.hook.UnetHook(lowvram=False)
        self.unet_hook.hook(
            model=self.unet,
            sd_ldm=self.sd_ldm,
            control_params=[self.control_param],
            process=self.process,
        )

    def sample(self, fn):
        """Run ``fn`` the way a generation runs sampling: inside ControlNet's hooked process.sample."""
        if self.unet_hook is None:
            return fn()
        self.process.sample_before_CN_hack = fn
        return self.process.sample()

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

    def enable_hypertile(self):
        """Mark encoder attention layers the way extensions-builtin/hypertile does, enabled."""
        layers = {}
        for name, module in self.sd_model.model.named_modules():
            if name.startswith("diffusion_model.input_blocks") and name.endswith("attn1"):
                setattr(module, "__webui_hypertile_params", types.SimpleNamespace(enabled=True))
                layers[name] = 1
        assert layers
        setattr(self.sd_model.model, "__webui_hypertile_layers", layers)

    def run(self, x, sigma, conds_list, cond, uncond, image_cond, s_min_uncond=0.0):
        self.captured.clear()
        self.unet_calls.clear()
        self.encoder_calls = 0
        x, sigma, image_cond = x.to(self.device), sigma.to(self.device), image_cond.to(self.device)
        cond = DictWithShape({key: value.to(self.device) for key, value in cond.items()})
        uncond = DictWithShape({key: value.to(self.device) for key, value in uncond.items()})
        with torch.inference_mode():
            return self.sample(lambda: self.denoiser(x, sigma, uncond, (conds_list, cond), 5.0, s_min_uncond, image_cond))

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
    with harness.pag_on(), torch.inference_mode():
        harness.sample(lambda: [outs.append(harness.inner(x, sigma, cond)) for x, sigma, cond, _ in main_calls])
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


MODES = ["unet", "base", "controlnet", "base_hypertile", "controlnet_hypertile"]


def make_harness(mode):
    harness = Harness(mode="unet" if mode == "unet" else "base")
    if mode.startswith("controlnet"):
        harness.install_controlnet()
    if mode.endswith("hypertile"):
        harness.enable_hypertile()
    return harness


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("layout", sorted(LAYOUTS))
def test_pag_cond_rows_match_old_full_pass_and_main_pass_is_unchanged(layout, mode):
    batch_size, repeats, cond_tokens, uncond_tokens, batch_cond_uncond = LAYOUTS[layout]
    controlnet = mode.startswith("controlnet")
    hypertile = mode.endswith("hypertile")
    inputs = make_inputs(batch_size, repeats, cond_tokens, uncond_tokens, marked=controlnet)
    n_cond = sum(repeats)

    # The baseline main pass goes through the unrecorded forward (the original sgm forward for "base").
    baseline = make_harness(mode)
    baseline.opts.batch_cond_uncond = batch_cond_uncond
    baseline.capture()
    baseline.run(*inputs)
    baseline_x_out = baseline.captured["x_out"]

    harness = make_harness(mode)
    harness.opts.batch_cond_uncond = batch_cond_uncond
    harness.enable_pag()
    harness.capture()
    harness.run(*inputs)

    main_calls = [call for call in harness.unet_calls if not call[3]]
    pag_calls = [call for call in harness.unet_calls if call[3]]
    # (a) the main pass is bitwise unchanged by recording.
    assert torch.equal(harness.captured["x_out"], baseline_x_out)
    # PAG mirrors the main pass's chunking, limited to cond rows, with the main call's own inputs.
    starts = [sum(call[0].shape[0] for call in main_calls[:i]) for i in range(len(main_calls))]
    # Hypertile keeps one PAG call per main call (its tile RNG draws per call); uncond-only calls run whole.
    expected_rows = []
    start = 0
    for x, *_ in main_calls:
        rows = min(x.shape[0], n_cond - start)
        if rows > 0:
            expected_rows.append(rows)
        elif hypertile:
            expected_rows.append(x.shape[0])
        start += x.shape[0]
    assert [call[0].shape[0] for call in pag_calls] == expected_rows
    for (main_x, main_sigma, main_cond, _), (pag_x, pag_sigma, pag_cond, _) in zip(main_calls, pag_calls):
        rows = pag_x.shape[0]
        assert pag_x.data_ptr() == main_x.data_ptr() and torch.equal(pag_x, main_x[:rows])
        assert torch.equal(pag_sigma, main_sigma[:rows])
        assert pag_cond["crossattn"].data_ptr() == main_cond["crossattn"].data_ptr()
        assert torch.equal(pag_cond["vector"], main_cond["vector"][:rows])
    if controlnet:
        # The ControlNet model ran for the main-pass calls only; PAG reused their residuals. Only the
        # uncond-only calls hypertile keeps (nothing recorded for them) recompute it, as before.
        uncond_only = sum(1 for (x, *_), start in zip(main_calls, starts) if start >= n_cond)
        assert harness.controlnet_calls == len(main_calls) + (uncond_only if hypertile else 0)
    if mode in ("base", "controlnet"):
        # PAG resumed at the middle block from the recorded encoder rows.
        assert harness.encoder_calls == len(main_calls)
    else:
        assert harness.encoder_calls == len(main_calls) + len(pag_calls)

    # (c) the cond rows equal the old full pass's cond rows (bitwise when a replay keeps the call's batch).
    pag_x_out = harness.pag_params.pag_x_out
    assert pag_x_out.shape[0] == n_cond
    expected = oracle_pag_cond_rows(harness, main_calls, n_cond)
    torch.testing.assert_close(pag_x_out, expected, rtol=1e-5, atol=1e-6)
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


@pytest.mark.parametrize("mode", ["base", "controlnet"])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_replay_reuses_recorded_rows_bitwise(batch_size, mode):
    """(b) Replayed rows equal a full recompute of the same rows: exact at the call's batch, ULP-level below it."""
    inputs = make_inputs(batch_size, [1] * batch_size, marked=mode == "controlnet")
    harness = make_harness(mode)
    harness.enable_pag(callbacks=False)
    memo = record_main_pass(harness, inputs, batch_size)
    assert harness.encoder_calls == 1
    assert all(rec.prefix is not None for rec in memo.calls)

    def replay(full_rows, reuse):
        saved = [(rec.prefix, rec.row_subset_ok) for rec in memo.calls]
        for rec in memo.calls:
            rec.row_subset_ok = not full_rows
            if not reuse:
                rec.prefix = None
        harness.encoder_calls = 0
        before = harness.controlnet_calls
        with harness.pag_on(), torch.inference_mode():
            out = harness.sample(lambda: harness.pag.pag_cond_rows_x_out(harness.inner, memo, False))
        for rec, (prefix, row_subset_ok) in zip(memo.calls, saved):
            rec.prefix, rec.row_subset_ok = prefix, row_subset_ok
        return out, harness.encoder_calls + harness.controlnet_calls - before

    reused_full, recomputed_parts = replay(full_rows=True, reuse=True)
    assert recomputed_parts == 0
    recomputed_full, recomputed_parts = replay(full_rows=True, reuse=False)
    assert recomputed_parts == (2 if mode == "controlnet" else 1)
    assert torch.equal(reused_full, recomputed_full)

    reused_cond, recomputed_parts = replay(full_rows=False, reuse=True)
    assert recomputed_parts == 0
    recomputed_cond, recomputed_parts = replay(full_rows=False, reuse=False)
    assert recomputed_parts == (2 if mode == "controlnet" else 1)
    torch.testing.assert_close(reused_cond, recomputed_cond, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(reused_cond, reused_full, rtol=1e-5, atol=1e-6)


def test_controlnet_replay_recomputes_when_inputs_changed_after_the_main_pass():
    inputs = make_inputs(1, [1], marked=True)
    harness = make_harness("controlnet")
    harness.enable_pag(callbacks=False)
    memo = record_main_pass(harness, inputs, 1)
    rec = memo.calls[0]
    expected = None
    with harness.pag_on(), torch.inference_mode():
        rec.row_subset_ok = False
        expected = harness.sample(lambda: harness.pag.pag_cond_rows_x_out(harness.inner, memo, False))
        # Inference tensors carry no version counter; a replaced input is what can be detected.
        rec.cond["c_concat"] = [rec.cond["c_concat"][0].clone()]
        before = harness.controlnet_calls
        out = harness.sample(lambda: harness.pag.pag_cond_rows_x_out(harness.inner, memo, False))
    assert rec.prefix is None
    assert harness.controlnet_calls == before + 1
    assert torch.equal(out, expected)


def test_controlnet_replay_recomputes_for_foreign_conditioning():
    inputs = make_inputs(1, [1], marked=True)
    harness = make_harness("controlnet")
    harness.enable_pag(callbacks=False)
    memo = record_main_pass(harness, inputs, 1)
    rec = memo.calls[0]
    cond = harness.row_memo.slice_cond_rows(rec.cond, rec.rows, 1)
    cond["crossattn"] = cond["crossattn"].clone()
    before = harness.controlnet_calls
    with harness.row_memo.replaying(rec, 1), torch.inference_mode():
        harness.sample(lambda: harness.inner(rec.x[:1], rec.sigma[:1], cond))
    assert harness.controlnet_calls == before + 1


def test_base_wrapper_main_pass_is_the_original_sgm_forward_bitwise():
    """The recorded main call runs the wrapper's copy of the sgm forward; it must match the original op for op."""
    harness = Harness(mode="base")
    gen = torch.Generator().manual_seed(5)
    x = torch.randn(3, 4, 8, 8, generator=gen)
    t = torch.rand(3, generator=gen) * 900
    context = torch.randn(3, 6, 16, generator=gen)
    y = torch.randn(3, 12, generator=gen)
    rec = harness.row_memo.CallRecord(0, x, t, {"crossattn": context, "vector": y}, cond_rows=2)
    with torch.inference_mode():
        original = harness.openaimodel.UNetModel.forward(harness.unet, x, timesteps=t, context=context, y=y)
        with harness.row_memo.recording(rec):
            recorded = harness.unet(x, timesteps=t, context=context, y=y)
    assert rec.prefix is not None
    assert torch.equal(recorded, original)


def legacy_controlnet_forward(harness, cn, x, hint, timesteps, context, y):
    """cldm.ControlNet.forward before guided-hint caching and the dtype_unet compute dtype (plain ControlNet)."""
    original_type = x.dtype
    x = x.to(cn.dtype)
    hint = hint.to(cn.dtype)
    timesteps = timesteps.to(cn.dtype)
    context = context.to(cn.dtype)
    if y is not None:
        y = y.to(cn.dtype)
    t_emb = harness.cldm.timestep_embedding(timesteps, cn.model_channels, repeat_only=False).to(cn.dtype)
    emb = cn.time_embed(t_emb)
    guided_hint = cn.input_hint_block(hint, emb, context)
    outs = []
    if cn.num_classes is not None:
        emb = emb + cn.label_emb(y)
    h = x
    for module, zero_conv in zip(cn.input_blocks, cn.zero_convs):
        if guided_hint is not None:
            h = module(h, emb, context)
            h += guided_hint
            guided_hint = None
        else:
            h = module(h, emb, context)
        outs.append(zero_conv(h, emb, context))
    h = cn.middle_block(h, emb, context)
    outs.append(cn.middle_block_out(h, emb, context))
    return [o.to(original_type) for o in outs]


def oracle_hooked_forward(harness, x, timesteps, context, y, hint=None):
    """The original ControlNet hooked forward for one plain SDXL unit, written out independently.

    The UNet part is the sgm forward (its own timestep embedding) plus the control residuals.
    """
    unet = harness.unet
    param = harness.control_param
    if (context[:, 0, :].abs() - 1024.0).abs().mean() < 1e-3:
        cond_mark = ((context[:, 0, :] + 1024.0).abs().mean(dim=1) > 1e-3).to(x.dtype)[:, None, None, None]
        ctx = context[:, 1:, :]  # the strided view unmark_prompt_context returns
    else:
        # Unmarked prompts (e.g. hires-pass conds): every row counts as cond.
        cond_mark = torch.ones(x.shape[0], 1, 1, 1, dtype=x.dtype)
        ctx = context
    hint = param.hint_cond if hint is None else hint
    control = legacy_controlnet_forward(harness, harness.controlnet.control_model, x, hint, timesteps, ctx, y)
    if param.cfg_injection or param.global_average_pooling:
        control = [c * cond_mark for c in control]
    control = [c * param.weight for c in control]
    if param.global_average_pooling:
        control = [torch.mean(c, dim=(2, 3), keepdim=True) for c in control]
    total = [0.0] * 10
    for idx, item in enumerate(control):
        total[idx] = item + total[idx]
    t_emb = harness.openaimodel.timestep_embedding(timesteps, unet.model_channels, repeat_only=False)
    emb = unet.time_embed(t_emb) + unet.label_emb(y)
    hs = []
    h = x
    for module in unet.input_blocks:
        h = module(h, emb, ctx)
        hs.append(h)
    h = unet.middle_block(h, emb, ctx) + total.pop()
    for module in unet.output_blocks:
        h = torch.cat([h, hs.pop() + total.pop()], dim=1)
        h = module(h, emb, ctx)
    return unet.out(h.type(x.dtype))


def unet_inputs(batch_size=2, seed=7, marked=True):
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(batch_size, 4, 8, 8, generator=gen)
    timesteps = torch.rand(batch_size, generator=gen) * 900
    context = torch.randn(batch_size, 5, 16, generator=gen)
    if marked:
        context[:, 0, :] = 1024.0
        context[batch_size // 2:, 0, :] = -1024.0
    y = torch.randn(batch_size, 12, generator=gen)
    return x, timesteps, context, y


@pytest.mark.parametrize("marked", [True, False], ids=["marked", "unmarked"])
@pytest.mark.parametrize("variant", ["balanced", "cfg_injection", "global_average_pooling"])
def test_controlnet_hooked_forward_matches_original_semantics(variant, marked):
    harness = Harness(mode="base")
    harness.install_controlnet(cfg_injection=variant == "cfg_injection", global_average_pooling=variant == "global_average_pooling")
    control_rows = []
    harness.controlnet.control_model.register_forward_pre_hook(lambda module, args, kwargs: control_rows.append(kwargs["x"].shape[0]), with_kwargs=True)
    x, timesteps, context, y = unet_inputs(marked=marked)
    with torch.inference_mode():
        out = harness.sample(lambda: harness.unet(x, timesteps=timesteps, context=context, y=y))
        expected = oracle_hooked_forward(harness, x, timesteps, context, y)
    if variant == "balanced" or not marked:
        assert control_rows == [2]
        assert torch.equal(out, expected)
    else:
        # Uncond-row residuals are zeroed by cond_mark: the ControlNet now runs on the cond row only.
        assert control_rows == [1]
        torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-6)


def test_controlnet_guided_hint_and_placement_are_computed_once_per_hint():
    harness = Harness(mode="base")
    harness.install_controlnet()
    hint_block_calls = []
    harness.controlnet.control_model.input_hint_block.register_forward_pre_hook(lambda module, args: hint_block_calls.append(1))
    fullvram_calls = []
    fullvram = harness.controlnet.fullvram
    harness.controlnet.fullvram = lambda: (fullvram_calls.append(1), fullvram())[1]
    x, timesteps, context, y = unet_inputs()
    with torch.inference_mode():
        first = harness.sample(lambda: harness.unet(x, timesteps=timesteps, context=context, y=y))
        second = harness.sample(lambda: harness.unet(x, timesteps=timesteps, context=context, y=y))
        assert len(hint_block_calls) == 1 and len(fullvram_calls) == 1
        assert torch.equal(first, second)
        harness.control_param.hint_cond = torch.rand(1, 3, 64, 64, generator=torch.Generator().manual_seed(9))
        third = harness.sample(lambda: harness.unet(x, timesteps=timesteps, context=context, y=y))
        # Outside the hooked process.sample (a leaked hook) ControlNet passes the UNet through untouched.
        controlnet_calls = harness.controlnet_calls
        plain = harness.unet(x, timesteps=timesteps, context=context, y=y)
        assert harness.controlnet_calls == controlnet_calls
        assert len(hint_block_calls) == 2
        assert torch.equal(third, oracle_hooked_forward(harness, x, timesteps, context, y))
    assert not torch.equal(first, third)
    assert torch.equal(plain, harness.openaimodel.UNetModel.forward(harness.unet, x, timesteps=timesteps, context=context, y=y))


def test_controlnet_unet_dtype_compute_is_bitwise_identical_under_bf16_autocast():
    """CN7: computing in dtype_unet instead of the fp32 construction dtype only removes casts autocast redoes."""
    harness = Harness(mode="unet")
    harness.install_controlnet()
    cn = harness.controlnet.control_model
    # bf16 weights where autocast computes in bf16; CPU autocast has no fp32 policy for group_norm/layer_norm
    # weights (CUDA autocast casts them), so the norms keep fp32 weights here.
    for module in cn.modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            module.to(torch.bfloat16)
    x, timesteps, context, y = unet_inputs()
    x, timesteps, context, y = x.bfloat16(), timesteps.bfloat16(), context[:, 1:, :].bfloat16(), y.bfloat16()
    hint = harness.control_param.hint_cond
    devices = harness.cldm.devices
    assert cn.dtype == torch.float32
    try:
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            legacy = legacy_controlnet_forward(harness, cn, x, hint, timesteps, context, y)
            devices.dtype_unet = torch.bfloat16
            current = cn(x=x, hint=hint, timesteps=timesteps, context=context, y=y)
            guided = cn.compute_guided_hint(hint)
            with_guided = cn(x=x, hint=hint, timesteps=timesteps, context=context, y=y, guided_hint=guided)
    finally:
        devices.dtype_unet = torch.float32
    assert all(o.dtype == torch.bfloat16 for o in current)
    assert all(torch.equal(a, b) for a, b in zip(legacy, current))
    assert all(torch.equal(a, b) for a, b in zip(legacy, with_guided))


def test_inpaint_only_pag_cond_rows_use_row_agnostic_cached_latents():
    """I5: hint latents cached at the hint's rows serve the 2-row main call and the 1-row PAG replay."""
    inputs = make_inputs(1, [1], marked=True)
    harness = make_harness("unet")
    harness.install_controlnet(preprocessor="inpaint_only", hint_channels=4)
    harness.enable_pag()
    harness.capture()
    harness.run(*inputs)
    main_calls = [call for call in harness.unet_calls if not call[3]]
    pag_calls = [call for call in harness.unet_calls if call[3]]
    assert [call[0].shape[0] for call in main_calls] == [2]
    assert [call[0].shape[0] for call in pag_calls] == [1]
    assert harness.control_param.used_hint_cond_latent.shape[0] == 1
    # inpaint_only post-processes with per-call tensors: never reused, recomputed on the cond row.
    assert harness.controlnet_calls == 2
    torch.testing.assert_close(harness.pag_params.pag_x_out, oracle_pag_cond_rows(harness, main_calls, 1), rtol=1e-5, atol=1e-6)


def test_reference_only_calls_are_replayed_whole_and_never_reused():
    """Reference units draw noise over all rows per call: PAG keeps the old full-row call."""
    inputs = make_inputs(1, [1], marked=True)
    harness = make_harness("base")
    harness.install_controlnet(preprocessor="reference_only", reference=True)
    harness.sd_ldm.sqrt_alphas_cumprod = torch.linspace(0.99, 0.05, 1000)
    harness.sd_ldm.sqrt_one_minus_alphas_cumprod = (1 - harness.sd_ldm.sqrt_alphas_cumprod ** 2).sqrt()
    harness.enable_pag()
    harness.capture()
    harness.run(*inputs)
    main_calls = [call for call in harness.unet_calls if not call[3]]
    pag_calls = [call for call in harness.unet_calls if call[3]]
    assert [call[0].shape[0] for call in main_calls] == [2]
    assert [call[0].shape[0] for call in pag_calls] == [2]
    # Main and PAG calls each ran the nested reference pass and their own encoder (nothing reused).
    assert harness.encoder_calls == 4
    assert harness.pag_params.pag_x_out.shape[0] == 1
    assert torch.isfinite(harness.pag_params.pag_x_out).all()


@pytest.mark.parametrize("layout", ["batched", "token_mismatch_split"])
def test_teacache_patched_unet_gets_whole_calls(layout):
    """TeaCache keys per-call-lane state on the rows it sees: PAG keeps the old whole-call lane."""
    batch_size, repeats, cond_tokens, uncond_tokens, batch_cond_uncond = LAYOUTS[layout]
    inputs = make_inputs(batch_size, repeats, cond_tokens, uncond_tokens)
    harness = make_harness("base")
    harness.unet._teacache_patched = True
    harness.enable_pag()
    harness.capture()
    harness.run(*inputs)
    main_calls = [call for call in harness.unet_calls if not call[3]]
    pag_calls = [call for call in harness.unet_calls if call[3]]
    assert [call[0].shape[0] for call in pag_calls] == [call[0].shape[0] for call in main_calls]
    assert torch.equal(harness.pag_params.pag_x_out, oracle_pag_cond_rows(harness, main_calls, sum(repeats)))


def test_controlnet_unet_timestep_embedding_is_the_sgm_one_computed_on_the_timesteps_device():
    """The hooked forward embeds timesteps like the UNet it replaces: sgm's util.timestep_embedding builds the
    frequencies on the timesteps' device. Frequencies built on the CPU and uploaded (ldm's util) can differ from
    a CUDA exp in the last bit, so on a GPU the hooked UNet would not compute the unhooked UNet's embedding."""
    from torch.overrides import TorchFunctionMode

    harness = Harness(mode="base")
    harness.install_controlnet()
    arange_devices = []

    class RecordArange(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            if func is torch.arange:
                arange_devices.append(kwargs.get("device"))
            return func(*args, **kwargs)

    x, timesteps, context, y = unet_inputs()
    with torch.inference_mode():
        with RecordArange():
            out = harness.sample(lambda: harness.unet(x, timesteps=timesteps, context=context, y=y))
        expected = oracle_hooked_forward(harness, x, timesteps, context, y)
    # The ControlNet's embedding and the hooked UNet's: both build their frequencies on the timesteps' device.
    assert len(arange_devices) >= 2
    assert all(device is not None and torch.device(device) == timesteps.device for device in arange_devices)
    assert torch.equal(out, expected)


def bf16_harness(preprocessor, hint_channels=3):
    """The harness UNet and ControlNet computing in bfloat16 under CPU autocast (norms keep fp32 weights; CPU
    autocast has no fp32 policy for them), with devices.dtype_unet = bfloat16 like the deployed runtime."""
    harness = Harness(mode="unet")
    harness.install_controlnet(preprocessor=preprocessor, hint_channels=hint_channels)
    # The real VAE encode returns latents in dtype_unet.
    fake_vae_latent = harness.hook.UnetHook.call_vae_using_process
    harness.hook.UnetHook.call_vae_using_process = staticmethod(lambda p, x, mask=None: fake_vae_latent(p, x, mask).to(torch.bfloat16))
    for module in (*harness.unet.modules(), *harness.controlnet.modules()):
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            module.to(torch.bfloat16)
    harness.cldm.devices.dtype_unet = torch.bfloat16
    return harness


def bf16_forward(harness, x, timesteps, context, y):
    with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
        return harness.sample(lambda: harness.unet(x.bfloat16(), timesteps=timesteps, context=context.bfloat16(), y=y.bfloat16()))


@pytest.mark.parametrize("preprocessor", ["inpaint_only", "tile_colorfix"])
def test_controlnet_eps_post_processing_is_float32_under_a_bf16_unet(preprocessor):
    """inpaint_only / colorfix rewrite the UNet's eps through x0 = a*x_t - b*eps and back (a, b ~ 10 at these
    timesteps). In bfloat16 the rounded coefficients and products perturbed eps by several ulps even where x0
    is kept; the conversion now runs in float32 and rounds once. Oracle: the same formulas in float64 on the
    same bf16 UNet output, rounded to bf16."""
    inpaint = preprocessor == "inpaint_only"
    harness = bf16_harness(preprocessor, hint_channels=4 if inpaint else 3)
    try:
        param = harness.control_param
        if inpaint:
            param.hint_cond[:, 3, :32] = 1.0  # inpaint the top half, keep the bottom half
            param.hint_cond[:, 3, 32:] = 0.0
        else:
            param.preprocessor["threshold_a"] = 2  # blur radius k
        x, timesteps, context, y = unet_inputs()
        timesteps = timesteps * 0.0 + torch.tensor([700.0, 950.0])
        out = bf16_forward(harness, x, timesteps, context, y)
        assert out.dtype == torch.bfloat16
        latent = param.used_hint_cond_latent
        # The same call without the post-processing (the name selects it; control and hint protocol are the same).
        param.preprocessor["name"] = "inpaint" if inpaint else "tile_resample"
        plain = bf16_forward(harness, x, timesteps, context, y)
    finally:
        harness.cldm.devices.dtype_unet = torch.float32

    ldm = harness.sd_ldm
    t = torch.round(timesteps.double()).long()

    def coef(table):
        return table.double()[t][:, None, None, None]

    a, b = coef(ldm.sqrt_recip_alphas_cumprod), coef(ldm.sqrt_recipm1_alphas_cumprod)
    xt, eps, x0_origin = x.bfloat16().double(), plain.double(), latent.double()
    x0_prd = a * xt - b * eps
    if inpaint:
        mask = torch.nn.functional.max_pool2d(param.hint_cond[:, 3:4].double(), (10, 10), stride=(8, 8), padding=1)
        assert 0 < mask.mean() < 1
        x0 = x0_prd * mask + x0_origin * (1 - mask)
    else:
        def blur(v, k=2):
            return torch.nn.functional.avg_pool2d(torch.nn.functional.pad(v, (k, k, k, k), mode="replicate"), (2 * k + 1, 2 * k + 1), stride=(1, 1))
        x0 = x0_prd - blur(x0_prd) + blur(x0_origin)
    w = param.weight
    expected = ((a * xt - x0) / b * w + eps * (1 - w)).bfloat16()
    ulp = torch.finfo(torch.bfloat16).eps * expected.double().abs().clamp_min(2.0 ** -8)
    assert ((out.double() - expected.double()).abs() <= ulp).all()
    assert (out == expected).double().mean() > 0.99
    if inpaint:
        # Where the mask keeps the prediction (x0 = x0_prd) with full weight, eps comes back unchanged.
        harness.cldm.devices.dtype_unet = torch.bfloat16
        try:
            param.preprocessor["name"] = "inpaint_only"
            param.weight = 1.0
            param.hint_cond[:, 3] = 1.0
            param.hint_cond = param.hint_cond.clone()  # the setter drops the hint-derived caches
            kept = bf16_forward(harness, x, timesteps, context, y)
            param.preprocessor["name"] = "inpaint"
            plain_full = bf16_forward(harness, x, timesteps, context, y)
        finally:
            harness.cldm.devices.dtype_unet = torch.float32
        assert torch.equal(kept, plain_full)


def test_controlnet_hires_pass_detection_with_same_size_hints_follows_a1111s_pass_flag():
    """Hires fix at scale 1 builds low-res and hires hints of the same size: the latent size cannot tell the
    passes apart, so every call used to count as the hires pass (hr_option, forced soft injection, ...)."""
    harness = Harness(mode="base")
    harness.install_controlnet()
    param = harness.control_param
    param.hr_hint_cond = torch.rand(1, 3, 64, 64, generator=torch.Generator().manual_seed(11))
    param.hr_option = harness.enums.HiResFixOption.HIGH_RES_ONLY
    x, timesteps, context, y = unet_inputs()
    with torch.inference_mode():
        harness.process.is_hr_pass = False
        low = harness.sample(lambda: harness.unet(x, timesteps=timesteps, context=context, y=y))
        assert harness.controlnet_calls == 0 and harness.unet_hook.is_in_high_res_fix is False
        plain = harness.openaimodel.UNetModel.forward(harness.unet, x, timesteps=timesteps, context=context[:, 1:], y=y)
        harness.process.is_hr_pass = True
        high = harness.sample(lambda: harness.unet(x, timesteps=timesteps, context=context, y=y))
        assert harness.controlnet_calls == 1 and harness.unet_hook.is_in_high_res_fix is True
        expected_high = oracle_hooked_forward(harness, x, timesteps, context, y, hint=param.hr_hint_cond)
    assert torch.equal(low, plain)
    assert torch.equal(high, expected_high)


def test_controlnet_hires_pass_detection_compares_both_dimensions():
    """A hires resize that keeps the height (1024x1024 -> 1536x1024): the low-res latent matches the low-res
    hint; comparing heights only picked the wider hires hint and failed on the residual shapes."""
    harness = Harness(mode="base")
    harness.install_controlnet()
    param = harness.control_param
    param.hr_hint_cond = torch.rand(1, 3, 64, 96, generator=torch.Generator().manual_seed(12))
    x, timesteps, context, y = unet_inputs()  # 8x8 latent = 64x64 pixels
    with torch.inference_mode():
        out = harness.sample(lambda: harness.unet(x, timesteps=timesteps, context=context, y=y))
        expected = oracle_hooked_forward(harness, x, timesteps, context, y)
    assert harness.unet_hook.is_in_high_res_fix is False
    assert torch.equal(out, expected)


@pytest.mark.parametrize("count", [9, 10, 13])
def test_controlnet_advanced_weighting_needs_one_weight_per_sdxl_residual(count):
    """SDXL ControlNets return 10 residuals; zip() used to drop residuals past a short list (the middle block
    first) and to apply the first 10 of 13 SD1.5 weights."""
    harness = Harness(mode="base")
    harness.install_controlnet(weight=0.7)
    harness.control_param.advanced_weighting = [0.7] * count
    x, timesteps, context, y = unet_inputs()
    with torch.inference_mode():
        if count != 10:
            with pytest.raises(ValueError, match="advanced_weighting"):
                harness.sample(lambda: harness.unet(x, timesteps=timesteps, context=context, y=y))
            return
        out = harness.sample(lambda: harness.unet(x, timesteps=timesteps, context=context, y=y))
        expected = oracle_hooked_forward(harness, x, timesteps, context, y)
    assert torch.equal(out, expected)


def test_controlnet_instantid_context_is_the_image_embed_rows_without_host_round_trips():
    """InstantID's ControlNet context is cond_emb * cond_mark + uncond_emb * (1 - cond_mark) (cond embeds on cond
    rows, uncond embeds on uncond rows), selected on the device from embeds cast once per request instead of
    moving cond_mark to the embeds' (CPU) device and back on every call (the former ImageEmbed.eval)."""
    harness = Harness(mode="base")
    harness.install_controlnet()
    param = harness.control_param
    param.control_model_type = harness.enums.ControlModelType.InstantID
    gen = torch.Generator().manual_seed(21)

    class Embed:  # ImageEmbed's fields only
        cond_emb = torch.randn(1, 3, 16, generator=gen)
        uncond_emb = torch.randn(1, 3, 16, generator=gen)

    param.control_context_override = Embed()
    contexts = []
    harness.controlnet.control_model.register_forward_pre_hook(lambda module, args, kwargs: contexts.append(kwargs["context"]), with_kwargs=True)
    x, timesteps, context, y = unet_inputs()  # row 0 cond, row 1 uncond
    with torch.inference_mode():
        harness.sample(lambda: harness.unet(x, timesteps=timesteps, context=context, y=y))
        harness.sample(lambda: harness.unet(x, timesteps=timesteps, context=context, y=y))
    cond_mark = torch.tensor([1.0, 0.0])[:, None, None]
    legacy = Embed.cond_emb * cond_mark + Embed.uncond_emb * (1 - cond_mark)  # the former ImageEmbed.eval
    assert len(contexts) == 2
    assert all(torch.equal(c, legacy) for c in contexts)
    assert param.control_context_device[0] is param.control_context_override
