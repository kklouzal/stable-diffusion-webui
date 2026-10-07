"""CPU differential tests for PAG's main-pass row memo with a tiny random SDXL-style UNet.

The real CFGDenoiser drives the main pass, the real Incantations PAG script runs its hidden pass, and the
real sgm UNetModel computes both. The oracle replays every logged main-pass UNet call whole with the PAG
perturbation enabled, which is what PAG computed before it was limited to cond rows.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from pathlib import Path
from unittest import mock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SGM_ROOT = ROOT / "repositories" / "generative-models"
INC_SCRIPTS = ROOT / "extensions" / "sd-webui-incantations" / "scripts"

pytestmark = pytest.mark.skipif(not (SGM_ROOT / "sgm").is_dir(), reason="generative-models repository is not checked out")


def _module(name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _load(name, path, replacements):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, replacements):
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return module


def import_sgm_openaimodel():
    sys.path.insert(0, str(SGM_ROOT))
    try:
        return importlib.import_module("sgm.modules.diffusionmodules.openaimodel")
    finally:
        sys.path.remove(str(SGM_ROOT))


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
            self.denoiser[:] = [c for c in self.denoiser if c is not fn]
            self.denoised[:] = [c for c in self.denoised if c is not fn]

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


class Harness:
    """Real CFGDenoiser + real PAG script over a tiny sgm UNet, with A1111 globals stubbed."""

    def __init__(self, unet_forward_wrapper=None):
        torch.manual_seed(0)
        self.openaimodel = import_sgm_openaimodel()
        self.unet = self.openaimodel.UNetModel(
            in_channels=4,
            model_channels=32,
            out_channels=4,
            num_res_blocks=1,
            attention_resolutions=[2],
            channel_mult=[1, 2],
            num_head_channels=16,
            use_linear_in_transformer=True,
            transformer_depth=[0, 1],
            context_dim=16,
            num_classes="sequential",
            adm_in_channels=12,
            spatial_transformer_attn_type="softmax",
        ).eval()
        with torch.no_grad():
            for param in self.unet.parameters():
                param.normal_(0.0, 0.08)

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
        self.shared = _module("modules.shared", opts=self.opts, state=state, sd_model=self.sd_model)
        script_callbacks = self.callbacks.module()

        row_memo = _load("modules.sd_unet_row_memo", ROOT / "modules" / "sd_unet_row_memo.py", {})
        self.row_memo = row_memo
        prompt_parser = _module(
            "modules.prompt_parser",
            reconstruct_multicond_batch=lambda cond, step: cond,
            reconstruct_cond_batch=lambda uncond, step: uncond,
        )
        modules_pkg = _module("modules", sd_unet_row_memo=row_memo)
        modules_pkg.__path__ = []
        graphs = _module("modules.openclaw_cuda_graphs", run=lambda fn, x, sigma, cond, denoiser=None: fn(x, sigma, cond=cond))
        modules_pkg.openclaw_cuda_graphs = graphs
        replacements = {
            "modules": modules_pkg,
            "modules.prompt_parser": prompt_parser,
            "modules.sd_samplers_common": _module(
                "modules.sd_samplers_common",
                InterruptedException=Exception,
                apply_refiner=lambda denoiser, sigma: False,
                store_latent=lambda latent: None,
            ),
            "modules.shared": self.shared,
            "modules.script_callbacks": script_callbacks,
            "modules.openclaw_cuda_graphs": graphs,
            "modules.sd_unet_row_memo": row_memo,
        }
        cfg_module = _load("cfg_denoiser_under_test", ROOT / "modules" / "sd_samplers_cfg_denoiser.py", replacements)

        scripts_pkg = _module("scripts")
        scripts_pkg.__path__ = [str(INC_SCRIPTS)]
        pag_replacements = dict(replacements)
        pag_replacements.update({
            "scripts": scripts_pkg,
            "modules.scripts": _module("modules.scripts", Script=object, AlwaysVisible=object()),
            "modules.headless_ui": _module("modules.headless_ui"),
            "modules.processing": _module("modules.processing", StableDiffusionProcessing=object),
        })
        with mock.patch.dict(sys.modules, pag_replacements):
            for name in [key for key in sys.modules if key.startswith("scripts.")]:
                del sys.modules[name]
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
        self.captured = {}

    def pag_modules(self):
        return [m for m in self.sd_model.network_layer_mapping.values() if "middle_block_1_transformer_blocks_0_attn1" in m.network_layer_name]

    def pag_enabled(self):
        return any(getattr(m, "pag_enable", False) for m in self.pag_modules())

    def enable_pag(self):
        self.script = self.pag.PAGExtensionScript()
        p = types.SimpleNamespace(incant_cfg_params={}, steps=4, cfg_scale=5.0, batch_size=1, extra_generation_params={})
        self.script.pag_process_batch(p, True, 3.0, 0, 150, False, "Constant", 0.0, 100.0, False)
        self.pag_params = p.incant_cfg_params["pag_params"]
        assert len(self.pag_params.crossattn_modules) == 1

    def capture(self):
        self.callbacks.denoised.append(lambda params: self.captured.update(x_out=params.x.clone()))

    def run(self, x, sigma, conds_list, cond, uncond, image_cond, s_min_uncond=0.0):
        self.captured.clear()
        self.unet_calls.clear()
        with torch.no_grad():
            denoised = self.denoiser(x, sigma, uncond, (conds_list, cond), 5.0, s_min_uncond, image_cond)
        return denoised


class DictWithShape(dict):
    """modules.prompt_parser.DictWithShape: SDXL conditioning with the cross-attention shape."""

    def __init__(self, x, shape=None):
        super().__init__()
        self.update(x)

    @property
    def shape(self):
        return self["crossattn"].shape


def make_inputs(batch_size, repeats, cond_tokens=4, uncond_tokens=4, seed=1):
    gen = torch.Generator().manual_seed(seed)
    n_cond = sum(repeats)
    x = torch.randn(batch_size, 4, 8, 8, generator=gen)
    sigma = torch.full((batch_size,), 1.7)
    cond = DictWithShape({"crossattn": torch.randn(n_cond, cond_tokens, 16, generator=gen), "vector": torch.randn(n_cond, 12, generator=gen)})
    uncond = DictWithShape({"crossattn": torch.randn(batch_size, uncond_tokens, 16, generator=gen), "vector": torch.randn(batch_size, 12, generator=gen)})
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
    for module in harness.pag_modules():
        module.pag_enable = True
    try:
        with torch.no_grad():
            for x, sigma, cond, _ in main_calls:
                c_in = (1.0 / (sigma ** 2 + 1.0) ** 0.5)[:, None, None, None]
                eps = harness.unet(x * c_in, timesteps=sigma * 100.0, context=cond["crossattn"], y=cond["vector"])
                outs.append(x - eps * sigma[:, None, None, None])
    finally:
        for module in harness.pag_modules():
            module.pag_enable = False
    return torch.cat(outs)[:n_cond]


LAYOUTS = {
    # name: (batch_size, repeats, cond_tokens, uncond_tokens, batch_cond_uncond, s_min_uncond)
    "batched": (1, [1], 4, 4, True, 0.0),
    "batch_size_2": (2, [1, 1], 4, 4, True, 0.0),
    "and_prompt_batched": (1, [2], 4, 4, True, 0.0),
    "token_mismatch_split": (1, [2], 8, 4, True, 0.0),
    "batching_disabled": (2, [1, 1], 4, 4, False, 0.0),
    "and_prompt_batching_disabled": (2, [2, 1], 4, 4, False, 0.0),
}


@pytest.mark.parametrize("layout", sorted(LAYOUTS))
def test_pag_cond_rows_match_old_full_pass_and_main_pass_is_unchanged(layout):
    batch_size, repeats, cond_tokens, uncond_tokens, batch_cond_uncond, s_min_uncond = LAYOUTS[layout]
    inputs = make_inputs(batch_size, repeats, cond_tokens, uncond_tokens)
    n_cond = sum(repeats)

    baseline = Harness()
    baseline.opts.batch_cond_uncond = batch_cond_uncond
    baseline.capture()
    baseline.run(*inputs, s_min_uncond=s_min_uncond)
    baseline_x_out = baseline.captured["x_out"]

    harness = Harness()
    harness.opts.batch_cond_uncond = batch_cond_uncond
    harness.enable_pag()
    harness.capture()
    harness.run(*inputs, s_min_uncond=s_min_uncond)

    main_calls = [call for call in harness.unet_calls if not call[3]]
    pag_calls = [call for call in harness.unet_calls if call[3]]
    # (a) the main pass is bitwise unchanged by recording.
    assert torch.equal(harness.captured["x_out"], baseline_x_out)
    # PAG mirrors the main pass's chunking, limited to cond rows.
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
        assert torch.equal(pag_cond["crossattn"], main_cond["crossattn"][:rows])
        assert torch.equal(pag_cond["vector"], main_cond["vector"][:rows])

    # (c) the cond rows equal the old full pass's cond rows.
    pag_x_out = harness.pag_params.pag_x_out
    assert pag_x_out.shape[0] == n_cond
    expected = oracle_pag_cond_rows(harness, main_calls, n_cond)
    torch.testing.assert_close(pag_x_out, expected, rtol=1e-5, atol=1e-6)
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
    torch.testing.assert_close(harness.pag_params.pag_x_out, oracle_pag_cond_rows(harness, main_calls, 1), rtol=0, atol=0)
