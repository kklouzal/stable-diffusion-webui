import importlib
import types
import unittest
from unittest import mock

import torch

utils = importlib.import_module("extensions.sd-webui-controlnet.tests.utils", "utils")

import scripts.hook as hook  # noqa: E402
from scripts.enums import ControlModelType  # noqa: E402
from scripts.hook import ControlParams, UnetHook, NEGATIVE_MARK_TOKEN, POSITIVE_MARK_TOKEN  # noqa: E402


class _Block(torch.nn.Module):
    def forward(self, h, emb, context):
        return h


class _OutBlock(torch.nn.Module):
    def forward(self, h, emb, context):
        return h[:, :4] + h[:, 4:]


class TinySD15UNet(torch.nn.Module):
    """The attributes the hooked forward uses: 12 input blocks (13 ControlNet residuals with the middle block)."""
    model_channels = 4

    def __init__(self):
        super().__init__()
        self.input_blocks = torch.nn.ModuleList([_Block() for _ in range(12)])
        self.middle_block = _Block()
        self.output_blocks = torch.nn.ModuleList([_OutBlock() for _ in range(12)])
        self.time_embed = torch.nn.Identity()
        self.out = torch.nn.Identity()

    def forward(self, x, timesteps=None, context=None, y=None, **kwargs):
        raise AssertionError("the hooked forward runs with control")


class FakeControlNet:
    """Returns 13 residuals for the rows it is given, in its own dtype; records them."""
    disable_memory_management = True

    def __init__(self, dtype):
        self.dtype = dtype
        self.outputs = []
        # Read by the inpaint-model check (4-channel latent input).
        self.control_model = types.SimpleNamespace(input_blocks=[[types.SimpleNamespace(in_channels=4)]])

    def __call__(self, x, hint, timesteps, context, y=None, control_type=None):
        gen = torch.Generator().manual_seed(len(self.outputs))
        out = [torch.randn(x.shape[0], 4, 8, 8, generator=gen).mul(3).to(self.dtype) for _ in range(13)]
        self.outputs.append(out)
        return out


class TestControlNetResiduals(unittest.TestCase):
    """The residuals added to the UNet equal (control rows zero-filled to the batch) * cond_mark * scale, the former
    formula, bit for bit and in its dtype, while the multiplies that cannot change a value are skipped."""

    def run_unet(self, rows, control_dtype, context_dtype, weight, cfg_injection, soft_injection):
        unet = TinySD15UNet()
        control_model = FakeControlNet(control_dtype)
        param = ControlParams(
            control_model=control_model, preprocessor={"name": "none"}, hint_cond=torch.zeros(1, 3, 64, 64),
            weight=weight, guidance_stopped=False, start_guidance_percent=0.0, stop_guidance_percent=1.0,
            advanced_weighting=None, control_model_type=ControlModelType.ControlNet, hr_hint_cond=None,
            global_average_pooling=False, soft_injection=soft_injection, cfg_injection=cfg_injection,
        )
        sd_ldm = types.SimpleNamespace(is_sdxl=False)
        process = types.SimpleNamespace(sample=lambda *args, **kwargs: None)
        net = UnetHook()
        net.hook(model=unet, sd_ldm=sd_ldm, control_params=[param], process=process)
        residuals = []

        def recording_adding(base, x, require_channel_alignment):
            if isinstance(x, torch.Tensor):
                residuals.append(x)
            return real_adding(base, x, require_channel_alignment)

        real_adding = hook.aligned_adding
        gen = torch.Generator().manual_seed(7)
        context = torch.randn(len(rows), 5, 16, generator=gen)
        mark = torch.tensor([POSITIVE_MARK_TOKEN if r == "c" else NEGATIVE_MARK_TOKEN for r in rows], dtype=torch.float32)
        context = torch.cat([mark[:, None, None].expand(-1, 1, 16), context], dim=1).to(context_dtype)
        x = torch.randn(len(rows), 4, 8, 8, generator=gen).to(context_dtype)
        try:
            net.sampling_active = True
            with mock.patch.object(hook, "aligned_adding", recording_adding):
                unet.forward(x, timesteps=torch.full((len(rows),), 500.0), context=context)
        finally:
            net.restore()

        # Oracle: the former residual formula on what the control model returned.
        cond = torch.tensor([1.0 if r == "c" else 0.0 for r in rows])[:, None, None, None].to(context_dtype)
        c_idx = [i for i, r in enumerate(rows) if r == "c"]
        partial = cfg_injection and 0 < len(c_idx) < len(rows) and c_idx == list(range(c_idx[0], c_idx[-1] + 1))
        outputs = control_model.outputs[0]
        if partial:
            outputs = [torch.zeros((len(rows), *c.shape[1:]), dtype=c.dtype).index_copy_(0, torch.tensor(c_idx), c) for c in outputs]
        if cfg_injection:
            outputs = [c * cond for c in outputs]
        scales = [weight * (0.825 ** float(12 - i)) for i in range(13)] if soft_injection else [weight] * 13
        expected = [c * s for c, s in zip(outputs, scales)]
        # The forward pops the middle residual first, then the output blocks' in reverse order.
        order = [12] + list(range(11, -1, -1))
        return residuals, [expected[i] for i in order], [control_model.outputs[0][i] for i in order]

    def test_residuals_match_the_former_formula(self):
        cases = 0
        for rows in ("cu", "ccuu", "cucu", "cc"):
            for control_dtype, context_dtype in ((torch.bfloat16, torch.float32), (torch.float32, torch.float32),
                                                 (torch.bfloat16, torch.bfloat16), (torch.float16, torch.float32)):
                for weight in (1.0, 0.7):
                    for cfg_injection in (False, True):
                        for soft_injection in (False, True):
                            with self.subTest(rows=rows, control=control_dtype, context=context_dtype, weight=weight,
                                              cfg_injection=cfg_injection, soft_injection=soft_injection):
                                got, expected, raw = self.run_unet(rows, control_dtype, context_dtype, weight,
                                                                   cfg_injection, soft_injection)
                                self.assertEqual(len(got), 13)
                                if weight == 1.0 and not cfg_injection and not soft_injection:
                                    # Nothing to multiply: the residuals are the control model's own tensors.
                                    self.assertTrue(all(g is r for g, r in zip(got, raw)))
                                for g, e in zip(got, expected):
                                    self.assertEqual(g.dtype, e.dtype)
                                    self.assertTrue(torch.equal(g, e))
                                    self.assertTrue(torch.equal(torch.signbit(g), torch.signbit(e)))
                                cases += 1
        self.assertEqual(cases, 128)
