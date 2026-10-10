"""Checkpoint ControlNets under a bfloat16 UNet on CUDA keep float16 weights and compute in float16 (cldm.controlnet_dtype).

CUDA is not needed: the build runs on the CPU (the policy reads the ControlNet device), and the forward is exercised under
CPU autocast, where float16 and bfloat16 autocast behave like CUDA's for the ops these tiny models use."""
import importlib
import types
import unittest
from unittest import mock

import torch

utils = importlib.import_module("extensions.sd-webui-controlnet.tests.utils", "utils")

from modules import devices  # noqa: E402
from scripts import cldm, controlnet_model_guess  # noqa: E402
import scripts.controlnet as controlnet_script  # noqa: E402
from scripts.controlnet import Script  # noqa: E402
from scripts.enums import ControlModelType  # noqa: E402

TINY_SDXL = dict(
    in_channels=4, model_channels=32, hint_channels=3, num_res_blocks=1, attention_resolutions=[2],
    channel_mult=(1, 2), transformer_depth=[0, 1], transformer_depth_middle=1, context_dim=16,
    num_head_channels=16, use_linear_in_transformer=True, num_classes="sequential", adm_in_channels=24,
    global_average_pooling=False, use_fp16=False,
)


def checkpoint(dtype=torch.float16, seed=0):
    """A float16 file's weights (a zero-initialized zero_conv would hide the residual numerics)."""
    torch.manual_seed(seed)
    model = cldm.ControlNet(**TINY_SDXL)
    return {k: (v + 0.05 * torch.randn(v.shape)).to(dtype) for k, v in model.state_dict().items()}


def env(dtype_unet, device):
    return mock.patch.multiple(devices, dtype_unet=dtype_unet, get_device_for=lambda name: torch.device(device))


def inputs(seed=1, size=16):
    g = torch.Generator().manual_seed(seed)
    return dict(
        x=torch.randn(2, 4, size, size, generator=g).to(torch.bfloat16),
        hint=torch.rand(2, 3, size * 8, size * 8, generator=g),
        timesteps=torch.tensor([600.0, 600.0]),
        context=torch.randn(2, 7, 16, generator=g).to(torch.bfloat16),
        y=torch.randn(2, 24, generator=g).to(torch.bfloat16),
    )


def build(state_dict, dtype, float16_name=None):
    model = cldm.PlugableControlModel(TINY_SDXL, state_dict, dtype=dtype).eval()
    model.control_model.float16_name = float16_name
    return model


_group_norm, _layer_norm = torch.nn.functional.group_norm, torch.nn.functional.layer_norm


def cuda_autocast_norms():
    """CUDA autocast runs group_norm and layer_norm in float32 (its float32 op list); CPU autocast does not, and its
    group_norm rejects float32 input with float16 weights. Emulate the CUDA policy: float32 operands, result in x's dtype."""
    def f32(t):
        return None if t is None else t.float()

    def group_norm(x, num_groups, weight=None, bias=None, eps=1e-5):
        with torch.autocast("cpu", enabled=False):
            return _group_norm(x.float(), num_groups, f32(weight), f32(bias), eps)

    def layer_norm(x, shape, weight=None, bias=None, eps=1e-5):
        with torch.autocast("cpu", enabled=False):
            return _layer_norm(x.float(), shape, f32(weight), f32(bias), eps)

    return mock.patch.multiple(torch.nn.functional, group_norm=group_norm, layer_norm=layer_norm)


def autocast(dtype):
    return torch.autocast("cpu", dtype=dtype or torch.bfloat16, enabled=dtype is not None)


def run(model, dtype_unet, outer_autocast, **kwargs):
    with env(dtype_unet, "cpu"), cuda_autocast_norms(), torch.no_grad(), autocast(outer_autocast):
        return model(**inputs(), **kwargs)


class TestPolicy(unittest.TestCase):
    def test_float16_only_for_a_bfloat16_unet_on_cuda(self):
        for dtype_unet, device, expected in (
                (torch.bfloat16, "cuda", torch.float16), (torch.bfloat16, "cpu", torch.bfloat16),
                (torch.float16, "cuda", torch.float16), (torch.float32, "cuda", torch.float32)):
            with self.subTest(dtype_unet=dtype_unet, device=device), env(dtype_unet, device):
                self.assertEqual(cldm.controlnet_dtype(), expected)

    def build_by_guess(self, dtype_unet, device):
        state_dict = checkpoint()
        with env(dtype_unet, device), mock.patch.object(controlnet_model_guess, "controlnet_sdxl_small_config", TINY_SDXL):
            built = controlnet_model_guess.build_model_by_guess(dict(state_dict), None, "/models/depth-sdxl.fp16.safetensors")
        self.assertEqual(built.type, ControlModelType.ControlNet)
        return state_dict, built

    def test_build_keeps_the_float16_file_weights_bitwise(self):
        state_dict, built = self.build_by_guess(torch.bfloat16, "cuda")
        cm = built.model.control_model
        self.assertEqual(cm.float16_name, "depth-sdxl.fp16.safetensors")
        loaded = cm.state_dict()
        for key, value in state_dict.items():
            self.assertEqual(loaded[key].dtype, torch.float16, key)
            self.assertTrue(torch.equal(loaded[key].view(torch.int16), value.view(torch.int16)), key)

    def test_other_configurations_build_in_the_unet_dtype(self):
        for dtype_unet, device in ((torch.bfloat16, "cpu"), (torch.float16, "cuda"), (torch.float32, "cuda")):
            with self.subTest(dtype_unet=dtype_unet, device=device):
                _, built = self.build_by_guess(dtype_unet, device)
                self.assertIsNone(built.model.control_model.float16_name)
                self.assertEqual({v.dtype for v in built.model.state_dict().values()}, {dtype_unet})

    def test_build_control_model_keeps_float16_and_casts_the_rest(self):
        state_dict = checkpoint()
        p = types.SimpleNamespace(sd_model=types.SimpleNamespace(dtype=torch.bfloat16))
        for device, expected in (("cuda", torch.float16), ("cpu", torch.bfloat16)):
            with self.subTest(device=device), env(torch.bfloat16, device), \
                    mock.patch.object(controlnet_model_guess, "controlnet_sdxl_small_config", TINY_SDXL), \
                    mock.patch.object(Script, "_resolve_model_path", staticmethod(lambda model: (model, "/m/" + model))), \
                    mock.patch.object(controlnet_script, "load_state_dict", lambda path: dict(state_dict)):
                built = Script.build_control_model(p, None, "depth.safetensors")
                self.assertEqual({v.dtype for v in built.control_model.model.state_dict().values()}, {expected})

    def test_weights_overflowing_float16_fail_the_load(self):
        state_dict = checkpoint(torch.float32)
        state_dict["middle_block_out.0.bias"][3] = 1e6
        with self.assertRaisesRegex(RuntimeError, r"overflow float16: \['middle_block_out.0.bias'\]"):
            build(state_dict, torch.float16)
        # A non-finite value the file already holds is not an overflow (and loads as before).
        state_dict["middle_block_out.0.bias"][3] = float("inf")
        build(state_dict, torch.float16)


class TestFloat16Forward(unittest.TestCase):
    def setUp(self):
        self.file = checkpoint()
        self.reference = [o.float() for o in run(build(self.file, torch.float32), torch.float32, None)]

    def rel(self, outs):
        return max(float((o.float() - r).norm() / r.norm()) for o, r in zip(outs, self.reference))

    def test_float16_activations_under_the_bfloat16_autocast(self):
        model = build(self.file, torch.float16, "tiny")
        zero_conv_out = []
        model.control_model.zero_convs[0].register_forward_hook(lambda m, a, out: zero_conv_out.append(out.dtype))
        outs = run(model, torch.bfloat16, torch.bfloat16)
        self.assertEqual(zero_conv_out, [torch.float16])  # the nested autocast, not the outer bfloat16 one
        self.assertEqual({o.dtype for o in outs}, {torch.bfloat16})  # residuals in the UNet's dtype
        self.assertEqual([o.shape for o in outs], [r.shape for r in self.reference])

        bf16 = build(self.file, torch.bfloat16)
        bf16_outs = run(bf16, torch.bfloat16, torch.bfloat16)
        # float16 against the float32 build of the same file: well under the bfloat16 path's error. The float16
        # result is rounded to bfloat16 on return, so its floor is the bfloat16 rounding (2^-9 relative).
        self.assertLess(self.rel(outs), 0.5 * self.rel(bf16_outs))
        self.assertLess(self.rel(outs), 1e-2)

    def test_guided_hint_is_float16_and_reusable(self):
        model = build(self.file, torch.float16, "tiny")
        with env(torch.bfloat16, "cpu"), cuda_autocast_norms(), torch.no_grad(), autocast(torch.bfloat16):
            guided = model.control_model.compute_guided_hint(inputs()["hint"])
        self.assertEqual(guided.dtype, torch.float16)
        direct = run(model, torch.bfloat16, torch.bfloat16)
        cached = run(model, torch.bfloat16, torch.bfloat16, guided_hint=guided)
        for a, b in zip(direct, cached):
            self.assertTrue(torch.equal(a, b))

    def test_float16_overflow_fails_closed_naming_the_model(self):
        model = build(self.file, torch.float16, "depth-sdxl.fp16.safetensors")
        with torch.no_grad():
            model.control_model.zero_convs[1][0].weight.fill_(30000.0)
        with self.assertRaisesRegex(RuntimeError, r"ControlNet depth-sdxl\.fp16\.safetensors produced non-finite residuals"):
            run(model, torch.bfloat16, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
