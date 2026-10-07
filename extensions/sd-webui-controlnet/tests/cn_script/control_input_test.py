import importlib
import types
import unittest
from unittest import mock

import numpy as np
import torch
from einops import rearrange

utils = importlib.import_module("extensions.sd-webui-controlnet.tests.utils", "utils")

from scripts import controlnet, hook  # noqa: E402
from scripts.controlnet import clear_all_secondary_control_models, get_pytorch_control  # noqa: E402
from scripts.hook import BasicTransformerBlockSGM, UnetHook  # noqa: E402


def former_get_pytorch_control(x):
    """The former convert-on-CPU implementation (independent oracle)."""
    y = torch.from_numpy(x)
    y = y.float() / 255.0
    y = rearrange(y, 'h w c -> 1 c h w')
    y = y.clone()
    y = y.to(controlnet.devices.get_device_for("controlnet"))
    return y.clone()


class TestGetPytorchControl(unittest.TestCase):
    def test_matches_former_path_values_and_layout(self):
        rng = np.random.default_rng(0)
        every_value = np.arange(256, dtype=np.uint8).reshape(16, 16, 1).repeat(3, axis=2)
        cases = [
            every_value,
            rng.integers(0, 256, (37, 53, 3), dtype=np.uint8),
            rng.integers(0, 256, (8, 8, 4), dtype=np.uint8),
            rng.integers(0, 256, (20, 30, 3), dtype=np.uint8)[:, ::2],  # strided view
            (rng.random((9, 7, 4)) * 255).astype(np.float32),  # inpaint maps are float32
        ]
        for x in cases:
            with self.subTest(dtype=x.dtype, shape=x.shape):
                expected = former_get_pytorch_control(x)
                got = get_pytorch_control(x)
                self.assertEqual(got.dtype, expected.dtype)
                self.assertEqual(got.shape, expected.shape)
                self.assertEqual(got.stride(), expected.stride())
                self.assertTrue(torch.equal(got, expected))
                got.add_(1)  # result owns its storage: the lookup table is untouched
        self.assertTrue(torch.equal(controlnet._UINT8_TO_UNIT, torch.arange(256, dtype=torch.float32) / 255.0))


class FakeSDXLUNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_blocks = torch.nn.ModuleList(
            [torch.nn.Sequential(BasicTransformerBlockSGM(32, 1, 32)) if i in (4, 7) else torch.nn.Identity() for i in range(9)])
        self.middle_block = torch.nn.Sequential(BasicTransformerBlockSGM(64, 1, 64))
        self.output_blocks = torch.nn.ModuleList([torch.nn.Identity() for _ in range(6)])

    def forward(self, x, timesteps=None, context=None, y=None, **kwargs):
        return x


class TestSecondaryHijackRegistry(unittest.TestCase):
    def test_style_align_hijacks_are_registered_and_restored_without_module_walks(self):
        unet = FakeSDXLUNet()
        attn = [m for m in unet.modules() if isinstance(m, BasicTransformerBlockSGM)]
        gn = [unet.middle_block] + [unet.input_blocks[i] for i in (4, 5, 7, 8)] + list(unet.output_blocks)
        originals = {m: (m._forward if m in attn else None, m.forward) for m in attn + gn}
        process = types.SimpleNamespace(sample=lambda *a, **k: None)
        sd_ldm = types.SimpleNamespace(is_sdxl=True)

        owner = UnetHook()
        owner.hook(unet, sd_ldm, [], process, batch_option_style_align=True)
        self.assertEqual(set(unet._controlnet_secondary_hijacks), set(attn + gn))
        for m in attn:
            self.assertNotEqual(m._forward, originals[m][0])
        for m in gn:
            self.assertNotEqual(m.forward, originals[m][1])
        owner.restore()

        walk = mock.Mock(side_effect=AssertionError("full UNet module walk"))
        with mock.patch.object(hook, "torch_dfs", walk):
            clear_all_secondary_control_models(unet)  # controlnet_main_entry / postprocess
            plain = UnetHook()
            plain.hook(unet, sd_ldm, [], process)  # a request without reference/StyleAlign
            plain.restore()
        walk.assert_not_called()
        self.assertEqual(unet._controlnet_secondary_hijacks, [])
        for m in attn:
            self.assertEqual(m._forward, originals[m][0])
        for m in gn:
            self.assertEqual(m.forward, originals[m][1])


if __name__ == "__main__":
    unittest.main()
