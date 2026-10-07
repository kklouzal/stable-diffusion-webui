import importlib
import types
import unittest

import torch

utils = importlib.import_module("extensions.sd-webui-controlnet.tests.utils", "utils")

from scripts.adapter import PlugableAdapter  # noqa: E402
from scripts.controlnet_lllite import LLLiteModule, PlugableControlLLLite, clear_all_lllite  # noqa: E402


class CountingAdapter(torch.nn.Module):
    """Features that identify the hint they were computed from."""

    def __init__(self, as_list=True):
        super().__init__()
        self.calls = 0
        self.as_list = as_list

    def forward(self, hint):
        self.calls += 1
        features = [hint.mean(dim=(1, 2, 3)) * (i + 1) for i in range(4)]
        return features if self.as_list else features[0]


class TestPlugableAdapterFeatures(unittest.TestCase):
    def test_features_follow_the_hint(self):
        """hook.py passes the low-res hint every step and the high-res hint in the hires pass."""
        for as_list in (True, False):
            with self.subTest(as_list=as_list):
                model = CountingAdapter(as_list)
                adapter = PlugableAdapter(model)
                low, high = torch.full((1, 3, 8, 8), 0.25), torch.full((1, 3, 16, 16), 0.75)
                for hint in (low, low, high, high, low):
                    out = adapter(hint=hint)
                    expected = [hint.mean(dim=(1, 2, 3)) * (i + 1) for i in range(4)]
                    if as_list:
                        self.assertIsInstance(out, list)
                        for o, e in zip(out, expected):
                            self.assertTrue(torch.equal(o, e))
                    else:
                        self.assertTrue(torch.equal(out, expected[0]))
                self.assertEqual(model.calls, 3)
                adapter.release_request_state()
                self.assertIsNone(adapter.control)
                self.assertIsNone(adapter.hint_source)


class TestLLLiteHook(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.dim, self.size = 16, 64
        module = LLLiteModule("m", is_conv2d=False, in_dim=self.dim, depth=1, cond_emb_dim=8, mlp_dim=12)
        self.state = {f"lllite_unet_input_blocks_4_1_transformer_blocks_0_attn1_to_q.{k}": v
                      for k, v in module.state_dict().items()}
        self.to_q = torch.nn.Linear(self.dim, self.dim, bias=False)
        block = types.SimpleNamespace(transformer_blocks=[types.SimpleNamespace(attn1=types.SimpleNamespace(to_q=self.to_q))])
        self.unet = types.SimpleNamespace(input_blocks=[None, None, None, None, [None, block]],
                                          current_sampling_percent=0.5, is_in_high_res_fix=False,
                                          current_h_shape=(2, self.dim, self.size // 8, self.size // 8))
        self.hint = torch.rand(1, 3, self.size, self.size)

    def tearDown(self):
        clear_all_lllite()

    def test_hooked_projection_matches_the_module(self):
        lllite = PlugableControlLLLite(self.state)
        oracle = LLLiteModule("m", is_conv2d=False, in_dim=self.dim, depth=1, cond_emb_dim=8, mlp_dim=12)
        oracle.load_state_dict({k.split(".", 1)[1]: v for k, v in self.state.items()})
        oracle.set_cond_image(self.hint * 2.0 - 1.0)
        lllite.hook(model=self.unet, cond=self.hint, weight=0.6, start=0.2, end=0.8)
        x = torch.randn(2, (self.size // 8) ** 2, self.dim)
        with torch.no_grad():
            expected = torch.nn.Linear.forward(self.to_q, x + oracle(x, self.unet.current_h_shape) * 0.6)
            self.assertTrue(torch.equal(self.to_q(x), expected))

            module = lllite.modules["lllite_unet_input_blocks_4_1_transformer_blocks_0_attn1_to_q"]
            module.to = lambda *a, **k: self.fail("module already on the call's device")
            self.assertTrue(torch.equal(self.to_q(x), expected))

            self.unet.current_sampling_percent = 0.9
            self.assertTrue(torch.equal(self.to_q(x), torch.nn.Linear.forward(self.to_q, x)))


if __name__ == "__main__":
    unittest.main()
