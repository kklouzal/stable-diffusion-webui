import importlib
import itertools
import unittest

import torch

utils = importlib.import_module("extensions.sd-webui-controlnet.tests.utils", "utils")

from scripts import cldm  # noqa: E402

TINY = dict(
    in_channels=4, model_channels=32, hint_channels=3, num_res_blocks=1, attention_resolutions=[2],
    channel_mult=(1, 2), transformer_depth=[0, 1], transformer_depth_middle=1, context_dim=16,
    num_head_channels=16, use_linear_in_transformer=True, num_classes="sequential", adm_in_channels=24,
    global_average_pooling=False,
)


BITS = {2: torch.int16, 4: torch.int32, 8: torch.int64}


def bits(tensor):
    return tensor.view(BITS[tensor.element_size()])


def old_build(config, state_dict, dtype):
    """The former construct + load + cast path (independent oracle)."""
    model = cldm.ControlNet(**config).cpu()
    model.load_state_dict(state_dict, strict=False)
    return model.to(dtype) if dtype is not None else model


class TestControlNetFromStateDict(unittest.TestCase):
    def checkpoint(self, config, dtype, seed=0):
        torch.manual_seed(seed)
        reference = cldm.ControlNet(**config)
        state_dict = {k: (v.double() + torch.randn(v.shape, dtype=torch.float64)).to(dtype) for k, v in reference.state_dict().items()}
        # NaNs with payloads too: casts may canonicalize them differently.
        nan_bits = {torch.float16: [0x7C01, 0x7E00, -0x3FF, 0x7FFF], torch.bfloat16: [0x7F81, 0x7FC0, -0x7F, 0x7FFF],
                    torch.float32: [0x7F800001, 0x7FC00000, -0x7FFFFF, 0x7FFFFFFF],
                    torch.float64: [0x7FF0000000000001, 0x7FF8000000000000, -1, 0x7FFFFFFFFFFFFFFF]}[dtype]
        for value in state_dict.values():
            flat = value.view(-1)
            if flat.numel() >= len(nan_bits):
                flat[:len(nan_bits)] = torch.tensor(nan_bits).to(BITS[value.element_size()]).view(dtype)
        return state_dict

    def test_weights_and_dtypes_match_former_path_bitwise(self):
        for use_fp16, ckpt_dtype, dtype in itertools.product(
                (False, True), (torch.float16, torch.bfloat16, torch.float32, torch.float64),
                (torch.bfloat16, torch.float16, torch.float32, None)):
            with self.subTest(use_fp16=use_fp16, checkpoint=ckpt_dtype, dtype=dtype):
                config = dict(TINY, use_fp16=use_fp16)
                state_dict = self.checkpoint(config, ckpt_dtype)
                expected = old_build(config, state_dict, dtype).state_dict()
                got = cldm.PlugableControlModel(config, state_dict, dtype=dtype).control_model.state_dict()
                self.assertEqual(list(got), list(expected))
                for key in expected:
                    self.assertEqual(got[key].dtype, expected[key].dtype, key)
                    self.assertFalse(got[key].is_meta, key)
                    self.assertTrue(torch.equal(bits(got[key]), bits(expected[key])), key)

    def test_checkpoint_must_match_architecture(self):
        config = dict(TINY, use_fp16=False)
        state_dict = self.checkpoint(config, torch.float16)
        missing = dict(state_dict)
        missing.pop("middle_block_out.0.weight")
        with self.assertRaisesRegex(RuntimeError, r"1 missing keys \['middle_block_out.0.weight'\]"):
            cldm.PlugableControlModel(config, missing, dtype=torch.bfloat16)
        unexpected = dict(state_dict, extra=torch.zeros(1))
        with self.assertRaisesRegex(RuntimeError, r"1 unexpected keys \['extra'\]"):
            cldm.PlugableControlModel(config, unexpected, dtype=torch.bfloat16)
        wrong_shape = dict(state_dict)
        wrong_shape["middle_block_out.0.bias"] = torch.zeros(7, dtype=torch.float16)
        with self.assertRaisesRegex(RuntimeError, "size mismatch"):
            cldm.PlugableControlModel(config, wrong_shape, dtype=torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
