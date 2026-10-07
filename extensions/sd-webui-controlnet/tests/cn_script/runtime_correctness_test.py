import hashlib
import importlib
import os
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
import torch

utils = importlib.import_module("extensions.sd-webui-controlnet.tests.utils", "utils")

from modules import devices  # noqa: E402
from modules.processing import StableDiffusionProcessingTxt2Img  # noqa: E402
from scripts.cldm import PlugableControlModel  # noqa: E402
from scripts.controlnet import Script  # noqa: E402
from scripts.hook import TorchCache, UnetHook, register_schedule  # noqa: E402


def legacy_hash(key):
    """TorchCache.hash before bfloat16 support (float16/float32 keys only)."""
    v = key.detach().cpu().numpy().astype(np.float32)
    v = np.ascontiguousarray((v * 1000.0).astype(np.int32).copy())
    return hashlib.sha1(v).hexdigest()


class TestTorchCacheBfloat16(unittest.TestCase):
    """ControlNet's VAE cache keys are tensors in devices.dtype_vae, which is bfloat16 under --dtype bfloat16."""

    def test_hash_matches_former_keys_and_accepts_bfloat16(self):
        x = torch.rand(1, 3, 16, 16, generator=torch.Generator().manual_seed(0)) * 2 - 1
        cache = TorchCache()
        for dtype in (torch.float16, torch.float32):
            self.assertEqual(cache.hash(x.to(dtype)), legacy_hash(x.to(dtype)))
        with self.assertRaises(TypeError):
            legacy_hash(x.bfloat16())
        self.assertEqual(cache.hash(x.bfloat16()), legacy_hash(x.bfloat16().float()))

    def test_vae_encode_with_a_bfloat16_vae(self):
        encoded = []

        def encode_first_stage(img):
            encoded.append(img.dtype)
            return torch.nn.functional.avg_pool2d(torch.cat([img, img[:, :1]], dim=1), 8)

        p = types.SimpleNamespace(sd_model=types.SimpleNamespace(
            encode_first_stage=encode_first_stage, get_first_stage_encoding=lambda z: z))
        x = torch.rand(1, 3, 16, 16, generator=torch.Generator().manual_seed(1))
        with mock.patch.object(devices, "dtype_vae", torch.bfloat16), \
                mock.patch.object(devices, "dtype_unet", torch.bfloat16), \
                mock.patch.object(devices, "device", torch.device("cpu")), \
                mock.patch.object(devices, "autocast", lambda *a, **k: torch.autocast("cpu", enabled=False)):
            latent = UnetHook.call_vae_using_process(p, x)
            again = UnetHook.call_vae_using_process(p, x)
        self.assertEqual(encoded, [torch.bfloat16])  # the second call hits the per-process cache
        self.assertEqual(latent.dtype, torch.bfloat16)
        self.assertEqual(tuple(latent.shape), (1, 4, 2, 2))
        self.assertTrue(torch.equal(latent, again))


class TestSdxlScheduleTables(unittest.TestCase):
    """register_schedule's tables are indexed with `.to(x)` on every UNet call of reference, inpaint_only and
    colorfix units; on the CPU each call was a pageable host-to-device copy (a stream sync)."""

    def test_tables_live_on_the_sampling_device_with_the_same_values(self):
        on_cpu, on_meta = types.SimpleNamespace(), types.SimpleNamespace()
        register_schedule(on_cpu)
        with mock.patch.object(devices, "device", torch.device("meta")):
            register_schedule(on_meta)
        names = ["betas", "alphas_cumprod_prev", "sqrt_alphas_cumprod", "sqrt_one_minus_alphas_cumprod",
                 "log_one_minus_alphas_cumprod", "sqrt_recip_alphas_cumprod", "sqrt_recipm1_alphas_cumprod"]
        for name in names:
            self.assertEqual(getattr(on_meta, name).device.type, "meta")
            self.assertEqual(getattr(on_cpu, name).dtype, torch.float32)
        # Values: the scaled-linear SDXL/SD schedule, float64 numpy rounded once to float32.
        betas = np.linspace(0.00085 ** 0.5, 0.0120 ** 0.5, 1000, dtype=np.float64) ** 2
        acp = np.cumprod(1.0 - betas)
        torch.testing.assert_close(on_cpu.sqrt_recipm1_alphas_cumprod, torch.tensor(np.sqrt(1.0 / acp - 1), dtype=torch.float32), rtol=1e-6, atol=0)


TINY_SDXL_CONTROLNET = dict(
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


# The union condition transformer is 320 wide (SDXL's model_channels) whatever the config says.
TINY_SDXL_UNION_CONTROLNET = dict(
    TINY_SDXL_CONTROLNET,
    model_channels=320,
    num_res_blocks=1,
    attention_resolutions=[],
    channel_mult=[1],
    num_head_channels=64,
    transformer_depth=[1],
    union_controlnet_num_control_type=8,
)


def cpu_controlnet_env():
    return mock.patch.multiple(devices, get_device_for=lambda name: torch.device("cpu"), dtype_unet=torch.float32)


def pre_hook_count(module):
    return len(module._forward_pre_hooks)


class TestAggressiveLowVram(unittest.TestCase):
    """A low-VRAM unit calls aggressive_lowvram() before every ControlNet forward on a model that is cached
    across requests: its pre-hooks must not pile up, a later full-VRAM use must drop them, and union models
    must bring their union parts to the device too."""

    def test_hooks_are_registered_once_and_removed_by_fullvram(self):
        model = PlugableControlModel(TINY_SDXL_CONTROLNET).eval()
        cm = model.control_model
        with cpu_controlnet_env(), torch.no_grad():
            for _ in range(3):
                model.aggressive_lowvram()
            hooked = [cm.time_embed, cm.input_hint_block, cm.label_emb, cm.middle_block, cm.middle_block_out, *cm.input_blocks, *cm.zero_convs]
            self.assertTrue(all(pre_hook_count(m) == 1 for m in hooked))
            x = torch.randn(2, 4, 8, 8)
            out = model(x=x, hint=torch.rand(1, 3, 64, 64), timesteps=torch.tensor([10.0, 500.0]), context=torch.randn(2, 5, 16), y=torch.randn(2, 12))
            self.assertEqual(len(out), 10)
            model.fullvram()
            self.assertTrue(all(pre_hook_count(m) == 0 for m in hooked))
            self.assertIsNone(model.gpu_component)

    def test_union_parts_are_hooked(self):
        model = PlugableControlModel(TINY_SDXL_UNION_CONTROLNET).eval()
        with torch.no_grad():
            for p in model.parameters():
                p.normal_(0, 0.02)  # task_embedding is created with torch.empty
        cm = model.control_model
        with cpu_controlnet_env(), torch.no_grad():
            model.aggressive_lowvram()
            for m in (cm.transformer_layes, cm.spatial_ch_projs, cm.control_add_embedding):
                self.assertEqual(pre_hook_count(m), 1)
            guided = cm.compute_guided_hint(torch.rand(1, 3, 64, 64), [1])
            self.assertEqual(tuple(guided.shape), (1, 320, 8, 8))
            self.assertTrue(torch.isfinite(guided).all())


class TestModelCacheKeyCheckpointIdentity(unittest.TestCase):
    """'difference' ControlNets bake the UNet's weights in at build time. With --no-hashing the checkpoint
    sha256 is None and checkpoint switches load in place into the same UNet object, so the key must carry
    the checkpoint file's identity."""

    def test_key_changes_with_the_checkpoint_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            cn_path = os.path.join(tmp, "control.safetensors")
            ckpt_a, ckpt_b = os.path.join(tmp, "a.safetensors"), os.path.join(tmp, "b.safetensors")
            for path in (cn_path, ckpt_a, ckpt_b):
                with open(path, "wb") as f:
                    f.write(b"x")
            unet = torch.nn.Identity()

            def key(checkpoint):
                info = types.SimpleNamespace(filename=checkpoint, sha256=None)
                p = types.SimpleNamespace(sd_model=types.SimpleNamespace(sd_checkpoint_info=info, dtype=torch.bfloat16))
                with mock.patch.object(Script, "_resolve_model_path", staticmethod(lambda model: (model, cn_path))):
                    return Script._model_cache_key(p, unet, "control")

            self.assertEqual(key(ckpt_a), key(ckpt_a))
            self.assertNotEqual(key(ckpt_a), key(ckpt_b))
            before = key(ckpt_a)
            os.utime(ckpt_a, ns=(1, 1))
            self.assertNotEqual(before, key(ckpt_a))


def a1111_hires_size(width, height, hr_scale, hr_resize_x, hr_resize_y):
    """The final hires pixel size of A1111's own calculate_target_resolution (+ its latent truncation)."""
    p = StableDiffusionProcessingTxt2Img.__new__(StableDiffusionProcessingTxt2Img)
    p.width, p.height, p.hr_scale, p.hr_resize_x, p.hr_resize_y = width, height, hr_scale, hr_resize_x, hr_resize_y
    p.extra_generation_params = {}
    p.truncate_x = p.truncate_y = 0
    p.applied_old_hires_behavior_to = None
    with mock.patch("modules.processing.opts", types.SimpleNamespace(use_old_hires_fix_width_height=False)):
        StableDiffusionProcessingTxt2Img.calculate_target_resolution(p)
    return (p.hr_upscale_to_y // 8 - p.truncate_y) * 8, (p.hr_upscale_to_x // 8 - p.truncate_x) * 8


class TestHiresTargetDimensions(unittest.TestCase):
    def test_matches_a1111_hires_size(self):
        cases = [
            (1024, 768, 1.5, 0, 0),
            (1024, 768, 1.0, 1536, 0),     # width only: the height follows the aspect ratio
            (1024, 768, 1.0, 0, 1152),     # height only
            (1024, 1024, 1.0, 1536, 1024),  # both: upscaled, then truncated to the requested size
            (832, 1216, 2.0, 0, 0),
        ]
        for width, height, hr_scale, hr_resize_x, hr_resize_y in cases:
            p = StableDiffusionProcessingTxt2Img.__new__(StableDiffusionProcessingTxt2Img)
            p.width, p.height, p.enable_hr = width, height, True
            p.hr_scale, p.hr_resize_x, p.hr_resize_y = hr_scale, hr_resize_x, hr_resize_y
            h, w, hr_y, hr_x = Script.get_target_dimensions(p)
            self.assertEqual((h, w), (height, width))
            self.assertEqual((hr_y, hr_x), a1111_hires_size(width, height, hr_scale, hr_resize_x, hr_resize_y), (width, height, hr_scale, hr_resize_x, hr_resize_y))


if __name__ == "__main__":
    unittest.main()
