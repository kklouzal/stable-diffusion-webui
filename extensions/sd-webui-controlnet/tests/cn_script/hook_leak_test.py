import importlib
import types
import unittest
from unittest import mock

import torch

utils = importlib.import_module("extensions.sd-webui-controlnet.tests.utils", "utils")

from modules import devices, script_callbacks  # noqa: E402
from scripts.controlnet import Script  # noqa: E402
from scripts.enums import ControlModelType  # noqa: E402
from scripts.hook import ControlParams, UnetHook  # noqa: E402


class FakeUNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, x, timesteps=None, context=None, y=None, **kwargs):
        self.calls.append(x)
        return x


def fake_processing(unet, body):
    sd_model = types.SimpleNamespace(model=types.SimpleNamespace(diffusion_model=unet), is_sdxl=False)
    return types.SimpleNamespace(sd_model=sd_model, sample=lambda *args, **kwargs: body())


def control_param():
    return ControlParams(
        control_model=None, preprocessor={"name": "none"}, hint_cond=torch.zeros(1, 3, 8, 8),
        weight=1.0, guidance_stopped=False, start_guidance_percent=0.0, stop_guidance_percent=1.0,
        advanced_weighting=None, control_model_type=ControlModelType.ControlNet, hr_hint_cond=None,
        global_average_pooling=False, soft_injection=False, cfg_injection=False,
    )


def denoiser_callback_registered(handler):
    return any(cb.callback == handler for cb in script_callbacks.callback_map["callbacks_cfg_denoiser"])


class TestHookLeakAfterFailedGeneration(unittest.TestCase):
    """A generation that raises inside p.sample (e.g. the NaN check) never runs
    Script.postprocess, and txt2img/img2img use different Script instances."""

    def setUp(self):
        self.unet = FakeUNet()
        self.baseline = self.unet.forward
        self.img2img, self.txt2img = Script(), Script()

        def failing_sample():
            raise devices.NansException("A tensor with NaNs was produced in Unet.")

        p = fake_processing(self.unet, failing_sample)
        self.img2img.latest_network = UnetHook()
        self.img2img.latest_network.hook(model=self.unet, sd_ldm=p.sd_model, control_params=[control_param()], process=p)
        self.leaked = self.img2img.latest_network
        with self.assertRaises(devices.NansException):
            p.sample(conditioning=[], unconditional_conditioning=[])

    def tearDown(self):
        UnetHook.restore_leaked(self.unet)

    def assert_unet_healed(self):
        self.assertEqual(self.unet.forward, self.baseline)
        for attr in ("_controlnet_forward_hook_owner", "_controlnet_forward_hook_wrapper",
                     "_controlnet_forward_hook_baseline", "_controlnet_forward_hook_restore"):
            self.assertFalse(hasattr(self.unet, attr), attr)
        self.assertFalse(denoiser_callback_registered(self.leaked.guidance_schedule_handler))

    def test_leaked_hook_passes_through_until_healed(self):
        self.assertTrue(denoiser_callback_registered(self.leaked.guidance_schedule_handler))
        # Stale control (hint sized for another request) is never applied.
        self.assertEqual(self.unet.forward("latent", timesteps="t", context=None), "latent")
        self.assertEqual(self.unet.calls, ["latent"])

    def test_request_without_controlnet_on_other_instance_heals_unet(self):
        self.assertIsNone(self.txt2img.latest_network)
        p = fake_processing(self.unet, lambda: None)
        with mock.patch.object(Script, "get_enabled_units", return_value=[]):
            self.txt2img.controlnet_main_entry(p)
        self.assert_unet_healed()
        self.assertIsNone(self.leaked.control_params)

    def test_request_with_controlnet_on_other_instance_hooks_cleanly(self):
        p = fake_processing(self.unet, lambda: None)
        with mock.patch.object(Script, "get_enabled_units", return_value=[]):
            self.txt2img.controlnet_main_entry(p)
        fresh = UnetHook()
        # controlnet_main_entry ends by hooking; before the fix this raised
        # "ControlNet UNet forward hook is owned by another live hook".
        fresh.hook(model=self.unet, sd_ldm=p.sd_model, control_params=[control_param()], process=p)
        self.assertIs(self.unet._controlnet_forward_hook_owner, fresh._forward_hook_owner_token)
        fresh.restore()
        self.assert_unet_healed()


class TestPostprocessRelease(unittest.TestCase):
    def test_postprocess_restores_without_forcing_gc_or_cache_release(self):
        unet = FakeUNet()
        baseline = unet.forward
        script = Script()
        p = fake_processing(unet, lambda: None)
        script.latest_network = UnetHook()
        script.latest_network.hook(model=unet, sd_ldm=p.sd_model, control_params=[control_param()], process=p)
        processed = types.SimpleNamespace(images=[], extra_generation_params={})
        with mock.patch("gc.collect") as collect, mock.patch.object(devices, "torch_gc") as torch_gc:
            script.postprocess(p, processed)
        collect.assert_not_called()
        torch_gc.assert_not_called()
        self.assertIsNone(script.latest_network)
        self.assertEqual(unet.forward, baseline)


if __name__ == "__main__":
    unittest.main()
