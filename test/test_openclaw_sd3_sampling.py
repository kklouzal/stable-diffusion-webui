import unittest

import torch

from test.helpers import available_test_device, load_source, module


def load_sd3_impls_module():
    mmdit = type("MMDiT", (torch.nn.Module,), {"__init__": lambda self, *args, **kwargs: super(type(self), self).__init__()})
    return load_source("test_sd3_impls_module", "modules/models/sd3/sd3_impls.py", {
        "modules": module("modules"),
        "modules.models": module("modules.models"),
        "modules.models.sd3": module("modules.models.sd3"),
        "modules.models.sd3.mmdit": module("modules.models.sd3.mmdit", MMDiT=mmdit),
    })


class OpenClawSD3SamplingTests(unittest.TestCase):
    def test_sampling_sigmas_are_created_on_requested_device(self):
        sd3_impls = load_sd3_impls_module()
        device = available_test_device()

        sampling = sd3_impls.ModelSamplingDiscreteFlow(device=device)

        self.assertEqual(sampling.sigmas.device.type, device.type)
        self.assertEqual(sampling.sigmas.dtype, torch.float32)
        self.assertEqual(tuple(sampling.sigmas.shape), (1000,))

    def test_preview_factors_are_created_on_requested_device(self):
        sd3_impls = load_sd3_impls_module()
        device = available_test_device()
        dtype = torch.float16 if device.type == "cuda" else torch.float32

        factors = sd3_impls._sd3_preview_factors(device, dtype)

        self.assertEqual(factors.device.type, device.type)
        self.assertEqual(factors.dtype, dtype)
        self.assertEqual(tuple(factors.shape), (16, 3))

    def test_preview_decode_keeps_projection_device_side_until_final_image(self):
        sd3_impls = load_sd3_impls_module()
        device = available_test_device()
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        latent = torch.zeros((1, 16, 2, 3), device=device, dtype=dtype)

        preview = sd3_impls.SD3LatentFormat().decode_latent_to_preview(latent)

        self.assertEqual(preview.mode, "RGB")
        self.assertEqual(preview.size, (3, 2))
        self.assertEqual(preview.getpixel((0, 0)), (127, 127, 127))


if __name__ == "__main__":
    unittest.main()
