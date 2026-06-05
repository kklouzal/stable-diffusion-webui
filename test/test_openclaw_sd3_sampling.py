import importlib.util
import os
import sys
import types
import unittest

import torch


sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def load_sd3_impls_module():
    module_name = "test_sd3_impls_module"
    originals = {}

    def put(name, module):
        originals[name] = sys.modules.get(name)
        sys.modules[name] = module

    modules_pkg = types.ModuleType("modules")
    models_pkg = types.ModuleType("modules.models")
    sd3_pkg = types.ModuleType("modules.models.sd3")
    mmdit_module = types.ModuleType("modules.models.sd3.mmdit")
    mmdit_module.MMDiT = type("MMDiT", (torch.nn.Module,), {"__init__": lambda self, *args, **kwargs: super(type(self), self).__init__()})

    for name, module in (
        ("modules", modules_pkg),
        ("modules.models", models_pkg),
        ("modules.models.sd3", sd3_pkg),
        ("modules.models.sd3.mmdit", mmdit_module),
    ):
        put(name, module)

    try:
        spec = importlib.util.spec_from_file_location(module_name, "modules/models/sd3/sd3_impls.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
        for name, original in originals.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original

    return module


def available_device():
    if torch.cuda.is_available():
        try:
            torch.empty((), device="cuda:0")
            return torch.device("cuda:0")
        except Exception:
            pass
    return torch.device("cpu")


class OpenClawSD3SamplingTests(unittest.TestCase):
    def test_sampling_sigmas_are_created_on_requested_device(self):
        sd3_impls = load_sd3_impls_module()
        device = available_device()

        sampling = sd3_impls.ModelSamplingDiscreteFlow(device=device)

        self.assertEqual(sampling.sigmas.device.type, device.type)
        self.assertEqual(sampling.sigmas.dtype, torch.float32)
        self.assertEqual(tuple(sampling.sigmas.shape), (1000,))


if __name__ == "__main__":
    unittest.main()
