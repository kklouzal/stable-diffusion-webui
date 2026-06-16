from __future__ import annotations

import ast
import importlib
import sys
import types
from pathlib import Path
import unittest

import torch

EXT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXT_ROOT))


def install_a1111_stubs() -> None:
    modules_pkg = types.ModuleType("modules")
    headless_ui_mod = types.ModuleType("modules.headless_ui")
    scripts_mod = types.ModuleType("modules.scripts")
    scripts_mod.Script = object
    scripts_mod.AlwaysVisible = object()
    sd_samplers_common_mod = types.ModuleType("modules.sd_samplers_common")
    sd_samplers_common_mod.setup_img2img_steps = lambda _p, steps=None: (steps, steps)
    sd_samplers_kdiffusion_mod = types.ModuleType("modules.sd_samplers_kdiffusion")

    class KDiffusionSampler:
        def get_sigmas(self, _p, _steps):
            return torch.ones(1)

        def sample_img2img(self, *args, **kwargs):
            return args, kwargs

    sd_samplers_kdiffusion_mod.KDiffusionSampler = KDiffusionSampler

    modules_pkg.headless_ui = headless_ui_mod
    modules_pkg.scripts = scripts_mod
    modules_pkg.sd_samplers_common = sd_samplers_common_mod
    modules_pkg.sd_samplers_kdiffusion = sd_samplers_kdiffusion_mod
    sys.modules.update(
        {
            "modules": modules_pkg,
            "modules.headless_ui": headless_ui_mod,
            "modules.scripts": scripts_mod,
            "modules.sd_samplers_common": sd_samplers_common_mod,
            "modules.sd_samplers_kdiffusion": sd_samplers_kdiffusion_mod,
        }
    )


def load_api_denoise_ramp_persist_helper():
    api_path = EXT_ROOT.parents[1] / "modules" / "api" / "api.py"
    tree = ast.parse(api_path.read_text(), filename=str(api_path))
    api_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Api")
    helper = next(
        node
        for node in api_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "persist_openclaw_denoise_ramp_args"
    )
    module = ast.Module(body=[helper], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {}
    exec(compile(module, str(api_path), "exec"), namespace)
    return namespace[helper.name]


class DenoiseRampTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        install_a1111_stubs()
        cls.ramp = importlib.import_module("scripts.openclaw_denoise_ramp")

    def test_negative_delta_preserves_tail_endpoints_and_advances_interior(self):
        sigmas = torch.linspace(10.0, 0.0, 11)
        p = types.SimpleNamespace(openclaw_denoise_step_delta=-0.1, extra_generation_params={})

        ramped = self.ramp.ramp_sigmas_for_img2img(p, sigmas, steps=10, t_enc=6)

        start = 10 - 6 - 1
        torch.testing.assert_close(ramped[:start], sigmas[:start])
        torch.testing.assert_close(ramped[start], sigmas[start])
        torch.testing.assert_close(ramped[-1], sigmas[-1])
        self.assertLess(ramped[start + 1], sigmas[start + 1])
        self.assertLess(ramped[start + 2], sigmas[start + 2])
        self.assertEqual(p.extra_generation_params["Denoise step delta"], "-0.100")
        self.assertEqual(p.extra_generation_params["Denoise ramp gamma"], "0.500")

    def test_zero_delta_returns_original_schedule_without_metadata(self):
        sigmas = torch.linspace(5.0, 0.0, 6)
        p = types.SimpleNamespace(openclaw_denoise_step_delta=0.0, extra_generation_params={})

        ramped = self.ramp.ramp_sigmas_for_img2img(p, sigmas, steps=5, t_enc=4)

        self.assertIs(ramped, sigmas)
        self.assertEqual(p.extra_generation_params, {})

    def test_api_default_persistence_keeps_selected_delta_until_changed(self):
        persist = load_api_denoise_ramp_persist_helper()

        script = types.SimpleNamespace(args_from=3, title=lambda: "OpenClaw Denoise Ramp")
        defaults = [None, None, None, 0.0]

        persist(defaults, script, [0.075])
        self.assertEqual(defaults[3], 0.075)

        next_request_args = defaults.copy()
        self.assertEqual(next_request_args[3], 0.075)

        persist(defaults, script, [0.0])
        self.assertEqual(defaults[3], 0.0)


if __name__ == "__main__":
    unittest.main()
