"""IncantBaseExtensionScript dispatch to its submodules, with the submodules replaced by recorders."""
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

EXT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = EXT_ROOT / "scripts"


def _load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class _Recorder:
    """Stands in for one Incantations submodule; records hook calls and can fail postprocess_batch."""

    calls = None

    def __init__(self):
        self.fail_postprocess = None

    def setup_ui(self, is_img2img):
        return []

    def get_infotext_fields(self):
        return []

    def get_paste_field_names(self):
        return []

    def get_xyz_axis_options(self):
        return []

    def postprocess_batch(self, p, *args, **kwargs):
        type(self).calls.append(type(self).__name__)
        if self.fail_postprocess is not None:
            raise self.fail_postprocess


def load_incantation_base(calls):
    stub_names = ("SEGExtensionScript", "PAGExtensionScript", "CFGCombinerScript")
    recorders = {name: type(name, (_Recorder,), {"calls": calls}) for name in stub_names}

    modules_pkg = types.ModuleType("modules")
    scripts_mod = types.ModuleType("modules.scripts")
    scripts_mod.Script = object
    scripts_mod.AlwaysVisible = object()
    script_callbacks_mod = types.ModuleType("modules.script_callbacks")
    script_callbacks_mod.on_before_ui = lambda callback: None
    processing_mod = types.ModuleType("modules.processing")
    processing_mod.StableDiffusionProcessing = object
    headless_ui_mod = types.ModuleType("modules.headless_ui")
    for name, module in (("scripts", scripts_mod), ("script_callbacks", script_callbacks_mod), ("processing", processing_mod), ("headless_ui", headless_ui_mod)):
        setattr(modules_pkg, name, module)

    scripts_pkg = types.ModuleType("scripts")
    scripts_pkg.__path__ = []
    incant_utils_pkg = types.ModuleType("scripts.incant_utils")
    incant_utils_pkg.__path__ = []
    fake_submodules = {}
    for module_name, class_name in (("pag", "PAGExtensionScript"), ("cfg_combiner", "CFGCombinerScript"), ("smoothed_energy_guidance", "SEGExtensionScript")):
        fake = types.ModuleType(f"scripts.{module_name}")
        setattr(fake, class_name, recorders[class_name])
        fake_submodules[f"scripts.{module_name}"] = fake

    stubs = {
        "modules": modules_pkg,
        "modules.scripts": scripts_mod,
        "modules.script_callbacks": script_callbacks_mod,
        "modules.processing": processing_mod,
        "modules.headless_ui": headless_ui_mod,
        "scripts": scripts_pkg,
        "scripts.incant_utils": incant_utils_pkg,
        **fake_submodules,
    }
    # Stub A1111 and the sibling submodules only while the script imports, so other test files keep theirs.
    with mock.patch.dict(sys.modules, stubs):
        incant_utils_pkg.timing = _load_file("scripts.incant_utils.timing", SCRIPTS_DIR / "incant_utils" / "timing.py")
        scripts_pkg.ui_wrapper = _load_file("scripts.ui_wrapper", SCRIPTS_DIR / "ui_wrapper.py")
        return _load_file("incantation_base_under_test", SCRIPTS_DIR / "incantation_base.py")


class PostprocessBatchCleanupTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.base = load_incantation_base(self.calls)
        self.script = self.base.IncantBaseExtensionScript()
        self.p = types.SimpleNamespace()
        self.modules = {type(m.module).__name__: m.module for m in self.base.submodules}

    def test_submodules_run_in_declared_order(self):
        self.script.postprocess_batch(self.p, images=None, batch_number=0)

        self.assertEqual(self.calls, ["SEGExtensionScript", "PAGExtensionScript", "CFGCombinerScript"])
        self.assertEqual(
            set(self.p.openclaw_extension_timings["extensions"]),
            {"Incantations.SEGExtensionScript", "Incantations.PAGExtensionScript", "Incantations.CFGCombinerScript"},
        )

    def test_failing_cleanup_does_not_skip_later_submodules(self):
        failure = RuntimeError("SEG cleanup failed")
        self.modules["SEGExtensionScript"].fail_postprocess = failure

        with self.assertRaises(RuntimeError) as raised:
            self.script.postprocess_batch(self.p, images=None, batch_number=0)

        self.assertIs(raised.exception, failure)
        self.assertEqual(self.calls, ["SEGExtensionScript", "PAGExtensionScript", "CFGCombinerScript"])

    def test_every_cleanup_failure_is_reported(self):
        seg_failure = RuntimeError("SEG cleanup failed")
        pag_failure = ValueError("PAG cleanup failed")
        self.modules["SEGExtensionScript"].fail_postprocess = seg_failure
        self.modules["PAGExtensionScript"].fail_postprocess = pag_failure

        with self.assertRaises(ExceptionGroup) as raised:
            self.script.postprocess_batch(self.p, images=None, batch_number=0)

        self.assertEqual(list(raised.exception.exceptions), [seg_failure, pag_failure])
        self.assertEqual(self.calls, ["SEGExtensionScript", "PAGExtensionScript", "CFGCombinerScript"])


if __name__ == "__main__":
    unittest.main()
