"""Exercise the actual argument plumbing without loading the GPU runtime."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

from test_generation_last import load_generation_last, restore_modules
import tempfile


ROOT = Path(__file__).resolve().parents[1]


class RequestError(Exception):
    def __init__(self, *, status_code, detail):
        self.status_code = status_code
        super().__init__(detail)


def load_plumbing():
    source = ast.parse((ROOT / "modules/api/api.py").read_text())
    api = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "Api")
    methods = [node for node in api.body if isinstance(node, ast.FunctionDef) and node.name in (
        "init_script_args", "persist_openclaw_denoise_ramp_args")]
    helpers = [node for node in source.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in (
        "ScriptArgsList", "_set_script_arg", "_assign_script_args")]
    scripts = ast.parse((ROOT / "modules/scripts.py").read_text())
    runner = next(node for node in scripts.body if isinstance(node, ast.ClassDef) and node.name == "ScriptRunner")
    hooks = [node for node in runner.body if isinstance(node, ast.FunctionDef) and node.name in ("_script_args_for", "run")]
    tree = ast.Module(body=helpers + [
        ast.ClassDef(name="Api", bases=[], keywords=[], body=methods, decorator_list=[]),
        ast.ClassDef(name="Runner", bases=[], keywords=[], body=hooks, decorator_list=[]),
    ], type_ignores=[])
    namespace = {"shared": SimpleNamespace(total_tqdm=SimpleNamespace(clear=lambda: None)), "HTTPException": RequestError}
    exec(compile(ast.fix_missing_locations(tree), "argument_plumbing", "exec"), namespace)
    return namespace["Api"], namespace["Runner"]


def script(name, start, end, alwayson=True):
    return SimpleNamespace(title=lambda: name, args_from=start, args_to=end, alwayson=alwayson)


class ScriptArgumentRangeTests(unittest.TestCase):
    def setUp(self):
        api, self.runner = load_plumbing()
        self.api = api()
        self.a = script("A", 1, 2)
        self.b = script("B", 2, 3)
        self.api.get_script = lambda name, runner: {"A": self.a, "B": self.b}[name]

    def prepare(self, requested, selected=None, selected_args=None, defaults=None):
        defaults = [0, "default-A", "default-B"] if defaults is None else defaults
        args = self.api.init_script_args(
            SimpleNamespace(alwayson_scripts=requested, script_args=selected_args),
            defaults, selected, 0, None)
        return SimpleNamespace(script_args=args, openclaw_script_arg_ranges=args.openclaw_script_arg_ranges)

    def test_overflow_is_independent_of_request_order_and_keeps_fixed_prefix(self):
        for names in (("A", "B"), ("B", "A")):
            with self.subTest(order=names):
                values = {"A": ["A0", "A1"], "B": ["B0"]}
                defaults = [0, "default-A", "default-B"]
                p = self.prepare({name: {"args": values[name]} for name in names}, defaults=defaults)
                self.assertEqual(list(self.runner._script_args_for(p, self.a)), values["A"])
                self.assertEqual(list(self.runner._script_args_for(p, self.b)), values["B"])
                self.assertEqual(p.script_args[:3], [0, "A0", "B0"])
                self.assertEqual(defaults, [0, "default-A", "default-B"])

    def test_persisted_defaults_never_overwrite_adjacent_script(self):
        ramp = script("OpenClaw Denoise Ramp", 1, 2)
        defaults = [0, "old", "neighbor"]
        self.api.persist_openclaw_denoise_ramp_args(defaults, ramp, ["new", "overflow"])
        self.assertEqual(defaults, [0, "new", "neighbor"])

    def test_both_overflows_and_empty_default_args_remain_isolated(self):
        p = self.prepare({"A": {"args": ["A0", "A1"]}, "B": {"args": ["B0", "B1"]}})
        self.assertEqual(list(self.runner._script_args_for(p, self.a)), ["A0", "A1"])
        self.assertEqual(list(self.runner._script_args_for(p, self.b)), ["B0", "B1"])
        p = self.prepare({"A": {"args": []}})
        self.assertEqual(list(self.runner._script_args_for(p, self.a)), ["default-A"])
        self.assertEqual(p.openclaw_script_arg_ranges, {})

    def test_malformed_alwayson_args_fail_before_assignment(self):
        for value in (None, "not a list", {"bad": 1}):
            with self.subTest(value=value), self.assertRaises(RequestError) as caught:
                self.prepare({"A": {"args": value}})
            self.assertEqual(caught.exception.status_code, 422)

    def test_selectable_variable_lengths_do_not_shift_other_scripts(self):
        for values in ([], ["selected"], ["selected", "extra"]):
            with self.subTest(values=values):
                selected = script("Selected", 1, 2, False)
                received = []
                selected.run = lambda p, *args: received.append(list(args))
                p = self.prepare({"B": {"args": ["B0"]}}, selected, values)
                runner = self.runner()
                runner.selectable_scripts = [selected]
                runner.run(p, *p.script_args)
                self.assertEqual(received, [values])
                self.assertEqual(list(self.runner._script_args_for(p, self.b)), ["B0"])

    def test_snapshot_uses_the_same_ranges_as_runtime(self):
        p = self.prepare({"A": {"args": ["A0", "A1"]}, "B": {"args": ["B0"]}})
        p.scripts = SimpleNamespace(alwayson_scripts=[self.a, self.b])
        with tempfile.TemporaryDirectory() as directory:
            module, _, previous = load_generation_last(Path(directory))
            try:
                parameters, limitations = {}, []
                module._capture_script_parameters(p, parameters, limitations, {"bytes": 0})
                self.assertEqual(parameters["alwayson_scripts"], {
                    "A": {"args": ["A0", "A1"]}, "B": {"args": ["B0"]}})
                self.assertEqual(limitations, [])
            finally:
                restore_modules(previous)

    def test_snapshot_captures_selectable_overflow_without_neighbors(self):
        selected = script("Selected", 1, 2, False)
        p = self.prepare({"B": {"args": ["B0"]}}, selected, ["S0", "S1"])
        p.scripts = SimpleNamespace(alwayson_scripts=[self.b], selectable_scripts=[selected])
        with tempfile.TemporaryDirectory() as directory:
            module, _, previous = load_generation_last(Path(directory))
            try:
                parameters, limitations = {}, []
                module._capture_script_parameters(p, parameters, limitations, {"bytes": 0})
                self.assertEqual(parameters["script_args"], ["S0", "S1"])
                self.assertEqual(parameters["alwayson_scripts"]["B"]["args"], ["B0"])
                self.assertEqual(limitations, [])
            finally:
                restore_modules(previous)


if __name__ == "__main__":
    unittest.main()
