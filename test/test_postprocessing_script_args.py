import sys
import types

from modules import headless_ui as gr

sys.modules.setdefault("modules.errors", types.SimpleNamespace(display=lambda *args, **kwargs: None))
sys.modules.setdefault(
    "modules.shared",
    types.SimpleNamespace(
        opts=types.SimpleNamespace(
            postprocessing_operation_order=[],
            postprocessing_disable_in_extras=[],
        )
    ),
)

from modules import scripts_postprocessing


class DummyPostprocessingScript(scripts_postprocessing.ScriptPostprocessing):
    name = "Dummy"

    def ui(self):
        return {
            "enabled": gr.Checkbox(value=True),
            "amount": gr.Slider(value=0.75),
            "mode": gr.Dropdown(value="Scale by"),
        }


class FilteredPostprocessingScript(scripts_postprocessing.ScriptPostprocessing):
    name = "Filtered"

    def ui(self):
        return {"value": gr.Number(value=3)}

class OtherPostprocessingScript(scripts_postprocessing.ScriptPostprocessing):
    name = "Other"

    def ui(self):
        return {"value": gr.Number(value=7)}

    def process(self, pp, value):
        pp.info.setdefault("order", []).append((self.name, value))


def make_runner(monkeypatch, script_classes, *, disabled=()):
    monkeypatch.setattr(
        scripts_postprocessing.shared,
        "opts",
        types.SimpleNamespace(
            postprocessing_operation_order=[],
            postprocessing_disable_in_extras=list(disabled),
        ),
    )
    runner = scripts_postprocessing.ScriptPostprocessingRunner()
    runner.initialize_scripts([
        types.SimpleNamespace(script_class=script_class, path=f"{script_class.name}.py")
        for script_class in script_classes
    ])
    return runner


def test_create_args_for_run_preserves_ui_defaults_for_omitted_keys(monkeypatch):
    runner = make_runner(monkeypatch, [DummyPostprocessingScript])

    args = runner.create_args_for_run({"Dummy": {"amount": 0.25}})

    script = runner.scripts[0]
    assert args[script.args_from:script.args_to] == [True, 0.25, "Scale by"]


def test_create_args_for_run_returns_empty_args_when_all_scripts_filtered(monkeypatch):
    runner = make_runner(monkeypatch, [FilteredPostprocessingScript], disabled=["Filtered"])

    assert runner.create_args_for_run({}) == []


def test_create_args_for_run_uses_explicit_script_order(monkeypatch):
    runner = make_runner(monkeypatch, [DummyPostprocessingScript, OtherPostprocessingScript])

    args = runner.create_args_for_run({}, scripts_order=["Other", "Dummy"])

    other_script, dummy_script = runner.scripts_in_preferred_order(["Other", "Dummy"])
    assert other_script.name == "Other"
    assert dummy_script.name == "Dummy"
    assert args[other_script.args_from:other_script.args_to] == [7]
    assert args[dummy_script.args_from:dummy_script.args_to] == [True, 0.75, "Scale by"]


def test_run_uses_explicit_script_order(monkeypatch):
    runner = make_runner(monkeypatch, [DummyPostprocessingScript, OtherPostprocessingScript])
    args = runner.create_args_for_run({}, scripts_order=["Other", "Dummy"])

    def record_dummy(self, pp, enabled, amount, mode):
        pp.info.setdefault("order", []).append((self.name, enabled, amount, mode))

    monkeypatch.setattr(DummyPostprocessingScript, "process", record_dummy)
    monkeypatch.setattr(
        scripts_postprocessing.shared,
        "state",
        types.SimpleNamespace(skipped=False, job=None),
        raising=False,
    )
    pp = scripts_postprocessing.PostprocessedImage(image=object())

    runner.run(pp, args, scripts_order=["Other", "Dummy"])

    assert pp.info["order"] == [("Other", 7), ("Dummy", True, 0.75, "Scale by")]
