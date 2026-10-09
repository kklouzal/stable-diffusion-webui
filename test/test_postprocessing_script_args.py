import types

from modules import headless_ui as gr
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



def test_blend_with_original_resizes_and_converts_processed_image():
    from PIL import Image

    original = Image.new("RGBA", (2, 2), (10, 20, 30, 255))
    processed = Image.new("RGB", (1, 1), (110, 120, 130))

    blended = scripts_postprocessing.blend_with_original(original, processed, 0.5)

    assert blended.size == original.size
    assert blended.mode == original.mode
    assert blended.getpixel((0, 0)) == (60, 70, 80, 255)


def test_blend_with_original_keeps_the_alpha_of_an_rgba_original():
    from PIL import Image

    # A face restorer's RGB result over an RGBA original: half transparent, half opaque.
    alpha = Image.new("L", (4, 2), 0)
    alpha.paste(255, (2, 0, 4, 2))
    original = Image.new("RGB", (4, 2), (10, 20, 30))
    original.putalpha(alpha)
    restored = Image.new("RGB", (4, 2), (110, 120, 130))

    for visibility, color in ((1.0, (110, 120, 130)), (0.5, (60, 70, 80))):
        blended = scripts_postprocessing.blend_with_original(original, restored, visibility)
        assert blended.mode == "RGBA"
        assert blended.getpixel((0, 0)) == (*color, 0) and blended.getpixel((3, 1)) == (*color, 255)


def test_blend_with_original_returns_an_rgb_result_at_full_visibility_unchanged():
    from PIL import Image

    original = Image.new("RGB", (2, 2), (10, 20, 30))
    restored = Image.new("RGB", (2, 2), (110, 120, 130))

    assert scripts_postprocessing.blend_with_original(original, restored, 1.0) is restored


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
