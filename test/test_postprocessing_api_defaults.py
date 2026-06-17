import ast
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace


def load_run_extras():
    source = Path("modules/postprocessing.py").read_text()
    tree = ast.parse(source)
    module = ast.Module(
        body=[node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_extras"],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace = {}
    exec(compile(module, "modules/postprocessing.py", "exec"), namespace)
    return namespace["run_extras"]


def test_postprocessing_runner_order_override_preserves_script_defaults(monkeypatch):
    from modules import scripts_postprocessing

    calls = []

    def make_script(name, order, args_from, default):
        script = SimpleNamespace(
            name=name,
            order=order,
            main_ui_only=False,
            controls=OrderedDict([("value", SimpleNamespace(value=default))]),
            args_from=args_from,
            args_to=args_from + 1,
        )
        script.process_firstpass = lambda _pp, **_kwargs: calls.append(("first", name))
        script.process = lambda _pp, **_kwargs: calls.append(("process", name))
        return script

    monkeypatch.setattr(
        scripts_postprocessing.shared,
        "opts",
        SimpleNamespace(postprocessing_operation_order=[], postprocessing_disable_in_extras=[]),
    )
    monkeypatch.setattr(scripts_postprocessing.shared, "state", SimpleNamespace(skipped=False, job=None), raising=False)

    runner = scripts_postprocessing.ScriptPostprocessingRunner()
    runner.scripts = [
        make_script("Upscale", 1000, 0, "upscale-default"),
        make_script("GFPGAN", 2000, 1, "gfpgan-default"),
    ]
    runner.ui_created = True

    args = runner.create_args_for_run({"GFPGAN": {"value": "requested"}}, scripts_order=["GFPGAN", "Upscale"])
    assert args == ["upscale-default", "requested"]

    pp = scripts_postprocessing.PostprocessedImage(SimpleNamespace())
    runner.run(pp, args, scripts_order=["GFPGAN", "Upscale"])

    assert calls == [
        ("first", "GFPGAN"),
        ("first", "Upscale"),
        ("process", "GFPGAN"),
        ("process", "Upscale"),
    ]


def test_run_extras_maps_upscale_first_to_postprocessing_order():
    run_extras = load_run_extras()
    observed = []

    class FakeRunner:
        def create_args_for_run(self, scripts_args, scripts_order=None):
            observed.append(("create", scripts_order, scripts_args))
            return ["arg"]

    def fake_run_postprocessing(*args, save_output=True, scripts_order=None):
        observed.append(("run", scripts_order, save_output, args[-1]))
        return "result"

    run_extras.__globals__["scripts"] = SimpleNamespace(scripts_postproc=FakeRunner())
    run_extras.__globals__["run_postprocessing"] = fake_run_postprocessing

    common_args = dict(
        extras_mode=0,
        resize_mode=0,
        image="image",
        image_folder="",
        input_dir="",
        output_dir="",
        show_extras_results=True,
        gfpgan_visibility=0.5,
        codeformer_visibility=0.25,
        codeformer_weight=0.75,
        upscaling_resize=2,
        upscaling_resize_w=512,
        upscaling_resize_h=512,
        upscaling_crop=True,
        extras_upscaler_1="Lanczos",
        extras_upscaler_2="None",
        extras_upscaler_2_visibility=0,
    )

    assert run_extras(**common_args, upscale_first=False) == "result"
    assert observed[0][1] == ["GFPGAN", "CodeFormer", "Upscale"]
    assert observed[1][1] == ["GFPGAN", "CodeFormer", "Upscale"]

    observed.clear()
    assert run_extras(**common_args, upscale_first=True) == "result"
    assert observed[0][1] == ["Upscale", "GFPGAN", "CodeFormer"]
    assert observed[1][1] == ["Upscale", "GFPGAN", "CodeFormer"]


def load_set_upscalers():
    source = Path("modules/api/api.py").read_text()
    tree = ast.parse(source)
    module = ast.Module(
        body=[node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "setUpscalers"],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace = {}
    exec(compile(module, "modules/api/api.py", "exec"), namespace)
    return namespace["setUpscalers"]


def test_api_extras_always_returns_images_despite_directory_gallery_toggle():
    set_upscalers = load_set_upscalers()
    req = SimpleNamespace(
        show_extras_results=False,
        upscaler_1="None",
        upscaler_2="None",
    )

    result = set_upscalers(req)

    assert result["show_extras_results"] is True
    assert result["extras_upscaler_1"] == "None"
    assert result["extras_upscaler_2"] == "None"
    assert "upscaler_1" not in result
    assert "upscaler_2" not in result
