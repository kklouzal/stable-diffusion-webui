import ast
from pathlib import Path
import types


def load_api_script_default_helpers():
    source = Path("modules/api/api.py").read_text()
    tree = ast.parse(source)
    wanted = {"script_default_ui_values", "_set_script_arg"}
    module = ast.Module(
        body=[node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace = {}
    exec(compile(module, "modules/api/api.py", "exec"), namespace)
    return namespace


def test_script_default_ui_values_prefers_finalized_controls_without_rebuilding_ui():
    helper = load_api_script_default_helpers()["script_default_ui_values"]
    script = types.SimpleNamespace(
        controls=[types.SimpleNamespace(value="ui-default")],
        is_img2img=False,
        ui=lambda _is_img2img: (_ for _ in ()).throw(AssertionError("ui() should not be rebuilt")),
    )

    assert helper(script) == ["ui-default"]


def test_script_default_ui_values_falls_back_for_scripts_without_controls():
    helper = load_api_script_default_helpers()["script_default_ui_values"]
    script = types.SimpleNamespace(
        controls=None,
        is_img2img=True,
        ui=lambda is_img2img: [types.SimpleNamespace(value=f"fallback-{is_img2img}")],
    )

    assert helper(script) == ["fallback-True"]


def test_set_script_arg_updates_existing_index():
    helper = load_api_script_default_helpers()["_set_script_arg"]
    script_args = [0, "old", "keep"]

    helper(script_args, 1, "new")

    assert script_args == [0, "new", "keep"]


def test_set_script_arg_extends_sparse_api_vectors():
    helper = load_api_script_default_helpers()["_set_script_arg"]
    script_args = [0]

    helper(script_args, 3, "value")
