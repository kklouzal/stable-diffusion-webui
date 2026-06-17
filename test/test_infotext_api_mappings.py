import ast
from pathlib import Path
import types


def load_infotext_conversion_helpers():
    source = Path("modules/infotext_utils.py").read_text()
    tree = ast.parse(source)
    wanted = {
        "_indexed_infotext_value",
        "inpainting_mask_invert_from_infotext",
        "inpainting_fill_from_infotext",
        "inpaint_full_res_from_infotext",
    }
    module = ast.Module(
        body=[node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace = {}
    exec(compile(module, "modules/infotext_utils.py", "exec"), namespace)
    return namespace


def load_api_infotext_value_helper():
    source = Path("modules/api/api.py").read_text()
    tree = ast.parse(source)
    helper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "api_infotext_value_for_field"
    )
    module = ast.Module(body=[helper], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {}
    exec(compile(module, "modules/api/api.py", "exec"), namespace)
    return namespace[helper.name]


def test_inpaint_infotext_labels_convert_to_api_values():
    helpers = load_infotext_conversion_helpers()
    params = {
        "Mask mode": "Inpaint not masked",
        "Masked content": "latent noise",
        "Inpaint area": "Only masked",
    }

    assert helpers["inpainting_mask_invert_from_infotext"](params) == 1
    assert helpers["inpainting_fill_from_infotext"](params) == 2
    assert helpers["inpaint_full_res_from_infotext"](params) is True


def test_inpaint_infotext_default_labels_convert_to_api_values():
    helpers = load_infotext_conversion_helpers()
    params = {
        "Mask mode": "Inpaint masked",
        "Masked content": "original",
        "Inpaint area": "Whole picture",
    }

    assert helpers["inpainting_mask_invert_from_infotext"](params) == 0
    assert helpers["inpainting_fill_from_infotext"](params) == 1
    assert helpers["inpaint_full_res_from_infotext"](params) is False


def test_unknown_inpaint_infotext_labels_are_ignored():
    helpers = load_infotext_conversion_helpers()
    params = {
        "Mask mode": "unexpected",
        "Masked content": "unexpected",
        "Inpaint area": "unexpected",
    }

    assert helpers["inpainting_mask_invert_from_infotext"](params) is None
    assert helpers["inpainting_fill_from_infotext"](params) is None
    assert helpers["inpaint_full_res_from_infotext"](params) is None


def test_api_script_infotext_unwraps_gradio_update_values():
    convert = load_api_infotext_value_helper()
    field = types.SimpleNamespace(
        label=None,
        function=lambda _params: {"__type__": "generic_update", "value": True},
    )

    assert convert(field, {}, bool) is True


def test_api_script_infotext_uses_update_value_type_for_none_defaults():
    convert = load_api_infotext_value_helper()
    field = types.SimpleNamespace(
        label=None,
        function=lambda _params: {"__type__": "generic_update", "value": "Clamp-Cosine (c=2.0)"},
    )

    assert convert(field, {}, type(None)) == "Clamp-Cosine (c=2.0)"


def test_api_script_infotext_coerces_bool_strings_without_truthiness_bug():
    convert = load_api_infotext_value_helper()
    field = types.SimpleNamespace(label="CFG Interval Enable", function=None)

    assert convert(field, {"CFG Interval Enable": "False"}, bool) is False
    assert convert(field, {"CFG Interval Enable": "True"}, bool) is True
