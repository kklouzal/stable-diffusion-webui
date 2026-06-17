import ast
from pathlib import Path


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
