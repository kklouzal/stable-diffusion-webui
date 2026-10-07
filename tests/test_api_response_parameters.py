import ast
import copy
import types
from pathlib import Path
from typing import Any

API_PATH = Path(__file__).resolve().parents[1] / "modules" / "api" / "api.py"


def load_response_parameters():
    """Load _response_parameters from api.py without importing the webui runtime."""
    tree = ast.parse(API_PATH.read_text(encoding="utf8"))
    wanted = {"_CONTROLNET_UNIT_IMAGE_FIELDS", "_response_parameters"}
    body = [
        node for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in wanted)
        or (isinstance(node, ast.Assign) and any(getattr(target, "id", None) in wanted for target in node.targets))
    ]
    namespace = {"Any": Any}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(API_PATH), "exec"), namespace)
    return namespace["_response_parameters"]


def make_request():
    unit = {"enabled": True, "module": "depth_zoe", "image": "BASE64-IMAGE", "mask": "BASE64-MASK", "weight": 0.69}
    return types.SimpleNamespace(
        prompt="p",
        init_images=None,
        control_net_image="BASE64-LEGACY",
        control_net_image2=None,
        control_net_weight=0.69,
        alwayson_scripts={
            "ControlNet": {"args": [unit, {"enabled": False, "image": None}]},
            "incantations": {"args": [True, 3.0]},
        },
    )


def test_controlnet_images_are_not_echoed_and_the_request_is_unchanged():
    response_parameters = load_response_parameters()
    request = make_request()
    before = copy.deepcopy(vars(request))

    params = response_parameters(request, include_images=False)

    assert params["control_net_image"] is None
    assert params["control_net_image2"] is None
    assert params["control_net_weight"] == 0.69
    units = params["alwayson_scripts"]["ControlNet"]["args"]
    assert units[0] == {"enabled": True, "module": "depth_zoe", "image": None, "mask": None, "weight": 0.69}
    assert units[1] == {"enabled": False, "image": None}
    assert params["alwayson_scripts"]["incantations"] == {"args": [True, 3.0]}
    assert vars(request) == before


def test_controlnet_images_are_echoed_when_images_are_requested():
    response_parameters = load_response_parameters()
    request = make_request()

    params = response_parameters(request, include_images=True)

    assert params == vars(request)
    assert params["alwayson_scripts"]["ControlNet"]["args"][0]["image"] == "BASE64-IMAGE"
