import ast
from pathlib import Path
from types import SimpleNamespace


def load_script_listing_api_class():
    source = Path("modules/api/api.py").read_text(encoding="utf8")
    module = ast.parse(source)
    api_class = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "Api")
    wanted = {"_script_lists", "get_scripts_list", "get_script_info"}
    methods = [node for node in api_class.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    subset = ast.Module(body=[ast.ClassDef(name="Api", bases=[], keywords=[], body=methods, decorator_list=[])], type_ignores=[])
    ast.fix_missing_locations(subset)

    namespace = {}
    exec(compile(subset, "<api-script-listing>", "exec"), namespace)
    return namespace["Api"]


def test_script_listing_endpoints_share_txt2img_img2img_sources(monkeypatch):
    api_class = load_script_listing_api_class()

    txt_info = object()
    img_info = object()
    txt2img_scripts = [
        SimpleNamespace(name="Selectable", api_info=txt_info),
        SimpleNamespace(name=None, api_info=None),
    ]
    img2img_scripts = [SimpleNamespace(name="Img Script", api_info=img_info)]

    scripts_stub = SimpleNamespace(
        scripts_txt2img=SimpleNamespace(scripts=txt2img_scripts),
        scripts_img2img=SimpleNamespace(scripts=img2img_scripts),
    )
    scripts_list_calls = []

    class ScriptsList(SimpleNamespace):
        def __init__(self, **kwargs):
            scripts_list_calls.append(kwargs)
            super().__init__(**kwargs)

    monkeypatch.setitem(api_class._script_lists.__globals__, "scripts", scripts_stub)
    monkeypatch.setitem(api_class.get_scripts_list.__globals__, "models", SimpleNamespace(ScriptsList=ScriptsList))

    api = api_class()

    scripts_list = api.get_scripts_list()
    assert scripts_list.txt2img == ["Selectable"]
    assert scripts_list.img2img == ["Img Script"]
    assert scripts_list_calls == [{"txt2img": ["Selectable"], "img2img": ["Img Script"]}]
    assert api.get_script_info() == [txt_info, img_info]

def load_latent_upscale_api_class():
    source = Path("modules/api/api.py").read_text(encoding="utf8")
    module = ast.parse(source)
    api_class = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "Api")
    method = next(node for node in api_class.body if isinstance(node, ast.FunctionDef) and node.name == "get_latent_upscale_modes")
    subset = ast.Module(body=[ast.ClassDef(name="Api", bases=[], keywords=[], body=[method], decorator_list=[])], type_ignores=[])
    ast.fix_missing_locations(subset)

    namespace = {}
    exec(compile(subset, "<api-latent-upscale-modes>", "exec"), namespace)
    return namespace["Api"]


def test_latent_upscale_modes_lists_shared_mode_names_in_order(monkeypatch):
    api_class = load_latent_upscale_api_class()
    shared_stub = SimpleNamespace(
        latent_upscale_modes={
            "Latent": {"mode": "bilinear", "antialias": False},
            "Latent (nearest)": {"mode": "nearest", "antialias": False},
        }
    )

    monkeypatch.setitem(api_class.get_latent_upscale_modes.__globals__, "shared", shared_stub)

    assert api_class().get_latent_upscale_modes() == [
        {"name": "Latent"},
        {"name": "Latent (nearest)"},
    ]
