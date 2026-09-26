import ast
import json
from pathlib import Path
from types import SimpleNamespace


def load_function(path, name, globals_dict):
    source = Path(path).read_text(encoding="utf8")
    module = ast.parse(source, filename=path)
    function = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == name)
    compiled = compile(ast.Module(body=[function], type_ignores=[]), path, "exec")
    namespace = dict(globals_dict)
    exec(compiled, namespace)
    return namespace[name]


def test_processed_js_with_image_paths_keeps_paths_aligned_to_images():
    processed_js_with_image_paths = load_function(
        "modules/api/api.py",
        "processed_js_with_image_paths",
        {"json": json},
    )

    grid = SimpleNamespace()
    saved_sample = SimpleNamespace(already_saved_as="/tmp/sample-1.png")
    presaved_sample = SimpleNamespace(already_saved_as="/tmp/sample-2.png")
    processed = SimpleNamespace(
        images=[grid, saved_sample, presaved_sample],
        js=lambda: json.dumps({
            "index_of_first_image": 1,
            "infotexts": ["grid infotext", "sample 1 infotext"],
        }),
    )

    data = json.loads(processed_js_with_image_paths(processed, {"extra_key": "extra value"}))

    assert data["image_paths"] == [None, "/tmp/sample-1.png", "/tmp/sample-2.png"]
    assert data["index_of_first_image"] == 1
    assert data["infotexts"] == ["grid infotext", "sample 1 infotext"]
    assert data["extra_key"] == "extra value"
