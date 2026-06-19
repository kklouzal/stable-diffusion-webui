import ast
import os
from pathlib import Path


def load_path_is_parent():
    source = Path("modules/ui_extra_networks.py").read_text()
    tree = ast.parse(source)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "path_is_parent")
    namespace = {"os": os}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "modules/ui_extra_networks.py", "exec"), namespace)
    return namespace["path_is_parent"]


def test_path_is_parent_accepts_same_path_and_descendants(tmp_path):
    path_is_parent = load_path_is_parent()
    parent = tmp_path / "models"
    child = parent / "nested" / "preview.png"
    child.parent.mkdir(parents=True)
    child.touch()

    assert path_is_parent(parent, parent)
    assert path_is_parent(parent, child)


def test_path_is_parent_rejects_sibling_prefix_paths(tmp_path):
    path_is_parent = load_path_is_parent()
    parent = tmp_path / "models"
    sibling = tmp_path / "models-other" / "preview.png"
    parent.mkdir()
    sibling.parent.mkdir()
    sibling.touch()

    assert not path_is_parent(parent, sibling)
