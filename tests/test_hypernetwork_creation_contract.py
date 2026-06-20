import ast
from pathlib import Path


def test_create_hypernetwork_returns_created_filename_for_ui_message():
    source = Path("modules/hypernetworks/hypernetwork.py").read_text(encoding="utf8")
    tree = ast.parse(source)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "create_hypernetwork")

    return_names = [node.value.id for node in ast.walk(function) if isinstance(node, ast.Return) and isinstance(node.value, ast.Name)]

    assert "fn" in return_names
