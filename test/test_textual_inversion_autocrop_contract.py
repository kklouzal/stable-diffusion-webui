import ast
from pathlib import Path


def test_face_detection_branch_checks_detection_count_not_array_truthiness():
    source = Path("modules/textual_inversion/autocrop.py").read_text(encoding="utf8")
    tree = ast.parse(source)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "image_face_points")

    bare_faces_checks = [
        node for node in ast.walk(function)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "faces"
    ]
    len_faces_checks = [
        node for node in ast.walk(function)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Call)
        and isinstance(node.test.left.func, ast.Name)
        and node.test.left.func.id == "len"
        and len(node.test.left.args) == 1
        and isinstance(node.test.left.args[0], ast.Name)
        and node.test.left.args[0].id == "faces"
    ]

    assert not bare_faces_checks
    assert len_faces_checks
