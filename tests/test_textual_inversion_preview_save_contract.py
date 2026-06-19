import ast
from pathlib import Path


def test_train_embedding_preview_branch_saves_generated_image_once():
    source = Path("modules/textual_inversion/textual_inversion.py").read_text(encoding="utf8")
    tree = ast.parse(source)
    train_embedding = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "train_embedding")

    save_image_calls = [
        node for node in ast.walk(train_embedding)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "save_image"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "images"
    ]

    assert len(save_image_calls) == 1
