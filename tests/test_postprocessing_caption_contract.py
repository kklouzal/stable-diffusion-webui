import ast
from pathlib import Path


def load_caption_helpers():
    module = ast.parse(Path("modules/postprocessing.py").read_text())
    functions = [
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name in {"combine_caption", "save_caption_sidecar"}
    ]
    code = compile(ast.Module(body=functions, type_ignores=[]), "modules/postprocessing.py", "exec")
    namespace = {"os": __import__("os")}
    exec(code, namespace)
    return namespace


def load_combine_caption():
    return load_caption_helpers()["combine_caption"]


def test_combine_caption_actions_with_existing_caption():
    combine_caption = load_combine_caption()

    assert combine_caption("old", "new", "Prepend") == "new old"
    assert combine_caption("old", "new", "Append") == "old new"
    assert combine_caption("old", "new", "Keep") == "old"
    assert combine_caption("old", "new", "Ignore") == "new"


def test_combine_caption_actions_without_existing_caption():
    combine_caption = load_combine_caption()

    assert combine_caption("", "new", "Prepend") == "new"
    assert combine_caption("", "new", "Append") == "new"
    assert combine_caption("", "new", "Keep") == "new"


def test_save_caption_sidecar_combines_existing_caption(tmp_path):
    save_caption_sidecar = load_caption_helpers()["save_caption_sidecar"]
    image_path = tmp_path / "result.png"
    caption_path = tmp_path / "result.txt"
    caption_path.write_text("old\n", encoding="utf8")

    save_caption_sidecar(str(image_path), "new", "Append")

    assert caption_path.read_text(encoding="utf8") == "old new"


def test_save_caption_sidecar_skips_empty_combined_caption(tmp_path):
    save_caption_sidecar = load_caption_helpers()["save_caption_sidecar"]
    image_path = tmp_path / "result.png"

    save_caption_sidecar(str(image_path), "   ", "Ignore")

    assert not (tmp_path / "result.txt").exists()
