import ast
from pathlib import Path


def load_combine_caption():
    module = ast.parse(Path("modules/postprocessing.py").read_text())
    function = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "combine_caption")
    code = compile(ast.Module(body=[function], type_ignores=[]), "modules/postprocessing.py", "exec")
    namespace = {}
    exec(code, namespace)
    return namespace["combine_caption"]


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
