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


def _function_from(path, name):
    tree = ast.parse(Path(path).read_text(encoding="utf8"))
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def test_txt2img_preview_params_are_shared_by_embedding_and_hypernetwork_training():
    embedding_train = _function_from("modules/textual_inversion/textual_inversion.py", "train_embedding")
    hypernetwork_train = _function_from("modules/hypernetworks/hypernetwork.py", "train_hypernetwork")

    helper_calls = [
        node
        for node in ast.walk(embedding_train)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "apply_txt2img_preview_params"
    ]
    hypernetwork_helper_calls = [
        node
        for node in ast.walk(hypernetwork_train)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "apply_txt2img_preview_params"
    ]
    direct_sampler_assignments = [
        node
        for function in [embedding_train, hypernetwork_train]
        for node in ast.walk(function)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Attribute) and target.attr == "sampler_name" for target in node.targets)
    ]

    assert len(helper_calls) == 1
    assert len(hypernetwork_helper_calls) == 1
    assert direct_sampler_assignments == []
