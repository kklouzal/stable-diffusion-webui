import ast
from pathlib import Path


def _function_from(path, name):
    tree = ast.parse(Path(path).read_text(encoding="utf8"))
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def _calls(function, name):
    """Calls of `name` in `function`, whether spelled name(...) or module.name(...)."""
    return [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id == name)
            or (isinstance(node.func, ast.Attribute) and node.func.attr == name)
        )
    ]


TRAINERS = [
    ("modules/textual_inversion/textual_inversion.py", "train_embedding"),
    ("modules/hypernetworks/hypernetwork.py", "train_hypernetwork"),
]


def test_training_previews_go_through_the_shared_preview_helper():
    for path, name in TRAINERS:
        trainer = _function_from(path, name)

        assert len(_calls(trainer, "render_training_preview")) == 1, name
        assert _calls(trainer, "save_image") == [], name
        assert _calls(trainer, "apply_txt2img_preview_params") == [], name
        assert [
            node
            for node in ast.walk(trainer)
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Attribute) and target.attr == "sampler_name" for target in node.targets)
        ] == [], name


def test_preview_helper_saves_the_image_once_and_restores_the_rng():
    helper = _function_from("modules/textual_inversion/textual_inversion.py", "render_training_preview")

    assert len(_calls(helper, "save_image")) == 1
    assert len(_calls(helper, "apply_txt2img_preview_params")) == 1

    restores = {
        node.func.attr
        for statement in ast.walk(helper)
        if isinstance(statement, ast.Try)
        for node in ast.walk(ast.Module(body=statement.finalbody, type_ignores=[]))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert {"set_rng_state", "set_rng_state_all"} <= restores
