import ast
from pathlib import Path


def _api_method(name):
    source = Path("modules/api/api.py").read_text(encoding="utf8")
    tree = ast.parse(source)
    api_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Api")
    return next(node for node in api_class.body if isinstance(node, ast.FunctionDef) and node.name == name)


def _return_call_names(function):
    returns = [node for node in ast.walk(function) if isinstance(node, ast.Return)]
    return {
        node.value.func.attr
        for node in returns
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute)
    }


def test_train_hypernetwork_reports_hypernetwork_and_ends_state_once():
    function = _api_method("train_hypernetwork")
    constants = {node.value for node in ast.walk(function) if isinstance(node, ast.Constant) and isinstance(node.value, str)}

    assert "train_hypernetwork" in constants
    assert "train hypernetwork complete: filename: " in constants
    assert "train hypernetwork error: " in constants
    assert "train_embedding" not in constants
    assert "train embedding complete: filename: " not in constants
    assert "train embedding error: " not in constants

    helper = _api_method("_run_training_task")
    state_end_calls = [
        node
        for node in ast.walk(helper)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "end"
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == "state"
    ]
    assert len(state_end_calls) == 1


def test_create_api_methods_use_create_response_helper():
    helper_calls = _return_call_names(_api_method("_run_create_task"))
    assert "_create_response" in helper_calls
    assert "_train_response" not in helper_calls

    for method_name in ["create_embedding", "create_hypernetwork"]:
        assert _return_call_names(_api_method(method_name)) == {"_run_create_task"}


def test_training_api_methods_use_train_response_helper():
    helper_calls = _return_call_names(_api_method("_run_training_task"))
    assert "_train_response" in helper_calls
    assert "_create_response" not in helper_calls

    for method_name in ["train_embedding", "train_hypernetwork"]:
        assert _return_call_names(_api_method(method_name)) == {"_run_training_task"}
