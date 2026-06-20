import ast
from pathlib import Path


def _api_method(name):
    source = Path("modules/api/api.py").read_text(encoding="utf8")
    tree = ast.parse(source)
    api_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Api")
    return next(node for node in api_class.body if isinstance(node, ast.FunctionDef) and node.name == name)


def test_train_hypernetwork_reports_hypernetwork_and_ends_state_once():
    function = _api_method("train_hypernetwork")
    constants = {node.value for node in ast.walk(function) if isinstance(node, ast.Constant) and isinstance(node.value, str)}
    state_end_calls = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "end"
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == "state"
    ]

    assert "train hypernetwork complete: filename: " in constants
    assert "train hypernetwork error: " in constants
    assert "train embedding complete: filename: " not in constants
    assert "train embedding error: " not in constants
    assert len(state_end_calls) == 1


def test_create_api_methods_return_create_response_on_assertion_errors():
    for method_name in ["create_embedding", "create_hypernetwork"]:
        function = _api_method(method_name)
        returns = [node for node in ast.walk(function) if isinstance(node, ast.Return)]
        call_names = {
            node.value.func.attr
            for node in returns
            if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute)
        }

        assert "CreateResponse" in call_names
        assert "TrainResponse" not in call_names
