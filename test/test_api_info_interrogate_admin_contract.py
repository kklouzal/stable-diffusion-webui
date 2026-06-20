import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


API_PATH = Path(__file__).resolve().parents[1] / "modules/api/api.py"


def get_api_method_node(name):
    tree = ast.parse(API_PATH.read_text(encoding="utf8"))
    api_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Api")
    return next(node for node in api_class.body if isinstance(node, ast.FunctionDef) and node.name == name)


def load_api_class_with_methods(*method_names):
    tree = ast.parse(API_PATH.read_text(encoding="utf8"))
    api_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Api")
    selected = [
        node
        for node in api_class.body
        if isinstance(node, ast.FunctionDef) and node.name in set(method_names)
    ]
    module = ast.Module(
        body=[ast.ClassDef(name="FakeApi", bases=[], keywords=[], body=selected, decorator_list=[])],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace = {"models": SimpleNamespace(InterrogateRequest=object)}
    exec(compile(module, str(API_PATH), "exec"), namespace)
    return namespace["FakeApi"]


def test_interrogate_api_uses_queue_lock_helper():
    source = ast.get_source_segment(API_PATH.read_text(encoding="utf8"), get_api_method_node("interrogateapi"))

    assert "self._call_with_queue_lock(interrogate_image)" in source
    assert "with self.queue_lock" not in source


def test_interrogate_api_preserves_model_dispatch_under_queue_lock():
    api_class = load_api_class_with_methods("_call_with_queue_lock", "interrogateapi")
    api = api_class()
    calls = []

    def fake_call_with_queue_lock(func, *args, **kwargs):
        calls.append(func.__name__)
        return func(*args, **kwargs)

    api._call_with_queue_lock = fake_call_with_queue_lock
    api.interrogateapi.__globals__.update(
        decode_base64_to_image=lambda image: SimpleNamespace(convert=lambda mode: f"converted:{mode}:{image}"),
        shared=SimpleNamespace(interrogator=SimpleNamespace(interrogate=Mock(return_value="clip caption"))),
        deepbooru=SimpleNamespace(model=SimpleNamespace(tag=Mock(return_value="booru tags"))),
        models=SimpleNamespace(InterrogateResponse=lambda **kwargs: kwargs),
    )

    assert api.interrogateapi(SimpleNamespace(image="input", model="clip")) == {"caption": "clip caption"}
    assert calls == ["interrogate_image"]

    assert api.interrogateapi(SimpleNamespace(image="input", model="deepdanbooru")) == {"caption": "booru tags"}
    assert calls == ["interrogate_image", "interrogate_image"]
