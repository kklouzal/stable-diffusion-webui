import ast
from pathlib import Path


def api_method_source(name):
    source = Path("modules/api/api.py").read_text(encoding="utf-8")
    api_class = next(node for node in ast.parse(source).body if isinstance(node, ast.ClassDef) and node.name == "Api")
    method = next(node for node in api_class.body if isinstance(node, ast.FunctionDef) and node.name == name)
    return ast.get_source_segment(source, method)


def test_img2img_decodes_every_input_image_before_starting_the_task():
    # Invalid init images or masks must be rejected before the request waits for queue_lock and starts its task.
    body = api_method_source("img2imgapi")
    started = body.index("self._run_generation_task(")
    assert body.index("mask = decode_base64_to_image(mask)") < started
    assert body.index("decoded_init_images = [decode_base64_to_image(x) for x in init_images]") < started
