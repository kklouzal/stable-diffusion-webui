import ast
from pathlib import Path


def api_method_source(name):
    source = Path("modules/api/api.py").read_text(encoding="utf-8")
    api_class = next(node for node in ast.parse(source).body if isinstance(node, ast.ClassDef) and node.name == "Api")
    method = next(node for node in api_class.body if isinstance(node, ast.FunctionDef) and node.name == name)
    return ast.get_source_segment(source, method)


def test_img2img_decodes_every_input_image_before_queueing_the_task():
    # Invalid init images or masks must be rejected before a task id enters progress.pending_tasks.
    body = api_method_source("img2imgapi")
    queued = body.index("add_task_to_queue(task_id)")
    assert body.index("mask = decode_base64_to_image(mask)") < queued
    assert body.index("decoded_init_images = [decode_base64_to_image(x) for x in init_images]") < queued


def test_generation_routes_clear_unfinished_tasks_in_finally():
    for name in ("text2imgapi", "img2imgapi"):
        body = api_method_source(name)
        finally_block = body[body.rindex("finally:"):]  # outermost cleanup block
        assert "self._clear_pending_task_unless_finished(task_id, task_finished)" in finally_block
