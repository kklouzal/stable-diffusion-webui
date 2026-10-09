"""Device-cache releases stay at job/model boundaries, not between generation phases."""

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _torch_gc_calls(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node

    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "torch_gc":
            scope = []
            guards = []
            parent = parents.get(node)
            while parent is not None:
                if isinstance(parent, (ast.FunctionDef, ast.ClassDef)):
                    scope.append(parent.name)
                elif isinstance(parent, ast.If):
                    guards.append(ast.unparse(parent.test))
                elif isinstance(parent, ast.ExceptHandler):
                    guards.append(f"except {ast.unparse(parent.type)}")
                parent = parents.get(parent)
            calls.append((".".join(reversed(scope)), guards))
    return calls


def test_processing_releases_device_cache_only_at_recovery_and_model_offload_points():
    calls = _torch_gc_calls(ROOT / "modules" / "processing.py")

    assert sorted(scope for scope, _guards in calls) == [
        "decode_latent_batch",
        "process_images_inner",
    ]
    guards = [guards for _scope, guards in calls]
    # OOM retry path of the batched VAE decode.
    assert ["except torch.cuda.OutOfMemoryError"] in guards
    # lowvram/medvram offload after decode.
    assert any(g and g[0] == "lowvram.is_enabled(shared.sd_model)" for g in guards)
    # Face restoration runs per image inside the request: the job's single release (State.end) covers it.
    assert not any("p.restore_faces" in g for g in guards)


def test_job_end_still_releases_device_cache():
    calls = _torch_gc_calls(ROOT / "modules" / "shared_state.py")

    assert ("State.end", []) in calls


def test_upscalers_and_face_restoration_leave_device_cache_release_to_the_job():
    # One release per job (State.begin/State.end): a release per tile pass or per face only stalls the request.
    for relative in (
        "modules/upscaler.py",
        "modules/upscaler_utils.py",
        "modules/face_restoration_utils.py",
        "extensions-builtin/SwinIR/scripts/swinir_model.py",
        "extensions-builtin/ScuNET/scripts/scunet_model.py",
    ):
        assert _torch_gc_calls(ROOT / relative) == [], relative
