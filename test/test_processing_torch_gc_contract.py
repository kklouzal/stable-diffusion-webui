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
        "process_images_inner",
        "process_images_inner",
    ]
    guards = [guards for _scope, guards in calls]
    # OOM retry path of the batched VAE decode.
    assert ["except torch.cuda.OutOfMemoryError"] in guards
    # lowvram/medvram offload after decode.
    assert any(g and g[0] == "lowvram.is_enabled(shared.sd_model)" for g in guards)
    # Face restoration loads its own models.
    assert sum(1 for g in guards if g and g[0] == "p.restore_faces") == 2


def test_job_end_still_releases_device_cache():
    calls = _torch_gc_calls(ROOT / "modules" / "shared_state.py")

    assert ("State.end", []) in calls
