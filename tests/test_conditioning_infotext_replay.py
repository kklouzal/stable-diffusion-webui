import ast
from pathlib import Path

PROCESSING_PATH = Path(__file__).resolve().parents[1] / "modules" / "processing.py"


def load_replay():
    """Load _replay_conditioning_infotext from processing.py without importing the webui runtime."""
    tree = ast.parse(PROCESSING_PATH.read_text(encoding="utf8"))
    body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_replay_conditioning_infotext"]
    namespace = {}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(PROCESSING_PATH), "exec"), namespace)
    return namespace["_replay_conditioning_infotext"]


def test_cache_hits_do_not_repeat_ti_hashes():
    replay = load_replay()
    target = {}
    written = {"TI hashes": "emb_a: 1234abcd, emb_b: 5678ef90", "Emphasis": "No norm"}

    for _ in range(4):  # n_iter > 1 replays the same cached entries each iteration
        replay(target, written)

    assert target == {"TI hashes": "emb_a: 1234abcd, emb_b: 5678ef90", "Emphasis": "No norm"}


def test_new_ti_hashes_are_prepended_once():
    replay = load_replay()
    target = {"TI hashes": "emb_b: 5678ef90"}

    replay(target, {"TI hashes": "emb_c: 0000ffff, emb_b: 5678ef90"})

    assert target["TI hashes"] == "emb_c: 0000ffff, emb_b: 5678ef90"
