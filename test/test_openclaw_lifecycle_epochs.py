import ast
from types import SimpleNamespace

from modules import openclaw_lifecycle_epochs


def test_checkpoint_commit_is_successful_and_change_only(monkeypatch):
    calls = []
    monkeypatch.setattr(openclaw_lifecycle_epochs.openclaw_cache_epochs, "bump_epoch", lambda dim, reason: calls.append((dim, reason)))
    assert not openclaw_lifecycle_epochs.publish_checkpoint_commit(changed=False)
    assert openclaw_lifecycle_epochs.publish_checkpoint_commit(changed=True, vae_bytes_changed=True, vae_object_changed=True)
    assert calls == [(dim, 'checkpoint_commit') for dim in ('checkpoint_object_epoch', 'source_bytes_epoch', 'vae_bytes_epoch', 'vae_object_epoch')]


def test_model_movement_only_effective_transition(monkeypatch):
    calls = []
    monkeypatch.setattr(openclaw_lifecycle_epochs.openclaw_cache_epochs, "bump_epoch", lambda dim, reason: calls.append((dim, reason)))
    model = SimpleNamespace()
    assert not openclaw_lifecycle_epochs.publish_model_movement_commit(model, before=None, to_cpu=True)
    assert not openclaw_lifecycle_epochs.publish_model_movement_commit(model, before="cpu", to_cpu=True)
    assert openclaw_lifecycle_epochs.publish_model_movement_commit(model, before="cpu", to_cpu=False)
    assert calls == [(dim, 'model_movement_commit') for dim in ('model_movement_epoch', 'device_epoch')]


def test_vae_pending_publishes_only_after_successful_finalization(monkeypatch):
    calls = []
    monkeypatch.setattr(openclaw_lifecycle_epochs.openclaw_cache_epochs, "bump_epoch", lambda dim, reason: calls.append((dim, reason)))
    model = SimpleNamespace()
    assert not openclaw_lifecycle_epochs.note_vae_commit(model, bytes_changed=True, object_changed=True, publish=False)
    assert calls == []
    pending = openclaw_lifecycle_epochs.take_pending_vae_commit(model)
    assert pending == (True, True)
    assert openclaw_lifecycle_epochs.note_vae_commit(model, bytes_changed=pending[0], object_changed=pending[1], publish=True)
    assert calls == [(dim, 'vae_commit') for dim in ('vae_bytes_epoch', 'vae_object_epoch')]


def test_vae_failure_can_discard_pending(monkeypatch):
    calls = []
    monkeypatch.setattr(openclaw_lifecycle_epochs.openclaw_cache_epochs, "bump_epoch", lambda dim, reason: calls.append((dim, reason)))
    model = SimpleNamespace()
    openclaw_lifecycle_epochs.note_vae_commit(model, bytes_changed=True, object_changed=True, publish=False)
    openclaw_lifecycle_epochs.discard_pending_vae_commit(model)
    assert openclaw_lifecycle_epochs.take_pending_vae_commit(model) == (False, False)
    assert calls == []


def test_static_commit_sites_follow_final_commit_points():
    models = open("modules/sd_models.py", encoding="utf-8").read()
    vae = open("modules/sd_vae.py", encoding="utf-8").read()
    assert models.index("model_data.set_sd_model(sd_model)", models.index("def reload_model_weights")) < models.index("publish_checkpoint_commit", models.index("def reload_model_weights"))
    assert models.index("Model loaded in", models.index("def load_model(")) < models.index("publish_checkpoint_commit", models.index("def load_model("))
    assert vae.index("finally:", vae.index("def reload_vae_weights")) < vae.index("publish=True", vae.index("def reload_vae_weights"))
    assert "Precision, attention, and compile epochs are intentionally deferred" in open("modules/openclaw_lifecycle_epochs.py", encoding="utf-8").read()

def test_lifecycle_reason_literals_are_validated_epoch_reasons():
    source = open("modules/openclaw_lifecycle_epochs.py", encoding="utf-8").read()
    tree = ast.parse(source)
    reason_literals = {
        keyword.value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for keyword in node.keywords
        if keyword.arg == "reason"
        and isinstance(keyword.value, ast.Constant)
        and isinstance(keyword.value.value, str)
    }
    assert reason_literals
    assert reason_literals <= openclaw_lifecycle_epochs.openclaw_cache_epochs.EPOCH_BUMP_REASONS


def test_load_vae_records_one_pending_commit_note():
    source = open("modules/sd_vae.py", encoding="utf-8").read()
    tree = ast.parse(source)
    load_vae = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "load_vae")
    pending_notes = [
        node
        for node in ast.walk(load_vae)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "note_vae_commit"
        and any(
            keyword.arg == "publish"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is False
            for keyword in node.keywords
        )
    ]
    assert len(pending_notes) == 1
