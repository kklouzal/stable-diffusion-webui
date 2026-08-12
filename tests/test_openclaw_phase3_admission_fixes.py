import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _function(path, name):
    tree = ast.parse((ROOT / path).read_text())
    return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)


def test_lora_execution_identity_has_no_transition_epochs():
    fn = _function("extensions-builtin/Lora/networks.py", "_execution_identity")
    source = ast.unparse(fn)
    assert "epoch_subset" not in source
    assert all(x in source for x in ("device", "dtype", "dtype_unet"))


def test_lora_source_key_is_immutable_source_plus_parser_contract():
    fn = _function("extensions-builtin/Lora/networks.py", "network_source_key")
    source = ast.unparse(fn)
    assert "epoch_subset" not in source
    assert "filename" in source
    assert "source_signature" in source
    assert "LORA_SOURCE_SCHEMA_REVISION" in source


def test_vae_reset_preserves_enablement_and_telemetry_contract():
    fn = _function("modules/openclaw_vae_decode_graphs.py", "set_enabled")
    source = ast.unparse(fn)
    assert "enabled is not None" in source
    assert "_clear_cache_locked()" in source
    assert "_LIFECYCLE_STATE.clear()" in source
    assert "manual_reset" in source
    assert "bump_epoch" not in source
    assert "_COUNTERS['invalidations'] += 1" in source
    assert 'for key in _COUNTERS' not in source
    assert "_FAILED_KEYS.clear" not in source  # clear helper owns atomic teardown


def test_vae_api_clear_without_enabled_is_reset_not_disable():
    fn = _function("modules/api/api.py", "set_vae_decode_graphs")
    source = ast.unparse(fn)
    assert "'enabled' in req" in source
    assert "else None" in source
