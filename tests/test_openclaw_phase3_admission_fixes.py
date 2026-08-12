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


def test_cuda_graph_first_request_returns_captured_output_contract():
    source = Path("modules/openclaw_cuda_graphs.py").read_text()
    assert 'graph.replay()\n                return _clone_static(static_out)' in source
    assert 'return capture_return' not in source
    assert '_record_bypass("cache_warmup")' not in source


def test_vae_graph_first_request_returns_captured_output_contract():
    source = Path("modules/openclaw_vae_decode_graphs.py").read_text()
    assert 'graph.replay()\n                return static_output.clone()' in source
    assert 'return warmup_output' not in source


def test_compile_cache_namespace_runtime_ownership_contract():
    source = Path("gb10/run.sh").read_text()
    for family in ("torchinductor", "triton", "cuda"):
        assert f'"${{OPENCLAW_COMPILE_CACHE_ROOT}}/{family}/${{OPENCLAW_COMPILE_CACHE_NAMESPACE}}"' in source
    assert 'install -d -o 2323 -g 2323 -m 0750 "${cache_namespace_path}"' in source
    assert "sudo setpriv --reuid=2323 --regid=2323 --clear-groups test -w" in source
    assert 'rm -rf "${cache_namespace_path}"' not in source
