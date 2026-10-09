from pathlib import Path


def test_gb10_smoke_exercises_precision_diagnostics_through_api_endpoint():
    smoke = Path("gb10/smoke-test.sh").read_text(encoding="utf8")

    assert "/sdapi/v1/openclaw/precision-map" in smoke
    assert "expected initialized precision diagnostics payload" in smoke
    assert "summary = payload.get('summary', {})" in smoke
    assert "from modules.api import api" not in smoke
    assert "_precision_storage_active" not in smoke


def test_gb10_smoke_checks_vae_decode_graph_status_endpoint():
    smoke = Path("gb10/smoke-test.sh").read_text(encoding="utf8")

    assert "/sdapi/v1/openclaw/vae-decode-graphs" in smoke
    assert "expected VAE decode graph status payload" in smoke
