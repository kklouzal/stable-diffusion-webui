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


def test_gb10_smoke_fails_when_the_container_cannot_use_cuda():
    smoke = Path("gb10/smoke-test.sh").read_text(encoding="utf8")

    assert "if not torch.cuda.is_available():\n    raise SystemExit(" in smoke
    # The quantized-Linear checks run unconditionally once CUDA is required.
    assert "if torch.cuda.is_available():" not in smoke


def test_gb10_smoke_requires_the_expected_scripts_in_either_tab():
    smoke = Path("gb10/smoke-test.sh").read_text(encoding="utf8")

    expected = smoke.split('EXPECTED_SCRIPTS="${EXPECTED_SCRIPTS:-', 1)[1].split('}"', 1)[0].split(",")
    for name in ("controlnet", "incantations", "teacache", "openclaw multi-sampler", "tiled diffusion", "ultimate sd upscale"):
        assert name in expected
    assert "'/sdapi/v1/scripts'," in smoke
    assert "missing = sorted(expected_scripts - set(payload['txt2img']) - set(payload['img2img']))" in smoke
