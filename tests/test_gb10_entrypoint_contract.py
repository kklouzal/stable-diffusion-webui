from pathlib import Path


def test_entrypoint_prepares_vaeapprox_download_directory_for_first_run():
    entrypoint = Path("docker/entrypoint.sh").read_text(encoding="utf-8")

    assert "$A1111_HOME/models/VAE-approx" in entrypoint
    assert "mkdir -p \"$A1111_HOME/tmp\" \"$A1111_HOME/models/ControlNet\" \"$A1111_HOME/models/VAE-approx\"" in entrypoint
    assert "$A1111_HOME/models/ControlNet" in entrypoint
    assert "$A1111_HOME/models/VAE-approx" in entrypoint
