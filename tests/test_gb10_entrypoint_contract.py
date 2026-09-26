import re
from pathlib import Path


def test_entrypoint_prepares_vaeapprox_download_directory_for_first_run():
    entrypoint = Path("docker/entrypoint.sh").read_text(encoding="utf-8")

    assert "$A1111_HOME/models/VAE-approx" in entrypoint
    assert "mkdir -p \"$A1111_HOME/tmp\" \"$A1111_HOME/models/ControlNet\" \"$A1111_HOME/models/VAE-approx\"" in entrypoint
    assert "$A1111_HOME/models/ControlNet" in entrypoint
    assert "$A1111_HOME/models/VAE-approx" in entrypoint


def test_bare_container_start_uses_the_api_only_launcher_default():
    entrypoint = Path("docker/entrypoint.sh").read_text(encoding="utf-8")
    launcher = Path("docker/launch-a1111.sh").read_text(encoding="utf-8")

    # The launcher owns the default flags; the entrypoint must not shadow them with a partial set.
    assert not re.search(r"^\s*(export\s+)?COMMANDLINE_ARGS=", entrypoint, re.MULTILINE)
    default = launcher.split('COMMANDLINE_ARGS="${COMMANDLINE_ARGS:-', 1)[1].split('}"', 1)[0]
    for flag in ("--nowebui", "--api", "--listen", "--dtype bfloat16"):
        assert flag in default
