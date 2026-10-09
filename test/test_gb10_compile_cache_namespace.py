import re
import subprocess
from pathlib import Path

RUN_SH = Path("gb10/run.sh")
DRIVER_VERSION_FILE = "/sys/module/nvidia/version"


def _namespace_block():
    source = RUN_SH.read_text(encoding="utf8")
    return re.search(r"^A1111_COMMIT_HASH=.*?(?=^A1111_VERSION_TAG=)", source, re.S | re.M).group(0)


def _derive(tmp_path, *, driver_version="580.178.04\n", namespace_env=""):
    fake_docker = tmp_path / "docker"
    fake_docker.write_text('#!/bin/sh\necho "$@" > "$(dirname "$0")/docker-args"\necho "torch-2.14.0a0+4fdf77b940.nv26.8-triton-3.8.0+git4c7e62b7-cuda-13.4.1.012"\n')
    fake_docker.chmod(0o755)
    driver_file = tmp_path / "nvidia-version"
    if driver_version is not None:
        driver_file.write_text(driver_version)
    block = _namespace_block().replace(DRIVER_VERSION_FILE, str(driver_file))
    script = f'set -euo pipefail\nsudo() {{ "$@"; }}\nDOCKER_BIN={fake_docker}\nIMAGE_TAG=local/gb10-a1111:latest\nPROJECT_ROOT={tmp_path}\n{namespace_env}\n{block}\necho "$OPENCLAW_COMPILE_CACHE_NAMESPACE"\n'
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True)


def test_namespace_is_keyed_by_image_compiler_stack_and_host_driver(tmp_path):
    result = _derive(tmp_path)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "torch-2.14.0a0_4fdf77b940.nv26.8-triton-3.8.0_git4c7e62b7-cuda-13.4.1.012-driver-580.178.04"
    docker_args = (tmp_path / "docker-args").read_text()
    assert "run --rm --network none --entrypoint python local/gb10-a1111:latest -c" in docker_args
    assert "--gpus" not in docker_args


def test_explicit_namespace_override_skips_the_image_query(tmp_path):
    result = _derive(tmp_path, namespace_env="OPENCLAW_COMPILE_CACHE_NAMESPACE=pinned")

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "pinned"
    assert not (tmp_path / "docker-args").exists()


def test_unreadable_driver_version_fails_the_deploy(tmp_path):
    result = _derive(tmp_path, driver_version=None)

    assert result.returncode == 1
    assert "cannot read the host NVIDIA driver version" in result.stderr


def test_namespace_does_not_track_the_a1111_commit(tmp_path):
    result = _derive(tmp_path, namespace_env="A1111_COMMIT_HASH=0123456789abcdef0123456789abcdef01234567")

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "torch-2.14.0a0_4fdf77b940.nv26.8-triton-3.8.0_git4c7e62b7-cuda-13.4.1.012-driver-580.178.04"


def test_namespace_dirs_are_owned_by_the_runtime_user_and_other_namespaces_are_kept():
    # Text contract: the block runs sudo install/setpriv/find against the host cache root.
    source = RUN_SH.read_text(encoding="utf8")
    block = source[source.index("COMPILE_CACHE_NAMESPACE_PATHS=("):source.index("TARGET_IMAGE_ID=")]

    for family in ("torchinductor", "triton", "cuda"):
        assert f'"${{OPENCLAW_COMPILE_CACHE_ROOT}}/{family}/${{OPENCLAW_COMPILE_CACHE_NAMESPACE}}"' in block
    assert 'install -d -o 2323 -g 2323 -m 0750 "${cache_namespace_path}"' in block
    assert "sudo setpriv --reuid=2323 --regid=2323 --clear-groups test -w" in block
    assert "rm -rf" not in block
    assert "other namespace dirs hold" in block
