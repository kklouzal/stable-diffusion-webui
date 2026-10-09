from pathlib import Path

PATCHER = Path("docker/patch-kornia-torch-jit-compat.py").resolve()


def _run_patcher(tmp_path, files):
    """Run the build-time patcher against a stand-in kornia package that shadows the installed one."""
    import os
    import subprocess
    import sys

    package = tmp_path / "site" / "kornia"
    for name, text in files.items():
        (package / name).parent.mkdir(parents=True, exist_ok=True)
        (package / name).write_text(text, encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": str(tmp_path / "site")}
    return package, subprocess.run([sys.executable, str(PATCHER)], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120)


def test_kornia_torch_jit_patch_removes_jit_script_decorators_and_asserts_clean_import(tmp_path):
    # Undecorated, the functions no longer need torch at import: the stand-in imports cleanly only once patched.
    package, result = _run_patcher(tmp_path, {
        "__init__.py": "from .geometry.ops import scale\n",
        "geometry/__init__.py": "",
        "geometry/ops.py": "@torch.jit.script\ndef scale(x):\n    return x * 2\n\n\n@torch.jit.script\ndef shift(x):\n    return x + 1\n",
        "utils.py": "def plain(x):\n    return x\n",
    })

    assert result.returncode == 0, result.stdout + result.stderr
    assert (package / "geometry/ops.py").read_text(encoding="utf-8") == "def scale(x):\n    return x * 2\n\n\ndef shift(x):\n    return x + 1\n"
    assert (package / "utils.py").read_text(encoding="utf-8") == "def plain(x):\n    return x\n"
    assert "files=1" in result.stdout
    assert "kornia import clean" in result.stdout


def test_kornia_torch_jit_patch_fails_the_build_when_the_import_still_warns(tmp_path):
    _package, result = _run_patcher(tmp_path, {
        "__init__.py": "import warnings\nwarnings.warn('deprecated torch API', DeprecationWarning)\n",
    })

    assert result.returncode != 0
    assert "kornia import clean" not in result.stdout


def test_dockerfile_runs_kornia_torch_jit_patch_after_runtime_requirements_install():
    dockerfile = Path("Dockerfile").read_text(encoding="utf8")

    assert "COPY docker/patch-kornia-torch-jit-compat.py /usr/local/bin/gb10-a1111-patch-kornia-torch-jit-compat" in dockerfile
    assert "&& /usr/local/bin/gb10-a1111-patch-controlnet-aux-compat-v2" in dockerfile
    assert "&& /usr/local/bin/gb10-a1111-patch-kornia-torch-jit-compat" in dockerfile
