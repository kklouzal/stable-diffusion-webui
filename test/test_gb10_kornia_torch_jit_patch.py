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
    # No bytecode from the import checks: only what the patcher compiles itself may be on disk afterwards.
    env = {**os.environ, "PYTHONPATH": str(tmp_path / "site"), "PYTHONDONTWRITEBYTECODE": "1"}
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


def _pyc_matches_source(path):
    """True when the timestamp pyc next to `path` is the one Python would use for the current source."""
    import importlib.util
    import struct

    pyc = Path(importlib.util.cache_from_source(str(path)))
    if not pyc.exists():
        return False
    flags, mtime, size = struct.unpack("<III", pyc.read_bytes()[4:16])
    stat = path.stat()
    return flags == 0 and mtime == int(stat.st_mtime) & 0xFFFFFFFF and size == stat.st_size & 0xFFFFFFFF


def test_patched_modules_get_fresh_bytecode_even_when_the_import_check_never_loads_them(tmp_path):
    # The runtime user cannot rewrite site-packages bytecode, so a stale pyc would be recompiled in memory at every start.
    package, result = _run_patcher(tmp_path, {
        "__init__.py": "",
        "filters/blur.py": "@torch.jit.script\ndef blur(x):\n    return x\n",
    })

    assert result.returncode == 0, result.stdout + result.stderr
    assert _pyc_matches_source(package / "filters/blur.py")


CONTROLNET_AUX_PATCHER = Path("docker/patch-controlnet-aux-compat.py").resolve()


def test_controlnet_aux_patch_compiles_every_file_it_rewrites(tmp_path):
    import os
    import subprocess
    import sys

    package = tmp_path / "site" / "controlnet_aux"
    files = {
        "__init__.py": "from .mediapipe_face import MediapipeFaceDetector\n",
        "mediapipe_face/__init__.py": "import mediapipe\n",
        "zoe/ops.py": "from timm.models.layers import DropPath\n",
    }
    for name, text in files.items():
        (package / name).parent.mkdir(parents=True, exist_ok=True)
        (package / name).write_text(text, encoding="utf-8")
    # No bytecode from the import checks: only what the patcher compiles itself may be on disk afterwards.
    env = {**os.environ, "PYTHONPATH": str(tmp_path / "site"), "PYTHONDONTWRITEBYTECODE": "1"}
    result = subprocess.run([sys.executable, str(CONTROLNET_AUX_PATCHER)], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120)

    assert result.returncode == 0, result.stdout + result.stderr
    assert (package / "zoe/ops.py").read_text(encoding="utf-8") == "from timm.layers import DropPath\n"
    assert "unavailable_reason = str(_gb10_mediapipe_exc)" in (package / "__init__.py").read_text(encoding="utf-8")
    assert _pyc_matches_source(package / "zoe/ops.py")
    assert _pyc_matches_source(package / "__init__.py")
