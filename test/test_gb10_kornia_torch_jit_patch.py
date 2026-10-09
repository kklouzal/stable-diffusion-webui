from pathlib import Path


def test_kornia_torch_jit_patch_removes_jit_script_decorators_and_asserts_clean_import():
    patcher = Path("docker/patch-kornia-torch-jit-compat.py").read_text(encoding="utf8")

    assert 'original.replace("@torch.jit.script\\n", "")' in patcher
    assert '"error::DeprecationWarning"' in patcher
    assert "import kornia; print('kornia import clean')" in patcher


def test_dockerfile_runs_kornia_torch_jit_patch_after_runtime_requirements_install():
    dockerfile = Path("Dockerfile").read_text(encoding="utf8")

    assert "COPY docker/patch-kornia-torch-jit-compat.py /usr/local/bin/gb10-a1111-patch-kornia-torch-jit-compat" in dockerfile
    assert "&& /usr/local/bin/gb10-a1111-patch-controlnet-aux-compat-v2" in dockerfile
    assert "&& /usr/local/bin/gb10-a1111-patch-kornia-torch-jit-compat" in dockerfile
