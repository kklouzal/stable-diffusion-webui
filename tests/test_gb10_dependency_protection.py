import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def load_script_module(name: str, relative_path: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prepare_resolver_excludes_all_nvidia_base_packages(tmp_path: Path):
    source = tmp_path / "requirements.txt"
    source.write_text("setuptools==69.5.1\nnumpy\nrequests\ntorch\n", encoding="utf-8")
    target = tmp_path / "resolver.txt"
    protected = tmp_path / "protected.txt"
    protected.write_text("setuptools\nnumpy\n", encoding="utf-8")
    audit = tmp_path / "audit.json"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "docker" / "prepare-resolver-input.py"),
            "--source",
            str(source),
            "--target",
            str(target),
            "--wheel-dir",
            str(tmp_path / "wheels"),
            "--audit",
            str(audit),
            "--protected-names-file",
            str(protected),
        ],
        check=True,
    )

    assert target.read_text(encoding="utf-8") == "requests\n"
    removed = {item["name"] for item in json.loads(audit.read_text())["removed"]}
    assert removed == {"numpy", "setuptools", "torch"}


def test_runtime_filter_excludes_all_nvidia_base_packages(tmp_path: Path):
    source = tmp_path / "resolved.txt"
    source.write_text(
        "numpy==2.3.2\nrequests==2.32.5\nsetuptools==69.5.1\ntorch==2.12.1\n",
        encoding="utf-8",
    )
    target = tmp_path / "runtime.txt"
    protected = tmp_path / "protected.txt"
    protected.write_text("numpy\nsetuptools\n", encoding="utf-8")
    audit = tmp_path / "audit.json"
    env = os.environ.copy()
    env.update(
        SOURCE=str(source),
        TARGET=str(target),
        AUDIT=str(audit),
        BASE_PROTECTED_NAMES_FILE=str(protected),
    )

    subprocess.run(
        [sys.executable, str(ROOT / "docker" / "filter-resolved-requirements.py")],
        check=True,
        env=env,
    )

    assert target.read_text(encoding="utf-8") == "requests==2.32.5\n"
    removed = {item["name"] for item in json.loads(audit.read_text())["removed"]}
    assert removed == {"numpy", "setuptools", "torch"}


def test_protected_resolver_stubs_preserve_versions_without_base_dependencies(tmp_path: Path):
    constraints = tmp_path / "constraints.txt"
    constraints.write_text("numpy==2.5.2\npip==26.2.1\n", encoding="utf-8")
    wheels = tmp_path / "wheels"
    requirements = tmp_path / "stubs.txt"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "docker" / "create-protected-package-stubs.py"),
            "--constraints",
            str(constraints),
            "--wheel-dir",
            str(wheels),
            "--requirements-out",
            str(requirements),
        ],
        check=True,
    )

    built = list(wheels.glob("*.whl"))
    assert [wheel.name for wheel in built] == ["numpy-2.5.2-py3-none-any.whl"]
    assert requirements.read_text().strip() == str(built[0].resolve())
    with zipfile.ZipFile(built[0]) as archive:
        metadata = archive.read("numpy-2.5.2.dist-info/METADATA").decode()
    assert "Name: numpy\n" in metadata
    assert "Version: 2.5.2\n" in metadata
    assert "Requires-Dist:" not in metadata


def test_stack_checker_treats_every_named_base_package_as_protected():
    checker = load_script_module("check_protected_stack", "docker/check-protected-stack.py")

    assert checker.is_protected("numpy", {"numpy"})
    data = {
        "exact": {name: {"present": True} for name in checker.REQUIRED_PRESENT},
        "base_protected_names": ["numpy"],
        "packages": {},
    }
    assert checker.validate_baseline(data) == ["NVIDIA base package is absent: numpy"]
