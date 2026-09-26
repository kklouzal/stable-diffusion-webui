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


def make_dist(site: Path, name: str, version: str, requires=(), installer="pip", direct_url=None) -> Path:
    dist_info = site / f"{name.replace('-', '_')}-{version}.dist-info"
    dist_info.mkdir(parents=True)
    metadata = f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
    metadata += "".join(f"Requires-Dist: {req}\n" for req in requires)
    (dist_info / "METADATA").write_text(metadata, encoding="utf-8")
    (dist_info / "INSTALLER").write_text(f"{installer}\n", encoding="utf-8")
    if direct_url:
        (dist_info / "direct_url.json").write_text(json.dumps({"url": direct_url, "dir_info": {}}), encoding="utf-8")
    return dist_info


def run_with_site(site: Path, *args: str) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(site)
    return subprocess.run([sys.executable, *args], env=env, capture_output=True, text=True)


def test_protected_resolver_stubs_declare_only_released_requirements(tmp_path: Path):
    site = tmp_path / "site"
    make_dist(site, "gb10-parent", "1.0", [
        "gb10-released>=2.0,<3",
        "gb10-protected>=1",
        "gb10-released[extra]<2.5",
        "gb10-other; extra == 'test'",
        "gb10-released<9; sys_platform == 'win32'",
    ])
    make_dist(site, "gb10-leaf", "1.0")
    constraints = tmp_path / "constraints.txt"
    constraints.write_text("gb10-parent==1.0\ngb10-leaf==1.0\n", encoding="utf-8")
    floors = tmp_path / "floors.txt"
    floors.write_text("gb10-released>=2.1\ngb10-other>=1\n", encoding="utf-8")
    dependents = tmp_path / "dependents.txt"

    result = run_with_site(
        site,
        str(ROOT / "docker" / "create-protected-package-stubs.py"),
        "--constraints", str(constraints),
        "--wheel-dir", str(tmp_path / "wheels"),
        "--requirements-out", str(tmp_path / "stubs.txt"),
        "--released-floors", str(floors),
        "--dependents-out", str(dependents),
    )
    assert result.returncode == 0, result.stderr

    assert dependents.read_text(encoding="utf-8") == "gb10-parent==1.0\n"
    with zipfile.ZipFile(tmp_path / "wheels" / "gb10_parent-1.0-py3-none-any.whl") as archive:
        metadata = archive.read("gb10_parent-1.0.dist-info/METADATA").decode()
    requires = [line for line in metadata.splitlines() if line.startswith("Requires-Dist:")]
    assert requires == ["Requires-Dist: gb10-released<2.5", "Requires-Dist: gb10-released<3,>=2.0"]
    with zipfile.ZipFile(tmp_path / "wheels" / "gb10_leaf-1.0-py3-none-any.whl") as archive:
        assert "Requires-Dist:" not in archive.read("gb10_leaf-1.0.dist-info/METADATA").decode()


def test_protected_resolver_stubs_reject_pin_that_does_not_match_install(tmp_path: Path):
    site = tmp_path / "site"
    make_dist(site, "gb10-parent", "1.1")
    constraints = tmp_path / "constraints.txt"
    constraints.write_text("gb10-parent==1.0\n", encoding="utf-8")
    floors = tmp_path / "floors.txt"
    floors.write_text("gb10-released>=2\n", encoding="utf-8")

    result = run_with_site(
        site,
        str(ROOT / "docker" / "create-protected-package-stubs.py"),
        "--constraints", str(constraints),
        "--wheel-dir", str(tmp_path / "wheels"),
        "--requirements-out", str(tmp_path / "stubs.txt"),
        "--released-floors", str(floors),
        "--dependents-out", str(tmp_path / "dependents.txt"),
    )
    assert result.returncode != 0
    assert "does not match installed 1.1" in result.stderr


def snapshot_site(tmp_path: Path) -> Path:
    site = tmp_path / "site"
    make_dist(site, "torch", "2.99.0", ["gb10-stack-dep>=1"])
    make_dist(site, "gb10-stack-dep", "1.0")
    make_dist(site, "gb10-stock", "1.5")
    make_dist(site, "gb10-local-build", "1.0+nv1")
    make_dist(site, "gb10-source-build", "1.0", direct_url="file:///opt/src/gb10-source-build")
    make_dist(site, "gb10-dpkg", "1.0", installer="debian")
    make_dist(site, "nvidia-gb10-lib", "1.0")
    return site


def test_base_snapshot_releases_stock_wheels_with_ngc_floor(tmp_path: Path):
    site = snapshot_site(tmp_path)
    released = tmp_path / "released.txt"
    released.write_text("# comment\ngb10_stock  # trailing comment\n", encoding="utf-8")
    out = tmp_path / "out"

    result = run_with_site(site, str(ROOT / "docker" / "snapshot-base-packages.py"), "--released", str(released), "--out-dir", str(out))
    assert result.returncode == 0, result.stderr

    assert (out / "base-python-released-floors.txt").read_text(encoding="utf-8") == "gb10-stock>=1.5\n"
    names = set((out / "base-python-protected-names.txt").read_text(encoding="utf-8").split())
    assert {"torch", "gb10-stack-dep", "gb10-local-build", "gb10-dpkg"} <= names
    assert "gb10-stock" not in names
    constraints = (out / "base-python-protected-constraints.txt").read_text(encoding="utf-8").splitlines()
    assert "torch==2.99.0" in constraints
    assert not any(line.startswith("gb10-stock==") for line in constraints)


def test_base_snapshot_rejects_packages_that_break_the_release_rule(tmp_path: Path):
    site = snapshot_site(tmp_path)
    released = tmp_path / "released.txt"
    released.write_text(
        "gb10-stack-dep\ngb10-local-build\ngb10-source-build\ngb10-dpkg\nnvidia-gb10-lib\ngb10-missing\n",
        encoding="utf-8",
    )

    result = run_with_site(site, str(ROOT / "docker" / "snapshot-base-packages.py"), "--released", str(released), "--out-dir", str(tmp_path / "out"))
    assert result.returncode != 0
    for expected in (
        "gb10-stack-dep: in the requirement closure of torch",
        "gb10-local-build: not a stock PyPI wheel (local version label 1.0+nv1)",
        "gb10-source-build: not a stock PyPI wheel (installed from file:///opt/src/gb10-source-build)",
        "gb10-dpkg: not a stock PyPI wheel (installed by debian, not a pip wheel)",
        "nvidia-gb10-lib: CUDA/NVIDIA package prefixes are never released",
        "gb10-missing: released but not installed in the base image",
    ):
        assert expected in result.stderr
    assert not (tmp_path / "out").exists()


def test_base_snapshot_rejects_inherited_packages_changed_before_protection(tmp_path: Path):
    site = snapshot_site(tmp_path)
    released = tmp_path / "released.txt"
    released.write_text("gb10-stock\n", encoding="utf-8")
    pristine = tmp_path / "pristine.txt"
    pristine.write_text("torch==2.99.0\ngb10-stack-dep==0.9\ngb10-gone==1.0\ngb10-local-build==0.1\n", encoding="utf-8")
    args = [str(ROOT / "docker" / "snapshot-base-packages.py"), "--released", str(released), "--out-dir", str(tmp_path / "out")]

    result = run_with_site(site, *args, "--pristine", str(pristine), "--rebuilt", "gb10_local_build")
    assert result.returncode != 0
    assert "gb10-stack-dep: NGC ships 0.9 but a build step changed it to 1.0" in result.stderr
    assert "gb10-gone: NGC ships 1.0 but a build step removed it" in result.stderr
    assert "gb10-local-build" not in result.stderr
    assert not (tmp_path / "out").exists()

    pristine.write_text("torch==2.99.0\ngb10-stack-dep==1.0\n", encoding="utf-8")
    result = run_with_site(site, *args, "--pristine", str(pristine))
    assert result.returncode == 0, result.stderr


def test_stack_checker_validates_released_floors_and_declared_requirements(tmp_path: Path, monkeypatch):
    import importlib.metadata as md

    checker = load_script_module("check_protected_stack_released", "docker/check-protected-stack.py")
    site = tmp_path / "site"
    dists = [
        md.PathDistribution(make_dist(site, "gb10-released", "2.5")),
        md.PathDistribution(make_dist(site, "gb10-floored", "0.9")),
        md.PathDistribution(make_dist(site, "gb10-capper", "1.0", ["gb10-released<2.4", "gb10-released>=9; extra == 'x'"])),
        md.PathDistribution(make_dist(site, "gb10-happy", "1.0", ["gb10-released>=2", "gb10-unreleased<0"])),
        md.PathDistribution(make_dist(site / "shadowed", "gb10-released", "1.0")),
    ]
    monkeypatch.setattr(checker.md, "distributions", lambda: iter(dists))
    floors = tmp_path / "floors.txt"
    floors.write_text("gb10-released>=2.0\ngb10-floored>=1.0\ngb10-absent>=1\n", encoding="utf-8")

    report, problems = checker.check_released(checker.load_released_floors(floors))

    assert report["gb10-released"] == {"floor": "2.0", "installed": "2.5"}
    assert sorted(problems) == [
        "gb10-capper requires gb10-released<2.4, installed 2.5",
        "released package below NGC floor: gb10-floored 0.9 < 1.0",
        "released package is absent: gb10-absent",
    ]
