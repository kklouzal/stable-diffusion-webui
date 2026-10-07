from __future__ import annotations

import importlib.util
import io
import os
import resource
import shutil
import signal
import subprocess
import sys
import types
from contextlib import redirect_stdout
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PATCHER = REPO_ROOT / "gb10" / "patch-controlnet-preprocessor-path.py"

OLD_ANNOTATOR_PATH = """import os
from modules import shared

models_path = shared.opts.data.get('control_net_modules_path', None)
if not models_path:
    models_path = getattr(shared.cmd_opts, 'controlnet_annotator_models_path', None)
if not models_path:
    models_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'downloads')

if not os.path.isabs(models_path):
    models_path = os.path.join(shared.data_path, models_path)

models_path = os.path.realpath(models_path)
os.makedirs(models_path, exist_ok=True)
print(f'ControlNet preprocessor location: {models_path}')
"""

OLD_CONTROLNET = '''def on_ui_settings():
    section = ('control_net', "ControlNet")
    shared.opts.add_option("control_net_models_path", shared.OptionInfo(
        "", "Extra path to scan for ControlNet models (e.g. training output directory)", section=section))
    shared.opts.add_option("control_net_modules_path", shared.OptionInfo(
        "", "Path to directory containing annotator model directories (overrides corresponding command line flag)", section=section).needs_reload_ui())
    shared.opts.add_option("control_net_unit_count", shared.OptionInfo(
        3, "Multi-ControlNet: ControlNet unit number", gr.Slider, {"minimum": 1, "maximum": 10, "step": 1}, section=section).needs_reload_ui())
'''


def make_controlnet_tree(tmp_path: Path) -> Path:
    root = tmp_path / "sd-webui-controlnet"
    (root / "annotator").mkdir(parents=True)
    (root / "scripts").mkdir()
    (root / "annotator" / "annotator_path.py").write_text(OLD_ANNOTATOR_PATH, encoding="utf-8")
    (root / "scripts" / "controlnet.py").write_text(OLD_CONTROLNET, encoding="utf-8")
    return root


def run_patcher(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(PATCHER), str(root)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )


def resolve_annotator_models_path(tmp_path, patched_root, opts_data, cmd_path=None, data_path=None):
    # Import from a disposable copy so module-level os.makedirs calls cannot touch
    # the source fixture and each case gets a fresh module path.
    annotator_dir = tmp_path / "import" / "extensions" / "sd-webui-controlnet" / "annotator"
    annotator_dir.mkdir(parents=True, exist_ok=True)
    module_path = annotator_dir / "annotator_path.py"
    shutil.copy2(patched_root / "annotator" / "annotator_path.py", module_path)

    shared = types.SimpleNamespace(
        opts=types.SimpleNamespace(data=dict(opts_data)),
        cmd_opts=types.SimpleNamespace(controlnet_annotator_models_path=cmd_path),
        data_path=str(data_path or tmp_path / "data"),
    )
    # The stubs live only while the module executes: a SimpleNamespace left as sys.modules["modules"] broke every
    # later test file that imports the real webui package (or stubs only some of its submodules).
    saved = {name: sys.modules.get(name) for name in ("modules", "modules.shared", "annotator_path")}
    sys.modules.pop("annotator_path", None)
    sys.modules["modules"] = types.SimpleNamespace(shared=shared)
    sys.modules["modules.shared"] = shared
    try:
        spec = importlib.util.spec_from_file_location("annotator_path", module_path)
        module = importlib.util.module_from_spec(spec)
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            spec.loader.exec_module(module)
    finally:
        for name, value in saved.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value

    return module.models_path, stdout.getvalue()


def test_patcher_adds_preprocessor_models_option_and_is_idempotent(tmp_path):
    root = make_controlnet_tree(tmp_path)

    first = run_patcher(root)
    second = run_patcher(root)

    assert "Patched ControlNet preprocessor path plumbing" in first.stdout
    assert "already patched" in second.stdout
    annotator = (root / "annotator" / "annotator_path.py").read_text(encoding="utf-8")
    controlnet = (root / "scripts" / "controlnet.py").read_text(encoding="utf-8")
    assert "control_net_preprocessor_models_path" in annotator
    assert "control_net_preprocessor_models_path" in controlnet
    assert annotator.index("control_net_preprocessor_models_path") < annotator.index("control_net_modules_path")


def test_preprocessor_models_path_takes_precedence_over_legacy_modules_path(tmp_path):
    root = make_controlnet_tree(tmp_path)
    run_patcher(root)

    resolved, stdout = resolve_annotator_models_path(
        tmp_path,
        root,
        {
            "control_net_modules_path": "/tmp/legacy-controlnet-modules",
            "control_net_preprocessor_models_path": "/tmp/preprocessor-controlnet-models",
        },
    )

    assert resolved == "/tmp/preprocessor-controlnet-models"
    assert "ControlNet preprocessor location: /tmp/preprocessor-controlnet-models" in stdout


def test_legacy_modules_path_and_command_line_flag_remain_fallbacks(tmp_path):
    root = make_controlnet_tree(tmp_path)
    run_patcher(root)

    resolved, _ = resolve_annotator_models_path(
        tmp_path,
        root,
        {"control_net_modules_path": "/tmp/legacy-controlnet-modules"},
        cmd_path="/tmp/cmd-controlnet-models",
    )
    assert resolved == "/tmp/legacy-controlnet-modules"

    resolved, _ = resolve_annotator_models_path(
        tmp_path,
        root,
        {},
        cmd_path="/tmp/cmd-controlnet-models",
    )
    assert resolved == "/tmp/cmd-controlnet-models"


def test_relative_preprocessor_path_is_resolved_under_shared_data_path(tmp_path):
    root = make_controlnet_tree(tmp_path)
    run_patcher(root)
    data_path = tmp_path / "a1111-data"

    resolved, _ = resolve_annotator_models_path(
        tmp_path,
        root,
        {"control_net_preprocessor_models_path": "controlnet-preprocessors"},
        data_path=data_path,
    )

    assert resolved == os.path.realpath(data_path / "controlnet-preprocessors")


def test_run_sh_applies_preprocessor_path_patch_to_mounted_controlnet_extension():
    run_sh = (REPO_ROOT / "gb10" / "run.sh").read_text(encoding="utf-8")

    assert 'CONTROLNET_ROOT="${HOST_ROOT}/Extensions/sd-webui-controlnet"' in run_sh
    assert "patch-controlnet-preprocessor-path.py" in run_sh
    assert run_sh.index('CONTROLNET_ROOT="') < run_sh.index("patch-controlnet-preprocessor-path.py")


def run_patcher_with_file_size_limit(root: Path, limit: int) -> subprocess.CompletedProcess[str]:
    """Run the patcher with RLIMIT_FSIZE=limit and SIGXFSZ ignored: a write past `limit` bytes fails with EFBIG
    partway through, as on a full disk."""
    def limit_file_size():
        signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
        resource.setrlimit(resource.RLIMIT_FSIZE, (limit, limit))

    return subprocess.run([sys.executable, str(PATCHER), str(root)], text=True, capture_output=True, preexec_fn=limit_file_size)


def snapshot(root: Path) -> dict[str, bytes]:
    return {str(path.relative_to(root)): path.read_bytes() for path in sorted(root.rglob("*")) if path.is_file()}


def test_write_failing_midway_leaves_the_target_intact(tmp_path):
    root = make_controlnet_tree(tmp_path)
    (root / "annotator" / "annotator_path.py").chmod(0o640)
    before = snapshot(root)

    result = run_patcher_with_file_size_limit(root, 64)

    assert result.returncode != 0 and "File too large" in result.stderr
    assert snapshot(root) == before  # no truncated target, no leftover temporary file
    run_patcher(root)
    assert (root / "annotator" / "annotator_path.py").stat().st_mode & 0o777 == 0o640


def test_every_target_is_validated_before_any_is_written(tmp_path):
    root = make_controlnet_tree(tmp_path)
    controlnet = root / "scripts" / "controlnet.py"
    controlnet.write_text(OLD_CONTROLNET.replace("annotator model directories", "annotator directories"), encoding="utf-8")
    before = snapshot(root)

    result = subprocess.run([sys.executable, str(PATCHER), str(root)], text=True, capture_output=True)

    assert snapshot(root) == before  # the patchable annotator_path.py was not written either
    assert result.returncode != 0 and "unsupported ControlNet source" in result.stderr


def test_crlf_and_unverifiable_results_fail_closed_without_writing(tmp_path):
    root = make_controlnet_tree(tmp_path)
    annotator = root / "annotator" / "annotator_path.py"
    for payload, message in (
        (OLD_ANNOTATOR_PATH.replace("\n", "\r\n").encode("utf-8"), "CRLF"),
        ((OLD_ANNOTATOR_PATH + "def broken(:\n").encode("utf-8"), "invalid Python"),
    ):
        annotator.write_bytes(payload)
        before = snapshot(root)
        result = subprocess.run([sys.executable, str(PATCHER), str(root)], text=True, capture_output=True)
        assert result.returncode != 0 and message in result.stderr, result.stderr
        assert snapshot(root) == before


def test_patched_tree_holds_each_new_block_once_and_no_stray_old_block(tmp_path):
    root = make_controlnet_tree(tmp_path)
    run_patcher(root)
    patcher = load_patcher_module()
    for relative, old, new in patcher.TARGETS:
        text = (root / relative).read_text(encoding="utf-8")
        assert text.count(new) == 1 and old not in text.replace(new, "")
        compile(text, str(relative), "exec")


def load_patcher_module():
    spec = importlib.util.spec_from_file_location("gb10_patch_controlnet_preprocessor_path", PATCHER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
