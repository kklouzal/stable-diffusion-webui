from __future__ import annotations

import importlib.util
import io
import os
import shutil
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
    sys.modules.pop("annotator_path", None)
    sys.modules["modules"] = types.SimpleNamespace(shared=shared)
    sys.modules["modules.shared"] = shared

    spec = importlib.util.spec_from_file_location("annotator_path", module_path)
    module = importlib.util.module_from_spec(spec)
    stdout = io.StringIO()
    with redirect_stdout(stdout):
        spec.loader.exec_module(module)

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
    assert run_sh.index("patch-controlnet-preprocessor-path.py") < run_sh.index("patch-controlnet-teed.py")
