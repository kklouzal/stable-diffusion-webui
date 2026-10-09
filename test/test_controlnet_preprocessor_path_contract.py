"""ControlNet annotator model directory resolution (extensions/sd-webui-controlnet/annotator/annotator_path.py).

Precedence: the control_net_preprocessor_models_path option, then the legacy control_net_modules_path option, then
--controlnet-annotator-models-path, then annotator/downloads/; a relative path resolves under the webui data path.
"""
from __future__ import annotations

import io
import os
import shutil
import types
from contextlib import redirect_stdout
from pathlib import Path

from test.helpers import load_source, module

ANNOTATOR_PATH = Path(__file__).resolve().parents[1] / "extensions" / "sd-webui-controlnet" / "annotator" / "annotator_path.py"


def resolve_annotator_models_path(tmp_path, opts_data, cmd_path=None, data_path=None):
    # Import from a disposable copy so the module-level os.makedirs calls cannot touch the source tree.
    annotator_dir = tmp_path / "import" / "extensions" / "sd-webui-controlnet" / "annotator"
    annotator_dir.mkdir(parents=True, exist_ok=True)
    module_path = annotator_dir / "annotator_path.py"
    shutil.copy2(ANNOTATOR_PATH, module_path)

    shared = module(
        "modules.shared",
        opts=types.SimpleNamespace(data=dict(opts_data)),
        cmd_opts=types.SimpleNamespace(controlnet_annotator_models_path=cmd_path),
        data_path=str(data_path or tmp_path / "data"),
    )
    stdout = io.StringIO()
    with redirect_stdout(stdout):
        annotator_path = load_source("annotator_path", module_path, {"modules": module("modules", shared=shared), "modules.shared": shared})

    return annotator_path.models_path, stdout.getvalue(), annotator_dir


def test_preprocessor_models_path_takes_precedence_over_legacy_modules_path(tmp_path):
    resolved, stdout, _ = resolve_annotator_models_path(
        tmp_path,
        {
            "control_net_modules_path": str(tmp_path / "legacy-controlnet-modules"),
            "control_net_preprocessor_models_path": str(tmp_path / "preprocessor-controlnet-models"),
        },
        cmd_path=str(tmp_path / "cmd-controlnet-models"),
    )

    assert resolved == os.path.realpath(tmp_path / "preprocessor-controlnet-models")
    assert f"ControlNet preprocessor location: {resolved}" in stdout


def test_legacy_modules_path_command_line_flag_and_downloads_remain_fallbacks(tmp_path):
    legacy, cmd = str(tmp_path / "legacy-controlnet-modules"), str(tmp_path / "cmd-controlnet-models")
    assert resolve_annotator_models_path(tmp_path, {"control_net_modules_path": legacy}, cmd_path=cmd)[0] == os.path.realpath(legacy)
    assert resolve_annotator_models_path(tmp_path, {"control_net_preprocessor_models_path": ""}, cmd_path=cmd)[0] == os.path.realpath(cmd)

    resolved, _, annotator_dir = resolve_annotator_models_path(tmp_path, {})
    assert resolved == os.path.realpath(annotator_dir / "downloads")
    assert os.path.isdir(resolved)


def test_relative_preprocessor_path_is_resolved_under_shared_data_path(tmp_path):
    data_path = tmp_path / "a1111-data"

    resolved, _, _ = resolve_annotator_models_path(
        tmp_path,
        {"control_net_preprocessor_models_path": "controlnet-preprocessors"},
        data_path=data_path,
    )

    assert resolved == os.path.realpath(data_path / "controlnet-preprocessors")
