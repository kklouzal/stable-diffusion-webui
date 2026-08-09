#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import sys

OLD_ANNOTATOR = """models_path = shared.opts.data.get('control_net_modules_path', None)
if not models_path:
    models_path = getattr(shared.cmd_opts, 'controlnet_annotator_models_path', None)
"""
NEW_ANNOTATOR = """models_path = shared.opts.data.get('control_net_preprocessor_models_path', None)
if not models_path:
    models_path = shared.opts.data.get('control_net_modules_path', None)
if not models_path:
    models_path = getattr(shared.cmd_opts, 'controlnet_annotator_models_path', None)
"""
OLD_CONTROLNET = """    shared.opts.add_option(\"control_net_modules_path\", shared.OptionInfo(
        \"\", \"Path to directory containing annotator model directories (overrides corresponding command line flag)\", section=section).needs_reload_ui())
"""
NEW_CONTROLNET = """    shared.opts.add_option(\"control_net_modules_path\", shared.OptionInfo(
        \"\", \"Legacy path to directory containing annotator/preprocessor model directories (overrides corresponding command line flag when no preprocessor-specific path is set)\", section=section).needs_reload_ui())
    shared.opts.add_option(\"control_net_preprocessor_models_path\", shared.OptionInfo(
        \"\", \"Path to directory containing annotator/preprocessor model directories (overrides legacy setting and corresponding command line flag)\", section=section).needs_reload_ui())
"""


def patch_file(path: pathlib.Path, old: str, new: str) -> bool:
    text = path.read_text(encoding="utf-8")
    if new in text:
        return False
    if old not in text:
        raise RuntimeError(f"expected patch target not found: {path}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    return True


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        raise SystemExit("usage: patch-controlnet-preprocessor-path.py /path/to/sd-webui-controlnet")
    root = pathlib.Path(argv[1]).resolve()
    changed = False
    changed |= patch_file(root / "annotator" / "annotator_path.py", OLD_ANNOTATOR, NEW_ANNOTATOR)
    changed |= patch_file(root / "scripts" / "controlnet.py", OLD_CONTROLNET, NEW_CONTROLNET)
    if changed:
        print(f"Patched ControlNet preprocessor path plumbing under {root}")
    else:
        print(f"ControlNet preprocessor path plumbing already patched under {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
