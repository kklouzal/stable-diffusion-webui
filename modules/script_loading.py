import hashlib
import importlib.util
import os
import re
import sys
from pathlib import Path

from modules import errors


loaded_scripts = {}


def module_name_for_path(path):
    """Return a stable, collision-resistant synthetic name for a script file."""
    resolved = Path(path).resolve()
    stem = re.sub(r"\W", "_", resolved.stem, flags=re.ASCII)
    if not stem or stem[0].isdigit():
        stem = f"script_{stem}"
    digest = hashlib.sha256(os.fsencode(resolved)).hexdigest()[:16]
    synthetic_name = f"a1111_dynamic_{stem}_{digest}"

    package_parts = []
    parent = resolved.parent
    while (parent / "__init__.py").is_file() and parent.name.isidentifier():
        package_parts.append(parent.name)
        parent = parent.parent
    if package_parts:
        synthetic_name = ".".join([*reversed(package_parts), synthetic_name])

    return synthetic_name


def load_module(path):
    path = os.fspath(path)
    module_name = module_name_for_path(path)
    module_spec = importlib.util.spec_from_file_location(module_name, path)
    if module_spec is None or module_spec.loader is None:
        raise ImportError(f"cannot create module spec for {path}")

    module = importlib.util.module_from_spec(module_spec)
    missing = object()
    previous = sys.modules.get(module_name, missing)
    sys.modules[module_name] = module
    try:
        module_spec.loader.exec_module(module)
    except BaseException:
        if sys.modules.get(module_name) is module:
            if previous is missing:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = previous
        raise

    loaded_scripts[path] = module
    return module


def preload_extensions(extensions_dir, parser, extension_list=None):
    if not os.path.isdir(extensions_dir):
        return

    extensions = extension_list if extension_list is not None else os.listdir(extensions_dir)
    for dirname in sorted(extensions):
        preload_script = os.path.join(extensions_dir, dirname, "preload.py")
        if not os.path.isfile(preload_script):
            continue

        try:
            module = load_module(preload_script)
            if hasattr(module, 'preload'):
                module.preload(parser)

        except Exception:
            errors.report(f"Error running preload() for {preload_script}", exc_info=True)
