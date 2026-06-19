import importlib.util
import sys
import types
from pathlib import Path


def load_extensions_module(tmp_path):
    errors = types.SimpleNamespace(report=lambda *args, **kwargs: None)
    cache = types.SimpleNamespace(cached_data_for_file=lambda *args, **kwargs: args[-1]())
    scripts = types.SimpleNamespace(ScriptFile=lambda basedir, filename, path: (basedir, filename, path))
    shared = types.SimpleNamespace(
        cmd_opts=types.SimpleNamespace(disable_all_extensions=False, disable_extra_extensions=False),
        opts=types.SimpleNamespace(disable_all_extensions="none", disabled_extensions=[]),
    )

    sys.modules["modules"] = types.ModuleType("modules")
    sys.modules["modules.shared"] = shared
    sys.modules["modules.errors"] = errors
    sys.modules["modules.cache"] = cache
    sys.modules["modules.scripts"] = scripts
    sys.modules["modules.gitpython_hack"] = types.SimpleNamespace(Repo=object)
    sys.modules["modules.paths_internal"] = types.SimpleNamespace(
        extensions_dir=str(tmp_path / "extensions"),
        extensions_builtin_dir=str(tmp_path / "extensions-builtin"),
        script_path=str(tmp_path),
    )

    spec = importlib.util.spec_from_file_location("extensions_under_test", "modules/extensions.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_metadata(path, name):
    path.mkdir(parents=True)
    (path / "metadata.ini").write_text(f"[Extension]\nName = {name}\n", encoding="utf8")


def test_metadata_name_is_normalized_and_used_for_extension_identity(tmp_path):
    extensions = load_extensions_module(tmp_path)
    ext_path = tmp_path / "sample-ext"
    write_metadata(ext_path, "Canonical.Ext")

    metadata = extensions.ExtensionMetadata(str(ext_path), "sample-ext")
    extension = extensions.Extension("sample-ext", str(ext_path), metadata=metadata)

    assert metadata.canonical_name == "canonical.ext"
    assert extension.canonical_name == "canonical.ext"


def test_extension_without_explicit_metadata_uses_created_metadata(tmp_path):
    extensions = load_extensions_module(tmp_path)
    ext_path = tmp_path / "plain-ext"
    ext_path.mkdir()

    extension = extensions.Extension("plain-ext", str(ext_path))

    assert extension.canonical_name == "plain-ext"


def test_list_extensions_keys_loaded_extensions_by_metadata_name(tmp_path):
    extensions = load_extensions_module(tmp_path)
    ext_path = tmp_path / "extensions" / "folder-name"
    write_metadata(ext_path, "CanonicalName")

    extensions.list_extensions()

    assert "canonicalname" in extensions.loaded_extensions
    assert "folder-name" not in extensions.loaded_extensions
    assert extensions.extensions[0].canonical_name == "canonicalname"
