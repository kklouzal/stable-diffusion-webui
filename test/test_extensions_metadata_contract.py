import importlib.util
import sys
import types

import pytest


@pytest.fixture(autouse=True)
def restore_module_stubs():
    saved = {name: module for name, module in sys.modules.items() if name == "modules" or name.startswith("modules.")}
    yield

    for name in [name for name in sys.modules if name == "modules" or name.startswith("modules.")]:
        if name not in saved:
            sys.modules.pop(name, None)

    for name, module in saved.items():
        sys.modules[name] = module


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


def test_rescan_keeps_the_registries_complete_until_the_new_ones_are_published(tmp_path):
    extensions = load_extensions_module(tmp_path)
    for name in ("first", "second"):
        (tmp_path / "extensions" / name).mkdir(parents=True)
    extensions.list_extensions()
    previous = list(extensions.extensions)
    observed = []
    create_metadata = extensions.ExtensionMetadata

    def observing_metadata(path, canonical_name):
        # what a concurrent reader (script_callbacks.find_extension/sort_callbacks) sees during the rescan
        observed.append(([x.name for x in extensions.extensions], len(extensions.extension_paths), sorted(extensions.loaded_extensions)))
        return create_metadata(path, canonical_name)

    extensions.ExtensionMetadata = observing_metadata
    extensions.list_extensions()

    assert observed == [(["first", "second"], 2, ["first", "second"])] * 2
    assert [x.name for x in extensions.extensions] == ["first", "second"]
    assert all(new is not old for new, old in zip(extensions.extensions, previous))
    assert sorted(extensions.extension_paths) == sorted(x.path for x in extensions.extensions)


def test_read_info_keeps_fields_when_the_repository_changes_while_it_is_read(tmp_path):
    extensions = load_extensions_module(tmp_path)
    ext_path = tmp_path / "extensions" / "git-ext"
    (ext_path / ".git").mkdir(parents=True)
    extension = extensions.Extension("git-ext", str(ext_path))

    def read_repository():
        extension.remote = "https://example.invalid/git-ext.git"
        extension.commit_hash = "0123456789abcdef"
        extension.have_info_from_repo = True

    def cached_data_for_file(subsection, title, filename, func, **kwargs):
        func()
        return None  # cache.cached_data_for_file: source revision changed during func(), nothing cached

    extension.do_read_info_from_repo = read_repository
    extensions.cache = types.SimpleNamespace(cached_data_for_file=cached_data_for_file)

    extension.read_info_from_repo()

    assert (extension.remote, extension.commit_hash) == ("https://example.invalid/git-ext.git", "0123456789abcdef")
