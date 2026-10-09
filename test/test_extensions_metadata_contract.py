import types

from test.helpers import load_source, module


def load_extensions_module(tmp_path):
    return load_source("extensions_under_test", "modules/extensions.py", {
        "modules": module("modules"),
        "modules.shared": module(
            "modules.shared",
            cmd_opts=types.SimpleNamespace(disable_all_extensions=False, disable_extra_extensions=False),
            opts=types.SimpleNamespace(disable_all_extensions="none", disabled_extensions=[]),
        ),
        "modules.errors": module("modules.errors", report=lambda *args, **kwargs: None),
        "modules.cache": module("modules.cache", cached_data_for_file=lambda *args, **kwargs: args[-1]()),
        "modules.scripts": module("modules.scripts", ScriptFile=lambda basedir, filename, path: (basedir, filename, path)),
        "modules.gitpython_hack": module("modules.gitpython_hack", Repo=object),
        "modules.paths_internal": module(
            "modules.paths_internal",
            extensions_dir=str(tmp_path / "extensions"),
            extensions_builtin_dir=str(tmp_path / "extensions-builtin"),
        ),
    })


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
