def test_extension_item_schema_allows_missing_git_metadata(initialize, tmp_path):
    # An extension without readable git metadata keeps branch/commit_date None; /sdapi/v1/extensions still lists it.
    from modules import extensions
    from modules.api import models

    extension = extensions.Extension("local-extension", str(tmp_path))
    extension.remote = "https://example.invalid/local-extension.git"

    item = models.ExtensionItem.model_validate({"name": extension.name, **extension.to_dict(), "enabled": extension.enabled})

    assert item.branch is None
    assert item.commit_date is None
