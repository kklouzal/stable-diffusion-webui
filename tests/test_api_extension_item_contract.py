from pathlib import Path


def test_extension_item_schema_allows_missing_git_metadata():
    model_source = Path("modules/api/models.py").read_text()
    extension_source = Path("modules/extensions.py").read_text()

    assert "self.branch = None" in extension_source
    assert "self.commit_date = None" in extension_source
    assert "branch: Optional[str]" in model_source
    assert "commit_date: Optional[int]" in model_source
