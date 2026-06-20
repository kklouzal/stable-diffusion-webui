from pathlib import Path


def test_cover_image_metadata_defaults_to_empty_list_and_rejects_negative_index():
    source = Path("modules/ui_extra_networks.py").read_text()

    assert "metadata.get('ssmd_cover_images', '[]')" in source
    assert "isinstance(cover_images, list)" in source
    assert "0 <= index < len(cover_images)" in source
    assert "metadata.get('ssmd_cover_images', {})" not in source


def test_extra_network_paths_use_commonpath_parent_check():
    source = Path("modules/ui_extra_networks.py").read_text()

    assert "def path_is_parent(parent_path, child_path):" in source
    assert "if path_is_parent(parentdir, abspath):" in source
    assert "if path_is_parent(absdir, filename):" in source
    assert "abspath.startswith(parentdir)" not in source
    assert "filename.startswith(absdir)" not in source
