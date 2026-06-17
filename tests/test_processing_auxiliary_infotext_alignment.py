from pathlib import Path


def test_returned_mask_images_keep_infotexts_aligned():
    source = Path("modules/processing.py").read_text()

    for append_call in (
        "output_images.append(image_mask)",
        "output_images.append(image_mask_composite)",
    ):
        append_at = source.index(append_call)
        branch_at = source.rfind("if opts.return_", 0, append_at)
        branch_body = source[branch_at:append_at]

        assert "infotexts.append(text)" in branch_body


def test_outpainting_mk2_marks_prepended_grid_as_non_sample():
    source = Path("scripts/outpainting_mk_2.py").read_text()

    grid_branch_at = source.index("return_grid = opts.return_grid and not unwanted_grid_because_of_img_count")
    processed_at = source.index("res = Processed(p, all_images", grid_branch_at)
    processed_block = source[grid_branch_at:source.index("if opts.samples_save:", processed_at)]

    assert "return_grid = opts.return_grid and not unwanted_grid_because_of_img_count" in processed_block
    assert "index_of_first_image = 1 if return_grid else 0" in processed_block
    assert "index_of_first_image=index_of_first_image" in processed_block

def test_xyz_grid_drops_stale_lone_image_infotexts():
    source = Path("scripts/xyz_grid.py").read_text()

    trim_at = source.index("processed.images = processed.images[:z_count + 1] if draw_grid else []")
    trim_block = source[trim_at:source.index("if draw_grid and opts.grid_save:", trim_at)]

    assert "processed.infotexts = processed.infotexts[:z_count + 1] if draw_grid else []" in trim_block
