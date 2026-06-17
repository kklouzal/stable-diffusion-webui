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


def test_loopback_marks_prepended_grid_as_non_sample():
    source = Path("scripts/loopback.py").read_text()

    grid_join_at = source.index("all_images = grids + all_images")
    processed_block = source[grid_join_at:source.index("return processed", grid_join_at)]

    assert "index_of_first_image = 1 if grids else 0" in processed_block
    assert "index_of_first_image=index_of_first_image" in processed_block


def test_loopback_saves_grid_with_initial_infotext():
    source = Path("scripts/loopback.py").read_text()

    save_call_at = source.index("images.save_image(grid")
    save_call = source[save_call_at:source.index(")", save_call_at)]

    assert "info=initial_info" in save_call


def test_sd_upscale_tracks_per_result_infotexts():
    source = Path("scripts/sd_upscale.py").read_text()

    result_images_at = source.index("result_images = []")
    processed_at = source.index("processed = Processed(p, result_images", result_images_at)
    processed_block = source[result_images_at:source.index("return processed", processed_at)]

    assert "result_infotexts = []" in processed_block
    assert "result_info = None" in processed_block
    assert "result_info = processed.info" in processed_block
    assert "result_infotexts.append(result_info)" in processed_block
    assert "info=result_info" in processed_block
    assert "infotexts=result_infotexts" in processed_block


def test_xyz_grid_marks_first_image_as_sample_when_grid_disabled():
    source = Path("scripts/xyz_grid.py").read_text()

    init_at = source.index("processed_result.images = [None] * list_size")
    init_block = source[init_at:source.index("idx = index(ix, iy, iz)", init_at)]

    assert "processed_result.index_of_first_image = 1 if draw_grid else 0" in init_block


def test_xyz_grid_drops_stale_lone_image_infotexts():
    source = Path("scripts/xyz_grid.py").read_text()

    trim_at = source.index("processed.images = processed.images[:z_count + 1] if draw_grid else []")
    trim_block = source[trim_at:source.index("if draw_grid and opts.grid_save:", trim_at)]

    assert "processed.infotexts = processed.infotexts[:z_count + 1] if draw_grid else []" in trim_block
