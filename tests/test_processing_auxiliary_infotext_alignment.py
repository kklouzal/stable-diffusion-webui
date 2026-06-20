from pathlib import Path


def test_returned_mask_images_keep_infotexts_aligned():
    source = Path("modules/processing.py").read_text()

    helper_at = source.index("def append_output_image(output_image):")
    helper_body = source[helper_at:source.index("append_output_image(image)", helper_at)]
    assert "infotexts.append(text)" in helper_body
    assert "output_images.append(output_image)" in helper_body

    for append_call in (
        "append_output_image(image_mask)",
        "append_output_image(image_mask_composite)",
    ):
        append_at = source.index(append_call)
        branch_at = source.rfind("if opts.return_", 0, append_at)
        branch_body = source[branch_at:append_at + len(append_call)]

        assert append_call in branch_body


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


def test_processing_branch_shape_helpers_keep_sensitive_semantics_local():
    source = Path("modules/processing.py").read_text()

    image_helper_at = source.index("def _image_to_chw_float32_array")
    image_helper_body = source[image_helper_at:source.index("def _full_masked_image_conditioning", image_helper_at)]
    assert "np.array(image).astype(np.float32) / 255.0" in image_helper_body
    assert "image = image * 2.0 - 1.0" in image_helper_body
    assert "return np.moveaxis(image, 2, 0)" in image_helper_body
    assert source.count("_image_to_chw_float32_array(") == 4

    helper_at = source.index("def _full_masked_image_conditioning")
    helper_body = source[helper_at:source.index("def txt2img_image_conditioning", helper_at)]
    assert "torch.ones(x.shape[0], 3, height, width, device=x.device) * 0.5" in helper_body
    assert "torch.nn.functional.pad" in helper_body
    assert source.count("return _full_masked_image_conditioning(sd_model, x, width, height)") == 2

    vae_helper_at = source.index("def add_vae_encoder_generation_param(self):")
    vae_helper_body = source[vae_helper_at:source.index("def setup_prompts", vae_helper_at)]
    assert "opts.sd_vae_encode_method != 'Full'" in vae_helper_body
    assert "self.extra_generation_params['VAE Encoder']" in vae_helper_body
    assert source.count("self.add_vae_encoder_generation_param()") == 3


def test_processing_loop_reuses_batch_slice_boundaries_without_resetting_seed_state():
    source = Path("modules/processing.py").read_text()

    helper_at = source.index("def _batch_slice_range")
    helper_body = source[helper_at:source.index("def program_version", helper_at)]
    assert "batch_start = batch_number * batch_size" in helper_body
    assert "return batch_start, batch_start + batch_size" in helper_body

    first_setup_at = source.index("p.seeds = p.all_seeds[batch_start:batch_end]")
    first_setup_block = source[source.rfind("batch_start, batch_end = _batch_slice_range", 0, first_setup_at):source.index("latent_channels =", first_setup_at)]
    assert "p.prompts = p.all_prompts[batch_start:batch_end]" in first_setup_block
    assert "p.negative_prompts = p.all_negative_prompts[batch_start:batch_end]" in first_setup_block
    assert "p.seeds = p.all_seeds[batch_start:batch_end]" in first_setup_block
    assert "p.subseeds = p.all_subseeds[batch_start:batch_end]" in first_setup_block

    reset_at = source.index("batch_params = scripts.PostprocessBatchListArgs")
    reset_block = source[source.rfind("batch_start, batch_end = _batch_slice_range", 0, reset_at):reset_at]
    assert "p.prompts = p.all_prompts[batch_start:batch_end]" in reset_block
    assert "p.negative_prompts = p.all_negative_prompts[batch_start:batch_end]" in reset_block
    assert "p.seeds =" not in reset_block
    assert "p.subseeds =" not in reset_block


def test_img2img_init_cache_helpers_share_payload_and_stats_boundaries():
    source = Path("modules/processing.py").read_text()

    attrs_at = source.index("_IMG2IMG_INIT_CACHE_ATTRS =")
    attrs_line = source[attrs_at:source.index("\n", attrs_at)]
    for attr in ("init_latent", "image_conditioning", "mask", "nmask", "mask_for_overlay", "overlay_images", "color_corrections", "paste_to"):
        assert attr in attrs_line

    restore_at = source.index("def _restore_img2img_init_cache")
    restore_body = source[restore_at:source.index("def _store_img2img_init_cache", restore_at)]
    assert "for attr in _IMG2IMG_INIT_CACHE_ATTRS" in restore_body
    assert "setattr(self, attr, _clone_cache_value(payload.get(attr)))" in restore_body

    store_at = source.index("def _store_img2img_init_cache")
    store_body = source[store_at:source.index("def init(self, all_prompts", store_at)]
    assert "for attr in _IMG2IMG_INIT_CACHE_ATTRS" in store_body
    assert '"is_using_inpainting_conditioning": self.is_using_inpainting_conditioning' in store_body
    assert '"extra_generation_params": _clone_cache_value(extra_generation_params)' in store_body

    stats_at = source.index("def _snapshot_img2img_init_cache_stats")
    stats_body = source[stats_at:source.index("def _img2img_init_cache_bypass_reason", stats_at)]
    assert stats_body.count("_snapshot_img2img_init_cache_stats(") == 4
    assert 'self.openclaw_img2img_init_cache_stats = dict(stats)' in stats_body
