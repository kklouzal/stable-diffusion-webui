from pathlib import Path


def _processing():
    """The real modules.processing."""
    from modules import processing

    return processing


def _load_script(name, **module_globals):
    """A private copy of scripts/<name>.py whose module globals in module_globals are replaced."""
    from test.helpers import load_source

    module = load_source(f"{name}_under_test", f"scripts/{name}.py")
    for key, value in module_globals.items():
        setattr(module, key, value)
    return module


def _processed(p, images_list, seed=-1, info="", index_of_first_image=0, infotexts=None, **_kwargs):
    """Records what a script hands to Processed (the real one reads most of a full processing object)."""
    from types import SimpleNamespace

    return SimpleNamespace(images=images_list, seed=seed, info=info, index_of_first_image=index_of_first_image, infotexts=infotexts)


def _image(size=(8, 8)):
    from PIL import Image

    return Image.new("RGB", size, "gray")


def _process_images_inner():
    import ast

    tree = ast.parse(Path("modules/processing.py").read_text(encoding="utf-8"))
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "process_images_inner")


def test_returned_mask_images_keep_infotexts_aligned():
    # process_images_inner runs the whole pipeline (model, sampler, VAE decode), so its output loop is checked on the AST.
    import ast

    inner = _process_images_inner()
    parents = {child: node for node in ast.walk(inner) for child in ast.iter_child_nodes(node)}
    helper = next(node for node in ast.walk(inner) if isinstance(node, ast.FunctionDef) and node.name == "append_output_image")
    helper_nodes = set(ast.walk(helper))

    # Every output image goes through the helper, which appends its infotext alongside.
    assert [ast.unparse(statement) for statement in helper.body] == ["infotexts.append(text)", "output_images.append(output_image)"]
    assert not [
        node for node in ast.walk(inner)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "output_images.append" and node not in helper_nodes
    ]

    branches = {}
    for node in ast.walk(inner):
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "append_output_image":
            branch = parents[node]
            while not isinstance(branch, (ast.If, ast.For)):
                branch = parents[branch]
            branches[ast.unparse(node.args[0])] = ast.unparse(branch.test) if isinstance(branch, ast.If) else None
    assert branches == {"image": None, "image_mask": "opts.return_mask", "image_mask_composite": "opts.return_mask_composite"}


def test_outpainting_mk2_marks_prepended_grid_as_non_sample(initialize, monkeypatch):
    from types import SimpleNamespace

    def process_images(p):
        return SimpleNamespace(seed=7, info="first info", images=[_image((p.width, p.height)) for _ in p.init_images])

    for batch_size, return_grid, expected_images, expected_first in ((2, True, 3, 1), (2, False, 2, 0), (1, True, 1, 0)):
        opts = SimpleNamespace(return_grid=return_grid, grid_only_if_multiple=True, samples_save=False, grid_save=False)
        module = _load_script("outpainting_mk_2", process_images=process_images, Processed=_processed, opts=opts, state=SimpleNamespace())
        p = SimpleNamespace(width=64, height=64, init_images=[_image((64, 64))], n_iter=1, batch_size=batch_size)

        res = module.Script().run(p, None, 64, 4, ["left"], 1.0, 0.05)

        assert len(res.images) == expected_images
        assert res.index_of_first_image == expected_first
        assert (res.seed, res.info) == (7, "first info")


def _run_loopback(monkeypatch, *, batch_count, loops, return_grid, grid_save=False):
    from types import SimpleNamespace

    saved = []
    seeds = iter(range(100, 200))

    def process_images(p):
        seed = next(seeds)
        return SimpleNamespace(seed=seed, info=f"info {seed}", images=[_image()])

    module = _load_script(
        "loopback",
        processing=SimpleNamespace(fix_seed=lambda p: None, setup_color_correction=lambda image: None, process_images=process_images),
        images=SimpleNamespace(image_grid=lambda imgs, rows=None: ("grid", len(imgs)), save_image=lambda image, *args, **kwargs: saved.append((image, args, kwargs))),
        opts=SimpleNamespace(img2img_color_correction=False, grid_save=grid_save, return_grid=return_grid, grid_format="png", grid_extended_filename=False),
        state=SimpleNamespace(interrupted=False, stopping_generation=False, skipped=False),
        Processed=_processed,
    )
    p = SimpleNamespace(n_iter=batch_count, denoising_strength=0.5, init_images=[_image()], prompt="a cat", inpainting_fill=0, seed=1, outpath_grids="grids")
    return module.Script().run(p, loops, 0.8, "Linear", "None"), saved


def test_loopback_marks_prepended_grid_as_non_sample(initialize, monkeypatch):
    processed, _ = _run_loopback(monkeypatch, batch_count=1, loops=3, return_grid=True)
    assert processed.images[0] == ("grid", 3) and len(processed.images) == 4
    assert processed.index_of_first_image == 1

    for batch_count, loops, return_grid in ((1, 3, False), (1, 1, True)):
        processed, _ = _run_loopback(monkeypatch, batch_count=batch_count, loops=loops, return_grid=return_grid)
        assert not any(isinstance(image, tuple) for image in processed.images)
        assert processed.index_of_first_image == 0


def test_loopback_saves_grid_with_initial_infotext(initialize, monkeypatch):
    processed, saved = _run_loopback(monkeypatch, batch_count=1, loops=3, return_grid=False, grid_save=True)

    [(grid, args, kwargs)] = saved
    assert grid == ("grid", 3)
    assert args[2] == 100  # the seed of the first iteration, like the info
    assert kwargs["info"] == "info 100" == processed.info


def test_sd_upscale_tracks_per_result_infotexts(initialize, monkeypatch):
    from types import SimpleNamespace

    from modules import images

    saved = []

    def process_images(p):
        return SimpleNamespace(seed=p.seed, info=f"info seed {p.seed}", images=[_image((p.width, p.height)) for _ in p.init_images])

    module = _load_script(
        "sd_upscale",
        processing=SimpleNamespace(fix_seed=lambda p: None, process_images=process_images),
        shared=SimpleNamespace(sd_upscalers=[SimpleNamespace(name="None")]),
        devices=SimpleNamespace(torch_gc=lambda: None),
        images=SimpleNamespace(flatten=images.flatten, split_grid=images.split_grid, combine_grid=images.combine_grid,
                               save_image=lambda image, *args, info=None, **kwargs: saved.append(info)),
        opts=SimpleNamespace(img2img_background_color="#ffffff", samples_save=True, samples_format="png"),
        state=SimpleNamespace(),
        Processed=_processed,
    )
    # A 96x64 image in 64x64 tiles with 32 px overlap is two tiles, one batch each; two upscales (n_iter).
    p = SimpleNamespace(seed=10, extra_generation_params={}, init_images=[_image((96, 64))], width=64, height=64, batch_size=1, n_iter=2, prompt="", outpath_samples="")

    processed = module.Script().run(p, None, 32, 0, 1.0)

    # Each result carries the infotext of its own first batch (seed 10, then seed 11), not the first result's.
    assert len(processed.images) == 2
    assert processed.infotexts == ["info seed 10", "info seed 11"] == saved
    assert processed.info == "info seed 10"


def _cell(x, y, z, ix, iy, iz):
    from types import SimpleNamespace

    return SimpleNamespace(images=[_image()], prompt=f"prompt {x}", seed=x, infotexts=[f"cell {x}"], width=8, height=8)


def test_xyz_grid_marks_first_image_as_sample_when_grid_disabled(initialize, monkeypatch):
    from types import SimpleNamespace

    module = _load_script("xyz_grid", state=SimpleNamespace())

    for draw_grid, expected_first, expected_images in ((False, 0, 2), (True, 1, 4)):
        result = module.draw_xyz_grid(
            SimpleNamespace(n_iter=1), xs=[1, 2], ys=[0], zs=[0], x_labels=["1", "2"], y_labels=[""], z_labels=[""], cell=_cell,
            draw_legend=False, include_lone_images=True, include_sub_grids=False, first_axes_processed="x", second_axes_processed="y",
            margin_size=0, draw_grid=draw_grid,
        )

        assert result.index_of_first_image == expected_first
        assert len(result.images) == len(result.infotexts) == expected_images
        assert result.infotexts[-2:] == ["cell 1", "cell 2"]


def test_xyz_grid_drops_stale_lone_image_infotexts(initialize, monkeypatch):
    import contextlib
    from types import SimpleNamespace

    def process_images(pc):
        return _cell(pc.seed, None, None, 0, 0, 0)

    module = _load_script(
        "xyz_grid",
        process_images=process_images,
        processing=SimpleNamespace(create_infotext=lambda pc, *args, **kwargs: f"grid {pc.extra_generation_params.get('X Values')}"),
        shared=SimpleNamespace(total_tqdm=SimpleNamespace(updateTotal=lambda total: None), state=SimpleNamespace(interrupted=False)),
        state=SimpleNamespace(stopping_generation=False),
        opts=SimpleNamespace(return_grid=True, img_max_size_mp=1, grid_save=False),
        SharedSettingsStackHelper=contextlib.nullcontext,
    )
    script = module.Script()
    script.current_axis_options = module.axis_options
    seed_axis = next(index for index, option in enumerate(module.axis_options) if option.label == "Seed")

    def run(draw_grid):
        p = SimpleNamespace(batch_size=1, n_iter=1, width=8, height=8, steps=1, seed=1, styles=[], extra_generation_params={}, outpath_grids="",
                            all_prompts=[""], all_seeds=[1], all_subseeds=[1])
        return script.run(p, seed_axis, "1, 2", None, 0, "", None, 0, "", None, False, False, False, True, False, False, False, 0, False, draw_grid)

    without_grid = run(draw_grid=False)
    assert without_grid.images == [] and without_grid.infotexts == []

    with_grid = run(draw_grid=True)
    assert len(with_grid.images) == 1
    assert with_grid.infotexts == ["grid 1, 2"]


def test_processing_branch_shape_helpers_keep_sensitive_semantics_local(initialize, monkeypatch):
    import ast
    from types import SimpleNamespace

    import numpy as np
    import torch
    from PIL import Image

    processing = _processing()

    pixels = (np.arange(2 * 3 * 3, dtype=np.uint8) * 14).reshape(2, 3, 3)
    expected = np.moveaxis(pixels.astype(np.float32) / 255.0, 2, 0)
    unsigned = processing._image_to_chw_float32_array(Image.fromarray(pixels))
    signed = processing._image_to_chw_float32_array(Image.fromarray(pixels), scale_to_signed=True)
    assert unsigned.dtype == signed.dtype == np.float32 and unsigned.shape == (3, 2, 3)
    assert np.array_equal(unsigned, expected) and np.array_equal(signed, expected * 2.0 - 1.0)

    encoded = []

    def images_tensor_to_samples(image, approximation=None, model=None):
        encoded.append((image, approximation))
        return image[:, :, ::8, ::8]

    monkeypatch.setattr(processing, "images_tensor_to_samples", images_tensor_to_samples)
    monkeypatch.setattr(processing.opts, "sd_vae_encode_method", "TAESD", raising=False)
    x = torch.zeros(2, 4, 2, 3, dtype=torch.bfloat16)
    for model in (
        SimpleNamespace(model=SimpleNamespace(conditioning_key="hybrid"), is_sdxl_inpaint=False),
        SimpleNamespace(model=SimpleNamespace(conditioning_key="crossattn"), is_sdxl_inpaint=True),
    ):
        conditioning = processing.txt2img_image_conditioning(model, x, 24, 16)
        image, approximation = encoded.pop()
        # The fully masked image (all 0.5) is encoded, then a mask channel of ones is prepended.
        assert torch.equal(image, torch.full((2, 3, 16, 24), 0.5)) and approximation == processing.approximation_indexes["TAESD"]
        assert conditioning.dtype == torch.bfloat16 and conditioning.shape == (2, 4, 2, 3)
        assert torch.equal(conditioning[:, :1], torch.ones(2, 1, 2, 3, dtype=torch.bfloat16))
        assert torch.equal(conditioning[:, 1:], torch.full((2, 3, 2, 3), 0.5, dtype=torch.bfloat16))

    p = processing.StableDiffusionProcessingTxt2Img.__new__(processing.StableDiffusionProcessingTxt2Img)
    p.extra_generation_params = {}
    p.add_vae_encoder_generation_param()
    assert p.extra_generation_params == {"VAE Encoder": "TAESD"}
    monkeypatch.setattr(processing.opts, "sd_vae_encode_method", "Full", raising=False)
    p.extra_generation_params = {}
    p.add_vae_encoder_generation_param()
    assert p.extra_generation_params == {}

    # Every image-to-latent branch converts and records the encoder through these helpers.
    tree = ast.parse(Path("modules/processing.py").read_text(encoding="utf-8"))
    helpers = {"_image_to_chw_float32_array", "_full_masked_image_conditioning", "self.add_vae_encoder_generation_param"}
    sites = []

    def collect(node, scope):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.ClassDef)):
                collect(child, scope + [child.name])
                continue
            if isinstance(child, ast.Call) and ast.unparse(child.func) in helpers:
                sites.append((ast.unparse(child.func), ".".join(scope)))
            collect(child, scope)

    collect(tree, [])
    assert sorted(sites) == sorted([
        ("_full_masked_image_conditioning", "txt2img_image_conditioning"),
        ("_full_masked_image_conditioning", "txt2img_image_conditioning"),
        ("_image_to_chw_float32_array", "StableDiffusionProcessingTxt2Img.sample"),
        ("_image_to_chw_float32_array", "StableDiffusionProcessingTxt2Img.sample"),
        ("_image_to_chw_float32_array", "StableDiffusionProcessingTxt2Img.sample_hr_pass"),
        ("_image_to_chw_float32_array", "StableDiffusionProcessingImg2Img.init"),
        ("self.add_vae_encoder_generation_param", "StableDiffusionProcessingTxt2Img.sample"),
        ("self.add_vae_encoder_generation_param", "StableDiffusionProcessingTxt2Img.sample_hr_pass"),
        ("self.add_vae_encoder_generation_param", "StableDiffusionProcessingImg2Img.init"),
    ])


def test_processing_loop_reuses_batch_slice_boundaries_without_resetting_seed_state(initialize, monkeypatch):
    # The two slicing sites sit inside process_images_inner (full pipeline), so they are checked on the AST.
    import ast

    processing = _processing()
    assert [processing._batch_slice_range(n, 3) for n in range(3)] == [(0, 3), (3, 6), (6, 9)]

    sites = []
    for node in ast.walk(_process_images_inner()):
        for body in (getattr(node, field, None) for field in ("body", "orelse")):
            if not isinstance(body, list):
                continue
            for index, statement in enumerate(body):
                if isinstance(statement, ast.Assign) and isinstance(statement.value, ast.Call) and ast.unparse(statement.value.func) == "_batch_slice_range":
                    assert ast.unparse(statement) == "batch_start, batch_end = _batch_slice_range(n, p.batch_size)"
                    sliced = []
                    for following in body[index + 1:]:
                        if not (isinstance(following, ast.Assign) and ast.unparse(following.value).endswith("[batch_start:batch_end]")):
                            break
                        sliced.append(ast.unparse(following))
                    assigned = {ast.unparse(target) for other in body if isinstance(other, ast.Assign) for target in other.targets}
                    sites.append((statement.lineno, sliced, assigned))

    (_, first, _), (_, reset, reset_assigned) = sorted(sites)
    assert first == [
        "p.prompts = p.all_prompts[batch_start:batch_end]",
        "p.negative_prompts = p.all_negative_prompts[batch_start:batch_end]",
        "p.seeds = p.all_seeds[batch_start:batch_end]",
        "p.subseeds = p.all_subseeds[batch_start:batch_end]",
    ]
    # The reset before postprocess_batch_list restores the prompt slices only; seed state set by scripts stays.
    assert reset == first[:2]
    assert not {"p.seeds", "p.subseeds"} & reset_assigned


def test_img2img_init_cache_helpers_share_payload_and_stats_boundaries(initialize, monkeypatch):
    import time

    import numpy as np
    import torch
    from PIL import Image

    processing = _processing()
    StableDiffusionProcessing = processing.StableDiffusionProcessing
    monkeypatch.setattr(StableDiffusionProcessing, "cached_img2img_init", [None, None])
    monkeypatch.setattr(StableDiffusionProcessing, "cached_img2img_init_stats", processing._cache_stats(last_hit=False, cached=False, bypass_reason=None))

    def make(**attrs):
        p = processing.StableDiffusionProcessingImg2Img.__new__(processing.StableDiffusionProcessingImg2Img)
        p.__dict__.update(attrs)
        return p

    def payload():
        return {
            "init_latent": torch.ones(1, 4, 2, 2), "image_conditioning": torch.zeros(1, 5, 1, 1), "mask": torch.ones(4, 2, 2),
            "nmask": torch.zeros(4, 2, 2), "mask_for_overlay": Image.new("L", (16, 16), 255), "overlay_images": [Image.new("RGBA", (16, 16))],
            "color_corrections": [np.ones(3)], "paste_to": (0, 0, 16, 16),
        }

    assert set(payload()) == set(processing._IMG2IMG_INIT_CACHE_ATTRS)
    source = make(**payload(), is_using_inpainting_conditioning=True, extra_generation_params={})
    source._store_img2img_init_cache(("key",), time.perf_counter(), {"VAE Encoder": "TAESD"})
    stats = source.openclaw_img2img_init_cache_stats
    assert (stats["last_hit"], stats["cached"], stats["bypass_reason"], stats["hits"], stats["misses"]) == (False, True, None, 0, 1)
    source.init_latent.add_(1)  # the cache holds clones
    source.color_corrections[0][:] = 0

    target = make(extra_generation_params={"Denoising strength": 0.5})
    assert target._restore_img2img_init_cache(("other key",)) is False
    assert target._restore_img2img_init_cache(("key",)) is True
    expected = payload()
    for attr in processing._IMG2IMG_INIT_CACHE_ATTRS:
        restored, value = getattr(target, attr), expected[attr]
        if torch.is_tensor(value):
            assert torch.equal(restored, value)
        elif isinstance(value, Image.Image):
            assert np.array_equal(np.asarray(restored), np.asarray(value))
        elif attr == "overlay_images":
            assert [np.asarray(image).tolist() for image in restored] == [np.asarray(image).tolist() for image in value]
        elif attr == "color_corrections":
            assert [array.tolist() for array in restored] == [array.tolist() for array in value]
        else:
            assert restored == value
    assert target.is_using_inpainting_conditioning is True
    assert target.extra_generation_params == {"Denoising strength": 0.5, "VAE Encoder": "TAESD"}
    stats = target.openclaw_img2img_init_cache_stats
    assert (stats["last_hit"], stats["cached"], stats["bypass_reason"], stats["hits"], stats["misses"]) == (True, True, None, 1, 1)
    assert stats == processing.StableDiffusionProcessingImg2Img.img2img_init_cache_status()

    monkeypatch.setattr(processing.opts, "persistent_img2img_init_cache", False, raising=False)
    bypassed = make()
    assert bypassed._img2img_init_cache_key([], False, False, False) is None
    stats = bypassed.openclaw_img2img_init_cache_stats
    assert (stats["last_hit"], stats["bypass_reason"]) == (False, "disabled")
