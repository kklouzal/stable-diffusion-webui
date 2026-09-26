import os
from pathlib import Path

from PIL import ImageOps, UnidentifiedImageError

from modules import images
from modules.infotext_utils import parse_generation_parameters
from modules.processing import process_images
from modules.shared import state
from modules.sd_models import get_closet_checkpoint_match
import modules.shared as shared
import modules.processing as processing
import modules.scripts


def process_batch(p, input, output_dir, inpaint_mask_dir, args, to_scale=False, scale_by=1.0, use_png_info=False, png_info_props=None, png_info_dir=None):
    output_dir = output_dir.strip()
    processing.fix_seed(p)

    if isinstance(input, str):
        batch_images = list(shared.walk_files(input, allowed_extensions=(".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff")))
    else:
        batch_images = [os.path.abspath(x.name) for x in input]

    is_inpaint_batch = False
    if inpaint_mask_dir:
        inpaint_masks = shared.listfiles(inpaint_mask_dir)
        is_inpaint_batch = bool(inpaint_masks)

        if is_inpaint_batch:
            print(f"\nInpaint batch is enabled. {len(inpaint_masks)} masks found.")

    print(f"Will process {len(batch_images)} images, creating {p.n_iter * p.batch_size} new images for each.")

    state.job_count = len(batch_images) * p.n_iter

    # extract "default" params to use in case getting png info fails
    prompt = p.prompt
    negative_prompt = p.negative_prompt
    seed = p.seed
    cfg_scale = p.cfg_scale
    sampler_name = p.sampler_name
    steps = p.steps
    override_settings = p.override_settings
    sd_model_checkpoint_override = get_closet_checkpoint_match(override_settings.get("sd_model_checkpoint", None))
    batch_results = None
    discard_further_results = False
    for i, image in enumerate(batch_images):
        state.job = f"{i+1} out of {len(batch_images)}"
        if state.skipped:
            state.skipped = False

        if state.interrupted or state.stopping_generation:
            break

        try:
            img = images.read(image)
        except UnidentifiedImageError as e:
            print(e)
            continue
        # Use the EXIF orientation of photos taken by smartphones.
        img = ImageOps.exif_transpose(img)

        if to_scale:
            p.width = int(img.width * scale_by)
            p.height = int(img.height * scale_by)

        p.init_images = [img] * p.batch_size

        image_path = Path(image)
        if is_inpaint_batch:
            # try to find corresponding mask for an image using simple filename matching
            if len(inpaint_masks) == 1:
                mask_image_path = inpaint_masks[0]
            else:
                # try to find corresponding mask for an image using simple filename matching
                mask_image_dir = Path(inpaint_mask_dir)
                masks_found = list(mask_image_dir.glob(f"{image_path.stem}.*"))

                if len(masks_found) == 0:
                    print(f"Warning: mask is not found for {image_path} in {mask_image_dir}. Skipping it.")
                    continue

                # it should contain only 1 matching mask
                # otherwise user has many masks with the same name but different extensions
                mask_image_path = masks_found[0]

            mask_image = images.read(mask_image_path)
            p.image_mask = mask_image

        if use_png_info:
            try:
                info_img = img
                if png_info_dir:
                    info_img_path = os.path.join(png_info_dir, os.path.basename(image))
                    info_img = images.read(info_img_path)
                geninfo, _ = images.read_info_from_image(info_img)
                parsed_parameters = parse_generation_parameters(geninfo)
                parsed_parameters = {k: v for k, v in parsed_parameters.items() if k in (png_info_props or {})}
            except Exception:
                parsed_parameters = {}

            p.prompt = prompt + (" " + parsed_parameters["Prompt"] if "Prompt" in parsed_parameters else "")
            p.negative_prompt = negative_prompt + (" " + parsed_parameters["Negative prompt"] if "Negative prompt" in parsed_parameters else "")
            p.seed = int(parsed_parameters.get("Seed", seed))
            p.cfg_scale = float(parsed_parameters.get("CFG scale", cfg_scale))
            p.sampler_name = parsed_parameters.get("Sampler", sampler_name)
            p.steps = int(parsed_parameters.get("Steps", steps))

            model_info = get_closet_checkpoint_match(parsed_parameters.get("Model hash", None))
            if model_info is not None:
                p.override_settings['sd_model_checkpoint'] = model_info.name
            elif sd_model_checkpoint_override:
                p.override_settings['sd_model_checkpoint'] = sd_model_checkpoint_override.name
            else:
                p.override_settings.pop("sd_model_checkpoint", None)

        if output_dir:
            p.outpath_samples = output_dir
            p.override_settings['save_to_dirs'] = False
            p.override_settings['save_images_replace_action'] = "Add number suffix"
            if p.n_iter > 1 or p.batch_size > 1:
                p.override_settings['samples_filename_pattern'] = f'{image_path.stem}-[generation_number]'
            else:
                p.override_settings['samples_filename_pattern'] = f'{image_path.stem}'

        proc = modules.scripts.scripts_img2img.run(p, *args)

        if proc is None:
            p.override_settings.pop('save_images_replace_action', None)
            proc = process_images(p)

        if not discard_further_results and proc:
            if batch_results:
                batch_results.images.extend(proc.images)
                batch_results.infotexts.extend(proc.infotexts)
            else:
                batch_results = proc

            if 0 <= shared.opts.img2img_batch_show_results_limit < len(batch_results.images):
                discard_further_results = True
                batch_results.images = batch_results.images[:int(shared.opts.img2img_batch_show_results_limit)]
                batch_results.infotexts = batch_results.infotexts[:int(shared.opts.img2img_batch_show_results_limit)]

    return batch_results
