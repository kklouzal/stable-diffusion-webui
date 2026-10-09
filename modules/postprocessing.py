import numpy as np
from PIL import Image

from modules import shared, images, scripts, scripts_postprocessing, ui_common, infotext_utils
from modules.shared import opts


def to_postprocessing_mode(image):
    """RGB or RGBA input for the postprocessing scripts.

    Images with transparency (LA, PA, La, P/L with a transparency key) become RGBA like RGBA input instead of
    losing the alpha (and showing the hidden colours); 16-bit grayscale is scaled to 8 bits (65535 -> 255, rounded)
    where convert("RGB") clipped every value above 255 to white.
    """
    if image.mode in ("RGBA", "RGB"):
        return image
    if image.has_transparency_data:
        return image.convert("RGBA")
    if image.mode.startswith("I;16"):
        # round(v / 257) in integers; Pillow's point() rejects I;16B/I;16L. fromarray drops info, which carries the
        # infotext read right after this.
        values = np.asarray(image).astype(np.uint32)
        eight_bit = Image.fromarray(((values * 2 + 257) // 514).astype(np.uint8))
        eight_bit.info.update(image.info)
        image = eight_bit
    return image.convert("RGB")


def run_postprocessing(extras_mode, image, image_folder, *args, scripts_order=None):
    """The API extras run: mode 1 processes the decoded PIL images of image_folder, any other mode the single image.
    Returns (output images, infotext HTML, ''); nothing is saved to disk."""
    # State.begin and State.end release cached device memory (devices.torch_gc) themselves.
    shared.state.begin(job="extras")
    try:
        outputs = []

        if extras_mode == 1:
            data_to_process = [(images.fix_image(img), '') for img in image_folder]
        else:
            assert image, 'image not selected'
            data_to_process = [(image, None)]

        infotext = ''

        shared.state.job_count = len(data_to_process)

        for image_data, name in data_to_process:
            shared.state.nextjob()
            shared.state.textinfo = name
            shared.state.skipped = False

            if shared.state.interrupted or shared.state.stopping_generation:
                break

            image_data = to_postprocessing_mode(image_data)

            parameters, existing_pnginfo = images.read_info_from_image(image_data)
            if parameters:
                existing_pnginfo["parameters"] = parameters

            initial_pp = scripts_postprocessing.PostprocessedImage(image_data)

            scripts.scripts_postproc.run(initial_pp, args, scripts_order=scripts_order)

            if shared.state.skipped:
                continue

            for pp in [initial_pp, *initial_pp.extra_images]:
                if shared.state.skipped:
                    break

                infotext = ", ".join([k if k == v else f'{k}: {infotext_utils.quote(v)}' for k, v in pp.info.items() if v is not None])

                if opts.enable_pnginfo:
                    # a dict per image: sharing existing_pnginfo gave every output of this input the last one's infotext
                    pp.image.info = {**existing_pnginfo, "postprocessing": infotext}

                shared.state.assign_current_image(pp.image)
                outputs.append(pp.image)

        return outputs, ui_common.plaintext_to_html(infotext), ''
    finally:
        # a failed input (e.g. an upscaler that cannot load) must not leave the "extras" job active
        shared.state.end()


def run_extras(extras_mode, resize_mode, image, image_folder, gfpgan_visibility, codeformer_visibility, codeformer_weight, upscaling_resize, upscaling_resize_w, upscaling_resize_h, upscaling_crop, extras_upscaler_1, extras_upscaler_2, extras_upscaler_2_visibility, upscale_first: bool, max_side_length: int = 0):
    """The /extra-single-image and /extra-batch-images handler: maps the request fields onto the postprocessing scripts."""

    scripts_order = ["Upscale", "GFPGAN", "CodeFormer"] if upscale_first else ["GFPGAN", "CodeFormer", "Upscale"]

    args = scripts.scripts_postproc.create_args_for_run({
        "Upscale": {
            "upscale_enabled": True,
            "upscale_mode": resize_mode,
            "upscale_by": upscaling_resize,
            "max_side_length": max_side_length,
            "upscale_to_width": upscaling_resize_w,
            "upscale_to_height": upscaling_resize_h,
            "upscale_crop": upscaling_crop,
            "upscaler_1_name": extras_upscaler_1,
            "upscaler_2_name": extras_upscaler_2,
            "upscaler_2_visibility": extras_upscaler_2_visibility,
        },
        "GFPGAN": {
            "enable": True,
            "gfpgan_visibility": gfpgan_visibility,
        },
        "CodeFormer": {
            "enable": True,
            "codeformer_visibility": codeformer_visibility,
            "codeformer_weight": codeformer_weight,
        },
    }, scripts_order=scripts_order)

    return run_postprocessing(extras_mode, image, image_folder, *args, scripts_order=scripts_order)
