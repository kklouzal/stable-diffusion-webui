"""Headless stand-in for the removed browser UI build, limited to the state the API server reads.

The API server builds no browser UI, but it depends on these results of building one:
- extension options: script_callbacks.ui_settings_callback(), then the "callbacks" options section and opts.reorder();
- txt2img/img2img script controls, args_from/args_to and infotext fields. Running each script's ui() in the UI section
  order (ordered_ui_categories) fixes the script argument indices that API requests and default script args use;
- the infotext paste fields per tab, which Api.apply_infotext maps back onto request fields;
- the postprocessing scripts' controls.
Script ui() code builds inert modules.headless_ui components.
"""

from modules import headless_ui as gr
from modules import infotext_utils, script_callbacks, scripts, sd_samplers, shared, shared_items
from modules.infotext_utils import PasteField
from modules.options import options_section


def ordered_ui_categories():
    user_order = {x.strip(): i * 2 + 1 for i, x in enumerate(shared.opts.ui_reorder_list)}

    for _, category in sorted(enumerate(shared_items.ui_reorder_categories()), key=lambda x: user_order.get(x[1], x[0] * 2 + 0)):
        yield category


def _styles_from_infotext(d):
    return d["Styles array"] if isinstance(d.get("Styles array"), list) else gr.update()


def _enable_hr_from_infotext(d):
    return "Denoising strength" in d and ("Hires upscale" in d or "Hires upscaler" in d or "Hires resize-1" in d)


def _txt2img_paste_fields():
    return [
        PasteField(None, "Prompt", api="prompt"),
        PasteField(None, "Negative prompt", api="negative_prompt"),
        PasteField(None, "CFG scale", api="cfg_scale"),
        PasteField(None, "Size-1", api="width"),
        PasteField(None, "Size-2", api="height"),
        PasteField(None, "Batch size", api="batch_size"),
        PasteField(None, _styles_from_infotext, api="styles"),
        PasteField(None, "Denoising strength", api="denoising_strength"),
        PasteField(None, _enable_hr_from_infotext, api="enable_hr"),
        PasteField(None, "Hires upscale", api="hr_scale"),
        PasteField(None, "Hires upscaler", api="hr_upscaler"),
        PasteField(None, "Hires steps", api="hr_second_pass_steps"),
        PasteField(None, "Hires resize-1", api="hr_resize_x"),
        PasteField(None, "Hires resize-2", api="hr_resize_y"),
        PasteField(None, "Hires checkpoint", api="hr_checkpoint_name"),
        PasteField(None, sd_samplers.get_hr_sampler_from_infotext, api="hr_sampler_name"),
        PasteField(None, sd_samplers.get_hr_scheduler_from_infotext, api="hr_scheduler"),
        PasteField(None, "Hires prompt", api="hr_prompt"),
        PasteField(None, "Hires negative prompt", api="hr_negative_prompt"),
        *scripts.scripts_txt2img.infotext_fields,
    ]


def _img2img_paste_fields():
    return [
        PasteField(None, "Prompt", api="prompt"),
        PasteField(None, "Negative prompt", api="negative_prompt"),
        PasteField(None, "CFG scale", api="cfg_scale"),
        PasteField(None, "Image CFG scale", api="image_cfg_scale"),
        PasteField(None, "Size-1", api="width"),
        PasteField(None, "Size-2", api="height"),
        PasteField(None, "Batch size", api="batch_size"),
        PasteField(None, _styles_from_infotext, api="styles"),
        PasteField(None, "Denoising strength", api="denoising_strength"),
        PasteField(None, "Mask blur", api="mask_blur"),
        PasteField(None, infotext_utils.inpainting_mask_invert_from_infotext, api="inpainting_mask_invert"),
        PasteField(None, infotext_utils.inpainting_fill_from_infotext, api="inpainting_fill"),
        PasteField(None, infotext_utils.inpaint_full_res_from_infotext, api="inpaint_full_res"),
        PasteField(None, "Masked area padding", api="inpaint_full_res_padding"),
        *scripts.scripts_img2img.infotext_fields,
    ]


def _setup_script_runner(runner, is_img2img):
    scripts.scripts_current = runner
    runner.initialize_scripts(is_img2img=is_img2img)
    runner.prepare_ui()
    for category in ordered_ui_categories():
        if category == "scripts":
            runner.setup_ui()
        runner.setup_ui_for_section(category)


def initialize_script_ui_state():
    infotext_utils.reset()
    script_callbacks.ui_settings_callback()

    _setup_script_runner(scripts.scripts_txt2img, is_img2img=False)
    infotext_utils.add_paste_fields("txt2img", None, _txt2img_paste_fields())

    _setup_script_runner(scripts.scripts_img2img, is_img2img=True)
    img2img_fields = _img2img_paste_fields()
    infotext_utils.add_paste_fields("img2img", None, img2img_fields)
    infotext_utils.add_paste_fields("inpaint", None, img2img_fields)
    scripts.scripts_current = None

    scripts.scripts_postproc.setup_ui()
    infotext_utils.add_paste_fields("extras", None, None)

    # Added last so that scripts have already registered their callbacks.
    shared.opts.data_labels.update(options_section(('callbacks', "Callbacks", "system"), {
        **shared_items.callbacks_order_settings(),
    }))
    shared.opts.reorder()
