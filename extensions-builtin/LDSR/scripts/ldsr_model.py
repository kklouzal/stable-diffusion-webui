"""API surface left behind by the removed LDSR upscaler.

The owner keeps every option key and CLI flag, so this extension still registers the options `ldsr_steps` and
`ldsr_cached` (here) and the `--ldsr-models-path` flag (preload.py), from the same places as before, so the order of
/sdapi/v1/options, /sdapi/v1/cmd-flags and the openapi schemas is unchanged. Nothing reads them.
"""

from modules import shared, script_callbacks


def on_ui_settings():
    from modules import headless_ui as gr

    shared.opts.add_option("ldsr_steps", shared.OptionInfo(100, "LDSR processing steps. Lower = faster", gr.Slider, {"minimum": 1, "maximum": 200, "step": 1}, section=('upscaling', "Upscaling")))
    shared.opts.add_option("ldsr_cached", shared.OptionInfo(False, "Cache LDSR model in memory", gr.Checkbox, {"interactive": True}, section=('upscaling', "Upscaling")))


script_callbacks.on_ui_settings(on_ui_settings)
