"""Names that extensions still import from modules.ui (multidiffusion-upscaler imports gr_show).

The browser UI was removed from this fork. The API server builds its script and infotext state in
modules/headless_setup.py.
"""


def gr_show(visible=True):
    return {"visible": visible, "__type__": "update"}
