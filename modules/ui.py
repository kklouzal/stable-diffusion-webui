"""Names that scripts and extensions still import from modules.ui.

The browser UI was removed from this fork. The API server builds its script and infotext state in
modules/headless_setup.py.
"""

random_symbol = '\U0001f3b2\ufe0f'  # 🎲️
reuse_symbol = '\u267b\ufe0f'  # ♻️
switch_values_symbol = '\U000021C5' # ⇅


def gr_show(visible=True):
    return {"visible": visible, "__type__": "update"}
