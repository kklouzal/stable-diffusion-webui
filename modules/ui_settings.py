"""Setting components for script UIs (extra-options-section shows chosen settings among a script's controls)."""

from modules import headless_ui as gr

from modules.shared import opts


def get_setting_component_args(key):
    info = opts.data_labels[key]
    args = info.component_args() if callable(info.component_args) else info.component_args or {}

    return {k: v for k, v in args.items() if k not in {'precision'}}


def create_setting_component(key):
    def fun():
        return opts.data[key] if key in opts.data else opts.data_labels[key].default

    info = opts.data_labels[key]
    t = type(info.default)

    args = get_setting_component_args(key)

    if info.component is not None:
        comp = info.component
    elif t == str:
        comp = gr.Textbox
    elif t == int:
        comp = gr.Number
    elif t == bool:
        comp = gr.Checkbox
    else:
        raise Exception(f'bad options item type: {t} for key {key}')

    return comp(label=info.label, value=fun(), elem_id=f"setting_{key}", **args)
