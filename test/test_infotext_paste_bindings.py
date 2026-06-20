import importlib
import sys
from types import SimpleNamespace


class FakeButton:
    def __init__(self):
        self.clicks = []

    def click(self, **kwargs):
        self.clicks.append(kwargs)


class FakeGallery:
    pass


def load_infotext_utils():
    sys.modules["modules.headless_ui"] = SimpleNamespace(
        Gallery=FakeGallery,
        components=SimpleNamespace(Component=object),
        update=lambda **kwargs: {"__type__": "generic_update", **kwargs},
    )
    sys.modules["modules.paths"] = SimpleNamespace(data_path="/tmp")
    sys.modules["modules.shared"] = SimpleNamespace(
        opts=SimpleNamespace(send_seed=True),
        cmd_opts=SimpleNamespace(hide_ui_dir_config=True, no_prompt_history=True),
    )
    sys.modules["modules.ui_tempdir"] = SimpleNamespace(is_ui_temp_path=lambda _path: False)
    sys.modules["modules.script_callbacks"] = SimpleNamespace(infotext_pasted_callback=lambda *_args, **_kwargs: None)
    sys.modules["modules.processing"] = SimpleNamespace()
    sys.modules["modules.infotext_versions"] = SimpleNamespace()
    sys.modules["modules.images"] = SimpleNamespace(read=lambda *_args, **_kwargs: None)
    sys.modules["modules.prompt_parser"] = SimpleNamespace()
    sys.modules["modules.errors"] = SimpleNamespace(report=lambda *_args, **_kwargs: None)
    sys.modules.pop("modules.infotext_utils", None)
    sys.modules.pop("modules.generation_parameters_copypaste", None)
    return importlib.import_module("modules.infotext_utils")


def test_source_tab_paste_binding_filters_inputs_and_outputs_by_same_names():
    infotext_utils = load_infotext_utils()
    button = FakeButton()
    source_prompt = object()
    source_seed = object()
    source_steps = object()
    source_extra = object()
    dest_prompt = object()
    dest_seed = object()
    dest_steps = object()
    dest_extra = object()

    infotext_utils.paste_fields["txt2img"] = {
        "init_img": None,
        "fields": [
            (source_prompt, "Prompt"),
            (source_seed, "Seed"),
            (source_steps, "Steps"),
            (source_extra, "Sampler"),
        ],
        "override_settings_component": None,
    }
    infotext_utils.paste_fields["img2img"] = {
        "init_img": None,
        "fields": [
            (dest_prompt, "Prompt"),
            (dest_seed, "Seed"),
            (dest_steps, "Steps"),
            (dest_extra, "Sampler"),
        ],
        "override_settings_component": None,
    }
    infotext_utils.registered_param_bindings[:] = [
        infotext_utils.ParamBinding(button, "img2img", source_tabname="txt2img", paste_field_names=[])
    ]

    infotext_utils.connect_paste_params_buttons()

    copy_click = button.clicks[0]
    assert copy_click["inputs"] == [source_prompt, source_seed, source_steps]
    assert copy_click["outputs"] == [dest_prompt, dest_seed, dest_steps]
    assert source_extra not in copy_click["inputs"]
    assert dest_extra not in copy_click["outputs"]
