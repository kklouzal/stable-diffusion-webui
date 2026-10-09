import importlib.util
import sys
from pathlib import Path


def load_headless_ui():
    # The module registers itself as `gradio` in sys.modules; undo that so other tests are unaffected.
    before = set(sys.modules)
    spec = importlib.util.spec_from_file_location("headless_ui_contract", Path("modules/headless_ui.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        for name in set(sys.modules) - before:
            del sys.modules[name]
    return module


def test_components_accept_the_event_methods_quicksettings_wiring_calls():
    gr = load_headless_ui()
    # Script ui() code and vendored ControlNet UI code wire these events on inert components.
    for component in (gr.Textbox(), gr.Slider(), gr.Dropdown(), gr.Checkbox()):
        for event in ("submit", "blur", "release", "change", "click", "then"):
            assert getattr(component, event)(fn=None, inputs=[], outputs=[]) is component


def test_components_expose_the_attributes_vendored_controlnet_reads():
    gr = load_headless_ui()
    image = gr.Image()
    # controlnet_ui_group.render wraps the upload preprocessor without calling it:
    # image.preprocess = functools.partial(svg_preprocess, preprocess=image.preprocess)
    assert image.preprocess("value") == "value"
    # infotext.py names gr.components.IOComponent (an unevaluated function-scope annotation, kept until ControlNet drops it).
    assert gr.components.IOComponent is gr.components.Component
