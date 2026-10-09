from test.helpers import load_source


def load_headless_ui():
    # The module registers itself as `gradio` and `gradio.components` (sys.modules.setdefault): hide those names while
    # it executes and restore them afterwards, so other tests are unaffected.
    return load_source("headless_ui_contract", "modules/headless_ui.py", {"gradio": None, "gradio.components": None})


def test_components_accept_the_event_methods_quicksettings_wiring_calls():
    gr = load_headless_ui()
    # Script ui() code and vendored ControlNet UI code wire these events on inert components.
    for component in (gr.Textbox(), gr.Slider(), gr.Dropdown(), gr.Checkbox()):
        for event in ("submit", "blur", "release", "change", "click", "then"):
            assert getattr(component, event)(fn=None, inputs=[], outputs=[]) is component


def test_components_module_serves_third_party_imports():
    gr = load_headless_ui()
    # MultiDiffusion imports gradio.components.Component; Detail Daemon checks isinstance(x, gr.components.Slider).
    assert gr.components.Component is gr.components.Slider.__mro__[1]
    assert isinstance(gr.Slider(), gr.components.Slider)
