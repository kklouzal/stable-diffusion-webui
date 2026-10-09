"""Inert UI component surface used by API/headless startup.

A1111 scripts historically describe their controls by constructing component
objects during startup. The GB10 fork no longer ships or launches a browser UI,
but API endpoints still need those script defaults and metadata. This module
provides the tiny component/event surface needed for that bookkeeping without
pulling in a browser-UI framework.
"""
from __future__ import annotations

import sys
from types import ModuleType
from typing import Any


class _FallbackUpdate(dict):
    def __init__(self, **kwargs: Any):
        super().__init__(kwargs)
        self["__type__"] = "generic_update"


class _FallbackComponent:
    # Keeps only what API script metadata reads (scripts.script_control_api_arg, script defaults, elem_id).
    def __init__(self, *args: Any, **kwargs: Any):
        self.value = kwargs.get("value", args[0] if args else None)
        self.label = kwargs.get("label")
        self.elem_id = kwargs.get("elem_id")
        self.choices = kwargs.get("choices") or []
        self.minimum = kwargs.get("minimum")
        self.maximum = kwargs.get("maximum")
        self.step = kwargs.get("step")

    def __enter__(self):
        return self

    def __exit__(self, exc_type=None, exc=None, tb=None):
        return False

    def then(self, *args: Any, **kwargs: Any): return self
    def click(self, *args: Any, **kwargs: Any): return self
    def change(self, *args: Any, **kwargs: Any): return self
    def submit(self, *args: Any, **kwargs: Any): return self
    def blur(self, *args: Any, **kwargs: Any): return self
    def release(self, *args: Any, **kwargs: Any): return self
    def select(self, *args: Any, **kwargs: Any): return self
    def upload(self, *args: Any, **kwargs: Any): return self
    def load(self, *args: Any, **kwargs: Any): return self
    def render(self, *args: Any, **kwargs: Any): return self

    @staticmethod
    def update(**kwargs: Any):
        return update(**kwargs)


class Interface(_FallbackComponent):
    """Base class for ControlNet's ModalInterface."""


class Blocks(_FallbackComponent):
    pass


def update(**kwargs: Any):
    return _FallbackUpdate(**kwargs)


def Warning(message: str):
    print(f"Warning: {message}")


_COMPONENT_NAMES = {
    "Accordion", "Button", "Checkbox", "CheckboxGroup", "Code", "ColorPicker", "Column", "Dropdown", "File",
    "Gallery", "Group", "HTML", "Image", "Info", "Markdown", "Number", "Plot", "Radio", "Row", "Slider", "State",
    "Tab", "TabItem", "Tabs", "Text", "Textbox", "UploadButton", "Video",
}


for _component_name in _COMPONENT_NAMES:
    globals()[_component_name] = type(_component_name, (_FallbackComponent,), {})


# Third-party extensions may still import the historical UI package name while
# declaring API script controls. Route those imports to this inert headless
# surface instead of requiring the real browser UI dependency.
components = ModuleType("gradio.components")
components.Component = _FallbackComponent
for _component_name in _COMPONENT_NAMES:
    setattr(components, _component_name, globals()[_component_name])

sys.modules.setdefault("gradio", sys.modules[__name__])
sys.modules.setdefault("gradio.components", components)
