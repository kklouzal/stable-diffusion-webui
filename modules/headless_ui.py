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
    def __init__(self, *args: Any, **kwargs: Any):
        self.args = args
        self.kwargs = kwargs
        self.value = kwargs.get("value", args[0] if args else None)
        self.label = kwargs.get("label")
        self.elem_id = kwargs.get("elem_id")
        self.visible = kwargs.get("visible", True)
        self.choices = kwargs.get("choices") or []
        self.elem_classes = kwargs.get("elem_classes") or []
        self.children = kwargs.get("children") or []
        self.id = kwargs.get("id", self.elem_id)
        self.selected = kwargs.get("selected")
        self.minimum = kwargs.get("minimum")
        self.maximum = kwargs.get("maximum")
        self.step = kwargs.get("step")
        self.multiselect = kwargs.get("multiselect", False)
        self.open = kwargs.get("open", self.value)
        self.do_not_save_to_config = kwargs.get("do_not_save_to_config", False)

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
    def preprocess(self, value: Any): return value

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
    "Accordion", "Audio", "Box", "Button", "Checkbox", "CheckboxGroup", "Code",
    "ColorPicker", "Column", "Dropdown", "File",
    "Files", "Gallery", "Group", "HTML", "HighlightedText", "Image", "Info", "Label",
    "Markdown", "Number", "Plot", "Radio", "Row", "SelectData", "Slider", "State", "Tab", "TabItem", "Tabs",
    "Text", "TextArea", "Textbox", "UploadButton", "Video",
}


for _component_name in _COMPONENT_NAMES:
    globals()[_component_name] = type(_component_name, (_FallbackComponent,), {})


# Third-party extensions may still import the historical UI package name while
# declaring API script controls. Route those imports to this inert headless
# surface instead of requiring the real browser UI dependency.
components = ModuleType("gradio.components")
components.Component = _FallbackComponent
components.IOComponent = _FallbackComponent  # named in vendored ControlNet annotations
for _component_name in _COMPONENT_NAMES:
    setattr(components, _component_name, globals()[_component_name])

sys.modules.setdefault("gradio", sys.modules[__name__])
sys.modules.setdefault("gradio.components", components)
