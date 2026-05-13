"""Inert UI component surface used by API/headless startup.

A1111 scripts historically describe their controls by constructing component
objects during startup. The GB10 fork no longer ships or launches a browser UI,
but API endpoints still need those script defaults and metadata. This module
provides the tiny component/event surface needed for that bookkeeping without
pulling in a browser-UI framework.
"""
from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import Any


def is_available() -> bool:
    return True


def version() -> str | None:
    return None


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

    def style(self, *args: Any, **kwargs: Any): return self
    def then(self, *args: Any, **kwargs: Any): return self
    def success(self, *args: Any, **kwargs: Any): return self
    def click(self, *args: Any, **kwargs: Any): return self
    def change(self, *args: Any, **kwargs: Any): return self
    def submit(self, *args: Any, **kwargs: Any): return self
    def release(self, *args: Any, **kwargs: Any): return self
    def select(self, *args: Any, **kwargs: Any): return self
    def upload(self, *args: Any, **kwargs: Any): return self
    def clear(self, *args: Any, **kwargs: Any): return self
    def load(self, *args: Any, **kwargs: Any): return self
    def render(self, *args: Any, **kwargs: Any): return self

    def get_block_name(self):
        return self.__class__.__name__.lower()

    def get_config(self):
        return {}

    @staticmethod
    def update(**kwargs: Any):
        return update(**kwargs)


class IOComponent(_FallbackComponent):
    pass


class Block(_FallbackComponent):
    pass


class BlockContext(_FallbackComponent):
    pass


class Blocks(_FallbackComponent):
    def queue(self, *args: Any, **kwargs: Any):
        return self

    def launch(self, *args: Any, **kwargs: Any):
        raise RuntimeError("Browser UI launch requested in headless-only build")

    def close(self):
        pass

    def get_config_file(self, *args: Any, **kwargs: Any):
        return {"components": []}


class Request:
    pass


def update(**kwargs: Any):
    return _FallbackUpdate(**kwargs)


def Warning(message: str):
    print(f"Warning: {message}")


_COMPONENT_NAMES = {
    "Accordion", "Audio", "Box", "Button", "Checkbox", "CheckboxGroup", "Code",
    "ColorPicker", "Column", "Dataframe", "Dataset", "Dropdown", "File",
    "Files", "Gallery", "Group", "HTML", "HighlightedText", "Image", "Info", "JSON", "Label",
    "Markdown", "Number", "Plot", "Radio", "Row", "SelectData", "Slider", "State", "Tab", "TabItem", "Tabs",
    "Text", "TextArea", "Textbox", "Video",
}


for _component_name in _COMPONENT_NAMES:
    globals()[_component_name] = type(_component_name, (_FallbackComponent,), {})


def __getattr__(name: str) -> Any:
    if name == "__version__":
        return None
    if name in _COMPONENT_NAMES:
        return globals()[name]
    if name == "themes":
        class ThemeClass(_FallbackComponent):
            @classmethod
            def load(cls, *args: Any, **kwargs: Any):
                return cls(*args, **kwargs)

            @classmethod
            def from_hub(cls, *args: Any, **kwargs: Any):
                return cls(*args, **kwargs)

            def dump(self, *args: Any, **kwargs: Any):
                pass

        return SimpleNamespace(
            Base=lambda *args, **kwargs: ThemeClass(*args, **kwargs),
            Default=lambda *args, **kwargs: ThemeClass(*args, **kwargs),
            ThemeClass=ThemeClass,
        )
    if name == "deprecation":
        return SimpleNamespace(UIDeprecationWarning=DeprecationWarning)
    if name == "utils":
        return SimpleNamespace(version_check=lambda: None, get_local_ip_address=lambda: "127.0.0.1")
    if name == "components":
        return components
    if name == "blocks":
        return blocks
    if name == "routes":
        return SimpleNamespace(templates=SimpleNamespace(TemplateResponse=lambda *args, **kwargs: None))
    raise AttributeError(name)


# Third-party extensions may still import the historical UI package name while
# declaring API script controls. Route those imports to this inert headless
# surface instead of requiring the real browser UI dependency.
components = ModuleType("gradio.components")
components.IOComponent = IOComponent
components.Component = _FallbackComponent
for _component_name in _COMPONENT_NAMES:
    setattr(components, _component_name, globals()[_component_name])

blocks = ModuleType("gradio.blocks")
blocks.Block = Block
blocks.BlockContext = BlockContext
blocks.Blocks = Blocks

routes = ModuleType("gradio.routes")
routes.templates = SimpleNamespace(TemplateResponse=lambda *args, **kwargs: None)

sys.modules.setdefault("gradio", sys.modules[__name__])
sys.modules.setdefault("gradio.components", components)
sys.modules.setdefault("gradio.blocks", blocks)
sys.modules.setdefault("gradio.routes", routes)
