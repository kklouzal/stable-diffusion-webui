"""Compatibility boundary for Gradio.

Import this module as gr from code that only needs Gradio component
metadata or UI construction helpers. When Gradio is installed, attributes are
proxied to the real package. When it is not installed, a minimal inert surface
is provided so API/headless code can import option/script metadata without
pulling Gradio in as a hard dependency.
"""
from __future__ import annotations

from types import ModuleType, SimpleNamespace
import sys
import warnings
from typing import Any

try:  # pragma: no cover - exercised in the real UI environment
    import gradio as _gradio
except Exception:  # keep API/headless imports independent from gradio
    _gradio = None


def is_available() -> bool:
    return _gradio is not None


def version() -> str | None:
    return getattr(_gradio, "__version__", None) if _gradio is not None else None


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
        self.choices = kwargs.get("choices")
        self.elem_classes = kwargs.get("elem_classes") or []

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
        raise RuntimeError("Gradio UI launch requested but gradio is not installed")

    def close(self):
        pass

    def get_config_file(self, *args: Any, **kwargs: Any):
        return {"components": []}


class Request:
    pass


def update(**kwargs: Any):
    if _gradio is not None:
        return _gradio.update(**kwargs)
    return _FallbackUpdate(**kwargs)


def Warning(message: str):
    if _gradio is not None:
        return _gradio.Warning(message)
    print(f"Warning: {message}")


_COMPONENT_NAMES = {
    "Accordion", "Audio", "Box", "Button", "Checkbox", "CheckboxGroup", "Code",
    "ColorPicker", "Column", "Dataframe", "Dataset", "Dropdown", "File",
    "Files", "Gallery", "Group", "HTML", "HighlightedText", "Image", "Info", "JSON", "Label",
    "Markdown", "Number", "Plot", "Radio", "Row", "SelectData", "Slider", "State", "Tab", "TabItem", "Tabs",
    "Text", "TextArea", "Textbox", "Video",
}


def __getattr__(name: str) -> Any:
    if _gradio is not None:
        return getattr(_gradio, name)
    if name == "__version__":
        return None
    if name in _COMPONENT_NAMES:
        return type(name, (_FallbackComponent,), {})
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
        return SimpleNamespace(GradioDeprecationWarning=DeprecationWarning)
    if name == "utils":
        return SimpleNamespace(version_check=lambda: None, get_local_ip_address=lambda: "127.0.0.1")
    if name == "components":
        return SimpleNamespace(IOComponent=IOComponent, Component=_FallbackComponent)
    if name == "blocks":
        return SimpleNamespace(Block=Block, BlockContext=BlockContext, Blocks=Blocks)
    if name == "routes":
        return SimpleNamespace(templates=SimpleNamespace(TemplateResponse=lambda *args, **kwargs: None))
    raise AttributeError(name)


def _fallback_module(name: str, **attrs: Any) -> ModuleType:
    module = ModuleType(name)
    module.__dict__.update(attrs)
    return module


def _install_fallback_import_modules() -> None:
    """Expose enough Gradio-shaped modules for legacy extensions.

    Repo-owned code imports this compatibility boundary directly, but installed
    third-party extensions may still use imports such as
    ``from gradio.components import Component``. These modules are inert and are
    intended only for API/headless startup without the real Gradio package.
    """
    current = sys.modules[__name__]
    sys.modules.setdefault("gradio", current)
    sys.modules.setdefault(
        "gradio.components",
        _fallback_module("gradio.components", IOComponent=IOComponent, Component=_FallbackComponent),
    )
    sys.modules.setdefault(
        "gradio.blocks",
        _fallback_module("gradio.blocks", Block=Block, BlockContext=BlockContext, Blocks=Blocks),
    )
    sys.modules.setdefault(
        "gradio.routes",
        _fallback_module("gradio.routes", templates=SimpleNamespace(TemplateResponse=lambda *args, **kwargs: None)),
    )
    sys.modules.setdefault(
        "gradio.deprecation",
        _fallback_module("gradio.deprecation", GradioDeprecationWarning=DeprecationWarning),
    )


# Let legacy/third-party extensions that still use import gradio as gr keep
# importing in Gradio-free API/headless mode. Repo-owned code should import this
# boundary explicitly as from modules import gradio_compat as gr.
if _gradio is None:
    _install_fallback_import_modules()
