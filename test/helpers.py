"""Shared support for the tests in test/ (and the extension suites, which run from the repo root).

Module stubs: a test that exercises one webui source file without importing the runtime around it installs stand-in
modules in sys.modules for exactly as long as it needs them, with `stub_modules` (a context manager; inside a fixture,
yield from within it) or `load_source`, or with `monkeypatch.setitem(sys.modules, ...)` inside a test. All of them
restore only the names they touched. Never use `mock.patch.dict(sys.modules)`: on exit it also drops every module
imported meanwhile (torch submodules, ...), so the next importer gets fresh copies whose classes no longer match the
ones other modules already bound. Never install a stub at collection time without restoring it: it stays for the rest
of the session and every later `from modules import shared` gets it.

This module imports only the standard library, so stub-only tests stay cheap; helpers that need torch import it when
called.
"""
from __future__ import annotations

import contextlib
import importlib.util
import sys
import types
from collections.abc import Iterator, Mapping
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST_FILES = Path(__file__).resolve().parent / "test_files"

_ABSENT = object()


def init_shared():
    """The real modules.shared, initialized once per process the way webui startup does (shared_init.initialize()).

    Test modules that import webui modules needing shared.opts/state at import time call this at module level first.
    Initializing only when opts is still None matters: a second shared_init.initialize() would build a new opts while
    modules imported since keep the one they bound with `from modules.shared import opts`."""
    from modules import shared, shared_init

    if shared.opts is None:
        shared_init.initialize()
    return shared


def module(name: str, *, package: bool = False, **attrs) -> types.ModuleType:
    """A new module `name` holding `attrs`. package=True gives it an empty __path__, so its submodules resolve only
    through sys.modules, never from disk."""
    mod = types.ModuleType(name)
    if package:
        mod.__path__ = []
    for key, value in attrs.items():
        setattr(mod, key, value)
    return mod


@contextlib.contextmanager
def stub_modules(stubs: Mapping[str, types.ModuleType | None]) -> Iterator[None]:
    """Install `stubs` in sys.modules for the block (a None value hides that name), then restore exactly the touched
    names: the previous module, or absent when there was none. Modules imported meanwhile under other names stay.

    `from package import name` reads the package attribute before sys.modules["package.name"]: to hand stubbed
    submodules to code that imports them that way while the real package is loaded, stub the package as well (or set
    the attribute with monkeypatch)."""
    previous = {name: sys.modules.get(name, _ABSENT) for name in stubs}
    try:
        for name, value in stubs.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value
        yield
    finally:
        for name, value in previous.items():
            if value is _ABSENT:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


def load_source(name: str, path: str | Path, stubs: Mapping[str, types.ModuleType | None] | None = None) -> types.ModuleType:
    """Execute the source file `path` (relative paths resolve against the repo root) as a new module `name` while
    `stubs` are installed, and return it. The module is sys.modules[name] while it executes (dataclasses and
    `typing.get_type_hints` look it up there); afterwards that name and the stubs are restored. Imports the module runs
    later, inside its functions, see whatever sys.modules holds then: keep the stubs installed around those calls."""
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    mod = importlib.util.module_from_spec(spec)
    with stub_modules({**(stubs or {}), name: mod}):
        spec.loader.exec_module(mod)
    return mod


def add_repositories_to_sys_path(*repos: str) -> None:
    """Put repositories/<repo> on sys.path (what modules/paths.py does at startup), for tests that import ldm/sgm
    without the rest of modules.paths' start-up side effects."""
    for repo in repos:
        path = str(ROOT / "repositories" / repo)
        if Path(path).is_dir() and path not in sys.path:
            sys.path.insert(0, path)


def error_over_rms(output, reference):
    """(max, mean) |output - reference| / rms(reference), against a float64 reference. Distances in bf16 ULPs mean
    nothing for outputs near zero, where affine terms cancel; this measures error as consumed downstream."""
    diff = (output.double() - reference).abs()
    rms = reference.pow(2).mean().sqrt()
    return (diff.max() / rms).item(), (diff.mean() / rms).item()


def random_activations(shape, memory_format=None):
    """bf16 CUDA activations with per-channel scales from 1e-3 to 10 and offsets of a few scales (seeded). For NCHW
    the scale is shared by each run of channels / 32 channels (one GroupNorm group), so group variances span
    1e-6..100, including the range where eps matters most."""
    import torch

    generator = torch.Generator(device="cuda").manual_seed(1234)
    if len(shape) == 4:
        channels = shape[1]
        scale = torch.logspace(-3, 1, 32, device="cuda").repeat_interleave(channels // 32)
        view = (1, channels, 1, 1)
    else:
        channels = shape[-1]
        scale = torch.logspace(-3, 1, channels, device="cuda")
        view = (channels,)
    offset = torch.randn(channels, device="cuda", generator=generator) * scale * 3
    x = torch.randn(shape, device="cuda", generator=generator) * scale.view(view) + offset.view(view)
    return x.to(torch.bfloat16).contiguous(memory_format=memory_format or torch.contiguous_format)


def randomize(module):
    """Fill `module`'s parameters with seeded values (weights ~ N(1, 0.25) for 1-d parameters, N(0, 0.25) otherwise)."""
    import torch

    generator = torch.Generator().manual_seed(4321)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.5 + (1.0 if parameter.dim() == 1 else 0.0))
    return module


class WatchingDict(dict):
    """A dict that records every mutation after which the watched key is missing (for registries that must never be
    observed without a published name)."""

    def __init__(self, *args, watch):
        super().__init__(*args)
        self.watch = watch
        self.missing_after = []

    def _check(self, op):
        if self.watch not in self:
            self.missing_after.append(op)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self._check("setitem")

    def __delitem__(self, key):
        super().__delitem__(key)
        self._check("delitem")

    def pop(self, *args):
        result = super().pop(*args)
        self._check("pop")
        return result

    def update(self, *args, **kwargs):
        super().update(*args, **kwargs)
        self._check("update")

    def clear(self):
        super().clear()
        self._check("clear")


def available_test_device():
    """cuda:0 when a CUDA device is present and accepts an allocation, else cpu."""
    import torch

    if torch.cuda.is_available():
        try:
            torch.empty((), device="cuda:0")
            return torch.device("cuda:0")
        except Exception:
            pass
    return torch.device("cpu")


@contextlib.contextmanager
def stubbed_generation_last(data_path: str | Path) -> Iterator[tuple[types.ModuleType, types.ModuleType]]:
    """modules/generation_last.py without the webui runtime, for the block: yields (generation_last, the
    modules.shared stand-in it reads opts/state from).

    Its webui imports are stand-ins (modules.paths pointing at data_path, modules.scripts holding the real
    script_arg_range compiled on its own) except the stdlib-only modules.persistent_artifact_cache, which is the
    real file. The stubs stay installed for the whole block: generation_last imports modules.errors when it reports
    a failure."""
    import ast

    scripts_source = ROOT / "modules" / "scripts.py"
    tree = ast.parse(scripts_source.read_text(encoding="utf-8"))
    body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "script_arg_range"]
    namespace = {}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(scripts_source), "exec"), namespace)

    shared = module("modules.shared", opts=types.SimpleNamespace(CLIP_stop_at_last_layers=2),
                    state=types.SimpleNamespace(interrupted=False, stopping_generation=False))
    scripts = module("modules.scripts", script_arg_range=namespace["script_arg_range"])
    package = module("modules", package=True, scripts=scripts)
    paths = module("modules.paths", data_path=str(data_path))
    with stub_modules({"modules": package, "modules.paths": paths, "modules.shared": shared, "modules.scripts": scripts}):
        package.persistent_artifact_cache = load_source("modules.persistent_artifact_cache", "modules/persistent_artifact_cache.py")
        yield load_source("modules.generation_last", "modules/generation_last.py"), shared
