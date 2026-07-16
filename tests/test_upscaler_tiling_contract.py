import contextlib
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

_UPSCALER_STUB_MODULES = []


def _setdefault_upscaler_stub(name, module):
    if name not in sys.modules:
        sys.modules[name] = module
        _UPSCALER_STUB_MODULES.append(name)
    return sys.modules[name]


def _cleanup_upscaler_import_stubs():
    while _UPSCALER_STUB_MODULES:
        name = _UPSCALER_STUB_MODULES.pop()
        module = sys.modules.pop(name, None)
        if "." in name:
            parent_name, attr = name.rsplit(".", 1)
            parent = sys.modules.get(parent_name)
            if parent is not None and hasattr(parent, attr) and getattr(parent, attr, None) is module:
                delattr(parent, attr)
    modules_pkg = sys.modules.get("modules")
    images_module = sys.modules.pop("modules.images", None)
    if modules_pkg is not None and hasattr(modules_pkg, "images") and getattr(modules_pkg, "images", None) is images_module:
        delattr(modules_pkg, "images")


import pytest
from PIL import Image

for name in ("numpy", "pytz", "pillow_avif"):
    if importlib.util.find_spec(name) is None:
        _setdefault_upscaler_stub(name, types.ModuleType(name))

if importlib.util.find_spec("piexif") is None:
    piexif = _setdefault_upscaler_stub("piexif", types.ModuleType("piexif"))
    piexif_helper = _setdefault_upscaler_stub("piexif.helper", types.ModuleType("piexif.helper"))
    piexif.helper = piexif_helper

shared = types.ModuleType("modules.shared")
shared.opts = SimpleNamespace(enable_upscale_progressbar=False)
shared.state = SimpleNamespace(interrupted=False, skipped=False)
shared.cmd_opts = SimpleNamespace(unix_filenames_sanitization=False, filenames_max_length=128)
_setdefault_upscaler_stub("modules.shared", shared)

_setdefault_upscaler_stub("modules.sd_samplers", types.ModuleType("modules.sd_samplers"))
_setdefault_upscaler_stub("modules.script_callbacks", types.ModuleType("modules.script_callbacks"))
_setdefault_upscaler_stub("modules.errors", types.ModuleType("modules.errors"))
paths_internal = types.ModuleType("modules.paths_internal")
webui_root = str(Path(__file__).parents[1])
paths_internal.roboto_ttf_file = ""
paths_internal.models_path = f"{webui_root}/models"
paths_internal.script_path = webui_root
paths_internal.data_path = webui_root
paths_internal.extensions_dir = f"{webui_root}/extensions"
paths_internal.extensions_builtin_dir = f"{webui_root}/extensions-builtin"
paths_internal.cwd = webui_root
_setdefault_upscaler_stub("modules.paths_internal", paths_internal)

from modules import images
_cleanup_upscaler_import_stubs()


def test_split_grid_clamps_overlap_larger_than_tile():
    img = Image.new("RGB", (64, 64), color=(32, 64, 96))

    grid = images.split_grid(img, tile_w=16, tile_h=16, overlap=48)
    assert grid.overlap == 15
    assert grid.tile_count > 1
    assert all(x >= 0 and y >= 0 for y, _h, row in grid.tiles for x, _w, _tile in row)


def test_tiled_upscale_2_clamps_overlap_larger_than_tile(monkeypatch):
    torch = pytest.importorskip("torch")
    devices = types.ModuleType("modules.devices")
    devices.without_autocast = lambda disable=False: contextlib.nullcontext()
    torch_utils = types.ModuleType("modules.torch_utils")
    torch_utils.get_param = lambda model: None
    _setdefault_upscaler_stub("modules.shared", shared)
    _setdefault_upscaler_stub("modules.devices", devices)
    _setdefault_upscaler_stub("modules.torch_utils", torch_utils)
    _setdefault_upscaler_stub("modules.images", images)
    from modules import upscaler_utils
    _cleanup_upscaler_import_stubs()

    monkeypatch.setattr(shared, "opts", SimpleNamespace(enable_upscale_progressbar=False))
    monkeypatch.setattr(upscaler_utils.shared, "opts", SimpleNamespace(enable_upscale_progressbar=False))
    monkeypatch.setattr(upscaler_utils.shared, "state", SimpleNamespace(interrupted=False, skipped=False))
    img = torch.arange(3 * 32 * 32, dtype=torch.float32).reshape(1, 3, 32, 32)

    result = upscaler_utils.tiled_upscale_2(
        img,
        lambda patch: patch,
        tile_size=16,
        tile_overlap=48,
        scale=1,
        device=torch.device("cpu"),
    )

    assert torch.equal(result, img)
