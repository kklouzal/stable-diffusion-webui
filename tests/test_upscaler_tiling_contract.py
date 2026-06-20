import sys
import types
from types import SimpleNamespace

import pytest
from PIL import Image

for name in ("numpy", "pytz", "pillow_avif"):
    sys.modules.setdefault(name, types.ModuleType(name))

piexif = sys.modules.setdefault("piexif", types.ModuleType("piexif"))
piexif_helper = sys.modules.setdefault("piexif.helper", types.ModuleType("piexif.helper"))
piexif.helper = piexif_helper

shared = types.ModuleType("modules.shared")
shared.opts = SimpleNamespace(enable_upscale_progressbar=False)
shared.state = SimpleNamespace(interrupted=False, skipped=False)
shared.cmd_opts = SimpleNamespace(unix_filenames_sanitization=False, filenames_max_length=128)
sys.modules["modules.shared"] = shared

sys.modules.setdefault("modules.sd_samplers", types.ModuleType("modules.sd_samplers"))
sys.modules.setdefault("modules.script_callbacks", types.ModuleType("modules.script_callbacks"))
sys.modules.setdefault("modules.errors", types.ModuleType("modules.errors"))
paths_internal = types.ModuleType("modules.paths_internal")
paths_internal.roboto_ttf_file = ""
sys.modules.setdefault("modules.paths_internal", paths_internal)

from modules import images


def test_split_grid_clamps_overlap_larger_than_tile():
    img = Image.new("RGB", (64, 64), color=(32, 64, 96))

    grid = images.split_grid(img, tile_w=16, tile_h=16, overlap=48)
    assert grid.overlap == 15
    assert grid.tile_count > 1
    assert all(x >= 0 and y >= 0 for y, _h, row in grid.tiles for x, _w, _tile in row)


def test_tiled_upscale_2_clamps_overlap_larger_than_tile(monkeypatch):
    torch = pytest.importorskip("torch")
    from modules import upscaler_utils

    monkeypatch.setattr(shared, "opts", SimpleNamespace(enable_upscale_progressbar=False))
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
