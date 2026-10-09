import torch
from PIL import Image

from test.helpers import init_shared

init_shared()

from modules import images, upscaler_utils  # noqa: E402


def test_split_grid_clamps_overlap_larger_than_tile():
    img = Image.new("RGB", (64, 64), color=(32, 64, 96))

    grid = images.split_grid(img, tile_w=16, tile_h=16, overlap=48)
    assert grid.overlap == 15
    assert grid.tile_count > 1
    assert all(x >= 0 and y >= 0 for y, _h, row in grid.tiles for x, _w, _tile in row)


def test_tiled_upscale_2_clamps_overlap_larger_than_tile(monkeypatch):
    monkeypatch.setattr(upscaler_utils.shared.opts, "enable_upscale_progressbar", False)
    monkeypatch.setattr(upscaler_utils.shared.state, "interrupted", False)
    monkeypatch.setattr(upscaler_utils.shared.state, "skipped", False)
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
