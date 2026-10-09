"""Hypertile U-Net tiling geometry (extensions-builtin/hypertile/hypertile.py). CPU-only."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

HYPERTILE_PATH = Path(__file__).resolve().parents[1] / "extensions-builtin" / "hypertile" / "hypertile.py"


@pytest.fixture(scope="module")
def hypertile():
    name = "hypertile_under_test"
    spec = importlib.util.spec_from_file_location(name, HYPERTILE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # @dataclass resolves the defining module through sys.modules
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(name, None)


@pytest.mark.parametrize(
    ("rows", "cols", "image_width", "image_height"),
    [(32, 64, 1024, 512), (64, 32, 512, 1024), (52, 76, 1216, 832), (38, 63, 1000, 600), (40, 40, 1280, 1280)],
)
def test_find_hw_candidates_returns_rows_then_columns(hypertile, rows, cols, image_width, image_height):
    assert hypertile.find_hw_candidates(rows * cols, image_width / image_height) == (rows, cols)


def test_primary_and_fallback_branches_agree_on_orientation(hypertile):
    # 91 = 7 * 13 with an aspect the rounding branch cannot hit exactly: the divisor search decides.
    assert hypertile.find_hw_candidates(91, 1.6) == (7, 13)
    assert hypertile.iterative_closest_divisors(91, 1.6) == (7, 13)
    # The rounding branch must use the same convention for an exact grid of the same orientation.
    assert hypertile.find_hw_candidates(7 * 13, 13 / 7) == (7, 13)


def _tiles_seen_by_attention(hypertile, rows, cols, *, tile_size=128, depth=0, swap_size=1):
    """Run the U-Net wrapper on a row-major token grid whose channels carry (row, col); return the tiles."""
    r, c = torch.meshgrid(torch.arange(rows), torch.arange(cols), indexing="ij")
    x = torch.stack([r.reshape(-1), c.reshape(-1)], dim=-1).float()[None]  # (1, rows*cols, 2), "b c h w -> b (h w) c"
    seen = []

    def attention(tokens, *args, **kwargs):
        seen.append(tokens.clone())
        return tokens

    params = hypertile.HypertileParams()
    params.forward = attention
    params.depth = depth
    params.tile_size = tile_size
    params.swap_size = swap_size
    params.aspect_ratio = (cols * 8) / (rows * 8)  # image width / height, as hypertile_hook_model sets it
    params.enabled = True
    hypertile.set_hypertile_seed(0)
    out = hypertile.self_attn_forward(params)(x)
    assert len(seen) == 1
    return x, out, seen[0]


@pytest.mark.parametrize(("rows", "cols"), [(32, 64), (64, 32), (48, 80), (32, 32)])
def test_unet_tiles_are_contiguous_spatial_rectangles(hypertile, rows, cols):
    x, out, tiles = _tiles_seen_by_attention(hypertile, rows, cols)

    assert torch.equal(out, x)  # tiling round-trips the token order
    assert tiles.shape[0] > 1  # the grid is large enough to be tiled
    for tile in tiles:
        tile_rows = tile[:, 0].long()
        tile_cols = tile[:, 1].long()
        r0, r1 = int(tile_rows.min()), int(tile_rows.max())
        c0, c1 = int(tile_cols.min()), int(tile_cols.max())
        height, width = r1 - r0 + 1, c1 - c0 + 1
        assert height * width == tile.shape[0], "tile is not a dense rectangle of the latent grid"
        expected_r, expected_c = torch.meshgrid(torch.arange(r0, r1 + 1), torch.arange(c0, c1 + 1), indexing="ij")
        assert torch.equal(tile_rows, expected_r.reshape(-1))
        assert torch.equal(tile_cols, expected_c.reshape(-1))


def test_tile_counts_follow_each_axis(hypertile):
    # latent tile 16 at depth 0: a 32x64 grid splits into 2 row bands and 4 column bands.
    _, _, tiles = _tiles_seen_by_attention(hypertile, 32, 64)
    assert tiles.shape[0] == 2 * 4
    assert tiles.shape[1] == 16 * 16


def test_disabled_wrapper_passes_through(hypertile):
    calls = []
    params = hypertile.HypertileParams()
    params.forward = lambda *args, **kwargs: calls.append((args, kwargs)) or "out"
    params.enabled = False
    assert hypertile.self_attn_forward(params)("x", context=None) == "out"
    assert calls == [(("x",), {"context": None})]
