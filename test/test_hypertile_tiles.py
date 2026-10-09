"""Hypertile U-Net tiling geometry (extensions-builtin/hypertile/hypertile.py). CPU-only."""

import pytest
import torch

from test.helpers import load_source


@pytest.fixture(scope="module")
def hypertile():
    return load_source("hypertile_under_test", "extensions-builtin/hypertile/hypertile.py")


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


def _hooked_depth0_layer(hypertile, width, height, tile_size_max):
    """A U-Net with one SD1.5 depth-0 attention layer, configured by the real hypertile_hook_model; returns a function
    that runs that layer on a (rows, cols) row-major token grid and returns the token tiles its attention saw."""
    seen = []

    class Attention(torch.nn.Module):
        def forward(self, tokens):
            seen.append(tokens.clone())
            return tokens

    model = torch.nn.Module()
    parent = model
    for part in "input_blocks.1.1.transformer_blocks.0".split("."):
        parent.add_module(part, torch.nn.Module())
        parent = getattr(parent, part)
    parent.add_module("attn1", Attention())
    hypertile.hypertile_hook_model(model, width, height, enable=True, tile_size_max=tile_size_max, swap_size=1, max_depth=0)

    def run(rows, cols):
        r, c = torch.meshgrid(torch.arange(rows), torch.arange(cols), indexing="ij")
        x = torch.stack([r.reshape(-1), c.reshape(-1)], dim=-1)[None]
        seen.clear()
        assert torch.equal(parent.attn1(x), x)  # tiling round-trips the token order
        return seen[0]

    return parent.attn1.__webui_hypertile_params, run


def _smallest_divisor_at_least(value, minimum):
    return next(d for d in range(min(minimum, value), value + 1) if value % d == 0)


# Every multiple of 64 from 512 to 2048 on both axes, plus multiples of 8 that are not multiples of 64.
SIZES = list(range(512, 2049, 64)) + [520, 776, 832, 1000, 1080, 1216, 1352, 1544]


@pytest.mark.parametrize("tile_size_max", [0, 128, 256, 384, 512])
def test_hook_keeps_the_configured_tile_size_and_splits_each_axis_by_it(hypertile, tile_size_max):
    latent_tile = max(128, tile_size_max) // 8
    for width in SIZES:
        for height in SIZES[::3]:
            params, run = _hooked_depth0_layer(hypertile, width, height, tile_size_max)
            assert params.tile_size == tile_size_max, (width, height)
            rows, cols = height // 8, width // 8
            tiles = run(rows, cols)
            tile_rows = _smallest_divisor_at_least(rows, latent_tile)
            tile_cols = _smallest_divisor_at_least(cols, latent_tile)
            # Each tile edge depends only on its own axis: the other axis (squareness, gcd) never shrinks it.
            assert tiles.shape == ((rows // tile_rows) * (cols // tile_cols), tile_rows * tile_cols, 2), (width, height)
            for tile in tiles:
                assert int(tile[:, 0].max() - tile[:, 0].min()) + 1 == tile_rows
                assert int(tile[:, 1].max() - tile[:, 1].min()) + 1 == tile_cols


def test_non_square_image_tiles_like_the_square_one_on_the_shared_axis(hypertile):
    # 1024x1152 used to get tile size 128 (largest power of two dividing gcd 128) instead of the configured 256:
    # 16-row tiles, 8x9 of them, where 1024x1024 gets 32-row tiles.
    _, run_square = _hooked_depth0_layer(hypertile, 1024, 1024, 256)
    _, run_wide = _hooked_depth0_layer(hypertile, 1152, 1024, 256)
    square, wide = run_square(128, 128), run_wide(128, 144)
    assert square.shape[0] == 4 * 4
    assert wide.shape[0] == 4 * 4
    assert int(wide[0][:, 0].max()) + 1 == int(square[0][:, 0].max()) + 1 == 32
