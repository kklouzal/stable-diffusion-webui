import logging
from typing import Callable

import numpy as np
import torch
import tqdm
from PIL import Image

from modules import devices, images, shared, torch_utils

logger = logging.getLogger(__name__)


_unit_lut_cache: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}


def _unit_lut(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """`v / 255` for every uint8 `v`: divided in float64 and rounded once to `dtype` on the CPU, then moved to
    `device`. Bitwise equal to the former numpy path (float64 `/ 255`, then `.to(device, dtype)`, which converts on the
    CPU side); see test/test_upscaler_device_conversions.py.
    """
    key = (device, dtype)
    lut = _unit_lut_cache.get(key)
    if lut is None:
        lut = torch.from_numpy(np.arange(256, dtype=np.uint8) / 255).to(dtype=dtype).to(device=device)
        _unit_lut_cache[key] = lut
    return lut


def pil_image_to_device_bgr(img: Image.Image, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """1xCxHxW BGR tensor of `img` in [0, 1]: only the uint8 pixels are copied to `device` and converted there
    through `_unit_lut`, with the canonical contiguous strides of the former CPU numpy path."""
    pixels = torch.from_numpy(np.array(img.convert("RGB"))).to(device=device)
    bgr = _unit_lut(device, dtype)[pixels.flip(2).long()]
    # HWC to CHW with the canonical contiguous strides of the CPU path, also for size-1 dims.
    return bgr.permute(2, 0, 1).clone(memory_format=torch.contiguous_format).unsqueeze(0)


def torch_bgr_to_pil_image(tensor: torch.Tensor) -> Image.Image:
    if tensor.ndim == 4:
        # If we're given a tensor with a batch dimension, squeeze it out
        # (but only if it's a batch of size 1).
        if tensor.shape[0] != 1:
            raise ValueError(f"{tensor.shape} does not describe a BCHW tensor")
        tensor = tensor.squeeze(0)
    assert tensor.ndim == 3, f"{tensor.shape} does not describe a CHW tensor"
    # Quantize on the tensor's device and copy back only uint8. Same ops and order as the former
    # numpy path: fp32 clamp, fp32 multiply by 255, round half to even, cast to uint8.
    arr = tensor.float().clamp(0, 1).mul_(255.0).round_().to(torch.uint8)
    arr = arr.flip(0).permute(1, 2, 0).contiguous()  # BGR CHW to RGB HWC
    return Image.fromarray(arr.cpu().numpy(), "RGB")


def upscale_pil_patch(model, img: Image.Image) -> Image.Image:
    """
    Upscale a given PIL image using the given model.
    """
    param = torch_utils.get_param(model)

    with torch.inference_mode():
        tensor = pil_image_to_device_bgr(img, param.device, param.dtype)
        with devices.without_autocast():
            return torch_bgr_to_pil_image(model(tensor))


def _keeping_alpha(img: Image.Image, upscale_rgb: Callable[[Image.Image], Image.Image]) -> Image.Image:
    """`upscale_rgb(img)`, except that an RGBA image keeps its alpha: models upscale colour only, so its RGB is
    upscaled and its alpha resized (LANCZOS) to the result's size and re-attached. Other modes go to `upscale_rgb`
    unchanged (it converts them to RGB). `upscale_rgb` returns its input when interrupted; then `img` comes back."""
    if img.mode != "RGBA":
        return upscale_rgb(img)
    rgb = img.convert("RGB")
    output = upscale_rgb(rgb)
    if output is rgb:
        return img
    output.putalpha(img.getchannel("A").resize(output.size, resample=Image.Resampling.LANCZOS))
    return output


def upscale_with_model(
    model: Callable[[torch.Tensor], torch.Tensor],
    img: Image.Image,
    *,
    tile_size: int,
    tile_overlap: int = 0,
    desc="tiled upscale",
) -> Image.Image:
    """`img` upscaled by `model` (tiled through `images.Grid` unless `tile_size` <= 0); see `_keeping_alpha` for RGBA
    images. An interrupted upscale returns `img`."""
    return _keeping_alpha(img, lambda rgb: _upscale_rgb_with_model(model, rgb, tile_size=tile_size, tile_overlap=tile_overlap, desc=desc))


def _upscale_rgb_with_model(model, img: Image.Image, *, tile_size: int, tile_overlap: int, desc: str) -> Image.Image:
    if tile_size <= 0:
        logger.debug("Upscaling %s without tiling", img)
        output = upscale_pil_patch(model, img)
        logger.debug("=> %s", output)
        return output

    # A tile larger than the image would be cropped at a negative offset, i.e. padded with black that the model
    # then blends into the image's edges; cap each tile side at the image side instead.
    grid = images.split_grid(img, min(tile_size, img.width), min(tile_size, img.height), tile_overlap)
    newtiles = []

    with tqdm.tqdm(total=grid.tile_count, desc=desc, disable=not shared.opts.enable_upscale_progressbar) as p:
        for y, h, row in grid.tiles:
            newrow = []
            for x, w, tile in row:
                if shared.state.interrupted:
                    return img
                output = upscale_pil_patch(model, tile)
                scale_factor = output.width // tile.width
                newrow.append([x * scale_factor, w * scale_factor, output])
                p.update(1)
            newtiles.append([y * scale_factor, h * scale_factor, newrow])

    newgrid = images.Grid(
        newtiles,
        tile_w=grid.tile_w * scale_factor,
        tile_h=grid.tile_h * scale_factor,
        image_w=grid.image_w * scale_factor,
        image_h=grid.image_h * scale_factor,
        overlap=grid.overlap * scale_factor,
    )
    return images.combine_grid(newgrid)


def tiled_upscale_2(
    img: torch.Tensor,
    model,
    *,
    tile_size: int,
    tile_overlap: int,
    scale: int,
    device: torch.device,
    desc="Tiled upscale",
):
    # Alternative implementation of `upscale_with_model` originally used by
    # SwinIR and ScuNET.  It differs from `upscale_with_model` in that tiling and
    # weighting is done in PyTorch space, as opposed to `images.Grid` doing it in
    # Pillow space.  Each tile's weight ramps up linearly over the `tile_overlap`
    # (times `scale`) output pixels along every edge it shares with another tile
    # and is 1 elsewhere, so overlapping tiles are cross-faded instead of
    # averaged evenly, which kept both tiles' border errors in the seam; with
    # `tile_overlap` 0 every weight is 1, the former plain average.
    # Returns None when interrupted or skipped before every tile ran.

    b, c, h, w = img.size()
    tile_size = min(tile_size, h, w)
    tile_overlap = max(0, min(tile_overlap, tile_size - 1))

    if tile_size <= 0:
        logger.debug("Upscaling %s without tiling", img.shape)
        return model(img.to(device=device))

    stride = tile_size - tile_overlap
    h_idx_list = list(range(0, h - tile_size, stride)) + [h - tile_size]
    w_idx_list = list(range(0, w - tile_size, stride)) + [w - tile_size]
    result = torch.zeros(
        b,
        c,
        h * scale,
        w * scale,
        device=device,
        dtype=img.dtype,
    )
    # Per-pixel weight sum; the same for every channel, so one channel broadcast over `result` suffices.
    weights = torch.zeros((1, 1, h * scale, w * scale), device=device, dtype=img.dtype)

    def edge_ramp(starts_inside: bool, ends_inside: bool) -> torch.Tensor:
        # min(1, distance in pixels from each shared edge / (ramp + 1)), in float64: never 0, so every output pixel
        # gets a positive weight sum.
        ramp = tile_overlap * scale
        distance = torch.arange(1, tile_size * scale + 1, dtype=torch.float64)
        weight = torch.ones_like(distance)
        if starts_inside:
            weight = torch.minimum(weight, distance / (ramp + 1))
        if ends_inside:
            weight = torch.minimum(weight, distance.flip(0) / (ramp + 1))
        return weight

    tile_weights = {}  # by which of the tile's edges are shared: at most 9 distinct tiles
    logger.debug("Upscaling %s to %s with tiles", img.shape, result.shape)
    with tqdm.tqdm(total=len(h_idx_list) * len(w_idx_list), desc=desc, disable=not shared.opts.enable_upscale_progressbar) as pbar:
        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                if shared.state.interrupted or shared.state.skipped:
                    # Pixels no tile has reached yet have weight 0; dividing would turn them into NaN.
                    return None

                # Only move this patch to the device if it's not already there.
                in_patch = img[
                    ...,
                    h_idx : h_idx + tile_size,
                    w_idx : w_idx + tile_size,
                ].to(device=device)

                out_patch = model(in_patch)

                edges = (h_idx > 0, h_idx + tile_size < h, w_idx > 0, w_idx + tile_size < w)
                tile_weight = tile_weights.get(edges)
                if tile_weight is None:
                    tile_weight = torch.outer(edge_ramp(*edges[:2]), edge_ramp(*edges[2:])).to(device=device, dtype=img.dtype)
                    tile_weights[edges] = tile_weight

                result[
                    ...,
                    h_idx * scale : (h_idx + tile_size) * scale,
                    w_idx * scale : (w_idx + tile_size) * scale,
                ].addcmul_(out_patch, tile_weight)

                weights[
                    ...,
                    h_idx * scale : (h_idx + tile_size) * scale,
                    w_idx * scale : (w_idx + tile_size) * scale,
                ].add_(tile_weight)

                pbar.update(1)

    output = result.div_(weights)

    return output


def upscale_2(
    img: Image.Image,
    model,
    *,
    tile_size: int,
    tile_overlap: int,
    scale: int,
    desc: str,
):
    """
    Convenience wrapper around `tiled_upscale_2` that handles PIL images.

    Like `upscale_with_model`, the model runs in its own dtype even when the caller (hires fix) is inside the
    sampler's autocast, an RGBA image keeps its alpha (`_keeping_alpha`), and an interrupted or skipped upscale
    returns `img` unchanged.
    """
    param = torch_utils.get_param(model)

    def upscale_rgb(rgb: Image.Image) -> Image.Image:
        with torch.inference_mode(), devices.without_autocast():
            # Uploaded once; bitwise the same tensor as the CPU float64 conversion followed by a per-tile copy.
            tensor = pil_image_to_device_bgr(rgb, param.device, param.dtype)
            output = tiled_upscale_2(
                tensor,
                model,
                tile_size=tile_size,
                tile_overlap=tile_overlap,
                scale=scale,
                desc=desc,
                device=param.device,
            )
            if output is None:
                return rgb
            return torch_bgr_to_pil_image(output)

    return _keeping_alpha(img, upscale_rgb)
