"""ControlNet input conversions done with OpenCV return exactly the arrays of the NumPy expressions they replace:
ControlNetUnit.combine_image_and_mask (RGB + mask channel -> RGBA) and choose_input_image's from_rgba_to_input
(RGBA -> RGB when the alpha channel is not a scribble)."""

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import cv2
import numpy as np
import pytest

CONTROLNET_ROOT = Path(__file__).resolve().parents[1] / "extensions" / "sd-webui-controlnet"


def extract(path, *names):
    """The function reached through names (class or function bodies), compiled without its decorators."""
    body = ast.parse(path.read_text(encoding="utf8")).body
    for name in names:
        node = next(n for n in ast.walk(ast.Module(body=body, type_ignores=[])) if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name)
        body = node.body
    node.decorator_list = []
    return node


def load(path, *names, **namespace):
    node = extract(path, *names)
    namespace = {"np": np, "cv2": cv2, "Optional": Optional, **namespace}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[node.name]


combine_image_and_mask = load(CONTROLNET_ROOT / "internal_controlnet" / "args.py", "ControlNetUnit", "combine_image_and_mask")
from_rgba_to_input = load(
    CONTROLNET_ROOT / "scripts" / "controlnet.py", "Script", "choose_input_image", "from_rgba_to_input",
    shared=SimpleNamespace(opts=SimpleNamespace(data={})), logger=SimpleNamespace(info=lambda *args: None),
    HWC3=lambda x: np.concatenate([x[:, :, None]] * 3, axis=2),
)


def images(seed=0):
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 256, (37, 53, 3), dtype=np.uint8)
    yield base
    yield base[:, ::2]  # strided
    yield np.asfortranarray(base)
    yield base[:1, :1]
    yield rng.integers(0, 256, (720, 1280, 3), dtype=np.uint8)


@pytest.mark.parametrize("with_mask", [False, True])
def test_combine_image_and_mask_equals_concatenate(with_mask):
    for image in images():
        mask = np.random.default_rng(1).integers(0, 256, image.shape, dtype=np.uint8) if with_mask else None
        expected = np.concatenate([image, (np.zeros_like(image) if mask is None else mask)[:, :, 0:1]], axis=2)
        actual = combine_image_and_mask(None, image, mask)
        assert actual.dtype == expected.dtype and actual.shape == expected.shape and np.array_equal(actual, expected)
        assert actual.flags.c_contiguous and not np.shares_memory(actual, image)

    with pytest.raises(ValueError):
        combine_image_and_mask(None, np.zeros((4, 4, 3), np.uint8), np.zeros((4, 5, 3), np.uint8))
    floats = np.ones((2, 2, 3), np.float32)  # not uint8: the NumPy path, as before
    assert np.array_equal(combine_image_and_mask(None, floats, None), np.concatenate([floats, np.zeros((2, 2, 1), np.float32)], axis=2))


def test_rgb_taken_from_rgba_equals_the_channel_slice():
    for image in images():
        rgba = combine_image_and_mask(None, image, None)  # alpha 0: not a scribble
        taken = from_rgba_to_input(rgba)
        assert taken.dtype == np.uint8 and np.array_equal(taken, rgba[:, :, :3]) and taken.flags.c_contiguous
        opaque = rgba.copy()
        opaque[:, :, 3] = 255
        assert np.array_equal(from_rgba_to_input(opaque), image)
