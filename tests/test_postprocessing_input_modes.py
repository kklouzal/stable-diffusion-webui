"""Extras input normalization: keep transparency, scale 16-bit grayscale instead of clipping it."""

import ast
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def _load_to_postprocessing_mode():
    # postprocessing imports the webui runtime; compile the one function in isolation.
    module = ast.parse(Path("modules/postprocessing.py").read_text(encoding="utf-8"))
    functions = [node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "to_postprocessing_mode"]
    namespace = {"np": np, "Image": Image}
    exec(compile(ast.Module(body=functions, type_ignores=[]), "modules/postprocessing.py", "exec"), namespace)
    return namespace["to_postprocessing_mode"]


to_postprocessing_mode = _load_to_postprocessing_mode()
VALUES = np.array([[0, 128, 129, 255, 256, 1000, 32767, 32896, 65534, 65535]], dtype=np.uint16)


@pytest.mark.parametrize("mode,dtype", [("I;16", "<u2"), ("I;16L", "<u2"), ("I;16B", ">u2")])
def test_16_bit_grayscale_is_scaled_and_rounded(mode, dtype):
    image = Image.frombytes(mode, (VALUES.shape[1], 1), VALUES.astype(dtype).tobytes())
    image.info["parameters"] = "prompt"

    converted = to_postprocessing_mode(image)

    expected = np.floor(VALUES[0].astype(np.float64) / 257.0 + 0.5).astype(np.uint8)  # round(v / 257), 65535 -> 255
    assert converted.mode == "RGB"
    assert np.array_equal(np.asarray(converted)[0], np.stack([expected] * 3, axis=-1))
    assert converted.info["parameters"] == "prompt"  # read_info_from_image runs on the converted image


def test_transparency_is_kept_as_rgba():
    la = Image.fromarray(np.array([[[10, 0], [200, 255]]], dtype=np.uint8), "LA")
    paletted = Image.new("P", (2, 1))
    paletted.putpalette([255, 0, 0, 0, 255, 0])
    paletted.putpixel((1, 0), 1)
    paletted.info["transparency"] = 0

    assert np.asarray(to_postprocessing_mode(la)).tolist() == [[[10, 10, 10, 0], [200, 200, 200, 255]]]
    assert np.asarray(to_postprocessing_mode(paletted)).tolist() == [[[255, 0, 0, 0], [0, 255, 0, 255]]]


def test_rgb_and_rgba_pass_through_and_opaque_modes_convert_as_before():
    rgb = Image.new("RGB", (2, 2), (1, 2, 3))
    rgba = Image.new("RGBA", (2, 2), (1, 2, 3, 4))
    gray = Image.fromarray(np.array([[0, 77], [128, 255]], dtype=np.uint8))

    assert to_postprocessing_mode(rgb) is rgb and to_postprocessing_mode(rgba) is rgba
    assert np.array_equal(np.asarray(to_postprocessing_mode(gray)), np.asarray(gray.convert("RGB")))
