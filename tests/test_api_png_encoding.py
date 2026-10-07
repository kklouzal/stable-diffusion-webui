import ast
import base64
import io
import os
import types
from pathlib import Path

from PIL import Image, PngImagePlugin

API_PATH = Path(__file__).resolve().parents[1] / "modules" / "api" / "api.py"


def load_encode_pil_to_base64(samples_format="png"):
    tree = ast.parse(API_PATH.read_text(encoding="utf8"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "encode_pil_to_base64")
    module = ast.Module(body=[function], type_ignores=[])
    namespace = {
        "io": io,
        "base64": base64,
        "PngImagePlugin": PngImagePlugin,
        "opts": types.SimpleNamespace(samples_format=samples_format, jpeg_quality=80, webp_lossless=False),
        "images": types.SimpleNamespace(geninfo_to_exif_bytes=lambda parameters: b""),
        "HTTPException": RuntimeError,
    }
    exec(compile(module, str(API_PATH), "exec"), namespace)
    return namespace["encode_pil_to_base64"]


def test_png_response_is_lossless_level_one_with_metadata():
    encode = load_encode_pil_to_base64()
    for mode in ("RGB", "RGBA"):
        image = Image.frombytes(mode, (96, 64), os.urandom(96 * 64 * len(mode)))
        image.info["parameters"] = "steps: 20"
        encoded = encode(image)

        metadata = PngImagePlugin.PngInfo()
        metadata.add_text("parameters", "steps: 20")
        expected = io.BytesIO()
        image.save(expected, format="PNG", pnginfo=metadata, compress_level=1)
        assert base64.b64decode(encoded) == expected.getvalue()

        with Image.open(io.BytesIO(base64.b64decode(encoded))) as decoded:
            assert decoded.mode == mode
            assert decoded.tobytes() == image.tobytes()
            assert decoded.text == {"parameters": "steps: 20"}


def test_strings_pass_through_unchanged():
    assert load_encode_pil_to_base64()("already-encoded") == "already-encoded"
