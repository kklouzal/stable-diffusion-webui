"""modules/png_writer.py against Pillow's PNG writer on the same images: same chunks and filtered scanlines, same
pixels for Pillow and for an independent decoder (OpenCV/libpng), output independent of the thread count."""

import importlib.util
import io
import random
import zlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, PngImagePlugin

from test.helpers import TEST_FILES

try:
    import cv2  # OpenCV's libpng decoder: an implementation independent of Pillow and zlib's Python binding
except ImportError:
    cv2 = None

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("png_writer_under_test", ROOT / "modules" / "png_writer.py")
png_writer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(png_writer)


def parse(data):
    """[(chunk type, body)] of a PNG file, checking the signature, every CRC and that nothing follows IEND."""
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    out, pos = [], 8
    while pos < len(data):
        length = int.from_bytes(data[pos:pos + 4], "big")
        cid, body = data[pos + 4:pos + 8], data[pos + 8:pos + 8 + length]
        assert int.from_bytes(data[pos + 8 + length:pos + 12 + length], "big") == zlib.crc32(body, zlib.crc32(cid))
        out.append((cid, body))
        pos += 12 + length
    assert pos == len(data) and out[-1] == (b"IEND", b"")
    return out


def split(data):
    """(non-IDAT chunks in order, decompressed IDAT stream), checking that the IDAT chunks are contiguous."""
    chunks = parse(data)
    kinds = [cid == b"IDAT" for cid, _ in chunks]
    first = kinds.index(True)
    assert all(kinds[first:first + sum(kinds)])
    decompressor = zlib.decompressobj()
    stream = decompressor.decompress(b"".join(body for cid, body in chunks if cid == b"IDAT"))
    assert decompressor.eof and not decompressor.unused_data  # complete stream, adler32 checked by zlib
    return [chunk for chunk in chunks if chunk[0] != b"IDAT"], stream


def pillow_png(image, pnginfo=None):
    output = io.BytesIO()
    image.save(output, format="PNG", pnginfo=pnginfo)
    return output.getvalue()


def synthetic(mode, width, height, seed=0):
    """Deterministic bands each favouring one filter: noise (None), repeated rows (Up), per-row random offset plus a
    horizontal ramp (Sub), smooth 2D ramp with noise (Paeth and the others)."""
    rng = np.random.default_rng(seed)
    channels = len(mode)
    y, x = np.mgrid[0:height, 0:width]
    x = np.repeat(x[..., None], channels, axis=2)
    y = np.repeat(y[..., None], channels, axis=2)
    band = (y * 4 // max(height, 1))[..., 0]
    noise = rng.integers(0, 256, (height, width, channels))
    repeated = np.broadcast_to(rng.integers(0, 256, (1, width, channels)), (height, width, channels))
    ramp = (x * 7 + rng.integers(0, 256, (height, 1, channels))) % 256
    smooth = (x * 3 + y * 5 + rng.integers(0, 3, (height, width, channels))) % 256
    array = np.select([band[..., None] == k for k in range(4)], [noise, repeated, ramp, smooth]).astype(np.uint8)
    return Image.fromarray(array)  # (h, w, 3) -> RGB, (h, w, 4) -> RGBA


def real(mode):
    with Image.open(TEST_FILES / "two-faces.jpg") as photo:
        image = photo.convert("RGB")
    if mode == "RGBA":
        image.putalpha(image.convert("L").transpose(Image.Transpose.FLIP_TOP_BOTTOM))
    return image


def text_info():
    info = PngImagePlugin.PngInfo()
    info.add_text("parameters", "a red apple on a wooden table\nSteps: 20, Sampler: DPM++ 2M, Seed: 1")
    info.add_text("caf\xe9", "latin-1 only: tEXt \xe9")
    info.add_text("extras", "not latin-1: iTXt — 日本 ✓")
    info.add_text("tagged", PngImagePlugin.iTXt("with language", "en", "Tagged"))
    info.add_text("zipped", "zTXt " * 50, zip=True)
    info.add_itxt("zipped-itxt", "compressed iTXt ✓ " * 50, zip=True)
    return info


def assert_equivalent(image, pnginfo=None, **encode_kwargs):
    ours = png_writer.encode(image, pnginfo, **encode_kwargs)
    reference = pillow_png(image, pnginfo)
    our_chunks, our_stream = split(ours)
    reference_chunks, reference_stream = split(reference)
    assert our_chunks == reference_chunks  # IHDR, text chunks (order, encoding), IEND
    assert our_stream == reference_stream  # identical filter choice and filtered bytes for every row

    with Image.open(io.BytesIO(ours)) as decoded, Image.open(io.BytesIO(reference)) as expected:
        decoded.load()
        expected.load()
        assert (decoded.mode, decoded.size) == (image.mode, image.size)
        assert decoded.tobytes() == image.tobytes()
        assert decoded.info == expected.info and decoded.text == expected.text

    if cv2 is None:  # present in the target image; elsewhere only the independent-decoder check is skipped
        return ours, reference
    independent = cv2.imdecode(np.frombuffer(ours, np.uint8), cv2.IMREAD_UNCHANGED)
    if independent.ndim == 2:
        independent = independent[..., None]
    independent = independent[..., [2, 1, 0, 3][:independent.shape[2]]]  # BGR(A) -> RGB(A)
    assert np.array_equal(independent.reshape(image.height, image.width, -1), np.asarray(image).reshape(image.height, image.width, -1))
    return ours, reference


@pytest.mark.parametrize("mode", ["RGB", "RGBA"])
@pytest.mark.parametrize("size", [(1, 1), (1, 9), (9, 1), (2, 3), (37, 23), (203, 151)])
def test_synthetic_images_match_pillow(mode, size):
    assert_equivalent(synthetic(mode, *size, seed=size[0] * 1000 + size[1]), text_info())


@pytest.mark.parametrize("mode", ["RGB", "RGBA"])
def test_every_filter_type_is_exercised_and_matches(mode, monkeypatch):
    image = synthetic(mode, 160, 120)
    _, stream = split(pillow_png(image))
    row = 160 * len(mode) + 1
    assert {stream[i] for i in range(0, len(stream), row)} == {0, 1, 2, 4}  # Average is only used with optimize
    assert_equivalent(image)
    monkeypatch.setattr(png_writer, "_PIECE_BYTES", 1)  # one row per piece, histories shorter than the window
    assert_equivalent(image)


@pytest.mark.parametrize("mode", ["RGB", "RGBA"])
@pytest.mark.parametrize("piece_bytes", [None, 2000])
def test_photo_matches_pillow_and_size_stays_close(mode, piece_bytes, monkeypatch):
    if piece_bytes is not None:
        monkeypatch.setattr(png_writer, "_PIECE_BYTES", piece_bytes)
    ours, reference = assert_equivalent(real(mode), text_info())
    if piece_bytes is None:  # tiny pieces restart Huffman blocks every few rows: valid, but larger
        assert abs(len(ours) / len(reference) - 1) < 0.01


def test_images_opened_from_files_keep_pillow_behaviour():
    # srgb/gamma/dpi/Software in info: Pillow's writer does not copy them, neither does png_writer.
    with Image.open(TEST_FILES / "img2img_basic.png") as image:
        assert image.mode == "RGBA" and {"srgb", "gamma", "dpi"} <= set(image.info)
        assert_equivalent(image, text_info())
    rgba = synthetic("RGBA", 20, 10)
    rgba.info["transparency"] = 3  # ignored by Pillow for RGBA
    assert_equivalent(rgba)


def test_output_does_not_depend_on_thread_count():
    image = synthetic("RGB", 300, 400)
    outputs = {png_writer.encode(image, text_info(), threads=threads) for threads in (1, 2, 3, 8)}
    assert len(outputs) == 1


def uncovered_cases():
    rgb = synthetic("RGB", 8, 8)
    icc = rgb.copy()
    icc.info["icc_profile"] = b"profile"
    keyed = rgb.copy()
    keyed.info["transparency"] = (1, 2, 3)
    stale = rgb.copy()
    stale.encoderinfo = {"dpi": (72, 72)}
    background = PngImagePlugin.PngInfo()
    background.add(b"bKGD", b"\0\0\0\0\0\0")
    private = PngImagePlugin.PngInfo()
    private.add(b"prVt", b"data", after_idat=True)
    return [
        (rgb.convert("L"), None), (rgb.convert("LA"), None), (rgb.convert("P"), None), (rgb.convert("1"), None),
        (Image.new("I;16", (4, 4)), None), (Image.new("RGB", (0, 3)), None),
        (icc, None), (keyed, None), (stale, None), (rgb, background), (rgb, private),
    ]


@pytest.mark.parametrize("index", range(len(uncovered_cases())))
def test_uncovered_inputs_are_left_to_pillow(index):
    image, pnginfo = uncovered_cases()[index]
    assert png_writer.encode(image, pnginfo) is None


def test_adler32_combine_matches_zlib():
    rng = random.Random(5)
    for _ in range(200):
        a, b = rng.randbytes(rng.choice([0, 1, 7, 65521, 70000])), rng.randbytes(rng.choice([0, 1, 5552, 65521, 65522, 200000]))
        assert png_writer.adler32_combine(zlib.adler32(a), zlib.adler32(b), len(b)) == zlib.adler32(a + b)
