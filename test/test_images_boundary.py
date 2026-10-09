"""modules/images.py correctness at the image boundary: tiling recombination, grid annotations on current Pillow,
untrusted EXIF parsing, transparency flattening, 16-bit to 8-bit rounding, eager decode and saved file names."""

import importlib.util
import io
import os
import random
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import piexif
import piexif.helper
import pytest
from PIL import Image, PngImagePlugin

ROOT = Path(__file__).resolve().parents[1]


def _module(name, **attrs):
    mod = types.ModuleType(name)
    mod.__dict__.update(attrs)
    return mod


@pytest.fixture
def images(monkeypatch):
    """A private copy of modules/images.py with only the webui modules it imports stubbed."""
    opts = SimpleNamespace(
        data={}, enable_pnginfo=True, jpeg_quality=80, webp_lossless=True, save_to_dirs=False, grid_save_to_dirs=False,
        samples_filename_pattern="", save_images_add_number=False, directories_filename_pattern="", export_for_4chan=False,
        target_side_length=4000, img_downscale_threshold=4, save_txt=False, font=None, n_rows=0, grid_prevent_empty_spots=False,
        grid_background_color="#ffffff", grid_text_active_color="#000000", grid_text_inactive_color="#999999",
        directories_max_prompt_words=8, save_images_replace_action="Replace", png_parallel_encoder=True,
    )
    stubs = {
        "modules.shared": _module(
            "modules.shared", opts=opts, cmd_opts=SimpleNamespace(unix_filenames_sanitization=False, filenames_max_length=128),
            state=SimpleNamespace(job_timestamp=""), prompt_styles=SimpleNamespace(get_style_prompts=lambda styles: []), sd_upscalers=[],
        ),
        "modules.script_callbacks": _module(
            "modules.script_callbacks",
            ImageSaveParams=lambda image, p, filename, pnginfo: SimpleNamespace(image=image, p=p, filename=filename, pnginfo=pnginfo),
            before_image_saved_callback=lambda params: None, image_saved_callback=lambda params: None,
        ),
        "modules.sd_samplers": _module("modules.sd_samplers", samplers_map={}),
        "modules.errors": _module("modules.errors", report=lambda *a, **k: None, display=lambda *a, **k: None),
        "modules.paths_internal": _module("modules.paths_internal", roboto_ttf_file=str(ROOT / "modules" / "Roboto-Regular.ttf")),
    }
    package = sys.modules.get("modules")
    for name, stub in stubs.items():
        monkeypatch.setitem(sys.modules, name, stub)
        if package is not None:
            monkeypatch.setattr(package, name.split(".")[1], stub, raising=False)

    spec = importlib.util.spec_from_file_location("c1_images_under_test", ROOT / "modules" / "images.py")
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def _random_rgb(w, h, seed=0):
    return Image.fromarray(np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8))


# --- split_grid / combine_grid ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("size,tile,overlap", [
    ((190, 190), 192, 8),     # narrower and shorter than the tile, within the overlap: was darkened at the edges
    ((190, 300), 192, 8),
    ((300, 186), 192, 8),
    ((100, 100), 192, 8),
    ((192, 192), 192, 8),
    ((500, 400), 192, 8),
    ((1024, 1024), 192, 8),
    ((64, 64), 16, 48),
    ((200, 200), 192, 0),
    ((1000, 1500), 1024, 64),  # SD upscale: tile = p.width
])
def test_combine_grid_of_untouched_tiles_is_the_image(images, size, tile, overlap):
    img = _random_rgb(*size)
    grid = images.split_grid(img, tile, tile, overlap)

    assert np.array_equal(np.asarray(images.combine_grid(grid)), np.asarray(img))


def test_combine_grid_of_scaled_tiles_matches_whole_image_scaling(images):
    """As upscaler_utils.upscale_with_model rebuilds the grid; nearest 2x of each tile == nearest 2x of the image."""
    img = _random_rgb(188, 150, seed=1)
    grid = images.split_grid(img, 192, 192, 8)
    scale = 2
    tiles = [[y * scale, h * scale, [[x * scale, w * scale, t.resize((t.width * scale, t.height * scale), Image.Resampling.NEAREST)] for x, w, t in row]] for y, h, row in grid.tiles]
    scaled = images.Grid(tiles, grid.tile_w * scale, grid.tile_h * scale, grid.image_w * scale, grid.image_h * scale, grid.overlap * scale)

    expected = img.resize((img.width * scale, img.height * scale), Image.Resampling.NEAREST)
    assert np.array_equal(np.asarray(images.combine_grid(scaled)), np.asarray(expected))


# --- grid annotations ------------------------------------------------------------------------------------------------

def test_grid_annotations_render_on_current_pillow(images):
    """multiline_textsize was removed in Pillow 10; x/y/z plot and prompt matrix legends raised AttributeError."""
    cell = 64
    grid = Image.new("RGB", (2 * cell, cell), "black")
    hor = [[images.GridAnnotation("a much longer label than the cell is wide")], [images.GridAnnotation("b", is_active=False)]]
    ver = [[images.GridAnnotation("")]]

    result = images.draw_grid_annotations(grid, cell, cell, hor, ver)

    assert result.width == grid.width
    assert result.height > grid.height  # a text band was added above the cells


# --- FilenameGenerator -----------------------------------------------------------------------------------------------

def test_hasprompt_without_prompt_returns_none(images):
    namegen = images.FilenameGenerator(SimpleNamespace(), seed=1, prompt=None, image=Image.new("RGB", (1, 1)))

    assert namegen.hasprompt("cat|dog") is None


# --- save_image ------------------------------------------------------------------------------------------------------

def test_save_image_truncates_multibyte_filename_to_the_byte_limit(images, tmp_path):
    name_max = os.statvfs(tmp_path).f_namemax
    stem = "猫" * name_max  # 3 bytes per character: fits the old character budget check, not the byte limit

    fullfn, _ = images.save_image(Image.new("RGB", (2, 2)), str(tmp_path), "", extension="png", forced_filename=stem, save_to_dirs=False)

    saved = Path(fullfn)
    assert saved.is_file()
    assert len(os.fsencode(saved.name)) <= name_max
    assert saved.stem == stem[: (name_max - 4) // 3]
    assert not list(tmp_path.glob("*.tmp"))


def test_truncate_to_fs_bytes_never_splits_a_character(images):
    for text in ("ascii-only", "猫a猫b", "é" * 7, "😀x😀"):
        for limit in range(0, 12):
            truncated = images.truncate_to_fs_bytes(text, limit)
            assert text.startswith(truncated)
            assert len(os.fsencode(truncated)) <= limit
            assert len(truncated) == len(text) or len(os.fsencode(text[: len(truncated) + 1])) > limit


def test_failed_save_leaves_no_temporary_file(images, tmp_path, monkeypatch):
    def failing_insert(exif, filename):
        raise ValueError("EXIF APP1 segment too long")

    monkeypatch.setattr(images.piexif, "insert", failing_insert)

    with pytest.raises(ValueError):
        images.save_image(Image.new("RGB", (2, 2)), str(tmp_path), "", extension="jpg", info="Steps: 20", forced_filename="x", save_to_dirs=False)

    assert list(tmp_path.iterdir()) == []


def test_sixteen_bit_images_round_to_eight_bits(images, tmp_path):
    """Every I;16 value v must become round(v / 257) (65535 -> 255); the point transform used to truncate."""
    values = np.arange(65536, dtype=np.uint16).reshape(256, 256)
    path = tmp_path / "i16.webp"

    images.save_image_with_geninfo(Image.fromarray(values), None, str(path))  # lossless WebP (opts.webp_lossless)

    with Image.open(path) as saved:
        got = np.asarray(saved.convert("RGB"))[..., 0].astype(np.int64)
    assert np.array_equal(got, (values.astype(np.int64) + 128) // 257)  # 257 is odd: no ties


# --- EXIF UserComment (untrusted) ------------------------------------------------------------------------------------

def _exif_with_comment(text):
    return piexif.dump({"Exif": {piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump(text, encoding="unicode")}})


@pytest.mark.parametrize("extension", [".jpg", ".webp", ".avif", ".png"])
def test_saved_geninfo_reads_back(images, tmp_path, extension):
    text = "a cat, Steps: 20, Sampler: DPM++ 2M, ünïcödé ✓"
    path = tmp_path / f"x{extension}"
    if extension == ".png":
        Image.new("RGB", (8, 8)).save(path, exif=_exif_with_comment(text))  # eXIf chunk, no parameters text
    else:
        images.save_image_with_geninfo(Image.new("RGB", (8, 8)), text, str(path))

    with Image.open(path) as image:
        assert images.read_info_from_image(image)[0] == text


# --- PNG encoder selection (test/test_png_writer.py covers the encoder itself) -----------------------------------------

def _decoded_png(path):
    with Image.open(path) as image:
        image.load()
        return image.mode, image.size, image.tobytes(), image.text


def _pillow_png(image, text, extra=None):
    info = PngImagePlugin.PngInfo()
    for key, value in {**(extra or {}), "parameters": text}.items():
        info.add_text(key, value)
    output = io.BytesIO()
    image.save(output, format="PNG", pnginfo=info)
    return output.getvalue()


def test_png_saves_use_the_parallel_writer_unless_disabled(images, tmp_path, monkeypatch):
    calls = []
    encode = images.png_writer.encode
    monkeypatch.setattr(images.png_writer, "encode", lambda *args: calls.append(encode(*args)) or calls[-1])
    image = _random_rgb(300, 200, seed=3)
    text = "a cat, Steps: 20, \xfcn\xefc\xf6d\xe9 \u2713"

    images.save_image_with_geninfo(image, text, str(tmp_path / "parallel.png"), existing_pnginfo={"extra": "x"})
    assert len(calls) == 1 and calls[0] is not None  # the parallel writer covered the image and wrote the file
    assert (tmp_path / "parallel.png").read_bytes() == calls[0]
    monkeypatch.setattr(images.opts, "png_parallel_encoder", False)
    images.save_image_with_geninfo(image, text, str(tmp_path / "pillow.png"), existing_pnginfo={"extra": "x"})
    assert len(calls) == 1

    assert (tmp_path / "pillow.png").read_bytes() == _pillow_png(image, text, {"extra": "x"})
    assert _decoded_png(tmp_path / "parallel.png") == _decoded_png(tmp_path / "pillow.png")
    assert _decoded_png(tmp_path / "parallel.png")[3] == {"extra": "x", "parameters": text}


def test_png_writer_failures_and_uncovered_modes_save_with_pillow(images, tmp_path, monkeypatch):
    reports = []
    monkeypatch.setattr(images.errors, "report", lambda *args, **kwargs: reports.append(args))
    gray = _random_rgb(40, 30).convert("L")
    images.save_image_with_geninfo(gray, "Steps: 20", str(tmp_path / "gray.png"))
    assert (tmp_path / "gray.png").read_bytes() == _pillow_png(gray, "Steps: 20") and not reports

    def broken(*args):
        raise RuntimeError("encoder defect")

    monkeypatch.setattr(images.png_writer, "encode", broken)
    image = _random_rgb(40, 30)
    images.save_image_with_geninfo(image, "Steps: 20", str(tmp_path / "x.png"))
    assert (tmp_path / "x.png").read_bytes() == _pillow_png(image, "Steps: 20")
    assert len(reports) == 1  # reported, not silent


def test_png_write_failure_removes_the_file_it_created(images, tmp_path, monkeypatch):
    class FullDisk(io.FileIO):
        def write(self, data):
            super().write(data[:10])
            raise OSError(28, "No space left on device")

    monkeypatch.setattr(images, "open", lambda path, mode: FullDisk(path, "w"), raising=False)
    with pytest.raises(OSError):
        images.save_image_with_geninfo(_random_rgb(40, 30), None, str(tmp_path / "x.png"))
    assert list(tmp_path.iterdir()) == []


def _tiff_with_ascii_user_comment(payload):
    """Little-endian TIFF/EXIF bytes whose Exif IFD holds UserComment (0x9286) with the ASCII type (2)."""
    import struct

    ifd0 = struct.pack("<HHHII", 1, 0x8769, 4, 1, 26) + struct.pack("<I", 0)  # one entry: Exif IFD pointer at 26
    exif_ifd = struct.pack("<HHHII", 1, 0x9286, 2, len(payload), 26 + 18) + struct.pack("<I", 0)
    return b"II*\x00" + struct.pack("<I", 8) + ifd0 + exif_ifd + payload


@pytest.mark.parametrize("prefix", [b"", b"Exif\x00\x00"])
def test_ascii_typed_user_comment_is_read(images, prefix):
    # EXIF specifies UNDEFINED with an 8-byte code prefix, but Pillow writes a str UserComment with the ASCII type.
    # piexif-based parsing read such comments (as UTF-8 text); they must not be dropped.
    text = "a cat, Steps: 20, Sampler: Euler, ünïcödé"
    utf8 = _tiff_with_ascii_user_comment(text.encode("utf8") + b"\x00")
    assert images.read_info_from_image(SimpleNamespace(info={"exif": prefix + utf8}))[0] == text

    exif = Image.Exif()
    exif.get_ifd(0x8769)[0x9286] = "Steps: 20, Sampler: Euler"
    assert images.read_info_from_image(SimpleNamespace(info={"exif": prefix + exif.tobytes().removeprefix(b"Exif\x00\x00")}))[0] == "Steps: 20, Sampler: Euler"


def test_exif_bytes_are_never_opened_as_a_file_path(images, tmp_path, monkeypatch):
    secret = tmp_path / "server-side.jpg"
    Image.new("RGB", (4, 4)).save(secret)
    piexif.insert(_exif_with_comment("SECRET PROMPT"), str(secret))

    upload = io.BytesIO()
    Image.new("RGB", (4, 4)).save(upload, format="WEBP", exif=os.fsencode(secret))  # EXIF chunk = a path
    upload.seek(0)
    monkeypatch.setattr(images.piexif, "load", lambda *a, **k: pytest.fail("piexif.load must not parse uploaded EXIF"))

    with Image.open(upload) as image:
        assert image.info["exif"] == os.fsencode(secret)
        assert images.read_info_from_image(image)[0] is None


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_crafted_exif_count_is_bounded(images):
    # 90 bytes: the Exif IFD pointer entry declares 0x30000001 LONGs; piexif.load builds a format string and tuple
    # of that length (several GB; found by fuzzing, it OOM-killed a 6 GB container). Pillow reads only what exists.
    crafted = bytes.fromhex(
        "4578696600004d4d002a000000080002011200030000000100010000876900043000000100000026000000000001928600070000001e00dd"
        "0034554e49434f444500c2680e65006c006c006f00200077006f0072006c0064"
    )
    assert images.read_info_from_image(SimpleNamespace(info={"exif": crafted, "parameters": "kept"}))[0] == "kept"


@pytest.mark.filterwarnings("ignore::UserWarning")  # Pillow warns about each corrupt IFD it skips
def test_malformed_exif_never_raises(images):
    valid = piexif.dump({
        "0th": {piexif.ImageIFD.Orientation: 1, piexif.ImageIFD.Software: b"x" * 10},
        "Exif": {piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump("hello", encoding="unicode"), piexif.ExifIFD.ISOSpeed: 100},
    })
    assert images.read_info_from_image(SimpleNamespace(info={"exif": valid}))[0] == "hello"
    rng = random.Random(0)
    for _ in range(3000):
        data = bytearray(valid)
        for _ in range(rng.randint(1, 5)):
            data[rng.randrange(len(data))] = rng.randrange(256)
        if rng.random() < 0.2:
            data = data[: rng.randrange(len(data))]
        geninfo, _ = images.read_info_from_image(SimpleNamespace(info={"exif": bytes(data)}))
        assert geninfo is None or isinstance(geninfo, str)


# --- flatten ---------------------------------------------------------------------------------------------------------

def _transparent_inputs():
    la = Image.merge("LA", (Image.frombytes("L", (2, 1), bytes([10, 40])), Image.frombytes("L", (2, 1), bytes([0, 255]))))
    p = Image.frombytes("P", (2, 1), bytes([0, 1]))
    p.putpalette([10, 20, 30, 40, 50, 60])
    p.info["transparency"] = 0  # single transparent palette index (PNG tRNS / GIF)
    rgb = Image.frombytes("RGB", (2, 1), bytes([10, 20, 30, 40, 50, 60]))
    rgb.info["transparency"] = (10, 20, 30)  # PNG colour key
    gray = Image.frombytes("L", (2, 1), bytes([10, 40]))
    gray.info["transparency"] = 10
    return {"LA": la, "P": p, "RGB": rgb, "L": gray}


@pytest.mark.parametrize("name", ["LA", "P", "RGB", "L"])
def test_flatten_fills_transparency_of_every_mode(images, name):
    """img2img_background_color: 'fill transparent parts of the input image with this color' held only for RGBA."""
    img = _transparent_inputs()[name]
    expected_opaque = img.convert("RGBA").getpixel((1, 0))[:3]

    flat = images.flatten(img, "#ff00ff")

    assert flat.mode == "RGB"
    assert flat.getpixel((0, 0)) == (255, 0, 255)
    assert flat.getpixel((1, 0)) == expected_opaque
    assert np.array_equal(np.asarray(flat), np.asarray(images.flatten(img.convert("RGBA"), "#ff00ff")))


def test_flatten_partial_alpha_matches_rgba_path(images):
    alpha = Image.frombytes("L", (4, 1), bytes([0, 64, 191, 255]))
    la = Image.merge("LA", (Image.frombytes("L", (4, 1), bytes([200, 200, 200, 200])), alpha))

    assert np.array_equal(np.asarray(images.flatten(la, "#000000")), np.asarray(images.flatten(la.convert("RGBA"), "#000000")))


@pytest.mark.parametrize("mode", ["1", "L", "P", "RGB", "I", "I;16", "F", "CMYK", "YCbCr"])
def test_flatten_without_transparency_is_plain_rgb_conversion(images, mode):
    img = _random_rgb(5, 3, seed=2).convert(mode) if mode not in ("I;16",) else Image.fromarray(np.arange(15, dtype=np.uint16).reshape(3, 5))
    assert not img.has_transparency_data

    assert np.array_equal(np.asarray(images.flatten(img, "#ff00ff")), np.asarray(img.convert("RGB")))


# --- read ------------------------------------------------------------------------------------------------------------

def test_read_rejects_truncated_data_at_the_boundary(images):
    data = io.BytesIO()
    _random_rgb(64, 64).save(data, format="PNG")
    raw = data.getvalue()

    with pytest.raises(OSError):
        images.read(io.BytesIO(raw[: len(raw) // 2]))

    assert np.array_equal(np.asarray(images.read(io.BytesIO(raw))), np.asarray(_random_rgb(64, 64)))


def test_read_refuses_more_than_max_pixels_from_the_header(images):
    """The pixel budget is checked before decoding: truncated pixel data over the budget raises the budget error, not
    the decoder's."""
    data = io.BytesIO()
    _random_rgb(40, 26).save(data, format="PNG")
    raw = data.getvalue()

    with pytest.raises(Image.DecompressionBombError):
        images.read(io.BytesIO(raw[: len(raw) // 2]), max_pixels=40 * 26 - 1)
    with pytest.raises(OSError):
        images.read(io.BytesIO(raw[: len(raw) // 2]), max_pixels=40 * 26)

    assert images.read(io.BytesIO(raw), max_pixels=40 * 26).size == (40, 26)
