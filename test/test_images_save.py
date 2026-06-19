import importlib.util
import sys
import types
from pathlib import Path
import os
from types import SimpleNamespace

from PIL import Image


def module(name, **attrs):
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    return mod


def load_images_module(monkeypatch):
    opts = SimpleNamespace(
        data={},
        enable_pnginfo=False,
        jpeg_quality=80,
        webp_lossless=False,
        save_to_dirs=False,
        grid_save_to_dirs=False,
        samples_filename_pattern="",
        save_images_add_number=False,
        directories_filename_pattern="",
        export_for_4chan=False,
        target_side_length=4000,
        img_downscale_threshold=4,
        save_txt=True,
        font=None,
        n_rows=0,
        grid_prevent_empty_spots=False,
        grid_background_color="#ffffff",
        grid_text_active_color="#000000",
        grid_text_inactive_color="#999999",
        directories_max_prompt_words=8,
        save_images_replace_action="Add number suffix",
    )
    shared = module(
        "modules.shared",
        opts=opts,
        cmd_opts=SimpleNamespace(unix_filenames_sanitization=False, filenames_max_length=128),
        state=SimpleNamespace(job_timestamp=""),
        sd_model=SimpleNamespace(sd_model_hash="", sd_checkpoint_info=SimpleNamespace(name_for_extra="model")),
        prompt_styles=SimpleNamespace(get_style_prompts=lambda styles: []),
        sd_upscalers=[],
    )

    class ImageSaveParams:
        def __init__(self, image, p, filename, pnginfo):
            self.image = image
            self.p = p
            self.filename = filename
            self.pnginfo = pnginfo

    callbacks = module(
        "modules.script_callbacks",
        ImageSaveParams=ImageSaveParams,
        ImageGridLoopParams=lambda imgs, cols, rows: SimpleNamespace(imgs=imgs, cols=cols, rows=rows),
        image_grid_callback=lambda params: None,
        before_image_saved_callback=lambda params: None,
        image_saved_callback=lambda params: None,
    )

    monkeypatch.setitem(sys.modules, "numpy", module("numpy", float32=float, uint8=int))
    monkeypatch.setitem(sys.modules, "pytz", module("pytz", timezone=lambda name: None, exceptions=SimpleNamespace(UnknownTimeZoneError=Exception)))
    piexif = module("piexif", load=lambda data: {}, dump=lambda data: b"", insert=lambda exif, filename: None, ExifIFD=SimpleNamespace(UserComment=0))
    piexif.helper = module("piexif.helper", UserComment=SimpleNamespace(load=lambda data: "", dump=lambda text, encoding=None: b""))
    monkeypatch.setitem(sys.modules, "piexif", piexif)
    monkeypatch.setitem(sys.modules, "piexif.helper", piexif.helper)
    monkeypatch.setitem(sys.modules, "pillow_avif", module("pillow_avif"))
    monkeypatch.setitem(sys.modules, "modules.sd_samplers", module("modules.sd_samplers", find_sampler_config=lambda name: SimpleNamespace(options={}), samplers_map={}))
    monkeypatch.setitem(sys.modules, "modules.shared", shared)
    monkeypatch.setitem(sys.modules, "modules.script_callbacks", callbacks)
    monkeypatch.setitem(sys.modules, "modules.errors", module("modules.errors", report=lambda *a, **k: None, display=lambda *a, **k: None))
    monkeypatch.setitem(sys.modules, "modules.paths_internal", module("modules.paths_internal", roboto_ttf_file=""))

    spec = importlib.util.spec_from_file_location("test_loaded_images", Path("modules/images.py"))
    images = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(images)
    return images


def test_save_image_reports_number_suffixed_duplicate_filename(tmp_path, monkeypatch):
    images = load_images_module(monkeypatch)

    original = tmp_path / "duplicate.png"
    original.write_bytes(b"existing")

    image = Image.new("RGB", (1, 1), color="white")

    fullfn, txt_fullfn = images.save_image(
        image,
        path=str(tmp_path),
        basename="",
        extension="png",
        info="extras metadata",
        forced_filename="duplicate",
        save_to_dirs=False,
    )

    assert fullfn == str(tmp_path / "duplicate-1.png")
    assert txt_fullfn == str(tmp_path / "duplicate-1.txt")
    assert image.already_saved_as == fullfn
    assert original.read_bytes() == b"existing"
    assert (tmp_path / "duplicate-1.png").is_file()
    assert (tmp_path / "duplicate-1.txt").read_text(encoding="utf8") == "extras metadata\n"


def test_save_image_creates_callback_rewritten_directory(tmp_path, monkeypatch):
    images = load_images_module(monkeypatch)

    callback_dir = tmp_path / "callback-target"

    def rewrite_filename(params):
        params.filename = str(callback_dir / "rewritten.webp")

    monkeypatch.setattr(images.script_callbacks, "before_image_saved_callback", rewrite_filename)

    image = Image.new("RGB", (1, 1), color="white")

    fullfn, txt_fullfn = images.save_image(
        image,
        path=str(tmp_path),
        basename="",
        extension="png",
        info="callback metadata",
        forced_filename="original",
        save_to_dirs=False,
    )

    assert fullfn == str(callback_dir / "rewritten.webp")
    assert txt_fullfn == str(callback_dir / "rewritten.txt")
    assert image.already_saved_as == fullfn
    assert (callback_dir / "rewritten.webp").is_file()
    assert (callback_dir / "rewritten.txt").read_text(encoding="utf8") == "callback metadata\n"


def test_save_image_reports_exported_4chan_jpg_path(tmp_path, monkeypatch):
    images = load_images_module(monkeypatch)
    images.opts.export_for_4chan = True
    images.opts.target_side_length = 2

    saved_callbacks = []
    monkeypatch.setattr(images.script_callbacks, "image_saved_callback", saved_callbacks.append)

    image = Image.new("RGB", (4, 2), color="white")

    fullfn, txt_fullfn = images.save_image(
        image,
        path=str(tmp_path),
        basename="",
        extension="png",
        info="export metadata",
        forced_filename="large",
        save_to_dirs=False,
    )

    assert fullfn == str(tmp_path / "large.jpg")
    assert txt_fullfn == str(tmp_path / "large.txt")
    assert image.already_saved_as == fullfn
    assert saved_callbacks[0].filename == fullfn
    assert saved_callbacks[0].image.size == (2, 1)
    assert (tmp_path / "large.png").is_file()
    assert (tmp_path / "large.jpg").is_file()
    assert (tmp_path / "large.txt").read_text(encoding="utf8") == "export metadata\n"

def test_save_image_truncates_long_filename_component_only(tmp_path, monkeypatch):
    images = load_images_module(monkeypatch)
    image = Image.new("RGB", (1, 1), color="white")
    max_stem_len = os.statvfs(tmp_path).f_namemax - 4

    fullfn, txt_fullfn = images.save_image(
        image,
        path=str(tmp_path),
        basename="",
        extension="png",
        info="long filename metadata",
        forced_filename="x" * (max_stem_len + 50),
        save_to_dirs=False,
    )

    saved_path = Path(fullfn)
    assert saved_path.parent == tmp_path
    assert saved_path.name == f"{'x' * max_stem_len}.png"
    assert txt_fullfn == str(tmp_path / f"{'x' * max_stem_len}.txt")
    assert saved_path.is_file()
