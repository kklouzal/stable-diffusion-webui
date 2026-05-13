import os
import tempfile
from collections import namedtuple
from pathlib import Path

from modules import headless_ui as gr

from PIL import PngImagePlugin

from modules import shared


Savedfile = namedtuple("Savedfile", ["name"])


def register_tmp_file(ui_app, filename):
    if hasattr(ui_app, 'temp_file_sets'):  # legacy UI 3.15
        ui_app.temp_file_sets[0] = ui_app.temp_file_sets[0] | {os.path.abspath(filename)}

    if hasattr(ui_app, 'temp_dirs'):  # legacy UI 3.9
        ui_app.temp_dirs = ui_app.temp_dirs | {os.path.abspath(os.path.dirname(filename))}


def check_tmp_file(ui_app, filename):
    if hasattr(ui_app, 'temp_file_sets'):
        return any(filename in fileset for fileset in ui_app.temp_file_sets)

    if hasattr(ui_app, 'temp_dirs'):
        return any(Path(temp_dir).resolve() in Path(filename).resolve().parents for temp_dir in ui_app.temp_dirs)

    return False


def save_pil_to_file(self, pil_image, dir=None, format="png"):
    already_saved_as = getattr(pil_image, 'already_saved_as', None)
    if already_saved_as and os.path.isfile(already_saved_as):
        register_tmp_file(shared.demo, already_saved_as)
        filename_with_mtime = f'{already_saved_as}?{os.path.getmtime(already_saved_as)}'
        register_tmp_file(shared.demo, filename_with_mtime)
        return filename_with_mtime

    if shared.opts.temp_dir != "":
        dir = shared.opts.temp_dir
    else:
        os.makedirs(dir, exist_ok=True)

    use_metadata = False
    metadata = PngImagePlugin.PngInfo()
    for key, value in pil_image.info.items():
        if isinstance(key, str) and isinstance(value, str):
            metadata.add_text(key, value)
            use_metadata = True

    file_obj = tempfile.NamedTemporaryFile(delete=False, suffix=".png", dir=dir)
    pil_image.save(file_obj, pnginfo=(metadata if use_metadata else None))
    return file_obj.name


def install_ui_tempdir_override():
    """override save to file function so that it also writes PNG info"""
    gr.components.IOComponent.pil_to_temp_file = save_pil_to_file


def on_tmpdir_changed():
    if shared.opts.temp_dir == "" or shared.demo is None:
        return

    os.makedirs(shared.opts.temp_dir, exist_ok=True)

    register_tmp_file(shared.demo, os.path.join(shared.opts.temp_dir, "x"))


def cleanup_tmpdr():
    temp_dir = shared.opts.temp_dir
    if temp_dir == "" or not os.path.isdir(temp_dir):
        return

    for root, _, files in os.walk(temp_dir, topdown=False):
        for name in files:
            _, extension = os.path.splitext(name)
            if extension != ".png":
                continue

            filename = os.path.join(root, name)
            os.remove(filename)


def is_ui_temp_path(path):
    """
    Check if the path is a temp dir used by the UI layer
    """
    path = Path(path)
    if shared.opts.temp_dir and path.is_relative_to(shared.opts.temp_dir):
        return True
    if ui_temp_dir := os.environ.get("GRADIO_TEMP_DIR"):
        if path.is_relative_to(ui_temp_dir):
            return True
    if path.is_relative_to(Path(tempfile.gettempdir()) / "gradio"):
        return True
    return False
