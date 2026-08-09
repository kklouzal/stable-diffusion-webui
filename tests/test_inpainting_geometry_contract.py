import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

_STUB_MODULES = []


def _setdefault_stub(name, module):
    if name not in sys.modules:
        sys.modules[name] = module
        _STUB_MODULES.append(name)
    return sys.modules[name]


def _cleanup_import_stubs():
    while _STUB_MODULES:
        name = _STUB_MODULES.pop()
        module = sys.modules.pop(name, None)
        if '.' in name:
            parent_name, attr = name.rsplit('.', 1)
            parent = sys.modules.get(parent_name)
            if parent is not None and hasattr(parent, attr) and getattr(parent, attr, None) is module:
                delattr(parent, attr)
    modules_pkg = sys.modules.get('modules')
    images_module = sys.modules.pop('modules.images', None)
    if modules_pkg is not None and hasattr(modules_pkg, 'images') and getattr(modules_pkg, 'images', None) is images_module:
        delattr(modules_pkg, 'images')


@pytest.fixture
def images_module():
    for name in ('numpy', 'pytz', 'pillow_avif'):
        if importlib.util.find_spec(name) is None:
            _setdefault_stub(name, types.ModuleType(name))

    if importlib.util.find_spec('piexif') is None:
        piexif = _setdefault_stub('piexif', types.ModuleType('piexif'))
        piexif_helper = _setdefault_stub('piexif.helper', types.ModuleType('piexif.helper'))
        piexif.helper = piexif_helper

    shared = types.ModuleType('modules.shared')
    shared.opts = SimpleNamespace(
        upscaler_for_img2img='None',
        n_rows=-1,
        grid_prevent_empty_spots=False,
        grid_background_color='#000000',
        grid_text_active_color='#000000',
        grid_text_inactive_color='#000000',
        font='',
    )
    shared.state = SimpleNamespace(interrupted=False, skipped=False)
    shared.cmd_opts = SimpleNamespace(unix_filenames_sanitization=False, filenames_max_length=128)
    shared.sd_upscalers = []
    _setdefault_stub('modules.shared', shared)

    _setdefault_stub('modules.sd_samplers', types.ModuleType('modules.sd_samplers'))
    _setdefault_stub('modules.script_callbacks', types.ModuleType('modules.script_callbacks'))
    _setdefault_stub('modules.errors', types.ModuleType('modules.errors'))
    paths_internal = types.ModuleType('modules.paths_internal')
    webui_root = str(Path(__file__).parents[1])
    paths_internal.roboto_ttf_file = ''
    paths_internal.models_path = f'{webui_root}/models'
    paths_internal.script_path = webui_root
    paths_internal.data_path = webui_root
    paths_internal.extensions_dir = f'{webui_root}/extensions'
    paths_internal.extensions_builtin_dir = f'{webui_root}/extensions-builtin'
    paths_internal.cwd = webui_root
    _setdefault_stub('modules.paths_internal', paths_internal)

    from modules import images
    yield images
    _cleanup_import_stubs()


def test_expand_crop_region_ceil_expands_to_requested_aspect_when_room_exists():
    from modules import masking

    crop = (0, 0, 1, 1)
    expanded = masking.expand_crop_region(crop, 768, 512, 512, 512)

    assert expanded == (0, 0, 2, 1)
    assert expanded[0] <= crop[0] and expanded[1] <= crop[1]
    assert expanded[2] >= crop[2] and expanded[3] >= crop[3]


def test_expand_crop_region_ceil_height_expansion_when_room_exists():
    from modules import masking

    crop = (0, 0, 3, 1)
    expanded = masking.expand_crop_region(crop, 512, 512, 512, 512)

    assert expanded == (0, 0, 3, 3)
    assert expanded[0] <= crop[0] and expanded[1] <= crop[1]
    assert expanded[2] >= crop[2] and expanded[3] >= crop[3]


def test_resize_image_crop_mode_covers_target_after_fractional_aspect_rounding(images_module):
    src = Image.new('RGB', (2, 3), color=(10, 20, 30))

    result = images_module.resize_image(1, src, 3, 2)

    assert result.size == (3, 2)


def test_resize_image_fill_mode_keeps_nonzero_intermediate_for_extreme_aspect(images_module):
    src = Image.new('RGB', (1, 2), color=(10, 20, 30))

    result = images_module.resize_image(2, src, 1, 1)

    assert result.size == (1, 1)
