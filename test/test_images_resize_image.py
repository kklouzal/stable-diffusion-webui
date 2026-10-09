import pytest
from PIL import Image

from test.helpers import init_shared

init_shared()

from modules import images  # noqa: E402


@pytest.fixture
def images_module(monkeypatch):
    monkeypatch.setattr(images.opts, "upscaler_for_img2img", "None")  # plain LANCZOS resizes, no upscaler model
    return images


def test_resize_image_crop_mode_covers_target_after_fractional_aspect_rounding(images_module):
    src = Image.new('RGB', (2, 3), color=(10, 20, 30))

    result = images_module.resize_image(1, src, 3, 2)

    assert result.size == (3, 2)


def test_resize_image_fill_mode_keeps_nonzero_intermediate_for_extreme_aspect(images_module):
    src = Image.new('RGB', (1, 2), color=(10, 20, 30))

    result = images_module.resize_image(2, src, 1, 1)

    assert result.size == (1, 1)
