from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from test.helpers import load_source, module, stub_modules


def _uncrop(image, dest_size, paste_loc):
    x, y, w, h = paste_loc
    base_image = Image.new('RGBA', dest_size)
    base_image.paste(image.resize((w, h)), (x, y))
    return base_image


@pytest.fixture()
def soft_inpainting():
    # The script imports modules.processing/images/shared inside its methods: the stubs stay installed for the test.
    with stub_modules({
        "modules": module("modules", package=True),
        "modules.headless_ui": module("modules.headless_ui"),
        "modules.ui_components": module("modules.ui_components", InputAccordion=object),
        "modules.scripts": module(
            "modules.scripts", Script=object, AlwaysVisible=object(),
            MaskBlendArgs=object, PostSampleArgs=object, PostProcessMaskOverlayArgs=object,
        ),
        "modules.torch_utils": module("modules.torch_utils", float64=lambda _tensor: torch.float64),
        "modules.processing": module("modules.processing", uncrop=_uncrop, create_binary_mask=lambda image, round=True: image.convert('L')),
        "modules.images": module(
            "modules.images", resize_image=lambda _mode, image, width, height: image.resize((width, height)),
            flatten=lambda image, _color: image.convert('RGB'),
        ),
        "modules.shared": module("modules.shared", opts=SimpleNamespace(img2img_background_color="#ffffff")),
    }):
        yield load_source("soft_inpainting_under_test", "extensions-builtin/soft-inpainting/scripts/soft_inpainting.py")


def reference_weighted_histogram_filter(img, kernel, kernel_center, percentile_min=0.0, percentile_max=1.0, min_width=1.0):
    """The original per-pixel implementation, kept verbatim as the test oracle."""
    vec = np.array

    kernel_min = -kernel_center
    kernel_max = vec(kernel.shape) - kernel_center

    def weighted_histogram_filter_single(idx):
        idx = vec(idx)
        min_index = np.maximum(0, idx + kernel_min)
        max_index = np.minimum(vec(img.shape), idx + kernel_max)
        window_shape = max_index - min_index

        class WeightedElement:
            def __init__(self, value, weight):
                self.value: float = value
                self.weight: float = weight
                self.window_min: float = 0.0
                self.window_max: float = 1.0

        values = []
        for window_tup in np.ndindex(tuple(window_shape)):
            window_index = vec(window_tup)
            image_index = window_index + min_index
            centered_kernel_index = image_index - idx
            kernel_index = centered_kernel_index + kernel_center
            element = WeightedElement(img[tuple(image_index)], kernel[tuple(kernel_index)])
            values.append(element)

        def sort_key(x: WeightedElement):
            return x.value

        values.sort(key=sort_key)

        sum = 0
        for i in range(len(values)):
            values[i].window_min = sum
            sum += values[i].weight
            values[i].window_max = sum

        window_min = sum * percentile_min
        window_max = sum * percentile_max
        window_width = window_max - window_min

        if window_width < min_width:
            window_center = (window_min + window_max) / 2
            window_min = window_center - min_width / 2
            window_max = window_center + min_width / 2

            if window_max > sum:
                window_max = sum
                window_min = sum - min_width

            if window_min < 0:
                window_min = 0
                window_max = min_width

        value = 0
        value_weight = 0

        for i in range(len(values)):
            if window_min >= values[i].window_max:
                continue
            if window_max <= values[i].window_min:
                break

            s = max(window_min, values[i].window_min)
            e = min(window_max, values[i].window_max)
            w = e - s

            value += values[i].value * w
            value_weight += w

        return value / value_weight if value_weight != 0 else 0

    img_out = img.copy()

    for index in np.ndindex(img.shape):
        img_out[index] = weighted_histogram_filter_single(index)

    return img_out


def assert_bitwise_equal(actual, expected):
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert np.array_equal(actual.view(np.uint8), expected.view(np.uint8))


def random_images(rng):
    for shape in [(1, 1), (1, 7), (6, 1), (2, 3), (5, 5), (13, 17), (24, 31)]:
        yield rng.uniform(0.0, 3.0, size=shape).astype(np.float32)
        # Few distinct values: exercises the stable tie order of the sort.
        yield (rng.integers(0, 4, size=shape) * 0.5).astype(np.float32)
        yield rng.exponential(5.0, size=shape).astype(np.float32)
        yield rng.normal(0.0, 2.0, size=shape).astype(np.float32)
    yield np.zeros((9, 11), dtype=np.float32)
    yield np.array([[-0.0, 0.0, 1.0], [0.0, -0.0, -1.0]], dtype=np.float32)


def test_weighted_histogram_filter_is_bit_identical_at_the_call_sites(soft_inpainting):
    rng = np.random.default_rng(1234)
    kernel, kernel_center = soft_inpainting.get_gaussian_kernel(stddev_radius=1.5, max_radius=2)
    for img in random_images(rng):
        # Exactly the two chained passes of apply_adaptive_masks.
        expected = reference_weighted_histogram_filter(img, kernel, kernel_center, percentile_min=0.9, percentile_max=1, min_width=1)
        actual = soft_inpainting.weighted_histogram_filter(img, kernel, kernel_center, percentile_min=0.9, percentile_max=1, min_width=1)
        assert_bitwise_equal(actual, expected)
        expected = reference_weighted_histogram_filter(expected, kernel, kernel_center, percentile_min=0.25, percentile_max=0.75, min_width=1)
        actual = soft_inpainting.weighted_histogram_filter(actual, kernel, kernel_center, percentile_min=0.25, percentile_max=0.75, min_width=1)
        assert_bitwise_equal(actual, expected)


def test_weighted_histogram_filter_is_bit_identical_for_random_kernels_and_windows(soft_inpainting):
    rng = np.random.default_rng(99)
    cases = 0
    for img in random_images(rng):
        for _ in range(3):
            kernel_shape = tuple(int(n) for n in rng.integers(1, 6, size=2))
            kernel = rng.uniform(0.0, 1.5, size=kernel_shape)
            kernel[rng.random(kernel_shape) < 0.2] = 0.0
            kernel_center = np.array([rng.integers(0, kernel_shape[0]), rng.integers(0, kernel_shape[1])])
            percentile_min, percentile_max = sorted(float(x) for x in rng.choice([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0], size=2))
            min_width = [1, 0.25, 2.5, 40.0][int(rng.integers(0, 4))]
            expected = reference_weighted_histogram_filter(img, kernel, kernel_center, percentile_min, percentile_max, min_width)
            actual = soft_inpainting.weighted_histogram_filter(img, kernel, kernel_center, percentile_min, percentile_max, min_width)
            assert_bitwise_equal(actual, expected)
            cases += 1
    # An all-zero kernel leaves no weight in any window.
    img = rng.uniform(0.0, 1.0, size=(4, 5)).astype(np.float32)
    kernel = np.zeros((3, 3))
    assert_bitwise_equal(soft_inpainting.weighted_histogram_filter(img, kernel, 1), reference_weighted_histogram_filter(img, kernel, 1))
    assert cases == 3 * 30


def test_weighted_histogram_filter_blocks_tall_images_without_changing_bits(soft_inpainting):
    rng = np.random.default_rng(7)
    kernel, kernel_center = soft_inpainting.get_gaussian_kernel(stddev_radius=1.5, max_radius=2)
    # 16384 // 1000 = 16 rows per block: one full block and a short last one, with windows crossing the seam.
    img = rng.uniform(0.0, 3.0, size=(19, 1000)).astype(np.float32)
    expected = reference_weighted_histogram_filter(img, kernel, kernel_center, 0.25, 0.75, 1)
    assert_bitwise_equal(soft_inpainting.weighted_histogram_filter(img, kernel, kernel_center, 0.25, 0.75, 1), expected)


def test_weighted_histogram_filter_rejects_unsupported_inputs(soft_inpainting):
    img = np.zeros((3, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="float64 kernel"):
        soft_inpainting.weighted_histogram_filter(img, np.ones((3, 3), dtype=np.float32), 1)
    with pytest.raises(ValueError, match="containing its center"):
        soft_inpainting.weighted_histogram_filter(img, np.ones((3, 3)), 3)
    with pytest.raises(ValueError, match="2-D image"):
        soft_inpainting.weighted_histogram_filter(np.zeros((2, 3, 3), dtype=np.float32), np.ones((3, 3)), 1)


def _inpaint_full_res_processing(**overrides):
    canvas = Image.new('RGB', (96, 80), (10, 20, 30))
    p = SimpleNamespace(
        image_mask=Image.new('L', (96, 80)),
        nmask=torch.full((4, 8, 8), 0.25),
        init_images=[canvas],
        paste_to=(16, 8, 48, 48),
        resize_mode=0,
        width=64,
        height=64,
        batch_size=1,
    )
    for key, value in overrides.items():
        setattr(p, key, value)
    return p


SETTINGS = (1, 0.5, 4, 0, 0.5, 2)


def test_apply_masks_uncrops_to_the_overlay_size_with_inpaint_full_res(soft_inpainting):
    script = soft_inpainting.Script()
    p = _inpaint_full_res_processing()
    samples = torch.zeros((1, 3, 64, 64))
    samples.already_decoded = True

    script.post_sample(p, SimpleNamespace(samples=samples), True, *SETTINGS)

    assert [mask.size for mask in script.masks_for_overlay] == [(96, 80)]
    assert [image.size for image in script.overlay_images] == [(96, 80)]
    overlay = script.overlay_images[0]
    # Outside the paste region the mask is empty, so the original pixels stay fully opaque.
    assert overlay.getpixel((0, 0)) == (10, 20, 30, 255)
    assert overlay.getpixel((95, 79)) == (10, 20, 30, 255)
    # Inside it the 0.25 latent mask makes the original partly transparent.
    assert overlay.getpixel((40, 32))[3] < 255

    ppmo = SimpleNamespace(index=0, mask_for_overlay="core", overlay_image="core")
    script.postprocess_maskoverlay(p, ppmo, True, *SETTINGS)
    assert ppmo.mask_for_overlay is script.masks_for_overlay[0]
    assert ppmo.overlay_image is script.overlay_images[0]


def test_masks_and_overlays_never_leak_into_a_later_batch_or_request(soft_inpainting, monkeypatch):
    script = soft_inpainting.Script()
    p = _inpaint_full_res_processing()
    samples = torch.zeros((1, 3, 64, 64))
    samples.already_decoded = True
    script.post_sample(p, SimpleNamespace(samples=samples), True, *SETTINGS)
    assert script.masks_for_overlay is not None

    # A later batch whose mask build fails must fall back to the core overlay, not reuse the previous masks.
    def boom(**_kwargs):
        raise RuntimeError("mask build failed")

    monkeypatch.setattr(soft_inpainting, "apply_masks", boom)
    with pytest.raises(RuntimeError, match="mask build failed"):
        script.post_sample(p, SimpleNamespace(samples=samples), True, *SETTINGS)
    ppmo = SimpleNamespace(index=0, mask_for_overlay="core", overlay_image="core")
    script.postprocess_maskoverlay(p, ppmo, True, *SETTINGS)
    assert (ppmo.mask_for_overlay, ppmo.overlay_image) == ("core", "core")

    # A request without a latent mask must not see an earlier request's masks either.
    script.masks_for_overlay, script.overlay_images = ["stale"], ["stale"]
    script.post_sample(_inpaint_full_res_processing(nmask=None), SimpleNamespace(samples=samples), True, *SETTINGS)
    assert (script.masks_for_overlay, script.overlay_images) == (None, None)

    # The finished request releases its full-resolution overlays.
    script.masks_for_overlay, script.overlay_images = ["done"], ["done"]
    script.postprocess(p, SimpleNamespace(), False, *SETTINGS)
    assert (script.masks_for_overlay, script.overlay_images) == (None, None)
