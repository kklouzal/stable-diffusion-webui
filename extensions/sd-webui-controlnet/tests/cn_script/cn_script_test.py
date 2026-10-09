import unittest
from types import SimpleNamespace
from unittest import mock
from PIL import Image, ImageOps
import cv2
import numpy as np

import importlib

utils = importlib.import_module("extensions.sd-webui-controlnet.tests.utils", "utils")


from scripts.enums import ResizeMode
from scripts.controlnet import prepare_mask, Script, set_numpy_seed
from internal_controlnet import external_code
from internal_controlnet.external_code import ControlNetUnit
from modules import processing, shared


class TestPrepareMask(unittest.TestCase):
    def test_prepare_mask(self):
        p = processing.StableDiffusionProcessingImg2Img()
        p.inpainting_mask_invert = True
        p.mask_blur_x = p.mask_blur_y = 5

        mask = Image.new("RGB", (10, 10), color="white")

        processed_mask = prepare_mask(mask, p)

        # Check that mask is correctly converted to grayscale
        self.assertTrue(processed_mask.mode, "L")

        # Check that mask colors are correctly inverted
        self.assertEqual(
            processed_mask.getpixel((0, 0)), 0
        )  # inverted white should be black

        p.inpainting_mask_invert = False
        processed_mask = prepare_mask(mask, p)

        # Check that mask colors are not inverted when 'inpainting_mask_invert' is False
        self.assertEqual(
            processed_mask.getpixel((0, 0)), 255
        )  # white should remain white

        p.mask_blur_x = p.mask_blur_y = 0
        mask = Image.new("RGB", (10, 10), color="black")
        processed_mask = prepare_mask(mask, p)

        # Check that mask is not blurred when 'mask_blur' is 0
        self.assertEqual(
            processed_mask.getpixel((0, 0)), 0
        )  # black should remain black


    def test_matches_the_core_inpaint_mask(self):
        """StableDiffusionProcessingImg2Img.init: create_binary_mask(round=mask_round), invert, blur."""
        alpha = np.zeros((40, 50, 4), np.uint8)
        alpha[10:20, 5:30, 3] = 255
        alpha[25, 30:40, 3] = 100  # soft alpha: dropped when rounding, kept as is otherwise
        rgb = np.random.default_rng(0).integers(0, 256, (40, 50, 3), dtype=np.uint8)
        for mask in (Image.fromarray(alpha, "RGBA"), Image.fromarray(rgb, "RGB"), Image.fromarray(rgb[..., 0], "L")):
            for mask_round in (True, False):
                for invert in (False, True):
                    with self.subTest(mode=mask.mode, mask_round=mask_round, invert=invert):
                        p = processing.StableDiffusionProcessingImg2Img(mask_round=mask_round, inpainting_mask_invert=invert)
                        p.mask_blur_x = p.mask_blur_y = 0
                        expected = processing.create_binary_mask(mask, round=mask_round)
                        if invert:
                            expected = ImageOps.invert(expected)
                        processed = prepare_mask(mask, p)
                        self.assertEqual(processed.mode, "L")
                        np.testing.assert_array_equal(np.asarray(processed), np.asarray(expected))
                        if mask.mode != "RGBA":  # L and RGB masks: unchanged grayscale conversion
                            gray = np.asarray(mask.convert("L"))
                            np.testing.assert_array_equal(np.asarray(processed), 255 - gray if invert else gray)
        # The transparent mask is its alpha channel, not the (black) color channels.
        p = processing.StableDiffusionProcessingImg2Img()
        p.mask_blur_x = p.mask_blur_y = 0
        self.assertEqual(prepare_mask(Image.fromarray(alpha, "RGBA"), p).getbbox(), (5, 10, 30, 20))


class TestCropWithA1111Mask(unittest.TestCase):
    """Inpaint "Only masked" crops the ControlNet input image to the core's crop region."""

    def processing(self, mask):
        p = processing.StableDiffusionProcessingImg2Img(
            width=64, height=32, inpaint_full_res=True, inpaint_full_res_padding=4, mask=mask)
        p.mask_blur_x = p.mask_blur_y = 0
        p.extra_generation_params = {}
        return p

    def crop(self, p, image):
        unit = ControlNetUnit(module="canny", inpaint_crop_input_image=True)
        return Script.try_crop_image_with_a1111_mask(p, unit, image, ResizeMode.RESIZE)

    def test_crop_region_is_the_cores(self):
        from modules import images, masking

        image = np.random.default_rng(2).integers(0, 256, (90, 120, 3), dtype=np.uint8)
        alpha = np.zeros((90, 120, 4), np.uint8)
        alpha[30:50, 40:70, 3] = 255
        p = self.processing(Image.fromarray(alpha, "RGBA"))
        core_mask = processing.create_binary_mask(p.image_mask, round=p.mask_round)
        region = masking.expand_crop_region(
            masking.get_crop_region_v2(core_mask, p.inpaint_full_res_padding), p.width, p.height, 120, 90)
        expected = np.stack([
            np.asarray(images.resize_image(ResizeMode.OUTER_FIT.int_value(), Image.fromarray(image[:, :, i]).crop(region), p.width, p.height))[:, :, 0]
            for i in range(3)
        ], axis=2)
        np.testing.assert_array_equal(self.crop(p, image), expected)

    def test_blank_mask_does_not_crop(self):
        image = np.random.default_rng(3).integers(0, 256, (90, 120, 3), dtype=np.uint8)
        for mask in (Image.new("L", (120, 90)), Image.new("RGBA", (120, 90))):
            with self.subTest(mode=mask.mode):
                self.assertIs(self.crop(self.processing(mask), image), image)


class TestDetectmapResizeInterpolation(unittest.TestCase):
    def test_gray_maps_with_few_levels_are_not_resized_nearest(self):
        rng = np.random.default_rng(4)
        gray = np.repeat(np.linspace(90, 140, 48)[:, None], 40, axis=1).astype(np.uint8)  # smooth, 51 levels
        _, up = Script.detectmap_proc(gray, "depth_zoe", ResizeMode.RESIZE, 96, 80)
        np.testing.assert_array_equal(up, cv2.resize(np.stack([gray] * 3, axis=2), (80, 96), interpolation=cv2.INTER_CUBIC))

        palette = rng.integers(0, 256, (12, 3), dtype=np.uint8)
        seg = palette[rng.integers(0, 12, (6, 5))].repeat(8, axis=0).repeat(8, axis=1)
        _, up = Script.detectmap_proc(seg, "seg_anime_face", ResizeMode.RESIZE, 96, 80)
        np.testing.assert_array_equal(up, cv2.resize(seg, (80, 96), interpolation=cv2.INTER_NEAREST))


class TestSetNumpySeed(unittest.TestCase):
    def test_seed_subseed_minus_one(self):
        p = processing.StableDiffusionProcessing()
        p.seed = -1
        p.subseed = -1
        p.all_seeds = [123, 456]
        expected_seed = (123 + 123) & 0xFFFFFFFF
        self.assertEqual(set_numpy_seed(p), expected_seed)

    def test_valid_seed_subseed(self):
        p = processing.StableDiffusionProcessing()
        p.seed = 50
        p.subseed = 100
        p.all_seeds = [123, 456]
        expected_seed = (50 + 100) & 0xFFFFFFFF
        self.assertEqual(set_numpy_seed(p), expected_seed)

    def test_invalid_seed_subseed(self):
        p = processing.StableDiffusionProcessing()
        p.seed = "invalid"
        p.subseed = 2.5
        p.all_seeds = [123, 456]
        self.assertEqual(set_numpy_seed(p), None)

    def test_empty_all_seeds(self):
        p = processing.StableDiffusionProcessing()
        p.seed = -1
        p.subseed = 2
        p.all_seeds = []
        self.assertEqual(set_numpy_seed(p), None)

    def test_random_state_change(self):
        p = processing.StableDiffusionProcessing()
        p.seed = 50
        p.subseed = 100
        p.all_seeds = [123, 456]
        expected_seed = (50 + 100) & 0xFFFFFFFF

        np.random.seed(0)  # set a known seed
        before_random = np.random.randint(0, 1000)  # get a random integer

        seed = set_numpy_seed(p)
        self.assertEqual(seed, expected_seed)

        after_random = np.random.randint(0, 1000)  # get another random integer

        self.assertNotEqual(before_random, after_random)


class MockImg2ImgProcessing(processing.StableDiffusionProcessing):
    """Mock the Img2Img processing as the WebUI version have dependency on
    `sd_model`."""

    def __init__(self, init_images, resize_mode, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_images = init_images
        self.resize_mode = resize_mode


class TestScript(unittest.TestCase):
    sample_base64_image = (
        "data:image/png;base64,"
        "iVBORw0KGgoAAAANSUhEUgAAARMAAAC3CAIAAAC+MS2jAAAAqUlEQVR4nO3BAQ"
        "0AAADCoPdPbQ8HFAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAA/wZOlAAB5tU+nAAAAABJRU5ErkJggg=="
    )

    sample_np_image = np.zeros(shape=[8, 8, 3], dtype=np.uint8)

    def test_choose_input_image(self):
        with self.subTest(name="no image"):
            with self.assertRaises(ValueError):
                Script.choose_input_image(
                    p=processing.StableDiffusionProcessing(),
                    unit=ControlNetUnit(),
                )

        with self.subTest(name="control net input"):
            _, resize_mode = Script.choose_input_image(
                p=MockImg2ImgProcessing(
                    init_images=[TestScript.sample_np_image],
                    resize_mode=ResizeMode.OUTER_FIT,
                ),
                unit=ControlNetUnit(
                    image=TestScript.sample_np_image,
                    module="none",
                    resize_mode=ResizeMode.INNER_FIT,
                ),
            )
            self.assertEqual(resize_mode, ResizeMode.INNER_FIT)

        with self.subTest(name="A1111 input"):
            _, resize_mode = Script.choose_input_image(
                p=MockImg2ImgProcessing(
                    init_images=[TestScript.sample_np_image],
                    resize_mode=ResizeMode.OUTER_FIT,
                ),
                unit=ControlNetUnit(
                    module="none",
                    resize_mode=ResizeMode.INNER_FIT,
                ),
            )
            self.assertEqual(resize_mode, ResizeMode.OUTER_FIT)


def _processing_with_controlnet_args(script_args, **attributes):
    """A processing stand-in whose ControlNet script owns script_args[0:3] (control_net_unit_count == 3)."""
    cn_script = SimpleNamespace(title=lambda: "ControlNet", args_from=0, args_to=3)
    p = SimpleNamespace(
        scripts=SimpleNamespace(alwayson_scripts=[cn_script]),
        script_args=script_args,
        extra_generation_params={},
        **attributes,
    )
    return p, cn_script


class TestControlNetUnitSources(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.dict(shared.opts.data, {"control_net_allow_script_control": True, "control_net_unit_count": 3})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_legacy_remote_fields_do_not_mutate_the_shared_default_units(self):
        # The API passes the same default ControlNetUnit objects to every request that sends no alwayson ControlNet
        # args; the legacy control_net_* fields must configure copies of them, not the defaults.
        defaults = [ControlNetUnit(), ControlNetUnit(), ControlNetUnit()]
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        remote, _ = _processing_with_controlnet_args(
            list(defaults), control_net_enabled=True, control_net_module="canny", control_net_image=image,
            control_net_weight=0.66,
        )
        units = Script.get_enabled_units(remote)
        self.assertEqual([(u.enabled, u.module, u.weight) for u in units], [(True, "canny", 0.66)])
        self.assertIs(units[0].image, image)
        for default in defaults:
            self.assertEqual((default.enabled, default.module, default.weight, default.image), (False, "none", 1.0, None))

        plain, _ = _processing_with_controlnet_args(list(defaults))
        self.assertEqual(Script.get_enabled_units(plain), [])
        self.assertEqual(plain.extra_generation_params, {})

    def test_units_beyond_the_fixed_slots_are_read_from_the_recorded_range(self):
        # modules/api/api.py _assign_script_args: a request with more units than slots fills the slots with the first
        # units, appends the full list and records its range in p.openclaw_script_arg_ranges.
        requested = [ControlNetUnit(enabled=True, module="none", weight=w) for w in (0.1, 0.2, 0.3, 0.4)]
        p, cn_script = _processing_with_controlnet_args(requested[:3] + requested)
        p.openclaw_script_arg_ranges = {id(cn_script): (3, 7)}
        self.assertEqual([u.weight for u in external_code.get_all_units_in_processing(p)], [0.1, 0.2, 0.3, 0.4])
        self.assertEqual([u.weight for u in Script.get_enabled_units(p)], [0.1, 0.2, 0.3, 0.4])

        del p.openclaw_script_arg_ranges
        self.assertEqual([u.weight for u in external_code.get_all_units_in_processing(p)], [0.1, 0.2, 0.3])


if __name__ == "__main__":
    unittest.main()
