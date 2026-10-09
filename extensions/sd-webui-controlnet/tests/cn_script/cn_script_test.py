import unittest
from types import SimpleNamespace
from unittest import mock
from PIL import Image
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
        p = processing.StableDiffusionProcessing()
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
