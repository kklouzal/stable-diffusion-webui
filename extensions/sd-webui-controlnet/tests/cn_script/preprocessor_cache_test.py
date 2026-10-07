import importlib
import unittest
from unittest import mock

import numpy as np
import torch

utils = importlib.import_module("extensions.sd-webui-controlnet.tests.utils", "utils")

from scripts.supported_preprocessor import CACHE_SIZE, Preprocessor  # noqa: E402
from scripts.preprocessor.legacy.legacy_preprocessors import DETERMINISTIC_LEGACY_PREPROCESSORS  # noqa: E402


class TestCacheablePreprocessors(unittest.TestCase):
    def test_cacheable_flags(self):
        for name in sorted(DETERMINISTIC_LEGACY_PREPROCESSORS) + ["canny", "blur_gaussian", "scribble_xdog"]:
            with self.subTest(name=name):
                self.assertTrue(Preprocessor.get_preprocessor(name).cacheable)
        # Random (shuffle), reads other settings (depth_leres++), a hit would be
        # slower than the work (none, invert), unaudited or tensor results.
        for name in ("shuffle", "depth_leres++", "none", "invert", "depth_hand_refiner", "clip_vision",
                     "ip-adapter_face_id", "reference_only", "tile_resample", "inpaint_only", "softedge_teed"):
            with self.subTest(name=name):
                self.assertFalse(Preprocessor.get_preprocessor(name).cacheable)


class TestLegacyResultCache(unittest.TestCase):
    def setUp(self):
        self.assertGreater(CACHE_SIZE, 0)
        self.preprocessor = Preprocessor.get_preprocessor("depth_zoe")
        self.preprocessor.clear_cache("test")
        self.calls = []

        def depth(img, res=512, **kwargs):
            self.calls.append(res)
            return (255 - img[:, :, 0]).copy(), True

        patcher = mock.patch.object(self.preprocessor, "call_function", depth)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.preprocessor.clear_cache, "test")
        self.image = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)

    def call(self, image, resolution=512, **kwargs):
        return self.preprocessor.cached_call(image, resolution=resolution, slider_1=0.0, slider_2=0.0, model="m", **kwargs).value

    def test_repeat_is_served_from_cache_as_an_independent_copy(self):
        first = self.call(self.image)
        first[0, 0, 0] = 7  # callers may mutate their result
        second = self.call(self.image.copy())
        self.assertEqual(self.calls, [512])
        np.testing.assert_array_equal(second, np.repeat((255 - self.image[:, :, :1]), 3, axis=2))
        self.assertIsNot(first, second)

    def test_input_and_parameters_are_part_of_the_key(self):
        self.call(self.image)
        changed = self.image.copy()
        changed[3, 3, 0] += 1
        self.call(changed)
        self.call(self.image, resolution=640)
        self.assertEqual(self.calls, [512, 512, 640])

    def test_the_units_controlnet_model_is_not_part_of_the_key(self):
        """Switching the ControlNet model (or the CLIP-on-CPU flag) with the same input reuses the depth map."""
        first = self.call(self.image)
        second = self.preprocessor.cached_call(self.image, resolution=512, slider_1=0.0, slider_2=0.0, model="other", low_vram=True).value
        self.assertEqual(self.calls, [512])
        np.testing.assert_array_equal(first, second)

    def test_calls_with_a_callback_always_run(self):
        seen = []
        for _ in range(2):
            self.call(self.image, json_pose_callback=seen.append)
        self.assertEqual(self.calls, [512, 512])


class TestMlsdNoLines(unittest.TestCase):
    def test_no_segment_is_an_empty_result_and_real_failures_raise(self):
        from annotator import mlsd
        from annotator.mlsd.utils import pred_lines

        image = np.zeros((64, 64, 3), dtype=np.uint8)
        # Flat heat map with zero displacement: every candidate is shorter than dist_thr.
        lines = pred_lines(image, lambda batch: torch.zeros(1, 9, 32, 32), [64, 64], 0.1, 0.1)
        self.assertEqual(lines.shape, (0, 4))

        class Failing(torch.nn.Module):
            def forward(self, batch):
                raise RuntimeError("CUDA out of memory")

        with mock.patch.object(mlsd, "mlsdmodel", Failing()):
            with self.assertRaisesRegex(RuntimeError, "out of memory"):
                mlsd.apply_mlsd(image, 0.1, 0.1)
        with mock.patch.object(mlsd, "mlsdmodel", torch.nn.Identity()), \
                mock.patch.object(mlsd, "pred_lines", return_value=np.zeros((0, 4))):
            np.testing.assert_array_equal(mlsd.apply_mlsd(image, 0.1, 0.1), np.zeros((64, 64), dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
