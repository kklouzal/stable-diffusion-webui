import importlib
import os
import tempfile
import threading
import time
import unittest
from unittest import mock

import cv2
import numpy as np

utils = importlib.import_module("extensions.sd-webui-controlnet.tests.utils", "utils")

from annotator import openpose  # noqa: E402
from annotator.openpose import animalpose, wholebody  # noqa: E402
from scripts.supported_preprocessor import Preprocessor  # noqa: E402


class _OnnxStub:
    """Stands in for Wholebody/AnimalPose: counts constructions and detects overlapping calls."""

    def __init__(self, *paths):
        type(self).constructed += 1
        self.active = 0

    def __call__(self, image):
        self.active += 1
        type(self).max_active = max(type(self).max_active, self.active)
        time.sleep(0.02)
        self.active -= 1
        return None if type(self) is _DwStub else []

    @staticmethod
    def format_result(keypoints_info):
        return []


class _DwStub(_OnnxStub):
    constructed = 0
    max_active = 0


class _AnimalStub(_OnnxStub):
    constructed = 0
    max_active = 0


class TestOnnxPoseNetsLoadOnce(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        for name in ("yolox_l.onnx", "dw-ll_ucoco_384.onnx",
                     "rtmpose-m_simcc-ap10k_pt-aic-coco_210e-256x256-7a041aa1_20230206.onnx"):
            open(os.path.join(self.tmp.name, name), "wb").close()
        for stub in (_DwStub, _AnimalStub):
            stub.constructed = stub.max_active = 0
        self.detector = openpose.OpenposeDetector()
        self.detector.model_dir = self.tmp.name
        self.image = np.zeros((64, 48, 3), dtype=np.uint8)

    def _run(self, n_threads, **kwargs):
        threads = [threading.Thread(target=self.detector, args=(self.image,), kwargs=kwargs) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    def test_dw_nets_are_parsed_once_and_never_run_concurrently(self):
        with mock.patch.object(wholebody, "Wholebody", _DwStub):
            self._run(1, use_dw_pose=True)
            self._run(4, use_dw_pose=True)
        self.assertEqual(_DwStub.constructed, 1)
        self.assertEqual(_DwStub.max_active, 1)

    def test_animal_nets_are_parsed_once_and_never_run_concurrently(self):
        with mock.patch.object(animalpose, "AnimalPose", _AnimalStub):
            self._run(1, use_animal_pose=True)
            self._run(4, use_animal_pose=True)
        self.assertEqual(_AnimalStub.constructed, 1)
        self.assertEqual(_AnimalStub.max_active, 1)


class TestAnylineResolution(unittest.TestCase):
    def setUp(self):
        self.anyline = Preprocessor.get_preprocessor("softedge_anyline")
        self.lineart = Preprocessor.get_preprocessor("lineart_standard")

        class Mteed:
            # Deterministic stand-in for the MTEED network (weights are not available offline).
            def __call__(self, img, safe_steps=2):
                return np.ascontiguousarray(img[:, :, 1])

        patcher = mock.patch.object(self.anyline, "model", Mteed())
        patcher.start()
        self.addCleanup(patcher.stop)

    @staticmethod
    def drawing(h, w):
        img = np.full((h, w, 3), 255, dtype=np.uint8)
        for i in range(0, max(h, w), 37):
            cv2.line(img, (i, 0), (w - 1, i), (20, 40, 60), 2)
            cv2.circle(img, (i % w, (3 * i) % h), 3, (0, 0, 0), -1)
        return img

    def test_resolution_not_multiple_of_64(self):
        # Previously lineart_standard re-resized the padded image to `resolution` and combine_layers raised.
        for (h, w), res in (((1000, 1000), 1000), ((750, 1000), 1000), ((512, 683), 520)):
            with self.subTest(h=h, w=w, res=res):
                out = self.anyline(self.drawing(h, w), res, slider_1=2)
                k = res / min(h, w)
                self.assertEqual(out.shape, (int(np.round(h * k)), int(np.round(w * k)), 3))
                self.assertEqual(out.dtype, np.uint8)

    def test_lineart_call_is_unchanged_for_multiples_of_64(self):
        # Exactness of the fix: for a multiple-of-64 resolution the padded image's short side is the resolution,
        # so lineart_standard gets the same arguments as before.
        from scripts.utils import resize_image_with_pad

        for (h, w), res in (((1024, 1024), 1024), ((750, 1000), 1280), ((600, 900), 512)):
            with self.subTest(h=h, w=w, res=res):
                img, _ = resize_image_with_pad(self.drawing(h, w), res)
                self.assertEqual(min(img.shape[:2]), res)

    def test_lineart_at_own_scale_is_not_resampled(self):
        # cv2 returns a same-size INTER_AREA resize unchanged, so lineart_standard sees the padded pixels as-is.
        from scripts.preprocessor.legacy.processor import lineart_standard, resize_image_with_pad

        img, _ = resize_image_with_pad(self.drawing(1000, 1000), 1000)
        same, _ = resize_image_with_pad(img, min(img.shape[:2]))
        np.testing.assert_array_equal(same, img)
        self.assertEqual(lineart_standard(img, res=min(img.shape[:2]))[0].shape, img.shape[:2])


if __name__ == "__main__":
    unittest.main()
