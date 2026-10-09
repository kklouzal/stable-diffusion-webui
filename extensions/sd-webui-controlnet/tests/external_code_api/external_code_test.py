import unittest
import importlib

import numpy as np

utils = importlib.import_module('extensions.sd-webui-controlnet.tests.utils', 'utils')


from scripts import external_code
from scripts.enums import ResizeMode


class TestPixelPerfectResolution(unittest.TestCase):
    def test_outer_fit(self):
        image = np.zeros((100, 100, 3))
        target_H, target_W = 50, 100
        resize_mode = ResizeMode.OUTER_FIT
        result = external_code.pixel_perfect_resolution(image, target_H, target_W, resize_mode)
        expected = 50  # manually computed expected result
        self.assertEqual(result, expected)

    def test_inner_fit(self):
        image = np.zeros((100, 100, 3))
        target_H, target_W = 50, 100
        resize_mode = ResizeMode.INNER_FIT
        result = external_code.pixel_perfect_resolution(image, target_H, target_W, resize_mode)
        expected = 100  # manually computed expected result
        self.assertEqual(result, expected)


if __name__ == '__main__':
    unittest.main()