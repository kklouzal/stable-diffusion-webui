import importlib
utils = importlib.import_module('extensions.sd-webui-controlnet.tests.utils', 'utils')


from scripts.utils import get_unique_axis0

import unittest
import numpy as np

class TestUniqueFunctions(unittest.TestCase):
    def test_get_unique_axis0(self):
        data = np.random.randint(0, 100, size=(100000, 3))
        data = np.concatenate((data, data))
        numpy_unique_res = np.unique(data, axis=0)
        get_unique_axis0_res = get_unique_axis0(data)
        self.assertEqual(np.array_equal(
            np.sort(numpy_unique_res, axis=0), np.sort(get_unique_axis0_res, axis=0),
        ), True)

if __name__ == '__main__':
    unittest.main()