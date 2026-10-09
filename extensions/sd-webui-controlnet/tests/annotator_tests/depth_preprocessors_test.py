import hashlib
import importlib
import unittest
from unittest import mock

import numpy as np
import torch
import torch.nn.functional as F

utils = importlib.import_module("extensions.sd-webui-controlnet.tests.utils", "utils")

from annotator.depth_anything import DepthAnythingDetector, DepthAnythingV1  # noqa: E402
from annotator.depth_anything_v2 import DepthAnythingV2Detector  # noqa: E402
from annotator.zoe import ZoeDetector  # noqa: E402
from scripts import utils as cn_utils  # noqa: E402
from scripts.preprocessor.legacy import processor  # noqa: E402

# sha256 of repr(sorted((key, shape))) of depth_anything_vitl14.pth (the v1 checkpoint), measured on the real file.
V1_CHECKPOINT_MANIFEST = "2a4168920991e6d0830392f835535a0da25e41fbc8c1cf5ffc4f4753ab87fe5b"


class _RecordingNet(torch.nn.Module):
    """Stands in for the depth network: records its input and returns a fixed relative-depth map at its size."""

    def __init__(self):
        super().__init__()
        self.inputs = []

    def forward(self, x):
        self.inputs.append(x)
        h, w = x.shape[-2:]
        yy, xx = torch.meshgrid(torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32), indexing="ij")
        return (torch.sin(yy / 7.0) + 0.37 * xx / w)[None]


def _detector(cls):
    det = cls.__new__(cls)
    det.device = torch.device("cpu")
    det.model = _RecordingNet()
    return det


class TestDepthAnythingDetectors(unittest.TestCase):
    def test_network_input_is_rgb(self):
        red = np.zeros((70, 90, 3), np.uint8)
        red[..., 0] = 255
        for cls in (DepthAnythingDetector, DepthAnythingV2Detector):
            with self.subTest(cls=cls.__name__):
                det = _detector(cls)
                det(red, colored=False)
                x = det.model.inputs[0][0]
                # NormalizeImage: (v - mean) / std with ImageNet RGB statistics.
                torch.testing.assert_close(x[0].mean(), torch.tensor((1.0 - 0.485) / 0.229))
                torch.testing.assert_close(x[2].mean(), torch.tensor((0.0 - 0.406) / 0.225))

    def test_depth_is_rounded_to_uint8(self):
        img = np.random.default_rng(0).integers(0, 256, (61, 83, 3), dtype=np.uint8)
        for cls in (DepthAnythingDetector, DepthAnythingV2Detector):
            with self.subTest(cls=cls.__name__):
                det = _detector(cls)
                out = det(img, colored=False)
                with torch.no_grad():
                    depth = det.model(det.model.inputs[0])
                depth = F.interpolate(depth[None], (61, 83), mode="bilinear", align_corners=False)[0, 0]
                expected = ((depth - depth.min()) / (depth.max() - depth.min()) * 255.0).numpy()
                np.testing.assert_array_equal(out, np.rint(expected).astype(np.uint8))
                self.assertFalse(np.array_equal(out, expected.astype(np.uint8)))

    def test_v1_backbone_is_built_locally_and_matches_the_checkpoint_layout(self):
        with mock.patch.object(torch.hub, "load", side_effect=AssertionError("torch.hub used")):
            net = DepthAnythingV1()
        manifest = sorted((k, tuple(v.shape)) for k, v in net.state_dict().items())
        self.assertEqual(hashlib.sha256(repr(manifest).encode()).hexdigest(), V1_CHECKPOINT_MANIFEST)


class TestZoeDetector(unittest.TestCase):
    def test_depth_is_rounded_to_uint8(self):
        depth = torch.linspace(0.5, 9.5, 37 * 53, dtype=torch.float32).reshape(1, 1, 37, 53)
        det = ZoeDetector.__new__(ZoeDetector)
        det.device = torch.device("cpu")
        det.model = mock.Mock(infer=mock.Mock(return_value=depth))
        out = det(np.zeros((37, 53, 3), np.uint8))
        d = depth[0, 0].numpy().copy()
        vmin, vmax = np.percentile(d, np.array([2, 85], dtype=d.dtype))
        expected = ((1.0 - (d - vmin) / (vmax - vmin)) * 255.0).clip(0, 255)
        np.testing.assert_array_equal(out, np.rint(expected).astype(np.uint8))
        self.assertFalse(np.array_equal(out, expected.astype(np.uint8)))


class TestDepthPreprocessorsSkipPadding(unittest.TestCase):
    """The depth networks run on the resized image itself: edge padding to a multiple of 64 would enter the output
    normalization (min/max, or ZoeDepth's 2nd/85th percentiles)."""

    def test_network_input_is_the_resized_image_without_padding(self):
        img = np.random.default_rng(1).integers(0, 256, (200, 140, 3), dtype=np.uint8)
        resized = cn_utils.resize_image_short_side(img, 512)
        self.assertEqual(resized.shape, (731, 512, 3))
        padded, remove_pad = cn_utils.resize_image_with_pad(img, 512)
        np.testing.assert_array_equal(remove_pad(padded), resized)

        for name, attr in (("depth_anything", "model_depth_anything"),
                           ("depth_anything_v2", "model_depth_anything_v2"),
                           ("zoe_depth", "model_zoe_depth")):
            with self.subTest(name=name):
                seen = []

                def model(x, seen=seen, **kwargs):
                    seen.append(x)
                    return np.full(x.shape[:2], 7, np.uint8)

                with mock.patch.object(processor, attr, model):
                    out, is_image = getattr(processor, name)(img, 512)
                np.testing.assert_array_equal(seen[0], resized)
                self.assertEqual(out.shape, (731, 512))
                self.assertTrue(is_image)
