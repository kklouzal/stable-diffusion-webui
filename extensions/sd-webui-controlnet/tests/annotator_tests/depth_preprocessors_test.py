import hashlib
import importlib
import unittest
from unittest import mock

import torch

utils = importlib.import_module("extensions.sd-webui-controlnet.tests.utils", "utils")

from annotator.depth_anything import DepthAnythingV1  # noqa: E402

# sha256 of repr(sorted((key, shape))) of depth_anything_vitl14.pth (the v1 checkpoint), measured on the real file.
V1_CHECKPOINT_MANIFEST = "2a4168920991e6d0830392f835535a0da25e41fbc8c1cf5ffc4f4753ab87fe5b"


class TestDepthAnythingV1Backbone(unittest.TestCase):
    def test_v1_backbone_is_built_locally_and_matches_the_checkpoint_layout(self):
        with mock.patch.object(torch.hub, "load", side_effect=AssertionError("torch.hub used")):
            net = DepthAnythingV1()
        manifest = sorted((k, tuple(v.shape)) for k, v in net.state_dict().items())
        self.assertEqual(hashlib.sha256(repr(manifest).encode()).hexdigest(), V1_CHECKPOINT_MANIFEST)
