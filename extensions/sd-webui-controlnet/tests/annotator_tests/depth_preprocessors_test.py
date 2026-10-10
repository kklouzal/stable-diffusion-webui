import hashlib
import importlib
import os
import threading
import unittest
from unittest import mock

import numpy as np
import torch
import torch.nn.functional as F

utils = importlib.import_module("extensions.sd-webui-controlnet.tests.utils", "utils")

from annotator import util as annotator_util  # noqa: E402
from annotator.depth_anything import DepthAnythingDetector, DepthAnythingV1  # noqa: E402
from annotator.depth_anything_v2 import DepthAnythingV2Detector  # noqa: E402
from annotator.zoe import UNUSED_CHECKPOINT_KEYS, ZoeDetector  # noqa: E402
from annotator.zoe.zoedepth.models.zoedepth.zoedepth_v1 import ZoeDepth  # noqa: E402
from annotator.zoe.zoedepth.utils.config import get_config  # noqa: E402
from depth_anything_v2.dpt import DepthAnythingV2  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
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


class _StandIn(torch.nn.Module):
    """Randomly initialized parameters, one frozen; a persistent buffer; a non-persistent buffer computed here."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(6, 4)
        self.conv = torch.nn.Conv2d(3, 2, 1)
        self.scale = torch.nn.Parameter(torch.ones(4), requires_grad=False)
        self.register_buffer("table", torch.zeros(5))
        self.register_buffer("index", torch.arange(7) * 3, persistent=False)
        self.built_on_meta = [p.is_meta for p in self.parameters()]


def _stand_in_checkpoint():
    g = torch.Generator().manual_seed(0)
    return {
        "linear.weight": torch.randn(6, 4, generator=g).t(),  # other strides than the parameter's
        "linear.bias": torch.randn(4, generator=g).half(),  # another dtype than the parameter's
        "conv.weight": torch.randn(2, 3, 1, 1, generator=g),
        "conv.bias": torch.randn(2, generator=g),
        "scale": torch.randn(4, generator=g),
        "table": torch.randn(5, generator=g),
        "old.index": torch.arange(3),
    }


def _assert_same_state(test, model, reference):
    """Bit for bit the same tensors (dtype, shape, strides, values) and the same parameter requires_grad."""
    tensors = dict(model.state_dict(keep_vars=True), **dict(model.named_buffers()))
    expected = dict(reference.state_dict(keep_vars=True), **dict(reference.named_buffers()))
    test.assertEqual(tensors.keys(), expected.keys())
    for key, value in expected.items():
        got = tensors[key]
        test.assertFalse(got.is_meta, key)
        test.assertEqual((got.dtype, got.shape, got.stride(), got.requires_grad),
                         (value.dtype, value.shape, value.stride(), value.requires_grad), key)
        test.assertTrue(torch.equal(got, value), key)


class TestBuildWithStateDict(unittest.TestCase):
    """annotator.util.build_with_state_dict: the module build() + load_state_dict leaves, without random init."""

    def test_same_state_as_build_and_load_state_dict(self):
        reference = _StandIn()
        incompatible = reference.load_state_dict(_stand_in_checkpoint(), strict=False)
        self.assertEqual((incompatible.missing_keys, incompatible.unexpected_keys), ([], ["old.index"]))

        model = annotator_util.build_with_state_dict(_StandIn, _stand_in_checkpoint(), frozenset({"old.index"}))
        self.assertEqual(model.built_on_meta, [True] * 5)
        self.assertEqual(reference.built_on_meta, [False] * 5)
        _assert_same_state(self, model, reference)
        self.assertEqual(model.index.tolist(), list(range(0, 21, 3)))
        self.assertFalse(torch.nn.Linear(2, 2).weight.is_meta)

    def test_checkpoint_tensors_of_the_model_layout_are_used_as_they_are(self):
        checkpoint = _stand_in_checkpoint()
        checkpoint["linear.weight"] = checkpoint["linear.weight"].contiguous()
        checkpoint["linear.bias"] = checkpoint["linear.bias"].float()
        model = annotator_util.build_with_state_dict(_StandIn, checkpoint, frozenset({"old.index"}))
        for key in ("linear.weight", "linear.bias", "table"):
            self.assertEqual(model.state_dict()[key].data_ptr(), checkpoint[key].data_ptr(), key)

    def test_key_set_changes_fail_closed(self):
        unused = frozenset({"old.index"})
        cases = {
            "missing key": ({k: v for k, v in _stand_in_checkpoint().items() if k != "table"}, unused),
            "unexpected key": (dict(_stand_in_checkpoint(), extra=torch.zeros(1)), unused),
            "absent unused key": (_stand_in_checkpoint(), unused | {"old.other"}),
        }
        for name, (checkpoint, unused_keys) in cases.items():
            with self.subTest(name), self.assertRaisesRegex(RuntimeError, "Unsupported _StandIn checkpoint"):
                annotator_util.build_with_state_dict(_StandIn, checkpoint, unused_keys)
        with self.assertRaisesRegex(RuntimeError, "size mismatch"):
            annotator_util.build_with_state_dict(_StandIn, dict(_stand_in_checkpoint(), table=torch.zeros(6)), unused)
        self.assertFalse(torch.nn.Linear(2, 2).weight.is_meta)

    def test_shared_parameters_fail_closed(self):
        def build():
            model = _StandIn()
            model.tied = torch.nn.Linear(6, 4)
            model.tied.weight = model.linear.weight
            return model

        checkpoint = dict(_stand_in_checkpoint(), **{"tied.weight": torch.zeros(4, 6), "tied.bias": torch.zeros(4)})
        with self.assertRaisesRegex(RuntimeError, r"shares parameters.*\['tied.weight'\]"):
            annotator_util.build_with_state_dict(build, checkpoint, frozenset({"old.index"}))

    def test_parameters_other_threads_build_meanwhile_are_real(self):
        other = []

        def build():
            thread = threading.Thread(target=lambda: other.append(torch.nn.Linear(2, 2)))
            thread.start()
            thread.join()
            return _StandIn()

        model = annotator_util.build_with_state_dict(build, _stand_in_checkpoint(), frozenset({"old.index"}))
        self.assertEqual(model.built_on_meta, [True] * 5)
        self.assertFalse(other[0].weight.is_meta)

    def test_a_failing_build_restores_register_parameter(self):
        original = torch.nn.Module.register_parameter
        with self.assertRaisesRegex(ValueError, "boom"):
            annotator_util.build_with_state_dict(mock.Mock(side_effect=ValueError("boom")), {})
        self.assertIs(torch.nn.Module.register_parameter, original)
        self.assertFalse(torch.nn.Linear(2, 2).weight.is_meta)

    def test_zoe_unused_keys_are_its_non_persistent_relative_position_indices(self):
        model = annotator_util._build_with_meta_parameters(
            lambda: ZoeDepth.build_from_config(get_config("zoedepth", "infer")))
        self.assertTrue(all(p.is_meta for p in model.parameters()))
        buffers = dict(model.named_buffers())
        self.assertFalse(any(b.is_meta for b in buffers.values()))
        self.assertEqual(len(UNUSED_CHECKPOINT_KEYS), 24)
        self.assertEqual(UNUSED_CHECKPOINT_KEYS, {
            key for key in buffers.keys() - model.state_dict().keys() if key.endswith(".attn.relative_position_index")})


ZOE_WEIGHTS = os.path.join(ZoeDetector.model_dir, "ZoeD_M12_N.pt")
V1_WEIGHTS = os.path.join(DepthAnythingDetector.model_dir, "depth_anything_vitl14.pth")
V2_WEIGHTS = os.path.join(DepthAnythingV2Detector.model_dir, "depth_anything_v2_vitl.safetensors")


class TestDepthLoadersMatchRandomInitAndLoad(unittest.TestCase):
    """With the real checkpoints: each detector's model is bit for bit the randomly initialized model the loaders
    built before, loaded with load_state_dict."""

    @unittest.skipUnless(os.path.exists(ZOE_WEIGHTS), "ZoeDepth checkpoint not downloaded")
    def test_zoe(self):
        det = ZoeDetector()
        det.device = torch.device("cpu")
        det.load_model()
        reference = ZoeDepth.build_from_config(get_config("zoedepth", "infer"))
        incompatible = reference.load_state_dict(torch.load(ZOE_WEIGHTS, map_location="cpu")["model"], strict=False)
        self.assertEqual((incompatible.missing_keys, set(incompatible.unexpected_keys)), ([], UNUSED_CHECKPOINT_KEYS))
        _assert_same_state(self, det.model, reference.eval())
        self.assertFalse(det.model.training)

    @unittest.skipUnless(os.path.exists(V1_WEIGHTS), "Depth Anything checkpoint not downloaded")
    def test_depth_anything(self):
        model = DepthAnythingDetector(torch.device("cpu")).model
        reference = DepthAnythingV1()
        reference.load_state_dict(torch.load(V1_WEIGHTS), strict=True)
        _assert_same_state(self, model, reference.eval())
        self.assertFalse(model.training)

    @unittest.skipUnless(os.path.exists(V2_WEIGHTS), "Depth Anything V2 checkpoint not downloaded")
    def test_depth_anything_v2(self):
        model = DepthAnythingV2Detector(torch.device("cpu")).model
        reference = DepthAnythingV2(encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024])
        reference.load_state_dict(load_file(V2_WEIGHTS))
        _assert_same_state(self, model, reference.eval())
        self.assertFalse(model.training)
