"""Preprocessor registry and legacy-wrapper contracts: the dropped preprocessors stay unregistered, the legacy
helpers keep their outputs, and the recolor/scribble_xdog/unload/transformer-id fixes hold. CPU only, no weights."""
import importlib
import unittest
from unittest import mock

import cv2
import numpy as np

utils = importlib.import_module("extensions.sd-webui-controlnet.tests.utils", "utils")

import scripts.preprocessor  # noqa: E402,F401  (registers every preprocessor)
from annotator.util import HWC3  # noqa: E402
from scripts.enums import StableDiffusionVersion, UnetBlockType  # noqa: E402
from scripts.ipadapter.presets import IPAdapterPreset, ipadapter_presets  # noqa: E402
from scripts.preprocessor.legacy import processor  # noqa: E402
from scripts.supported_preprocessor import Preprocessor  # noqa: E402

DROPPED = (
    "segmentation", "oneformer_ade20k", "oneformer_coco", "normal_dsine", "mediapipe_face", "depth_hand_refiner",
    "instant_id_face_embedding", "instant_id_face_keypoints", "ip-adapter_face_id", "ip-adapter_face_id_plus",
    "ip-adapter_pulid", "facexlib",
)


class TestRegistry(unittest.TestCase):
    def test_dropped_preprocessors_are_not_registered(self):
        for name in DROPPED:
            with self.subTest(name=name):
                self.assertIsNone(Preprocessor.get_preprocessor(name))
        self.assertNotIn("Instant-ID", Preprocessor.get_all_preprocessor_tags())

    def test_kept_preprocessors_are_registered(self):
        for name in ("none", "canny", "depth_zoe", "depth_anything", "depth_anything_v2", "anime_face_segment",
                     "mobile_sam", "ip-adapter-auto", "ip-adapter_clip_sd15", "inpaint", "inpaint_only"):
            with self.subTest(name=name):
                self.assertIsNotNone(Preprocessor.get_preprocessor(name))
        self.assertEqual(Preprocessor.get_preprocessor("inpaint").label, "inpaint_global_harmonious")
        self.assertEqual(Preprocessor.get_preprocessor("inpaint_only").label, "inpaint_only")
        self.assertEqual(
            [p.label for p in Preprocessor.get_filtered_preprocessors("Segmentation")],
            ["none", "seg_anime_face", "mobile_sam"],
        )
        self.assertEqual(Preprocessor.get_default_preprocessor("Segmentation").label, "seg_anime_face")

    def test_every_ip_adapter_preset_maps_to_a_registered_preprocessor(self):
        for preset in ipadapter_presets:
            with self.subTest(model=preset.model):
                self.assertIsNotNone(Preprocessor.get_preprocessor(preset.module))
        with self.assertRaisesRegex(AssertionError, "not found in ipadapter presets"):
            IPAdapterPreset.match_model("ip-adapter-faceid_sdxl [12345678]")

    def test_ip_adapter_auto_asserts_before_using_the_match(self):
        auto = Preprocessor.get_preprocessor("ip-adapter-auto")
        with mock.patch.object(type(auto), "get_preprocessor_by_model", staticmethod(lambda model: None)):
            with self.assertRaisesRegex(AssertionError, "no registered preprocessor"):
                auto(np.zeros((8, 8, 3), np.uint8), model="ip-adapter_sdxl")


def _old_resize_image_with_pad(input_image, resolution):
    """The former processor.py implementation (HWC3, then pad to a multiple of 64)."""
    img = HWC3(input_image)
    H_raw, W_raw, _ = img.shape
    k = float(resolution) / float(min(H_raw, W_raw))
    interpolation = cv2.INTER_CUBIC if k > 1 else cv2.INTER_AREA
    H_target = int(np.round(float(H_raw) * k))
    W_target = int(np.round(float(W_raw) * k))
    img = cv2.resize(img, (W_target, H_target), interpolation=interpolation)
    H_pad = int(np.ceil(float(H_target) / 64.0) * 64 - H_target)
    W_pad = int(np.ceil(float(W_target) / 64.0) * 64 - W_target)
    img_padded = np.pad(img, [[0, H_pad], [0, W_pad], [0, 0]], mode="edge")
    return img_padded, lambda x: x[:H_target, :W_target].copy()


class TestLegacyHelpers(unittest.TestCase):
    def test_resize_image_with_pad_matches_former_implementation(self):
        rng = np.random.default_rng(0)
        for shape in ((37, 53), (37, 53, 1), (37, 53, 3), (37, 53, 4)):
            img = rng.integers(0, 256, size=shape, dtype=np.uint8)
            for res in (24, 37, 100):
                with self.subTest(shape=shape, res=res):
                    new, new_remove = processor.resize_image_with_pad(img, res)
                    old, old_remove = _old_resize_image_with_pad(img, res)
                    np.testing.assert_array_equal(new, old)
                    self.assertTrue(new.flags["C_CONTIGUOUS"])
                    np.testing.assert_array_equal(new_remove(new), old_remove(old))

    def test_safe_and_filter_variants_pass_their_flags(self):
        image = np.zeros((64, 64, 3), np.uint8)
        expected = {
            "hed": ("model_hed", dict(is_safe=False)),
            "hed_safe": ("model_hed", dict(is_safe=True)),
            "pidinet": ("model_pidinet", dict(is_safe=False, apply_fliter=False)),
            "pidinet_safe": ("model_pidinet", dict(is_safe=True, apply_fliter=False)),
            "pidinet_sketch": ("model_pidinet", dict(is_safe=False, apply_fliter=True)),
        }
        for name, (global_name, kwargs) in expected.items():
            with self.subTest(name=name):
                model = mock.Mock(return_value=np.zeros((64, 64), np.uint8))
                with mock.patch.object(processor, global_name, model):
                    Preprocessor.get_preprocessor(name).call_function(img=image, res=64, thr_a=None, thr_b=None)
                self.assertEqual(model.call_args.kwargs, kwargs)


class TestRecolor(unittest.TestCase):
    def test_luminance_is_lab_l_of_the_rgb_input(self):
        red = np.zeros((4, 4, 3), np.uint8)
        red[..., 0] = 255
        result, is_image = processor.recolor_luminance(red, thr_a=1.0)
        self.assertTrue(is_image)
        self.assertEqual(int(result[0, 0, 0]), 136)  # L of pure red; reading it as BGR (pure blue) gives 82
        rng = np.random.default_rng(1)
        img = rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)
        result, _ = processor.recolor_luminance(img, thr_a=1.0)
        np.testing.assert_array_equal(result, np.repeat(cv2.cvtColor(img, cv2.COLOR_RGB2LAB)[..., :1], 3, axis=2))

    def test_intensity_is_the_max_channel(self):
        rng = np.random.default_rng(2)
        img = rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)
        result, _ = processor.recolor_intensity(img, thr_a=1.0)
        np.testing.assert_array_equal(result, np.repeat(img.max(axis=2, keepdims=True), 3, axis=2))


class TestScribbleXdog(unittest.TestCase):
    def test_strong_edges_survive_the_threshold(self):
        # Vertical lines of every grey level on white: some line centres reach an edge strength in [128, 143], where
        # the former uint8 2 * (255 - dog) wrapped below the default threshold of 32 and dropped the edge.
        img = np.full((64, 512, 3), 255, np.uint8)
        for i, x in enumerate(range(4, 508, 4)):
            img[:, x] = 2 * i
        g1 = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 0.5)
        g2 = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 5.0)
        strength = (255 - (255 - np.min(g2 - g1, axis=2)).clip(0, 255).astype(np.uint8)).astype(np.int64)
        wrapping = (strength >= 128) & (strength < 144)
        self.assertTrue(wrapping.any())
        result = Preprocessor.get_preprocessor("scribble_xdog")(img, 64, slider_1=32)
        np.testing.assert_array_equal(result[..., 0] == 255, 2 * strength > 32)


class TestUnloadMetadata(unittest.TestCase):
    def test_revision_releases_the_clip_g_encoder(self):
        for name in ("revision_clipvision", "revision_ignore_prompt"):
            with self.subTest(name=name):
                p = Preprocessor.get_preprocessor(name)
                encoder = mock.Mock()
                with mock.patch.dict(processor.clip_encoder, {"clip_g": encoder}), \
                        mock.patch.object(p, "call_function", return_value=({"image_embeds": None}, False)):
                    p(np.zeros((8, 8, 3), np.uint8), slider_1=0.0)
                    self.assertTrue(p.unload())
                    encoder.unload_model.assert_called_once_with()
                    self.assertIsNone(processor.clip_encoder["clip_g"])
                self.assertFalse(p.unload())

    def test_pidinet_scribble_releases_pidinet(self):
        import annotator.pidinet as pidinet_annotator

        p = Preprocessor.get_preprocessor("pidinet_scribble")
        with mock.patch.object(processor, "model_pidinet", object()), \
                mock.patch.object(pidinet_annotator, "unload_pid_model") as unload_pid_model, \
                mock.patch.object(p, "call_function", return_value=(np.zeros((8, 8), np.uint8), True)):
            p(np.zeros((8, 8, 3), np.uint8))
            self.assertTrue(p.unload())
            unload_pid_model.assert_called_once_with()


class TestTransformerIds(unittest.TestCase):
    def test_sd1x_ids_are_filed_by_block_type_in_call_order(self):
        ids = StableDiffusionVersion.SD1x.transformer_ids
        self.assertEqual({t.block_type for t in ids.input_ids}, {UnetBlockType.INPUT})
        self.assertEqual({t.block_type for t in ids.output_ids}, {UnetBlockType.OUTPUT})
        self.assertEqual({t.block_type for t in ids.middle_ids}, {UnetBlockType.MIDDLE})
        self.assertEqual((len(ids.input_ids), len(ids.output_ids), len(ids.middle_ids)), (6, 9, 1))
        # plugable_ipadapter enumerates chain(input, output, middle): the same order as before the fix.
        order = [t.transformer_index for t in (*ids.input_ids, *ids.output_ids, *ids.middle_ids)]
        self.assertEqual(order, [0, 1, 2, 3, 4, 5, *range(7, 16), 6])
        self.assertEqual([t.transformer_index for t in ids.to_list()], list(range(16)))


if __name__ == "__main__":
    unittest.main()
