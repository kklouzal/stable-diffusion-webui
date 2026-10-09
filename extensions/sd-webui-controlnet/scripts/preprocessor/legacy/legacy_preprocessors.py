"""Registers the legacy table in preprocessor_compiled.py as Preprocessor objects. New preprocessors are written as
Preprocessor subclasses (see model_free_preprocessors.py) instead of table entries."""

from annotator.util import HWC3
from .preprocessor_compiled import legacy_preprocessors
from ...supported_preprocessor import Preprocessor, PreprocessorParameter


# Legacy preprocessors whose result is a pure function of the result-cache key
# (input image, resolution, sliders, ControlNet options, device/dtype): no RNG,
# no other global settings, weights from fixed files. Checked per annotator.
# Excluded on purpose: depth_leres++ (reads opts.depthmap_script_boost_rmax),
# identity-like and cheap ones (reference_*, tile_*, recolor_*, threshold, color:
# hashing the input costs about as much as the work), and the clip ones (tensor results).
DETERMINISTIC_LEGACY_PREPROCESSORS = frozenset((
    "depth", "depth_leres", "depth_zoe", "depth_anything", "depth_anything_v2",
    "normal_map", "normal_bae",
    "lineart", "lineart_coarse", "lineart_anime", "lineart_anime_denoise", "lineart_standard",
    "mlsd",
    "hed", "hed_safe", "scribble_hed", "pidinet", "pidinet_safe", "pidinet_sketch", "pidinet_scribble",
    "anime_face_segment",
    "openpose", "openpose_face", "openpose_faceonly", "openpose_full", "openpose_hand",
    "dw_openpose_full", "animal_openpose", "densepose", "densepose_parula",
))


class LegacyPreprocessor(Preprocessor):
    def __init__(self, name: str, legacy_dict):
        super().__init__(name)
        self.cacheable = name in DETERMINISTIC_LEGACY_PREPROCESSORS
        self._label = legacy_dict["label"]
        self.call_function = legacy_dict["call_function"]
        self.unload_function = legacy_dict["unload_function"]
        self.managed_model = legacy_dict["managed_model"]
        self.do_not_need_model = legacy_dict["model_free"]
        self.sorting_priority = legacy_dict["priority"]
        self.tags = legacy_dict["tags"]
        self.returns_image = legacy_dict.get("returns_image", True)
        self.accepts_mask = legacy_dict.get("accepts_mask", False)
        self.requires_mask = legacy_dict.get("requires_mask", False)
        # The table's "resolution" and "slider_3" data are not read: every legacy entry keeps the inherited
        # slider_resolution (64..2048, 512, step 8) and the hidden slider_3. module_detail and
        # ControlNetUnit.bound_check_params expose exactly those defaults.

        if legacy_dict["slider_1"] is None:
            self.slider_1 = PreprocessorParameter(visible=False)
        else:
            self.slider_1 = PreprocessorParameter(
                **legacy_dict["slider_1"], visible=True
            )

        if legacy_dict["slider_2"] is None:
            self.slider_2 = PreprocessorParameter(visible=False)
        else:
            self.slider_2 = PreprocessorParameter(
                **legacy_dict["slider_2"], visible=True
            )

        self.active_with_model = False

    def unload(self):
        if self.active_with_model:
            self.unload_function()
            self.active_with_model = False
            return True
        return False

    def __call__(
        self,
        input_image,
        resolution=512,
        slider_1=None,
        slider_2=None,
        slider_3=None,
        **kwargs
    ):
        # Legacy Preprocessors does not have slider 3
        del slider_3

        result, is_image = self.call_function(
            img=input_image, res=resolution, thr_a=slider_1, thr_b=slider_2, **kwargs
        )
        if self.managed_model is not None:
            self.active_with_model = True

        if is_image:
            result = HWC3(result)

        return result


unknown = DETERMINISTIC_LEGACY_PREPROCESSORS - legacy_preprocessors.keys()
assert not unknown, f"cacheable legacy preprocessors not registered: {sorted(unknown)}"

for name, data in legacy_preprocessors.items():
    p = LegacyPreprocessor(name, data)
    Preprocessor.add_supported_preprocessor(p)
