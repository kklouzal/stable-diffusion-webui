import functools
import math
import tracemalloc
import os
import logging
from copy import copy
from typing import Any, List, NamedTuple, Optional, Tuple
import modules.scripts as scripts
from internal_controlnet.cache_contract import AtomicLRU, callable_identity, controlnet_option_snapshot, runtime_identity
from modules import shared, devices, script_callbacks, processing, masking, images
import gradio as gr
import time

from einops import rearrange

# Register all preprocessors.
import scripts.preprocessor as preprocessor_init  # noqa
from annotator.util import HWC3
from internal_controlnet.external_code import ControlNetUnit
from scripts import global_state, hook, external_code, controlnet_version, utils
from scripts.controlnet_lora import bind_control_lora, unbind_control_lora
from scripts.controlnet_lllite import clear_all_lllite
from scripts.ipadapter.plugable_ipadapter import clear_all_ip_adapter
from scripts.utils import load_state_dict, get_unique_axis0, align_dim_latent
from scripts.hook import ControlParams, UnetHook, HackedImageRNG
from scripts.enums import (
    ControlModelType,
    InputMode,
    StableDiffusionVersion,
    HiResFixOption,
    ControlMode,
    ResizeMode,
)
from scripts.logging import logger
from scripts.supported_preprocessor import Preprocessor
from modules.processing import StableDiffusionProcessingImg2Img, StableDiffusionProcessingTxt2Img, StableDiffusionProcessing
from modules.images import save_image
from scripts.infotext import Infotext

import cv2
import numpy as np
import torch

from PIL import Image
from scripts.lvminthin import lvmin_thin, nake_nms
from scripts.controlnet_model_guess import build_model_by_guess, ControlModel
from scripts.hook import restore_secondary_hijacks


def clear_all_secondary_control_models(m):
    restore_secondary_hijacks(m)
    clear_all_lllite()
    clear_all_ip_adapter()


def find_closest_lora_model_name(search: str):
    if not search:
        return None
    if search in global_state.cn_models:
        return search
    search = search.lower()
    if search in global_state.cn_models_names:
        return global_state.cn_models_names.get(search)
    applicable = [name for name in global_state.cn_models_names.keys()
                  if search in name.lower()]
    if not applicable:
        return None
    applicable = sorted(applicable, key=lambda name: len(name))
    return global_state.cn_models_names[applicable[0]]


global_state.update_cn_models()
logger.info(f"ControlNet {controlnet_version.version_flag}")


def prepare_mask(
    mask: Image.Image, p: processing.StableDiffusionProcessingImg2Img
) -> Image.Image:
    """
    The img2img inpaint mask exactly as StableDiffusionProcessingImg2Img.init prepares the core's
    (processing.prepare_image_mask): binary, stretched onto the init image `p.init_images[0]` when its size differs,
    inverted when `p.inpainting_mask_invert`, then blurred by `p.mask_blur_x`/`p.mask_blur_y`. The result has the
    init image's size, so a crop region computed on it is in the core's (init image) coordinates.

    Args:
        mask (Image.Image): The input mask as a PIL Image object.
        p: The img2img processing object (the only kind that carries an inpaint mask).

    Returns:
        mask (Image.Image): The prepared mask, mode "L", of the init image's size.
    """
    return processing.prepare_image_mask(
        mask, p.init_images[0].size, mask_round=p.mask_round, invert=p.inpainting_mask_invert,
        blur_x=p.mask_blur_x, blur_y=p.mask_blur_y)


def set_numpy_seed(p: processing.StableDiffusionProcessing) -> Optional[int]:
    """
    Set the random seed for NumPy based on the provided parameters.

    Args:
        p (processing.StableDiffusionProcessing): The instance of the StableDiffusionProcessing class.

    Returns:
        Optional[int]: The computed random seed if successful, or None if an exception occurs.

    This function sets the random seed for NumPy using the seed and subseed values from the given instance of
    StableDiffusionProcessing. If either seed or subseed is -1, it uses the first value from `all_seeds`.
    Otherwise, it takes the maximum of the provided seed value and 0.

    The final random seed is computed by adding the seed and subseed values, applying a bitwise AND operation
    with 0xFFFFFFFF to ensure it fits within a 32-bit integer.
    """
    try:
        tmp_seed = int(p.all_seeds[0] if p.seed == -1 else max(int(p.seed), 0))
        tmp_subseed = int(p.all_seeds[0] if p.subseed == -1 else max(int(p.subseed), 0))
        seed = (tmp_seed + tmp_subseed) & 0xFFFFFFFF
        np.random.seed(seed)
        return seed
    except Exception as e:
        logger.warning(e)
        logger.warning('Warning: Failed to use consistent random seed.')
        return None


# v / 255 for every uint8 v, divided on the CPU in float32 exactly like the
# former per-pixel path. (On CUDA a division by a Python scalar multiplies by
# the reciprocal, which can differ in the last bit, so it is not used.)
_UINT8_TO_UNIT = torch.arange(256, dtype=torch.float32) / 255.0


def get_pytorch_control(x: np.ndarray) -> torch.Tensor:
    """HWC array -> 1CHW float32 in [0, 1] on the ControlNet device. uint8 maps
    upload as uint8 (a quarter of the float bytes) and are expanded on the device
    by exact table lookup; the result's values and strides equal the former
    convert-on-CPU path."""
    device = devices.get_device_for("controlnet")
    if x.dtype == np.uint8:
        y = _UINT8_TO_UNIT.to(device)[torch.from_numpy(x).to(device).long()]
    else:
        y = (torch.from_numpy(x).float() / 255.0).to(device)
    return rearrange(y, 'h w c -> 1 c h w')


def get_control(
    p: StableDiffusionProcessing,
    unit: ControlNetUnit,
    control_model_type: ControlModelType,
    preprocessor: Preprocessor,
):
    """Get input for a ControlNet unit."""
    high_res_fix = isinstance(p, StableDiffusionProcessingTxt2Img) and getattr(p, 'enable_hr', False)
    h, w, hr_y, hr_x = Script.get_target_dimensions(p)
    input_image, resize_mode = Script.choose_input_image(p, unit)
    if isinstance(input_image, list):
        assert unit.accepts_multiple_inputs
        input_images = input_image
    else: # Following operations are only for single input image.
        input_image = Script.try_crop_image_with_a1111_mask(p, unit, input_image, resize_mode)
        input_image = input_image.copy()  # C-contiguous copy
        if unit.module == 'inpaint_only+lama' and resize_mode == ResizeMode.OUTER_FIT:
            # inpaint_only+lama is special and required outpaint fix
            _, input_image = Script.detectmap_proc(input_image, unit.module, resize_mode, hr_y, hr_x)
        input_images = [input_image]

    if unit.pixel_perfect:
        unit.processor_res = external_code.pixel_perfect_resolution(
            input_images[0],
            target_H=h,
            target_W=w,
            resize_mode=resize_mode,
        )
    # Preprocessor result may depend on numpy random operations, use the
    # random seed in `StableDiffusionProcessing` to make the
    # preprocessor result reproducable.
    # Currently following preprocessors use numpy random:
    # - shuffle
    seed = set_numpy_seed(p)
    logger.debug(f"Use numpy seed {seed}.")
    logger.info(f"Using preprocessor: {unit.module}")
    logger.info(f'preprocessor resolution = {unit.processor_res}')

    detected_maps = []
    def store_detected_map(detected_map, module: str) -> None:
        if unit.save_detected_map:
            detected_maps.append((detected_map, module))

    def preprocess_input_image(input_image: np.ndarray):
        """ Preprocess single input image. """
        result = preprocessor.cached_call(
            input_image,
            resolution=unit.processor_res,
            slider_1=unit.threshold_a,
            slider_2=unit.threshold_b,
            low_vram=(
                "clip" in unit.module and
                shared.opts.data.get("controlnet_clip_detector_on_cpu", False)
            ),
            model=unit.model,
        )
        detected_map = result.value
        is_image = preprocessor.returns_image
        # TODO: Refactor img control detection logic.
        if high_res_fix:
            if is_image:
                hr_control, hr_detected_map = Script.detectmap_proc(detected_map, unit.module, resize_mode, hr_y, hr_x)
                store_detected_map(hr_detected_map, unit.module)
            else:
                hr_control = detected_map
        else:
            hr_control = None

        if is_image:
            control, detected_map = Script.detectmap_proc(detected_map, unit.module, resize_mode, h, w)
            store_detected_map(detected_map, unit.module)
        else:
            control = detected_map
            for image in result.display_images:
                store_detected_map(image, unit.module)

        if control_model_type == ControlModelType.T2I_StyleAdapter:
            control = control['last_hidden_state']

        if control_model_type == ControlModelType.ReVision:
            control = control['image_embeds']

        return control, hr_control

    controls, hr_controls = list(zip(*[preprocess_input_image(img) for img in input_images]))
    assert len(controls) == len(hr_controls)
    return controls, hr_controls, detected_maps


class BuiltControlModel(NamedTuple):
    """A model_load_cache entry: the built model and, for a 'difference' model, the checkpoint revision
    (Script._checkpoint_revision) whose UNet weights its build added; None for models independent of it."""
    control_model: ControlModel
    base_revision: Any


class Script(scripts.Script, metaclass=(
    utils.TimeMeta if logger.level == logging.DEBUG else type)):

    model_load_cache = AtomicLRU(2, "controlnet-model")

    def __init__(self) -> None:
        super().__init__()
        self.latest_network = None
        self.enabled_units: List[ControlNetUnit] = []
        self.detected_map = []
        self.post_processors = []
        self.noise_modifier = None

    def title(self):
        return "ControlNet"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        """API-only: one inert State per unit (`control_net_unit_count`). Each value is the default ControlNetUnit
        that the API's default script args and /sdapi/v1/script-info report; no browser UI is built."""
        return tuple(gr.State(ControlNetUnit()) for _ in range(shared.opts.data.get("control_net_unit_count", 3)))

    @staticmethod
    def _resolve_model_path(model):
        model_path = global_state.cn_models.get(model, None)
        resolved_model = model
        if model_path is None:
            resolved_model = find_closest_lora_model_name(model)
            model_path = global_state.cn_models.get(resolved_model, None)
        if model_path is None:
            raise RuntimeError(f"model not found: {model}")
        model_path = model_path.strip('"')
        if not os.path.exists(model_path):
            raise ValueError(f"file not found: {model_path}")
        return resolved_model, os.path.realpath(model_path)

    @staticmethod
    def _model_cache_key(p, unet, model):
        resolved_model, model_path = Script._resolve_model_path(model)
        stat = os.stat(model_path)
        source_revision = (
            model_path, stat.st_dev, stat.st_ino, stat.st_size,
            stat.st_mtime_ns, stat.st_ctime_ns,
        )
        sd_model = p.sd_model
        # The loaded checkpoint is not part of the key: only 'difference' models depend on it, and
        # load_control_model rebuilds those when it changes (see _checkpoint_revision).
        base_revision = (type(unet).__module__, type(unet).__qualname__, id(unet))
        return (
            "controlnet-model", 1, resolved_model, source_revision, base_revision,
            str(sd_model.dtype), str(getattr(devices, "dtype_unet", None)),
            str(getattr(devices, "device", None)), runtime_identity(torch),
            callable_identity(build_model_by_guess), controlnet_option_snapshot(shared.opts.data),
        )

    @staticmethod
    def _return_cached_model(control_model):
        if control_model.type == ControlModelType.Controlllite:
            return None
        if not control_model.type.allow_context_sharing:
            return ControlModel(copy(control_model.model), control_model.type)
        return control_model

    @staticmethod
    def _checkpoint_revision(sd_model):
        """Identity of the checkpoint whose weights the UNet holds: its sha256, and its file (path, size, mtime),
        since checkpoint switches load in place into the same UNet object and --no-hashing leaves sha256 None."""
        checkpoint_info = getattr(sd_model, "sd_checkpoint_info", None)
        checkpoint_file = getattr(checkpoint_info, "filename", None)
        file_revision = None
        if checkpoint_file and os.path.isfile(checkpoint_file):
            checkpoint_stat = os.stat(checkpoint_file)
            file_revision = (os.path.realpath(checkpoint_file), checkpoint_stat.st_size, checkpoint_stat.st_mtime_ns)
        return getattr(checkpoint_info, "sha256", None), file_revision

    @staticmethod
    def load_control_model(p, unet, model) -> ControlModel:
        max_size = shared.opts.data.get("control_net_model_cache_size", 2)
        Script.model_load_cache.set_max_size(max_size)
        key = Script._model_cache_key(p, unet, model)
        checkpoint_revision = Script._checkpoint_revision(p.sd_model)

        def build():
            return Script.build_control_model(p, unet, model, checkpoint_revision)

        built = Script.model_load_cache.get_or_compute(key, build)
        if built.base_revision is not None and built.base_revision != checkpoint_revision:
            # A 'difference' model holds the UNet weights of the checkpoint it was built against.
            Script.model_load_cache.discard(key, "checkpoint-changed")
            built = Script.model_load_cache.get_or_compute(key, build)
        control_model = built.control_model
        if control_model.type == ControlModelType.Controlllite:
            # Mutable per-unit context is not equivalent across requests.
            Script.model_load_cache.discard(key, "volatile-controlllite")
            return control_model
        cached = Script._return_cached_model(control_model)
        logger.info(f"ControlNet model cache lookup: {Script.model_load_cache.info()['last_lookup']['reason']}")
        return cached

    @staticmethod
    def build_control_model(p, unet, model, checkpoint_revision=None) -> "BuiltControlModel":
        if model is None or model == 'None':
            raise RuntimeError("You have not selected any ControlNet Model.")

        model, model_path = Script._resolve_model_path(model)

        logger.info(f"Loading model: {model}")
        state_dict = load_state_dict(model_path)
        # build_model_by_guess adds the UNet's current weights to a 'difference' model's deltas.
        depends_on_checkpoint = 'difference' in state_dict and unet is not None
        control_model = build_model_by_guess(state_dict, unet, model_path)
        control_model.model.to('cpu', dtype=p.sd_model.dtype)
        logger.info(f"ControlNet model {model}({control_model.type}) loaded.")
        return BuiltControlModel(control_model, checkpoint_revision if depends_on_checkpoint else None)

    @staticmethod
    def normalize_remote_resize_mode(value, default):
        if isinstance(value, ResizeMode):
            return value
        aliases = {
            0: ResizeMode.RESIZE,
            1: ResizeMode.INNER_FIT,
            2: ResizeMode.OUTER_FIT,
            "0": ResizeMode.RESIZE,
            "1": ResizeMode.INNER_FIT,
            "2": ResizeMode.OUTER_FIT,
            "Just resize": ResizeMode.RESIZE,
            "Just Resize": ResizeMode.RESIZE,
            "Crop and resize": ResizeMode.INNER_FIT,
            "Crop and Resize": ResizeMode.INNER_FIT,
            "Resize and fill": ResizeMode.OUTER_FIT,
            "Resize and Fill": ResizeMode.OUTER_FIT,
        }
        if value in aliases:
            return aliases[value]
        try:
            return ResizeMode(value)
        except Exception:
            return default

    @staticmethod
    def normalize_remote_control_mode(value, default):
        if isinstance(value, ControlMode):
            return value
        aliases = {
            0: ControlMode.BALANCED,
            1: ControlMode.PROMPT,
            2: ControlMode.CONTROL,
            "0": ControlMode.BALANCED,
            "1": ControlMode.PROMPT,
            "2": ControlMode.CONTROL,
            "Balanced": ControlMode.BALANCED,
            "My prompt is more important": ControlMode.PROMPT,
            "ControlNet is more important": ControlMode.CONTROL,
        }
        if value in aliases:
            return aliases[value]
        try:
            return ControlMode(value)
        except Exception:
            return default

    @staticmethod
    def normalize_guidance_interval(start, end):
        def as_float(value, default):
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        start = min(1.0, max(0.0, as_float(start, 0.0)))
        end = min(1.0, max(0.0, as_float(end, 1.0)))
        if end < start:
            end = start
        return start, end

    @staticmethod
    def get_remote_call(p, attribute, default=None, idx=0, strict=False):
        if not shared.opts.data.get("control_net_allow_script_control", False):
            return default

        def get_element(obj, strict=False):
            if not isinstance(obj, list):
                return obj if not strict or idx == 0 else None
            elif idx < len(obj):
                return obj[idx]
            else:
                return None

        if idx > 0:
            indexed_attribute_value = getattr(p, f"{attribute}{idx + 1}", None)
            if indexed_attribute_value is not None:
                return indexed_attribute_value

        attribute_value = get_element(getattr(p, attribute, None), strict)
        return attribute_value if attribute_value is not None else default

    @staticmethod
    def parse_remote_call(p, unit: ControlNetUnit, idx):
        selector = Script.get_remote_call

        unit.enabled = selector(p, "control_net_enabled", unit.enabled, idx, strict=True)
        unit.module = selector(p, "control_net_module", unit.module, idx)
        unit.model = selector(p, "control_net_model", unit.model, idx)
        unit.weight = selector(p, "control_net_weight", unit.weight, idx)
        unit.image = selector(p, "control_net_image", unit.image, idx)
        unit.resize_mode = Script.normalize_remote_resize_mode(selector(p, "control_net_resize_mode", unit.resize_mode, idx), unit.resize_mode)
        unit.low_vram = selector(p, "control_net_lowvram", unit.low_vram, idx)
        unit.processor_res = selector(p, "control_net_pres", unit.processor_res, idx)
        unit.threshold_a = selector(p, "control_net_pthr_a", unit.threshold_a, idx)
        unit.threshold_b = selector(p, "control_net_pthr_b", unit.threshold_b, idx)
        guidance_start = selector(p, "control_net_guidance_start", unit.guidance_start, idx)
        guidance_end = selector(p, "control_net_guidance_end", unit.guidance_end, idx)
        # The API maps the legacy alias control_net_guidance_strength to control_net_guidance_end.
        unit.guidance_start, unit.guidance_end = Script.normalize_guidance_interval(guidance_start, guidance_end)
        unit.control_mode = Script.normalize_remote_control_mode(selector(p, "control_net_control_mode", unit.control_mode, idx), unit.control_mode)
        unit.pixel_perfect = selector(p, "control_net_pixel_perfect", unit.pixel_perfect, idx)

        return unit

    @staticmethod
    def detectmap_proc(detected_map, module, resize_mode, h, w):

        if 'inpaint' in module:
            detected_map = detected_map.astype(np.float32)
        else:
            detected_map = HWC3(detected_map)

        def high_quality_resize(x, size):
            # Written by lvmin
            # Super high-quality control map up-scaling, considering binary, seg, and one-pixel edges

            inpaint_mask = None
            if x.ndim == 3 and x.shape[2] == 4:
                inpaint_mask = x[:, :, 3]
                x = x[:, :, 0:3]

            if x.shape[0] != size[1] or x.shape[1] != size[0]:
                new_size_is_smaller = (size[0] * size[1]) < (x.shape[0] * x.shape[1])
                new_size_is_bigger = (size[0] * size[1]) > (x.shape[0] * x.shape[1])
                unique_color_count = len(get_unique_axis0(x.reshape(-1, x.shape[2])))
                is_one_pixel_edge = False
                is_binary = False
                if unique_color_count == 2:
                    is_binary = np.min(x) < 16 and np.max(x) > 240
                    if is_binary:
                        xc = x
                        xc = cv2.erode(xc, np.ones(shape=(3, 3), dtype=np.uint8), iterations=1)
                        xc = cv2.dilate(xc, np.ones(shape=(3, 3), dtype=np.uint8), iterations=1)
                        one_pixel_edge_count = np.where(xc < x)[0].shape[0]
                        all_edge_count = np.where(x > 127)[0].shape[0]
                        is_one_pixel_edge = one_pixel_edge_count * 2 > all_edge_count

                # Few colors means a segmentation/color-coded map: keep the labels exact. A gray map (all channels
                # equal) is an intensity map instead, e.g. a smooth low-contrast depth map with under 200 levels,
                # which nearest-neighbour scaling would turn into visible steps.
                is_gray = bool((x[:, :, 0] == x[:, :, 1]).all() and (x[:, :, 0] == x[:, :, 2]).all())
                if 2 < unique_color_count < 200 and not is_gray:
                    interpolation = cv2.INTER_NEAREST
                elif new_size_is_smaller:
                    interpolation = cv2.INTER_AREA
                else:
                    interpolation = cv2.INTER_CUBIC  # Must be CUBIC because we now use nms. NEVER CHANGE THIS

                y = cv2.resize(x, size, interpolation=interpolation)
                if inpaint_mask is not None:
                    inpaint_mask = cv2.resize(inpaint_mask, size, interpolation=interpolation)

                if is_binary:
                    y = np.mean(y.astype(np.float32), axis=2).clip(0, 255).astype(np.uint8)
                    if is_one_pixel_edge:
                        y = nake_nms(y)
                        _, y = cv2.threshold(y, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        y = lvmin_thin(y, prunings=new_size_is_bigger)
                    else:
                        _, y = cv2.threshold(y, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    y = np.stack([y] * 3, axis=2)
            else:
                y = x

            if inpaint_mask is not None:
                inpaint_mask = (inpaint_mask > 127).astype(np.float32) * 255.0
                inpaint_mask = inpaint_mask[:, :, None].clip(0, 255).astype(np.uint8)
                y = np.concatenate([y, inpaint_mask], axis=2)

            return y

        if resize_mode == ResizeMode.RESIZE:
            detected_map = high_quality_resize(detected_map, (w, h))
            detected_map = detected_map.copy()  # C-contiguous; never aliases the preprocessor result
            return get_pytorch_control(detected_map), detected_map

        old_h, old_w, _ = detected_map.shape
        old_w = float(old_w)
        old_h = float(old_h)
        k0 = float(h) / old_h
        k1 = float(w) / old_w

        safeint = lambda x: int(np.round(x))

        if resize_mode == ResizeMode.OUTER_FIT:
            k = min(k0, k1)
            borders = np.concatenate([detected_map[0, :, :], detected_map[-1, :, :], detected_map[:, 0, :], detected_map[:, -1, :]], axis=0)
            high_quality_border_color = np.median(borders, axis=0).astype(detected_map.dtype)
            if len(high_quality_border_color) == 4:
                # Inpaint hijack
                high_quality_border_color[3] = 255
            high_quality_background = np.tile(high_quality_border_color[None, None], [h, w, 1])
            detected_map = high_quality_resize(detected_map, (safeint(old_w * k), safeint(old_h * k)))
            new_h, new_w, _ = detected_map.shape
            pad_h = max(0, (h - new_h) // 2)
            pad_w = max(0, (w - new_w) // 2)
            high_quality_background[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = detected_map
            detected_map = high_quality_background
            detected_map = detected_map.copy()
            return get_pytorch_control(detected_map), detected_map
        else:
            k = max(k0, k1)
            detected_map = high_quality_resize(detected_map, (safeint(old_w * k), safeint(old_h * k)))
            new_h, new_w, _ = detected_map.shape
            pad_h = max(0, (new_h - h) // 2)
            pad_w = max(0, (new_w - w) // 2)
            detected_map = detected_map[pad_h:pad_h+h, pad_w:pad_w+w]
            detected_map = detected_map.copy()
            return get_pytorch_control(detected_map), detected_map

    @staticmethod
    def get_enabled_units(p):
        def unfold_merged(unit: ControlNetUnit) -> List[ControlNetUnit]:
            """Unfolds a merged unit to multiple units. Keeps the unit merged for
            preprocessors that can accept multiple input images.
            """
            if unit.input_mode != InputMode.MERGE:
                return [unit]

            if unit.accepts_multiple_inputs:
                unit.input_mode = InputMode.SIMPLE
                return [unit]

            assert isinstance(unit.image, list)
            result = []
            for image in unit.image:
                u = unit.model_copy()
                u.image = [image]
                u.input_mode = InputMode.SIMPLE
                u.weight = unit.weight / len(unit.image)
                result.append(u)
            return result

        # Copies: the script args may be the API's shared default units (the same objects for every request), and
        # parse_remote_call / unfold_merged / pixel-perfect / the inpaint fallback assign onto the units.
        units = [unit.model_copy() for unit in external_code.get_all_units_in_processing(p)]
        if len(units) == 0:
            # fill null groups from legacy remote-call fields, including indexed
            # control_net_*2/control_net_*3 aliases accepted by the API model.
            for idx in range(external_code.get_max_models_num()):
                remote_unit = Script.parse_remote_call(p, ControlNetUnit(), idx)
                if remote_unit.enabled:
                    units.append(remote_unit)

        enabled_units = []
        for idx, unit in enumerate(units):
            local_unit = Script.parse_remote_call(p, unit, idx)
            if not local_unit.enabled:
                continue
            enabled_units.extend(unfold_merged(local_unit))

        Infotext.write_infotext(enabled_units, p)
        return enabled_units

    @staticmethod
    def choose_input_image(
            p: processing.StableDiffusionProcessing,
            unit: ControlNetUnit,
        ) -> Tuple[np.ndarray, ResizeMode]:
        """ Choose input image from following sources with descending priority:
         - unit.image: ControlNet unit input image.
         - p.init_images: A1111 img2img input image.

        Returns:
            - The input image in ndarray form.
            - The resize mode.
        """
        def from_rgba_to_input(img: np.ndarray) -> np.ndarray:
            if (
                shared.opts.data.get("controlnet_ignore_noninpaint_mask", False) or
                (img[:, :, 3] <= 5).all() or
                (img[:, :, 3] >= 250).all()
            ):
                # Take RGB: the values of img[:, :, :3] as a contiguous array. OpenCV drops the alpha channel ~10x
                # faster than the strided copy get_control made of that view (3.6 -> 0.35 ms at 1280x1280).
                if img.dtype == np.uint8 and img.size and img.shape[2] == 4:
                    return cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
                return img[:, :, :3]
            logger.info("Canvas scribble mode. Using mask scribble as input.")
            return HWC3(img[:, :, 3])

        image = unit.get_input_images_rgba()
        a1111_image = getattr(p, "init_images", [None])[0]

        resize_mode = unit.resize_mode

        if image is not None:
            assert isinstance(image, list)
            # Inpaint mask or CLIP mask.
            if unit.is_inpaint or unit.uses_clip:
                # RGBA
                input_image = image
            else:
                # RGB
                input_image = [from_rgba_to_input(img) for img in image]

            if len(input_image) == 1:
                input_image = input_image[0]
        elif a1111_image is not None:
            input_image = HWC3(np.asarray(a1111_image))
            a1111_i2i_resize_mode = getattr(p, "resize_mode", None)
            assert a1111_i2i_resize_mode is not None
            resize_mode = external_code.resize_mode_from_value(a1111_i2i_resize_mode)

            a1111_mask_image : Optional[Image.Image] = getattr(p, "image_mask", None)
            if unit.is_inpaint:
                if a1111_mask_image is not None:
                    a1111_mask = np.array(prepare_mask(a1111_mask_image, p))
                    assert a1111_mask.ndim == 2
                    assert a1111_mask.shape[0] == input_image.shape[0]
                    assert a1111_mask.shape[1] == input_image.shape[1]
                    input_image = np.concatenate([input_image[:, :, 0:3], a1111_mask[:, :, None]], axis=2)
                else:
                    input_image = np.concatenate([
                        input_image[:, :, 0:3],
                        np.zeros_like(input_image, dtype=np.uint8)[:, :, 0:1],
                    ], axis=2)
        else:
            raise ValueError("controlnet is enabled but no input image is given")

        assert isinstance(input_image, (np.ndarray, list))
        return input_image, resize_mode

    @staticmethod
    def try_crop_image_with_a1111_mask(
        p: StableDiffusionProcessing,
        unit: ControlNetUnit,
        input_image: np.ndarray,
        resize_mode: ResizeMode,
    ) -> np.ndarray:
        """
        Crop ControlNet input image based on A1111 inpaint mask given.
        This logic is crutial in upscale scripts, as they use A1111 mask + inpaint_full_res
        to crop tiles.
        """
        # Note: The method determining whether the active script is an upscale script is purely
        # based on `extra_generation_params` these scripts attach on `p`, and subject to change
        # in the future.
        # TODO: Change this to a more robust condition once A1111 offers a way to verify script name.
        is_upscale_script = any("upscale" in k.lower() and "Hires" not in k for k in getattr(p, "extra_generation_params", {}).keys())
        logger.debug(f"is_upscale_script={is_upscale_script}")
        # Note: `inpaint_full_res` is "inpaint area" on UI. The flag is `True` when "Only masked"
        # option is selected.
        a1111_mask_image : Optional[Image.Image] = getattr(p, "image_mask", None)
        is_only_masked_inpaint = (
            issubclass(type(p), StableDiffusionProcessingImg2Img) and
            p.inpaint_full_res and
            a1111_mask_image is not None
        )
        if (
            'reference' not in unit.module
            and is_only_masked_inpaint
            and (is_upscale_script or unit.inpaint_crop_input_image)
        ):
            mask = prepare_mask(a1111_mask_image, p)
            crop_region = masking.get_crop_region_v2(mask, p.inpaint_full_res_padding)
            if crop_region is None:
                # Blank mask: the core does not crop either; it falls back to plain img2img.
                return input_image
            crop_region = masking.expand_crop_region(crop_region, p.width, p.height, mask.width, mask.height)

            logger.debug("Crop input image based on A1111 mask.")
            input_image = [input_image[:, :, i] for i in range(input_image.shape[2])]
            input_image = [Image.fromarray(x) for x in input_image]

            input_image = [
                images.resize_image(resize_mode.int_value(), i, mask.width, mask.height)
                for i in input_image
            ]

            input_image = [x.crop(crop_region) for x in input_image]
            input_image = [
                images.resize_image(ResizeMode.OUTER_FIT.int_value(), x, p.width, p.height)
                for x in input_image
            ]

            input_image = [np.asarray(x)[:, :, 0] for x in input_image]
            input_image = np.stack(input_image, axis=2)
        return input_image

    @staticmethod
    def check_sd_version_compatible(unit: ControlNetUnit) -> None:
        """
        Checks whether the given ControlNet unit has model compatible with the currently
        active sd model. An exception is thrown if ControlNet unit is detected to be
        incompatible.
        """
        sd_version = global_state.get_sd_version()
        assert sd_version != StableDiffusionVersion.UNKNOWN

        if "revision" in unit.module.lower() and sd_version != StableDiffusionVersion.SDXL:
            raise Exception(f"Preprocessor 'revision' only supports SDXL. Current SD base model is {sd_version}.")

        # No need to check if the ControlModelType does not require model to be present.
        if unit.model is None or unit.model.lower() == "none":
            return

        cnet_sd_version = StableDiffusionVersion.detect_from_model_name(unit.model)

        if cnet_sd_version == StableDiffusionVersion.UNKNOWN:
            logger.warning(f"Unable to determine version for ControlNet model '{unit.model}'.")
            return

        if not sd_version.is_compatible_with(cnet_sd_version):
            raise Exception(f"ControlNet model {unit.model}({cnet_sd_version}) is not compatible with sd model({sd_version})")

    @staticmethod
    def get_target_dimensions(p: StableDiffusionProcessing) -> Tuple[int, int, int, int]:
        """Returns (h, w, hr_h, hr_w)."""
        h = align_dim_latent(p.height)
        w = align_dim_latent(p.width)

        high_res_fix = (
            isinstance(p, StableDiffusionProcessingTxt2Img)
            and getattr(p, 'enable_hr', False)
        )
        if high_res_fix:
            # The hires latent size (in pixels) that StableDiffusionProcessingTxt2Img.calculate_target_resolution and
            # sample_hr_pass produce. This runs in process(), before p.init() computes hr_upscale_to_*/truncate_*, so it
            # repeats that arithmetic: the upscale target (a scaled size is floored with the core's 1e-9 slack, since
            # e.g. 800 * 1.15 == 919.9999999999999), the latent of that target, minus the latent rows/columns the
            # core truncates when both resize dimensions are given (which keeps ceil(requested / 8) latents).
            truncate_y = truncate_x = 0
            if p.hr_resize_x == 0 and p.hr_resize_y == 0:
                hr_y = math.floor(p.height * p.hr_scale + 1e-9)
                hr_x = math.floor(p.width * p.hr_scale + 1e-9)
            elif p.hr_resize_y == 0:
                hr_y, hr_x = p.hr_resize_x * p.height // p.width, p.hr_resize_x
            elif p.hr_resize_x == 0:
                hr_y, hr_x = p.hr_resize_y, p.hr_resize_y * p.width // p.height
            else:
                if p.width / p.height < p.hr_resize_x / p.hr_resize_y:
                    hr_y, hr_x = p.hr_resize_x * p.height // p.width, p.hr_resize_x
                else:
                    hr_y, hr_x = p.hr_resize_y, p.hr_resize_y * p.width // p.height
                truncate_y = (hr_y - p.hr_resize_y) // 8
                truncate_x = (hr_x - p.hr_resize_x) // 8
            hr_y = align_dim_latent(hr_y) - truncate_y * 8
            hr_x = align_dim_latent(hr_x) - truncate_x * 8
        else:
            hr_y = h
            hr_x = w

        return h, w, hr_y, hr_x

    def controlnet_main_entry(self, p):
        sd_ldm = p.sd_model
        unet = sd_ldm.model.diffusion_model
        self.noise_modifier = None

        setattr(p, 'controlnet_control_loras', [])

        # A failed generation skips postprocess, and its hook may belong to the
        # other (txt2img/img2img) Script instance; heal the shared UNet first.
        UnetHook.restore_leaked(unet)
        if self.latest_network is not None:
            # always restore (~0.05s)
            self.latest_network.restore()

        # always clear (~0.05s)
        clear_all_secondary_control_models(unet)

        self.enabled_units = Script.get_enabled_units(p)

        if len(self.enabled_units) == 0:
            self.latest_network = None
            return

        detected_maps = []
        forward_params: List[ControlParams] = []
        post_processors = []

        high_res_fix = isinstance(p, StableDiffusionProcessingTxt2Img) and getattr(p, 'enable_hr', False)

        for unit in self.enabled_units:
            Script.check_sd_version_compatible(unit)
            if (
                'inpaint_only' == unit.module and
                issubclass(type(p), StableDiffusionProcessingImg2Img) and
                p.image_mask is not None
            ):
                logger.warning('A1111 inpaint and ControlNet inpaint duplicated. Falls back to inpaint_global_harmonious.')
                unit.module = 'inpaint'

            preprocessor = Preprocessor.get_preprocessor(unit.module)
            assert preprocessor is not None

            if preprocessor.do_not_need_model:
                model_net = None
                if 'reference' in unit.module:
                    control_model_type = ControlModelType.AttentionInjection
                elif 'revision' in unit.module:
                    control_model_type = ControlModelType.ReVision
                else:
                    raise Exception("Unable to determine control_model_type.")
            else:
                model_net, control_model_type = Script.load_control_model(p, unet, unit.model)
                model_net.reset()

                if model_net is not None and devices.fp8 and control_model_type == ControlModelType.ControlNet:
                    for _module in model_net.modules(): # FIXME: let's only apply fp8 to ControlNet for now
                        if isinstance(_module, (torch.nn.Conv2d, torch.nn.Linear)):
                            _module.to(torch.float8_e4m3fn)

                if control_model_type == ControlModelType.ControlLoRA:
                    control_lora = model_net.control_model
                    bind_control_lora(unet, control_lora)
                    p.controlnet_control_loras.append(control_lora)

            if unit.effective_region_mask is not None:
                assert control_model_type.supports_effective_region_mask, f"`effective_region_mask` not supported for {control_model_type}"

            if unit.ipadapter_input is not None:
                # Use ipadapter_input from API call.
                assert control_model_type == ControlModelType.IPAdapter
                controls = unit.ipadapter_input
                hr_controls = unit.ipadapter_input
            else:
                controls, hr_controls, additional_maps = get_control(
                    p, unit, control_model_type, preprocessor)
                detected_maps.extend(additional_maps)

            if len(controls) == len(hr_controls) == 1:
                control = controls[0]
                hr_control = hr_controls[0]
            else:
                control = controls
                hr_control = hr_controls

            preprocessor_dict = dict(
                name=unit.module,
                preprocessor_resolution=unit.processor_res,
                threshold_a=unit.threshold_a,
                threshold_b=unit.threshold_b
            )

            global_average_pooling = (
                control_model_type.is_controlnet and
                model_net.control_model.global_average_pooling
            )

            if control_model_type == ControlModelType.ControlNetUnion:
                logger.info(f"ControlNetUnion control type: {unit.union_control_type}")

            forward_param = ControlParams(
                control_model=model_net,
                preprocessor=preprocessor_dict,
                hint_cond=control,
                weight=unit.weight,
                guidance_stopped=False,
                start_guidance_percent=unit.guidance_start,
                stop_guidance_percent=unit.guidance_end,
                advanced_weighting=unit.advanced_weighting,
                control_model_type=control_model_type,
                global_average_pooling=global_average_pooling,
                hr_hint_cond=hr_control,
                hr_option=unit.hr_option if high_res_fix else HiResFixOption.BOTH,
                soft_injection=unit.control_mode != ControlMode.BALANCED,
                cfg_injection=unit.control_mode == ControlMode.CONTROL,
                effective_region_mask=(
                    get_pytorch_control(unit.effective_region_mask)[:, 0:1, :, :]
                    if unit.effective_region_mask is not None
                    else None
                ),
                # TODO: Implement merge of units with the same union model.
                union_control_types=[unit.union_control_type],
            )
            forward_params.append(forward_param)

            if 'inpaint_only' in unit.module:
                final_inpaint_feed = hr_control if hr_control is not None else control
                final_inpaint_feed = final_inpaint_feed.detach().cpu().numpy().copy()
                final_inpaint_mask = final_inpaint_feed[:, 3, :, :].astype(np.float32)
                final_inpaint_raw = final_inpaint_feed[:, :3].astype(np.float32)
                sigma = shared.opts.data.get("control_net_inpaint_blur_sigma", 7)
                final_inpaint_mask_post_cv = []
                for m in final_inpaint_mask:
                    m = cv2.dilate(m, np.ones((sigma, sigma), dtype=np.uint8))
                    m = cv2.blur(m, (sigma, sigma))[None, None]
                    final_inpaint_mask_post_cv.append(m)
                final_inpaint_mask = np.concatenate(final_inpaint_mask_post_cv, axis=0)
                # Bind this unit's maps now: a closure over the loop variables would give every unit the last unit's.
                post_processors.append(functools.partial(
                    inpaint_only_post_processing,
                    final_inpaint_raw=torch.from_numpy(final_inpaint_raw.copy()),
                    final_inpaint_mask=torch.from_numpy(final_inpaint_mask.copy()),
                ))

            if 'recolor' in unit.module:
                final_feed = hr_control if hr_control is not None else control
                final_feed = final_feed.detach().cpu().numpy().copy()
                final_feed = final_feed[:, 0, :, :].astype(np.float32)
                final_feed = (final_feed * 255).clip(0, 255).astype(np.uint8)
                # Bound per unit (see inpaint_only above).
                if 'luminance' in unit.module:
                    post_processors.append(functools.partial(
                        recolor_post_processing, final_feed=final_feed, to_code=cv2.COLOR_RGB2LAB, from_code=cv2.COLOR_LAB2RGB, channel=0))

                if 'intensity' in unit.module:
                    post_processors.append(functools.partial(
                        recolor_post_processing, final_feed=final_feed, to_code=cv2.COLOR_RGB2HSV, from_code=cv2.COLOR_HSV2RGB, channel=2))

            if '+lama' in unit.module:
                forward_param.used_hint_cond_latent = hook.UnetHook.call_vae_using_process(p, control)
                self.noise_modifier = forward_param.used_hint_cond_latent

            del model_net

        is_low_vram = any(unit.low_vram for unit in self.enabled_units)

        for i, (param, unit) in enumerate(zip(forward_params, self.enabled_units)):
            if param.control_model_type == ControlModelType.IPAdapter:
                if param.advanced_weighting is not None:
                    logger.info(f"IP-Adapter using advanced weighting {param.advanced_weighting}")
                    assert len(param.advanced_weighting) == global_state.get_sd_version().transformer_block_num
                    # Convert advanced weighting list to dict
                    weight = {
                        i: w
                        for i, w in enumerate(param.advanced_weighting)
                        if w > 0
                    }
                else:
                    weight = param.weight

                h, w, hr_y, hr_x = Script.get_target_dimensions(p)
                param.control_model.hook(
                    model=unet,
                    preprocessor_outputs=param.hint_cond,
                    weight=weight,
                    dtype=torch.float32,
                    start=param.start_guidance_percent,
                    end=param.stop_guidance_percent,
                    latent_width=w // 8,
                    latent_height=h // 8,
                    effective_region_mask=param.effective_region_mask,
                )
            if param.control_model_type == ControlModelType.Controlllite:
                param.control_model.hook(
                    model=unet,
                    cond=param.hint_cond,
                    weight=param.weight,
                    start=param.start_guidance_percent,
                    end=param.stop_guidance_percent
                )
            if param.control_model_type == ControlModelType.InstantID:
                # For instant_id we always expect ip-adapter model followed
                # by ControlNet model.
                assert i > 0, "InstantID control model should follow ipadapter model."
                ip_adapter_param = forward_params[i - 1]
                assert ip_adapter_param.control_model_type == ControlModelType.IPAdapter, \
                        "InstantID control model should follow ipadapter model."
                control_model = ip_adapter_param.control_model
                assert hasattr(control_model, "image_emb")
                param.control_context_override = control_model.image_emb

        self.latest_network = UnetHook(lowvram=is_low_vram)
        self.latest_network.hook(model=unet, sd_ldm=sd_ldm, control_params=forward_params, process=p)

        self.detected_map = detected_maps
        self.post_processors = post_processors

    def controlnet_hack(self, p):
        t = time.time()
        if getattr(shared.cmd_opts, 'controlnet_tracemalloc', False):
            tracemalloc.start()
            setattr(self, "malloc_begin", tracemalloc.take_snapshot())

        self.controlnet_main_entry(p)
        if getattr(shared.cmd_opts, 'controlnet_tracemalloc', False):
            logger.info("After hook malloc:")
            for stat in tracemalloc.take_snapshot().compare_to(self.malloc_begin, "lineno")[:10]:
                logger.info(stat)

        if len(self.enabled_units) > 0:
            logger.info(f'ControlNet Hooked - Time = {time.time() - t}')

    @staticmethod
    def process_has_sdxl_refiner(p):
        return getattr(p, 'refiner_checkpoint', None) is not None

    def process(self, p, *args, **kwargs):
        if not Script.process_has_sdxl_refiner(p):
            self.controlnet_hack(p)
        return

    def before_process_batch(self, p, *args, **kwargs):
        if self.noise_modifier is not None:
            p.rng = HackedImageRNG(rng=p.rng,
                                   noise_modifier=self.noise_modifier,
                                   sd_model=p.sd_model)
        self.noise_modifier = None
        if Script.process_has_sdxl_refiner(p):
            self.controlnet_hack(p)
        return

    def before_hr(self, p, *args, **kwargs):
        # sample_hr_pass computes p.hr_c/p.hr_uc (calculate_hr_conds) after the hook's process_sample marked the
        # first pass's conds -- always unless hires_fix_use_firstpass_conds -- and runs this right before sampling
        # with them. Unmarked, unmark_prompt_context reads every hires row as cond: cond-only control
        # ("ControlNet is more important"), IP-Adapter/InstantID uncond embeds and reference style fidelity
        # would treat the uncond rows as cond rows.
        if self.latest_network is not None and self.latest_network.sampling_active:
            UnetHook.mark_hires_conds(p)

    def postprocess_batch(self, p, *args, **kwargs):
        images = kwargs.get('images', [])
        for post_processor in self.post_processors:
            for i in range(len(images)):
                images[i] = post_processor(images[i], i)
        return

    def postprocess(self, p, processed, *args):
        sd_ldm = p.sd_model
        unet = sd_ldm.model.diffusion_model

        clear_all_secondary_control_models(unet)

        self.noise_modifier = None

        for control_lora in getattr(p, 'controlnet_control_loras', []):
            unbind_control_lora(control_lora)
        p.controlnet_control_loras = []

        self.post_processors = []
        setattr(p, 'controlnet_vae_cache', None)

        processor_params_flag = (', '.join(getattr(processed, 'extra_generation_params', []))).lower()

        self.enabled_units.clear()

        if shared.opts.data.get("control_net_detectmap_autosaving", False) and self.latest_network is not None:
            for detect_map, module in self.detected_map:
                detectmap_dir = os.path.join(shared.opts.data.get("control_net_detectedmap_dir", ""), module)
                if not os.path.isabs(detectmap_dir):
                    detectmap_dir = os.path.join(p.outpath_samples, detectmap_dir)
                if module != "none":
                    os.makedirs(detectmap_dir, exist_ok=True)
                    img = Image.fromarray(detect_map.clip(0, 255).astype(np.uint8))
                    save_image(img, detectmap_dir, module)

        if self.latest_network is None:
            return

        if not shared.opts.data.get("control_net_no_detectmap", False):
            if 'sd upscale' not in processor_params_flag:
                if self.detected_map is not None:
                    for detect_map, module in self.detected_map:
                        if detect_map is None:
                            continue
                        detect_map = detect_map.copy()
                        detect_map = external_code.visualize_inpaint_mask(detect_map)
                        processed.images.extend([
                            Image.fromarray(
                                detect_map.clip(0, 255).astype(np.uint8)
                            )
                        ])

        self.latest_network.restore()
        self.latest_network = None
        self.detected_map.clear()

        # No gc.collect()/torch_gc() here: request-end memory release is the core
        # pipeline's policy; forcing it again per request only churns the allocator.
        if getattr(shared.cmd_opts, 'controlnet_tracemalloc', False):
            logger.info("After generation:")
            for stat in tracemalloc.take_snapshot().compare_to(self.malloc_begin, "lineno")[:10]:
                logger.info(stat)
            tracemalloc.stop()


def inpaint_only_post_processing(x, i, final_inpaint_raw, final_inpaint_mask):
    """Paste the unit's unmasked source pixels `final_inpaint_raw[i]` (CHW, [0, 1]) back over output image `x` (CHW,
    [0, 1]), blended by its dilated and blurred mask `final_inpaint_mask[i]` (1HW)."""
    if i >= final_inpaint_raw.shape[0]:
        i = 0
    _, H, W = x.shape
    _, _, Hmask, Wmask = final_inpaint_mask.shape
    if Hmask != H or Wmask != W:
        logger.error('Error: ControlNet find post-processing resolution mismatch. This could be related to other extensions hacked processing.')
        return x
    r = final_inpaint_raw[i].to(x.dtype).to(x.device)
    m = final_inpaint_mask[i].to(x.dtype).to(x.device)
    y = m * x.clip(0, 1) + (1 - m) * r
    y = y.clip(0, 1)
    return y


def recolor_post_processing(x, i, final_feed, to_code, from_code, channel):
    """Replace `channel` of output image `x` (CHW, [0, 1]) in the `to_code` colour space with the recolor
    preprocessor's map `final_feed[i]` (uint8, NHW), then convert back with `from_code`."""
    if i >= final_feed.shape[0]:
        i = 0
    C, H, W = x.shape
    _, Hfeed, Wfeed = final_feed.shape
    if Hfeed != H or Wfeed != W or C != 3:
        logger.error('Error: ControlNet find post-processing resolution mismatch. This could be related to other extensions hacked processing.')
        return x
    h = x.detach().cpu().numpy().transpose((1, 2, 0))
    h = (h * 255).clip(0, 255).astype(np.uint8)
    h = cv2.cvtColor(h, to_code)
    h[:, :, channel] = final_feed[i]
    h = cv2.cvtColor(h, from_code)
    h = (h.astype(np.float32) / 255.0).transpose((2, 0, 1))
    y = torch.from_numpy(h).clip(0, 1).to(x)
    return y


def on_ui_settings():
    section = ('control_net', "ControlNet")
    shared.opts.add_option("control_net_detectedmap_dir", shared.OptionInfo(
        global_state.default_detectedmap_dir, "Directory for detected maps auto saving", section=section))
    shared.opts.add_option("control_net_models_path", shared.OptionInfo(
        "", "Extra path to scan for ControlNet models (e.g. training output directory)", section=section))
    shared.opts.add_option("control_net_modules_path", shared.OptionInfo(
        "", "Legacy path to directory containing annotator/preprocessor model directories (overrides corresponding command line flag when no preprocessor-specific path is set)", section=section).needs_reload_ui())
    shared.opts.add_option("control_net_preprocessor_models_path", shared.OptionInfo(
        "", "Path to directory containing annotator/preprocessor model directories (overrides legacy setting and corresponding command line flag)", section=section).needs_reload_ui())
    shared.opts.add_option("control_net_unit_count", shared.OptionInfo(
        3, "Multi-ControlNet: ControlNet unit number", gr.Slider, {"minimum": 1, "maximum": 10, "step": 1}, section=section).needs_reload_ui())
    shared.opts.add_option("control_net_model_cache_size", shared.OptionInfo(
        2, "Model cache size", gr.Slider, {"minimum": 1, "maximum": 10, "step": 1}, section=section).needs_reload_ui())
    shared.opts.add_option("control_net_inpaint_blur_sigma", shared.OptionInfo(
        7, "ControlNet inpainting Gaussian blur sigma", gr.Slider, {"minimum": 0, "maximum": 64, "step": 1}, section=section))
    shared.opts.add_option("control_net_no_detectmap", shared.OptionInfo(
        False, "Do not append detectmap to output", gr.Checkbox, {"interactive": True}, section=section))
    shared.opts.add_option("control_net_detectmap_autosaving", shared.OptionInfo(
        False, "Allow detectmap auto saving", gr.Checkbox, {"interactive": True}, section=section))
    shared.opts.add_option("control_net_allow_script_control", shared.OptionInfo(
        False, "Allow other script to control this extension", gr.Checkbox, {"interactive": True}, section=section))
    shared.opts.add_option("control_net_sync_field_args", shared.OptionInfo(
        True, "Paste ControlNet parameters in infotext", gr.Checkbox, {"interactive": True}, section=section))
    shared.opts.add_option("controlnet_show_batch_images_in_ui", shared.OptionInfo(
        False, "Show batch images in gradio gallery output", gr.Checkbox, {"interactive": True}, section=section))
    shared.opts.add_option("controlnet_increment_seed_during_batch", shared.OptionInfo(
        False, "Increment seed after each controlnet batch iteration", gr.Checkbox, {"interactive": True}, section=section))
    shared.opts.add_option("controlnet_disable_openpose_edit", shared.OptionInfo(
        False, "Disable openpose edit", gr.Checkbox, {"interactive": True}, section=section))
    shared.opts.add_option("controlnet_disable_photopea_edit", shared.OptionInfo(
        False, "Disable photopea edit", gr.Checkbox, {"interactive": True}, section=section))
    shared.opts.add_option("controlnet_photopea_warning", shared.OptionInfo(
        True, "Photopea popup warning", gr.Checkbox, {"interactive": True}, section=section))
    shared.opts.add_option("controlnet_ignore_noninpaint_mask", shared.OptionInfo(
        False, "Ignore mask on ControlNet input image if control type is not inpaint",
        gr.Checkbox, {"interactive": True}, section=section))
    shared.opts.add_option("controlnet_clip_detector_on_cpu", shared.OptionInfo(
        False, "Load CLIP preprocessor model on CPU",
        gr.Checkbox, {"interactive": True}, section=section))
    shared.opts.add_option("controlnet_control_type_dropdown", shared.OptionInfo(
        False, "Display control type as dropdown",
        gr.Checkbox, {"interactive": True}, section=section).needs_reload_ui())


script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_infotext_pasted(Infotext.on_infotext_pasted)
