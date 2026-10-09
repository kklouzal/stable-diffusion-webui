from typing import List, Any, Optional, Union, Dict
import numpy as np

from modules import scripts, processing, shared
from modules.api import api
from .args import ControlNetUnit, RESIZE_MODE_ALIASES
from scripts import global_state
from scripts.logging import logger
from scripts.enums import (
    ResizeMode,
    ControlMode,  # noqa: F401
)
from scripts.supported_preprocessor import (
    Preprocessor,
    PreprocessorParameter,  # noqa: F401
)
from scripts.utils import visualize_inpaint_mask  # noqa: F401 (public re-export)

import torch
import base64
import io
from modules.safe import unsafe_torch_load


def get_api_version() -> int:
    return 3


def resize_mode_from_value(value: Union[str, int, ResizeMode]) -> ResizeMode:
    if isinstance(value, str):
        return ResizeMode(RESIZE_MODE_ALIASES.get(value, value))
    elif isinstance(value, int):
        assert value >= 0
        if value == 3:  # 'Just Resize (Latent upscale)'
            return ResizeMode.RESIZE

        if value >= len(ResizeMode):
            logger.warning(
                f"Unrecognized ResizeMode int value {value}. Fall back to RESIZE."
            )
            return ResizeMode.RESIZE

        return [e for e in ResizeMode][value]
    else:
        return value


def pixel_perfect_resolution(
    image: np.ndarray,
    target_H: int,
    target_W: int,
    resize_mode: ResizeMode,
) -> int:
    """
    Calculate the estimated resolution for resizing an image while preserving aspect ratio.

    The function first calculates scaling factors for height and width of the image based on the target
    height and width. Then, based on the chosen resize mode, it either takes the smaller or the larger
    scaling factor to estimate the new resolution.

    If the resize mode is OUTER_FIT, the function uses the smaller scaling factor, ensuring the whole image
    fits within the target dimensions, potentially leaving some empty space.

    If the resize mode is not OUTER_FIT, the function uses the larger scaling factor, ensuring the target
    dimensions are fully filled, potentially cropping the image.

    After calculating the estimated resolution, the function prints some debugging information.

    Args:
        image (np.ndarray): A 3D numpy array representing an image. The dimensions represent [height, width, channels].
        target_H (int): The target height for the image.
        target_W (int): The target width for the image.
        resize_mode (ResizeMode): The mode for resizing.

    Returns:
        int: The estimated resolution after resizing.
    """
    raw_H, raw_W, _ = image.shape

    k0 = float(target_H) / float(raw_H)
    k1 = float(target_W) / float(raw_W)

    if resize_mode == ResizeMode.OUTER_FIT:
        estimation = min(k0, k1) * float(min(raw_H, raw_W))
    else:
        estimation = max(k0, k1) * float(min(raw_H, raw_W))

    logger.debug("Pixel Perfect Computation:")
    logger.debug(f"resize_mode = {resize_mode}")
    logger.debug(f"raw_H = {raw_H}")
    logger.debug(f"raw_W = {raw_W}")
    logger.debug(f"target_H = {target_H}")
    logger.debug(f"target_W = {target_W}")
    logger.debug(f"estimation = {estimation}")

    return int(np.round(estimation))


def to_base64_nparray(encoding: str) -> np.ndarray:
    """
    Convert a base64 image into the image type the extension uses
    """

    return np.array(api.decode_base64_to_image(encoding)).astype("uint8")


def get_all_units_in_processing(
    p: processing.StableDiffusionProcessing,
) -> List[ControlNetUnit]:
    """
    Fetch ControlNet processing units from a StableDiffusionProcessing.

    The units come from ControlNet's slice of `p.script_args`: the range the API recorded in
    `p.openclaw_script_arg_ranges` when a request passed more units than ControlNet has slots
    (modules/api/api.py _assign_script_args), else the script's fixed `args_from:args_to` slots.
    ControlNetUnit args are returned as the script-arg objects themselves (callers that mutate
    them must copy); dict args are parsed into new units.
    """

    cn_script = find_cn_script(p.scripts)
    if cn_script is None:
        return []
    start, end = getattr(p, "openclaw_script_arg_ranges", {}).get(
        id(cn_script), (cn_script.args_from, cn_script.args_to)
    )
    return get_all_units_from(p.script_args[start:end])


def get_all_units_from(script_args: List[Any]) -> List[ControlNetUnit]:
    """
    Fetch ControlNet processing units from ControlNet script arguments.
    Use `get_all_units_in_processing` to fetch the units of a processing object.
    """

    all_units = [
        to_processing_unit(script_arg)
        for script_arg in script_args
        if isinstance(script_arg, (ControlNetUnit, dict))
    ]
    if not all_units:
        logger.warning(
            "No ControlNetUnit detected in args. It is very likely that you are having an extension conflict."
            f"Here are args received by ControlNet: {script_args}."
        )

    return all_units


def get_max_models_num():
    """
    Fetch the maximum number of allowed ControlNet models.
    """

    max_models_num = shared.opts.data.get("control_net_unit_count", 3)
    return max_models_num


def to_processing_unit(unit: Union[Dict, ControlNetUnit]) -> ControlNetUnit:
    """
    Convert different types to processing unit.
    """
    if isinstance(unit, dict):
        return ControlNetUnit.from_dict(unit)

    assert isinstance(unit, ControlNetUnit)
    return unit


def get_models(update: bool = False) -> List[str]:
    """
    Fetch the list of available models.
    Each value is a valid candidate of `ControlNetUnit.model`.

    Keyword arguments:
    update -- Whether to refresh the list from disk. (default False)
    """

    if update:
        global_state.update_cn_models()

    return list(global_state.cn_models_names.values())


def get_modules(alias_names: bool = False) -> List[str]:
    """
    Fetch the list of available preprocessors.
    Each value is a valid candidate of `ControlNetUnit.module`.

    Keyword arguments:
    alias_names -- Whether to get the ui alias names instead of internal keys
    """
    return [
        (p.label if alias_names else p.name)
        for p in Preprocessor.get_sorted_preprocessors()
    ]


def get_modules_detail(alias_names: bool = False) -> Dict[str, Any]:
    """
    get the detail of all preprocessors including
    sliders: the slider config in Auto1111 webUI

    Keyword arguments:
    alias_names -- Whether to get the module detail with alias names instead of internal keys
    """

    _module_detail = {}
    _module_list = get_modules(False)
    _module_list_alias = get_modules(True)

    _output_list = _module_list if not alias_names else _module_list_alias
    for module_name in _output_list:
        preprocessor = Preprocessor.get_preprocessor(module_name)
        assert preprocessor is not None
        _module_detail[module_name] = dict(
            model_free=preprocessor.do_not_need_model,
            sliders=[
                s.api_json
                for s in (
                    preprocessor.slider_resolution,
                    preprocessor.slider_1,
                    preprocessor.slider_2,
                    preprocessor.slider_3,
                )
                if s.visible
            ],
        )

    return _module_detail


def find_cn_script(script_runner: scripts.ScriptRunner) -> Optional[scripts.Script]:
    """
    Find the ControlNet script in `script_runner`. Returns `None` if `script_runner` does not contain a ControlNet script.
    """

    if script_runner is None:
        return None

    for script in script_runner.alwayson_scripts:
        if is_cn_script(script):
            return script


def is_cn_script(script: scripts.Script) -> bool:
    """
    Determine whether `script` is a ControlNet script.
    """

    return script.title().lower() == "controlnet"


# TODO: Add model constraint
ControlNetUnit.cls_match_model = lambda model: True
ControlNetUnit.cls_match_module = (
    lambda module: Preprocessor.get_preprocessor(module) is not None
)
ControlNetUnit.cls_get_preprocessor = Preprocessor.get_preprocessor
ControlNetUnit.cls_decode_base64 = to_base64_nparray


def decode_base64(b: str) -> torch.Tensor:
    decoded_bytes = base64.b64decode(b)
    # API input is data, never code: tensors and plain containers only, whatever
    # TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD says (it only changes the default).
    return unsafe_torch_load(io.BytesIO(decoded_bytes), weights_only=True)


ControlNetUnit.cls_torch_load_base64 = decode_base64
ControlNetUnit.cls_logger = logger

logger.debug("ControlNetUnit initialized")
