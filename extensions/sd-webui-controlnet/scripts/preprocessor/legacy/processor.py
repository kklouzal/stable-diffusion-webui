import cv2
import numpy as np
import torch
from typing import Callable, Tuple

from modules.safe import Extra
from modules import devices
from annotator.util import HWC3
from scripts import utils
from scripts.logging import logger


def torch_handler(module: str, name: str):
    """ Allow all torch access. Bypass A1111 safety whitelist. """
    if module == 'torch':
        return getattr(torch, name)
    if module == 'torch._tensor':
        # depth_anything dep.
        return getattr(torch._tensor, name)


def resize_image_with_pad(img, res):
    """scripts.utils.resize_image_with_pad on HWC3(img): 2-D, 1-channel and RGBA input become RGB first."""
    return utils.resize_image_with_pad(HWC3(img), res)


def resize_image_short_side(img, res):
    """scripts.utils.resize_image_short_side on HWC3(img): the resized image without the 64-multiple edge padding."""
    return utils.resize_image_short_side(HWC3(img), res)


def tile_resample(img, res=512, thr_a=1.0, **kwargs):
    img = HWC3(img)
    if thr_a < 1.1:
        return img, True
    H, W, C = img.shape
    H = int(float(H) / float(thr_a))
    W = int(float(W) / float(thr_a))
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    return img, True


def threshold(img, res=512, thr_a=127, **kwargs):
    img, remove_pad = resize_image_with_pad(img, res)
    result = np.zeros_like(img, dtype=np.uint8)
    result[np.min(img, axis=2) > thr_a] = 255
    return remove_pad(result), True


def identity(img, **kwargs):
    return img, True


def _scribble_from_edges(result):
    """Thin a soft-edge map to a binary scribble (NMS, blur, threshold)."""
    from annotator.util import nms
    result = nms(result, 127, 3.0)
    result = cv2.GaussianBlur(result, (0, 0), 3.0)
    result[result > 4] = 255
    result[result < 255] = 0
    return result


model_hed = None


def hed(img, res=512, is_safe=False, **kwargs):
    img, remove_pad = resize_image_with_pad(img, res)
    global model_hed
    if model_hed is None:
        from annotator.hed import apply_hed
        model_hed = apply_hed
    result = model_hed(img, is_safe=is_safe)
    return remove_pad(result), True


def unload_hed():
    global model_hed
    if model_hed is not None:
        from annotator.hed import unload_hed_model
        unload_hed_model()


def scribble_hed(img, res=512, **kwargs):
    result, _ = hed(img, res)
    return _scribble_from_edges(result), True


model_mlsd = None


def mlsd(img, res=512, thr_a=0.1, thr_b=0.1, **kwargs):
    thr_v, thr_d = thr_a, thr_b
    img, remove_pad = resize_image_with_pad(img, res)
    global model_mlsd
    if model_mlsd is None:
        from annotator.mlsd import apply_mlsd
        model_mlsd = apply_mlsd
    result = model_mlsd(img, thr_v, thr_d)
    return remove_pad(result), True


def unload_mlsd():
    global model_mlsd
    if model_mlsd is not None:
        from annotator.mlsd import unload_mlsd_model
        unload_mlsd_model()


model_depth_anything = None


def depth_anything(img, res:int = 512, colored:bool = True, **kwargs):
    # No padding: the network takes any size, and the min-max normalization of its output must not see edge
    # padding (the reference runs on the image as it is).
    img = resize_image_short_side(img, res)
    global model_depth_anything
    if model_depth_anything is None:
        with Extra(torch_handler):
            from annotator.depth_anything import DepthAnythingDetector
            device = devices.get_device_for("controlnet")
            model_depth_anything = DepthAnythingDetector(device)
    return model_depth_anything(img, colored=colored), True


def unload_depth_anything():
    if model_depth_anything is not None:
        model_depth_anything.unload_model()


model_depth_anything_v2 = None


def depth_anything_v2(img, res:int = 512, colored:bool = True, **kwargs):
    # No padding: the network takes any size, and the min-max normalization of its output must not see edge
    # padding (the reference runs on the image as it is).
    img = resize_image_short_side(img, res)
    global model_depth_anything_v2
    if model_depth_anything_v2 is None:
        with Extra(torch_handler):
            from annotator.depth_anything_v2 import DepthAnythingV2Detector
            device = devices.get_device_for("controlnet")
            model_depth_anything_v2 = DepthAnythingV2Detector(device)
    return model_depth_anything_v2(img, colored=colored), True


def unload_depth_anything_v2():
    if model_depth_anything_v2 is not None:
        model_depth_anything_v2.unload_model()


model_midas = None


def midas(img, res=512, a=np.pi * 2.0, **kwargs):
    img, remove_pad = resize_image_with_pad(img, res)
    global model_midas
    if model_midas is None:
        from annotator.midas import apply_midas
        model_midas = apply_midas
    result, _ = model_midas(img, a)
    return remove_pad(result), True


def midas_normal(img, res=512, a=np.pi * 2.0, thr_a=0.4, **kwargs):  # bg_th -> thr_a
    bg_th = thr_a
    img, remove_pad = resize_image_with_pad(img, res)
    global model_midas
    if model_midas is None:
        from annotator.midas import apply_midas
        model_midas = apply_midas
    _, result = model_midas(img, a, bg_th)
    return remove_pad(result), True


def unload_midas():
    global model_midas
    if model_midas is not None:
        from annotator.midas import unload_midas_model
        unload_midas_model()


model_leres = None


def leres(img, res=512, a=np.pi * 2.0, thr_a=0, thr_b=0, boost=False, **kwargs):
    img, remove_pad = resize_image_with_pad(img, res)
    global model_leres
    if model_leres is None:
        from annotator.leres import apply_leres
        model_leres = apply_leres
    result = model_leres(img, thr_a, thr_b, boost=boost)
    return remove_pad(result), True


def unload_leres():
    global model_leres
    if model_leres is not None:
        from annotator.leres import unload_leres_model
        unload_leres_model()


class OpenposeModel(object):
    def __init__(self) -> None:
        self.model_openpose = None

    def run_model(
            self,
            img: np.ndarray,
            include_body: bool,
            include_hand: bool,
            include_face: bool,
            use_dw_pose: bool = False,
            use_animal_pose: bool = False,
            json_pose_callback: Callable[[str], None] = None,
            res: int = 512,
            **kwargs  # Ignore rest of kwargs
    ) -> Tuple[np.ndarray, bool]:
        """Run the openpose model. Returns a tuple of
        - result image
        - is_image flag

        The JSON format pose string is passed to `json_pose_callback`.
        """
        if json_pose_callback is None:
            json_pose_callback = lambda x: None

        img, remove_pad = resize_image_with_pad(img, res)

        if self.model_openpose is None:
            from annotator.openpose import OpenposeDetector
            self.model_openpose = OpenposeDetector()

        return remove_pad(self.model_openpose(
            img,
            include_body=include_body,
            include_hand=include_hand,
            include_face=include_face,
            use_dw_pose=use_dw_pose,
            use_animal_pose=use_animal_pose,
            json_pose_callback=json_pose_callback
        )), True

    def unload(self):
        if self.model_openpose is not None:
            self.model_openpose.unload_model()


g_openpose_model = OpenposeModel()


model_pidinet = None


def pidinet(img, res=512, is_safe=False, apply_fliter=False, **kwargs):
    img, remove_pad = resize_image_with_pad(img, res)
    global model_pidinet
    if model_pidinet is None:
        from annotator.pidinet import apply_pidinet
        model_pidinet = apply_pidinet
    result = model_pidinet(img, is_safe=is_safe, apply_fliter=apply_fliter)
    return remove_pad(result), True


def scribble_pidinet(img, res=512, **kwargs):
    result, _ = pidinet(img, res)
    return _scribble_from_edges(result), True


def unload_pidinet():
    global model_pidinet
    if model_pidinet is not None:
        from annotator.pidinet import unload_pid_model
        unload_pid_model()


clip_encoder = {
    'clip_g': None,
    'clip_h': None,
    'clip_vitl': None,
}


def clip(img, res=512, config='clip_vitl', low_vram=False, **kwargs):
    global clip_encoder
    if clip_encoder[config] is None:
        from annotator.clipvision import ClipVisionDetector
        if low_vram:
            logger.info("Loading CLIP model on CPU.")
        clip_encoder[config] = ClipVisionDetector(config, low_vram)
    result = clip_encoder[config](img)
    return result, False


def unload_clip(config='clip_vitl'):
    global clip_encoder
    if clip_encoder[config] is not None:
        clip_encoder[config].unload_model()
        clip_encoder[config] = None


model_color = None


def color(img, res=512, **kwargs):
    img = HWC3(img)
    global model_color
    if model_color is None:
        from annotator.color import apply_color
        model_color = apply_color
    result = model_color(img, res=res)
    return result, True


def lineart_standard(img, res=512, **kwargs):
    img, remove_pad = resize_image_with_pad(img, res)
    x = img.astype(np.float32)
    g = cv2.GaussianBlur(x, (0, 0), 6.0)
    intensity = np.min(g - x, axis=2).clip(0, 255)
    intensity /= max(16, np.median(intensity[intensity > 8]))
    intensity *= 127
    result = intensity.clip(0, 255).astype(np.uint8)
    return remove_pad(result), True


model_lineart = None


def lineart(img, res=512, **kwargs):
    img, remove_pad = resize_image_with_pad(img, res)
    global model_lineart
    if model_lineart is None:
        from annotator.lineart import LineartDetector
        model_lineart = LineartDetector(LineartDetector.model_default)

    # applied auto inversion
    result = 255 - model_lineart(img)
    return remove_pad(result), True


def unload_lineart():
    global model_lineart
    if model_lineart is not None:
        model_lineart.unload_model()


model_lineart_coarse = None


def lineart_coarse(img, res=512, **kwargs):
    img, remove_pad = resize_image_with_pad(img, res)
    global model_lineart_coarse
    if model_lineart_coarse is None:
        from annotator.lineart import LineartDetector
        model_lineart_coarse = LineartDetector(LineartDetector.model_coarse)

    # applied auto inversion
    result = 255 - model_lineart_coarse(img)
    return remove_pad(result), True


def unload_lineart_coarse():
    global model_lineart_coarse
    if model_lineart_coarse is not None:
        model_lineart_coarse.unload_model()


model_lineart_anime = None


def lineart_anime(img, res=512, **kwargs):
    img, remove_pad = resize_image_with_pad(img, res)
    global model_lineart_anime
    if model_lineart_anime is None:
        from annotator.lineart_anime import LineartAnimeDetector
        model_lineart_anime = LineartAnimeDetector()

    # applied auto inversion
    result = 255 - model_lineart_anime(img)
    return remove_pad(result), True


def unload_lineart_anime():
    global model_lineart_anime
    if model_lineart_anime is not None:
        model_lineart_anime.unload_model()


model_manga_line = None


def lineart_anime_denoise(img, res=512, **kwargs):
    img, remove_pad = resize_image_with_pad(img, res)
    global model_manga_line
    if model_manga_line is None:
        from annotator.manga_line import MangaLineExtration
        model_manga_line = MangaLineExtration()

    # applied auto inversion
    result = model_manga_line(img)
    return remove_pad(result), True


def unload_lineart_anime_denoise():
    global model_manga_line
    if model_manga_line is not None:
        model_manga_line.unload_model()


model_zoe_depth = None


def zoe_depth(img, res=512, **kwargs):
    # No padding: ZoeDepth pads and resizes internally, and its 2nd/85th-percentile normalization must not count edge
    # padding. The reference ZoeDetector (controlnet_aux 0.0.10, and the lllyasviel ControlNet 1.1 annotator its
    # resize_image comes from) runs on the resized image without padding.
    img = resize_image_short_side(img, res)
    global model_zoe_depth
    if model_zoe_depth is None:
        from annotator.zoe import ZoeDetector
        model_zoe_depth = ZoeDetector()
    return model_zoe_depth(img), True


def unload_zoe_depth():
    global model_zoe_depth
    if model_zoe_depth is not None:
        model_zoe_depth.unload_model()


model_normal_bae = None


def normal_bae(img, res=512, **kwargs):
    img, remove_pad = resize_image_with_pad(img, res)
    global model_normal_bae
    if model_normal_bae is None:
        from annotator.normalbae import NormalBaeDetector
        model_normal_bae = NormalBaeDetector()
    result = model_normal_bae(img)
    return remove_pad(result), True


def unload_normal_bae():
    global model_normal_bae
    if model_normal_bae is not None:
        model_normal_bae.unload_model()


def _recolor(img, conversion, channel, gamma):
    """One channel of the RGB input in another colour space, gamma-adjusted, as a grey RGB map."""
    result = cv2.cvtColor(HWC3(img), conversion)[:, :, channel].astype(np.float32) / 255.0
    result = result ** gamma
    result = (result * 255.0).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(result, cv2.COLOR_GRAY2RGB), True


def recolor_luminance(img, res=512, thr_a=1.0, **kwargs):
    # LAB L of the RGB input: controlnet.py's post-processor writes it back through RGB2LAB/LAB2RGB.
    return _recolor(img, cv2.COLOR_RGB2LAB, 0, thr_a)


def recolor_intensity(img, res=512, thr_a=1.0, **kwargs):
    return _recolor(img, cv2.COLOR_RGB2HSV, 2, thr_a)


model_anime_face_segment = None


def anime_face_segment(img, res=512, **kwargs):
    img, remove_pad = resize_image_with_pad(img, res)
    global model_anime_face_segment
    if model_anime_face_segment is None:
        from annotator.anime_face_segment import AnimeFaceSegment
        model_anime_face_segment = AnimeFaceSegment()

    result = model_anime_face_segment(img)
    return remove_pad(result), True


def unload_anime_face_segment():
    global model_anime_face_segment
    if model_anime_face_segment is not None:
        model_anime_face_segment.unload_model()



def densepose(img, res=512, cmap="viridis", **kwargs):
    img, remove_pad = resize_image_with_pad(img, res)
    from annotator.densepose import apply_densepose
    result = apply_densepose(img, cmap=cmap)
    return remove_pad(result), True


def unload_densepose():
    from annotator.densepose import unload_model
    unload_model()
