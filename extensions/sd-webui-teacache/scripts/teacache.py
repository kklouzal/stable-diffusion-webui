from __future__ import annotations

from typing import Optional

import torch
from sgm.modules.diffusionmodules.openaimodel import timestep_embedding

from modules import headless_ui as gr
from modules import processing, script_callbacks, scripts
from modules.sd_hijack_unet import th
from modules.sd_samplers_common import setup_img2img_steps
from modules.ui_components import InputAccordion

_cache = None

DEFAULT_THRESHOLD = 0.25
DEFAULT_MAX_CONSECUTIVE = 4
DEFAULT_START = 0.35
DEFAULT_END = 0.90

SDXL_POLYNOMIAL_COEFFICIENTS = (
    4.72656327e-03,
    1.09937816e+00,
    4.82785530e+00,
    -2.93749209e+01,
    4.22227031e+01,
)

# One Python bool conversion remains in TeaCacheSession.update_condition: the
# UNet branch must know whether to reuse a cached residual or compute the full
# path. Distance math stays on the residual tensor device until that boundary.


def clamp_float(value, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = default
    return min(max(value, minimum), maximum)


def clamp_int(value, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(float(value))
    except (TypeError, ValueError):
        value = default
    return min(max(value, minimum), maximum)


def normalize_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def normalize_args(args):
    values = [False, DEFAULT_THRESHOLD, DEFAULT_MAX_CONSECUTIVE, DEFAULT_START, DEFAULT_END]
    for index, value in enumerate(args[:len(values)]):
        values[index] = value
    enabled = normalize_bool(values[0])
    threshold = clamp_float(values[1], DEFAULT_THRESHOLD, 0.0, 1.0)
    max_consecutive = clamp_int(values[2], DEFAULT_MAX_CONSECUTIVE, 0, 150)
    start = clamp_float(values[3], DEFAULT_START, 0.0, 1.0)
    end = clamp_float(values[4], DEFAULT_END, 0.0, 1.0)
    if end < start:
        start, end = end, start
    return enabled, threshold, max_consecutive, start, end


def relative_l1_distance(prev: torch.Tensor, curr: torch.Tensor) -> torch.Tensor:
    prev_f = prev.float()
    curr_f = curr.float()
    baseline = prev_f.abs().mean().clamp_min(torch.finfo(prev_f.dtype).eps)
    return (prev_f - curr_f).abs().mean() / baseline


def sdxl_polynomial_distance(relative_distance: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
    result = coeffs[-1]
    for index in range(coeffs.shape[0] - 2, -1, -1):
        result = result * relative_distance + coeffs[index]
    return result


def _tensor_signature(tensor: Optional[torch.Tensor]):
    if tensor is None:
        return None
    return (tuple(tensor.shape), str(tensor.dtype), str(tensor.device))


def _call_signature(
    h: torch.Tensor,
    timesteps: Optional[torch.Tensor],
    context: Optional[torch.Tensor],
    y: Optional[torch.Tensor],
    kwargs: dict,
):
    return (
        _tensor_signature(h),
        _tensor_signature(timesteps),
        _tensor_signature(context),
        _tensor_signature(y),
        tuple(sorted(kwargs.keys())),
    )


def _has_masked_denoising(p: processing.StableDiffusionProcessing) -> bool:
    return any(getattr(p, name, None) is not None for name in ("mask", "nmask", "image_mask"))


class TeaCacheSession:
    def __init__(self, threshold: float, max_consecutive: int, start: float, end: float, steps: int, initial_step: int = 1, disabled_reason: str = ""):
        self.threshold = threshold
        self.max_consecutive = max_consecutive
        self.start = start
        self.end = end
        self.steps = steps
        self.disabled_reason = disabled_reason

        self.current_step = initial_step
        self.call_index = 0
        self.residuals: dict[int, tuple[tuple, torch.Tensor]] = {}
        self.previous_fb: dict[int, torch.Tensor] = {}
        self.distances: dict[int, torch.Tensor] = {}
        self._threshold_tensors: dict[tuple[str, torch.dtype], torch.Tensor] = {}
        self._coefficient_tensors: dict[tuple[str, torch.dtype], torch.Tensor] = {}
        self.consecutive_hits = 0
        self.use_cache = True

    def _device_scalar(self, value: float, reference: torch.Tensor, cache: dict[tuple[str, torch.dtype], torch.Tensor]) -> torch.Tensor:
        key = (str(reference.device), reference.dtype)
        tensor = cache.get(key)
        if tensor is None:
            tensor = reference.new_tensor(value)
            cache[key] = tensor
        return tensor

    def _coefficient_tensor(self, reference: torch.Tensor) -> torch.Tensor:
        key = (str(reference.device), reference.dtype)
        coeffs = self._coefficient_tensors.get(key)
        if coeffs is None:
            coeffs = reference.new_tensor(SDXL_POLYNOMIAL_COEFFICIENTS)
            self._coefficient_tensors[key] = coeffs
        return coeffs

    def update_condition(self, first_block_residual: torch.Tensor, signature: tuple):
        self.use_cache = not self.disabled_reason
        # check step range
        progress = self.current_step / max(1, self.steps)
        if not (self.start < progress <= self.end):
            self.use_cache = False
        # check max consecutive cache hits
        if self.max_consecutive > 0 and self.consecutive_hits >= self.max_consecutive:
            self.use_cache = False
        # check cached value exists for this exact UNet call shape/conditioning lane
        previous_fb = self.previous_fb.get(self.call_index)
        cached = self.residuals.get(self.call_index)
        if previous_fb is None or cached is None or cached[0] != signature:
            self.use_cache = False

        if self.use_cache:
            # NoobAI XL vpred v1.0 coefficients used by upstream TeaCache for SDXL-like UNets.
            distance = self.distances.get(self.call_index)
            if distance is None or distance.device != first_block_residual.device:
                distance = first_block_residual.new_zeros(())
            relative_distance = relative_l1_distance(previous_fb, first_block_residual)
            distance = distance + sdxl_polynomial_distance(relative_distance, self._coefficient_tensor(relative_distance))
            # Intentional sync point: Python must choose cached vs full UNet branch.
            # The relative-distance and polynomial math above remain on GPU.
            if bool(torch.ge(distance, self._device_scalar(self.threshold, distance, self._threshold_tensors))):
                self.use_cache = False
                self.distances[self.call_index] = distance.detach().zero_()
            else:
                self.distances[self.call_index] = distance.detach()
                if self.call_index == 0:
                    self.consecutive_hits += 1

        self.previous_fb[self.call_index] = first_block_residual.detach().clone()

    def next_step(self):
        self.current_step += 1
        self.call_index = 0
        self.use_cache = True

    def can_use_current_residual(self) -> bool:
        return self.use_cache and self.call_index in self.residuals

    def current_residual(self, signature: tuple) -> Optional[torch.Tensor]:
        cached = self.residuals.get(self.call_index)
        if not self.use_cache or cached is None or cached[0] != signature:
            return None
        return cached[1]

    def store_current_residual(self, signature: tuple, residual: torch.Tensor):
        self.residuals[self.call_index] = (signature, residual.detach().clone())


class TeaCacheScript(scripts.Script):
    def __init__(self):
        self.original_forward = None

    def title(self):
        return "TeaCache"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with InputAccordion(False, label=self.title()) as enabled:
            with gr.Row():
                threshold = gr.Slider(
                    label="Cache threshold",
                    info="Higher caches more aggressively.",
                    minimum=0.0, maximum=1.0, value=DEFAULT_THRESHOLD, step=0.01,
                )
            with gr.Row():
                max_consecutive = gr.Number(
                    label="Max consecutive cached steps",
                    minimum=0, maximum=150, value=DEFAULT_MAX_CONSECUTIVE, step=1,
                )
                start = gr.Slider(
                    label="Start",
                    minimum=0.0, maximum=1.0, value=DEFAULT_START, step=0.01,
                )
                end = gr.Slider(
                    label="End",
                    minimum=0.0, maximum=1.0, value=DEFAULT_END, step=0.01,
                )

        infotext_keys = ["TeaCache threshold", "TeaCache max consecutive", "TeaCache start", "TeaCache end"]
        self.infotext_fields = [
            (enabled, lambda d: any(key in d for key in infotext_keys)),
            (threshold, "TeaCache threshold"),
            (max_consecutive, "TeaCache max consecutive"),
            (start, "TeaCache start"),
            (end, "TeaCache end"),
        ]

        components = [enabled, threshold, max_consecutive, start, end]

        return components

    def process(self, p: processing.StableDiffusionProcessing, *args):
        # patch model forward method
        enabled, _, _, _, _ = normalize_args(args)
        if not enabled:
            # fix model if patch was not reverted (due to exception, oom)
            unet = p.sd_model.model.diffusion_model
            if getattr(unet, "_teacache_patched", False):
                self.postprocess(p)
            return
        unet = p.sd_model.model.diffusion_model
        if not self.original_forward:
            self.original_forward = unet.forward
        unet.forward = patched_forward.__get__(unet)
        unet._teacache_patched = True
        unet._openclaw_teacache_original_forward = self.original_forward

    def process_before_every_sampling(self, p: processing.StableDiffusionProcessing, *args, **kwargs):
        # initialize and configure cache
        global _cache
        enabled, threshold, max_consecutive, start, end = normalize_args(args)
        if not enabled:
            return
        disabled_reason = ""
        if not getattr(p.sd_model, "is_sdxl", False):
            disabled_reason = "non-SDXL model"
        elif _has_masked_denoising(p):
            disabled_reason = "masked/inpaint denoising"

        # initial step based on denoise strength
        total_steps = p.steps
        initial_step = 1
        if getattr(self, "is_img2img", False):
            total_steps, steps = setup_img2img_steps(p)
            initial_step = total_steps - steps
        elif getattr(p, "is_hr_pass", False):
            total_steps = getattr(p, "hr_second_pass_steps", 0) or p.steps
            total_steps, steps = setup_img2img_steps(p, total_steps)  # hires fix doesn't reduce steps
            initial_step = total_steps - steps
        _cache = TeaCacheSession(threshold, max_consecutive, start, end, max(1, total_steps), initial_step, disabled_reason)

        # set infotext
        p.extra_generation_params["TeaCache threshold"] = threshold
        if max_consecutive > 0:
            p.extra_generation_params["TeaCache max consecutive"] = max_consecutive
        if start > 0.0:
            p.extra_generation_params["TeaCache start"] = start
        if end < 1.0:
            p.extra_generation_params["TeaCache end"] = end
        if disabled_reason:
            p.extra_generation_params["TeaCache disabled reason"] = disabled_reason

    def postprocess(self, p: processing.StableDiffusionProcessing, *args):
        # restore model, clear cache
        global _cache
        unet = p.sd_model.model.diffusion_model
        if not getattr(unet, "_teacache_patched", False):
            return
        original_forward = getattr(unet, "_openclaw_teacache_original_forward", None) or self.original_forward
        if original_forward is not None:
            unet.forward = original_forward
        unet._teacache_patched = False
        if hasattr(unet, "_openclaw_teacache_original_forward"):
            delattr(unet, "_openclaw_teacache_original_forward")
        self.original_forward = None
        _cache = None


def patched_forward(
    self,
    x: torch.Tensor,
    timesteps: Optional[torch.Tensor] = None,
    context: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    """
    Apply the model to an input batch.
    :param x: an [N x C x ...] Tensor of inputs.
    :param timesteps: a 1-D batch of timesteps.
    :param context: conditioning plugged in via crossattn
    :param y: an [N] Tensor of labels, if class-conditional.
    :return: an [N x C x ...] Tensor of outputs.
    """

    global _cache

    assert (y is not None) == (
        self.num_classes is not None
    ), "must specify y if and only if the model is class-conditional"
    hs = []
    t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
    emb = self.time_embed(t_emb)

    if self.num_classes is not None:
        assert y.shape[0] == x.shape[0]
        emb = emb + self.label_emb(y)

    h = x

    # call first two blocks
    h = self.input_blocks[0](h, emb, context)
    hs.append(h)
    original_h = h
    h = self.input_blocks[1](h, emb, context)
    hs.append(h)

    # check cache condition
    if _cache is None:
        original_forward = getattr(self, "_openclaw_teacache_original_forward", None)
        if original_forward is None:
            raise RuntimeError("TeaCache patched forward has no active cache or original forward")
        return original_forward(x, timesteps=timesteps, context=context, y=y, **kwargs)

    signature = _call_signature(h, timesteps, context, y, kwargs)
    first_block_residual = h - original_h
    _cache.update_condition(first_block_residual, signature)

    # use cache or call full model
    cached_residual = _cache.current_residual(signature)
    if cached_residual is not None:
        h = h + cached_residual
    else:
        original_h = h
        for module in self.input_blocks[2:]:
            h = module(h, emb, context)
            hs.append(h)
        h = self.middle_block(h, emb, context)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)

        if _cache.call_index == 0:
            _cache.consecutive_hits = 0
        _cache.store_current_residual(signature, h - original_h)

    _cache.call_index += 1

    h = h.to(dtype=x.dtype)

    return self.out(h)


def next_step(*args):
    global _cache
    if _cache is not None:
        _cache.next_step()


script_callbacks.on_cfg_after_cfg(next_step)
