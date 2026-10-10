from __future__ import annotations

from typing import Optional
import threading

import torch
from sgm.modules.diffusionmodules.openaimodel import timestep_embedding

from modules import headless_ui as gr
from modules import processing, script_callbacks, scripts
from modules.sd_hijack_unet import th
from modules.sd_samplers_common import setup_img2img_steps
from modules.ui_components import InputAccordion

_cache = None
_cache_lock = threading.RLock()

def _get_cache():
    with _cache_lock:
        return _cache

def _set_cache(session):
    global _cache
    with _cache_lock:
        _cache = session

DEFAULT_THRESHOLD = 0.25
DEFAULT_MAX_CONSECUTIVE = 4
DEFAULT_START = 0.35
DEFAULT_END = 0.90

# Rescale polynomial upstream TeaCache fit on NoobAI-XL vpred v1.0 and applies to every SDXL-like UNet. It maps the
# first-block relative L1 change to an estimate of the output change; an eps-prediction model (the production
# checkpoint) has no fit of its own, so the threshold is calibrated against this v-pred fit. Recorded in infotext.
SDXL_POLYNOMIAL_FIT = "NoobAI-XL v-pred fit"
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
    return (prev_f - curr_f).abs_().mean() / baseline


def sdxl_polynomial_distance(relative_distance: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
    result = coeffs[-1]
    for index in range(coeffs.shape[0] - 2, -1, -1):
        result = result * relative_distance + coeffs[index]
    return result


def _tensor_signature(tensor: Optional[torch.Tensor]):
    if tensor is None:
        return None
    return (tuple(tensor.shape), str(tensor.dtype), str(tensor.device))


def _conditioning_change(stored: tuple, current: tuple):
    """Whether a lane's conditioning differs from the conditioning its cached residual was computed with.

    Returns False, True (decided without touching tensor values), or a 0-dim bool tensor on the
    conditioning device, folded into the single refresh sync. SDXL's first block (conv + ResBlock) never
    sees the cross-attention context, so the first-block distance cannot notice a prompt-schedule switch
    or a lane whose call order shifted onto a same-shaped call with different conditioning.
    """
    changed = False
    for previous, value in zip(stored, current):
        if previous is None and value is None:
            continue
        if previous is None or value is None or previous.shape != value.shape:
            return True
        differs = torch.ne(previous, value).any()
        changed = differs if changed is False else torch.logical_or(changed, differs)
    return changed


def _call_signature(
    h: torch.Tensor,
    timesteps: Optional[torch.Tensor],
    context: Optional[torch.Tensor],
    y: Optional[torch.Tensor],
):
    # Extra forward kwargs are not part of the key: _patched_forward_inner never caches a call that has any.
    return (
        _tensor_signature(h),
        _tensor_signature(timesteps),
        _tensor_signature(context),
        _tensor_signature(y),
    )


def _call_original_forward(unet, x, timesteps=None, context=None, y=None, **kwargs) -> torch.Tensor:
    original_forward = getattr(unet, "_openclaw_teacache_original_forward", None)
    if original_forward is None:
        raise RuntimeError("TeaCache patched forward has no active cache or original forward")
    return original_forward(x, timesteps=timesteps, context=context, y=y, **kwargs)


def _teacache_patch_is_live(unet) -> bool:
    """True when ``unet.forward`` is TeaCache's own bound patch, i.e. nothing was installed above it.

    Recognized by the function marker rather than identity so a patch bound by an earlier load of this
    script module (script reload after a failed generation) is still recognized as TeaCache-owned.
    """
    forward = getattr(unet, "forward", None)
    return (
        getattr(getattr(forward, "__func__", None), "_openclaw_teacache_patch", False)
        and getattr(forward, "__self__", None) is unet
    )


def _restore_patched_unet(unet) -> None:
    if unet is None or not getattr(unet, "_teacache_patched", False):
        return
    if not _teacache_patch_is_live(unet):
        # A failed generation skips postprocess, so the patch can still be installed when the next request's
        # ControlNet hook captures it as its baseline and wraps above it. Writing unet.forward here would drop that
        # wrapper (ControlNet silently not applied) and strand its ownership markers (every later ControlNet hook
        # raises). Leave the pass-through patch in place; it delegates while it is not on top, and a later
        # process()/postprocess() restores it once the wrapper above has restored its own baseline.
        return
    original_forward = getattr(unet, "_openclaw_teacache_original_forward", None)
    if original_forward is not None:
        unet.forward = original_forward
    unet._teacache_patched = False
    if hasattr(unet, "_openclaw_teacache_original_forward"):
        delattr(unet, "_openclaw_teacache_original_forward")


def _has_masked_denoising(p: processing.StableDiffusionProcessing) -> bool:
    return any(getattr(p, name, None) is not None for name in ("mask", "nmask", "image_mask"))


def _unet_has_external_forward_hook(unet) -> bool:
    # ControlNet's live ownership marker, or a TeaCache patch that is no longer on top, means another callable
    # wraps the UNet: TeaCache never wraps above it nor restores across its ownership boundary.
    return (
        getattr(unet, "_controlnet_forward_hook_owner", None) is not None
        or (getattr(unet, "_teacache_patched", False) and not _teacache_patch_is_live(unet))
    )


def _has_external_unet_forward_hook(p: processing.StableDiffusionProcessing) -> bool:
    return _unet_has_external_forward_hook(getattr(getattr(getattr(p, "sd_model", None), "model", None), "diffusion_model", None))


class TeaCacheSession:
    def __init__(self, threshold: float, max_consecutive: int, start: float, end: float, steps: int, initial_step: int = 1, disabled_reason: str = ""):
        self.threshold = threshold
        self.max_consecutive = max_consecutive
        self.start = start
        self.end = end
        self.steps = steps
        self.disabled_reason = disabled_reason

        self.current_step = initial_step
        # Per-call-lane state, keyed by the UNet call's index within the denoiser step (call_index).
        self.call_index = 0
        self.residuals: dict[int, tuple[tuple, torch.Tensor]] = {}
        # (context, y) the lane's cached residual was computed with; a lane missing here was stored without them.
        self.residual_conditioning: dict[int, tuple[Optional[torch.Tensor], Optional[torch.Tensor]]] = {}
        self.previous_fb: dict[int, torch.Tensor] = {}
        self.distances: dict[int, torch.Tensor] = {}
        self._threshold_tensors: dict[tuple[str, torch.dtype], torch.Tensor] = {}
        self._coefficient_tensors: dict[tuple[str, torch.dtype], torch.Tensor] = {}
        # Consecutive cache hits per lane: every lane decides on its own distance, so max_consecutive must bound
        # each lane's staleness. A shared lane-0 counter, advanced or reset by lane 0 before later lanes checked
        # it, put those lanes (e.g. PAG's identical-input replay) out of refresh phase with lane 0.
        self.consecutive_hits: dict[int, int] = {}
        self.use_cache = True

    def _device_constant(
        self,
        value: float | tuple[float, ...],
        reference: torch.Tensor,
        cache: dict[tuple[str, torch.dtype], torch.Tensor],
    ) -> torch.Tensor:
        key = (str(reference.device), reference.dtype)
        tensor = cache.get(key)
        if tensor is None:
            tensor = reference.new_tensor(value)
            cache[key] = tensor
        return tensor

    def update_condition(
        self,
        first_block_residual: torch.Tensor,
        signature: tuple,
        context: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
    ):
        # One owned fp32 copy per call, kept as the lane's previous residual: the next call's distance then
        # converts only the current residual (bf16 -> fp32 is exact, so the values are unchanged), and the
        # stored copy replaces the separate clone that kept the lane independent of the producer's tensor.
        current_fb = first_block_residual.detach().to(dtype=torch.float32, copy=True)
        lane = self.call_index
        self.use_cache = not self.disabled_reason
        # check step range
        progress = self.current_step / max(1, self.steps)
        if not (self.start < progress <= self.end):
            self.use_cache = False
        # check max consecutive cache hits of this lane
        hits = self.consecutive_hits.get(lane, 0)
        if self.max_consecutive > 0 and hits >= self.max_consecutive:
            self.use_cache = False
        # check cached value exists for this exact UNet call shape/conditioning lane
        previous_fb = self.previous_fb.get(lane)
        cached = self.residuals.get(lane)
        if previous_fb is None or cached is None or cached[0] != signature:
            self.use_cache = False
        conditioning_changed = False
        if self.use_cache:
            conditioning_changed = _conditioning_change(self.residual_conditioning.get(lane, (None, None)), (context, y))
            if conditioning_changed is True:
                self.use_cache = False

        if self.use_cache:
            distance = self.distances.get(lane)
            if distance is None or distance.device != current_fb.device:
                distance = current_fb.new_zeros(())
            relative_distance = relative_l1_distance(previous_fb, current_fb)
            coeffs = self._device_constant(SDXL_POLYNOMIAL_COEFFICIENTS, relative_distance, self._coefficient_tensors)
            distance = distance + sdxl_polynomial_distance(relative_distance, coeffs)
            threshold = self._device_constant(self.threshold, distance, self._threshold_tensors)
            should_refresh = torch.logical_or(torch.logical_not(torch.isfinite(distance)), torch.ge(distance, threshold))
            if conditioning_changed is not False:
                should_refresh = torch.logical_or(should_refresh, conditioning_changed.to(should_refresh.device))
            # Intentional sync point: Python must choose cached vs full UNet branch.
            # The relative-distance and polynomial math above remain on GPU.
            if bool(should_refresh):
                self.use_cache = False
                self.distances[lane] = distance.detach().zero_()
            else:
                self.distances[lane] = distance.detach()
                self.consecutive_hits[lane] = hits + 1

        self.previous_fb[lane] = current_fb

    def next_step(self):
        self.current_step += 1
        self.call_index = 0

    def current_residual(self, signature: tuple) -> Optional[torch.Tensor]:
        cached = self.residuals.get(self.call_index)
        if not self.use_cache or cached is None or cached[0] != signature:
            return None
        return cached[1]

    def reset_current_distance(self, reference: torch.Tensor):
        self.distances[self.call_index] = reference.detach().new_zeros(())

    def store_current_residual(
        self,
        signature: tuple,
        residual: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
    ):
        residual = residual.detach()
        self.residuals[self.call_index] = (signature, residual.clone())
        # Copies: the producer may reuse or edit its conditioning buffers after this call returns.
        self.residual_conditioning[self.call_index] = tuple(
            None if value is None else value.detach().clone() for value in (context, y)
        )
        self.consecutive_hits[self.call_index] = 0
        self.reset_current_distance(residual)


class TeaCacheScript(scripts.Script):
    def __init__(self):
        self.patched_unet = None

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
        if not enabled or _has_external_unet_forward_hook(p) or _has_masked_denoising(p):
            # Fix and clear any prior patch/session if a previous run ended through
            # exception/OOM or if the model object changed before cleanup.
            self.postprocess(p)
            return
        unet = p.sd_model.model.diffusion_model
        if self.patched_unet is not None and self.patched_unet is not unet and getattr(self.patched_unet, "_teacache_patched", False):
            # Model/refiner switches can replace the active UNet object before the
            # previous postprocess hook observes the old one. Restore the owned
            # patch before installing a patch on the new UNet.
            self.postprocess(p)
        original_forward = getattr(unet, "_openclaw_teacache_original_forward", None)
        if original_forward is None:
            if getattr(unet, "_teacache_patched", False):
                raise RuntimeError("TeaCache UNet patch is missing its original forward")
            original_forward = unet.forward
        self.patched_unet = unet
        unet.forward = patched_forward.__get__(unet)
        unet._teacache_patched = True
        unet._openclaw_teacache_original_forward = original_forward

    def process_before_every_sampling(self, p: processing.StableDiffusionProcessing, *args, **kwargs):
        # initialize and configure cache
        enabled, threshold, max_consecutive, start, end = normalize_args(args)
        if not enabled:
            return
        disabled_reason = ""
        if not getattr(p.sd_model, "is_sdxl", False):
            disabled_reason = "non-SDXL model"
        elif _has_external_unet_forward_hook(p):
            disabled_reason = "external UNet forward hook"
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
        _set_cache(TeaCacheSession(threshold, max_consecutive, start, end, max(1, total_steps), initial_step, disabled_reason))

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
        else:
            p.extra_generation_params["TeaCache rescale"] = SDXL_POLYNOMIAL_FIT

    def postprocess(self, p: processing.StableDiffusionProcessing | None, *args):
        # restore model, clear cache
        unet = self.patched_unet
        if unet is None and p is not None:
            unet = p.sd_model.model.diffusion_model
        _restore_patched_unet(unet)
        _set_cache(None)
        self.patched_unet = None


def _patched_forward_inner(
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

    cache = _get_cache()

    assert (y is not None) == (
        self.num_classes is not None
    ), "must specify y if and only if the model is class-conditional"

    if cache is None or cache.disabled_reason or kwargs or _unet_has_external_forward_hook(self):
        return _call_original_forward(self, x, timesteps=timesteps, context=context, y=y, **kwargs)

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

    signature = _call_signature(h, timesteps, context, y)
    first_block_residual = h - original_h
    cache.update_condition(first_block_residual, signature, context, y)

    # use cache or call full model
    cached_residual = cache.current_residual(signature)
    if cached_residual is not None:
        h = h + cached_residual
    else:
        original_h = h
        try:
            for module in self.input_blocks[2:]:
                h = module(h, emb, context)
                hs.append(h)
            h = self.middle_block(h, emb, context)
            for module in self.output_blocks:
                h = th.cat([h, hs.pop()], dim=1)
                h = module(h, emb, context)
        except Exception:
            cache.reset_current_distance(original_h)
            raise

        cache.store_current_residual(signature, h - original_h, context, y)

    cache.call_index += 1

    h = h.to(dtype=x.dtype)

    return self.out(h)


def patched_forward(
    self,
    x: torch.Tensor,
    timesteps: Optional[torch.Tensor] = None,
    context: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    try:
        return _patched_forward_inner(self, x, timesteps=timesteps, context=context, y=y, **kwargs)
    except Exception:
        if getattr(self, "_teacache_patched", False):
            _restore_patched_unet(self)
            _set_cache(None)
        raise


# Marks the function every TeaCache UNet patch is bound from (see _teacache_patch_is_live).
patched_forward._openclaw_teacache_patch = True


def next_step(*args):
    cache = _get_cache()
    if cache is not None:
        cache.next_step()


script_callbacks.on_cfg_after_cfg(next_step)
