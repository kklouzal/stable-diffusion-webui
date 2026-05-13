from __future__ import annotations

from modules import headless_ui as gr
import torch

from modules import scripts, sd_samplers_common, sd_samplers_kdiffusion

_ORIGINAL_GET_SIGMAS = getattr(sd_samplers_kdiffusion.KDiffusionSampler, "_openclaw_original_get_sigmas", None)
if _ORIGINAL_GET_SIGMAS is None:
    _ORIGINAL_GET_SIGMAS = sd_samplers_kdiffusion.KDiffusionSampler.get_sigmas
    sd_samplers_kdiffusion.KDiffusionSampler._openclaw_original_get_sigmas = _ORIGINAL_GET_SIGMAS

_ORIGINAL_SAMPLE_IMG2IMG = getattr(sd_samplers_kdiffusion.KDiffusionSampler, "_openclaw_original_sample_img2img", None)
if _ORIGINAL_SAMPLE_IMG2IMG is None:
    _ORIGINAL_SAMPLE_IMG2IMG = sd_samplers_kdiffusion.KDiffusionSampler.sample_img2img
    sd_samplers_kdiffusion.KDiffusionSampler._openclaw_original_sample_img2img = _ORIGINAL_SAMPLE_IMG2IMG


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def _interp_sigmas(sigmas: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    max_pos = sigmas.shape[0] - 1
    positions = torch.clamp(positions, 0, max_pos)
    left = torch.floor(positions).long()
    right = torch.clamp(left + 1, max=max_pos)
    frac = (positions - left.to(positions.dtype)).to(sigmas.dtype)
    return sigmas[left] * (1.0 - frac) + sigmas[right] * frac


def ramp_sigmas_for_img2img(p, sigmas: torch.Tensor, steps: int, t_enc: int | None = None) -> torch.Tensor:
    delta = _safe_float(getattr(p, "openclaw_denoise_step_delta", 0.0), 0.0)
    if abs(delta) < 1e-9 or t_enc is None:
        return sigmas

    total_transitions = max(1, min(steps, sigmas.shape[0] - 1))
    t_enc = max(0, min(int(t_enc), total_transitions))
    if t_enc <= 1:
        return sigmas

    # A1111 img2img uses sigmas[steps - t_enc - 1:]. Keep the exact same
    # start/end points and step count, and only bend spacing inside that tail.
    # This avoids the earlier bad behavior where a per-step absolute-strength
    # remap could leave the final image under-denoised/noisy.
    start = max(0, total_transitions - t_enc - 1)
    tail_len = sigmas.shape[0] - start
    if tail_len <= 2:
        return sigmas

    device = sigmas.device
    dtype = sigmas.dtype
    progress = torch.linspace(0.0, 1.0, tail_len, device=device)

    # Small, bounded curvature: + values linger higher/noisier a little longer
    # then catch up; - values drop noise a little faster. Endpoints are fixed.
    gamma = max(0.5, min(1.5, 1.0 + float(delta) * 5.0))
    curved = torch.pow(progress, gamma)
    positions = float(start) + curved * float(tail_len - 1)

    ramped_tail = _interp_sigmas(sigmas, positions).to(dtype)
    ramped_tail[0] = sigmas[start]
    ramped_tail[-1] = sigmas[-1]

    out = sigmas.clone()
    out[start:] = ramped_tail
    p.extra_generation_params["Denoise step delta"] = f"{delta:+.3f}"
    p.extra_generation_params["Denoise ramp gamma"] = f"{gamma:.3f}"
    return out


def _patched_get_sigmas(self, p, steps):
    sigmas = _ORIGINAL_GET_SIGMAS(self, p, steps)
    if not getattr(p, "openclaw_denoise_ramp_active", False):
        return sigmas
    return ramp_sigmas_for_img2img(p, sigmas, steps, getattr(p, "openclaw_denoise_ramp_t_enc", None))


def _patched_sample_img2img(self, p, x, noise, conditioning, unconditional_conditioning, steps=None, image_conditioning=None):
    internal_steps, t_enc = sd_samplers_common.setup_img2img_steps(p, steps)
    previous_active = getattr(p, "openclaw_denoise_ramp_active", False)
    previous_t_enc = getattr(p, "openclaw_denoise_ramp_t_enc", None)
    p.openclaw_denoise_ramp_active = True
    p.openclaw_denoise_ramp_t_enc = t_enc
    try:
        return _ORIGINAL_SAMPLE_IMG2IMG(self, p, x, noise, conditioning, unconditional_conditioning, steps=steps, image_conditioning=image_conditioning)
    finally:
        p.openclaw_denoise_ramp_active = previous_active
        p.openclaw_denoise_ramp_t_enc = previous_t_enc


sd_samplers_kdiffusion.KDiffusionSampler.get_sigmas = _patched_get_sigmas
sd_samplers_kdiffusion.KDiffusionSampler.sample_img2img = _patched_sample_img2img


class OpenClawDenoiseRampScript(scripts.Script):
    def title(self):
        return "OpenClaw Denoise Ramp"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        delta = gr.Number(value=0.0, visible=False, precision=3, label="Denoise step delta")
        return [delta]

    def process(self, p, delta=0.0):
        p.openclaw_denoise_step_delta = max(-0.1, min(0.1, _safe_float(delta, 0.0)))
