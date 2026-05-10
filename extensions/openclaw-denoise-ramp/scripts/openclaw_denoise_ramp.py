from __future__ import annotations

import gradio as gr
import torch

from modules import scripts, sd_samplers_kdiffusion

_ORIGINAL_GET_SIGMAS = getattr(sd_samplers_kdiffusion.KDiffusionSampler, "_openclaw_original_get_sigmas", None)
if _ORIGINAL_GET_SIGMAS is None:
    _ORIGINAL_GET_SIGMAS = sd_samplers_kdiffusion.KDiffusionSampler.get_sigmas
    sd_samplers_kdiffusion.KDiffusionSampler._openclaw_original_get_sigmas = _ORIGINAL_GET_SIGMAS


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


def _ramp_sigmas_for_img2img(p, sigmas: torch.Tensor, steps: int) -> torch.Tensor:
    delta = _safe_float(getattr(p, "openclaw_denoise_step_delta", 0.0), 0.0)
    if abs(delta) < 1e-9 or not hasattr(p, "denoising_strength"):
        return sigmas

    base_strength = max(0.0, min(_safe_float(getattr(p, "denoising_strength", 0.0), 0.0), 0.999))
    if base_strength <= 0.0:
        return sigmas

    total_transitions = max(1, min(steps, sigmas.shape[0] - 1))
    t_enc = int(base_strength * total_transitions)
    if t_enc <= 1:
        return sigmas

    start = max(0, total_transitions - t_enc - 1)
    tail_len = sigmas.shape[0] - start
    if tail_len <= 2:
        return sigmas

    device = sigmas.device
    dtype = sigmas.dtype
    j = torch.arange(tail_len, device=device, dtype=torch.float32)

    strengths = torch.clamp(torch.tensor(base_strength, device=device) + torch.tensor(delta, device=device) * j, 0.001, 0.999)
    desired_positions = (1.0 - strengths) * float(total_transitions)
    base_positions = torch.linspace(float(start), float(sigmas.shape[0] - 1), tail_len, device=device)

    if delta > 0:
        # Stronger-per-step means linger closer to the noisier/high-sigma side,
        # but never reverse the sigma direction.
        positions = torch.minimum(desired_positions, base_positions)
    else:
        # Negative values move down the schedule faster/more conservatively.
        positions = torch.maximum(desired_positions, base_positions)

    fixed = [float(positions[0].item())]
    eps = 1e-4
    last_index = float(sigmas.shape[0] - 1)
    for idx in range(1, tail_len - 1):
        fixed.append(min(last_index - eps * (tail_len - idx), max(float(positions[idx].item()), fixed[-1] + eps)))
    fixed.append(last_index)
    positions = torch.tensor(fixed, device=device, dtype=torch.float32)

    ramped_tail = _interp_sigmas(sigmas, positions).to(dtype)
    ramped_tail[-1] = sigmas[-1]

    out = sigmas.clone()
    out[start:] = ramped_tail
    p.extra_generation_params["Denoise step delta"] = f"{delta:+.3f}"
    return out


def _patched_get_sigmas(self, p, steps):
    sigmas = _ORIGINAL_GET_SIGMAS(self, p, steps)
    return _ramp_sigmas_for_img2img(p, sigmas, steps)


sd_samplers_kdiffusion.KDiffusionSampler.get_sigmas = _patched_get_sigmas


class OpenClawDenoiseRampScript(scripts.Script):
    def title(self):
        return "OpenClaw Denoise Ramp"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        delta = gr.Number(value=0.0, visible=False, precision=3, label="Denoise step delta")
        return [delta]

    def process(self, p, delta=0.0):
        p.openclaw_denoise_step_delta = max(-0.05, min(0.05, _safe_float(delta, 0.0)))
