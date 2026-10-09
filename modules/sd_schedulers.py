import dataclasses
import torch
import k_diffusion
import numpy as np
from scipy import stats

from modules import shared


def to_d(x, sigma, denoised):
    """Converts a denoiser output to a Karras ODE derivative."""
    return (x - denoised) / sigma


k_diffusion.sampling.to_d = to_d


@dataclasses.dataclass
class Scheduler:
    name: str
    label: str
    function: any

    default_rho: float = -1
    need_inner_model: bool = False
    aliases: list = None


def scheduler_labels():
    return [scheduler.label for scheduler in schedulers]


def uniform(n, sigma_min, sigma_max, inner_model, device):
    n = _validate_step_count(n)
    return inner_model.get_sigmas(n).to(device)


def _as_sigma(value, device):
    return torch.as_tensor(value, device=device, dtype=torch.float32).reshape(())


def _append_zero(sigmas):
    return torch.cat([sigmas, sigmas.new_zeros(1)])


def _loglinear_interp_sigmas(sigmas, num_steps):
    log_values = sigmas.flip(0).log().reshape(1, 1, -1)
    interped = torch.nn.functional.interpolate(log_values, size=num_steps, mode="linear", align_corners=True).reshape(-1)
    return interped.exp().flip(0)


def _validate_step_count(n):
    if n <= 0:
        raise ValueError("scheduler step count must be positive")
    return n


def _sigmas_from_timesteps(inner_model, timesteps, device):
    # Every model wrapper is a k_diffusion DiscreteSchedule, whose t_to_sigma takes a timestep vector.
    return inner_model.t_to_sigma(timesteps).to(device=device, dtype=torch.float32).reshape(-1)


def sgm_uniform(n, sigma_min, sigma_max, inner_model, device):
    n = _validate_step_count(n)
    start = inner_model.sigma_to_t(_as_sigma(sigma_max, device))
    end = inner_model.sigma_to_t(_as_sigma(sigma_min, device))
    timesteps = torch.linspace(start, end, n + 1, device=device)[:-1]
    return _append_zero(_sigmas_from_timesteps(inner_model, timesteps, device))


def get_align_your_steps_sigmas(n, sigma_min, sigma_max, device):
    n = _validate_step_count(n)

    # https://research.nvidia.com/labs/toronto-ai/AlignYourSteps/howto.html
    if shared.sd_model.is_sdxl:
        base_sigmas = [14.615, 6.315, 3.771, 2.181, 1.342, 0.862, 0.555, 0.380, 0.234, 0.113, 0.029]
    else:
        # Default to SD 1.5 sigmas.
        base_sigmas = [14.615, 6.475, 3.861, 2.697, 1.886, 1.396, 0.963, 0.652, 0.399, 0.152, 0.029]

    # Interpolate in float64 and round once: the reference (NVIDIA's how-to, upstream np.interp) works in float64, and
    # float32 log/interpolate/exp drifts up to 16 ulp from it.
    sigmas = torch.as_tensor(base_sigmas, device=device, dtype=torch.float64)
    if n != sigmas.numel():
        sigmas = _loglinear_interp_sigmas(sigmas, n)

    return _append_zero(sigmas.to(torch.float32))


def kl_optimal(n, sigma_min, sigma_max, device):
    """KL-optimal schedule (Sabour et al., "Align Your Steps", arXiv:2404.14507): n sigmas from sigma_max to sigma_min,
    evenly spaced in arctan(sigma), then the terminal 0 every k-diffusion schedule ends with, so that the last step
    denoises to sigma 0 instead of returning a latent that still carries sigma_min noise."""
    n = _validate_step_count(n)
    alpha_min = torch.arctan(_as_sigma(sigma_min, device))
    alpha_max = torch.arctan(_as_sigma(sigma_max, device))
    ramp = torch.arange(n, device=device, dtype=torch.float32) / max(n - 1, 1)
    sigmas = torch.tan(ramp * alpha_min + (1.0 - ramp) * alpha_max)
    return _append_zero(sigmas)


def simple_scheduler(n, sigma_min, sigma_max, inner_model, device):
    n = _validate_step_count(n)
    sigmas = torch.as_tensor(inner_model.sigmas, device=device, dtype=torch.float32)
    ss = len(inner_model.sigmas) / n
    # float64 like the reference loop's int(i * ss): float32 products truncate to the neighbouring table index for
    # 112 of the step counts 1..1000 (30 among them).
    indices = -(1 + (torch.arange(n, device=device, dtype=torch.float64) * ss).to(torch.long))
    return _append_zero(sigmas[indices])


def normal_scheduler(n, sigma_min, sigma_max, inner_model, device, sgm=False, floor=False):
    n = _validate_step_count(n)
    start = inner_model.sigma_to_t(_as_sigma(sigma_max, device))
    end = inner_model.sigma_to_t(_as_sigma(sigma_min, device))

    if sgm:
        timesteps = torch.linspace(start, end, n + 1, device=device)[:-1]
    else:
        timesteps = torch.linspace(start, end, n, device=device)

    return _append_zero(_sigmas_from_timesteps(inner_model, timesteps, device))


def ddim_scheduler(n, sigma_min, sigma_max, inner_model, device):
    n = _validate_step_count(n)
    sigmas = torch.as_tensor(inner_model.sigmas, device=device, dtype=torch.float32)
    ss = max(len(inner_model.sigmas) // n, 1)
    indices = torch.arange(1, len(inner_model.sigmas), ss, device=device)
    return _append_zero(sigmas[indices].flip(0))


def beta_scheduler(n, sigma_min, sigma_max, inner_model, device):
    n = _validate_step_count(n)

    # From "Beta Sampling is All You Need" [arXiv:2407.12173] (Lee et. al, 2024)
    alpha = shared.opts.beta_dist_alpha
    beta = shared.opts.beta_dist_beta
    start = inner_model.sigma_to_t(_as_sigma(sigma_max, device))
    end = inner_model.sigma_to_t(_as_sigma(sigma_min, device))
    # sigma_to_t follows the model wrapper's sigma buffers (CUDA), not `device`, so build the curve there. Keep float32:
    # quantized sigma_to_t returns integer timesteps, which the interpolation must not truncate.
    curve = torch.as_tensor(stats.beta.ppf(np.linspace(1, 0, n), alpha, beta), device=start.device, dtype=torch.float32)
    timesteps = end + curve * (start - end)
    return _append_zero(_sigmas_from_timesteps(inner_model, timesteps, device))


schedulers = [
    Scheduler('automatic', 'Automatic', None),
    Scheduler('uniform', 'Uniform', uniform, need_inner_model=True),
    Scheduler('karras', 'Karras', k_diffusion.sampling.get_sigmas_karras, default_rho=7.0),
    Scheduler('exponential', 'Exponential', k_diffusion.sampling.get_sigmas_exponential),
    Scheduler('polyexponential', 'Polyexponential', k_diffusion.sampling.get_sigmas_polyexponential, default_rho=1.0),
    Scheduler('sgm_uniform', 'SGM Uniform', sgm_uniform, need_inner_model=True, aliases=["SGMUniform"]),
    Scheduler('kl_optimal', 'KL Optimal', kl_optimal),
    Scheduler('align_your_steps', 'Align Your Steps', get_align_your_steps_sigmas),
    Scheduler('simple', 'Simple', simple_scheduler, need_inner_model=True),
    Scheduler('normal', 'Normal', normal_scheduler, need_inner_model=True),
    Scheduler('ddim', 'DDIM', ddim_scheduler, need_inner_model=True),
    Scheduler('beta', 'Beta', beta_scheduler, need_inner_model=True),
]

def _scheduler_lookup_keys(scheduler):
    return (scheduler.name, scheduler.label, *(scheduler.aliases or []))


schedulers_map = {key: scheduler for scheduler in schedulers for key in _scheduler_lookup_keys(scheduler)}
