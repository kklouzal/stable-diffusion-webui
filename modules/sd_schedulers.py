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


def uniform(n, sigma_min, sigma_max, inner_model, device):
    n = _validate_step_count(n)
    return inner_model.get_sigmas(n).to(device)


def _as_sigma(value, device, dtype=torch.float32):
    return torch.as_tensor(value, device=device, dtype=dtype).reshape(())


def _stack_sigmas(sigmas, device, dtype=torch.float32):
    return torch.stack([_as_sigma(sigma, device, dtype) for sigma in sigmas])


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


def _sigmas_from_timesteps(inner_model, timesteps, device, dtype=torch.float32):
    try:
        sigmas = inner_model.t_to_sigma(timesteps)
    except Exception:
        sigmas = None

    if torch.is_tensor(sigmas) and sigmas.ndim > 0:
        return sigmas.to(device=device, dtype=dtype).reshape(-1)

    return _stack_sigmas((inner_model.t_to_sigma(ts) for ts in timesteps), device, dtype)


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

    sigmas = torch.as_tensor(base_sigmas, device=device, dtype=torch.float32)
    if n != sigmas.numel():
        sigmas = _loglinear_interp_sigmas(sigmas, n)

    return _append_zero(sigmas)


def kl_optimal(n, sigma_min, sigma_max, device):
    n = _validate_step_count(n)
    alpha_min = torch.arctan(_as_sigma(sigma_min, device))
    alpha_max = torch.arctan(_as_sigma(sigma_max, device))
    step_indices = torch.arange(n + 1, device=device)
    sigmas = torch.tan(step_indices / n * alpha_min + (1.0 - step_indices / n) * alpha_max)
    return sigmas


def simple_scheduler(n, sigma_min, sigma_max, inner_model, device):
    n = _validate_step_count(n)
    sigmas = torch.as_tensor(inner_model.sigmas, device=device, dtype=torch.float32)
    ss = len(inner_model.sigmas) / n
    indices = -(1 + (torch.arange(n, device=device, dtype=torch.float32) * ss).to(torch.long))
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
    curve = torch.as_tensor(stats.beta.ppf(np.linspace(1, 0, n), alpha, beta), device=device, dtype=torch.float32)

    start = inner_model.sigma_to_t(_as_sigma(sigma_max, device))
    end = inner_model.sigma_to_t(_as_sigma(sigma_min, device))
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

schedulers_map = {**{x.name: x for x in schedulers}, **{x.label: x for x in schedulers}}
