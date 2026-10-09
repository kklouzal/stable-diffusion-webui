import math

import torch

######################### DynThresh Core #########################

def cfg_relative(x_out, conds_list, denoised_uncond):
    """Per image, the weighted sum over its conds of (cond - uncond); same shape and dtype as denoised_uncond.

    With one weight-1 prompt per image whose cond rows lead x_out in order (A1111's batch without AND
    prompts), this is ``x_out[:n] - denoised_uncond`` in one kernel. That is bitwise what the general
    accumulation feeds DynThresh: ``0 + d * 1.0`` differs from ``d`` only for d = -0.0, i.e. cond = -0.0
    and uncond = +0.0, where its only use ``uncond + relative * scale`` is +0.0 for either zero.
    """
    n = denoised_uncond.shape[0]
    if len(conds_list) == n and all(len(conds) == 1 and conds[0][0] == i and conds[0][1] == 1.0 for i, conds in enumerate(conds_list)):
        return x_out[:n] - denoised_uncond
    relative = torch.zeros_like(denoised_uncond)
    for i, conds in enumerate(conds_list):
        for cond_index, weight in conds:
            relative[i] += (x_out[cond_index] - denoised_uncond[i]) * weight
    return relative


def _abs_max(x, dim=None):
    """max |x| (over ``dim``, kept) as one reduction; max is exact and NaN-propagating like ``x.abs().amax()``."""
    return torch.linalg.vector_norm(x, ord=math.inf, dim=dim, keepdim=dim is not None)



class DynThresh:

    Modes = ("Constant", "Linear Down", "Cosine Down", "Half Cosine Down", "Linear Up", "Cosine Up", "Half Cosine Up", "Power Up", "Power Down", "Linear Repeating", "Cosine Repeating", "Sawtooth")

    def __init__(self, mimic_scale, threshold_percentile, mimic_mode, mimic_scale_min, cfg_mode, cfg_scale_min, sched_val, max_steps, separate_feature_channels, scaling_startpoint, variability_measure, interpolate_phi):
        # step and max_steps drive the scale schedules; the A1111 CustomCFGDenoiser sets both from the sampler on
        # every step, so this max_steps only matters to direct users.
        self.mimic_scale = mimic_scale
        self.threshold_percentile = threshold_percentile
        self.mimic_mode = mimic_mode
        self.cfg_mode = cfg_mode
        self.max_steps = max_steps
        self.cfg_scale_min = cfg_scale_min
        self.mimic_scale_min = mimic_scale_min
        self.sched_val = sched_val
        self.sep_feat_channels = separate_feature_channels
        self.scaling_startpoint = scaling_startpoint
        self.variability_measure = variability_measure
        self.interpolate_phi = interpolate_phi

    def interpret_scale(self, scale, mode, min):
        scale -= min
        max_step_index = max(self.max_steps - 1, 1)
        frac = self.step / max_step_index
        if mode == "Constant":
            pass
        elif mode == "Linear Down":
            scale *= 1.0 - frac
        elif mode == "Half Cosine Down":
            scale *= math.cos(frac)
        elif mode == "Cosine Down":
            scale *= math.cos(frac * 1.5707)
        elif mode == "Linear Up":
            scale *= frac
        elif mode == "Half Cosine Up":
            scale *= 1.0 - math.cos(frac)
        elif mode == "Cosine Up":
            scale *= 1.0 - math.cos(frac * 1.5707)
        elif mode == "Power Up":
            scale *= math.pow(frac, self.sched_val)
        elif mode == "Power Down":
            scale *= 1.0 - math.pow(frac, self.sched_val)
        elif mode == "Linear Repeating":
            portion = (frac * self.sched_val) % 1.0
            scale *= (0.5 - portion) * 2 if portion < 0.5 else (portion - 0.5) * 2
        elif mode == "Cosine Repeating":
            scale *= math.cos(frac * 6.28318 * self.sched_val) * 0.5 + 0.5
        elif mode == "Sawtooth":
            scale *= (frac * self.sched_val) % 1.0
        scale += min
        return scale

    @staticmethod
    def _stats_dtype(dtype):
        return torch.float64 if dtype == torch.float64 else torch.float32

    @staticmethod
    def _safe_denominator(value):
        eps = torch.finfo(value.dtype).eps
        return value.clamp_min(eps)

    def dynthresh_from_relative(self, relative, uncond, cfg_scale):
        """Apply Dynamic Thresholding from an already aggregated CFG delta.

        Reductions and scaling statistics intentionally run in fp32 for fp16 /
        bf16 friendliness, then cast back to the original latent dtype. This
        avoids unstable half-precision quantile/std/division without changing
        the outward sampler dtype.

        Runs once per denoiser step on the latent batch, so it avoids kernels
        that cannot change a bit of the result: AD references are one fused
        |x| max reduction, and at percentile 100 the MEAN clamp is skipped
        because |cfg_centered| <= cfg_scaleref <= max_scaleref elementwise.
        """
        orig_dtype = uncond.dtype
        stats_dtype = self._stats_dtype(orig_dtype)
        uncond_f = uncond.to(dtype=stats_dtype)
        relative_f = relative.to(dtype=stats_dtype)
        mimic_scale = self.interpret_scale(self.mimic_scale, self.mimic_mode, self.mimic_scale_min)
        cfg_scale = self.interpret_scale(cfg_scale, self.cfg_mode, self.cfg_scale_min)

        mim_target = uncond_f + relative_f * mimic_scale
        cfg_target = uncond_f + relative_f * cfg_scale

        mim_flattened = mim_target.flatten(2)
        cfg_flattened = cfg_target.flatten(2)
        mim_means = mim_flattened.mean(dim=2).unsqueeze(2)
        cfg_means = cfg_flattened.mean(dim=2).unsqueeze(2)
        mim_centered = mim_flattened - mim_means
        cfg_centered = cfg_flattened - cfg_means

        if self.sep_feat_channels:
            if self.variability_measure == 'STD':
                mim_scaleref = mim_centered.std(dim=2, correction=0).unsqueeze(2)
                cfg_scaleref = cfg_centered.std(dim=2, correction=0).unsqueeze(2)
            else: # 'AD'
                mim_scaleref = _abs_max(mim_centered, dim=2)
                if self.threshold_percentile >= 1.0:
                    cfg_scaleref = _abs_max(cfg_centered, dim=2)
                else:
                    cfg_scaleref = torch.quantile(cfg_centered.abs(), self.threshold_percentile, dim=2).unsqueeze(2)
        else:
            if self.variability_measure == 'STD':
                mim_scaleref = mim_centered.std(correction=0)
                cfg_scaleref = cfg_centered.std(correction=0)
            else: # 'AD'
                mim_scaleref = _abs_max(mim_centered)
                if self.threshold_percentile >= 1.0:
                    cfg_scaleref = _abs_max(cfg_centered)
                else:
                    cfg_scaleref = torch.quantile(cfg_centered.abs(), self.threshold_percentile)

        cfg_scaleref = self._safe_denominator(cfg_scaleref)
        mim_scaleref = self._safe_denominator(mim_scaleref)

        if self.scaling_startpoint == 'ZERO':
            scaling_factor = mim_scaleref / cfg_scaleref
            result = cfg_flattened * scaling_factor
        else: # 'MEAN'
            if self.variability_measure == 'STD':
                cfg_renormalized = (cfg_centered / cfg_scaleref).mul_(mim_scaleref)
            else: # 'AD'
                # Both references are already >= eps (or NaN), so their maximum needs no clamp.
                max_scaleref = torch.maximum(mim_scaleref, cfg_scaleref)
                if self.threshold_percentile < 1.0:
                    cfg_centered = cfg_centered.clamp(-max_scaleref, max_scaleref)
                cfg_renormalized = (cfg_centered / max_scaleref).mul_(mim_scaleref)
            result = cfg_renormalized.add_(cfg_means)

        actual_res = result.unflatten(2, mim_target.shape[2:])

        if self.interpolate_phi != 1.0:
            actual_res = actual_res * self.interpolate_phi + cfg_target * (1.0 - self.interpolate_phi)

        return actual_res.to(dtype=orig_dtype)
