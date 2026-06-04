import logging
import math

import torch

logger = logging.getLogger(__name__)

######################### DynThresh Core #########################

class DynThresh:

    Modes = ("Constant", "Linear Down", "Cosine Down", "Half Cosine Down", "Linear Up", "Cosine Up", "Half Cosine Up", "Power Up", "Power Down", "Linear Repeating", "Cosine Repeating", "Sawtooth")
    Startpoints = ("MEAN", "ZERO")
    Variabilities = ("AD", "STD")
    _EXPERIMENT_MODE3_COEFS = (
        (0.298, 0.207, 0.208, 0.0),
        (0.187, 0.286, 0.173, 0.0),
        (-0.158, 0.189, 0.264, 0.0),
        (-0.184, -0.271, -0.473, 1.0),
    )
    _experiment_mode3_matrix_cache = {}  # noqa: RUF012 - shared tensor cache keyed by device/dtype.

    def __init__(self, mimic_scale, threshold_percentile, mimic_mode, mimic_scale_min, cfg_mode, cfg_scale_min, sched_val, experiment_mode, max_steps, separate_feature_channels, scaling_startpoint, variability_measure, interpolate_phi):
        self.mimic_scale = mimic_scale
        self.threshold_percentile = threshold_percentile
        self.mimic_mode = mimic_mode
        self.cfg_mode = cfg_mode
        self.max_steps = max_steps
        self.cfg_scale_min = cfg_scale_min
        self.mimic_scale_min = mimic_scale_min
        self.experiment_mode = experiment_mode
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

    @classmethod
    def _experiment_mode3_matrices(cls, device, dtype):
        device = torch.device(device)
        key = (device.type, device.index, dtype)
        matrices = cls._experiment_mode3_matrix_cache.get(key)
        if matrices is None:
            coefs = torch.tensor(cls._EXPERIMENT_MODE3_COEFS, device=device, dtype=dtype)
            matrices = (coefs, torch.linalg.inv(coefs))
            cls._experiment_mode3_matrix_cache[key] = matrices
        return matrices

    def dynthresh_from_relative(self, relative, uncond, cfg_scale):
        """Apply Dynamic Thresholding from an already aggregated CFG delta.

        Reductions and scaling statistics intentionally run in fp32 for fp16 /
        bf16 friendliness, then cast back to the original latent dtype. This
        avoids unstable half-precision quantile/std/division without changing
        the outward sampler dtype.
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
                mim_scaleref = mim_centered.std(dim=2).unsqueeze(2)
                cfg_scaleref = cfg_centered.std(dim=2).unsqueeze(2)
            else: # 'AD'
                mim_scaleref = mim_centered.abs().amax(dim=2).unsqueeze(2)
                cfg_abs = cfg_centered.abs()
                if self.threshold_percentile >= 1.0:
                    cfg_scaleref = cfg_abs.amax(dim=2).unsqueeze(2)
                else:
                    cfg_scaleref = torch.quantile(cfg_abs, self.threshold_percentile, dim=2).unsqueeze(2)
        else:
            if self.variability_measure == 'STD':
                mim_scaleref = mim_centered.std()
                cfg_scaleref = cfg_centered.std()
            else: # 'AD'
                mim_scaleref = mim_centered.abs().amax()
                cfg_abs = cfg_centered.abs()
                if self.threshold_percentile >= 1.0:
                    cfg_scaleref = cfg_abs.amax()
                else:
                    cfg_scaleref = torch.quantile(cfg_abs, self.threshold_percentile)

        cfg_scaleref = self._safe_denominator(cfg_scaleref)
        mim_scaleref = self._safe_denominator(mim_scaleref)

        if self.scaling_startpoint == 'ZERO':
            scaling_factor = mim_scaleref / cfg_scaleref
            result = cfg_flattened * scaling_factor
        else: # 'MEAN'
            if self.variability_measure == 'STD':
                cfg_renormalized = (cfg_centered / cfg_scaleref) * mim_scaleref
            else: # 'AD'
                max_scaleref = self._safe_denominator(torch.maximum(mim_scaleref, cfg_scaleref))
                cfg_clamped = cfg_centered.clamp(-max_scaleref, max_scaleref)
                cfg_renormalized = (cfg_clamped / max_scaleref) * mim_scaleref
            result = cfg_renormalized + cfg_means

        actual_res = result.unflatten(2, mim_target.shape[2:])

        if self.interpolate_phi != 1.0:
            actual_res = actual_res * self.interpolate_phi + cfg_target * (1.0 - self.interpolate_phi)

        if self.experiment_mode == 1:
            actual_res[:, 1].mul_(torch.where(actual_res[:, 0] > 1.0, 0.5, 1.0))
            actual_res[:, 1].mul_(torch.where(actual_res[:, 1] > 1.0, 0.5, 1.0))
            actual_res[:, 2].mul_(torch.where(actual_res[:, 2] > 1.5, 0.5, 1.0))
        elif self.experiment_mode == 2:
            over_scale = actual_res.abs().amax(dim=1, keepdim=True) > 1.5
            actual_res = actual_res * torch.where(over_scale, 0.7, 1.0)
        elif self.experiment_mode == 3:
            coefs, inv_coefs = self._experiment_mode3_matrices(actual_res.device, stats_dtype)
            res_rgb = torch.einsum("laxy,ab -> lbxy", actual_res, coefs)
            rgb_channel_max = res_rgb[0, :3].amax(dim=(1, 2))
            max_rgb = rgb_channel_max.amax()
            max_w = res_rgb[0, 3].amax()
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "experiment_mode=3 max values: r=%s, g=%s, b=%s, w=%s, rgb=%s",
                    rgb_channel_max[0],
                    rgb_channel_max[1],
                    rgb_channel_max[2],
                    max_w,
                    max_rgb,
                )
            if self.step / max(self.max_steps - 1, 1) > 0.2:
                should_scale = (max_rgb < 2.0) & (max_w < 3.0)
            else:
                should_scale = (max_rgb > 2.4) & (max_w > 3.0)
            scale = torch.where(
                should_scale,
                self._safe_denominator(max_rgb / 2.4),
                torch.ones_like(max_rgb),
            )
            res_rgb = res_rgb / scale
            actual_res = torch.einsum("laxy,ab -> lbxy", res_rgb, inv_coefs)

        return actual_res.to(dtype=orig_dtype)

    def dynthresh(self, cond, uncond, cfg_scale, weights):
        # uncond shape is (batch, 4, height, width)
        if uncond.shape[0] <= 0 or cond.shape[0] % uncond.shape[0] != 0:
            raise ValueError("Expected # of conds per batch to be constant across batches")
        conds_per_batch = cond.shape[0] // uncond.shape[0]
        cond_stacked = cond.reshape((-1, conds_per_batch, *uncond.shape[1:]))
        diff = cond_stacked - uncond.unsqueeze(1)
        if weights is not None:
            diff = diff * weights.to(device=diff.device, dtype=diff.dtype)
        relative = diff.sum(1)
        return self.dynthresh_from_relative(relative, uncond, cfg_scale)
