import logging
import time
import torch
import torch.nn.functional as F
from modules import scripts, script_callbacks
from modules.script_callbacks import CFGDenoiserParams
from modules.processing import StableDiffusionProcessing
from scripts.ui_wrapper import UIWrapper

logger = logging.getLogger(__name__)
_SANF_KERNEL_CACHE = {}


def sanf_gaussian_blur3(x):
        """Small cached 3x3 Gaussian blur for PAG SANF saliency maps."""
        squeeze_batch = x.ndim == 3
        x_in = x.unsqueeze(0) if squeeze_batch else x
        if min(x_in.shape[-2:]) < 2:
            return x
        channels = x_in.shape[-3]
        key = (x_in.device, x_in.dtype, channels)
        kernel = _SANF_KERNEL_CACHE.get(key)
        if kernel is None:
                coords = torch.arange(-1, 2, device=x_in.device, dtype=torch.float32)
                gaussian_1d = torch.exp(-0.5 * coords.pow(2)).to(dtype=x_in.dtype)
                gaussian_1d = gaussian_1d / gaussian_1d.sum()
                base = torch.mm(gaussian_1d[:, None], gaussian_1d[None, :])
                kernel = base.expand(channels, 1, 3, 3).contiguous()
                _SANF_KERNEL_CACHE[key] = kernel
        x_blur = F.conv2d(F.pad(x_in, (1, 1, 1, 1), mode='reflect'), kernel, groups=channels)
        return x_blur.squeeze(0) if squeeze_batch else x_blur


def _record_cfg_timing(cfg_dict, hook_name, elapsed):
        timings = cfg_dict.setdefault("openclaw_extension_timings", {})
        hook = timings.setdefault(hook_name, {"total_seconds": 0.0, "calls": 0})
        hook["total_seconds"] = round(float(hook.get("total_seconds") or 0.0) + float(elapsed), 6)
        hook["calls"] = int(hook.get("calls") or 0) + 1


def _merge_cfg_timings(p, cfg_dict):
        cfg_timings = (cfg_dict or {}).get("openclaw_extension_timings") or {}
        if not cfg_timings:
                return
        timings = getattr(p, "openclaw_extension_timings", None)
        if timings is None:
                timings = p.openclaw_extension_timings = {"total_seconds": 0.0, "extensions": {}}
        ext = timings["extensions"].setdefault("Incantations.CFGCombinerScript", {"total_seconds": 0.0, "calls": 0, "hooks": {}})
        for hook_name, hook in cfg_timings.items():
                elapsed = float(hook.get("total_seconds") or 0.0)
                calls = int(hook.get("calls") or 0)
                timings["total_seconds"] = round(float(timings.get("total_seconds") or 0.0) + elapsed, 6)
                ext["total_seconds"] = round(float(ext.get("total_seconds") or 0.0) + elapsed, 6)
                ext["calls"] = int(ext.get("calls") or 0) + calls
                ext["hooks"][hook_name] = round(float(ext["hooks"].get(hook_name) or 0.0) + elapsed, 6)
        cfg_dict["openclaw_extension_timings"] = {}


class CFGCombinerScript(UIWrapper):
        """Owns GB10 Incantations CFG denoiser composition.

        PAG and the CFG interval scheduler need to change A1111's
        ``CFGDenoiser.combine_denoised`` result.  The abandoned upstream
        extension used A1111's generic patch stack and attempted to unpatch on
        every denoiser callback.  That made ownership unclear and could bypass
        wrappers from quality-critical extensions such as Dynamic Thresholding /
        CFG-Fix.

        GB10 keeps this lifecycle explicit:
        - capture the currently-installed combine_denoised callable once
        - install one wrapper for the active batch
        - delegate base CFG to the captured callable so external wrappers still
          compose
        - restore only if the denoiser still points at our exact wrapper
        """
        def __init__(self):
                self._cfg_denoiser_callback = None

        # Extension title in menu UI
        def title(self):
                return "CFG Combiner"

        # Decide to show menu in txt2img or img2img
        def show(self, is_img2img):
                return scripts.AlwaysVisible

        # Setup menu ui detail
        def setup_ui(self, is_img2img):
            self.infotext_fields = []
            self.paste_field_names = []
            return []

        def before_process(self, p: StableDiffusionProcessing, *args, **kwargs):
            logger.debug("CFGCombinerScript before_process")
            if not hasattr(p, 'incant_cfg_params'):
                p.incant_cfg_params = {}

        def process(self, p: StableDiffusionProcessing, *args, **kwargs):
            pass

        def before_process_batch(self, p: StableDiffusionProcessing, *args, **kwargs):
            pass

        def process_batch(self, p: StableDiffusionProcessing, *args, **kwargs):
            """Register only when PAG/CFG interval state exists for this batch.

            The combiner is a no-op without PAG/CFG interval parameters, but
            registering it anyway adds a CFG denoiser callback on every sampler
            step and patches ``combine_denoised`` on the first step. Skipping
            that inactive path preserves output semantics and removes avoidable
            per-step Python work for normal generations.
            """
            logger.debug("CFGCombinerScript process_batch")
            self.remove_callbacks()
            if not getattr(p, 'incant_cfg_params', None) or p.incant_cfg_params.get('pag_params') is None:
                return
            cfg_denoise_lambda = lambda params: self.on_cfg_denoiser_callback(params, p.incant_cfg_params)
            self._cfg_denoiser_callback = cfg_denoise_lambda
            script_callbacks.on_cfg_denoiser(cfg_denoise_lambda)
            logger.debug('Hooked CFG combiner callback')

        def postprocess_batch(self, p: StableDiffusionProcessing, *args, **kwargs):
            logger.debug("CFGCombinerScript postprocess_batch")
            cfg_dict = getattr(p, 'incant_cfg_params', None)
            _merge_cfg_timings(p, cfg_dict)
            self.restore_cfg_denoiser(cfg_dict)
            self.remove_callbacks()

        def unhook_callbacks(self, cfg_dict = None):
            self.restore_cfg_denoiser(cfg_dict)
            self.remove_callbacks()

        def remove_callbacks(self):
            if self._cfg_denoiser_callback is not None:
                    script_callbacks.remove_callbacks_for_function(self._cfg_denoiser_callback)
                    self._cfg_denoiser_callback = None

        def get_xyz_axis_options(self) -> dict:
            return {}

        def on_cfg_denoiser_callback(self, params: CFGDenoiserParams, cfg_dict: dict):
            """Callback for when the CFG denoiser is available.

            Installs one owned wrapper for the batch. Later callbacks for the
            same denoiser reuse the wrapper and only observe the mutable state
            in cfg_dict / pag_params.
            """
            self.patch_cfg_denoiser(params.denoiser, cfg_dict)

        def patch_cfg_denoiser(self, denoiser, cfg_dict: dict):
            """Install the GB10 combine_denoised wrapper once for this denoiser."""
            if not cfg_dict:
                    logger.error("Unable to patch CFG Denoiser, no dict passed as cfg_dict")
                    return
            if not denoiser:
                    logger.error("Unable to patch CFG Denoiser, denoiser is None")
                    return

            wrapped = cfg_dict.get('wrapped_combine_denoised')
            if cfg_dict.get('denoiser') is denoiser and wrapped is not None and denoiser.combine_denoised is wrapped:
                    return

            if cfg_dict.get('denoiser') is not None and cfg_dict.get('denoiser') is not denoiser:
                    self.restore_cfg_denoiser(cfg_dict)

            original_func = denoiser.combine_denoised

            def gb10_combine_denoised(x_out, conds_list, uncond, cond_scale):
                    return combine_denoised_pass_conds_list(
                            x_out,
                            conds_list,
                            uncond,
                            cond_scale,
                            original_func=original_func,
                            cfg_dict=cfg_dict,
                    )

            gb10_combine_denoised.__name__ = 'gb10_incantations_combine_denoised'
            denoiser.combine_denoised = gb10_combine_denoised
            denoiser._gb10_incantations_original_combine_denoised = original_func
            denoiser._gb10_incantations_wrapped_combine_denoised = gb10_combine_denoised
            cfg_dict['denoiser'] = denoiser
            cfg_dict['original_combine_denoised'] = original_func
            cfg_dict['wrapped_combine_denoised'] = gb10_combine_denoised

        def restore_cfg_denoiser(self, cfg_dict = None):
            """Restore combine_denoised if it still points at our exact wrapper.

            If another extension has wrapped our wrapper after installation,
            do not clobber it. That is safer than blindly restoring and deleting
            someone else's outer wrapper.
            """
            if cfg_dict is None:
                    return
            denoiser = cfg_dict.get('denoiser')
            original = cfg_dict.get('original_combine_denoised')
            wrapped = cfg_dict.get('wrapped_combine_denoised')
            if denoiser is None or original is None or wrapped is None:
                    return

            if getattr(denoiser, 'combine_denoised', None) is wrapped:
                    denoiser.combine_denoised = original
                    for attr in (
                            '_gb10_incantations_original_combine_denoised',
                            '_gb10_incantations_wrapped_combine_denoised',
                    ):
                            if hasattr(denoiser, attr):
                                    delattr(denoiser, attr)
            else:
                    logger.warning("Not restoring combine_denoised because another wrapper replaced the GB10 wrapper")

            cfg_dict['denoiser'] = None
            cfg_dict['original_combine_denoised'] = None
            cfg_dict['wrapped_combine_denoised'] = None


def combine_denoised_pass_conds_list(*args, **kwargs):
        """Owned combine_denoised wrapper for PAG and CFG interval scheduling.

        The captured original_func is intentionally called for the base CFG path
        so Dynamic Thresholding / CFG-Fix and similar extensions can still
        rescale the base CFG result before PAG is added.
        """
        original_func = kwargs.get('original_func')
        cfg_dict = kwargs.get('cfg_dict') or {}
        pag_params = cfg_dict.get('pag_params')
        if original_func is None:
                raise RuntimeError("GB10 CFG combiner missing original combine_denoised function")

        if pag_params is None:
                return original_func(*args)

        def new_combine_denoised(x_out, conds_list, uncond, cond_scale):
                # SDXL passes dict conditioning here; keep the original object for
                # the captured combiner, but use the cross-attention tensor for
                # shape/index math in GB10's PAG/SANF path.
                uncond_tensor = uncond.get('crossattn') if isinstance(uncond, dict) else uncond
                if uncond_tensor is None:
                        raise RuntimeError("GB10 CFG combiner could not derive unconditional tensor")
                denoised_uncond = x_out[-uncond_tensor.shape[0]:]

                ### Variables
                # 0. Standard CFG Value
                cfg_scale = cond_scale

                # 1. CFG Interval
                # Overrides cfg_scale if pag_params is not None
                if pag_params is not None and pag_params.cfg_interval_enable:
                        cfg_scale = pag_params.cfg_interval_scheduled_value

                # Build the base CFG result by delegating to the captured original combiner.
                # This is intentionally important for compatibility with extensions such
                # as Dynamic Thresholding / CFG-Fix, which wrap combine_denoised to rescale
                # the CFG result. Older local code recomputed CFG here and accidentally
                # bypassed those wrappers whenever PAG was active.
                original_started = time.perf_counter()
                try:
                        denoised = original_func(x_out, conds_list, uncond, cfg_scale)
                finally:
                        _record_cfg_timing(cfg_dict, "combine_original", time.perf_counter() - original_started)

                # 2. PAG
                pag_x_out = None
                pag_scale = None
                run_pag = False
                if pag_params is not None:
                        pag_active = pag_params.pag_active
                        pag_x_out = pag_params.pag_x_out
                        pag_scale = pag_params.pag_scale

                        if not pag_active or not (pag_params.pag_start_step <= pag_params.step <= pag_params.pag_end_step) or pag_scale <= 0:
                                run_pag = False
                        elif pag_x_out is None:
                                logger.warning("PAG was requested but no PAG denoised output is available; using base CFG only")
                        else:
                                run_pag = pag_active

                # Dynamic Thresholding can be composed cleanly with the base CFG path above.
                # PAG SANF replaces the CFG contribution with a saliency-selected CFG/PAG
                # blend, so it cannot faithfully preserve a dynamically-thresholded base.
                # In that case keep the previous SANF behavior rather than pretending both
                # rescalers are fully applied.
                use_saliency_map = False
                if pag_params is not None:
                        use_saliency_map = pag_params.pag_sanf
                if use_saliency_map and run_pag:
                        denoised = denoised_uncond.clone()

                ### Add PAG guidance on top of the base CFG result
                for i, conds in enumerate(conds_list):
                        for cond_index, weight in conds:
                                if pag_params is None or not run_pag:
                                        continue
                                try:
                                        pag_index = cond_index if cond_index < pag_x_out.shape[0] else i
                                        pag_delta = x_out[cond_index] - pag_x_out[pag_index]
                                        pag_x = pag_delta * (weight * pag_scale)

                                        if not use_saliency_map:
                                                pag_blend_started = time.perf_counter()
                                                try:
                                                        denoised[i] += pag_x
                                                finally:
                                                        _record_cfg_timing(cfg_dict, "combine_pag_blend", time.perf_counter() - pag_blend_started)
                                                continue

                                        # Saliency Adaptive Noise Fusion arXiv.2311.10329v5
                                        sanf_started = time.perf_counter()
                                        try:
                                                model_delta = x_out[cond_index] - denoised_uncond[i]
                                                cfg_x = model_delta * (weight * cfg_scale)
                                                omega_rt = sanf_gaussian_blur3(cfg_x.abs()).float()
                                                omega_rs = sanf_gaussian_blur3(pag_x.abs()).float()
                                                soft_rt = torch.softmax(omega_rt, dim=0)
                                                soft_rs = torch.softmax(omega_rs, dim=0)

                                                m = torch.stack([soft_rt, soft_rs], dim=0) # 2 c h w
                                                _, argmax_indices = torch.max(m, dim=0)
                                                m1 = (argmax_indices == 0).to(dtype=cfg_x.dtype)
                                                sal_cfg = cfg_x * m1 + pag_x * (1 - m1)
                                                denoised[i] += sal_cfg
                                        finally:
                                                _record_cfg_timing(cfg_dict, "combine_sanf_blend", time.perf_counter() - sanf_started)
                                except Exception as e:
                                        logger.exception("Exception in combine_denoised_pass_conds_list - %s", e)

                return denoised
        return new_combine_denoised(*args)
