import logging
import time
from os import environ
import math

import modules.scripts as scripts
import gradio as gr

from modules import script_callbacks, shared
from modules.script_callbacks import CFGDenoiserParams
from modules.processing import StableDiffusionProcessing

from scripts.ui_wrapper import UIWrapper
from scripts.incant_utils import module_hooks

import torch
from torch.nn import functional as F

logger = logging.getLogger(__name__)
logger.setLevel(environ.get("SD_WEBUI_LOG_LEVEL", logging.INFO))

"""
An unofficial implementation of "Smoothed Energy Guidance for SDXL" for Automatic1111 WebUI.

@article{hong2024smoothed,
  title={Smoothed Energy Guidance: Guiding Diffusion Models with Reduced Energy Curvature of Attention},
  author={Hong, Susung},
  journal={arXiv preprint arXiv:2408.00760},
  year={2024}
}

Parts of the code are based off the author's official implementation at https://github.com/SusungHong/SEG-SDXL

Author: v0xie
GitHub URL: https://github.com/v0xie/sd-webui-incantations

"""


class SEGStateParams:
        def __init__(self):
                self.seg_active: bool = False      # SEG guidance scale
                self.seg_blur_sigma: float = 1.0
                self.seg_blur_threshold: float = 15.0 # 2^13 ~= 8192
                self.seg_start_step: int = 0
                self.seg_end_step: int = 150 
                self.crossattn_modules = [] # callable lambda
                self.openclaw_extension_timings = {}




def _record_seg_timing(seg_params, hook_name, elapsed):
        timings = seg_params.openclaw_extension_timings
        elapsed = float(elapsed)
        hook = timings.setdefault(hook_name, {"total_seconds": 0.0, "calls": 0})
        hook["total_seconds"] = round(float(hook.get("total_seconds") or 0.0) + elapsed, 6)
        hook["calls"] = int(hook.get("calls") or 0) + 1


def _merge_seg_timings(p, seg_params):
        if not seg_params.openclaw_extension_timings:
                return
        timings = getattr(p, "openclaw_extension_timings", None)
        if timings is None:
                timings = p.openclaw_extension_timings = {"total_seconds": 0.0, "extensions": {}}
        ext = timings["extensions"].setdefault("Incantations.SEGExtensionScript", {"total_seconds": 0.0, "calls": 0, "hooks": {}})
        for hook_name, hook in seg_params.openclaw_extension_timings.items():
                elapsed = float(hook.get("total_seconds") or 0.0)
                calls = int(hook.get("calls") or 0)
                timings["total_seconds"] = round(float(timings.get("total_seconds") or 0.0) + elapsed, 6)
                ext["total_seconds"] = round(float(ext.get("total_seconds") or 0.0) + elapsed, 6)
                ext["calls"] = int(ext.get("calls") or 0) + calls
                ext["hooks"][hook_name] = round(float(ext["hooks"].get(hook_name) or 0.0) + elapsed, 6)
        seg_params.openclaw_extension_timings = {}

class SEGExtensionScript(UIWrapper):
        def __init__(self):
                self.paste_field_names = []
                self.infotext_fields = []
                self._cfg_denoiser_callback = None
                self._seg_hook_handles = []

        # Extension title in menu UI
        def title(self) -> str:
                return "Smoothed Energy Guidance"

        # Decide to show menu in txt2img or img2img
        def show(self, is_img2img):
                return scripts.AlwaysVisible

        # Setup menu ui detail
        def setup_ui(self, is_img2img) -> list:
                with gr.Accordion('Smoothed Energy Guidance', open=False):
                        active = gr.Checkbox(value=False, default=False, label="SEG Active", elem_id='seg_active', info="Recommended to keep CFG Scale fixed at 3.0, use Sigma to adjust.")
                        with gr.Row():
                                seg_blur_sigma = gr.Slider(value = 11.0, minimum = 0.0, maximum = 11.0, step = 0.5, label="SEG Blur Sigma", elem_id = 'seg_blur_sigma', info="Exponential (2^n). Values >= 11 are infinite blur")
                        with gr.Row():
                                start_step = gr.Slider(value = 0, minimum = 0, maximum = 150, step = 1, label="SEG Start Step", elem_id = 'seg_start_step', info="")
                                end_step = gr.Slider(value = 150, minimum = 0, maximum = 150, step = 1, label="SEG End Step", elem_id = 'seg_end_step', info="")

                params = [active, seg_blur_sigma, start_step, end_step]
                                
                self.infotext_fields = [
                        (active, lambda d: gr.Checkbox.update(value='SEG Active' in d)),
                        (seg_blur_sigma, 'SEG Blur Sigma'),
                        (start_step, 'SEG Start Step'),
                        (end_step, 'SEG End Step'),
                ]
                for p in params:
                        p.do_not_save_to_config = True
                        self.paste_field_names.append(p.elem_id)

                return params

        def process_batch(self, p: StableDiffusionProcessing, *args, **kwargs):
               self.seg_process_batch(p, *args, **kwargs)

        def seg_process_batch(self, p: StableDiffusionProcessing, active, seg_blur_sigma, start_step, end_step, *args, **kwargs):
                # Clean previous hook handles before registering this batch.
                self.remove_all_hooks()

                active = getattr(p, "seg_active", active)
                if active is False:
                        return
                seg_blur_sigma = getattr(p, "seg_blur_sigma", seg_blur_sigma)
                if seg_blur_sigma == 0.0:
                        logger.info("SEG Blur Sigma is 0, skipping SEG")
                        return
                start_step = getattr(p, "seg_start_step", start_step)
                end_step = getattr(p, "seg_end_step", end_step)

                if active:
                        p.extra_generation_params.update({
                                "SEG Active": active,
                                "SEG Blur Sigma": seg_blur_sigma,
                                "SEG Start Step": start_step,
                                "SEG End Step": end_step,
                        })
                self.create_hook(p, active, seg_blur_sigma, start_step, end_step)

        def create_hook(self, p: StableDiffusionProcessing, active, seg_blur_sigma, start_step, end_step, *args, **kwargs):
                # Create a list of parameters for each concept
                seg_params = SEGStateParams()

                # Add to p's incant_cfg_params
                if not hasattr(p, 'incant_cfg_params'):
                        logger.error("No incant_cfg_params found in p")
                p.incant_cfg_params['seg_params'] = seg_params
                
                seg_params.seg_active = active 
                seg_params.seg_blur_sigma = seg_blur_sigma
                seg_params.seg_blur_threshold = 10.5
                seg_params.seg_start_step = start_step
                seg_params.seg_end_step = end_step

                # Get all the qv modules
                self_attn_modules = self.get_cross_attn_modules()
                if len(self_attn_modules) == 0:
                        logger.error("No self attention modules found, cannot proceed with SEG")
                        return
                seg_params.crossattn_modules = self_attn_modules

                self.remove_callbacks()
                cfg_denoise_lambda = lambda callback_params: self.on_cfg_denoiser_callback(callback_params, seg_params)
                self._cfg_denoiser_callback = cfg_denoise_lambda
                if seg_params.seg_active:
                        self.ready_hijack_forward(seg_params.crossattn_modules, seg_blur_sigma, seg_params.seg_blur_threshold, p.height, p.width)

                logger.debug('Hooked callbacks')
                script_callbacks.on_cfg_denoiser(cfg_denoise_lambda)

        def postprocess_batch(self, p, *args, **kwargs):
                self.seg_postprocess_batch(p, *args, **kwargs)

        def seg_postprocess_batch(self, p, active, seg_blur_sigma, start_step, end_step, *args, **kwargs):
                seg_params = getattr(p, "incant_cfg_params", {}).get("seg_params") if getattr(p, "incant_cfg_params", None) else None
                if seg_params is not None:
                        _merge_seg_timings(p, seg_params)
                self.remove_all_hooks()
                self.remove_callbacks()
                logger.debug('Removed SEG hooks and callbacks')
                active = getattr(p, "seg_active", active)
                if active is False:
                        return

        def remove_callbacks(self):
                if self._cfg_denoiser_callback is not None:
                        script_callbacks.remove_callbacks_for_function(self._cfg_denoiser_callback)
                        self._cfg_denoiser_callback = None

        def remove_all_hooks(self):
                for handle in self._seg_hook_handles:
                        handle.remove()
                self._seg_hook_handles = []

                self_attn_modules = self.get_cross_attn_modules()
                for module in self_attn_modules:
                        module_hooks.modules_remove_field(module.to_q, 'seg_enable')
                        module_hooks.modules_remove_field(module.to_q, 'seg_parent_module')

        def unhook_callbacks(self, seg_params: SEGStateParams = None):
                self.remove_all_hooks()
                self.remove_callbacks()

        def ready_hijack_forward(self, selfattn_modules, seg_blur_sigma, seg_blur_threshold, height, width):
                for module in selfattn_modules:
                        module_hooks.modules_add_field(module.to_q, 'seg_enable', False)
                        module_hooks.modules_add_field(module.to_q, 'seg_parent_module', [module])

                def seg_to_q_hook(module, input, kwargs, output):
                        if not hasattr(module, 'seg_enable'):
                                return
                        if not module.seg_enable:
                                return
                        batch_size, seq_len, inner_dim = input[0].shape
                        h = module.seg_parent_module[0].heads
                        head_dim = inner_dim // h

                        module_attn_size = seq_len
                        downscale_h = max(1, int((module_attn_size * (height / width)) ** 0.5))
                        while downscale_h > 1 and module_attn_size % downscale_h != 0:
                                downscale_h -= 1
                        downscale_w = module_attn_size // downscale_h
                        if downscale_h * downscale_w != module_attn_size:
                                logger.warning("SEG could not derive exact attention shape for seq_len=%s, image=%sx%s; skipping blur", seq_len, height, width)
                                return

                        # actual sigma value is calculated as 2 ^ sigma
                        is_inf_blur = seg_blur_sigma > seg_blur_threshold
                        blur_sigma_exp = 2 ** seg_blur_sigma
                        kernel_size = math.ceil(6 * blur_sigma_exp) + 1 - math.ceil(6 * blur_sigma_exp) % 2

                        # SEG mutates only the conditional half of A1111's paired
                        # CFG attention batch. A1111 does not guarantee cond+uncond
                        # are always evaluated together: token-length mismatches,
                        # disabled batch-cond-uncond, skip-uncond paths, and hidden
                        # extension passes can call attention with a singleton or
                        # otherwise unpaired batch. torch.chunk(2) may return only one
                        # chunk for batch=1, and even-sized unpaired batches can be
                        # semantically wrong, so skip unless this forward looks like a
                        # valid paired batch.
                        output_batch = output.shape[0]
                        if output_batch < 2 or output_batch % 2 != 0 or batch_size != output_batch:
                                logger.debug(
                                        "SEG skipping unpaired to_q batch: input_batch=%s output_batch=%s seq_len=%s",
                                        batch_size,
                                        output_batch,
                                        seq_len,
                                )
                                return

                        half_batch = output_batch // 2
                        q_uncond, q = output.split(half_batch, dim=0)
                        q = q.view(half_batch, -1, h, head_dim).transpose(1, 2) # (batch, num_heads, seq_len, head_dim)
                        q = q.permute(0, 1, 3, 2).reshape(half_batch * h, head_dim, downscale_h, downscale_w) # (batch * num_heads, head_dim, height, width)

                        if is_inf_blur:
                                q = gaussian_blur_inf(q, 1.0, blur_sigma_exp)
                        else:
                                q = gaussian_blur_2d(q, kernel_size, blur_sigma_exp)

                        q = q.reshape(half_batch, h, head_dim, downscale_h * downscale_w) # (batch, num_heads, head_dim, seq_len)
                        q = q.view(half_batch, h * head_dim, seq_len).transpose(1, 2) # (batch, inner_dim, seq_len)
                        q = torch.cat((q_uncond, q), dim=0)

                        return q

                # Create hooks and keep RemovableHandles so cleanup does not need
                # to rewrite PyTorch hook tables globally.
                for module in selfattn_modules:
                        self._seg_hook_handles.append(module_hooks.module_add_forward_hook(module.to_q, seg_to_q_hook, hook_type="forward", with_kwargs=True))

        def get_middle_block_modules(self):
                """ Get all attention modules from the middle block 
                Refere to page 22 of the SEG paper, Appendix A.2
                
                """
                middle_block_modules = module_hooks.get_modules(
                        network_layer_name_filter = 'middle_block_',
                        module_name_filter = 'CrossAttention'
                )
                middle_block_modules = [m for m in middle_block_modules if 'attn1' in m.network_layer_name]
                return middle_block_modules

        def get_cross_attn_modules(self):
                """ Get all cross attention modules """
                return self.get_middle_block_modules()

        def on_cfg_denoiser_callback(self, params: CFGDenoiserParams, seg_params: SEGStateParams):
                started = time.perf_counter()
                try:
                        self._on_cfg_denoiser_callback(params, seg_params)
                finally:
                        _record_seg_timing(seg_params, "cfg_denoiser_callback", time.perf_counter() - started)

        def _on_cfg_denoiser_callback(self, params: CFGDenoiserParams, seg_params: SEGStateParams):
                # Keep SEG hooks installed for the batch; per-step work only toggles
                # the hook flag. Removing hooks here disables SEG entirely.
                if not seg_params.seg_active:
                        return

                in_interval = seg_params.seg_start_step <= params.sampling_step <= seg_params.seg_end_step
                should_enable = in_interval and getattr(shared.opts, 'batch_cond_uncond', False)
                if not should_enable:
                        logger.debug(
                                "SEG disabled for this step: in_interval=%s batch_cond_uncond=%s",
                                in_interval,
                                getattr(shared.opts, 'batch_cond_uncond', False),
                        )
                for module in seg_params.crossattn_modules:
                        if hasattr(module.to_q, 'seg_enable'):
                                module.to_q.seg_enable = should_enable

        def get_xyz_axis_options(self) -> dict:
                xyz_grid = [x for x in scripts.scripts_data if x.script_class.__module__ in ("xyz_grid.py", "scripts.xyz_grid")][0].module
                extra_axis_options = {
                        xyz_grid.AxisOption("[SEG] Active", str, seg_apply_override('seg_active', boolean=True), choices=xyz_grid.boolean_choice(reverse=True)),
                        xyz_grid.AxisOption("[SEG] SEG Blur Sigma", float, seg_apply_field("seg_blur_sigma")),
                        xyz_grid.AxisOption("[SEG] SEG Start Step", int, seg_apply_field("seg_start_step")),
                        xyz_grid.AxisOption("[SEG] SEG End Step", int, seg_apply_field("seg_end_step")),
                }
                return extra_axis_options



# XYZ Plot
# Based on @mcmonkey4eva's XYZ Plot implementation here: https://github.com/mcmonkeyprojects/sd-dynamic-thresholding/blob/master/scripts/dynamic_thresholding.py
def seg_apply_override(field, boolean: bool = False):
    def fun(p, x, xs):
        if boolean:
            x = True if x.lower() == "true" else False
        setattr(p, field, x)
        if not hasattr(p, "seg_active"):
                setattr(p, "seg_active", True)
        if 'cfg_interval_' in field and not hasattr(p, "cfg_interval_enable"):
            setattr(p, "cfg_interval_enable", True)
    return fun


def seg_apply_field(field):
    def fun(p, x, xs):
        if not hasattr(p, "seg_active"):
                setattr(p, "seg_active", True)
        setattr(p, field, x)
    return fun


# Gaussian blur
# taken from https://github.com/SusungHong/SEG-SDXL/blob/master/pipeline_seg.py
_GAUSSIAN_KERNEL_CACHE = {}


def gaussian_blur_2d(img, kernel_size, sigma):
        height = img.shape[-1]
        kernel_size = min(kernel_size, height - (height % 2 - 1))
        channels = img.shape[-3]
        key = (img.device, img.dtype, channels, kernel_size, float(sigma))
        kernel2d = _GAUSSIAN_KERNEL_CACHE.get(key)
        if kernel2d is None:
                ksize_half = (kernel_size - 1) * 0.5
                x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size, device=img.device, dtype=torch.float32)
                pdf = torch.exp(-0.5 * (x / sigma).pow(2))
                x_kernel = (pdf / pdf.sum()).to(dtype=img.dtype)
                base_kernel = torch.mm(x_kernel[:, None], x_kernel[None, :])
                kernel2d = base_kernel.expand(channels, 1, base_kernel.shape[0], base_kernel.shape[1]).contiguous()
                _GAUSSIAN_KERNEL_CACHE[key] = kernel2d

        padding = [kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2]
        img = F.pad(img, padding, mode="reflect")
        img = F.conv2d(img, kernel2d, groups=channels)

        return img


def gaussian_blur_inf(img, kernel_size, sigma):
        return img.mean(dim=(-2, -1), keepdim=True).expand_as(img)