import logging
from os import environ
import functools
import math

from modules import headless_ui as gr

from modules import script_callbacks, shared
from modules.script_callbacks import CFGDenoiserParams
from modules.processing import StableDiffusionProcessing

from scripts.ui_wrapper import UIWrapper, cond_crossattn, xyz_field_setter
from scripts.incant_utils import module_hooks, timing

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
                self.seg_active: bool = False
                self.seg_blur_sigma: float = 1.0
                # Blur sigmas above this are the infinite blur (a global mean); the UI's 11.0 maximum is one.
                self.seg_blur_threshold: float = 10.5
                self.seg_start_step: int = 0
                self.seg_end_step: int = 150
                self.crossattn_modules = [] # the hooked middle-block self-attention modules
                self.openclaw_extension_timings = {}
                # (cond rows, uncond rows) of the current step's CFG batch, set by the cfg_denoiser callback.
                self.cfg_rows = None


def cfg_row_counts(text_cond, text_uncond):
        """(cond rows, uncond rows) of A1111's CFG batch from CFGDenoiserParams' conditioning (tensor or SDXL dict)."""
        return int(cond_crossattn(text_cond).shape[0]), int(cond_crossattn(text_uncond).shape[0])


def seg_attention_grid(seq_len, height, width):
        """(rows, cols) of the token grid of a UNet attention layer with seq_len tokens (seq index row * cols + col).

        The UNet sees the (height // 8, width // 8) latent and every downsample is a stride-2 conv with padding 1,
        i.e. ceil(n / 2) per side, so the layer's grid is the level of that chain with seq_len tokens. Sizes that are
        not multiples of 64 make that grid's aspect differ from height / width (1080x1920: 34x60, not 30x68). A latent
        of another size (e.g. a hires pass) falls back to the divisor pair of seq_len closest to height / width.
        """
        rows, cols = max(1, height // 8), max(1, width // 8)
        while rows * cols > seq_len:
                rows, cols = (rows + 1) // 2, (cols + 1) // 2
                if rows * cols == 1:
                        break
        if rows * cols == seq_len:
                return rows, cols
        aspect = math.log(height / width)
        rows = min((r for r in range(1, seq_len + 1) if seq_len % r == 0), key=lambda r: abs(math.log(r * r / seq_len) - aspect))
        return rows, seq_len // rows


def _blur_seg_uncond_queries(output, n_cond, *, heads, head_dim, downscale_h, downscale_w, kernel_size, sigma, is_inf_blur):
        """Blur the queries of the uncond rows ``[n_cond:]`` of a full CFG batch (legacy GB10 SEG, tuned behavior).

        A1111 orders the CFG batch [cond rows, uncond rows]; legacy SEG blurs the uncond rows, so CFG pushes
        away from the smoothed-attention uncond prediction. With one cond row per uncond row (no AND prompts)
        this is exactly the tail half of the batch. n_cond counts to_q rows (cond rows x hypertile tiles).
        output is the to_q result (batch, H*W, heads*head_dim) with seq index h*W + w. It is a fresh
        tensor owned by the hook, so the blurred rows are written into it in place (one rounding copy,
        no concatenated copy of the batch) and it is returned.
        """
        n_blur = output.shape[0] - n_cond
        q_blur = output[n_cond:]
        if is_inf_blur:
                seq_len = downscale_h * downscale_w
                q_blur = q_blur.view(n_blur, -1, heads, head_dim).transpose(1, 2)
                q_blur = q_blur.permute(0, 1, 3, 2).reshape(
                        n_blur * heads, head_dim, downscale_h, downscale_w
                )
                q_blur = gaussian_blur_inf(q_blur)
                q_blur = q_blur.reshape(n_blur, heads, head_dim, seq_len)
                q_blur = q_blur.view(n_blur, heads * head_dim, seq_len).transpose(1, 2)
        else:
                q_blur = _gaussian_blur_queries_fp32(q_blur, downscale_h, downscale_w, kernel_size, sigma)
        # Every value is computed from output before this write; copy_ rounds like .to(output.dtype).
        output[n_cond:].copy_(q_blur)
        return output


class SEGExtensionScript(UIWrapper):
        def __init__(self):
                super().__init__()
                self._seg_hooked_modules = []

        # Setup menu ui detail
        def setup_ui(self, is_img2img) -> list:
                with gr.Accordion('Smoothed Energy Guidance', open=False):
                        active = gr.Checkbox(value=False, default=False, label="SEG Active", elem_id='seg_active', info="Recommended to keep CFG Scale fixed at 3.0, use Sigma to adjust.")
                        with gr.Row():
                                seg_blur_sigma = gr.Slider(value = 11.0, minimum = 0.0, maximum = 11.0, step = 0.5, label="SEG Blur Sigma", elem_id = 'seg_blur_sigma', info="Exponential (2^n). Values >= 11 are infinite blur")
                        with gr.Row():
                                start_step = gr.Slider(value = 0, minimum = 0, maximum = 150, step = 1, label="SEG Start Step", elem_id = 'seg_start_step', info="")
                                end_step = gr.Slider(value = 150, minimum = 0, maximum = 150, step = 1, label="SEG End Step", elem_id = 'seg_end_step', info="")

                self.infotext_fields = [
                        (active, lambda d: gr.Checkbox.update(value='SEG Active' in d)),
                        (seg_blur_sigma, 'SEG Blur Sigma'),
                        (start_step, 'SEG Start Step'),
                        (end_step, 'SEG End Step'),
                ]
                return [active, seg_blur_sigma, start_step, end_step]

        def process_batch(self, p: StableDiffusionProcessing, *args, **kwargs):
               self.seg_process_batch(p, *args, **kwargs)

        def seg_process_batch(self, p: StableDiffusionProcessing, active, seg_blur_sigma, start_step, end_step, *args, **kwargs):
                # Clean previous hook handles and callbacks before registering this batch (a failed
                # generation skips postprocess_batch, and an inactive batch registers nothing new).
                self.remove_all_hooks()
                self.remove_callbacks()

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

        def create_hook(self, p: StableDiffusionProcessing, active, seg_blur_sigma, start_step, end_step):
                seg_params = SEGStateParams()
                p.incant_cfg_params['seg_params'] = seg_params

                seg_params.seg_active = active
                seg_params.seg_blur_sigma = seg_blur_sigma
                seg_params.seg_start_step = start_step
                seg_params.seg_end_step = end_step

                # Get all the qv modules
                self_attn_modules = self.get_cross_attn_modules()
                if len(self_attn_modules) == 0:
                        # The request asked for SEG and its infotext already says "SEG Active"; never render
                        # without it. (module_hooks.get_modules returns [] for a model without a layer mapping.)
                        raise RuntimeError("SEG: no middle-block self-attention modules found on the loaded model")
                seg_params.crossattn_modules = self_attn_modules

                def cfg_denoise_callback(callback_params):
                        return self.on_cfg_denoiser_callback(callback_params, seg_params)

                if seg_params.seg_active:
                        self.ready_hijack_forward(seg_params, seg_blur_sigma, p.height, p.width)

                logger.debug('Hooked callbacks')
                script_callbacks.on_cfg_denoiser(self.track_callback(cfg_denoise_callback))

        def postprocess_batch(self, p, *args, **kwargs):
                seg_params = (getattr(p, "incant_cfg_params", None) or {}).get("seg_params")
                if seg_params is not None:
                        timing.merge_into_processing(p, "Incantations.SEGExtensionScript", seg_params.openclaw_extension_timings)
                        seg_params.openclaw_extension_timings = {}
                self.remove_all_hooks()
                self.remove_callbacks()
                logger.debug('Removed SEG hooks and callbacks')

        def remove_all_hooks(self):
                self.remove_hook_handles()
                for module in self._seg_hooked_modules:
                        module_hooks.modules_remove_field(module.to_q, 'seg_enable')
                        module_hooks.modules_remove_field(module.to_q, 'seg_parent_module')
                self._seg_hooked_modules = []

        def ready_hijack_forward(self, seg_params: SEGStateParams, seg_blur_sigma, height, width):
                selfattn_modules = seg_params.crossattn_modules
                self._seg_hooked_modules = list(selfattn_modules)
                is_inf_blur = seg_blur_sigma > seg_params.seg_blur_threshold
                blur_sigma_exp = 2 ** seg_blur_sigma
                geometry_cache = {}
                for module in selfattn_modules:
                        module_hooks.modules_add_field(module.to_q, 'seg_enable', False)
                        module_hooks.modules_add_field(module.to_q, 'seg_parent_module', [module])

                # The fields above are added before this hook is installed and removed only after its handles.
                def seg_to_q_hook(module, input, kwargs, output):
                        if not module.seg_enable:
                                return
                        batch_size, seq_len, inner_dim = input[0].shape
                        h = module.seg_parent_module[0].heads
                        head_dim = inner_dim // h

                        cache_key = (seq_len, height, width)
                        geometry = geometry_cache.get(cache_key)
                        if geometry is None:
                                downscale_h, downscale_w = seg_attention_grid(seq_len, height, width)
                                kernel_size = math.ceil(6 * blur_sigma_exp) + 1 - math.ceil(6 * blur_sigma_exp) % 2
                                geometry = (downscale_h, downscale_w, kernel_size)
                                geometry_cache[cache_key] = geometry
                        downscale_h, downscale_w, kernel_size = geometry

                        # SEG blurs the uncond rows of A1111's full CFG batch [cond rows, uncond rows].
                        # A1111 does not always evaluate that batch in one call: token-length mismatches
                        # without padding, disabled batch-cond-uncond, skip-uncond (NGMS / skip early CFG)
                        # and hidden extension passes call the UNet on cond-only, uncond-only or partial
                        # batches, and AND prompts give more cond rows than uncond rows. Splitting such a
                        # batch in half would blur cond rows, i.e. guide toward the smoothed prediction, so
                        # blur only a call that is this step's full CFG batch. Hypertile, when it tiles this
                        # layer, makes every row `tiles` consecutive rows ("(b nh nw)"), so the batch is a
                        # whole multiple of the CFG rows and the uncond rows start at n_cond * tiles.
                        output_batch = output.shape[0]
                        cfg_rows = seg_params.cfg_rows
                        cfg_batch = cfg_rows[0] + cfg_rows[1] if cfg_rows is not None else 0
                        if (
                                cfg_rows is None
                                or cfg_rows[0] < 1
                                or cfg_rows[1] < 1
                                or output_batch % cfg_batch != 0
                                or batch_size != output_batch
                        ):
                                logger.debug(
                                        "SEG skipping to_q batch that is not the full CFG batch: input_batch=%s output_batch=%s cfg_rows=%s seq_len=%s",
                                        batch_size,
                                        output_batch,
                                        cfg_rows,
                                        seq_len,
                                )
                                return

                        return _blur_seg_uncond_queries(
                                output,
                                cfg_rows[0] * (output_batch // cfg_batch),
                                heads=h,
                                head_dim=head_dim,
                                downscale_h=downscale_h,
                                downscale_w=downscale_w,
                                kernel_size=kernel_size,
                                sigma=blur_sigma_exp,
                                is_inf_blur=is_inf_blur,
                        )

                # Keep RemovableHandles so cleanup does not need to rewrite PyTorch hook tables globally.
                for module in selfattn_modules:
                        self.add_forward_hook(module.to_q, seg_to_q_hook)

        def get_cross_attn_modules(self):
                """ The middle block's self-attention (attn1) modules """
                middle_block_modules = module_hooks.get_modules(
                        network_layer_name_filter = 'middle_block_',
                        module_name_filter = 'CrossAttention'
                )
                return [m for m in middle_block_modules if 'attn1' in m.network_layer_name]

        def on_cfg_denoiser_callback(self, params: CFGDenoiserParams, seg_params: SEGStateParams):
                with timing.timed(seg_params.openclaw_extension_timings, "cfg_denoiser_callback"):
                        self._on_cfg_denoiser_callback(params, seg_params)

        def _on_cfg_denoiser_callback(self, params: CFGDenoiserParams, seg_params: SEGStateParams):
                # Keep SEG hooks installed for the batch; per-step work only toggles
                # the hook flag. Removing hooks here disables SEG entirely.
                if not seg_params.seg_active:
                        return

                in_interval = seg_params.seg_start_step <= params.sampling_step <= seg_params.seg_end_step
                should_enable = in_interval and getattr(shared.opts, 'batch_cond_uncond', False)
                seg_params.cfg_rows = cfg_row_counts(params.text_cond, params.text_uncond) if should_enable else None
                if not should_enable:
                        logger.debug(
                                "SEG disabled for this step: in_interval=%s batch_cond_uncond=%s",
                                in_interval,
                                getattr(shared.opts, 'batch_cond_uncond', False),
                        )
                for module in seg_params.crossattn_modules:
                        module.to_q.seg_enable = should_enable

        def get_xyz_axis_options(self, xyz_grid) -> list:
                return [
                        xyz_grid.AxisOption("[SEG] Active", str, xyz_field_setter('seg_active', 'seg_active', boolean=True), choices=xyz_grid.boolean_choice(reverse=True)),
                        xyz_grid.AxisOption("[SEG] SEG Blur Sigma", float, xyz_field_setter("seg_blur_sigma", 'seg_active')),
                        xyz_grid.AxisOption("[SEG] SEG Start Step", int, xyz_field_setter("seg_start_step", 'seg_active')),
                        xyz_grid.AxisOption("[SEG] SEG End Step", int, xyz_field_setter("seg_end_step", 'seg_active')),
                ]


# Gaussian blur
# Separable form of the reflect-padded depthwise blur in
# https://github.com/SusungHong/SEG-SDXL/blob/master/pipeline_seg.py
@functools.lru_cache(maxsize=16)
def _gaussian_blur_operator(n, kernel_size, sigma, device):
        """(n, n) fp32 matrix A with A @ x == 1-D Gaussian conv of F.pad(x, mode='reflect') along x's first axis.

        Built by blurring the identity, so taps that reflect onto the same source
        pixel are summed exactly as in the padded convolution. The tensor is
        shared by the cache; callers must not mutate it. It is built on the CPU
        with CPU autocast off: the first call happens inside the UNet forward,
        and an active CPU autocast would otherwise run the conv in bf16 and
        cache a bf16 operator.
        """
        with torch.autocast("cpu", enabled=False):
                ksize_half = (kernel_size - 1) * 0.5
                x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size, dtype=torch.float32)
                pdf = torch.exp(-0.5 * (x / sigma).pow(2))
                taps = pdf / pdf.sum()
                pad = kernel_size // 2
                unit_pixels = F.pad(torch.eye(n, dtype=torch.float32).unsqueeze(1), [pad, pad], mode="reflect")
                # Row j of the conv output is the response to unit pixel j, i.e. column j of A.
                operator = F.conv1d(unit_pixels, taps.view(1, 1, kernel_size)).squeeze(1).T.contiguous()
        return operator.to(device)


def _gaussian_blur_queries_fp32(q, height, width, kernel_size, sigma):
        """Reflect-padded Gaussian blur of (batch, height*width, channels) queries over their (height, width) grid.

        Returns the fp32 result; the caller rounds it to the query dtype. Same math as the reference
        k x k outer-product depthwise conv, done as two small GEMMs on the native query layout without
        permute copies. Taps and accumulation are fp32, where the reference rounded its 2-D taps to the
        query dtype; with TF32 matmul (devices.enable_tf32) the taps still keep more mantissa bits than bf16.
        """
        min_spatial = min(height, width)
        kernel_size = min(kernel_size, min_spatial - (min_spatial % 2 - 1))
        blur_h = _gaussian_blur_operator(height, kernel_size, float(sigma), q.device)
        blur_w = _gaussian_blur_operator(width, kernel_size, float(sigma), q.device)
        batch, _, channels = q.shape
        # Autocast would otherwise run these matmuls in bf16, rounding the taps and the intermediate.
        with torch.autocast(q.device.type, enabled=False):
                q_blur = blur_h @ q.float().reshape(batch, height, width * channels)
                q_blur = blur_w @ q_blur.view(batch * height, width, channels)
        return q_blur.view(batch, height * width, channels)


def gaussian_blur_inf(img):
        """The infinite-sigma blur: every pixel becomes the spatial mean of its channel."""
        return img.mean(dim=(-2, -1), keepdim=True).expand_as(img)
