import logging
import time
from os import environ
import modules.scripts as scripts
import gradio as gr
from scripts.ui_wrapper import UIWrapper
from modules import script_callbacks
from modules.script_callbacks import CFGDenoiserParams, CFGDenoisedParams
from modules.processing import StableDiffusionProcessing
from modules.sd_samplers_cfg_denoiser import catenate_conds, subscript_cond
from modules import shared
from scripts.incant_utils import module_hooks

import math
import torch
from torch.nn import functional as F


logger = logging.getLogger(__name__)
logger.setLevel(environ.get("SD_WEBUI_LOG_LEVEL", logging.INFO))

"""
An unofficial implementation of "Self-Rectifying Diffusion Sampling with Perturbed-Attention Guidance" for Automatic1111 WebUI.

@misc{ahn2024selfrectifying,
      title={Self-Rectifying Diffusion Sampling with Perturbed-Attention Guidance}, 
      author={Donghoon Ahn and Hyoungwon Cho and Jaewon Min and Wooseok Jang and Jungwoo Kim and SeonHwa Kim and Hyun Hee Park and Kyong Hwan Jin and Seungryong Kim},
      year={2024},
      eprint={2403.17377},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

Include noise interval for CFG and PAG guidance in the sampling process from "Applying Guidance in a Limited Interval Improves
Sample and Distribution Quality in Diffusion Models"

@misc{kynkäänniemi2024applying,
      title={Applying Guidance in a Limited Interval Improves Sample and Distribution Quality in Diffusion Models}, 
      author={Tuomas Kynkäänniemi and Miika Aittala and Tero Karras and Samuli Laine and Timo Aila and Jaakko Lehtinen},
      year={2024},
      eprint={2404.07724},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

Include CFG schedulers from "Analysis of Classifier-Free Guidance Weight Schedulers"

@misc{wang2024analysis,
      title={Analysis of Classifier-Free Guidance Weight Schedulers}, 
      author={Xi Wang and Nicolas Dufour and Nefeli Andreou and Marie-Paule Cani and Victoria Fernandez Abrevaya and David Picard and Vicky Kalogeiton},
      year={2024},
      eprint={2404.13040},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

Saliency-adaptive noise fusion from arXiv:2311.10329 "High-fidelity Person-centric Subject-to-Image Synthesis"
@misc{wang2024highfidelity,
      title={High-fidelity Person-centric Subject-to-Image Synthesis}, 
      author={Yibin Wang and Weizhong Zhang and Jianwei Zheng and Cheng Jin},
      year={2024},
      eprint={2311.10329},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

Author: v0xie
GitHub URL: https://github.com/v0xie/sd-webui-incantations

"""


SCHEDULES = [
        'Constant',
        'Clamp-Linear (c=4.0)',
        'Clamp-Linear (c=2.0)',
        'Clamp-Linear (c=1.0)',
        'Linear',
        'Inverse-Linear',
        'Cosine',
        'Clamp-Cosine (c=4.0)',
        'Clamp-Cosine (c=2.0)',
        'Clamp-Cosine (c=1.0)',
        'Sine',
        'Interval',
        'PCS (s=0.01)',
        'PCS (s=0.1)',
        'PCS (s=1.0)',
        'PCS (s=2.0)',
        'PCS (s=4.0)',
]


class PAGStateParams:
        def __init__(self):
                self.pag_active: bool = False      # PAG guidance scale
                self.pag_sanf: bool = False # saliency-adaptive noise fusion, handled in cfg_combiner
                self.pag_scale: int = -1      # PAG guidance scale
                self.pag_start_step: int = 0
                self.pag_end_step: int = 150 
                self.cfg_interval_enable: bool = False
                self.cfg_interval_schedule: str = 'Constant'
                self.cfg_interval_low: float = 0
                self.cfg_interval_high: float = 50.0
                self.cfg_interval_scheduled_value: float = 7.0
                self.step : int = 0 
                self.max_sampling_step : int = 1 
                self.guidance_scale: int = -1 # CFG
                self.current_noise_level: float = 100.0
                self.x_in = None
                self.text_cond = None
                self.image_cond = None
                self.sigma = None
                self.text_uncond = None
                self.make_condition_dict = None # callable lambda
                self.crossattn_modules = [] # callable lambda
                self.pag_x_out = None
                self.batch_size = -1      # Batch size
                self.openclaw_extension_timings = {}


def cond_crossattn(cond):
        """Return the cross-attention tensor for plain or SDXL dict conditioning."""
        return cond.get('crossattn') if isinstance(cond, dict) else cond


def cond_batch_size(cond):
        tensor = cond_crossattn(cond)
        if tensor is None:
                raise RuntimeError("PAG conditioning is missing a cross-attention tensor")
        return tensor.shape[0]


def cond_token_count(cond):
        tensor = cond_crossattn(cond)
        if tensor is None:
                raise RuntimeError("PAG conditioning is missing a cross-attention tensor")
        return tensor.shape[1]


def _seg_to_q_modules():
        """Return SEG-managed to_q modules currently installed on the shared model.

        PAG runs an internal denoising pass to compute perturbed-attention output.
        That pass mirrors A1111 batching and may execute cond/uncond separately; SEG
        should not leak into it because SEG assumes a paired CFG attention batch.
        """
        try:
                mapping = getattr(shared.sd_model, 'network_layer_mapping', {}) or {}
        except Exception:
                return []

        modules = []
        seen = set()
        for parent in mapping.values():
                to_q = getattr(parent, 'to_q', None)
                if to_q is None or not hasattr(to_q, 'seg_enable'):
                        continue
                ident = id(to_q)
                if ident in seen:
                        continue
                seen.add(ident)
                modules.append(to_q)
        return modules


def _suspend_seg_for_pag_hidden_pass():
        saved = []
        for to_q in _seg_to_q_modules():
                saved.append((to_q, getattr(to_q, 'seg_enable', False)))
                setattr(to_q, 'seg_enable', False)
        return saved


def _restore_seg_after_pag_hidden_pass(saved):
        for to_q, enabled in saved:
                try:
                        setattr(to_q, 'seg_enable', enabled)
                except Exception:
                        pass


def pag_inner_model_x_out(inner_model, x_in, sigma_in, tensor, uncond, image_cond_in, make_condition_dict, batch_size):
        """Run PAG's hidden denoising pass with A1111's cond/uncond batching rules.

        A1111 only concatenates positive and negative conditioning when their
        token lengths match or prompt-padding is enabled. Otherwise it runs the
        positive batches and the unconditional batch separately. PAG's extra
        pass must mirror that split path; blindly calling ``catenate_conds`` for
        SDXL prompt/negative pairs with different token counts makes the
        callback fail before ``pag_x_out`` can be produced.
        """
        if cond_token_count(tensor) == cond_token_count(uncond):
                cond_in = catenate_conds([tensor, uncond])
                return inner_model(x_in, sigma_in, cond=make_condition_dict(cond_in, image_cond_in))

        x_out = torch.zeros_like(x_in)
        denoise_batch_size = max(1, batch_size * 2 if shared.opts.batch_cond_uncond else batch_size)
        for batch_offset in range(0, cond_batch_size(tensor), denoise_batch_size):
                a = batch_offset
                b = min(a + denoise_batch_size, cond_batch_size(tensor))
                x_out[a:b] = inner_model(
                        x_in[a:b],
                        sigma_in[a:b],
                        cond=make_condition_dict(subscript_cond(tensor, a, b), image_cond_in[a:b]),
                )

        uncond_count = cond_batch_size(uncond)
        x_out[-uncond_count:] = inner_model(
                x_in[-uncond_count:],
                sigma_in[-uncond_count:],
                cond=make_condition_dict(uncond, image_cond_in[-uncond_count:]),
        )
        return x_out




def _record_pag_timing(pag_params, hook_name, elapsed):
        timings = pag_params.openclaw_extension_timings
        elapsed = float(elapsed)
        hook = timings.setdefault(hook_name, {"total_seconds": 0.0, "calls": 0})
        hook["total_seconds"] = round(float(hook.get("total_seconds") or 0.0) + elapsed, 6)
        hook["calls"] = int(hook.get("calls") or 0) + 1


def _record_pag_detail(pag_params, detail_name, elapsed):
        timings = pag_params.openclaw_extension_timings
        elapsed = float(elapsed)
        details = timings.setdefault("details", {})
        detail = details.setdefault(detail_name, {"total_seconds": 0.0, "calls": 0})
        detail["total_seconds"] = round(float(detail.get("total_seconds") or 0.0) + elapsed, 6)
        detail["calls"] = int(detail.get("calls") or 0) + 1


def _merge_pag_timings(p, pag_params):
        if not pag_params.openclaw_extension_timings:
                return
        timings = getattr(p, "openclaw_extension_timings", None)
        if timings is None:
                timings = p.openclaw_extension_timings = {"total_seconds": 0.0, "extensions": {}}
        ext = timings["extensions"].setdefault("Incantations.PAGExtensionScript", {"total_seconds": 0.0, "calls": 0, "hooks": {}})
        detail_timings = pag_params.openclaw_extension_timings.pop("details", {})
        for hook_name, hook in pag_params.openclaw_extension_timings.items():
                elapsed = float(hook.get("total_seconds") or 0.0)
                calls = int(hook.get("calls") or 0)
                timings["total_seconds"] = round(float(timings.get("total_seconds") or 0.0) + elapsed, 6)
                ext["total_seconds"] = round(float(ext.get("total_seconds") or 0.0) + elapsed, 6)
                ext["calls"] = int(ext.get("calls") or 0) + calls
                ext["hooks"][hook_name] = round(float(ext["hooks"].get(hook_name) or 0.0) + elapsed, 6)
        if detail_timings:
                ext["details"] = ext.get("details", {})
                for detail_name, detail in detail_timings.items():
                        elapsed = float(detail.get("total_seconds") or 0.0)
                        calls = int(detail.get("calls") or 0)
                        existing = ext["details"].setdefault(detail_name, {"total_seconds": 0.0, "calls": 0})
                        existing["total_seconds"] = round(float(existing.get("total_seconds") or 0.0) + elapsed, 6)
                        existing["calls"] = int(existing.get("calls") or 0) + calls
        pag_params.openclaw_extension_timings = {}

class PAGExtensionScript(UIWrapper):
        def __init__(self):
                self._cfg_denoiser_callback = None
                self._cfg_denoised_callback = None
                self._pag_hook_handles = []

        # Extension title in menu UI
        def title(self) -> str:
                return "Perturbed Attention Guidance"

        # Decide to show menu in txt2img or img2img
        def show(self, is_img2img):
                return scripts.AlwaysVisible

        # Setup menu ui detail
        def setup_ui(self, is_img2img) -> list:
                with gr.Accordion('Perturbed Attention Guidance', open=False):
                        active = gr.Checkbox(value=False, default=False, label="PAG Active", elem_id='pag_active')
                        pag_sanf = gr.Checkbox(value=False, default=False, label="Use Saliency-Adaptive Noise Fusion", elem_id='pag_sanf')
                        with gr.Row():
                                pag_scale = gr.Slider(value = 0, minimum = 0, maximum = 20.0, step = 0.5, label="PAG Scale", elem_id = 'pag_scale', info="")
                        with gr.Row():
                                start_step = gr.Slider(value = 0, minimum = 0, maximum = 150, step = 1, label="PAG Start Step", elem_id = 'pag_start_step', info="")
                                end_step = gr.Slider(value = 150, minimum = 0, maximum = 150, step = 1, label="PAG End Step", elem_id = 'pag_end_step', info="")

                with gr.Accordion('CFG Scheduler', open=False):
                        cfg_interval_enable = gr.Checkbox(value=False, default=False, label="Enable CFG Scheduler", elem_id='cfg_interval_enable', info="If enabled, applies CFG only within noise interval with the selected schedule type. PAG must be enabled (scale can be 0). SDXL recommend CFG=15; CFG interval (0.28, 5.42]")
                        with gr.Row():
                                cfg_schedule = gr.Dropdown(
                                        value='Constant',
                                        choices= SCHEDULES,
                                        label="CFG Schedule Type", 
                                        elem_id='cfg_interval_schedule', 
                                )
                                cfg_interval_low = gr.Slider(value = 0, minimum = 0, maximum = 100, step = 0.1, label="CFG Noise Interval Low", elem_id = 'cfg_interval_low', info="")
                                cfg_interval_high = gr.Slider(value = 100, minimum = 0, maximum = 100, step = 0.1, label="CFG Noise Interval High", elem_id = 'cfg_interval_high', info="")
                                
                active.do_not_save_to_config = True
                pag_sanf.do_not_save_to_config = True
                pag_scale.do_not_save_to_config = True
                start_step.do_not_save_to_config = True
                end_step.do_not_save_to_config = True
                cfg_interval_enable.do_not_save_to_config = True
                cfg_schedule.do_not_save_to_config = True
                cfg_interval_low.do_not_save_to_config = True
                cfg_interval_high.do_not_save_to_config = True
                self.infotext_fields = [
                        (active, lambda d: gr.Checkbox.update(value='PAG Active' in d)),
                        (pag_sanf, lambda d: gr.Checkbox.update(value='PAG SANF' in d)),
                        (pag_scale, 'PAG Scale'),
                        (start_step, 'PAG Start Step'),
                        (end_step, 'PAG End Step'),
                        (cfg_interval_enable, 'CFG Interval Enable'),
                        (cfg_schedule, 'CFG Interval Schedule'),
                        (cfg_interval_low, 'CFG Interval Low'),
                        (cfg_interval_high, 'CFG Interval High')
                ]
                self.paste_field_names = [
                        'pag_active',
                        'pag_sanf',
                        'pag_scale',
                        'pag_start_step',
                        'pag_end_step',
                        'cfg_interval_enable',
                        'cfg_interval_schedule',
                        'cfg_interval_low',
                        'cfg_interval_high',
                ]
                return [active, pag_scale, start_step, end_step, cfg_interval_enable, cfg_schedule, cfg_interval_low, cfg_interval_high, pag_sanf]

        def process_batch(self, p: StableDiffusionProcessing, *args, **kwargs):
               self.pag_process_batch(p, *args, **kwargs)

        def pag_process_batch(self, p: StableDiffusionProcessing, active, pag_scale, start_step, end_step, cfg_interval_enable, cfg_schedule, cfg_interval_low, cfg_interval_high, pag_sanf, *args, **kwargs):
                # Clean previous hook handles before registering this batch.
                self.remove_all_hooks()

                active = getattr(p, "pag_active", active)
                pag_sanf = getattr(p, "pag_sanf", pag_sanf)
                cfg_interval_enable = getattr(p, "cfg_interval_enable", cfg_interval_enable)
                if active is False and cfg_interval_enable is False:
                        return
                pag_scale = getattr(p, "pag_scale", pag_scale)
                start_step = getattr(p, "pag_start_step", start_step)
                end_step = getattr(p, "pag_end_step", end_step)

                cfg_schedule = getattr(p, "cfg_interval_schedule", cfg_schedule)
                cfg_interval_low = getattr(p, "cfg_interval_low", cfg_interval_low)
                cfg_interval_high = getattr(p, "cfg_interval_high", cfg_interval_high)

                if active:
                        p.extra_generation_params.update({
                                "PAG Active": active,
                                "PAG SANF": pag_sanf,
                                "PAG Scale": pag_scale,
                                "PAG Start Step": start_step,
                                "PAG End Step": end_step,
                        })
                if cfg_interval_enable:
                        p.extra_generation_params.update({
                                "CFG Interval Enable": cfg_interval_enable,
                                "CFG Interval Schedule": cfg_schedule,
                                "CFG Interval Low": cfg_interval_low,
                                "CFG Interval High": cfg_interval_high
                        })
                self.create_hook(p, active, pag_scale, start_step, end_step, cfg_interval_enable, cfg_schedule, cfg_interval_low, cfg_interval_high, pag_sanf)

        def create_hook(self, p: StableDiffusionProcessing, active, pag_scale, start_step, end_step, cfg_interval_enable, cfg_schedule, cfg_interval_low, cfg_interval_high, pag_sanf, *args, **kwargs):
                # Create a list of parameters for each concept
                pag_params = PAGStateParams()

                # Add to p's incant_cfg_params
                if not hasattr(p, 'incant_cfg_params'):
                        logger.error("No incant_cfg_params found in p")
                p.incant_cfg_params['pag_params'] = pag_params

                # Preserve any setup timing already recorded before state was attached.
                _record_pag_timing(pag_params, "create_hook_setup", 0.0)
                
                pag_params.pag_active = active 
                pag_params.pag_sanf = pag_sanf 
                pag_params.pag_scale = pag_scale
                pag_params.pag_start_step = start_step
                pag_params.pag_end_step = end_step
                pag_params.cfg_interval_enable = cfg_interval_enable
                pag_params.cfg_interval_schedule = cfg_schedule
                pag_params.max_sampling_step = p.steps
                pag_params.guidance_scale = p.cfg_scale
                pag_params.batch_size = p.batch_size
                pag_params.denoiser = None
                pag_params.cfg_interval_scheduled_value = p.cfg_scale

                if pag_params.cfg_interval_enable:
                       # Refer to 3.1 Practice in the paper
                       # We want to round high and low noise levels to the nearest integer index
                       low_index = find_closest_index(cfg_interval_low, pag_params.max_sampling_step)
                       high_index = find_closest_index(cfg_interval_high, pag_params.max_sampling_step)
                       pag_params.cfg_interval_low = calculate_noise_level(low_index, pag_params.max_sampling_step)
                       pag_params.cfg_interval_high = calculate_noise_level(high_index, pag_params.max_sampling_step)
                       logger.debug(f"Step Aligned CFG Interval (low, high): ({low_index}, {high_index}), Step Aligned CFG Interval: ({round(pag_params.cfg_interval_low, 4)}, {round(pag_params.cfg_interval_high, 4)})")

                # Get all the qv modules
                cross_attn_modules = self.get_cross_attn_modules()
                if len(cross_attn_modules) == 0:
                        logger.error("No cross attention modules found, cannot proceed with PAG")
                        return
                pag_params.crossattn_modules = [m for m in cross_attn_modules if 'CrossAttention' in m.__class__.__name__]

                # Use lambdas to bind per-batch state without globals.
                self.remove_callbacks()
                cfg_denoise_lambda = lambda callback_params: self.on_cfg_denoiser_callback(callback_params, pag_params)
                cfg_denoised_lambda = lambda callback_params: self.on_cfg_denoised_callback(callback_params, pag_params)
                self._cfg_denoiser_callback = cfg_denoise_lambda
                self._cfg_denoised_callback = cfg_denoised_lambda
                if pag_params.pag_active:
                        self.ready_hijack_forward(pag_params.crossattn_modules, pag_scale)

                logger.debug('Hooked PAG callbacks')
                script_callbacks.on_cfg_denoiser(cfg_denoise_lambda)
                script_callbacks.on_cfg_denoised(cfg_denoised_lambda)



        def postprocess_batch(self, p, *args, **kwargs):
                self.pag_postprocess_batch(p, *args, **kwargs)

        def pag_postprocess_batch(self, p, active, *args, **kwargs):
                pag_params = getattr(p, "incant_cfg_params", {}).get("pag_params") if getattr(p, "incant_cfg_params", None) else None
                if pag_params is not None:
                        _merge_pag_timings(p, pag_params)
                self.remove_all_hooks()
                self.remove_callbacks()
                logger.debug('Removed PAG hooks and callbacks')
                active = getattr(p, "pag_active", active)
                if active is False:
                        return

        def remove_callbacks(self):
                if self._cfg_denoiser_callback is not None:
                        script_callbacks.remove_callbacks_for_function(self._cfg_denoiser_callback)
                        self._cfg_denoiser_callback = None
                if self._cfg_denoised_callback is not None:
                        script_callbacks.remove_callbacks_for_function(self._cfg_denoised_callback)
                        self._cfg_denoised_callback = None

        def remove_all_hooks(self):
                for handle in self._pag_hook_handles:
                        handle.remove()
                self._pag_hook_handles = []

                cross_attn_modules = self.get_cross_attn_modules()
                for module in cross_attn_modules:
                        to_v = getattr(module, 'to_v', None)
                        module_hooks.modules_remove_field(module, 'pag_enable')
                        module_hooks.modules_remove_field(module, 'pag_last_to_v')
                        if to_v is not None:
                                module_hooks.modules_remove_field(to_v, 'pag_parent_module')

        def unhook_callbacks(self, pag_params: PAGStateParams = None):
                self.remove_all_hooks()
                self.remove_callbacks()


        def ready_hijack_forward(self, crossattn_modules, pag_scale):
                """ Create hooks in the forward pass of the cross attention modules
                Copies the output of the to_v module to the parent module
                Then applies the PAG perturbation to the output of the cross attention module (multiplication by identity)
                """

                # add field for last_to_v
                for module in crossattn_modules:
                        to_v = getattr(module, 'to_v', None)
                        module_hooks.modules_add_field(module, 'pag_enable', False)
                        module_hooks.modules_add_field(module, 'pag_last_to_v', None)
                        if to_v is not None:
                                module_hooks.modules_add_field(to_v, 'pag_parent_module', [module])

                def to_v_pre_hook(module, input, kwargs, output):
                        """ Copy the output of the to_v module to the parent module """
                        parent_module = getattr(module, 'pag_parent_module', None)
                        # copy the output of the to_v module to the parent module
                        setattr(parent_module[0], 'pag_last_to_v', output.detach())

                def pag_pre_hook(module, input, kwargs, output):
                        if hasattr(module, 'pag_enable') and getattr(module, 'pag_enable', False) is False:
                                return
                        if not hasattr(module, 'pag_last_to_v'):
                                return

                        # get the last to_v output and save it
                        last_to_v = getattr(module, 'pag_last_to_v', None)

                        batch_size, seq_len, inner_dim = output.shape
                        if last_to_v is not None:
                                # Multiplication by an expanded identity matrix is exactly this slice.
                                # Avoid allocating the identity tensor and launching an einsum per attention call.
                                return last_to_v[:, :seq_len, :]
                        return output

                # Create hooks and keep RemovableHandles so cleanup does not need
                # to rewrite PyTorch hook tables globally.
                for module in crossattn_modules:
                        self._pag_hook_handles.append(module_hooks.module_add_forward_hook(module, pag_pre_hook, hook_type="forward", with_kwargs=True))
                        to_v = getattr(module, 'to_v', None)
                        if to_v is not None:
                                self._pag_hook_handles.append(module_hooks.module_add_forward_hook(to_v, to_v_pre_hook, hook_type="forward", with_kwargs=True))

        def get_middle_block_modules(self):
                """ Get all attention modules from the middle block 
                Refere to page 22 of the PAG paper, Appendix A.2
                
                """
                try:
                        m = shared.sd_model
                        nlm = m.network_layer_mapping
                        middle_block_modules = [m for m in nlm.values() if 'middle_block_1_transformer_blocks_0_attn1' in m.network_layer_name and 'CrossAttention' in m.__class__.__name__]
                        return middle_block_modules
                except AttributeError:
                        logger.exception("AttributeError in get_middle_block_modules", stack_info=True)
                        return []
                except Exception:
                        logger.exception("Exception in get_middle_block_modules", stack_info=True)
                        return []

        def get_cross_attn_modules(self):
                """ Get all cross attention modules """
                return self.get_middle_block_modules()

        def on_cfg_denoiser_callback(self, params: CFGDenoiserParams, pag_params: PAGStateParams):
                started = time.perf_counter()
                try:
                        self._on_cfg_denoiser_callback(params, pag_params)
                finally:
                        _record_pag_timing(pag_params, "cfg_denoiser_callback", time.perf_counter() - started)

        def _on_cfg_denoiser_callback(self, params: CFGDenoiserParams, pag_params: PAGStateParams):
                # Keep PAG hooks installed for the batch; per-step work only updates
                # mutable state. Removing hooks here disables the extra PAG pass.
                pag_params.step = params.sampling_step
                pag_params.pag_x_out = None

                # CFG Interval. Keep rho fixed to the upstream/default curve for now;
                # changing it is quality-affecting and should be a separate tuning pass.
                pag_params.current_noise_level = calculate_noise_level(
                        i = pag_params.step,
                        N = pag_params.max_sampling_step,
                )

                if pag_params.cfg_interval_enable:
                        # Calculate noise interval for every schedule, including Constant.
                        start = pag_params.cfg_interval_low
                        end = pag_params.cfg_interval_high
                        begin_range = start if start <= end else end
                        end_range = end if start <= end else start
                        scheduled_cfg_scale = cfg_scheduler(
                                pag_params.cfg_interval_schedule,
                                pag_params.step,
                                pag_params.max_sampling_step,
                                pag_params.guidance_scale,
                        )
                        pag_params.cfg_interval_scheduled_value = (
                                scheduled_cfg_scale
                                if begin_range <= pag_params.current_noise_level <= end_range
                                else 1.0
                        )

                # Run PAG only if active and within interval
                if not pag_params.pag_active or pag_params.pag_scale <= 0:
                        return
                if not pag_params.pag_start_step <= params.sampling_step <= pag_params.pag_end_step or pag_params.pag_scale <= 0:
                        return

                if isinstance(params.text_cond, dict):
                        pag_params.text_cond = {key: value.detach() for key, value in params.text_cond.items()}
                        if isinstance(params.text_uncond, dict):
                                pag_params.text_uncond = {key: value.detach() for key, value in params.text_uncond.items()}
                        else:
                                pag_params.text_uncond = params.text_uncond.detach()
                else:
                        pag_params.text_cond = params.text_cond.detach()
                        pag_params.text_uncond = params.text_uncond.detach()

                pag_params.x_in = params.x.detach()
                pag_params.sigma = params.sigma.detach()
                pag_params.image_cond = params.image_cond.detach() if params.image_cond is not None else None
                pag_params.denoiser = params.denoiser
                pag_params.make_condition_dict = get_make_condition_dict_fn(params.text_uncond)


        def on_cfg_denoised_callback(self, params: CFGDenoisedParams, pag_params: PAGStateParams):
                started = time.perf_counter()
                try:
                        self._on_cfg_denoised_callback(params, pag_params)
                finally:
                        _record_pag_timing(pag_params, "cfg_denoised_callback", time.perf_counter() - started)

        def _on_cfg_denoised_callback(self, params: CFGDenoisedParams, pag_params: PAGStateParams):
                """ Callback function for the CFGDenoisedParams 
                Refer to pg.22 A.2 of the PAG paper for how CFG and PAG combine
                
                """
                # Run only within interval
                # Run PAG only if active and within interval
                if not pag_params.pag_active or pag_params.pag_scale <= 0:
                        return
                if not pag_params.pag_start_step <= params.sampling_step <= pag_params.pag_end_step or pag_params.pag_scale <= 0:
                        return

                # passed from on_cfg_denoiser_callback
                x_in = pag_params.x_in
                if x_in is None or pag_params.text_cond is None or pag_params.text_uncond is None:
                        logger.warning("Skipping PAG extra pass because denoiser state was not captured")
                        return
                tensor = pag_params.text_cond
                uncond = pag_params.text_uncond
                image_cond_in = pag_params.image_cond
                sigma_in = pag_params.sigma
                
                make_condition_dict = pag_params.make_condition_dict or get_make_condition_dict_fn(uncond)

                # set pag_enable to True for the hooked cross attention modules
                for module in pag_params.crossattn_modules:
                        setattr(module, 'pag_enable', True)

                seg_saved_state = _suspend_seg_for_pag_hidden_pass()
                try:
                        # get the PAG guidance (is there a way to optimize this so we don't have to calculate it twice?)
                        hidden_started = time.perf_counter()
                        try:
                                pag_params.pag_x_out = pag_inner_model_x_out(
                                        params.inner_model,
                                        x_in,
                                        sigma_in,
                                        tensor,
                                        uncond,
                                        image_cond_in,
                                        make_condition_dict,
                                        pag_params.batch_size,
                                )
                        finally:
                                _record_pag_detail(pag_params, "pag_hidden_denoise", time.perf_counter() - hidden_started)
                finally:
                        _restore_seg_after_pag_hidden_pass(seg_saved_state)
                        # set pag_enable to False even if the hidden PAG pass raises
                        for module in pag_params.crossattn_modules:
                                setattr(module, 'pag_enable', False)
                        pag_params.x_in = None
                        pag_params.text_cond = None
                        pag_params.text_uncond = None
                        pag_params.image_cond = None
                        pag_params.sigma = None
        
        def get_xyz_axis_options(self) -> dict:
                xyz_grid = [x for x in scripts.scripts_data if x.script_class.__module__ in ("xyz_grid.py", "scripts.xyz_grid")][0].module
                extra_axis_options = {
                        xyz_grid.AxisOption("[PAG] Active", str, pag_apply_override('pag_active', boolean=True), choices=xyz_grid.boolean_choice(reverse=True)),
                        xyz_grid.AxisOption("[PAG] SANF", str, pag_apply_override('pag_sanf', boolean=True), choices=xyz_grid.boolean_choice(reverse=True)),
                        xyz_grid.AxisOption("[PAG] PAG Scale", float, pag_apply_field("pag_scale")),
                        xyz_grid.AxisOption("[PAG] PAG Start Step", int, pag_apply_field("pag_start_step")),
                        xyz_grid.AxisOption("[PAG] PAG End Step", int, pag_apply_field("pag_end_step")),
                        xyz_grid.AxisOption("[PAG] Enable CFG Scheduler", str, pag_apply_override('cfg_interval_enable', boolean=True), choices=xyz_grid.boolean_choice(reverse=True)),
                        xyz_grid.AxisOption("[PAG] CFG Noise Interval Low", float, pag_apply_field("cfg_interval_low")),
                        xyz_grid.AxisOption("[PAG] CFG Noise Interval High", float, pag_apply_field("cfg_interval_high")),
                        xyz_grid.AxisOption("[PAG] CFG Schedule Type", str, pag_apply_override('cfg_interval_schedule', boolean=False), choices=lambda: SCHEDULES),
                        #xyz_grid.AxisOption("[PAG] ctnms_alpha", float, pag_apply_field("pag_ctnms_alpha")),
                }
                return extra_axis_options



# from modules/sd_samplers_cfg_denoiser.py:187-195
def get_make_condition_dict_fn(text_uncond):
        if shared.sd_model.model.conditioning_key == "crossattn-adm":
                make_condition_dict = lambda c_crossattn, c_adm: {"c_crossattn": [c_crossattn], "c_adm": c_adm}
        else:
                if isinstance(text_uncond, dict):
                        make_condition_dict = lambda c_crossattn, c_concat: {**c_crossattn, "c_concat": [c_concat]}
                else:
                        make_condition_dict = lambda c_crossattn, c_concat: {"c_crossattn": [c_crossattn], "c_concat": [c_concat]}
        return make_condition_dict


def calculate_noise_level(i, N, sigma_min=0.002, sigma_max=80.0, rho=3):
    """
    Calculate the noise level for a given sampling step index.

    Parameters:
    i (int): Index of the current sampling step (0-based index).
    N (int): Total number of sampling steps.
    sigma_min (float): Minimum sigma value for min noise level, default 0.002.
    sigma_max (float): Maximum sigma value for max noise level, default 80.0.
    rho (int): Discretization parameter, default 3 for SD-XL, 7 for EDM2.

    Returns:
    float: Calculated noise level for the given step.
    """
    if i == 0:
        return sigma_max
    if i >= N:
        return 0.0
    sigma_max_p = sigma_max ** (1/rho)
    sigma_min_p = sigma_min ** (1/rho)
    inner_term = sigma_max_p + (i / (N - 1)) * (sigma_min_p - sigma_max_p)
    noise_level = inner_term ** rho

    return noise_level


def find_closest_index(noise_level: float, N: int, sigma_min=0.002, sigma_max=80.0, rho=3, tol=1e-6):
    """
    Given a noise level, find the closest integer index in the range [0, N-1] that corresponds to the noise level.

    Parameters:
    noise_level (float): Target noise level to find the closest index for.
    N (int): Total number of sampling steps.
    sigma_min (float): Minimum sigma value for min noise level, default 0.002.
    sigma_max (float): Maximum sigma value for max noise level, default 80.0.
    rho (int): Discretization parameter, default 3 for SD-XL, 7 for EDM2.

    Returns:
    int: The closest index to the specified noise level.
    """
    # Min/max noise levels for the given range
    if noise_level <= sigma_min:
        return N
    if noise_level >= sigma_max:
        return 0
        #return N - 1
    
    low, high = 0, N - 1
    while low <= high:
        mid = (low + high) // 2
        mid_nl = calculate_noise_level(mid, N)
        if abs(mid_nl - noise_level) < tol:
            return mid
        elif mid_nl < noise_level:
            high = mid - 1
        else:
            low = mid + 1
    
    # If exact match not found, return the index with noise level closest to the target
    return low if abs(calculate_noise_level(low, N) - noise_level) < abs(calculate_noise_level(high, N) - noise_level) else high


### CFG Schedulers


def cfg_scheduler(schedule: str, step: int, max_steps: int, w0: float) -> float:
        """
        Constant scheduler for CFG guidance weight.

        Parameters:
        step (int): Current sampling step.
        max_steps (int): Total number of sampling steps.
        w0 (float): Constant value for the guidance weight.

        Returns:
        float: Scheduled guidance weight value.
        """
        match schedule:
                case 'Constant':
                        return constant_schedule(step, max_steps, w0)
                case 'Linear':
                        return linear_schedule(step, max_steps, w0)
                case 'Clamp-Linear (c=4.0)':
                        return clamp_linear_schedule(step, max_steps, w0, 4.0)
                case 'Clamp-Linear (c=2.0)':
                        return clamp_linear_schedule(step, max_steps, w0, 2.0)
                case 'Clamp-Linear (c=1.0)':
                        return clamp_linear_schedule(step, max_steps, w0, 1.0)
                case 'Inverse-Linear':
                        return invlinear_schedule(step, max_steps, w0)
                case 'PCS (s=0.01)':
                        return powered_cosine_schedule(step, max_steps, w0, 0.01)
                case 'PCS (s=0.1)':
                        return powered_cosine_schedule(step, max_steps, w0, 0.1)
                case 'PCS (s=1.0)':
                        return powered_cosine_schedule(step, max_steps, w0, 1.0)
                case 'PCS (s=2.0)':
                        return powered_cosine_schedule(step, max_steps, w0, 2.0)
                case 'PCS (s=4.0)':
                        return powered_cosine_schedule(step, max_steps, w0, 4.0)
                case 'Clamp-Cosine (c=4.0)':
                        return clamp_cosine_schedule(step, max_steps, w0, 4.0)
                case 'Clamp-Cosine (c=2.0)':
                        return clamp_cosine_schedule(step, max_steps, w0, 2.0)
                case 'Clamp-Cosine (c=1.0)':
                        return clamp_cosine_schedule(step, max_steps, w0, 1.0)
                case 'Cosine':
                        return cosine_schedule(step, max_steps, w0)
                case 'Sine':
                        return sine_schedule(step, max_steps, w0)
                case 'V-Shape':
                        return v_shape_schedule(step, max_steps, w0)
                case 'A-Shape':
                        return a_shape_schedule(step, max_steps, w0)
                case 'Interval':
                        return interval_schedule(step, max_steps, w0, 0.25, 5.42)
                case _:
                        logger.error(f"Invalid CFG schedule: {schedule}")
                        return constant_schedule(step, max_steps, w0)


def constant_schedule(step: int, max_steps: int, w0: float):
        """
        Constant scheduler for CFG guidance weight.
        """
        return w0


def linear_schedule(step: int, max_steps: int, w0: float):
        """
        Normalized linear scheduler for CFG guidance weight.
        Such that integral 0-> T ~ w(t) dt  = w*T
        """
        # return w0 * (1 - step / max_steps)
        return w0 * 2 * (1 - step / max_steps)


def clamp_linear_schedule(step: int, max_steps: int, w0: float, c: float):
        """
        Normalized clamp-linear scheduler for CFG guidance weight.
        """
        return max(c, linear_schedule(step, max_steps, w0))


def clamp_cosine_schedule(step: int, max_steps: int, w0: float, c: float):
        """
        Normalized clamp-cosine scheduler for CFG guidance weight.
        """
        return max(c, cosine_schedule(step, max_steps, w0))


def invlinear_schedule(step: int, max_steps: int, w0: float):
        """ 
        Normalized inverse linear scheduler for CFG guidance weight.
        """
        # return w0 * (step / max_steps)
        return w0 * 2 * (step / max_steps)


def powered_cosine_schedule(step: int, max_steps: int, w0: float, s: float):
        """
        Normalized cosine scheduler for CFG guidance weight.
        """
        return w0 * ((1 - math.cos(math.pi * ((max_steps - step) / max_steps)**s))/2.0)


def cosine_schedule(step: int, max_steps: int, w0: float):
        """
        Normalized cosine scheduler for CFG guidance weight.
        """
        return w0 * (1 + math.cos(math.pi * step / max_steps))


def sine_schedule(step: int, max_steps: int, w0: float):
        """
        Normalized sine scheduler for CFG guidance weight.
        """
        return w0 * (math.sin((math.pi * step / max_steps) - (math.pi / 2)) + 1) 


def v_shape_schedule(step: int, max_steps: int, w0: float):
        """
        Normalized V-shape scheduler for CFG guidance weight.
        """
        if step < max_steps / 2:
                return invlinear_schedule(step, max_steps, w0)
        return linear_schedule(step, max_steps, w0)


def a_shape_schedule(step: int, max_steps: int, w0: float):
        """
        Normalized A-shape scheduler for CFG guidance weight.
        """
        if step < max_steps / 2:
                return linear_schedule(step, max_steps, w0)
        return invlinear_schedule(step, max_steps, w0)


def interval_schedule(step: int, max_steps: int, w0: float, low: float, high: float):
        """
        Normalized interval scheduler for CFG guidance weight.
        """
        if low <= step <= high:
                return w0
        return 1.0



# XYZ Plot
# Based on @mcmonkey4eva's XYZ Plot implementation here: https://github.com/mcmonkeyprojects/sd-dynamic-thresholding/blob/master/scripts/dynamic_thresholding.py
def pag_apply_override(field, boolean: bool = False):
    def fun(p, x, xs):
        if boolean:
            x = True if x.lower() == "true" else False
        setattr(p, field, x)
        if not hasattr(p, "pag_active"):
                setattr(p, "pag_active", True)
        if 'cfg_interval_' in field and not hasattr(p, "cfg_interval_enable"):
            setattr(p, "cfg_interval_enable", True)
    return fun


def pag_apply_field(field):
    def fun(p, x, xs):
        if not hasattr(p, "pag_active"):
                setattr(p, "pag_active", True)
        setattr(p, field, x)
    return fun
