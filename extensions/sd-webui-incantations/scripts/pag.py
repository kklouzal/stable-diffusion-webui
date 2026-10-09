import logging
import weakref
from contextlib import suppress
from os import environ
from modules import headless_ui as gr
from scripts.ui_wrapper import UIWrapper, cond_crossattn, sampler_step, xyz_field_setter
from modules import script_callbacks
from modules.script_callbacks import CFGDenoiserParams, CFGDenoisedParams
from modules.processing import StableDiffusionProcessing
from modules import shared, sd_unet_row_memo
from scripts.incant_utils import module_hooks, timing

import torch


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


class PAGStateParams:
        def __init__(self):
                self.pag_active: bool = False
                self.pag_sanf: bool = False # saliency-adaptive noise fusion, handled in cfg_combiner
                self.pag_scale: float = -1      # PAG guidance scale
                self.pag_start_step: int = 0
                self.pag_end_step: int = 150
                self.step: int = 0  # sampler step of the denoiser call in progress (set by the cfg_denoiser callback)
                self.crossattn_modules = [] # the hooked middle-block self-attention modules
                self.pag_x_out = None
                self.openclaw_extension_timings = {}
                self.seg_q_modules = None


def cond_batch_size(cond):
        tensor = cond_crossattn(cond)
        if tensor is None:
                raise RuntimeError("PAG conditioning is missing a cross-attention tensor")
        return tensor.shape[0]


def _seg_to_q_modules():
        """Return SEG-managed to_q modules currently installed on the shared model.

        PAG runs an internal denoising pass to compute perturbed-attention output.
        That pass evaluates cond rows only; SEG must not leak into it because SEG
        treats any even-sized attention batch as a paired cond/uncond CFG batch.
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


def _suspend_seg_for_pag_hidden_pass(seg_q_modules):
        saved = []
        for to_q in seg_q_modules:
                saved.append((to_q, getattr(to_q, 'seg_enable', False)))
                to_q.seg_enable = False
        return saved


def _restore_seg_after_pag_hidden_pass(saved):
        for to_q, enabled in saved:
                with suppress(Exception):
                        to_q.seg_enable = enabled


def pag_cond_rows_x_out(inner_model, memo, preserve_call_sequence, whole_calls=False):
        """Run PAG's perturbed pass on the cond rows of the recorded main-pass calls.

        Every PAG input row is an exact duplicate of a main-pass row and the
        combiner reads only the cond rows, which A1111 always places first. So
        each recorded call is replayed with its own inputs, limited to its
        leading cond rows, in the main pass's order and chunking. A call whose
        rows are coupled (``row_subset_ok`` cleared by the UNet forward that
        ran it) is replayed whole, as is every call when ``whole_calls`` is
        set. Calls holding only uncond rows are skipped unless the call
        sequence must be preserved: hypertile draws its tile layout per UNet
        call, and row-coupled calls may draw from the global RNG, so those
        replays keep the previous one-call-per-main-call behavior and discard
        the uncond output. The UNet forward serving a replay may reuse what it
        recorded for the call (modules/sd_unet_row_memo.py).
        """
        outs = []
        covered = 0
        for rec in memo.calls:
                cond_rows = rec.cond_rows
                whole = whole_calls or not rec.row_subset_ok
                if cond_rows == 0 and not whole and not preserve_call_sequence:
                        continue
                rows = cond_rows if cond_rows and not whole else rec.rows
                with sd_unet_row_memo.replaying(rec, rows):
                        out = inner_model(
                                rec.x[:rows],
                                rec.sigma[:rows],
                                cond=sd_unet_row_memo.slice_cond_rows(rec.cond, rec.rows, rows),
                        )
                if cond_rows:
                        outs.append(out[:cond_rows])
                        covered += cond_rows
        if covered != memo.n_cond:
                raise RuntimeError(f"PAG: the recorded main pass covers {covered} of {memo.n_cond} cond rows")
        return outs[0] if len(outs) == 1 else torch.cat(outs)


class PAGExtensionScript(UIWrapper):
        def __init__(self):
                super().__init__()
                self._pag_hooked_modules = []
                self._recorded_denoiser = None

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

                # The CFG Scheduler ("CFG Interval") was removed. Its four inputs stay so the positional Incantations
                # args keep their slots: API callers (the controller) send [False, "Constant", 0.0, 100.0] there and
                # PAG SANF after them (README, "A1111 API argument order"). Enabling it fails the request; the other
                # three are ignored, as they always were while it was off.
                cfg_interval_enable = gr.Checkbox(value=False, label="Enable CFG Scheduler", elem_id='cfg_interval_enable', info="Removed; must stay off")
                cfg_schedule = gr.Dropdown(value='Constant', choices=['Constant'], label="CFG Schedule Type", elem_id='cfg_interval_schedule', info="Removed; ignored")
                cfg_interval_low = gr.Slider(value = 0, minimum = 0, maximum = 100, step = 0.1, label="CFG Noise Interval Low", elem_id = 'cfg_interval_low', info="Removed; ignored")
                cfg_interval_high = gr.Slider(value = 100, minimum = 0, maximum = 100, step = 0.1, label="CFG Noise Interval High", elem_id = 'cfg_interval_high', info="Removed; ignored")

                self.infotext_fields = [
                        (active, lambda d: gr.Checkbox.update(value='PAG Active' in d)),
                        (pag_sanf, lambda d: gr.Checkbox.update(value='PAG SANF' in d)),
                        (pag_scale, 'PAG Scale'),
                        (start_step, 'PAG Start Step'),
                        (end_step, 'PAG End Step'),
                        # Re-running the infotext of an image made with the CFG Scheduler fails instead of rendering without it.
                        (cfg_interval_enable, 'CFG Interval Enable'),
                ]
                return [active, pag_scale, start_step, end_step, cfg_interval_enable, cfg_schedule, cfg_interval_low, cfg_interval_high, pag_sanf]

        def process_batch(self, p: StableDiffusionProcessing, *args, **kwargs):
               self.pag_process_batch(p, *args, **kwargs)

        def pag_process_batch(self, p: StableDiffusionProcessing, active, pag_scale, start_step, end_step, cfg_interval_enable, cfg_schedule, cfg_interval_low, cfg_interval_high, pag_sanf, *args, **kwargs):
                # Clean previous hook handles/callbacks before registering this batch.
                self.remove_all_hooks()
                self.remove_callbacks()
                self.remove_main_pass_recorder()

                # cfg_schedule, cfg_interval_low and cfg_interval_high are the removed CFG Scheduler's placeholder slots.
                if cfg_interval_enable:
                        raise ValueError("Incantations: the PAG CFG Scheduler (CFG Interval) was removed; cfg_interval_enable must be false")

                active = getattr(p, "pag_active", active)
                if not active:
                        return
                pag_sanf = getattr(p, "pag_sanf", pag_sanf)
                pag_scale = getattr(p, "pag_scale", pag_scale)
                start_step = getattr(p, "pag_start_step", start_step)
                end_step = getattr(p, "pag_end_step", end_step)

                crossattn_modules = self.get_cross_attn_modules()
                if not crossattn_modules:
                        # Requested PAG must never render without PAG, under an infotext that says "PAG Active".
                        # (module_hooks.get_modules returns [] for a model without a layer mapping.)
                        raise RuntimeError("PAG: no middle-block self-attention modules found on the loaded model")

                p.extra_generation_params.update({
                        "PAG Active": active,
                        "PAG SANF": pag_sanf,
                        "PAG Scale": pag_scale,
                        "PAG Start Step": start_step,
                        "PAG End Step": end_step,
                })
                self.create_hook(p, active, pag_scale, start_step, end_step, pag_sanf, crossattn_modules)

        def create_hook(self, p: StableDiffusionProcessing, active, pag_scale, start_step, end_step, pag_sanf, crossattn_modules):
                pag_params = PAGStateParams()
                p.incant_cfg_params['pag_params'] = pag_params

                # A zero-time entry: the API's openclaw_extension_timings has always listed this hook.
                timing.record(pag_params.openclaw_extension_timings, "create_hook_setup", 0.0)

                pag_params.pag_active = active
                pag_params.pag_sanf = pag_sanf
                pag_params.pag_scale = pag_scale
                pag_params.pag_start_step = start_step
                pag_params.pag_end_step = end_step
                pag_params.crossattn_modules = crossattn_modules
                self.ready_hijack_forward(crossattn_modules)

                def cfg_denoise_callback(callback_params):
                        return self.on_cfg_denoiser_callback(callback_params, pag_params)

                def cfg_denoised_callback(callback_params):
                        return self.on_cfg_denoised_callback(callback_params, pag_params)

                script_callbacks.on_cfg_denoiser(self.track_callback(cfg_denoise_callback))
                script_callbacks.on_cfg_denoised(self.track_callback(cfg_denoised_callback))
                logger.debug('Hooked PAG callbacks')

        def postprocess_batch(self, p, *args, **kwargs):
                pag_params = (getattr(p, "incant_cfg_params", None) or {}).get("pag_params")
                if pag_params is not None:
                        timing.merge_into_processing(p, "Incantations.PAGExtensionScript", pag_params.openclaw_extension_timings)
                        pag_params.openclaw_extension_timings = {}
                self.remove_all_hooks()
                self.remove_callbacks()
                self.remove_main_pass_recorder()
                logger.debug('Removed PAG hooks and callbacks')

        def recorded_denoiser(self):
                return self._recorded_denoiser() if self._recorded_denoiser is not None else None

        def _drop_main_pass_memo(self):
                memo = sd_unet_row_memo.disarm(self.recorded_denoiser())
                if memo is not None:
                        memo.clear()

        def remove_main_pass_recorder(self):
                """Release the main-pass memo and unwrap the recorded denoiser (batch end, next batch, failures)."""
                denoiser = self.recorded_denoiser()
                if denoiser is None:
                        self._recorded_denoiser = None
                        return
                self._drop_main_pass_memo()
                self._recorded_denoiser = None
                if not sd_unet_row_memo.uninstall(denoiser):
                        logger.warning("Not removing the PAG main-pass recorder because another wrapper replaced run_inner_model")

        def remove_all_hooks(self):
                self.remove_hook_handles()
                for module in self._pag_hooked_modules:
                        to_v = getattr(module, 'to_v', None)
                        module_hooks.modules_remove_field(module, 'pag_enable')
                        module_hooks.modules_remove_field(module, 'pag_last_to_v')
                        if to_v is not None:
                                module_hooks.modules_remove_field(to_v, 'pag_parent_module')
                self._pag_hooked_modules = []

        def ready_hijack_forward(self, crossattn_modules):
                """ Create hooks in the forward pass of the cross attention modules
                Copies the output of the to_v module to the parent module
                Then applies the PAG perturbation to the output of the cross attention module: with the identity
                attention map the output is to_out(to_v(x))
                """

                # add field for last_to_v
                self._pag_hooked_modules = list(crossattn_modules)
                for module in crossattn_modules:
                        to_v = getattr(module, 'to_v', None)
                        module_hooks.modules_add_field(module, 'pag_enable', False)
                        module_hooks.modules_add_field(module, 'pag_last_to_v', None)
                        if to_v is not None:
                                module_hooks.modules_add_field(to_v, 'pag_parent_module', [module])

                # The fields above are added before these hooks are installed and removed only after their handles.
                def to_v_forward_hook(module, input, kwargs, output):
                        """ Copy the output of the to_v module to the parent module """
                        module.pag_parent_module[0].pag_last_to_v = output.detach()

                def pag_forward_hook(module, input, kwargs, output):
                        if not module.pag_enable:
                                return

                        last_to_v = module.pag_last_to_v
                        if last_to_v is None:
                                raise RuntimeError("PAG: the perturbed attention call ran without a to_v output")
                        batch, seq_len, _ = output.shape
                        if last_to_v.shape[0] != batch or last_to_v.shape[1] < seq_len:
                                # Hypertile tiling this layer calls to_v on (batch * tiles) tile rows.
                                raise RuntimeError(
                                        f"PAG: the attention input was split into {last_to_v.shape[0]} rows of {last_to_v.shape[1]} tokens, "
                                        f"not the {batch} rows of {seq_len} tokens of its output (hypertile tiling the middle block?)"
                                )
                        # The identity attention map makes the attention output the values themselves, which then go
                        # through the output projection like any attention output (Ahn et al. 2024; diffusers
                        # PAGIdentitySelfAttnProcessor): to_out(to_v(x)). A context longer than x (ControlNet
                        # reference banks) keeps the values of the layer's own tokens, the leading seq_len.
                        return module.to_out(last_to_v[:, :seq_len, :])

                # Keep RemovableHandles so cleanup does not need to rewrite PyTorch hook tables globally.
                for module in crossattn_modules:
                        self.add_forward_hook(module, pag_forward_hook)
                        to_v = getattr(module, 'to_v', None)
                        if to_v is not None:
                                self.add_forward_hook(to_v, to_v_forward_hook)

        def get_cross_attn_modules(self):
                """ The middle block's self-attention modules; refer to page 22 of the PAG paper, Appendix A.2 """
                return module_hooks.get_modules(
                        network_layer_name_filter='middle_block_1_transformer_blocks_0_attn1',
                        module_name_filter='CrossAttention',
                )

        def on_cfg_denoiser_callback(self, params: CFGDenoiserParams, pag_params: PAGStateParams):
                with timing.timed(pag_params.openclaw_extension_timings, "cfg_denoiser_callback"):
                        self._on_cfg_denoiser_callback(params, pag_params)

        def _on_cfg_denoiser_callback(self, params: CFGDenoiserParams, pag_params: PAGStateParams):
                # Keep PAG hooks installed for the batch; per-step work only updates
                # mutable state. Removing hooks here disables the extra PAG pass.
                pag_params.pag_x_out = None

                # Run PAG only if active and within interval
                if not pag_params.pag_active or pag_params.pag_scale <= 0:
                        return
                denoiser = getattr(params, 'denoiser', None)
                if denoiser is None:
                        raise RuntimeError("PAG needs CFGDenoiserParams.denoiser for its step and to record the main denoiser pass")
                # The cfg_denoised callback of this denoiser call reads the same step.
                pag_params.step = sampler_step(denoiser)
                if not pag_params.pag_start_step <= pag_params.step <= pag_params.pag_end_step:
                        self._drop_main_pass_memo()
                        return

                # Record this step's main-pass UNet calls; the PAG pass replays their cond rows.
                if self.recorded_denoiser() is not denoiser:
                        self.remove_main_pass_recorder()
                        # Weak: a request that fails mid-step must not keep its denoiser and memo alive.
                        self._recorded_denoiser = weakref.ref(denoiser)
                sd_unet_row_memo.arm(denoiser, cond_batch_size(params.text_cond))


        def on_cfg_denoised_callback(self, params: CFGDenoisedParams, pag_params: PAGStateParams):
                with timing.timed(pag_params.openclaw_extension_timings, "cfg_denoised_callback"):
                        self._on_cfg_denoised_callback(params, pag_params)

        def _on_cfg_denoised_callback(self, params: CFGDenoisedParams, pag_params: PAGStateParams):
                """ Callback function for the CFGDenoisedParams
                Refer to pg.22 A.2 of the PAG paper for how CFG and PAG combine

                """
                # Run PAG only if active and within interval
                if not pag_params.pag_active or pag_params.pag_scale <= 0:
                        return
                if not pag_params.pag_start_step <= pag_params.step <= pag_params.pag_end_step:
                        return

                memo = sd_unet_row_memo.disarm(self.recorded_denoiser())
                if memo is None:
                        raise RuntimeError("PAG: the main denoiser pass of this step was not recorded")
                # TeaCache keeps per-call-lane state (cached residuals, a refresh distance averaged over all
                # rows of the lane): its PAG lane must keep seeing the whole calls it saw before.
                unet = getattr(getattr(shared.sd_model, 'model', None), 'diffusion_model', None)
                whole_calls = bool(getattr(unet, '_teacache_patched', False))

                # set pag_enable to True for the hooked cross attention modules
                for module in pag_params.crossattn_modules:
                        module.pag_enable = True

                if pag_params.seg_q_modules is None:
                        pag_params.seg_q_modules = _seg_to_q_modules()
                seg_saved_state = _suspend_seg_for_pag_hidden_pass(pag_params.seg_q_modules)
                try:
                        with timing.timed(pag_params.openclaw_extension_timings.setdefault("details", {}), "pag_hidden_denoise"):
                                pag_params.pag_x_out = pag_cond_rows_x_out(
                                        params.inner_model,
                                        memo,
                                        preserve_call_sequence=sd_unet_row_memo.hypertile_unet_enabled(getattr(shared.sd_model, 'model', None)),
                                        whole_calls=whole_calls,
                                )
                finally:
                        memo.clear()
                        _restore_seg_after_pag_hidden_pass(seg_saved_state)
                        # set pag_enable to False even if the hidden PAG pass raises
                        for module in pag_params.crossattn_modules:
                                module.pag_enable = False

        def get_xyz_axis_options(self, xyz_grid) -> list:
                return [
                        xyz_grid.AxisOption("[PAG] Active", str, xyz_field_setter('pag_active', 'pag_active', boolean=True), choices=xyz_grid.boolean_choice(reverse=True)),
                        xyz_grid.AxisOption("[PAG] SANF", str, xyz_field_setter('pag_sanf', 'pag_active', boolean=True), choices=xyz_grid.boolean_choice(reverse=True)),
                        xyz_grid.AxisOption("[PAG] PAG Scale", float, xyz_field_setter("pag_scale", 'pag_active')),
                        xyz_grid.AxisOption("[PAG] PAG Start Step", int, xyz_field_setter("pag_start_step", 'pag_active')),
                        xyz_grid.AxisOption("[PAG] PAG End Step", int, xyz_field_setter("pag_end_step", 'pag_active')),
                ]
