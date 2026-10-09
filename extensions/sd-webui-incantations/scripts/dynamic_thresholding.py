##################
# Stable Diffusion Dynamic Thresholding (CFG Scale Fix)
#
# Author: Alex 'mcmonkey' Goodwin
# GitHub URL: https://github.com/mcmonkeyprojects/sd-dynamic-thresholding
#
##################

import logging

from modules import headless_ui as gr
import dynthres_core
from modules import scripts, script_callbacks, sd_samplers, sd_samplers_common, sd_samplers_timesteps
from modules.sd_samplers_kdiffusion import CFGDenoiserKDiffusion as cfgdenoisekdiff
from scripts.ui_wrapper import add_xyz_axis_options, cond_crossattn

logger = logging.getLogger(__name__)

# The timestep samplers (DDIM, DDIM CFG++, PLMS, UniPC) run a CompVisSampler whose CFGDenoiserTimesteps has no
# k-diffusion CFG denoiser to wrap: swapping in CustomCFGDenoiser would silently compute the wrong math.
UNSUPPORTED_SAMPLERS = frozenset(sampler.name for sampler in sd_samplers_timesteps.samplers_data_timesteps)

######################### Data values #########################
MODES_WITH_VALUE = ["Power Up", "Power Down", "Linear Repeating", "Cosine Repeating", "Sawtooth"]

######################### Script class entrypoint #########################
class Script(scripts.Script):

    def title(self):
        return "Dynamic Thresholding (CFG Scale Fix)"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion("Dynamic Thresholding (CFG Scale Fix)", open=False, elem_id="dynthres_" + ("img2img" if is_img2img else "txt2img")):
            with gr.Row():
                enabled = gr.Checkbox(value=False, label="Enable Dynamic Thresholding (CFG Scale Fix)", elem_classes=["dynthres-enabled"], elem_id='dynthres_enabled')
            with gr.Group():
                mimic_scale = gr.Slider(minimum=1.0, maximum=30.0, step=0.5, label='Mimic CFG Scale', value=7.0, elem_id='dynthres_mimic_scale')
                with gr.Accordion("Advanced Options", open=False, elem_id='dynthres_advanced_opts'):
                    with gr.Row():
                        threshold_percentile = gr.Slider(minimum=90.0, value=100.0, maximum=100.0, step=0.05, label='Top percentile of latents to clamp', elem_id='dynthres_threshold_percentile')
                        interpolate_phi = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="Interpolate Phi", value=1.0, elem_id='dynthres_interpolate_phi')
                    with gr.Row():
                        mimic_mode = gr.Dropdown(dynthres_core.DynThresh.Modes, value="Constant", label="Mimic Scale Scheduler", elem_id='dynthres_mimic_mode')
                        cfg_mode = gr.Dropdown(dynthres_core.DynThresh.Modes, value="Constant", label="CFG Scale Scheduler", elem_id='dynthres_cfg_mode')
                    mimic_scale_min = gr.Slider(minimum=0.0, maximum=30.0, step=0.5, value=0.0, label="Minimum value of the Mimic Scale Scheduler", elem_id='dynthres_mimic_scale_min')
                    cfg_scale_min = gr.Slider(minimum=0.0, maximum=30.0, step=0.5, value=0.0, label="Minimum value of the CFG Scale Scheduler", elem_id='dynthres_cfg_scale_min')
                    sched_val = gr.Slider(minimum=0.0, maximum=40.0, step=0.5, value=4.0, label="Scheduler Value", info="Value unique to the scheduler mode - for Power Up/Down, this is the power. For Linear/Cosine Repeating, this is the number of repeats per image.", elem_id='dynthres_sched_val')
                    with gr.Row():
                        separate_feature_channels = gr.Checkbox(value=True, label="Separate Feature Channels", elem_id='dynthres_separate_feature_channels')
                        scaling_startpoint = gr.Radio(["ZERO", "MEAN"], value="MEAN", label="Scaling Startpoint")
                        variability_measure = gr.Radio(["STD", "AD"], value="AD", label="Variability Measure")
        self.infotext_fields = (
            (enabled, lambda d: gr.Checkbox.update(value="Dynamic thresholding enabled" in d)),
            (mimic_scale, "Mimic scale"),
            (separate_feature_channels, "Separate Feature Channels"),
            (scaling_startpoint, lambda d: gr.Radio.update(value=d.get("Scaling Startpoint", "MEAN"))),
            (variability_measure, lambda d: gr.Radio.update(value=d.get("Variability Measure", "AD"))),
            (interpolate_phi, "Interpolate Phi"),
            (threshold_percentile, "Threshold percentile"),
            (mimic_scale_min, "Mimic scale minimum"),
            (mimic_mode, lambda d: gr.Dropdown.update(value=d.get("Mimic mode", "Constant"))),
            (cfg_mode, lambda d: gr.Dropdown.update(value=d.get("CFG mode", "Constant"))),
            (cfg_scale_min, "CFG scale minimum"),
            (sched_val, "Scheduler value"))
        return [enabled, mimic_scale, threshold_percentile, mimic_mode, mimic_scale_min, cfg_mode, cfg_scale_min, sched_val, separate_feature_channels, scaling_startpoint, variability_measure, interpolate_phi]

    last_id = 0
    # Renamed samplers this script registered in all_samplers_map and has not removed yet. Generations are
    # serialized (queue_lock), so at most one is in use; any other is left by a batch whose generation failed
    # before postprocess_batch.
    registered_samplers = set()

    @staticmethod
    def _unregister_samplers(names):
        for name in names:
            sd_samplers.all_samplers_map.pop(name, None)
            Script.registered_samplers.discard(name)

    def _restore_original_sampler(self, p):
        if not hasattr(p, 'orig_sampler_name'):
            return
        p.sampler_name = p.orig_sampler_name
        self._unregister_samplers(p.fixed_samplers)
        if p.sampler is not None:
            p.sampler = sd_samplers.create_sampler(p.sampler_name, p.sd_model)
        del p.fixed_samplers
        del p.orig_sampler_name

    def process_batch(self, p, enabled, mimic_scale, threshold_percentile, mimic_mode, mimic_scale_min, cfg_mode, cfg_scale_min, sched_val, separate_feature_channels, scaling_startpoint, variability_measure, interpolate_phi, batch_number, prompts, seeds, subseeds):
        self._restore_original_sampler(p)
        self._unregister_samplers(list(Script.registered_samplers))
        enabled = getattr(p, 'dynthres_enabled', enabled)
        if not enabled:
            return
        orig_sampler_name = p.sampler_name
        if orig_sampler_name in UNSUPPORTED_SAMPLERS:
            raise RuntimeError(f"Cannot use sampler {orig_sampler_name} with Dynamic Thresholding")
        mimic_scale = getattr(p, 'dynthres_mimic_scale', mimic_scale)
        separate_feature_channels = getattr(p, 'dynthres_separate_feature_channels', separate_feature_channels)
        scaling_startpoint = getattr(p, 'dynthres_scaling_startpoint', scaling_startpoint)
        variability_measure = getattr(p, 'dynthres_variability_measure', variability_measure)
        interpolate_phi = getattr(p, 'dynthres_interpolate_phi', interpolate_phi)
        threshold_percentile = getattr(p, 'dynthres_threshold_percentile', threshold_percentile)
        mimic_mode = getattr(p, 'dynthres_mimic_mode', mimic_mode)
        mimic_scale_min = getattr(p, 'dynthres_mimic_scale_min', mimic_scale_min)
        cfg_mode = getattr(p, 'dynthres_cfg_mode', cfg_mode)
        cfg_scale_min = getattr(p, 'dynthres_cfg_scale_min', cfg_scale_min)
        sched_val = getattr(p, 'dynthres_scheduler_val', sched_val)
        p.extra_generation_params["Dynamic thresholding enabled"] = True
        p.extra_generation_params["Mimic scale"] = mimic_scale
        p.extra_generation_params["Separate Feature Channels"] = separate_feature_channels
        p.extra_generation_params["Scaling Startpoint"] = scaling_startpoint
        p.extra_generation_params["Variability Measure"] = variability_measure
        p.extra_generation_params["Interpolate Phi"] = interpolate_phi
        p.extra_generation_params["Threshold percentile"] = threshold_percentile
        p.extra_generation_params["Sampler"] = orig_sampler_name
        if mimic_mode != "Constant":
            p.extra_generation_params["Mimic mode"] = mimic_mode
            p.extra_generation_params["Mimic scale minimum"] = mimic_scale_min
        if cfg_mode != "Constant":
            p.extra_generation_params["CFG mode"] = cfg_mode
            p.extra_generation_params["CFG scale minimum"] = cfg_scale_min
        if cfg_mode in MODES_WITH_VALUE or mimic_mode in MODES_WITH_VALUE:
            p.extra_generation_params["Scheduler value"] = sched_val
        # A fresh ID per batch gives this batch's renamed sampler its own all_samplers_map key, so it never collides
        # with one a failed batch left registered.
        Script.last_id += 1
        # Percentage to portion
        threshold_percentile *= 0.01

        def make_sampler(orig_sampler_name):
            fixed_sampler_name = f"{orig_sampler_name}_dynthres{Script.last_id}"

            # Make a placeholder sampler
            sampler = sd_samplers.all_samplers_map[orig_sampler_name]
            dt_data = dynthres_core.DynThresh(mimic_scale, threshold_percentile, mimic_mode, mimic_scale_min, cfg_mode, cfg_scale_min, sched_val, p.steps, separate_feature_channels, scaling_startpoint, variability_measure, interpolate_phi)
            def new_constructor(model):
                result = sampler.constructor(model)
                cfg = CustomCFGDenoiser(result, dt_data)
                result.model_wrap_cfg = cfg
                return result
            new_sampler = sd_samplers_common.SamplerData(fixed_sampler_name, new_constructor, sampler.aliases, sampler.options)
            return fixed_sampler_name, new_sampler

        # Apply for usage
        p.orig_sampler_name = orig_sampler_name
        p.sampler_name, new_sampler = make_sampler(orig_sampler_name)
        sd_samplers.all_samplers_map[p.sampler_name] = new_sampler
        Script.registered_samplers.add(p.sampler_name)
        p.fixed_samplers = [p.sampler_name]

        if p.sampler is not None:
            p.sampler = sd_samplers.create_sampler(p.sampler_name, p.sd_model)

    def postprocess_batch(self, p, *args, **kwargs):
        self._restore_original_sampler(p)

######################### K-Diffusion Implementation logic #########################

class CustomCFGDenoiser(cfgdenoisekdiff):
    def __init__(self, model, dt_data):
        super().__init__(model)
        self.main_class = dt_data

    def combine_denoised(self, x_out, conds_list, uncond, cond_scale):
        denoised_uncond = x_out[-cond_crossattn(uncond).shape[0]:]
        self.main_class.step = self.step
        self.main_class.max_steps = self.total_steps

        relative = dynthres_core.cfg_relative(x_out, conds_list, denoised_uncond)
        return self.main_class.dynthresh_from_relative(relative, denoised_uncond, cond_scale)

######################### XYZ Plot Script Support logic #########################

def make_axis_options(xyz_grid):
    def apply_mimic_scale(p, x, xs):
        if x != 0:
            p.dynthres_enabled = True
            p.dynthres_mimic_scale = x
        else:
            p.dynthres_enabled = False
    def confirm_scheduler(p, xs):
        for x in xs:
            if x not in dynthres_core.DynThresh.Modes:
                raise RuntimeError(f"Unknown Scheduler: {x}")
    return [
        xyz_grid.AxisOption("[DynThres] Mimic Scale", float, apply_mimic_scale),
        xyz_grid.AxisOption("[DynThres] Separate Feature Channels", int,
                            xyz_grid.apply_field("dynthres_separate_feature_channels")),
        xyz_grid.AxisOption("[DynThres] Scaling Startpoint", str, xyz_grid.apply_field("dynthres_scaling_startpoint"), choices=lambda:['ZERO', 'MEAN']),
        xyz_grid.AxisOption("[DynThres] Variability Measure", str, xyz_grid.apply_field("dynthres_variability_measure"), choices=lambda:['STD', 'AD']),
        xyz_grid.AxisOption("[DynThres] Interpolate Phi", float, xyz_grid.apply_field("dynthres_interpolate_phi")),
        xyz_grid.AxisOption("[DynThres] Threshold Percentile", float, xyz_grid.apply_field("dynthres_threshold_percentile")),
        xyz_grid.AxisOption("[DynThres] Mimic Scheduler", str, xyz_grid.apply_field("dynthres_mimic_mode"), confirm=confirm_scheduler, choices=lambda: dynthres_core.DynThresh.Modes),
        xyz_grid.AxisOption("[DynThres] Mimic minimum", float, xyz_grid.apply_field("dynthres_mimic_scale_min")),
        xyz_grid.AxisOption("[DynThres] CFG Scheduler", str, xyz_grid.apply_field("dynthres_cfg_mode"), confirm=confirm_scheduler, choices=lambda: dynthres_core.DynThresh.Modes),
        xyz_grid.AxisOption("[DynThres] CFG minimum", float, xyz_grid.apply_field("dynthres_cfg_scale_min")),
        xyz_grid.AxisOption("[DynThres] Scheduler value", float, xyz_grid.apply_field("dynthres_scheduler_val"))
    ]

def callback_before_ui():
    try:
        xyz_grid = scripts.loaded_script_module("xyz_grid.py")
        add_xyz_axis_options(xyz_grid, make_axis_options(xyz_grid))
    except Exception:
        logger.exception("Failed to add Dynamic Thresholding support for X/Y/Z Plot Script")

script_callbacks.on_before_ui(callback_before_ui)
