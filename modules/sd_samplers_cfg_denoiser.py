import torch
from modules import prompt_parser, sd_samplers_common

from modules.shared import opts, state
import modules.shared as shared
from modules.script_callbacks import CFGDenoiserParams, cfg_denoiser_callback
from modules.script_callbacks import CFGDenoisedParams, cfg_denoised_callback
from modules.script_callbacks import AfterCFGCallbackParams, cfg_after_cfg_callback


def catenate_conds(conds):
    if not isinstance(conds[0], dict):
        return torch.cat(conds)

    return {key: torch.cat([x[key] for x in conds]) for key in conds[0].keys()}


def subscript_cond(cond, a, b):
    if not isinstance(cond, dict):
        return cond[a:b]

    return {key: vec[a:b] for key, vec in cond.items()}


def copy_condition_dict(cond):
    try:
        return cond.__class__(cond)
    except Exception:
        return dict(cond)


def pad_cond(tensor, repeats, empty):
    if not isinstance(tensor, dict):
        return torch.cat([tensor, empty.repeat((tensor.shape[0], repeats, 1))], axis=1)

    padded = copy_condition_dict(tensor)
    padded['crossattn'] = pad_cond(tensor['crossattn'], repeats, empty)
    return padded


class CFGDenoiser(torch.nn.Module):
    """
    Classifier free guidance denoiser. A wrapper for stable diffusion model (specifically for unet)
    that can take a noisy picture and produce a noise-free picture using two guidances (prompts)
    instead of one. Originally, the second prompt is just an empty string, but we use non-empty
    negative prompt.
    """

    def __init__(self, sampler):
        super().__init__()
        self.model_wrap = None
        self.mask = None
        self.nmask = None
        self.init_latent = None
        self.steps = None
        """number of steps as specified by user in UI"""

        self.total_steps = None
        """expected number of calls to denoiser calculated from self.steps and specifics of the selected sampler"""

        self.step = 0
        self.image_cfg_scale = None
        self.padded_cond_uncond = False
        self.padded_cond_uncond_v0 = False
        self.sampler = sampler
        self.model_wrap = None
        self.p = None

        self.cond_scale_miltiplier = 1.0
        self._cfg_index_layout = None
        self._cfg_index_cache = {}

        self.need_last_noise_uncond = False
        self.last_noise_uncond = None

        # NOTE: masking before denoising can cause the original latents to be oversmoothed
        # as the original latents do not have noise
        self.mask_before_denoising = False

    def batch_index_tensors(self, repeats, denoised_image_indexes, device):
        """Return immutable CFG batch indexes, reusing them while the prompt layout is stable."""

        layout = (tuple(repeats), tuple(denoised_image_indexes))
        if layout != self._cfg_index_layout:
            self._cfg_index_layout = layout
            self._cfg_index_cache.clear()

        device = torch.device(device)
        device_key = (device.type, device.index)
        cached = self._cfg_index_cache.get(device_key)
        if cached is None:
            batch_index = torch.repeat_interleave(
                torch.arange(len(repeats), device=device),
                torch.as_tensor(repeats, device=device),
            )
            denoised_index = torch.as_tensor(denoised_image_indexes, device=device, dtype=torch.long)
            cached = (batch_index, denoised_index)
            self._cfg_index_cache[device_key] = cached

        return cached

    @property
    def inner_model(self):
        raise NotImplementedError()

    def combine_denoised(self, x_out, conds_list, uncond, cond_scale):
        denoised_uncond = x_out[-uncond.shape[0]:]
        denoised = torch.clone(denoised_uncond)

        for i, conds in enumerate(conds_list):
            for cond_index, weight in conds:
                denoised[i] += (x_out[cond_index] - denoised_uncond[i]) * (weight * cond_scale)

        return denoised

    def combine_denoised_for_edit_model(self, x_out, cond_scale):
        out_cond, out_img_cond, out_uncond = x_out.chunk(3)
        denoised = out_uncond + cond_scale * (out_cond - out_img_cond) + self.image_cfg_scale * (out_img_cond - out_uncond)

        return denoised

    def get_pred_x0(self, x_in, x_out, sigma):
        return x_out

    def update_inner_model(self):
        self.model_wrap = None

        c, uc = self.p.get_conds()
        self.sampler.sampler_extra_args['cond'] = c
        self.sampler.sampler_extra_args['uncond'] = uc

    def run_inner_model(self, x, sigma, cond):
        from modules import openclaw_cuda_graphs
        return openclaw_cuda_graphs.run(self.inner_model, x, sigma, cond, denoiser=self)

    def pad_cond_uncond(self, cond, uncond):
        empty = shared.sd_model.cond_stage_model_empty_prompt
        num_repeats = (cond.shape[1] - uncond.shape[1]) // empty.shape[1]

        if num_repeats < 0:
            cond = pad_cond(cond, -num_repeats, empty)
            self.padded_cond_uncond = True
        elif num_repeats > 0:
            uncond = pad_cond(uncond, num_repeats, empty)
            self.padded_cond_uncond = True

        return cond, uncond

    def pad_cond_uncond_v0(self, cond, uncond):
        """
        Pads the 'uncond' tensor to match the shape of the 'cond' tensor.

        If 'uncond' is a dictionary, it is assumed that the 'crossattn' key holds the tensor to be padded.
        If 'uncond' is a tensor, it is padded directly.

        If the number of columns in 'uncond' is less than the number of columns in 'cond', the last column of 'uncond'
        is repeated to match the number of columns in 'cond'.

        If the number of columns in 'uncond' is greater than the number of columns in 'cond', 'uncond' is truncated
        to match the number of columns in 'cond'.

        Args:
            cond (torch.Tensor or DictWithShape): The condition tensor to match the shape of 'uncond'.
            uncond (torch.Tensor or DictWithShape): The tensor to be padded, or a dictionary containing the tensor to be padded.

        Returns:
            tuple: A tuple containing the 'cond' tensor and the padded 'uncond' tensor.

        Note:
            This is the padding that was always used in DDIM before version 1.6.0
        """

        is_dict_cond = isinstance(uncond, dict)
        uncond_vec = uncond['crossattn'] if is_dict_cond else uncond

        if uncond_vec.shape[1] < cond.shape[1]:
            last_vector = uncond_vec[:, -1:]
            last_vector_repeated = last_vector.repeat([1, cond.shape[1] - uncond_vec.shape[1], 1])
            uncond_vec = torch.hstack([uncond_vec, last_vector_repeated])
            self.padded_cond_uncond_v0 = True
        elif uncond_vec.shape[1] > cond.shape[1]:
            uncond_vec = uncond_vec[:, :cond.shape[1]]
            self.padded_cond_uncond_v0 = True

        if is_dict_cond:
            uncond = copy_condition_dict(uncond)
            uncond['crossattn'] = uncond_vec
        else:
            uncond = uncond_vec

        return cond, uncond

    def forward(self, x, sigma, uncond, cond, cond_scale, s_min_uncond, image_cond):
        if state.interrupted or state.skipped:
            raise sd_samplers_common.InterruptedException

        if sd_samplers_common.apply_refiner(self, sigma):
            cond = self.sampler.sampler_extra_args['cond']
            uncond = self.sampler.sampler_extra_args['uncond']

        # at self.image_cfg_scale == 1.0 produced results for edit model are the same as with normal sampling,
        # so is_edit_model is set to False to support AND composition.
        is_edit_model = shared.sd_model.cond_stage_key == "edit" and self.image_cfg_scale is not None and self.image_cfg_scale != 1.0

        conds_list, tensor = prompt_parser.reconstruct_multicond_batch(cond, self.step)
        uncond = prompt_parser.reconstruct_cond_batch(uncond, self.step)

        assert not is_edit_model or all(len(conds) == 1 for conds in conds_list), "AND is not supported for InstructPix2Pix checkpoint (unless using Image CFG scale = 1.0)"

        # If we use masks, blending between the denoised and original latent images occurs here.
        def apply_blend(current_latent):
            blended_latent = current_latent * self.nmask + self.init_latent * self.mask

            if self.p.scripts is not None:
                from modules import scripts
                mba = scripts.MaskBlendArgs(current_latent, self.nmask, self.init_latent, self.mask, blended_latent, denoiser=self, sigma=sigma)
                self.p.scripts.on_mask_blend(self.p, mba)
                blended_latent = mba.blended_latent

            return blended_latent

        # Blend in the original latents (before)
        if self.mask_before_denoising and self.mask is not None:
            x = apply_blend(x)

        batch_size = len(conds_list)
        repeats = [len(conds_list[i]) for i in range(batch_size)]
        denoised_image_indexes = [item[0][0] for item in conds_list]

        if shared.sd_model.model.conditioning_key == "crossattn-adm":
            image_uncond = torch.zeros_like(image_cond)
            make_condition_dict = lambda c_crossattn, c_adm: {"c_crossattn": [c_crossattn], "c_adm": c_adm}
        else:
            image_uncond = image_cond
            if isinstance(uncond, dict):
                make_condition_dict = lambda c_crossattn, c_concat: {**c_crossattn, "c_concat": [c_concat]}
            else:
                make_condition_dict = lambda c_crossattn, c_concat: {"c_crossattn": [c_crossattn], "c_concat": [c_concat]}

        x_repeat_index, _ = self.batch_index_tensors(repeats, denoised_image_indexes, x.device)
        sigma_repeat_index, _ = self.batch_index_tensors(repeats, denoised_image_indexes, sigma.device)
        image_cond_repeat_index, _ = self.batch_index_tensors(repeats, denoised_image_indexes, image_cond.device)

        x_repeated = x.index_select(0, x_repeat_index)
        sigma_repeated = sigma.index_select(0, sigma_repeat_index)
        image_cond_repeated = image_cond.index_select(0, image_cond_repeat_index)

        if not is_edit_model:
            x_in = torch.cat([x_repeated, x])
            sigma_in = torch.cat([sigma_repeated, sigma])
            image_cond_in = torch.cat([image_cond_repeated, image_uncond])
        else:
            x_in = torch.cat([x_repeated, x, x])
            sigma_in = torch.cat([sigma_repeated, sigma, sigma])
            image_cond_in = torch.cat([image_cond_repeated, image_uncond, torch.zeros_like(self.init_latent)])

        denoiser_params = CFGDenoiserParams(x_in, image_cond_in, sigma_in, state.sampling_step, state.sampling_steps, tensor, uncond, self)
        cfg_denoiser_callback(denoiser_params)
        x_in = denoiser_params.x
        image_cond_in = denoiser_params.image_cond
        sigma_in = denoiser_params.sigma
        tensor = denoiser_params.text_cond
        uncond = denoiser_params.text_uncond
        skip_uncond = False
        can_skip_uncond = not self.need_last_noise_uncond

        if can_skip_uncond and shared.opts.skip_early_cond != 0. and self.step / self.total_steps <= shared.opts.skip_early_cond:
            skip_uncond = True
            self.p.extra_generation_params["Skip Early CFG"] = shared.opts.skip_early_cond
        elif can_skip_uncond and (self.step % 2 or shared.opts.s_min_uncond_all) and s_min_uncond > 0 and sigma[0] < s_min_uncond and not is_edit_model:
            skip_uncond = True
            self.p.extra_generation_params["NGMS"] = s_min_uncond
            if shared.opts.s_min_uncond_all:
                self.p.extra_generation_params["NGMS all steps"] = shared.opts.s_min_uncond_all

        if skip_uncond:
            x_in = x_in[:-batch_size]
            sigma_in = sigma_in[:-batch_size]
            image_cond_in = image_cond_in[:-batch_size]

        self.padded_cond_uncond = False
        self.padded_cond_uncond_v0 = False
        if shared.opts.pad_cond_uncond_v0 and tensor.shape[1] != uncond.shape[1]:
            tensor, uncond = self.pad_cond_uncond_v0(tensor, uncond)
        elif shared.opts.pad_cond_uncond and tensor.shape[1] != uncond.shape[1]:
            tensor, uncond = self.pad_cond_uncond(tensor, uncond)

        if tensor.shape[1] == uncond.shape[1] or skip_uncond:
            if is_edit_model:
                cond_in = catenate_conds([tensor, uncond, uncond])
            elif skip_uncond:
                cond_in = tensor
            else:
                cond_in = catenate_conds([tensor, uncond])

            if shared.opts.batch_cond_uncond:
                x_out = self.run_inner_model(x_in, sigma_in, make_condition_dict(cond_in, image_cond_in))
            else:
                x_out = torch.zeros_like(x_in)
                for batch_offset in range(0, x_out.shape[0], batch_size):
                    a = batch_offset
                    b = a + batch_size
                    x_out[a:b] = self.run_inner_model(x_in[a:b], sigma_in[a:b], make_condition_dict(subscript_cond(cond_in, a, b), image_cond_in[a:b]))
        else:
            x_out = torch.zeros_like(x_in)
            model_batch_size = batch_size*2 if shared.opts.batch_cond_uncond else batch_size
            for batch_offset in range(0, tensor.shape[0], model_batch_size):
                a = batch_offset
                b = min(a + model_batch_size, tensor.shape[0])
                c_crossattn = subscript_cond(tensor, a, b)
                x_out[a:b] = self.run_inner_model(x_in[a:b], sigma_in[a:b], make_condition_dict(c_crossattn, image_cond_in[a:b]))

            if is_edit_model:
                uncond_in = catenate_conds([uncond, uncond])
                edit_uncond_start = tensor.shape[0]
                for batch_offset in range(0, uncond_in.shape[0], model_batch_size):
                    a = batch_offset
                    b = min(a + model_batch_size, uncond_in.shape[0])
                    x_slice = slice(edit_uncond_start + a, edit_uncond_start + b)
                    x_out[x_slice] = self.run_inner_model(x_in[x_slice], sigma_in[x_slice], make_condition_dict(subscript_cond(uncond_in, a, b), image_cond_in[x_slice]))
            elif not skip_uncond:
                x_out[-uncond.shape[0]:] = self.run_inner_model(x_in[-uncond.shape[0]:], sigma_in[-uncond.shape[0]:], make_condition_dict(uncond, image_cond_in[-uncond.shape[0]:]))

        _, denoised_image_indexes_tensor = self.batch_index_tensors(repeats, denoised_image_indexes, x_out.device)
        if skip_uncond:
            fake_uncond = x_out.index_select(0, denoised_image_indexes_tensor)
            x_out = torch.cat([x_out, fake_uncond])  # we skipped uncond denoising, so we put cond-denoised image to where the uncond-denoised image should be

        denoised_params = CFGDenoisedParams(x_out, state.sampling_step, state.sampling_steps, self.inner_model)
        cfg_denoised_callback(denoised_params)
        x_out = denoised_params.x

        if self.need_last_noise_uncond:
            self.last_noise_uncond = torch.clone(x_out[-uncond.shape[0]:])

        if is_edit_model:
            denoised = self.combine_denoised_for_edit_model(x_out, cond_scale * self.cond_scale_miltiplier)
        elif skip_uncond:
            denoised = self.combine_denoised(x_out, conds_list, uncond, 1.0)
        else:
            denoised = self.combine_denoised(x_out, conds_list, uncond, cond_scale * self.cond_scale_miltiplier)

        # Blend in the original latents (after)
        if not self.mask_before_denoising and self.mask is not None:
            denoised = apply_blend(denoised)

        _, denoised_image_indexes_x_tensor = self.batch_index_tensors(repeats, denoised_image_indexes, x_in.device)
        x_in_denoised = x_in.index_select(0, denoised_image_indexes_x_tensor)
        x_out_denoised = x_out.index_select(0, denoised_image_indexes_tensor)
        self.sampler.last_latent = self.get_pred_x0(x_in_denoised, x_out_denoised, sigma)

        if opts.live_preview_content == "Prompt":
            preview = self.sampler.last_latent
        elif opts.live_preview_content == "Negative prompt":
            preview = self.get_pred_x0(x_in[-uncond.shape[0]:], x_out[-uncond.shape[0]:], sigma)
        else:
            _, denoised_image_indexes_preview = self.batch_index_tensors(repeats, denoised_image_indexes, denoised.device)
            preview = self.get_pred_x0(x_in_denoised, denoised.index_select(0, denoised_image_indexes_preview), sigma)

        sd_samplers_common.store_latent(preview)

        after_cfg_callback_params = AfterCFGCallbackParams(denoised, state.sampling_step, state.sampling_steps)
        cfg_after_cfg_callback(after_cfg_callback_params)
        denoised = after_cfg_callback_params.x

        self.step += 1
        return denoised

