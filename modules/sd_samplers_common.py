import inspect
import math
from collections import namedtuple
import torch
from PIL import Image
from modules import devices, images, sd_vae_approx, sd_vae_taesd, shared
from modules.sd_unet_row_memo import tensor_version
from modules.shared import opts, state
import k_diffusion.sampling


SamplerDataTuple = namedtuple('SamplerData', ['name', 'constructor', 'aliases', 'options'])


class SamplerData(SamplerDataTuple):
    def total_steps(self, steps):
        if self.options.get("second_order", False):
            steps = steps * 2

        return steps


def _floor_step_count(value):
    """floor() of a step count computed from a decimal denoising strength.

    Decimal strengths are not exact binary fractions, so a quotient or product that is mathematically an integer can
    come out a few ulp below it (33 / 0.55 == 59.99999999999999, 0.29 * 100 == 28.999999999999996) and int() would
    drop a whole step. Any value within 1e-9 of an integer is that integer: genuine fractions of these step counts are
    at least ~1e-3 away from one, and the float64 error stays far below 1e-9."""
    return math.floor(value + 1e-9)


def setup_img2img_steps(p, steps=None):
    if opts.img2img_fix_steps or steps is not None:
        requested_steps = (steps or p.steps)
        steps = _floor_step_count(requested_steps / min(p.denoising_strength, 0.999)) if p.denoising_strength > 0 else 0
        t_enc = requested_steps - 1
    else:
        steps = p.steps
        t_enc = _floor_step_count(min(p.denoising_strength, 0.999) * steps)

    return steps, t_enc


approximation_indexes = {"Full": 0, "Approx NN": 1, "Approx cheap": 2, "TAESD": 3}


def samples_to_images_tensor(sample, approximation=None, model=None):
    """Transforms 4-channel latent space images into 3-channel RGB image tensors, with values in range [-1, 1]."""

    if approximation is None or (shared.state.interrupted and opts.live_preview_fast_interrupt):
        approximation = approximation_indexes.get(opts.show_progress_type, 0)

        from modules import lowvram
        if approximation == 0 and lowvram.is_enabled(shared.sd_model) and not shared.opts.live_preview_allow_lowvram_full:
            approximation = 1

    if approximation == 2:
        x_sample = sd_vae_approx.cheap_approximation(sample)
    elif approximation == 1:
        x_sample = sd_vae_approx.model()(sample.to(devices.device, devices.dtype)).detach()
    elif approximation == 3:
        x_sample = sd_vae_taesd.decoder_model()(sample.to(devices.device, devices.dtype)).detach()
        x_sample = x_sample * 2 - 1
    else:
        if model is None:
            model = shared.sd_model
        with torch.no_grad(), devices.without_autocast(): # fixes an issue with unstable VAEs that are flaky even in fp32
            x_sample = model.decode_first_stage(vae_decode_input(model, sample))

    return x_sample


def float_images_to_uint8(tensor):
    """Quantize clamped [0, 1] CHW image tensors (one image or a batch) to HWC uint8 tensors on their device.

    8-bit sRGB code values are round(255 * E') (IEC 61966-2-1); truncation biases every pixel by -0.5 code and maps
    a quarter of exact uint8 round trips (u / 255 * 2 - 1 -> u) to u - 1. The product is formed in float32: bf16/fp16
    cannot hold 255 * x exactly, and bf16 tensors have no NumPy dtype.
    """
    return tensor.float().mul(255.0).round_().to(torch.uint8).movedim(-3, -1).contiguous()


def single_sample_to_image(sample, approximation=None):
    x_sample = samples_to_images_tensor(sample.unsqueeze(0), approximation)[0].float() * 0.5 + 0.5
    x_sample = torch.clamp(x_sample, min=0.0, max=1.0)

    return Image.fromarray(float_images_to_uint8(x_sample).cpu().numpy())


def vae_decode_input(model, x):
    """The latent x as model.decode_first_stage takes it. SDXL's (sd_models_xl.decode_first_stage) scales it in float32
    and rounds it to the VAE dtype once, so it gets x unrounded; the others scale in the VAE dtype they are given."""
    if getattr(model, "decode_first_stage_takes_float32", False):
        return x.float()
    return x.to(model.first_stage_model.dtype)


def decode_first_stage(model, x):
    x = x.float() if getattr(model, "decode_first_stage_takes_float32", False) else x.to(devices.dtype_vae)
    approx_index = approximation_indexes.get(opts.sd_vae_decode_method, 0)
    from modules import openclaw_vae_decode_graphs
    decoded = openclaw_vae_decode_graphs.run(model, x, approx_index)
    if decoded is not None:
        return decoded
    return samples_to_images_tensor(x, approx_index, model)


def sample_to_image(samples, index=0, approximation=None):
    return single_sample_to_image(samples[index], approximation)


def samples_to_image_grid(samples, approximation=None):
    return images.image_grid([single_sample_to_image(sample, approximation) for sample in samples])


def images_tensor_to_samples(image, approximation=None, model=None):
    '''image[0, 1] -> latent'''
    if approximation is None:
        approximation = approximation_indexes.get(opts.sd_vae_encode_method, 0)

    if approximation == 3:
        image = image.to(devices.device, devices.dtype)
        x_latent = sd_vae_taesd.encoder_model()(image)
    else:
        if model is None:
            model = shared.sd_model
        model.first_stage_model.to(devices.dtype_vae)

        # [0, 1] -> [-1, 1] in the caller's dtype, then one cast to the VAE's: in a 16-bit VAE dtype the affine rounded
        # twice (u / 255 and then 2x - 1), off by up to half a code. A caller passing the VAE dtype gets the old result.
        image = (image.to(shared.device) * 2 - 1).to(dtype=devices.dtype_vae)
        if len(image) > 1:
            try:
                x_latent = model.get_first_stage_encoding(model.encode_first_stage(image))
            except torch.cuda.OutOfMemoryError:
                devices.torch_gc()
                x_latent = torch.stack([
                    model.get_first_stage_encoding(
                        model.encode_first_stage(torch.unsqueeze(img, 0))
                    )[0]
                    for img in image
                ])
        else:
            x_latent = model.get_first_stage_encoding(model.encode_first_stage(image))

    return x_latent


def store_latent(decoded):
    state.current_latent = decoded

    if opts.live_previews_enable and opts.show_progress_every_n_steps > 0 and shared.state.sampling_step % opts.show_progress_every_n_steps == 0:
        if not shared.parallel_processing_allowed:
            shared.state.assign_current_image(sample_to_image(decoded))


def is_sampler_using_eta_noise_seed_delta(p):
    """returns whether sampler from config will use eta noise seed delta for image creation"""

    from modules import sd_samplers
    sampler_config = sd_samplers.find_sampler_config(p.sampler_name)

    eta = p.eta

    if eta is None and p.sampler is not None:
        eta = p.sampler.eta

    if eta is None and sampler_config is not None:
        eta = 0 if sampler_config.options.get("default_eta_is_0", False) else 1.0

    if eta == 0:
        return False

    return sampler_config.options.get("uses_ensd", False)


class InterruptedException(BaseException):
    pass


def replace_torchsde_browinan():
    import torchsde._brownian.brownian_interval

    def torchsde_randn(size, dtype, device, seed):
        return devices.randn_local(seed, size).to(device=device, dtype=dtype)

    torchsde._brownian.brownian_interval._randn = torchsde_randn


replace_torchsde_browinan()


def cpu_sigmas(model_wrap):
    """The k-diffusion wrapper's sigma table (model_wrap.sigmas, on the device) as a CPU tensor, copied once per table.

    One wrapper is built per sampling run (and per refiner switch), so get_sigmas, its schedule cache key and the
    refiner switch read the table without a device-to-host copy and synchronization on every call. A replaced table,
    or one mutated in place where the version counter is tracked, is copied again. Generation's inference tensors have
    no version counter; nothing mutates a wrapper's table in place inside inference mode (per-request schedule options
    build a new wrapper; see openclaw_cuda_graphs._schedule_signature)."""
    sigmas = model_wrap.sigmas
    version = tensor_version(sigmas)
    cached = getattr(model_wrap, "openclaw_cpu_sigmas", None)
    if cached is None or cached[0] is not sigmas or cached[1] != version:
        cached = (sigmas, version, sigmas.detach().to(devices.cpu))
        model_wrap.openclaw_cpu_sigmas = cached
    return cached[2]


def apply_refiner(cfg_denoiser, sigma=None):
    if opts.refiner_switch_by_sample_steps or sigma is None:
        completed_ratio = cfg_denoiser.step / cfg_denoiser.total_steps
        cfg_denoiser.p.extra_generation_params["Refiner switch by sampling steps"] = True

    elif cfg_denoiser.p.refiner_checkpoint_info is None or shared.sd_model.sd_checkpoint_info == cfg_denoiser.p.refiner_checkpoint_info:
        return False  # no refiner, or switched to it already: skip reading sigma, which only feeds the switch decision

    else:
        # torch.max(sigma) only to handle rare case where we might have different sigmas in the same batch. Read once
        # to the host and matched against the CPU copy of the table: the same float32 arithmetic as on the device.
        try:
            sigmas = cpu_sigmas(cfg_denoiser.inner_model)
        except AttributeError:  # for samplers that don't use sigmas (DDIM) sigma is actually the timestep
            timestep = torch.max(sigma).to(dtype=int)
        else:
            timestep = torch.argmin(torch.abs(sigmas - torch.max(sigma).to(devices.cpu)))
        completed_ratio = (999 - timestep) / 1000

    refiner_switch_at = cfg_denoiser.p.refiner_switch_at
    refiner_checkpoint_info = cfg_denoiser.p.refiner_checkpoint_info

    if refiner_switch_at is not None and completed_ratio < refiner_switch_at:
        return False

    if refiner_checkpoint_info is None or shared.sd_model.sd_checkpoint_info == refiner_checkpoint_info:
        return False

    if getattr(cfg_denoiser.p, "enable_hr", False):
        is_second_pass = cfg_denoiser.p.is_hr_pass

        if opts.hires_fix_refiner_pass == "first pass" and is_second_pass:
            return False

        if opts.hires_fix_refiner_pass == "second pass" and not is_second_pass:
            return False

        if opts.hires_fix_refiner_pass != "second pass":
            cfg_denoiser.p.extra_generation_params['Hires refiner'] = opts.hires_fix_refiner_pass

    cfg_denoiser.p.extra_generation_params['Refiner'] = refiner_checkpoint_info.short_title
    cfg_denoiser.p.extra_generation_params['Refiner switch at'] = refiner_switch_at

    from modules import sd_models, extra_networks
    with sd_models.SkipWritingToConfig():
        sd_models.reload_model_weights(info=refiner_checkpoint_info)

    # The load computes the refiner's empty-prompt padding with sd_models.get_empty_cond, which resets every extra
    # network so that no LoRA reaches the padding. The refiner stage runs with the request's networks again.
    active_extra_network_data = getattr(cfg_denoiser.p, "_active_extra_network_data", None)
    if active_extra_network_data is not None:
        with devices.autocast():
            extra_networks.activate(cfg_denoiser.p, active_extra_network_data)

    devices.torch_gc()
    cfg_denoiser.p.setup_conds()
    cfg_denoiser.update_inner_model()

    return True


class TorchHijack:
    """This is here to replace torch.randn_like of k-diffusion.

    k-diffusion has random_sampler argument for most samplers, but not for all, so
    this is needed to properly replace every use of torch.randn_like.

    We need to replace to make images generated in batches to be same as images generated individually."""

    def __init__(self, p):
        self.rng = p.rng

    def __getattr__(self, item):
        if item == 'randn_like':
            return self.randn_like

        if hasattr(torch, item):
            return getattr(torch, item)

        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{item}'")

    def randn_like(self, x):
        return self.rng.next()


sigma_params_defaults = {'s_churn': 0.0, 's_tmin': 0.0, 's_tmax': float('inf'), 's_noise': 1.0}
"""k-diffusion's defaults for its stochasticity parameters: a sampler function is called with these when not given."""

sigma_params_infotext = {'s_churn': 'Sigma churn', 's_tmin': 'Sigma tmin', 's_tmax': 'Sigma tmax', 's_noise': 'Sigma noise'}


def sigma_params_kwargs(p, param_names):
    """Keyword arguments for the stochasticity parameters (s_churn, s_tmin, s_tmax, s_noise) a sampler function takes.

    The values are p's: the request's, else the settings' (StableDiffusionProcessing.fill_fields_from_opts; s_tmax 0 is
    infinity). A value is passed, and recorded in infotext, only where it differs from k-diffusion's default, so the
    default call stays exactly the sampler function's own."""
    kwargs = {}
    for name in param_names:
        value = getattr(p, name, None)
        if value is None:
            value = getattr(opts, name)
        if name == 's_tmax':
            value = value or float('inf')

        if value != sigma_params_defaults[name]:
            kwargs[name] = value
            p.extra_generation_params[sigma_params_infotext[name]] = value

    return kwargs


class Sampler:
    def __init__(self, funcname):
        self.funcname = funcname
        self.func = funcname
        self.extra_params = []
        self.stop_at = None
        self.eta = None
        self.config: SamplerData = None  # set by the function calling the constructor
        self.last_latent = None
        self.s_min_uncond = None

        self.eta_option_field = 'eta_ancestral'
        self.eta_infotext_field = 'Eta'
        self.eta_default = 1.0

        self.conditioning_key = getattr(shared.sd_model.model, 'conditioning_key', 'crossattn')

        self.p = None
        self.model_wrap_cfg = None
        self.sampler_extra_args = None
        self.options = {}

    def callback_state(self, d):
        step = d['i']

        if self.stop_at is not None and step > self.stop_at:
            raise InterruptedException

        state.sampling_step = step
        shared.total_tqdm.update()

    def launch_sampling(self, steps, func):
        self.model_wrap_cfg.steps = steps
        self.model_wrap_cfg.total_steps = self.config.total_steps(steps)
        state.sampling_steps = steps
        state.sampling_step = 0

        try:
            return func()
        except RecursionError:
            print(
                'Encountered RecursionError during sampling, returning last latent. '
                'rho >5 with a polyexponential scheduler may cause this error. '
                'You should try to use a smaller rho value instead.'
            )
            return self.last_latent
        except InterruptedException:
            return self.last_latent

    def initialize(self, p) -> dict:
        self.p = p
        self.model_wrap_cfg.p = p
        self.model_wrap_cfg.mask = p.mask if hasattr(p, 'mask') else None
        self.model_wrap_cfg.nmask = p.nmask if hasattr(p, 'nmask') else None
        self.model_wrap_cfg.step = 0
        self.model_wrap_cfg.image_cfg_scale = getattr(p, 'image_cfg_scale', None)
        self.eta = p.eta if p.eta is not None else getattr(opts, self.eta_option_field, 0.0)
        self.s_min_uncond = getattr(p, 's_min_uncond', 0.0)

        k_diffusion.sampling.torch = TorchHijack(p)

        parameters = inspect.signature(self.func).parameters
        extra_params_kwargs = sigma_params_kwargs(p, [name for name in self.extra_params if name in parameters])

        if 'eta' in parameters:
            if self.eta != self.eta_default:
                p.extra_generation_params[self.eta_infotext_field] = self.eta

            extra_params_kwargs['eta'] = self.eta

        return extra_params_kwargs

    def create_noise_sampler(self, x, sigmas, p):
        """For DPM++ SDE: manually create noise sampler to enable deterministic results across different batch sizes"""
        if shared.opts.no_dpmpp_sde_batch_determinism:
            return None

        from k_diffusion.sampling import BrownianTreeNoiseSampler
        sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
        current_iter_seeds = p.all_seeds[p.iteration * p.batch_size:(p.iteration + 1) * p.batch_size]
        return BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=current_iter_seeds)

    def set_sampler_extra_args(self, p, conditioning, unconditional_conditioning, image_conditioning):
        self.sampler_extra_args = {
            'cond': conditioning,
            'image_cond': image_conditioning,
            'uncond': unconditional_conditioning,
            'cond_scale': p.cfg_scale,
            's_min_uncond': self.s_min_uncond,
        }
        return self.sampler_extra_args

    def sample(self, p, x, conditioning, unconditional_conditioning, steps=None, image_conditioning=None):
        raise NotImplementedError()

    def sample_img2img(self, p, x, noise, conditioning, unconditional_conditioning, steps=None, image_conditioning=None):
        raise NotImplementedError()

    def add_infotext(self, p):
        if self.model_wrap_cfg.padded_cond_uncond:
            p.extra_generation_params["Pad conds"] = True

        if self.model_wrap_cfg.padded_cond_uncond_v0:
            p.extra_generation_params["Pad conds v0"] = True
