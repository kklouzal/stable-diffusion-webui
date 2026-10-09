"""DDIM, DDIM CFG++, PLMS and UniPC against test-local oracles.

DDIM, DDIM CFG++ and PLMS are checked against loops written the way ldm's reference DDIMSampler/PLMSSampler run them
(every timestep, down to alphas_cumprod[0]). UniPC's oracle copies modules/sd_samplers_timesteps_impl.py (UniPCCFG/unipc
at c98dc893) and modules/models/diffusion/uni_pc/uni_pc.py (the discrete NoiseScheduleVP and the UniPC multistep solver), reduced to
the paths the sampler table runs: data prediction without thresholding, multistep, no final denoise-to-zero. They
share no code with the samplers under test. Both sides get the same fake CFG denoiser, inputs and noise; the result
and every callback payload must be bit-identical (torch.equal).
"""

import itertools
import types

import pytest
import torch

from test.helpers import init_shared

shared = init_shared()

# sd_samplers first: importing the sampler modules on their own runs into an import cycle.
from modules import sd_samplers, sd_samplers_timesteps, sd_samplers_timesteps_impl  # noqa: E402,F401

import k_diffusion.sampling  # noqa: E402  (importable once modules.paths has run)


class FakeDenoiser:
    """Stands in for CFGDenoiserTimesteps: an eps prediction mixed by CFG, plus the CFG++ attributes it honours."""

    def __init__(self, alphas_cumprod, weights):
        self.inner_model = types.SimpleNamespace(inner_model=types.SimpleNamespace(alphas_cumprod=alphas_cumprod))
        self.weights = weights
        self.cond_scale_miltiplier = 1.0
        self.need_last_noise_uncond = False
        self.last_noise_uncond = None

    def __call__(self, x, t, cond, uncond, cond_scale, image_cond=None, s_min_uncond=0.0):
        t = t.to(x.dtype).reshape(-1, 1, 1, 1) / 1000
        noise_cond = torch.einsum("bchw,cd->bdhw", x, self.weights) * (1 - t) + cond
        noise_uncond = torch.tanh(x) * 0.25 + uncond * t
        if self.need_last_noise_uncond:
            self.last_noise_uncond = noise_uncond
        return noise_uncond + (noise_cond - noise_uncond) * (cond_scale * self.cond_scale_miltiplier)


class SeededNoise:
    """Replaces k_diffusion.sampling.torch for one run, like sd_samplers_common.TorchHijack."""

    def __init__(self, seed):
        self.generator = torch.Generator().manual_seed(seed)

    def randn_like(self, x):
        return torch.randn(x.shape, generator=self.generator, dtype=x.dtype)


def make_inputs(dtype):
    generator = torch.Generator().manual_seed(1234)
    x = torch.randn(2, 4, 8, 8, generator=generator).to(dtype)
    extra_args = {
        "cond": torch.randn(2, 4, 8, 8, generator=generator).to(dtype) * 0.1,
        "uncond": torch.randn(2, 4, 8, 8, generator=generator).to(dtype) * 0.1,
        "cond_scale": 7.0,
        "image_cond": None,
        "s_min_uncond": 0.0,
    }
    weights = torch.randn(4, 4, generator=generator).to(dtype) * 0.1
    alphas_cumprod = torch.cumprod(1 - torch.linspace(0.00085 ** 0.5, 0.012 ** 0.5, 1000, dtype=torch.float64) ** 2, 0).float()
    return x, extra_args, alphas_cumprod, weights


def run_sampler(monkeypatch, sampler, x, timesteps, extra_args, alphas_cumprod, weights, **kwargs):
    monkeypatch.setattr(k_diffusion.sampling, "torch", SeededNoise(99))
    model = FakeDenoiser(alphas_cumprod, weights)
    callbacks = []

    def callback(d):
        callbacks.append({key: value.clone() if torch.is_tensor(value) else value for key, value in d.items()})

    out = sampler(model, x.clone(), timesteps, extra_args=extra_args, callback=callback, disable=True, **kwargs)
    return out, callbacks, (model.cond_scale_miltiplier, model.need_last_noise_uncond)


def assert_same_run(actual, expected):
    actual_x, actual_callbacks, actual_model_state = actual
    expected_x, expected_callbacks, expected_model_state = expected
    assert torch.equal(actual_x, expected_x)
    assert actual_model_state == expected_model_state
    assert len(actual_callbacks) == len(expected_callbacks)
    for got, want in zip(actual_callbacks, expected_callbacks, strict=True):
        assert got.keys() == want.keys()
        for key, value in want.items():
            if torch.is_tensor(value):
                assert torch.equal(got[key], value), key
            else:
                assert got[key] == value, key


# --- oracle: DDIM, DDIM CFG++ and PLMS, structured as ldm's DDIMSampler.ddim_sampling/decode and PLMSSampler ---
# (repositories/stable-diffusion-stability-ai ldm/models/diffusion/ddim.py, plms.py; make_ddim_sampling_parameters in
# ldm/modules/diffusionmodules/util.py): every timestep is visited, index = total_steps - i - 1, and the step from the
# first timestep goes to alphas_cumprod[0]. CFG++ follows DDIM with the uncond noise as the direction.

def oracle_ddim_parameters(alphas_cumprod, timesteps, eta):
    """make_ddim_sampling_parameters: a_t, a_prev = [alphacums[0]] + alphacums[ddim_timesteps[:-1]], sigmas (float64)."""
    alphas = alphas_cumprod[timesteps]
    alphas_prev = torch.cat([alphas_cumprod[:1], alphas_cumprod[timesteps[:-1]]]).double()
    sigmas = eta * torch.sqrt((1 - alphas_prev) / (1 - alphas.double()) * (1 - alphas.double() / alphas_prev))
    return alphas, alphas_prev, sigmas, torch.sqrt(1 - alphas)


def oracle_full(x, value):
    """torch.full((b, 1, 1, 1), schedule[index]) in x's dtype, as the reference's per-step constants."""
    return torch.full((x.shape[0], 1, 1, 1), float(value), dtype=x.dtype)


@torch.no_grad()
def oracle_ddim_loop(model, x, timesteps, extra_args, callback, eta, cfgpp):
    alphas, alphas_prev, sigmas, sqrt_one_minus_alphas = oracle_ddim_parameters(model.inner_model.inner_model.alphas_cumprod, timesteps, eta)
    if cfgpp:
        model.cond_scale_miltiplier = 1 / 12.5
        model.need_last_noise_uncond = True

    total_steps = timesteps.shape[0]
    for i, step in enumerate(torch.flip(timesteps, [0])):
        index = total_steps - i - 1
        e_t = model(x, torch.full((x.shape[0],), int(step), dtype=x.dtype), **extra_args)
        direction = model.last_noise_uncond if cfgpp else e_t

        a_t, a_prev = oracle_full(x, alphas[index]), oracle_full(x, alphas_prev[index])
        sigma_t, sqrt_one_minus_at = oracle_full(x, sigmas[index]), oracle_full(x, sqrt_one_minus_alphas[index])

        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        dir_xt = (1. - a_prev - sigma_t ** 2).sqrt() * direction
        noise = sigma_t * k_diffusion.sampling.torch.randn_like(x)
        x = a_prev.sqrt() * pred_x0 + dir_xt + noise

        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': 0, 'sigma_hat': 0, 'denoised': pred_x0})

    return x


def oracle_ddim(model, x, timesteps, extra_args=None, callback=None, disable=None, eta=0.0):
    return oracle_ddim_loop(model, x, timesteps, extra_args or {}, callback, eta, cfgpp=False)


def oracle_ddim_cfgpp(model, x, timesteps, extra_args=None, callback=None, disable=None, eta=0.0):
    return oracle_ddim_loop(model, x, timesteps, extra_args or {}, callback, eta, cfgpp=True)


@torch.no_grad()
def oracle_plms(model, x, timesteps, extra_args=None, callback=None, disable=None):
    alphas, alphas_prev, _sigmas, sqrt_one_minus_alphas = oracle_ddim_parameters(model.inner_model.inner_model.alphas_cumprod, timesteps, 0.0)
    extra_args = extra_args or {}
    time_range = torch.flip(timesteps, [0])
    total_steps = timesteps.shape[0]
    old_eps = []

    for i, step in enumerate(time_range):
        index = total_steps - i - 1
        ts = torch.full((x.shape[0],), int(step), dtype=x.dtype)
        ts_next = torch.full((x.shape[0],), int(time_range[min(i + 1, len(time_range) - 1)]), dtype=x.dtype)

        def get_x_prev_and_pred_x0(e_t, index):
            a_t, a_prev, sqrt_one_minus_at = oracle_full(x, alphas[index]), oracle_full(x, alphas_prev[index]), oracle_full(x, sqrt_one_minus_alphas[index])
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
            dir_xt = (1. - a_prev).sqrt() * e_t
            return a_prev.sqrt() * pred_x0 + dir_xt, pred_x0

        e_t = model(x, ts, **extra_args)
        if len(old_eps) == 0:
            # Pseudo Improved Euler (2nd order)
            x_prev, pred_x0 = get_x_prev_and_pred_x0(e_t, index)
            e_t_next = model(x_prev, ts_next, **extra_args)
            e_t_prime = (e_t + e_t_next) / 2
        elif len(old_eps) == 1:
            e_t_prime = (3 * e_t - old_eps[-1]) / 2
        elif len(old_eps) == 2:
            e_t_prime = (23 * e_t - 16 * old_eps[-1] + 5 * old_eps[-2]) / 12
        else:
            e_t_prime = (55 * e_t - 59 * old_eps[-1] + 37 * old_eps[-2] - 9 * old_eps[-3]) / 24
        x, pred_x0 = get_x_prev_and_pred_x0(e_t_prime, index)

        old_eps.append(e_t)
        if len(old_eps) >= 4:
            old_eps.pop(0)

        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': 0, 'sigma_hat': 0, 'denoised': pred_x0})

    return x


# --- oracle: UniPC (uni_pc.py and UniPCCFG/unipc at c98dc893, as unipc() runs them) ---

def oracle_interpolate_fn(x, xp, yp):
    N, K = x.shape[0], xp.shape[1]
    all_x = torch.cat([x.unsqueeze(2), xp.unsqueeze(0).repeat((N, 1, 1))], dim=2)
    sorted_all_x, x_indices = torch.sort(all_x, dim=2)
    x_idx = torch.argmin(x_indices, dim=2)
    cand_start_idx = x_idx - 1
    start_idx = torch.where(
        torch.eq(x_idx, 0),
        x_idx.new_tensor(1),
        torch.where(
            torch.eq(x_idx, K), x_idx.new_tensor(K - 2), cand_start_idx,
        ),
    )
    end_idx = torch.where(torch.eq(start_idx, cand_start_idx), start_idx + 2, start_idx + 1)
    start_x = torch.gather(sorted_all_x, dim=2, index=start_idx.unsqueeze(2)).squeeze(2)
    end_x = torch.gather(sorted_all_x, dim=2, index=end_idx.unsqueeze(2)).squeeze(2)
    start_idx2 = torch.where(
        torch.eq(x_idx, 0),
        x_idx.new_tensor(0),
        torch.where(
            torch.eq(x_idx, K), x_idx.new_tensor(K - 2), cand_start_idx,
        ),
    )
    y_positions_expanded = yp.unsqueeze(0).expand(N, -1, -1)
    start_y = torch.gather(y_positions_expanded, dim=2, index=start_idx2.unsqueeze(2)).squeeze(2)
    end_y = torch.gather(y_positions_expanded, dim=2, index=(start_idx2 + 1).unsqueeze(2)).squeeze(2)
    cand = start_y + (x - start_x) * (end_y - start_y) / (end_x - start_x)
    return cand


def oracle_expand_dims(v, dims):
    return v[(...,) + (None,)*(dims - 1)]


class OracleDiscreteSchedule:
    def __init__(self, alphas_cumprod):
        log_alphas = 0.5 * torch.log(alphas_cumprod)
        self.total_N = len(log_alphas)
        self.T = 1.
        self.t_array = torch.linspace(0., 1., self.total_N + 1)[1:].reshape((1, -1))
        self.log_alpha_array = log_alphas.reshape((1, -1,))

    def marginal_log_mean_coeff(self, t):
        return oracle_interpolate_fn(t.reshape((-1, 1)), self.t_array.to(t.device), self.log_alpha_array.to(t.device)).reshape((-1))

    def marginal_alpha(self, t):
        return torch.exp(self.marginal_log_mean_coeff(t))

    def marginal_std(self, t):
        return torch.sqrt(1. - torch.exp(2. * self.marginal_log_mean_coeff(t)))

    def marginal_lambda(self, t):
        log_mean_coeff = self.marginal_log_mean_coeff(t)
        log_std = 0.5 * torch.log(1. - torch.exp(2. * log_mean_coeff))
        return log_mean_coeff - log_std

    def inverse_lambda(self, lamb):
        log_alpha = -0.5 * torch.logaddexp(lamb.new_zeros((1,)), -2. * lamb)
        t = oracle_interpolate_fn(log_alpha.reshape((-1, 1)), torch.flip(self.log_alpha_array.to(lamb.device), [1]), torch.flip(self.t_array.to(lamb.device), [1]))
        return t.reshape((-1,))


class OracleUniPC:
    def __init__(self, cfg_model, extra_args, callback, noise_schedule, variant):
        self.cfg_model = cfg_model
        self.extra_args = extra_args
        self.callback = callback
        self.noise_schedule = noise_schedule
        self.variant = variant
        self.index = 0

    def after_update(self, x, model_x):
        if self.callback is not None:
            self.callback({'x': x, 'i': self.index, 'sigma': 0, 'sigma_hat': 0, 'denoised': model_x})
        self.index += 1

    def model_fn(self, x, t):
        t_input = (t - 1. / self.noise_schedule.total_N) * 1000.
        noise = self.cfg_model(x, t_input, **self.extra_args)
        dims = x.dim()
        alpha_t, sigma_t = self.noise_schedule.marginal_alpha(t), self.noise_schedule.marginal_std(t)
        x0 = (x - oracle_expand_dims(sigma_t, dims) * noise) / oracle_expand_dims(alpha_t, dims)
        return x0

    def get_time_steps(self, skip_type, t_T, t_0, N, device):
        if skip_type == 'logSNR':
            lambda_T = self.noise_schedule.marginal_lambda(torch.as_tensor(t_T, device=device)).reshape(())
            lambda_0 = self.noise_schedule.marginal_lambda(torch.as_tensor(t_0, device=device)).reshape(())
            logSNR_steps = torch.linspace(lambda_T, lambda_0, N + 1, device=device)
            return self.noise_schedule.inverse_lambda(logSNR_steps)
        elif skip_type == 'time_uniform':
            return torch.linspace(t_T, t_0, N + 1, device=device)
        elif skip_type == 'time_quadratic':
            t_order = 2
            t = torch.linspace(t_T**(1. / t_order), t_0**(1. / t_order), N + 1, device=device).pow(t_order)
            return t
        raise AssertionError(skip_type)

    def multistep_uni_pc_update(self, x, model_prev_list, t_prev_list, t, order, use_corrector):
        if len(t.shape) == 0:
            t = t.view(-1)
        if 'bh' in self.variant:
            return self.multistep_uni_pc_bh_update(x, model_prev_list, t_prev_list, t, order, use_corrector)
        assert self.variant == 'vary_coeff'
        return self.multistep_uni_pc_vary_update(x, model_prev_list, t_prev_list, t, order, use_corrector)

    def multistep_uni_pc_vary_update(self, x, model_prev_list, t_prev_list, t, order, use_corrector):
        ns = self.noise_schedule
        dims = x.dim()

        t_prev_0 = t_prev_list[-1]
        lambda_prev_0 = ns.marginal_lambda(t_prev_0)
        lambda_t = ns.marginal_lambda(t)
        model_prev_0 = model_prev_list[-1]
        sigma_prev_0, sigma_t = ns.marginal_std(t_prev_0), ns.marginal_std(t)
        log_alpha_t = ns.marginal_log_mean_coeff(t)
        alpha_t = torch.exp(log_alpha_t)

        h = lambda_t - lambda_prev_0

        rks = []
        D1s = []
        for i in range(1, order):
            t_prev_i = t_prev_list[-(i + 1)]
            model_prev_i = model_prev_list[-(i + 1)]
            lambda_prev_i = ns.marginal_lambda(t_prev_i)
            rk = ((lambda_prev_i - lambda_prev_0) / h)[0]
            rks.append(rk)
            D1s.append((model_prev_i - model_prev_0) / rk)

        rks.append(h.new_ones(()))
        rks = torch.stack(rks)

        K = len(rks)
        C = []

        col = torch.ones_like(rks)
        for k in range(1, K + 1):
            C.append(col)
            col = col * rks / (k + 1)
        C = torch.stack(C, dim=1)

        if len(D1s) > 0:
            D1s = torch.stack(D1s, dim=1)
            A_p = torch.linalg.inv_ex(C[:-1, :-1], check_errors=False)[0]

        if use_corrector:
            A_c = torch.linalg.inv_ex(C, check_errors=False)[0]

        hh = -h
        h_phi_1 = torch.expm1(hh)
        h_phi_ks = []
        factorial_k = 1
        h_phi_k = h_phi_1
        for k in range(1, K + 2):
            h_phi_ks.append(h_phi_k)
            h_phi_k = h_phi_k / hh - 1 / factorial_k
            factorial_k *= (k + 1)

        model_t = None
        x_t_ = (
            oracle_expand_dims(sigma_t / sigma_prev_0, dims) * x
            - oracle_expand_dims(alpha_t * h_phi_1, dims) * model_prev_0
        )
        x_t = x_t_
        if len(D1s) > 0:
            for k in range(K - 1):
                x_t = x_t - oracle_expand_dims(alpha_t * h_phi_ks[k + 1], dims) * torch.einsum('bkchw,k->bchw', D1s, A_p[k])
        if use_corrector:
            model_t = self.model_fn(x_t, t)
            D1_t = (model_t - model_prev_0)
            x_t = x_t_
            k = 0
            for k in range(K - 1):
                x_t = x_t - oracle_expand_dims(alpha_t * h_phi_ks[k + 1], dims) * torch.einsum('bkchw,k->bchw', D1s, A_c[k][:-1])
            x_t = x_t - oracle_expand_dims(alpha_t * h_phi_ks[K], dims) * (D1_t * A_c[k][-1])
        return x_t, model_t

    def multistep_uni_pc_bh_update(self, x, model_prev_list, t_prev_list, t, order, use_corrector):
        ns = self.noise_schedule
        dims = x.dim()

        t_prev_0 = t_prev_list[-1]
        lambda_prev_0 = ns.marginal_lambda(t_prev_0)
        lambda_t = ns.marginal_lambda(t)
        model_prev_0 = model_prev_list[-1]
        sigma_prev_0, sigma_t = ns.marginal_std(t_prev_0), ns.marginal_std(t)
        log_alpha_t = ns.marginal_log_mean_coeff(t)
        alpha_t = torch.exp(log_alpha_t)

        h = lambda_t - lambda_prev_0

        rks = []
        D1s = []
        for i in range(1, order):
            t_prev_i = t_prev_list[-(i + 1)]
            model_prev_i = model_prev_list[-(i + 1)]
            lambda_prev_i = ns.marginal_lambda(t_prev_i)
            rk = ((lambda_prev_i - lambda_prev_0) / h)[0]
            rks.append(rk)
            D1s.append((model_prev_i - model_prev_0) / rk)

        rks.append(h.new_ones(()))
        rks = torch.stack(rks)

        R = []
        b = []

        hh = -h[0]
        h_phi_1 = torch.expm1(hh)
        h_phi_k = h_phi_1 / hh - 1

        factorial_i = 1

        if self.variant == 'bh1':
            B_h = hh
        else:
            assert self.variant == 'bh2'
            B_h = torch.expm1(hh)

        for i in range(1, order + 1):
            R.append(torch.pow(rks, i - 1))
            b.append(h_phi_k * factorial_i / B_h)
            factorial_i *= (i + 1)
            h_phi_k = h_phi_k / hh - 1 / factorial_i

        R = torch.stack(R)
        b = torch.stack(b)

        if len(D1s) > 0:
            D1s = torch.stack(D1s, dim=1)
            if order == 2:
                rhos_p = b.new_full((1,), 0.5)
            else:
                rhos_p = torch.linalg.solve_ex(R[:-1, :-1], b[:-1], check_errors=False)[0]
            pred_res = torch.einsum('k,bkchw->bchw', rhos_p, D1s)
        else:
            D1s = None
            pred_res = 0

        if use_corrector:
            if order == 1:
                rhos_c = b.new_full((1,), 0.5)
            else:
                rhos_c = torch.linalg.solve_ex(R, b, check_errors=False)[0]

        model_t = None
        x_t_ = (
            oracle_expand_dims(sigma_t / sigma_prev_0, dims) * x
            - oracle_expand_dims(alpha_t * h_phi_1, dims)* model_prev_0
        )
        x_t = x_t_ - oracle_expand_dims(alpha_t * B_h, dims) * pred_res

        if use_corrector:
            model_t = self.model_fn(x_t, t)
            if D1s is not None:
                corr_res = torch.einsum('k,bkchw->bchw', rhos_c[:-1], D1s)
            else:
                corr_res = 0
            D1_t = (model_t - model_prev_0)
            x_t = x_t_ - oracle_expand_dims(alpha_t * B_h, dims) * (corr_res + rhos_c[-1] * D1_t)
        return x_t, model_t

    def sample(self, x, steps, t_start, order, skip_type, lower_order_final):
        t_0 = 1. / self.noise_schedule.total_N
        t_T = self.noise_schedule.T if t_start is None else t_start
        device = x.device
        timesteps = self.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=steps, device=device)
        with torch.no_grad():
            vec_t = timesteps[0].expand((x.shape[0]))
            model_prev_list = [self.model_fn(x, vec_t)]
            t_prev_list = [vec_t]
            for init_order in range(1, order):
                vec_t = timesteps[init_order].expand(x.shape[0])
                x, model_x = self.multistep_uni_pc_update(x, model_prev_list, t_prev_list, vec_t, init_order, use_corrector=True)
                if model_x is None:
                    model_x = self.model_fn(x, vec_t)
                self.after_update(x, model_x)
                model_prev_list.append(model_x)
                t_prev_list.append(vec_t)

            for step in range(order, steps + 1):
                vec_t = timesteps[step].expand(x.shape[0])
                if lower_order_final:
                    step_order = min(order, steps + 1 - step)
                else:
                    step_order = order
                use_corrector = step != steps
                x, model_x = self.multistep_uni_pc_update(x, model_prev_list, t_prev_list, vec_t, step_order, use_corrector=use_corrector)
                callback_model_x = model_x
                if callback_model_x is None and step == steps:
                    callback_model_x = x
                self.after_update(x, callback_model_x)
                for i in range(order - 1):
                    t_prev_list[i] = t_prev_list[i + 1]
                    model_prev_list[i] = model_prev_list[i + 1]
                t_prev_list[-1] = vec_t
                if step < steps:
                    if model_x is None:
                        model_x = self.model_fn(x, vec_t)
                    model_prev_list[-1] = model_x
        return x


def oracle_unipc(opts):
    def unipc(model, x, timesteps, extra_args=None, callback=None, disable=None, is_img2img=False):
        ns = OracleDiscreteSchedule(model.inner_model.inner_model.alphas_cumprod)
        t_start = timesteps[-1] / 1000 + 1 / 1000 if is_img2img else None
        sampler = OracleUniPC(model, {} if extra_args is None else extra_args, callback, ns, opts.uni_pc_variant)
        return sampler.sample(x, len(timesteps), t_start, opts.uni_pc_order, opts.uni_pc_skip_type, opts.uni_pc_lower_order_final)

    return unipc


# --- tests ---

SAMPLERS = {
    "ddim": (sd_samplers_timesteps_impl.ddim, oracle_ddim),
    "ddim_cfgpp": (sd_samplers_timesteps_impl.ddim_cfgpp, oracle_ddim_cfgpp),
    "plms": (sd_samplers_timesteps_impl.plms, oracle_plms),
}


@pytest.mark.parametrize("name,eta", [("ddim", 0.0), ("ddim", 0.5), ("ddim_cfgpp", 0.0), ("ddim_cfgpp", 0.5), ("plms", None)])
@pytest.mark.parametrize("steps,t_enc", [(1, None), (4, None), (20, None), (20, 12)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_timestep_samplers_match_the_reference_oracle(monkeypatch, name, eta, steps, t_enc, dtype):
    x, extra_args, alphas_cumprod, weights = make_inputs(dtype)
    timesteps = sd_samplers_timesteps._make_timesteps(steps, torch.device("cpu"))
    if t_enc is not None:  # img2img passes the first t_enc timesteps
        timesteps = timesteps[:t_enc]
    production, oracle = SAMPLERS[name]
    kwargs = {} if eta is None else {"eta": eta}  # PLMS takes no eta (the reference asserts eta == 0)

    expected = run_sampler(monkeypatch, oracle, x, timesteps, extra_args, alphas_cumprod, weights, **kwargs)
    actual = run_sampler(monkeypatch, production, x, timesteps, extra_args, alphas_cumprod, weights, **kwargs)

    assert len(expected[1]) == len(timesteps)
    assert_same_run(actual, expected)


class TimestepRecorder(FakeDenoiser):
    def __init__(self, alphas_cumprod, weights):
        super().__init__(alphas_cumprod, weights)
        self.timesteps = []

    def __call__(self, x, t, **kwargs):
        self.timesteps.append(int(t[0]))
        return super().__call__(x, t, **kwargs)


@pytest.mark.parametrize("name", list(SAMPLERS))
@pytest.mark.parametrize("steps", [1, 4, 20])
def test_every_timestep_is_evaluated_down_to_the_first(monkeypatch, name, steps):
    x, extra_args, alphas_cumprod, weights = make_inputs(torch.float64)
    timesteps = sd_samplers_timesteps._make_timesteps(steps, torch.device("cpu"))
    monkeypatch.setattr(k_diffusion.sampling, "torch", SeededNoise(99))
    model = TimestepRecorder(alphas_cumprod, weights)

    SAMPLERS[name][0](model, x, timesteps, extra_args=extra_args, disable=True)

    expected = [int(t) for t in torch.flip(timesteps, [0])]
    if name == "plms":  # the first step is a pseudo improved Euler step: it also evaluates the next timestep
        expected.insert(1, expected[min(1, len(expected) - 1)])
    assert model.timesteps == expected
    assert model.timesteps[-1] == 1  # the first DDIM timestep; its step goes to alphas_cumprod[0]


@pytest.mark.parametrize("name", list(SAMPLERS))
def test_exact_noise_prediction_ends_at_the_alphas_cumprod_0_marginal(monkeypatch, name):
    """A denoiser returning the exact noise of a known x0 must, at eta 0, end at the alphas_cumprod[0] marginal of x0.
    Stopping one step early (the old loop) leaves timestep 1's noise level, about 0.03 relative to x0 here."""
    alphas_cumprod = make_inputs(torch.float64)[2].double()
    generator = torch.Generator().manual_seed(5)
    x0 = torch.randn(1, 4, 8, 8, generator=generator, dtype=torch.float64)
    noise = torch.randn(1, 4, 8, 8, generator=generator, dtype=torch.float64)
    timesteps = sd_samplers_timesteps._make_timesteps(20, torch.device("cpu"))

    class ExactNoise:
        inner_model = types.SimpleNamespace(inner_model=types.SimpleNamespace(alphas_cumprod=alphas_cumprod))
        cond_scale_miltiplier = 1.0
        need_last_noise_uncond = False
        last_noise_uncond = None

        def __call__(self, x, t):
            a = alphas_cumprod[int(t[0])]
            self.last_noise_uncond = (x - a.sqrt() * x0) / (1 - a).sqrt()
            return self.last_noise_uncond

    monkeypatch.setattr(k_diffusion.sampling, "torch", SeededNoise(99))
    a_T = alphas_cumprod[timesteps[-1]]
    x_T = a_T.sqrt() * x0 + (1 - a_T).sqrt() * noise

    out = SAMPLERS[name][0](ExactNoise(), x_T, timesteps, disable=True)

    a_0 = alphas_cumprod[0]
    torch.testing.assert_close(out, a_0.sqrt() * x0 + (1 - a_0).sqrt() * noise, rtol=0, atol=1e-9)


@pytest.mark.parametrize("variant,skip_type", list(itertools.product(["bh1", "bh2", "vary_coeff"], ["time_uniform", "time_quadratic", "logSNR"])))
@pytest.mark.parametrize("order", [1, 2, 3])
@pytest.mark.parametrize("lower_order_final", [True, False])
@pytest.mark.parametrize("is_img2img", [False, True])
def test_unipc_matches_the_oracle(monkeypatch, variant, skip_type, order, lower_order_final, is_img2img):
    x, extra_args, alphas_cumprod, weights = make_inputs(torch.float32)
    timesteps = sd_samplers_timesteps._make_timesteps(6, torch.device("cpu"))
    kwargs = {}
    if is_img2img:  # img2img passes the first t_enc timesteps and is_img2img=True
        timesteps = timesteps[:4]
        kwargs["is_img2img"] = True
    opts = types.SimpleNamespace(uni_pc_variant=variant, uni_pc_skip_type=skip_type, uni_pc_order=order, uni_pc_lower_order_final=lower_order_final)
    monkeypatch.setattr(sd_samplers_timesteps_impl, "shared", types.SimpleNamespace(opts=opts))

    expected = run_sampler(monkeypatch, oracle_unipc(opts), x, timesteps, extra_args, alphas_cumprod, weights, **kwargs)
    actual = run_sampler(monkeypatch, sd_samplers_timesteps_impl.unipc, x, timesteps, extra_args, alphas_cumprod, weights, **kwargs)

    assert len(expected[1]) == len(timesteps)
    assert_same_run(actual, expected)
