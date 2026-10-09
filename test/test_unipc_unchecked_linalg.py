import pytest
import torch

from modules.models.diffusion.uni_pc import uni_pc


def _checked_oracle(monkeypatch):
    """The pre-change behaviour: torch.linalg.solve/inv, which validate (and sync) on every call."""
    monkeypatch.setattr(uni_pc.UniPC, "_solve", lambda self, A, B: torch.linalg.solve(A, B))
    monkeypatch.setattr(uni_pc.UniPC, "_inv", lambda self, A: torch.linalg.inv(A))


def _sample(variant, order, predict_x0, dtype):
    generator = torch.Generator().manual_seed(1234)
    x = torch.randn(2, 4, 8, 8, generator=generator, dtype=dtype)
    weights = torch.randn(4, 4, generator=generator, dtype=dtype) * 0.1

    def model_fn(x_in, t_in, cond=None, uncond=None):
        return torch.einsum("bchw,cd->bdhw", x_in, weights) + t_in.reshape(-1, 1, 1, 1) * 0.05

    alphas_cumprod = torch.cumprod(1 - torch.linspace(0.00085, 0.012, 1000, dtype=torch.float64), 0).to(dtype)
    sampler = uni_pc.UniPC(model_fn, uni_pc.NoiseScheduleVP("discrete", alphas_cumprod=alphas_cumprod), predict_x0=predict_x0, variant=variant)
    return sampler.sample(x, steps=8, skip_type="time_uniform", order=order, lower_order_final=True)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("variant", ["bh1", "bh2", "vary_coeff"])
@pytest.mark.parametrize("order", [1, 2, 3])
@pytest.mark.parametrize("predict_x0", [True, False])
def test_unchecked_solves_reproduce_the_checked_sampler_bit_for_bit(monkeypatch, dtype, variant, order, predict_x0):
    unchecked = _sample(variant, order, predict_x0, dtype)
    with monkeypatch.context() as patch:
        _checked_oracle(patch)
        checked = _sample(variant, order, predict_x0, dtype)

    assert torch.isfinite(unchecked).all()
    assert torch.equal(unchecked, checked)


def test_singular_systems_still_fail_the_sampling_run():
    sampler = uni_pc.UniPC(lambda x, t: x, uni_pc.NoiseScheduleVP("linear"))
    singular = torch.tensor([[1.0, 1.0], [1.0, 1.0]])

    sampler._solve(singular, torch.ones(2))
    sampler._inv(torch.eye(2))
    with pytest.raises(torch.linalg.LinAlgError, match="singular"):
        sampler.check_linalg_infos()
    assert sampler.linalg_infos == []

    sampler._inv(torch.eye(2))
    sampler.check_linalg_infos()


@pytest.mark.parametrize("skip_type", ["time_uniform", "time_quadratic", "logSNR"])
def test_every_skip_type_samples(skip_type):
    # logSNR used to pass shape-(1,) lambdas to torch.linspace, which only accepts 0-dim tensor endpoints.
    generator = torch.Generator().manual_seed(7)
    x = torch.randn(1, 4, 8, 8, generator=generator)
    alphas_cumprod = torch.cumprod(1 - torch.linspace(0.00085, 0.012, 1000, dtype=torch.float64), 0).float()
    sampler = uni_pc.UniPC(lambda x_in, t_in, cond=None, uncond=None: x_in * 0.1, uni_pc.NoiseScheduleVP("discrete", alphas_cumprod=alphas_cumprod), predict_x0=True, variant="bh2")

    out = sampler.sample(x, steps=6, skip_type=skip_type, order=2, lower_order_final=True)

    assert torch.isfinite(out).all()


def test_logsnr_time_steps_match_the_host_scalar_reference():
    alphas_cumprod = torch.cumprod(1 - torch.linspace(0.00085, 0.012, 1000, dtype=torch.float64), 0).float()
    ns = uni_pc.NoiseScheduleVP("discrete", alphas_cumprod=alphas_cumprod)
    sampler = uni_pc.UniPC(lambda x, t: x, ns)
    device = torch.device("cpu")

    # Reference: the original implementation, which read both endpoints back as Python floats.
    lambda_T = ns.marginal_lambda(torch.tensor(1.0)).item()
    lambda_0 = ns.marginal_lambda(torch.tensor(1.0 / 1000)).item()
    expected = ns.inverse_lambda(torch.linspace(lambda_T, lambda_0, 9))

    actual = sampler.get_time_steps("logSNR", 1.0, 1.0 / 1000, 8, device)

    assert actual.shape == (9,)
    assert torch.equal(actual, expected)
