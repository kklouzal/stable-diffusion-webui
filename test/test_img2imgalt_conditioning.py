"""scripts/img2imgalt.py's noise inversion sends the model the conditioning layout CFGDenoiser sends.

SD-XL conditioning is a dict (crossattn, vector): it used to be passed to torch.cat, which raises TypeError. It is now
batched per key with c_concat added, as CFGDenoiser does; tensor (SD1/SD2) conditioning keeps the exact tensors it had.
The decode prompts are encoded as processing encodes prompts: with the image size and, for the negative prompt,
is_negative_prompt (SD-XL zeroes an empty one). Runs the real script functions and k-diffusion's CompVisDenoiser
against a recording stub model.
"""

from types import SimpleNamespace

import pytest
import torch

from test.helpers import add_repositories_to_sys_path, init_shared, load_source

shared = init_shared()
add_repositories_to_sys_path("k-diffusion")


@pytest.fixture
def img2imgalt(monkeypatch):
    script = load_source("img2imgalt_under_test", "scripts/img2imgalt.py")
    monkeypatch.setattr(script, "sd_samplers_common", SimpleNamespace(store_latent=lambda latent: None))
    monkeypatch.setattr(shared.state, "sampling_step", 0)
    monkeypatch.setattr(shared.state, "sampling_steps", 0)
    return script


class StubModel:
    parameterization = "eps"

    def __init__(self, sdxl):
        self.sdxl = sdxl
        self.alphas_cumprod = torch.linspace(0.9991, 0.0047, 1000)
        self.conds = []
        self.prompts = []

    def get_learned_conditioning(self, prompts):
        self.prompts.append(prompts)
        rows = torch.stack([torch.full((77, 8), float(len(text) + 1)) for text in prompts])
        if self.sdxl:
            return {"crossattn": rows, "vector": rows[:, 0, :6] * 2}
        return rows

    def apply_model(self, x, t, cond):
        self.conds.append(cond)
        crossattn = cond["crossattn"] if self.sdxl else cond["c_crossattn"][0]
        return x * 0.1 + crossattn.mean(dim=(1, 2)).reshape(-1, 1, 1, 1) * 0.01


def processing_p(model, batch_size=2):
    latent = torch.randn(batch_size, 4, 8, 8, generator=torch.Generator().manual_seed(0))
    return SimpleNamespace(init_latent=latent, image_conditioning=torch.zeros(batch_size, 5, 1, 1), sd_model=model, batch_size=batch_size, width=832, height=1216)


@pytest.mark.parametrize("finder", ["find_noise_for_image", "find_noise_for_image_sigma_adjustment"])
def test_sdxl_dict_conditioning_is_batched_per_key_with_c_concat(monkeypatch, img2imgalt, finder):
    model = StubModel(sdxl=True)
    monkeypatch.setattr(shared, "sd_model", model, raising=False)
    p = processing_p(model)
    cond, uncond = model.get_learned_conditioning(["a b", "a b"]), model.get_learned_conditioning(["", ""])

    noise = getattr(img2imgalt, finder)(p, cond, uncond, 1.5, 4)

    assert noise.shape == p.init_latent.shape and torch.isfinite(noise).all()
    assert len(model.conds) == 4
    for sent in model.conds:
        assert set(sent) == {"crossattn", "vector", "c_concat"}
        assert torch.equal(sent["crossattn"], torch.cat([uncond["crossattn"], cond["crossattn"]]))
        assert torch.equal(sent["vector"], torch.cat([uncond["vector"], cond["vector"]]))
        assert len(sent["c_concat"]) == 1 and torch.equal(sent["c_concat"][0], torch.cat([p.image_conditioning] * 2))


@pytest.mark.parametrize("finder", ["find_noise_for_image", "find_noise_for_image_sigma_adjustment"])
def test_tensor_conditioning_keeps_its_exact_layout(monkeypatch, img2imgalt, finder):
    model = StubModel(sdxl=False)
    monkeypatch.setattr(shared, "sd_model", model, raising=False)
    p = processing_p(model)
    cond, uncond = model.get_learned_conditioning(["a b", "a b"]), model.get_learned_conditioning(["", ""])

    getattr(img2imgalt, finder)(p, cond, uncond, 1.5, 4)

    for sent in model.conds:
        assert list(sent) == ["c_concat", "c_crossattn"]
        assert torch.equal(sent["c_crossattn"][0], torch.cat([uncond, cond]))
        assert torch.equal(sent["c_concat"][0], torch.cat([p.image_conditioning] * 2))


def test_decode_prompts_are_encoded_with_the_image_size_and_negative_flag(monkeypatch, img2imgalt):
    model = StubModel(sdxl=True)
    monkeypatch.setattr(shared, "sd_model", model, raising=False)
    p = processing_p(model, batch_size=1)
    p.extra_generation_params, p.seed, p.subseed_strength, p.seed_resize_from_h, p.seed_resize_from_w = {}, 1, 0.0, 0, 0
    sampler = SimpleNamespace(model_wrap=SimpleNamespace(get_sigmas=lambda steps: torch.ones(steps + 1)), sample_img2img=lambda p, x, noise, *args, **kwargs: noise)
    monkeypatch.setattr(img2imgalt, "sd_samplers", SimpleNamespace(create_sampler=lambda name, model: sampler))
    monkeypatch.setattr(img2imgalt, "processing", SimpleNamespace(
        process_images=lambda p: p.sample(None, None, [1], [0], 0.0, ["prompt"]),
        create_random_tensors=lambda shape, **kwargs: torch.zeros((1, *shape)),
    ))

    img2imgalt.Script().run(p, None, True, True, "decode me", "", True, 3, True, 1.0, 0.0, False)

    positive, negative = model.prompts
    assert list(positive) == ["decode me"] and not positive.is_negative_prompt
    assert list(negative) == [""] and negative.is_negative_prompt
    assert (positive.width, positive.height) == (negative.width, negative.height) == (832, 1216)
    assert len(model.conds) == 3 and all("vector" in sent for sent in model.conds)
