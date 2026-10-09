"""SD-XL's empty negative prompt is zeroed (sgm force_zero_embeddings), not encoded: when a longer positive prompt makes
the uncond need padding (pad_cond_uncond), a zeroed row is padded with zeros, not with the empty prompt's encoding.

The mark travels explicitly: sd_models_xl returns ZeroedTextConditioning, prompt_parser keeps it on each schedule row,
reconstruct_cond_batch reports the zeroed rows, and CFGDenoiser.pad_cond_uncond reads them."""

from types import SimpleNamespace

import pytest
import torch

from test.helpers import init_shared

shared = init_shared()

from modules import paths, sd_models_xl  # noqa: E402,F401  (paths puts repositories/ on sys.path for sgm)
from modules import prompt_parser, sd_models, sd_samplers, sd_samplers_cfg_denoiser  # noqa: E402,F401  (sd_samplers first: import cycle)

TOKENS, CHANNELS = 77, 8


class Conditioner:
    """sgm GeneralConditioner stand-in: text rows are 1 + row index unless force-zeroed, like sgm."""

    def __init__(self):
        self.embedders = []

    def __call__(self, batch, force_zero_embeddings=()):
        rows = len(batch["txt"])
        crossattn = torch.arange(1, rows + 1, dtype=torch.float32).reshape(rows, 1, 1).expand(rows, TOKENS, CHANNELS).clone()
        vector = torch.ones(rows, 4)
        if "txt" in force_zero_embeddings:
            crossattn.zero_()
        return {"crossattn": crossattn, "vector": vector}


def encode(texts, negative):
    model = SimpleNamespace(conditioner=Conditioner())
    return sd_models_xl.get_learned_conditioning(model, prompt_parser.SdConditioning(texts, is_negative_prompt=negative))


def test_only_an_all_empty_negative_batch_is_marked_zeroed():
    assert isinstance(encode(["", ""], negative=True), prompt_parser.ZeroedTextConditioning)
    assert not isinstance(encode(["", "blurry"], negative=True), prompt_parser.ZeroedTextConditioning)
    assert not isinstance(encode([""], negative=False), prompt_parser.ZeroedTextConditioning)
    zeroed = encode([""], negative=True)
    assert torch.count_nonzero(zeroed["crossattn"]) == 0


def learned_uncond(negative_prompts, steps=10):
    model = SimpleNamespace(get_learned_conditioning=lambda texts: encode(list(texts), negative=texts.is_negative_prompt))
    return prompt_parser.get_learned_conditioning(model, prompt_parser.SdConditioning(negative_prompts, is_negative_prompt=True), steps)


def test_rows_keep_the_mark_through_schedules_and_batch_reconstruction():
    # A prompt's schedule texts are encoded together: "[:blurry:5]" encodes "" with "blurry", which sgm does not zero.
    uncond = learned_uncond(["", "blurry", "[:blurry:5]", "[|]"])

    for step in (1, 3, 7):
        assert prompt_parser.reconstruct_cond_batch(uncond, step).zeroed_text_rows == (True, False, False, True)
    assert all(isinstance(entry.cond, prompt_parser.ZeroedTextConditioning) for entry in uncond[3])


@pytest.fixture
def denoiser(monkeypatch):
    empty = torch.full((1, TOKENS, CHANNELS), 0.5)
    model_data = sd_models.SdModelData()
    model_data.sd_model, model_data.was_loaded_at_least_once = SimpleNamespace(cond_stage_model_empty_prompt=empty), True
    monkeypatch.setattr(sd_models, "model_data", model_data)
    return sd_samplers_cfg_denoiser.CFGDenoiser(SimpleNamespace()), empty


def long_cond(rows):
    return prompt_parser.DictWithShape({"crossattn": torch.full((rows, 2 * TOKENS, CHANNELS), 9.0), "vector": torch.ones(rows, 4)})


def test_zeroed_rows_are_padded_with_zeros_and_encoded_rows_with_the_empty_prompt(denoiser):
    denoiser, empty = denoiser
    uncond = prompt_parser.reconstruct_cond_batch(learned_uncond(["", "blurry"]), 1)

    _cond, padded = denoiser.pad_cond_uncond(long_cond(2), uncond)

    assert padded["crossattn"].shape == (2, 2 * TOKENS, CHANNELS)
    assert torch.count_nonzero(padded["crossattn"][0]) == 0
    assert torch.equal(padded["crossattn"][1, :TOKENS], uncond["crossattn"][1])
    assert torch.equal(padded["crossattn"][1, TOKENS:], empty[0])
    assert denoiser.padded_cond_uncond


def test_unmarked_uncond_is_padded_with_the_empty_prompt_as_before(denoiser):
    denoiser, empty = denoiser
    uncond = prompt_parser.DictWithShape({"crossattn": torch.zeros(2, TOKENS, CHANNELS), "vector": torch.ones(2, 4)})

    _cond, padded = denoiser.pad_cond_uncond(long_cond(2), uncond)

    expected = torch.cat([uncond["crossattn"], empty.repeat((2, 1, 1))], axis=1)
    assert torch.equal(padded["crossattn"], expected)


def test_tensor_conds_pad_exactly_as_before():
    tensor = torch.randn(3, TOKENS, CHANNELS)
    empty = torch.randn(1, TOKENS, CHANNELS)

    padded = sd_samplers_cfg_denoiser.pad_cond(tensor, 2, empty)

    assert torch.equal(padded, torch.cat([tensor, empty.repeat((3, 2, 1))], axis=1))
