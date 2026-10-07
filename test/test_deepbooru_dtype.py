import contextlib

import torch
from PIL import Image

from modules import shared, shared_init

if getattr(shared, "opts", None) is None:
    shared_init.initialize()

from modules import deepbooru, devices


class FakeDeepDanbooru(torch.nn.Module):
    tags = ["1girl", "rating:safe", "outdoors", "sky"]

    def forward(self, x):
        return torch.tensor([[0.9, 0.99, 0.2, 0.7]], dtype=x.dtype)


def test_tag_multi_accepts_bfloat16_model_output(monkeypatch):
    # With --dtype bfloat16 the model output is bf16, which numpy cannot represent.
    monkeypatch.setattr(devices, "device", torch.device("cpu"), raising=False)
    monkeypatch.setattr(devices, "dtype", torch.bfloat16, raising=False)
    # The subject is the bf16 -> numpy conversion; CUDA autocast would query the (absent) driver on CPU hosts.
    monkeypatch.setattr(devices, "autocast", contextlib.nullcontext)
    for key, value in {"interrogate_deepbooru_score_threshold": 0.5, "deepbooru_use_spaces": True, "deepbooru_escape": False,
                       "deepbooru_sort_alpha": False, "interrogate_return_ranks": True, "deepbooru_filter_tags": ""}.items():
        monkeypatch.setattr(shared.opts, key, value, raising=False)
    tagger = deepbooru.DeepDanbooru()
    tagger.model = FakeDeepDanbooru()

    assert tagger.tag_multi(Image.new("RGB", (64, 64))) == "(1girl:0.898), (sky:0.699)"
