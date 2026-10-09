"""Float32 text encoders (sd_models.float32_text_encoder_names) against a float64 reference.

Runs the real hijacked SDXL OpenCLIP-G path (sd_hijack_open_clip.FrozenOpenCLIPEmbedder2WithCustomWords over
sgm's FrozenOpenCLIPEmbedder2, open_clip's bundled tokenizer) on a small randomly initialized text transformer of
the same architecture: penultimate hidden state with Original emphasis, and the pooled projection.
"""

import copy
from types import SimpleNamespace

import pytest
import torch

from test.helpers import init_shared

shared = init_shared()

from modules import devices, sd_models  # noqa: E402,F401  (sd_models imports sd_hijack in startup order)
from modules import sd_hijack, sd_hijack_clip, sd_hijack_open_clip  # noqa: E402


def _clip_g(width=128, heads=4, layers=6):
    import open_clip
    from sgm.modules.encoders.modules import FrozenOpenCLIPEmbedder2

    torch.manual_seed(0)
    clip = open_clip.model.CLIP(
        embed_dim=64,
        vision_cfg=open_clip.model.CLIPVisionCfg(layers=1, width=64, patch_size=16, image_size=32),
        text_cfg=open_clip.model.CLIPTextCfg(context_length=77, vocab_size=49408, width=width, heads=heads, layers=layers),
    )
    del clip.visual
    with torch.no_grad():
        # Trained CLIP weights are far from open_clip's tiny init: give the residual stream realistic magnitudes.
        for name, parameter in clip.named_parameters():
            if parameter.ndim == 2 and "embedding" not in name:
                parameter.normal_(0, 1.5 / parameter.shape[-1] ** 0.5)
        clip.token_embedding.weight.normal_(0, 0.5)
    embedder = FrozenOpenCLIPEmbedder2.__new__(FrozenOpenCLIPEmbedder2)
    torch.nn.Module.__init__(embedder)
    embedder.model = clip.eval()
    embedder.device = "cpu"
    embedder.max_length = 77
    embedder.return_pooled = True
    embedder.layer = "penultimate"
    embedder.layer_idx = 1
    embedder.legacy = False
    return embedder


def _encode(embedder, texts):
    hijack = sd_hijack.StableDiffusionModelHijack()
    wrapper = sd_hijack_open_clip.FrozenOpenCLIPEmbedder2WithCustomWords(embedder, hijack)
    with torch.no_grad():
        return wrapper(texts)


def _relative_l2(actual, reference):
    return ((actual.double() - reference).norm() / reference.norm()).item()


@pytest.fixture
def options(monkeypatch):
    monkeypatch.setattr(sd_hijack_clip, "opts", SimpleNamespace(
        emphasis="Original", comma_padding_backtrack=20, textual_inversion_add_hashes_to_infotext=False,
        use_old_emphasis_implementation=False, textual_inversion_templates_dir="",
    ), raising=False)


def test_float32_text_encoder_is_far_closer_to_float64_than_bfloat16(options):
    texts = ["a photograph of an astronaut riding a horse, (highly detailed:1.2), [blurry]", ""]
    embedder = _clip_g()
    reference_z, reference_pooled = _encode(copy.deepcopy(embedder).double(), texts)

    float32_z, float32_pooled = _encode(copy.deepcopy(embedder).float(), texts)
    bfloat16_z, bfloat16_pooled = _encode(copy.deepcopy(embedder).to(torch.bfloat16), texts)  # the old bf16 TE

    assert float32_z.dtype == torch.float32 and float32_pooled.dtype == torch.float32
    float32_error, bfloat16_error = _relative_l2(float32_z, reference_z), _relative_l2(bfloat16_z, reference_z)
    assert float32_error < 1e-5
    assert bfloat16_error > 1e-3
    assert bfloat16_error > 100 * float32_error
    assert _relative_l2(float32_pooled, reference_pooled) < 1e-5 < 1e-3 < _relative_l2(bfloat16_pooled, reference_pooled)


def test_float32_text_encoder_runs_with_autocast_off_and_ieee_matmuls(monkeypatch):
    seen = []
    model = torch.nn.Linear(2, 2)

    def without_autocast():
        seen.append("without_autocast")
        return torch.autocast("cpu", enabled=False)

    monkeypatch.setattr(devices, "without_autocast", without_autocast)
    matmul = torch.backends.cuda.matmul
    saved = matmul.fp32_precision
    try:
        matmul.fp32_precision = "tf32"  # devices.enable_tf32
        with sd_hijack_clip.text_encoder_precision(model):
            seen.append(matmul.fp32_precision)
        assert matmul.fp32_precision == "tf32"
        with pytest.raises(RuntimeError, match="encode failed"), sd_hijack_clip.text_encoder_precision(model):
            raise RuntimeError("encode failed")
        assert matmul.fp32_precision == "tf32"  # restored on failure too

        model.to(torch.bfloat16)  # fp8/TorchAO storage keeps a lower-precision text encoder: caller's autocast
        with sd_hijack_clip.text_encoder_precision(model):
            seen.append(matmul.fp32_precision)
    finally:
        matmul.fp32_precision = saved

    assert seen == ["without_autocast", "ieee", "without_autocast", "tf32"]


@pytest.mark.parametrize("table_dtype", [torch.float32, torch.bfloat16])
def test_textual_inversion_vectors_take_the_embedding_table_dtype(monkeypatch, table_dtype):
    # --upcast-sampling made cond_cast_unet round the vector to the bf16 UNet dtype even for a float32 table.
    monkeypatch.setattr(devices, "unet_needs_upcast", True)
    monkeypatch.setattr(devices, "dtype_unet", torch.bfloat16)
    table = torch.nn.Embedding(10, 4).to(table_dtype)
    vector = torch.tensor([[1 + 2.0 ** -12, 2.0, 3.0, 4.0]])  # not representable in bfloat16
    embeddings = SimpleNamespace(fixes=[[(1, SimpleNamespace(vec=vector))]])
    layer = sd_hijack.EmbeddingsWithFixes(table, embeddings)

    with torch.no_grad():
        out = layer(torch.tensor([[0, 5, 6, 7]]))

    assert out.dtype == table_dtype
    assert torch.equal(out[0, 2], vector[0].to(table_dtype))
