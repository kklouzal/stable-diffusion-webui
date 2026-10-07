"""Text conditioning: emphasis numerics and textual inversion placement across chunk boundaries."""
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")


def test_original_emphasis_is_identity_for_unit_multipliers_with_bf16_z():
    from modules import sd_emphasis

    z = (torch.randn(2, 77, 768, generator=torch.Generator().manual_seed(0)) * 3 + 0.25).to(torch.bfloat16)
    emphasis = sd_emphasis.EmphasisOriginal()
    emphasis.z = z
    emphasis.multipliers = torch.ones(2, 77)

    emphasis.after_transformers()

    assert emphasis.z.dtype == torch.float32
    assert torch.equal(emphasis.z, z.float())


def test_original_emphasis_matches_fp64_mean_restoration():
    from modules import sd_emphasis

    g = torch.Generator().manual_seed(1)
    z = (torch.randn(2, 77, 768, generator=g) + 0.1).to(torch.bfloat16)
    multipliers = torch.ones(2, 77)
    multipliers[0, 5:9] = 1.1 ** 3
    multipliers[1, 20:30] = 1 / 1.1
    emphasis = sd_emphasis.EmphasisOriginal()
    emphasis.z = z
    emphasis.multipliers = multipliers

    emphasis.after_transformers()

    z64 = z.double() * multipliers.double()[..., None]
    expected = z64 * (z.double().mean() / z64.mean())
    assert (emphasis.z.double() - expected).abs().max() <= 1e-5 * expected.abs().max()


class _Tokenizing:
    COMMA, WORD, EMB, START, END = 1, 2, 7, 49406, 49407

    def __init__(self, monkeypatch, backtrack):
        from modules import sd_hijack_clip

        monkeypatch.setattr(sd_hijack_clip, "opts", SimpleNamespace(emphasis="Original", comma_padding_backtrack=backtrack))
        embedding = SimpleNamespace(name="emb", vectors=3)

        class FakeEmbeddingDB:
            def find_embedding_at_position(self, tokens, offset):
                return (embedding, 1) if tokens[offset] == _Tokenizing.EMB else (None, None)

        class Model(sd_hijack_clip.TextConditionalModel):
            def tokenize(self, texts):
                ids = {",": _Tokenizing.COMMA, "emb": _Tokenizing.EMB}
                return [[ids.get(word, _Tokenizing.WORD) for word in text.split()] for text in texts]

        self.model = Model()
        self.model.hijack = SimpleNamespace(embedding_db=FakeEmbeddingDB())
        self.model.comma_token = self.COMMA
        self.model.id_start, self.model.id_end, self.model.id_pad = self.START, self.END, self.END
        self.embedding = embedding


@pytest.mark.parametrize("backtrack", [0, 20])
def test_comma_backtrack_moves_textual_inversion_fixes_with_their_placeholders(monkeypatch, backtrack):
    t = _Tokenizing(monkeypatch, backtrack)
    # 60 words, a comma, a 3-vector embedding and 20 words: the chunk fills 14 tokens after the comma, so a
    # backtrack of 20 moves the embedding to the second chunk.
    prompt = " ".join(["w"] * 60 + [",", "emb"] + ["w"] * 20)

    chunks, _token_count = t.model.tokenize_line(prompt)

    placed = []
    for index, chunk in enumerate(chunks):
        assert len(chunk.tokens) == 77
        body = chunk.tokens[1:-1]  # fix offsets exclude the start token
        placeholders = [i for i, token in enumerate(body) if token == 0]
        covered = sorted(i for fix in chunk.fixes for i in range(fix.offset, fix.offset + fix.embedding.vectors))
        assert placeholders == covered
        placed += [(index, fix.offset) for fix in chunk.fixes]
    assert placed == ([(1, 0)] if backtrack else [(0, 61)])


def test_ti_hashes_are_listed_once_when_both_sdxl_encoders_report_them(monkeypatch):
    from modules import sd_hijack_clip

    t = _Tokenizing(monkeypatch, 0)
    monkeypatch.setattr(sd_hijack_clip, "opts", SimpleNamespace(emphasis="Original", comma_padding_backtrack=0, textual_inversion_add_hashes_to_infotext=True))
    t.embedding.shorthash = "abc123"
    t.model.hijack.extra_generation_params = {"TI hashes": "other: 999"}  # written by the other encoder
    t.model.process_tokens = lambda tokens, multipliers: torch.zeros(len(tokens), 77, 4)

    for _encoder in ("clip_l", "clip_g"):
        t.model.forward(["w , emb w"])

    assert t.model.hijack.extra_generation_params["TI hashes"] == "emb: abc123, other: 999"


def test_clip_l_tokenization_applies_ftfy_like_the_reference_tokenizer_and_clip_g():
    """transformers 4 CLIPTokenizer ran ftfy.fix_text before BPE; open_clip (SDXL clip_g) still does."""
    import os
    from transformers import CLIPTokenizer
    import open_clip.tokenizer
    from modules import sd_hijack_clip

    path = os.environ.get("GB10_A1111_CLIP_VIT_LARGE_PATCH14_PATH")
    if not path or not os.path.isdir(path):
        pytest.skip("local clip-vit-large-patch14 tokenizer not available")
    wrapped = SimpleNamespace(tokenizer=CLIPTokenizer.from_pretrained(path, local_files_only=True))
    clip_l = sd_hijack_clip.FrozenCLIPEmbedderWithCustomWords(wrapped, SimpleNamespace())
    texts = ["a girl’s “red” hat", "ｆｕｌｌ width", "a&amp;b", "mojibake cafÃ©", "plain, text"]

    assert clip_l.tokenize(texts) == [open_clip.tokenizer._tokenizer.encode(text) for text in texts]


def test_sdxl_embedding_shapes_are_validated():
    from modules.textual_inversion import textual_inversion as ti

    ok = ti.create_embedding_from_data({"clip_l": torch.zeros(2, 768), "clip_g": torch.zeros(2, 1280)}, "ok")
    assert (ok.vectors, ok.shape) == (2, 2048)
    single = ti.create_embedding_from_data({"clip_l": torch.zeros(768), "clip_g": torch.zeros(1280)}, "one")
    assert single.vectors == 1 and tuple(single.vec["clip_l"].shape) == (1, 768)
    for clip_l, clip_g in [((3, 768), (2, 1280)), ((2, 1280), (2, 768)), ((2, 768, 1), (2, 1280))]:
        with pytest.raises(Exception, match="SDXL embedding"):
            ti.create_embedding_from_data({"clip_l": torch.zeros(clip_l), "clip_g": torch.zeros(clip_g)}, "bad")


def test_unchanged_reload_still_publishes_expected_shape(monkeypatch):
    from modules.textual_inversion import textual_inversion as ti

    monkeypatch.setattr(ti.EmbeddingDatabase, "get_expected_shape", lambda self: 2048)
    db = ti.EmbeddingDatabase()
    assert db.expected_shape == -1

    assert not db.load_textual_inversion_embeddings(force_reload=True)  # no folders: same (empty) maps
    assert db.expected_shape == 2048
