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
    expected = z64 * (z.double().mean(dim=(1, 2), keepdim=True) / z64.mean(dim=(1, 2), keepdim=True))
    assert (emphasis.z.double() - expected).abs().max() <= 1e-5 * expected.abs().max()


def _original_emphasis(z, multipliers):
    from modules import sd_emphasis

    emphasis = sd_emphasis.EmphasisOriginal()
    emphasis.z = z
    emphasis.multipliers = multipliers
    emphasis.after_transformers()
    return emphasis.z


def test_original_emphasis_row_does_not_depend_on_its_batch():
    g = torch.Generator().manual_seed(2)
    z = (torch.randn(3, 77, 768, generator=g) + 0.3).to(torch.bfloat16)
    multipliers = torch.ones(3, 77)
    multipliers[0, 5:9] = 1.5
    multipliers[2, 40:50] = 0.7

    batched = _original_emphasis(z, multipliers)

    for row in range(3):
        assert torch.equal(batched[row], _original_emphasis(z[row:row + 1], multipliers[row:row + 1])[0])
    assert torch.equal(batched[1], z[1].float())  # the unemphasized prompt is left exactly as encoded


@pytest.mark.parametrize("width", [768, 1280])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_original_emphasis_single_row_is_bit_identical_to_the_whole_tensor_mean(width, dtype):
    """One prompt per encode (the production batch): the row mean is computed exactly as the old whole-tensor mean."""
    g = torch.Generator().manual_seed(width)
    for seed in range(8):
        z = (torch.randn(1, 77 * (1 + seed % 3), width, generator=g) * 2 + 0.2).to(dtype)
        multipliers = 1 + torch.rand(1, z.shape[1], generator=g)
        zf = z.float()
        weighted = zf * multipliers[..., None]
        old = weighted * (zf.mean() / weighted.mean())

        assert torch.equal(_original_emphasis(z, multipliers), old)


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


def test_embedding_replaced_in_place_is_republished_without_hashes(monkeypatch, tmp_path):
    # Under --no-hashing every embedding hash is '', so a same-shape replacement (equal size, mtime preserved) was
    # only noticed through the file identity: without it the refresh kept the old vectors and the TI epoch.
    import os
    import safetensors.torch
    from modules import hashes, openclaw_cache_epochs
    from modules.textual_inversion import textual_inversion as ti

    monkeypatch.setattr(hashes, "sha256", lambda *args, **kwargs: None)
    monkeypatch.setattr(ti.EmbeddingDatabase, "get_expected_shape", lambda self: 2048)  # 768-wide files are skipped
    db = ti.EmbeddingDatabase()
    db.add_embedding_dir(str(tmp_path))
    path = tmp_path / "style.safetensors"
    safetensors.torch.save_file({"emb_params": torch.zeros(1, 768)}, str(path))
    assert db.load_textual_inversion_embeddings(force_reload=True)
    assert torch.equal(db.skipped_embeddings["style"].vec.cpu(), torch.zeros(1, 768))

    original = os.stat(path)
    replacement = tmp_path / "replacement.tmp"
    safetensors.torch.save_file({"emb_params": torch.ones(1, 768)}, str(replacement))
    os.utime(replacement, ns=(original.st_atime_ns, original.st_mtime_ns))
    os.replace(replacement, path)
    assert os.stat(path).st_size == original.st_size
    epoch = openclaw_cache_epochs.epoch_subset(("textual_inversion_epoch",))

    assert db.load_textual_inversion_embeddings(force_reload=True)
    assert torch.equal(db.skipped_embeddings["style"].vec.cpu(), torch.ones(1, 768))
    assert openclaw_cache_epochs.epoch_subset(("textual_inversion_epoch",)) != epoch
    assert not db.load_textual_inversion_embeddings(force_reload=True)  # unchanged file: nothing to publish


def _embedding_db(monkeypatch, tmp_path):
    from modules import hashes
    from modules.textual_inversion import textual_inversion as ti

    monkeypatch.setattr(hashes, "sha256", lambda *args, **kwargs: None)
    monkeypatch.setattr(ti.EmbeddingDatabase, "get_expected_shape", lambda self: 2048)  # 768-wide files are skipped
    db = ti.EmbeddingDatabase()
    db.add_embedding_dir(str(tmp_path))
    return ti, db


def _save_embedding(path, value):
    import safetensors.torch

    path.parent.mkdir(parents=True, exist_ok=True)
    safetensors.torch.save_file({"emb_params": torch.full((1, 768), float(value))}, str(path))


def test_embedding_changes_in_subfolders_are_reloaded(monkeypatch, tmp_path):
    """Only the top folder's mtime was checked: files added to or rewritten in a subfolder stayed unloaded until a
    forced reload."""
    _ti, db = _embedding_db(monkeypatch, tmp_path)
    _save_embedding(tmp_path / "styles" / "first.safetensors", 0)
    assert db.load_textual_inversion_embeddings()
    assert not db.load_textual_inversion_embeddings()  # unchanged tree: no reload

    _save_embedding(tmp_path / "styles" / "second.safetensors", 1)
    assert db.load_textual_inversion_embeddings()
    assert sorted(db.skipped_embeddings) == ["first", "second"]

    _save_embedding(tmp_path / "styles" / "first.safetensors", 2)
    assert db.load_textual_inversion_embeddings()
    assert torch.equal(db.skipped_embeddings["first"].vec.cpu(), torch.full((1, 768), 2.0))

    (tmp_path / "styles" / "second.safetensors").unlink()
    assert db.load_textual_inversion_embeddings()
    assert sorted(db.skipped_embeddings) == ["first"]


def test_duplicate_embedding_names_load_one_file_deterministically_and_are_reported(monkeypatch, tmp_path):
    ti, db = _embedding_db(monkeypatch, tmp_path)
    reports = []
    monkeypatch.setattr(ti.errors, "report", lambda message, **_kwargs: reports.append(message))
    for folder, value in (("b", 2), ("a", 1)):
        _save_embedding(tmp_path / folder / "style.safetensors", value)
    _save_embedding(tmp_path / "style.safetensors", 0)

    assert db.load_textual_inversion_embeddings()

    assert torch.equal(db.skipped_embeddings["style"].vec.cpu(), torch.zeros(1, 768))  # top folder first, then a/, b/
    assert reports == [
        f"Textual inversion embedding {tmp_path / 'a' / 'style.safetensors'} is not loaded: {tmp_path / 'style.safetensors'} has the same name 'style'",
        f"Textual inversion embedding {tmp_path / 'b' / 'style.safetensors'} is not loaded: {tmp_path / 'style.safetensors'} has the same name 'style'",
    ]


def _save_wide_embedding(path, value):
    import safetensors.torch

    path.parent.mkdir(parents=True, exist_ok=True)
    safetensors.torch.save_file({"emb_params": torch.full((1, 2048), float(value))}, str(path))


def test_a_skipped_embedding_does_not_block_a_fitting_one_of_the_same_name(monkeypatch, tmp_path):
    """sd15/foo (768 wide, skipped for a 2048-wide model) sorts before sdxl/foo: the fitting sdxl/foo loads and replaces
    the skipped entry. A second fitting foo, or a skipped-shape foo after a loaded one, is still reported."""
    ti, db = _embedding_db(monkeypatch, tmp_path)
    monkeypatch.setattr(ti.shared, "sd_model", SimpleNamespace(cond_stage_model=SimpleNamespace(tokenize=lambda names: [[len(names[0])]])), raising=False)
    reports = []
    monkeypatch.setattr(ti.errors, "report", lambda message, **_kwargs: reports.append(message))
    _save_embedding(tmp_path / "sd15" / "foo.safetensors", 1)
    _save_wide_embedding(tmp_path / "sdxl" / "foo.safetensors", 2)
    _save_embedding(tmp_path / "sd15" / "bar.safetensors", 3)

    assert db.load_textual_inversion_embeddings()

    assert reports == []
    assert list(db.word_embeddings) == ["foo"]
    assert db.word_embeddings["foo"].filename == str(tmp_path / "sdxl" / "foo.safetensors")
    assert torch.equal(db.word_embeddings["foo"].vec.cpu(), torch.full((1, 2048), 2.0))
    assert sorted(db.skipped_embeddings) == ["bar"]

    _save_wide_embedding(tmp_path / "zz" / "foo.safetensors", 4)
    _save_embedding(tmp_path / "zz" / "later" / "foo.safetensors", 5)

    assert not db.load_textual_inversion_embeddings()  # both new files are rejected: the published maps do not change

    assert db.word_embeddings["foo"].filename == str(tmp_path / "sdxl" / "foo.safetensors")
    assert sorted(db.skipped_embeddings) == ["bar"]
    assert reports == [
        f"Textual inversion embedding {tmp_path / 'zz' / 'foo.safetensors'} is not loaded: {tmp_path / 'sdxl' / 'foo.safetensors'} has the same name 'foo'",
        f"Textual inversion embedding {tmp_path / 'zz' / 'later' / 'foo.safetensors'} is not loaded: {tmp_path / 'sdxl' / 'foo.safetensors'} has the same name 'foo'",
    ]
