"""The conditioning caches (E05 c/uc/hires slots, E06 per-call token memo, E07 textual inversion database): keys,
atomic publication, and commits that take the epoch transaction before their own lock."""

import ast
import contextlib
import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
PROCESSING = ROOT / "modules" / "processing.py"
CLEAR_COND_CACHE = ROOT / "extensions" / "openclaw-clear-cond-cache" / "scripts" / "openclaw_clear_cond_cache.py"
CONDITIONING_EPOCHS = ("checkpoint_object_epoch", "conditioner_epoch", "textual_inversion_epoch", "tokenizer_epoch", "conditioning_hook_epoch", "device_epoch")


def _cached_params_return_elements():
    tree = ast.parse(PROCESSING.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "StableDiffusionProcessing")
    fn = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "cached_params")
    ret = next(node for node in ast.walk(fn) if isinstance(node, ast.Return))
    return {ast.unparse(item) for item in ret.value.elts}


def test_conditioning_key_covers_parser_tokenization_and_effective_network_state():
    elements = _cached_params_return_elements()
    required = {
        "cache_namespace",
        "effective_network_state",
        "opts.CLIP_stop_at_last_layers",
        "opts.sdxl_clip_l_skip",
        "opts.emphasis",
        "opts.use_old_emphasis_implementation",
        "opts.comma_padding_backtrack",
        "required_prompts",
        "extra_network_data",
    }
    assert required <= elements


@pytest.fixture
def processing(initialize, monkeypatch):
    from modules import processing as processing_module

    # Other test files leave stub "modules" packages in sys.modules; shared.sd_model resolves modules.sd_models through it.
    monkeypatch.setitem(sys.modules, "modules", processing_module.modules)
    monkeypatch.setattr(processing_module.shared, "sd_model", SimpleNamespace(sd_checkpoint_info="checkpoint"), raising=False)
    monkeypatch.setattr(processing_module.devices, "autocast", contextlib.nullcontext)
    for name, value in {"use_old_scheduling": False, "sdxl_refiner_low_aesthetic_score": 2.5, "sdxl_refiner_high_aesthetic_score": 6.0}.items():
        monkeypatch.setattr(processing_module.opts, name, value, raising=False)
    for cls, slots in ((processing_module.StableDiffusionProcessing, ("cached_c", "cached_uc")), (processing_module.StableDiffusionProcessingTxt2Img, ("cached_hr_c", "cached_hr_uc"))):
        for slot in slots:
            monkeypatch.setattr(cls, slot, [None, None])
    processing_module.openclaw_cache_epochs.reset_for_tests()
    yield processing_module
    processing_module.openclaw_cache_epochs.reset_for_tests()


@pytest.fixture
def p(processing, monkeypatch):
    monkeypatch.setattr(processing.StableDiffusionProcessing, "active_lora_cond_signature", lambda self: ("lora", "none"))
    p = processing.StableDiffusionProcessingTxt2Img.__new__(processing.StableDiffusionProcessingTxt2Img)
    p.__dict__.update(width=1024, height=1024, extra_generation_params={})
    return p


def _prompts(processing, text="a cat"):
    return processing.prompt_parser.SdConditioning([text], width=1024, height=1024)


def _encoder(calls):
    def encode(model, required_prompts, steps, hires_steps, use_old_scheduling):
        calls.append(required_prompts[0])
        if required_prompts[0] == "fails":
            raise RuntimeError("encoder failed")
        return f"cond:{required_prompts[0]}"
    return encode


def _transaction_entered_signal(monkeypatch, epochs):
    """Make epoch_transaction() set the returned event once a caller holds it."""
    entered = threading.Event()
    real_transaction = epochs.epoch_transaction

    @contextlib.contextmanager
    def transaction():
        with real_transaction():
            entered.set()
            yield

    monkeypatch.setattr(epochs, "epoch_transaction", transaction)
    return entered


def _assert_commit_takes_the_epoch_transaction_first(epochs, entered, lock, commit, lock_name):
    """Run commit in a worker while this thread holds the commit's own lock. The worker must take the epoch
    transaction before blocking on that lock, and a concurrent epoch bump must then wait until the commit finishes."""
    bumped = threading.Event()
    worker = threading.Thread(target=commit, daemon=True)
    bumper = threading.Thread(target=lambda: (epochs.bump_epoch("conditioner_epoch", reason="conditioning_cleared"), bumped.set()), daemon=True)
    lock.acquire()
    try:
        worker.start()
        assert entered.wait(5), f"the epoch transaction must be taken before the {lock_name}"
        bumper.start()
        assert not bumped.wait(0.1)
    finally:
        lock.release()
        worker.join(5)
        if bumper.ident is not None:
            bumper.join(5)
    assert not worker.is_alive() and bumped.is_set()


def test_conditioning_key_tracks_dependency_epochs_and_effective_network_state(processing, p, monkeypatch):
    epochs = processing.openclaw_cache_epochs
    prompts = _prompts(processing)

    def key(namespace="c"):
        return p.cached_params(namespace, prompts, 20, None)

    seen = [key()]  # keys hold the prompt list (unhashable); compared with ==
    for dimension in CONDITIONING_EPOCHS:
        epochs.bump_epoch(dimension, reason="other")
        assert key() not in seen, dimension
        seen.append(key())

    current = key()
    # A LoRA transition reaches the key through the published effective network state, not through its epoch.
    for dimension in ("lora_applied_epoch", "vae_object_epoch", "vae_bytes_epoch"):
        epochs.bump_epoch(dimension, reason="other")
    assert key() == current
    monkeypatch.setattr(processing.StableDiffusionProcessing, "active_lora_cond_signature", lambda self: ("lora", "alpha:0.5"))
    assert key() != current

    # Each slot has its own namespace; sampling inputs are not conditioning inputs.
    keys = [key(namespace) for namespace in ("c", "uc", "hr_c", "hr_uc")]
    assert all(keys.count(namespace_key) == 1 for namespace_key in keys)
    current = key()
    p.__dict__.update(seed=1234, subseed=5, sampler_name="Euler a", scheduler="Karras", cfg_scale=3.0)
    assert key() == current


def test_effective_network_state_is_the_published_lora_identity(processing, monkeypatch):
    p = processing.StableDiffusionProcessingTxt2Img.__new__(processing.StableDiffusionProcessingTxt2Img)
    monkeypatch.setitem(sys.modules, "networks", SimpleNamespace(current_network_state_identity=lambda: ("published", "alpha")))
    assert p.active_lora_cond_signature() == ("published", "alpha")
    monkeypatch.setitem(sys.modules, "networks", None)  # the Lora extension is not loaded
    assert p.active_lora_cond_signature() == ()


def test_failed_conditioning_is_not_published_and_keeps_the_previous_entry(processing, p):
    cache = [None, None]
    calls = []
    encode = _encoder(calls)

    assert p.get_conds_with_caching("c", encode, _prompts(processing, "ok"), 20, cache, None) == "cond:ok"
    published = list(cache)
    assert published[:2] == [p.cached_params("c", _prompts(processing, "ok"), 20, None), "cond:ok"]
    with pytest.raises(RuntimeError, match="encoder failed"):
        p.get_conds_with_caching("c", encode, _prompts(processing, "fails"), 20, cache, None)
    assert cache == published
    assert p.get_conds_with_caching("c", encode, _prompts(processing, "ok"), 20, cache, None) == "cond:ok"
    assert calls == ["ok", "fails"]

    empty = [None, None]
    with pytest.raises(RuntimeError, match="encoder failed"):
        p.get_conds_with_caching("uc", encode, _prompts(processing, "fails"), 20, empty, None)
    assert empty == [None, None]


def test_conditioning_publication_takes_the_epoch_transaction_before_the_cache_lock(processing, p, monkeypatch):
    epochs = processing.openclaw_cache_epochs
    entered = _transaction_entered_signal(monkeypatch, epochs)
    cache = [None, None]
    results = []
    _assert_commit_takes_the_epoch_transaction_first(
        epochs, entered, processing.StableDiffusionProcessing.conditioning_cache_lock,
        lambda: results.append(p.get_conds_with_caching("c", _encoder([]), _prompts(processing), 20, cache, None)),
        "conditioning cache lock",
    )

    # The entry was keyed and published inside one transaction, before the concurrent bump landed.
    assert results == ["cond:a cat"]
    assert dict(cache[0][1])["conditioner_epoch"] == 0
    assert dict(epochs.epoch_subset(("conditioner_epoch",)))["conditioner_epoch"] == 1


def test_conditioning_cache_telemetry_reports_dependency_classes_and_slot_occupancy(processing, p, monkeypatch):
    epochs = processing.openclaw_cache_epochs
    observed, sizes = [], []
    monkeypatch.setattr(epochs, "observe", lambda family, event, *, reason, semantic_key=None, count=1: observed.append((family, event, reason, semantic_key)))
    monkeypatch.setattr(epochs, "set_size", lambda family, *, current_size=None, capacity=None: sizes.append((family, current_size, capacity)))
    calls = []
    encode = _encoder(calls)
    prompts = _prompts(processing)
    slot = processing.StableDiffusionProcessing.cached_c

    def run_and_key():
        key = p.cached_params("c", prompts, 20, None)
        p.get_conds_with_caching("c", encode, prompts, 20, slot, None)
        return key

    first = run_and_key()
    hit = run_and_key()
    epochs.bump_epoch("conditioner_epoch", reason="conditioning_cleared")
    changed = run_and_key()
    assert first == hit != changed
    assert observed == [
        ("E05", "miss", "cache_miss", first),
        ("E05", "publish", "published", first),
        ("E05", "hit", "cache_hit", first),
        ("E05", "miss", "dependency_changed", changed),
        ("E05", "publish", "published", changed),
    ]
    assert calls == ["a cat", "a cat"]
    # Size: the filled slots of the four process-wide conditioning caches.
    assert sizes == [("E05", 1, 4), ("E05", 1, 4)]
    p.get_conds_with_caching("hr_uc", encode, prompts, 20, processing.StableDiffusionProcessingTxt2Img.cached_hr_uc, None)
    assert sizes[-1] == ("E05", 2, 4)


def test_conditioning_cache_telemetry_is_instrumented_and_reports_only_key_digests(processing, p):
    epochs = processing.openclaw_cache_epochs
    p.get_conds_with_caching("c", _encoder([]), _prompts(processing, "ultra secret prompt"), 20, [None, None], None)

    snapshot = epochs.snapshot()
    families = {item["id"]: item for item in snapshot["families"]}
    assert all(families[family]["instrumented"] for family in ("E05", "E06", "E07"))
    assert families["E05"]["semantic_key_digests"]
    assert "ultra secret prompt" not in json.dumps(snapshot)


def test_prompt_token_memo_is_call_local_and_keyed_by_tokenizer_epoch(initialize, monkeypatch):
    from modules import sd_hijack_clip

    epochs = sd_hijack_clip.openclaw_cache_epochs
    observed, sizes, tokenized = [], [], []
    monkeypatch.setattr(epochs, "observe", lambda family, event, *, reason, semantic_key=None, count=1: observed.append((family, event, semantic_key)))
    monkeypatch.setattr(epochs, "set_size", lambda family, *, current_size=None, capacity=None: sizes.append((family, current_size, capacity)))

    class Model(sd_hijack_clip.TextConditionalModel):
        def tokenize_line(self, line):
            tokenized.append(line)
            return [f"chunks:{line}"], len(line)

    model = Model()
    epoch = epochs.epoch_subset(("tokenizer_epoch",))
    assert model.process_texts(["a cat", "dog", "a cat"]) == ([["chunks:a cat"], ["chunks:dog"], ["chunks:a cat"]], 5)
    assert tokenized == ["a cat", "dog"]
    assert observed == [
        ("E06", "miss", (epoch, "a cat")),
        ("E06", "publish", (epoch, "a cat")),
        ("E06", "miss", (epoch, "dog")),
        ("E06", "publish", (epoch, "dog")),
        ("E06", "hit", (epoch, "a cat")),
    ]
    assert sizes == [("E06", 0, 2)]  # nothing is retained after the call

    model.process_texts(["a cat"])
    assert tokenized == ["a cat", "dog", "a cat"]  # the next call tokenizes again


@pytest.fixture
def embedding_db(initialize, monkeypatch, tmp_path):
    from modules import hashes
    from modules.textual_inversion import textual_inversion as ti

    monkeypatch.setattr(hashes, "sha256", lambda *args, **kwargs: None)
    monkeypatch.setattr(ti.EmbeddingDatabase, "get_expected_shape", lambda self: 2048)  # 768-wide files are skipped
    ti.openclaw_cache_epochs.reset_for_tests()
    db = ti.EmbeddingDatabase()
    db.add_embedding_dir(str(tmp_path))
    yield ti, db, tmp_path
    ti.openclaw_cache_epochs.reset_for_tests()


def _save_embedding(directory, name, value=0.0):
    import safetensors.torch
    import torch

    safetensors.torch.save_file({"emb_params": torch.full((1, 768), value)}, str(directory / f"{name}.safetensors"))


def test_ti_reload_publishes_staged_maps_then_bumps_ti_and_tokenizer_epochs_once(embedding_db, monkeypatch):
    ti, db, directory = embedding_db
    epochs = ti.openclaw_cache_epochs
    events = []
    real_bump, real_load_from_file = epochs.bump_epoch, ti.EmbeddingDatabase.load_from_file

    def bump(dimension, *, reason):
        events.append(("bump", dimension, reason, sorted(db.skipped_embeddings)))
        return real_bump(dimension, reason=reason)

    def load_from_file(self, path, filename):
        events.append(("stage", filename, sorted(db.skipped_embeddings)))  # the live maps while staging
        return real_load_from_file(self, path, filename)

    monkeypatch.setattr(epochs, "bump_epoch", bump)
    monkeypatch.setattr(ti.EmbeddingDatabase, "load_from_file", load_from_file)

    assert not db.load_textual_inversion_embeddings(force_reload=True)  # empty folder: same (empty) maps
    _save_embedding(directory, "style")
    assert db.load_textual_inversion_embeddings(force_reload=True)
    assert events == [
        ("stage", "style.safetensors", []),
        ("bump", "textual_inversion_epoch", "textual_inversion_reloaded", ["style"]),
        ("bump", "tokenizer_epoch", "textual_inversion_reloaded", ["style"]),
    ]

    events.clear()
    assert not db.load_textual_inversion_embeddings(force_reload=True)  # unchanged: nothing published, no bump
    assert events == [("stage", "style.safetensors", ["style"])]
    assert dict(epochs.epoch_subset(("textual_inversion_epoch", "tokenizer_epoch"))) == {"textual_inversion_epoch": 1, "tokenizer_epoch": 1}


def test_ti_reload_takes_the_epoch_transaction_before_the_publication_lock(embedding_db, monkeypatch):
    ti, db, directory = embedding_db
    epochs = ti.openclaw_cache_epochs
    entered = _transaction_entered_signal(monkeypatch, epochs)
    _save_embedding(directory, "style")
    results = []
    _assert_commit_takes_the_epoch_transaction_first(
        epochs, entered, db._publication_lock,
        lambda: results.append(db.load_textual_inversion_embeddings(force_reload=True)),
        "TI publication lock",
    )

    assert results == [True]
    assert sorted(db.skipped_embeddings) == ["style"]
    assert dict(epochs.epoch_subset(("textual_inversion_epoch", "tokenizer_epoch", "conditioner_epoch"))) == {
        "textual_inversion_epoch": 1, "tokenizer_epoch": 1, "conditioner_epoch": 1,
    }


def _with_items(node):
    return [ast.unparse(item.context_expr) for item in node.items]


def test_clear_cond_cache_commits_inside_the_epoch_transaction_then_the_cache_lock():
    # Statically checked: importing the extension script installs process-wide sd_models hooks. Its behavior (slots
    # cleared, each epoch bumped once, granular targets) is tested in extensions/openclaw-clear-cond-cache/tests.
    tree = ast.parse(CLEAR_COND_CACHE.read_text(encoding="utf-8"))
    clear = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "clear_cond_cache")
    transaction = next(node for node in ast.walk(clear) if isinstance(node, ast.With) and _with_items(node) == ["openclaw_cache_epochs.epoch_transaction()"])
    cache_lock = next(node for node in transaction.body if isinstance(node, ast.With) and _with_items(node) == ["StableDiffusionProcessing.conditioning_cache_lock"])
    statements = sorted((node for node in ast.walk(ast.Module(body=cache_lock.body, type_ignores=[])) if isinstance(node, (ast.Assign, ast.Expr))), key=lambda node: node.lineno)
    order = [ast.unparse(node) for node in statements]
    clears = [order.index(f"{slot} = [None, None]") for slot in (
        "StableDiffusionProcessing.cached_c", "StableDiffusionProcessing.cached_uc",
        "StableDiffusionProcessingTxt2Img.cached_hr_c", "StableDiffusionProcessingTxt2Img.cached_hr_uc",
    )]
    bumps = [order.index(f"openclaw_cache_epochs.bump_epoch('{epoch}', reason='{reason}')") for epoch, reason in (
        ("conditioner_epoch", "conditioning_cleared"), ("conditioning_hook_epoch", "conditioning_hook_changed"),
    )]
    assert max(clears) < min(bumps)
