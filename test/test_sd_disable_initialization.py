"""The model-loading replacement contexts act on the loading thread only and restore exactly what they replaced.

A model load runs on one thread (the startup load thread, or a generation request switching checkpoints) while API
handlers keep running on others, so none of these replacements may be visible to another thread.
"""

import math
import threading

import pytest
import torch

from modules import paths  # noqa: F401  (puts repositories/ on sys.path for sd_disable_initialization's imports)
from modules import sd_disable_initialization, shared

# Blocks on different threads are serialized; a model load still running on a background thread (a test that ran
# initialize()) can hold the lock for a while, so entering waits generously rather than failing.
ENTER_TIMEOUT = 600

HOOK_CLASSES = (torch.nn.Module, torch.nn.Linear, torch.nn.Conv2d, torch.nn.MultiheadAttention, torch.nn.LayerNorm, torch.nn.GroupNorm)


@pytest.fixture
def ram_optimization(monkeypatch):
    monkeypatch.setattr(shared.cmd_opts, "disable_model_loading_ram_optimization", False)


@pytest.fixture
def class_dicts():
    """Snapshot the attributes the contexts replace, and put them back even if a context leaks one."""
    fields = ("__init__", "to", "load_state_dict", "_load_from_state_dict")
    clip = sd_disable_initialization.ldm.modules.encoders.modules.CLIPTextModel
    targets = [(cls, field) for cls in HOOK_CLASSES for field in fields] + [(clip, "from_pretrained")]
    targets += [(torch.nn.init, name) for name in ("kaiming_uniform_", "_no_grad_normal_", "_no_grad_uniform_")]
    missing = object()
    saved = [(obj, field, vars(obj).get(field, missing)) for obj, field in targets]
    yield {(obj, field): value for obj, field, value in saved if value is not missing}
    for obj, field, value in saved:
        if value is missing:
            if field in vars(obj):
                delattr(obj, field)
        elif vars(obj).get(field, missing) is not value:
            setattr(obj, field, value)


def run_inside(context, body=None):
    """Hold `context` open on a worker thread; return (stop, thread, results) once it is inside the block."""
    entered, stop, results = threading.Event(), threading.Event(), {}

    def worker():
        with context:
            if body is not None:
                results.update(body())
            entered.set()
            stop.wait(ENTER_TIMEOUT)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    assert entered.wait(ENTER_TIMEOUT), "worker never entered the context"
    return stop, thread, results


def finish(stop, thread):
    stop.set()
    thread.join(30)
    assert not thread.is_alive()


def test_initialize_on_meta_does_not_touch_other_threads(ram_optimization, class_dicts):
    def body():
        layer = torch.nn.Linear(2, 2)
        return {"meta": layer.weight.is_meta, "to_result_is_layer": layer.to("cpu") is layer}

    stop, thread, inside = run_inside(sd_disable_initialization.InitializeOnMeta(), body)
    try:
        layer = torch.nn.Linear(2, 2)
        conv = torch.nn.Conv2d(1, 1, 1)
        moved = layer.to(torch.float64)
    finally:
        finish(stop, thread)

    assert not layer.weight.is_meta and not conv.weight.is_meta
    assert moved is layer and layer.weight.dtype == torch.float64
    assert inside == {"meta": True, "to_result_is_layer": True}
    for obj, field in ((torch.nn.Linear, "__init__"), (torch.nn.Conv2d, "__init__"), (torch.nn.Module, "to")):
        assert vars(obj)[field] is class_dicts[(obj, field)]


def test_load_state_dict_on_meta_does_not_feed_other_threads_from_the_checkpoint(ram_optimization, class_dicts):
    checkpoint = {"child.weight": torch.full((2, 2), 7.0), "child.bias": torch.full((2,), 7.0)}
    stop, thread, _ = run_inside(sd_disable_initialization.LoadStateDictOnMeta(checkpoint, device="cpu"))
    try:
        parent = torch.nn.Module()
        parent.child = torch.nn.Linear(2, 2)
        parent.load_state_dict({"child.weight": torch.ones(2, 2), "child.bias": torch.zeros(2)})
    finally:
        finish(stop, thread)

    torch.testing.assert_close(parent.child.weight.detach(), torch.ones(2, 2))
    torch.testing.assert_close(parent.child.bias.detach(), torch.zeros(2))
    assert set(checkpoint) == {"child.weight", "child.bias"}, "another thread's load consumed the checkpoint's tensors"


def test_disable_initialization_does_not_disable_init_on_other_threads(class_dicts):
    stop, thread, _ = run_inside(sd_disable_initialization.DisableInitialization(disable_clip=False))
    try:
        weight = torch.full((4, 4), math.nan)
        torch.nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        bias = torch.full((4,), math.nan)
        torch.nn.init.uniform_(bias, -1, 1)
        embedding = torch.full((4,), math.nan)
        torch.nn.init.normal_(embedding)
    finally:
        finish(stop, thread)

    assert torch.isfinite(weight).all() and torch.isfinite(bias).all() and torch.isfinite(embedding).all()


def test_load_state_dict_on_meta_restores_inherited_hooks_exactly(ram_optimization, class_dicts):
    for cls in HOOK_CLASSES[1:]:  # without the Lora extension's hooks these classes inherit Module's
        if "_load_from_state_dict" in vars(cls):
            delattr(cls, "_load_from_state_dict")

    leftover = {"child.weight": torch.full((2, 2), 7.0), "child.bias": torch.full((2,), 7.0)}  # keys this load never consumes
    with sd_disable_initialization.LoadStateDictOnMeta(leftover, device="cpu"):
        pass

    for cls in HOOK_CLASSES[1:]:
        assert "_load_from_state_dict" not in vars(cls), f"{cls.__name__} kept a hook that closes over the old checkpoint"
    assert vars(torch.nn.Module)["_load_from_state_dict"] is class_dicts[(torch.nn.Module, "_load_from_state_dict")]

    parent = torch.nn.Module()
    parent.child = torch.nn.Linear(2, 2)
    parent.load_state_dict({"child.weight": torch.ones(2, 2), "child.bias": torch.zeros(2)})
    torch.testing.assert_close(parent.child.weight.detach(), torch.ones(2, 2))


def test_load_state_dict_on_meta_wraps_own_class_hooks_once(ram_optimization, class_dicts):
    calls = []
    original = torch.nn.Module._load_from_state_dict

    def own_hook(self, *args, **kwargs):  # stands in for the Lora extension's Linear hook
        calls.append(type(self).__name__)
        return original(self, *args, **kwargs)

    torch.nn.Linear._load_from_state_dict = own_hook
    checkpoint = {"child.weight": torch.ones(2, 2), "child.bias": torch.zeros(2)}
    parent = torch.nn.Module()
    parent.child = torch.nn.Linear(2, 2)
    with sd_disable_initialization.LoadStateDictOnMeta(checkpoint, device="cpu"):
        result = parent.load_state_dict(checkpoint, strict=False)

    assert calls == ["Linear"]
    assert checkpoint == {}, "the checkpoint's tensors are consumed as they load"
    torch.testing.assert_close(parent.child.weight.detach(), torch.ones(2, 2))
    assert result is not None, "the replaced load_state_dict must return torch's result"
    assert list(result.missing_keys) == [] and list(result.unexpected_keys) == []
    assert vars(torch.nn.Linear)["_load_from_state_dict"] is own_hook


def test_clip_from_pretrained_is_restored_to_the_inherited_classmethod(class_dicts):
    clip = sd_disable_initialization.ldm.modules.encoders.modules.CLIPTextModel
    with sd_disable_initialization.DisableInitialization(disable_clip=True):
        pass

    assert "from_pretrained" not in vars(clip), "a bound method of CLIPTextModel was left on the class"

    class Subclass(clip):
        pass

    assert clip.from_pretrained.__self__ is clip
    assert Subclass.from_pretrained.__self__ is Subclass


def test_patched_load_state_dict_keeps_the_torch_contract(ram_optimization, class_dicts):
    layer = torch.nn.Linear(2, 2)
    weights = {"weight": torch.ones(2, 2), "bias": torch.zeros(2)}
    with sd_disable_initialization.LoadStateDictOnMeta({}, device="cpu"):
        result = layer.load_state_dict(weights, strict=False, assign=True)

    assert list(result.missing_keys) == [] and list(result.unexpected_keys) == []
    assert layer.weight.data_ptr() == weights["weight"].data_ptr(), "assign=True must reach torch"


def test_root_level_parameter_keys_load_with_the_default_dtype(ram_optimization, class_dicts):
    class Root(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))

    model = Root()
    checkpoint = {"scale": torch.tensor([1.5, 2.5], dtype=torch.float32)}
    with sd_disable_initialization.LoadStateDictOnMeta(checkpoint, device="cpu", weight_dtype_conversion={"": torch.float64}):
        model.load_state_dict(checkpoint, strict=False)

    torch.testing.assert_close(model.scale.detach(), torch.tensor([1.5, 2.5], dtype=torch.float64))


def test_failed_install_restores_everything_and_releases_the_lock(monkeypatch, class_dicts):
    class BrokenOpenClip:
        def __getattr__(self, name):
            raise RuntimeError("open_clip unavailable")

    monkeypatch.setattr(sd_disable_initialization, "open_clip", BrokenOpenClip())
    with pytest.raises(RuntimeError, match="open_clip unavailable"):
        with sd_disable_initialization.DisableInitialization(disable_clip=True):
            pass

    for name in ("kaiming_uniform_", "_no_grad_normal_", "_no_grad_uniform_"):
        assert vars(torch.nn.init)[name] is class_dicts[(torch.nn.init, name)]

    entered = threading.Event()

    def other():
        with sd_disable_initialization.DisableInitialization(disable_clip=False):
            entered.set()

    thread = threading.Thread(target=other, daemon=True)
    thread.start()
    thread.join(ENTER_TIMEOUT)
    assert entered.is_set(), "the replacement lock stayed held after the failed install"


def test_local_files_only_requests_stay_local(monkeypatch, class_dicts):
    import transformers.configuration_utils

    calls = []

    def cached_file(url, filename, local_files_only=False, **kwargs):
        calls.append(local_files_only)
        raise OSError("not in the local cache")

    monkeypatch.setattr(transformers.configuration_utils, "cached_file", cached_file)
    with sd_disable_initialization.DisableInitialization(disable_clip=True):
        with pytest.raises(OSError):
            transformers.configuration_utils.cached_file("some/repo", "config.json", local_files_only=True)

    assert calls == [True], "a local-only lookup went to the network"


def test_clip_added_tokens_lookup_is_skipped(monkeypatch, class_dicts):
    """openai/clip-vit-large-patch14 has no added_tokens.json: the tokenizer lookup answers None without a request."""
    import transformers.tokenization_utils_base

    calls = []

    def cached_file(path_or_repo_id, filename, local_files_only=False, **kwargs):
        calls.append((path_or_repo_id, filename, local_files_only))
        return f"/hf-cache/{filename}"

    monkeypatch.setattr(transformers.tokenization_utils_base, "cached_file", cached_file)
    with sd_disable_initialization.DisableInitialization(disable_clip=True):
        added_tokens = transformers.tokenization_utils_base.cached_file("openai/clip-vit-large-patch14", "added_tokens.json")
        vocab = transformers.tokenization_utils_base.cached_file("openai/clip-vit-large-patch14", "vocab.json")

    assert added_tokens is None
    assert vocab == "/hf-cache/vocab.json"
    assert calls == [("openai/clip-vit-large-patch14", "vocab.json", True)]
