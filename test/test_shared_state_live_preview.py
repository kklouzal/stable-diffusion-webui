"""Live previews are decoded on the sampler thread, on request of a /progress poll (D-11): a poll never runs device
work concurrently with sampling or a CUDA graph capture."""

import threading
from types import SimpleNamespace

import pytest

from test.helpers import init_shared

shared = init_shared()

from modules import shared_state  # noqa: E402


@pytest.fixture
def state(monkeypatch):
    import modules.sd_samplers

    decoded = []

    def sample_to_image(latent):
        decoded.append((latent, threading.get_ident()))
        return SimpleNamespace(mode="RGB", latent=latent)

    monkeypatch.setattr(modules.sd_samplers, "sample_to_image", sample_to_image)
    for name, value in {"live_previews_enable": True, "show_progress_every_n_steps": 2, "show_progress_grid": False, "live_previews_image_format": "png"}.items():
        monkeypatch.setattr(shared.opts, name, value, raising=False)
    monkeypatch.setattr(shared, "parallel_processing_allowed", True)
    monkeypatch.setattr(shared_state.devices, "torch_gc", lambda: None)
    state = shared_state.State()
    state.begin("test")
    state.decoded = decoded
    return state


def _step(state, step, latent):
    state.sampling_step = step
    state.current_latent = latent


def test_previews_are_decoded_on_the_storing_thread_only_when_requested(state):
    _step(state, 2, "l2")
    assert state.decoded == [] and state.current_image is None  # nobody asked

    poller = threading.Thread(target=state.request_current_image)
    poller.start()
    poller.join()
    assert state.decoded == []  # the poll itself decodes nothing

    _step(state, 3, "l3")
    assert state.decoded == [("l3", threading.get_ident())]
    assert state.current_image.latent == "l3" and state.current_image_sampling_step == 3
    assert not state.current_image_requested

    state.request_current_image()
    _step(state, 4, "l4")  # within the preview period of the last preview
    assert len(state.decoded) == 1 and state.current_image_requested
    _step(state, 5, "l5")
    assert [latent for latent, _ in state.decoded] == ["l3", "l5"]


@pytest.mark.parametrize("setting", [("live_previews_enable", False), ("show_progress_every_n_steps", -1)])
def test_disabled_previews_are_not_decoded(state, monkeypatch, setting):
    monkeypatch.setattr(shared.opts, *setting, raising=False)
    state.request_current_image()
    _step(state, 10, "l10")
    assert state.decoded == []


def test_without_parallel_processing_store_latent_owns_the_previews(state, monkeypatch):
    monkeypatch.setattr(shared, "parallel_processing_allowed", False)
    state.request_current_image()
    _step(state, 10, "l10")
    assert state.decoded == []


def test_begin_clears_a_pending_request(state):
    state.request_current_image()
    state.begin("next")
    assert not state.current_image_requested and state.current_latent is None


def _progressapi(state, current_image):
    import ast
    from pathlib import Path

    source = Path("modules/api/api.py").read_text(encoding="utf-8")
    api_class = next(node for node in ast.parse(source).body if isinstance(node, ast.ClassDef) and node.name == "Api")
    method = next(node for node in api_class.body if isinstance(node, ast.FunctionDef) and node.name == "progressapi")
    method.args.defaults = []
    module = ast.Module(body=[ast.ClassDef(name="Api", bases=[], keywords=[], body=[method], decorator_list=[])], type_ignores=[])
    state.job_count, state.job_no, state.sampling_steps, state.time_start = 1, 0, 10, 1.0
    state.current_image = current_image
    namespace = {
        "shared": SimpleNamespace(state=state),
        "progress_module": SimpleNamespace(current_task=None, calculate_progress_and_eta=lambda *a, **k: (0.5, 1.0)),
        "encode_pil_to_base64": lambda image: f"b64:{image}",
        "models": SimpleNamespace(ProgressResponse=lambda **kwargs: kwargs, ProgressRequest=object),
    }
    exec(compile(ast.fix_missing_locations(module), "modules/api/api.py", "exec"), namespace)
    return namespace["Api"]().progressapi


def test_progress_returns_the_sampler_preview_and_requests_the_next(state):
    progressapi = _progressapi(state, "preview")

    assert progressapi(SimpleNamespace(skip_current_image=True))["current_image"] is None
    assert not state.current_image_requested  # skip_current_image asks for nothing (B-03)

    assert progressapi(SimpleNamespace(skip_current_image=False))["current_image"] == "b64:preview"
    assert state.current_image_requested
    assert state.decoded == []
