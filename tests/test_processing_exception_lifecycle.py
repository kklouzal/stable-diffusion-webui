import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest


def _function(tree, name):
    return next(node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name)


def _load_process_images(**namespace):
    tree = ast.parse(Path("modules/processing.py").read_text())
    function = _function(tree, "process_images")
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace |= {"StableDiffusionProcessing": object, "Processed": object}
    exec(compile(module, "modules/processing.py", "exec"), namespace)
    return namespace["process_images"]


def test_generation_exception_deactivates_extra_networks_and_restores_state():
    events = []
    active_data = {"lora": ["example"]}
    p = SimpleNamespace(
        scripts=None,
        sd_model=object(),
        disable_extra_networks=False,
        override_settings_restore_afterwards=True,
        get_token_merging_ratio=lambda: 0.5,
    )

    class GenerationFailure(RuntimeError):
        pass

    def process_images_inner(processing):
        processing._active_extra_network_data = active_data
        events.append("generate")
        raise GenerationFailure

    process_images = _load_process_images(
        store_processing_override_settings=lambda processing: {"stored": True},
        apply_processing_override_settings=lambda processing: events.append("apply-overrides"),
        restore_processing_override_settings=lambda stored: events.append(("restore-overrides", stored)),
        process_images_inner=process_images_inner,
        sd_models=SimpleNamespace(apply_token_merging=lambda model, ratio: events.append(("token-merging", ratio))),
        sd_samplers=SimpleNamespace(fix_p_invalid_sampler_and_scheduler=lambda processing: events.append("fix-sampler")),
        profiling=SimpleNamespace(Profiler=nullcontext),
        extra_networks=SimpleNamespace(deactivate=lambda processing, data: events.append(("deactivate", data))),
    )

    with pytest.raises(GenerationFailure):
        process_images(p)

    assert ("deactivate", active_data) in events
    assert events.index(("deactivate", active_data)) < events.index(("token-merging", 0))
    assert events[-1] == ("restore-overrides", {"stored": True})
    assert p._active_extra_network_data is None


def test_extra_network_cleanup_is_owned_by_processing_finally():
    tree = ast.parse(Path("modules/processing.py").read_text())
    outer = _function(tree, "process_images")
    inner = _function(tree, "process_images_inner")
    outer_try = next(node for node in outer.body if isinstance(node, ast.Try))
    outer_finally_source = ast.unparse(ast.Module(body=outer_try.finalbody, type_ignores=[]))

    assert "active_extra_network_data = p._active_extra_network_data" in outer_finally_source
    assert "extra_networks.deactivate(p, active_extra_network_data)" in outer_finally_source
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "deactivate"
        for node in ast.walk(inner)
    )


def test_model_and_override_cleanup_survive_extra_network_cleanup_failure():
    tree = ast.parse(Path("modules/processing.py").read_text())
    outer = _function(tree, "process_images")
    outer_try = next(node for node in outer.body if isinstance(node, ast.Try))
    cleanup_try = next(node for node in outer_try.finalbody if isinstance(node, ast.Try))
    guaranteed_cleanup_source = ast.unparse(ast.Module(body=cleanup_try.finalbody, type_ignores=[]))

    assert "p._active_extra_network_data = None" in guaranteed_cleanup_source
    assert "sd_models.apply_token_merging(p.sd_model, 0)" in guaranteed_cleanup_source
    assert "restore_processing_override_settings(stored_opts)" in guaranteed_cleanup_source


def test_processing_tracks_activation_before_it_can_raise():
    source = Path("modules/processing.py").read_text()
    start = source.index("def process_images_inner")
    end = source.index("@dataclass(repr=False)", start)
    inner = source[start:end]
    activate_at = inner.index("extra_networks.activate(p, p.extra_network_data)")
    tracked_at = inner.rindex("p._active_extra_network_data = p.extra_network_data", 0, activate_at)

    assert tracked_at < activate_at


def test_cleanup_failure_still_restores_model_and_overrides():
    events = []
    active_data = {"hypernet": ["example"]}
    p = SimpleNamespace(
        scripts=None,
        sd_model=object(),
        disable_extra_networks=False,
        override_settings_restore_afterwards=True,
        get_token_merging_ratio=lambda: 0.5,
    )

    class CleanupFailure(RuntimeError):
        pass

    def process_images_inner(processing):
        processing._active_extra_network_data = active_data
        return object()

    def deactivate(processing, data):
        events.append(("deactivate", data))
        raise CleanupFailure

    process_images = _load_process_images(
        store_processing_override_settings=lambda processing: {"stored": True},
        apply_processing_override_settings=lambda processing: None,
        restore_processing_override_settings=lambda stored: events.append(("restore-overrides", stored)),
        process_images_inner=process_images_inner,
        sd_models=SimpleNamespace(apply_token_merging=lambda model, ratio: events.append(("token-merging", ratio))),
        sd_samplers=SimpleNamespace(fix_p_invalid_sampler_and_scheduler=lambda processing: None),
        profiling=SimpleNamespace(Profiler=nullcontext),
        extra_networks=SimpleNamespace(deactivate=deactivate),
    )

    with pytest.raises(CleanupFailure):
        process_images(p)

    assert events[-3:] == [
        ("deactivate", active_data),
        ("token-merging", 0),
        ("restore-overrides", {"stored": True}),
    ]
    assert p._active_extra_network_data is None
