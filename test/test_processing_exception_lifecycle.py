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
    # process_images' finally deactivates (the two behavioral tests here); the inner loop must never deactivate itself,
    # or a failure after its own deactivation would deactivate twice. It runs the whole pipeline, so this is AST-checked.
    tree = ast.parse(Path("modules/processing.py").read_text())
    inner = _function(tree, "process_images_inner")

    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "deactivate"
        for node in ast.walk(inner)
    )


def test_processing_tracks_activation_before_it_can_raise():
    # activate can raise midway; process_images' finally deactivates only what was recorded, so the record comes first.
    inner = _function(ast.parse(Path("modules/processing.py").read_text()), "process_images_inner")
    [activate] = [node for node in ast.walk(inner) if isinstance(node, ast.Call) and ast.unparse(node.func) == "extra_networks.activate"]
    assert ast.unparse(activate) == "extra_networks.activate(p, p.extra_network_data)"

    # The statements directly before the one that (at any nesting level) activates.
    preceding = []
    for node in ast.walk(inner):
        for body in (getattr(node, field, None) for field in ("body", "orelse")):
            if isinstance(body, list):
                preceding += [ast.unparse(body[index - 1]) for index, statement in enumerate(body) if index and activate in set(ast.walk(statement))]

    assert "p._active_extra_network_data = p.extra_network_data" in preceding


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
        generation_last=SimpleNamespace(persist_or_report=lambda processing, snapshot: events.append(("persist", snapshot))),
    )

    with pytest.raises(CleanupFailure):
        process_images(p)

    assert events[-3:] == [
        ("deactivate", active_data),
        ("token-merging", 0),
        ("restore-overrides", {"stored": True}),
    ]
    assert p._active_extra_network_data is None
    # the request failed in its cleanup: it is not the last completed generation
    assert not any(event[0] == "persist" for event in events)


def test_last_generation_snapshot_persists_after_cleanup_succeeded():
    events = []
    active_data = {"lora": ["example"]}
    snapshot = {"schema_version": 3}
    result = object()
    p = SimpleNamespace(
        scripts=None,
        sd_model=object(),
        disable_extra_networks=False,
        override_settings_restore_afterwards=True,
        get_token_merging_ratio=lambda: 0.5,
    )

    def process_images_inner(processing):
        processing._active_extra_network_data = active_data
        # built in process_images_inner, while the request's state is live
        processing._generation_last_snapshot = snapshot
        return result

    process_images = _load_process_images(
        store_processing_override_settings=lambda processing: {"stored": True},
        apply_processing_override_settings=lambda processing: None,
        restore_processing_override_settings=lambda stored: events.append(("restore-overrides", stored)),
        process_images_inner=process_images_inner,
        sd_models=SimpleNamespace(apply_token_merging=lambda model, ratio: events.append(("token-merging", ratio))),
        sd_samplers=SimpleNamespace(fix_p_invalid_sampler_and_scheduler=lambda processing: None),
        profiling=SimpleNamespace(Profiler=nullcontext),
        extra_networks=SimpleNamespace(deactivate=lambda processing, data: events.append(("deactivate", data))),
        generation_last=SimpleNamespace(persist_or_report=lambda processing, captured: events.append(("persist", captured))),
    )

    assert process_images(p) is result
    assert events == [
        ("token-merging", 0.5),
        ("deactivate", active_data),
        ("token-merging", 0),
        ("restore-overrides", {"stored": True}),
        ("persist", snapshot),
    ]
    assert p._generation_last_snapshot is None


def test_last_generation_snapshot_is_built_in_the_inner_loop_and_not_persisted_there():
    inner = _function(ast.parse(Path("modules/processing.py").read_text()), "process_images_inner")
    calls = {ast.unparse(node.func) for node in ast.walk(inner) if isinstance(node, ast.Call)}
    assert "generation_last.snapshot_or_report" in calls
    assert not calls & {"generation_last.persist_or_report", "generation_last.capture_or_report", "generation_last.capture_completed_generation"}
