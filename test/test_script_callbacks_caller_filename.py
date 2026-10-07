import inspect
import time

import pytest

from modules import script_callbacks

# Oracle: the previous inspect.stack()-based lookup, compiled under script_callbacks' filename so its own frame
# is skipped exactly like the real add_callback's.
_ORACLE_SOURCE = '''
def oracle_add_callback(callbacks, fun, *, name=None, category='unknown', filename=None):
    stack = [x for x in inspect.stack() if x.filename != __file__]
    callbacks.append(stack[0].filename if stack else 'unknown file')
'''


def _oracle():
    namespace = {"inspect": inspect, "__file__": script_callbacks.__file__}
    exec(compile(_ORACLE_SOURCE, script_callbacks.__file__, "exec"), namespace)
    return namespace["oracle_add_callback"]


def _resolved_with(monkeypatch, impl, register):
    lists = {key: [] for key in script_callbacks.callback_map}
    with monkeypatch.context() as patch:
        patch.setattr(script_callbacks, "callback_map", lists)
        if impl is not None:
            patch.setattr(script_callbacks, "add_callback", impl)
        register()
    (entry,) = [item for items in lists.values() for item in items]
    return entry if isinstance(entry, str) else entry.script


def _via_extension_script(filename):
    code = compile("script_callbacks.on_cfg_denoiser(lambda params: None)\n", filename, "exec")
    return lambda: exec(code, {"script_callbacks": script_callbacks})


@pytest.mark.parametrize("register, expected", [
    pytest.param(lambda: script_callbacks.on_app_started(lambda demo, app: None), __file__, id="test-module"),
    pytest.param(_via_extension_script("/nonexistent/extensions/fake-ext/scripts/fake.py"), "/nonexistent/extensions/fake-ext/scripts/fake.py", id="missing-file"),
    pytest.param(_via_extension_script(__file__), __file__, id="existing-file"),
    pytest.param(_via_extension_script("<string>"), "<string>", id="pseudo-file"),
])
def test_caller_filename_matches_the_inspect_stack_lookup(monkeypatch, register, expected):
    assert _resolved_with(monkeypatch, _oracle(), register) == expected
    assert _resolved_with(monkeypatch, None, register) == expected


def test_registration_no_longer_builds_the_whole_stack(monkeypatch):
    def frames_only(*args, **kwargs):
        raise AssertionError("add_callback must not call inspect.stack()")

    monkeypatch.setattr(inspect, "stack", frames_only)
    started = time.perf_counter()
    for _ in range(200):
        _resolved_with(monkeypatch, None, lambda: script_callbacks.on_cfg_denoiser(lambda params: None))
    assert (time.perf_counter() - started) / 200 < 0.005
