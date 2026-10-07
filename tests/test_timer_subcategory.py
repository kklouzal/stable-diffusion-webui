import types

from modules import timer


def test_subcategory_time_is_recorded_once(monkeypatch):
    now = [0.0]
    clock = types.SimpleNamespace(time=lambda: now[0], perf_counter=lambda: now[0])
    monkeypatch.setattr(timer, "time", clock)

    t = timer.Timer()
    now[0] = 1.0
    t.record("prepare")
    with t.subcategory("extensions"):
        now[0] = 3.0
        t.record("first")
        now[0] = 6.0  # time after the last inner record
    t.record("after")

    assert t.records == {"prepare": 1.0, "extensions/first": 2.0, "extensions": 5.0, "after": 0.0}
    assert t.total == 6.0
    assert t.summary() == "6.0s (prepare: 1.0s, extensions: 5.0s)"
