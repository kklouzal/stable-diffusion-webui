"""Timing records shared by the Incantations submodules.

Submodules accumulate per-batch hook timings as ``{hook: {"total_seconds", "calls"}}``
(PAG additionally keeps ``{"details": {name: {"total_seconds", "calls"}}}``) and fold
them into ``p.openclaw_extension_timings`` at postprocess time. That processing-level
record is what Processed and the API expose::

    {"total_seconds": s, "extensions": {name: {"total_seconds": s, "calls": n,
                                               "hooks": {hook: s}, "details": {...}}}}

Seconds are rounded to 6 decimals at every accumulation step.
"""


def _accumulate(entry, elapsed, calls):
    entry["total_seconds"] = round(float(entry.get("total_seconds") or 0.0) + elapsed, 6)
    entry["calls"] = int(entry.get("calls") or 0) + calls


def record(timings, name, elapsed):
    """Add one call taking ``elapsed`` seconds to ``timings[name]``."""
    _accumulate(timings.setdefault(name, {"total_seconds": 0.0, "calls": 0}), float(elapsed), 1)


def merge_into_processing(p, extension, hook_timings):
    """Fold ``hook_timings`` into ``p.openclaw_extension_timings["extensions"][extension]``.

    The caller keeps ownership of ``hook_timings`` and resets it after merging.
    """
    if not hook_timings:
        return
    timings = getattr(p, "openclaw_extension_timings", None)
    if timings is None:
        timings = p.openclaw_extension_timings = {"total_seconds": 0.0, "extensions": {}}
    ext = timings["extensions"].setdefault(extension, {"total_seconds": 0.0, "calls": 0, "hooks": {}})
    for hook_name, hook in hook_timings.items():
        if hook_name == "details":
            continue
        elapsed = float(hook.get("total_seconds") or 0.0)
        timings["total_seconds"] = round(float(timings.get("total_seconds") or 0.0) + elapsed, 6)
        _accumulate(ext, elapsed, int(hook.get("calls") or 0))
        ext["hooks"][hook_name] = round(float(ext["hooks"].get(hook_name) or 0.0) + elapsed, 6)
    details = hook_timings.get("details")
    if details:
        ext_details = ext.setdefault("details", {})
        for detail_name, detail in details.items():
            entry = ext_details.setdefault(detail_name, {"total_seconds": 0.0, "calls": 0})
            _accumulate(entry, float(detail.get("total_seconds") or 0.0), int(detail.get("calls") or 0))
