"""One grammar for OpenClaw runtime environment knobs.

Booleans: unset or empty -> default; "1"/"true"/"yes"/"on" -> True and "0"/"false"/"no"/"off" -> False
(case-insensitive, surrounding whitespace ignored); any other value raises ValueError.
Integers: unset or empty -> default; otherwise a base-10 integer >= minimum, else ValueError.

Callers read their knobs once at import or startup, so a malformed value fails there instead of
silently falling back to a default.
"""

from __future__ import annotations

import os

_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"0", "false", "no", "off"})


def env_bool(name: str, default):
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    normalized = value.strip().lower()
    if normalized in _TRUE:
        return True
    if normalized in _FALSE:
        return False
    raise ValueError(f"{name}={value!r} is not a boolean; use 1/true/yes/on or 0/false/no/off")


def env_int(name: str, default: int, *, minimum: int) -> int:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    try:
        parsed = int(value)
    except ValueError:
        raise ValueError(f"{name}={value!r} is not an integer") from None
    if parsed < minimum:
        raise ValueError(f"{name}={parsed} is below the minimum {minimum}")
    return parsed
