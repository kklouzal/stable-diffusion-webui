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


def bool_token(value: str) -> bool | None:
    """The boolean grammar above for one string: True, False, or None for any other text."""
    normalized = value.strip().lower()
    if normalized in _TRUE:
        return True
    if normalized in _FALSE:
        return False
    return None


def parse_bool(value: str, what: str) -> bool:
    """bool_token for one non-empty string; other text raises ValueError, which names the value `what`."""
    parsed = bool_token(value)
    if parsed is None:
        raise ValueError(f"{what}={value!r} is not a boolean; use 1/true/yes/on or 0/false/no/off")
    return parsed


def json_bool(data, key: str, default):
    """Field `key` of a JSON object as a boolean: missing -> default; null -> False, as bool(None); true/false, 0/1, or
    text in the grammar above ("false" is False: bool() made every non-empty string True). Anything else raises
    ValueError."""
    if not isinstance(data, dict) or key not in data:
        return default
    value = data[key]
    if value is None or isinstance(value, bool):
        return bool(value)
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        return parse_bool(value, key)
    raise ValueError(f"{key}={value!r} is not a boolean")


def env_bool(name: str, default):
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return parse_bool(value, name)


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
