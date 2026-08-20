"""Configuration loading with recursive merging and environment expansion."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)")


def deep_update(destination: dict, source: dict) -> dict:
    """Recursively merge ``source`` into ``destination``."""
    for key, value in source.items():
        if isinstance(value, dict) and key in destination:
            if not isinstance(destination[key], dict):
                raise TypeError(f"Cannot merge a mapping into non-mapping key {key!r}")
            deep_update(destination[key], value)
        else:
            destination[key] = value
    return destination


def _expand(value: Any, *, location: str = "config") -> Any:
    if isinstance(value, dict):
        return {key: _expand(item, location=f"{location}.{key}") for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item, location=f"{location}[{index}]") for index, item in enumerate(value)]
    if not isinstance(value, str):
        return value

    names = {first or second for first, second in _ENV_PATTERN.findall(value)}
    missing = sorted(name for name in names if not os.environ.get(name))
    if missing:
        joined = ", ".join(missing)
        raise EnvironmentError(
            f"Required environment variable(s) {joined} are not set for {location}: {value!r}"
        )
    return os.path.expandvars(value)


def load_config(config_file: str | os.PathLike[str]) -> dict:
    """Load JSON, merge an optional parent, then expand variables recursively.

    Parent paths are resolved relative to the child file. Unset ``${NAME}``
    references fail immediately with a location-aware error.
    """
    if not config_file:
        raise ValueError("A configuration path is required")
    path = Path(config_file).expanduser().resolve()
    with path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise TypeError(f"Configuration root must be a JSON object: {path}")

    merged: dict = {}
    parent = raw.get("parent")
    if parent:
        parent_path = _expand(parent, location="config.parent")
        parent_path = Path(parent_path).expanduser()
        if not parent_path.is_absolute():
            parent_path = path.parent / parent_path
        deep_update(merged, load_config(parent_path))
    child = dict(raw)
    child.pop("parent", None)
    deep_update(merged, child)
    return _expand(merged)
