"""In-process worker — calls backend modules directly, no subprocess.

Tabs can use:
  - stream_events(cfg, module_name)  -> generator of raw event dicts
  - stream_logs(cfg, module_name)    -> generator of formatted log strings

This replaces the previous subprocess.Popen approach. Backend modules
(`train`, `infer`, `evaluate`, `inspect_dataset`, `auth_status`) are now
imported and called in-process; their `emit()` calls are captured by the
`CapturingEmitter` in `common.py` and replayed to the UI as event dicts.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

from ._helpers import BACKEND, format_event

if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

# Lazy-import common only when needed so the UI loads even if backend deps
# (torch, transformers, etc.) aren't installed yet.
_common = None


def _get_common():
    global _common
    if _common is None:
        import importlib
        _common = importlib.import_module("common")
    return _common


# Backend module name -> module import path
MODULE_MAP = {
    "train.py":          "train",
    "infer.py":          "infer",
    "evaluate.py":       "evaluate",
    "inspect_dataset.py": "inspect_dataset",
    "auth_status.py":    "auth_status",
}


def _resolve_module(script_name: str):
    """Resolve a script name like 'train.py' to a backend module name."""
    if script_name in MODULE_MAP:
        return MODULE_MAP[script_name]
    # Allow callers to pass a bare module name too.
    return script_name.removesuffix(".py")


def stream_events(config: dict, script_name: str):
    """Generator over parsed event dicts from an in-process worker run."""
    common = _get_common()
    module_name = _resolve_module(script_name)
    yield from common.run_in_process(module_name, config)


def stream_logs(config: dict, script_name: str, *, tail: int = 400):
    """Yield accumulating log text for a backend run (formatted display strings)."""
    lines: list[str] = []
    for event in stream_events(config, script_name):
        lines.append(format_event(event))
        yield "\n".join(lines[-tail:])
