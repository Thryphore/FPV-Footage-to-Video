"""Persist last-used UI paths between app launches."""

from __future__ import annotations

import json
import os
from pathlib import Path


def prefs_path() -> Path:
    root = Path(os.environ.get("LOCALAPPDATA") or (Path.home() / ".config"))
    return root / "FPV-Footage-to-Video" / "prefs.json"


def load_prefs() -> dict:
    path = prefs_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def save_prefs(data: dict) -> None:
    path = prefs_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError:
        pass
