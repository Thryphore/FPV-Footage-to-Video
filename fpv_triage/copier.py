"""Copy keeper clips into the output folder."""

from __future__ import annotations

import shutil
from pathlib import Path


def unique_destination(output_dir: Path, source: Path) -> Path:
    """Avoid overwrites when two cards share the same filename."""
    candidate = output_dir / source.name
    if not candidate.exists():
        return candidate

    stem = source.stem
    suffix = source.suffix
    n = 2
    while True:
        candidate = output_dir / f"{stem}_{n}{suffix}"
        if not candidate.exists():
            return candidate
        n += 1


def copy_keeper(source: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    dest = unique_destination(output_dir, source)
    shutil.copy2(source, dest)
    return dest
