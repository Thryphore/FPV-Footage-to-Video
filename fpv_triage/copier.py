"""Copy keeper clips into the output folder."""

from __future__ import annotations

import shutil
from pathlib import Path


def unique_destination(
    output_dir: Path,
    source: Path,
    reserved: set[str] | None = None,
) -> Path:
    """Avoid overwrites when two cards share the same filename."""
    reserved = reserved if reserved is not None else set()
    candidate = output_dir / source.name
    if not candidate.exists() and candidate.name not in reserved:
        return candidate

    stem = source.stem
    suffix = source.suffix
    n = 2
    while True:
        candidate = output_dir / f"{stem}_{n}{suffix}"
        if not candidate.exists() and candidate.name not in reserved:
            return candidate
        n += 1


def reserve_destination(
    output_dir: Path,
    source: Path,
    reserved: set[str],
) -> Path:
    """Pick a free destination name and mark it reserved before copy starts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    dest = unique_destination(output_dir, source, reserved)
    reserved.add(dest.name)
    return dest


def copy_to_destination(source: Path, dest: Path) -> Path:
    """
    Copy to dest atomically: write a .part file, then replace.

    Lets other work continue while copying, without exposing a half-written MP4.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".part")
    try:
        if partial.exists():
            partial.unlink()
        shutil.copy2(source, partial)
        partial.replace(dest)
    except Exception:
        if partial.exists():
            try:
                partial.unlink()
            except OSError:
                pass
        raise
    return dest


def copy_keeper(source: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    dest = unique_destination(output_dir, source)
    return copy_to_destination(source, dest)
