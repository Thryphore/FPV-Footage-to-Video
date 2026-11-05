"""Discover video files under a source path."""

from __future__ import annotations

from pathlib import Path

from .analyzer import VIDEO_EXTENSIONS


def find_videos(source: Path, extensions: set[str] | None = None) -> list[Path]:
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(f"Source path does not exist: {source}")

    exts = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in (extensions or VIDEO_EXTENSIONS)}
    files: list[Path] = []

    if source.is_file():
        if source.suffix.lower() in exts:
            files.append(source)
        return files

    for path in source.rglob("*"):
        if path.is_file() and path.suffix.lower() in exts:
            files.append(path)

    return sorted(files)
