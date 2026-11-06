"""Discover video files under a source path."""

from __future__ import annotations

from pathlib import Path

from .analyzer import VIDEO_EXTENSIONS

LRF_EXTENSION = ".lrf"


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


def find_lrf_files(source: Path) -> list[Path]:
    """Find DJI low-resolution proxy (.LRF) files under a folder/drive."""
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(f"Source path does not exist: {source}")
    if source.is_file():
        return [source] if source.suffix.lower() == LRF_EXTENSION else []
    return sorted(
        path
        for path in source.rglob("*")
        if path.is_file() and path.suffix.lower() == LRF_EXTENSION
    )


def delete_lrf_files(source: Path) -> tuple[list[Path], list[tuple[Path, str]]]:
    """
    Delete all .LRF files under source.

    Returns (deleted paths, failures as (path, error)).
    """
    deleted: list[Path] = []
    failed: list[tuple[Path, str]] = []
    for path in find_lrf_files(source):
        try:
            path.unlink()
            deleted.append(path)
        except OSError as exc:
            failed.append((path, str(exc)))
    return deleted, failed
