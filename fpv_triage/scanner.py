"""Discover video files under a source path."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from .analyzer import VIDEO_EXTENSIONS

LRF_EXTENSION = ".lrf"
SRT_EXTENSION = ".srt"

SourceInput = Path | str | Sequence[Path | str]


def normalize_sources(source: SourceInput) -> list[Path]:
    """Accept one path or a sequence; drop blanks; require at least one."""
    if isinstance(source, (str, Path)):
        items: list[Path | str] = [source]
    else:
        items = list(source)
    paths = [Path(p) for p in items if str(p).strip()]
    if not paths:
        raise FileNotFoundError("No source folder provided")
    return paths


def _find_videos_one(source: Path, extensions: set[str]) -> list[Path]:
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(f"Source path does not exist: {source}")

    if source.is_file():
        return [source] if source.suffix.lower() in extensions else []

    return sorted(
        path
        for path in source.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    )


def find_videos(source: SourceInput, extensions: set[str] | None = None) -> list[Path]:
    """Find videos under one or more source folders (deduped by resolved path)."""
    exts = {
        e.lower() if e.startswith(".") else f".{e.lower()}"
        for e in (extensions or VIDEO_EXTENSIONS)
    }
    seen: set[str] = set()
    files: list[Path] = []
    for root in normalize_sources(source):
        for path in _find_videos_one(root, exts):
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            files.append(path)
    return files


def _find_ext_one(source: Path, extension: str) -> list[Path]:
    source = Path(source)
    ext = extension.lower() if extension.startswith(".") else f".{extension.lower()}"
    if not source.exists():
        raise FileNotFoundError(f"Source path does not exist: {source}")
    if source.is_file():
        return [source] if source.suffix.lower() == ext else []
    return sorted(
        path
        for path in source.rglob("*")
        if path.is_file() and path.suffix.lower() == ext
    )


def find_files_with_extension(source: SourceInput, extension: str) -> list[Path]:
    """Find files with a given extension under one or more folders/drives."""
    seen: set[str] = set()
    files: list[Path] = []
    for root in normalize_sources(source):
        for path in _find_ext_one(root, extension):
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            files.append(path)
    return files


def delete_files_with_extension(
    source: SourceInput, extension: str
) -> tuple[list[Path], list[tuple[Path, str]]]:
    """
    Delete all files with extension under source (one folder or several).

    Returns (deleted paths, failures as (path, error)).
    """
    deleted: list[Path] = []
    failed: list[tuple[Path, str]] = []
    for path in find_files_with_extension(source, extension):
        try:
            path.unlink()
            deleted.append(path)
        except OSError as exc:
            failed.append((path, str(exc)))
    return deleted, failed


def find_lrf_files(source: SourceInput) -> list[Path]:
    """Find DJI low-resolution proxy (.LRF) files under one or more folders/drives."""
    return find_files_with_extension(source, LRF_EXTENSION)


def delete_lrf_files(source: SourceInput) -> tuple[list[Path], list[tuple[Path, str]]]:
    """Delete all .LRF files under source (one folder or several)."""
    return delete_files_with_extension(source, LRF_EXTENSION)


def find_srt_files(source: SourceInput) -> list[Path]:
    """Find FPV/goggles subtitle (.SRT) files under one or more folders/drives."""
    return find_files_with_extension(source, SRT_EXTENSION)


def delete_srt_files(source: SourceInput) -> tuple[list[Path], list[tuple[Path, str]]]:
    """Delete all .SRT files under source (one folder or several)."""
    return delete_files_with_extension(source, SRT_EXTENSION)
