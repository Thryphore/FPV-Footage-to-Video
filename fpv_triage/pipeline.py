"""Shared triage pipeline used by the CLI and desktop app."""

from __future__ import annotations

import json
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .analyzer import AnalyzerConfig, FlightAnalysis, analyze_video
from .copier import copy_to_destination, reserve_destination
from .describer import DEFAULT_MODEL_ID, VideoDescriber, write_notes_file
from .scanner import find_videos


ProgressCallback = Callable[[int, int, str], None]
ResultCallback = Callable[[FlightAnalysis, str | None], None]


@dataclass
class TriageSummary:
    scanned: int
    kept: int
    rejected: int
    report_path: Path
    results: list[dict]


@dataclass
class _PendingKeep:
    analysis: FlightAnalysis
    timing: dict[str, float]
    t_clip: float
    copied_to: str | None
    notes_path: str | None
    copy_future: Future | None = None
    copy_t0: float | None = None
    order: int = 0


@dataclass
class _PipelineState:
    results: list[dict] = field(default_factory=list)
    kept: int = 0
    rejected: int = 0
    reserved_names: set[str] = field(default_factory=set)
    pending: deque[_PendingKeep] = field(default_factory=deque)
    finished_keeps: list[_PendingKeep] = field(default_factory=list)


def _finish_keep(
    job: _PendingKeep,
    *,
    on_result: ResultCallback | None,
) -> dict:
    if job.copy_future is not None and job.copy_t0 is not None:
        try:
            dest = job.copy_future.result()
            job.timing["copy"] = time.perf_counter() - job.copy_t0
            job.copied_to = str(dest)
        except Exception as exc:  # noqa: BLE001
            job.timing["copy"] = time.perf_counter() - job.copy_t0
            job.copied_to = None
            job.analysis.reason = f"copy failed: {exc}"

    job.timing["total"] = time.perf_counter() - job.t_clip
    job.analysis.timing = {k: round(v, 3) for k, v in job.timing.items()}

    entry = job.analysis.to_dict()
    entry["copied_to"] = job.copied_to
    entry["notes_path"] = job.notes_path

    if on_result:
        on_result(job.analysis, job.copied_to)
    return entry


def _reap_pending(
    state: _PipelineState,
    *,
    on_result: ResultCallback | None,
    wait: bool,
) -> None:
    """Emit completed keep jobs in order (head-of-queue only unless wait)."""
    while state.pending:
        head = state.pending[0]
        if head.copy_future is not None and not head.copy_future.done():
            if not wait:
                return
            # Block only when draining / shutting down.
        entry = _finish_keep(state.pending.popleft(), on_result=on_result)
        state.results.append(entry)


def run_triage(
    source: Path,
    output: Path,
    config: AnalyzerConfig,
    *,
    copy_videos: bool = True,
    dry_run: bool | None = None,
    describe: bool = True,
    model_id: str | None = None,
    extensions: set[str] | None = None,
    report_path: Path | None = None,
    should_cancel: Callable[[], bool] | None = None,
    on_progress: ProgressCallback | None = None,
    on_result: ResultCallback | None = None,
) -> TriageSummary:
    # dry_run=True is the old CLI flag for "don't copy videos".
    if dry_run is not None:
        copy_videos = not bool(dry_run)

    videos = find_videos(source, extensions)
    total = len(videos)
    describer: VideoDescriber | None = (
        VideoDescriber(model_id=model_id or DEFAULT_MODEL_ID) if describe else None
    )
    state = _PipelineState()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    # One background copy is enough to overlap with analyze/AI; two helps when
    # copies are huge and the next keep arrives before the previous finishes.
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="fpv-copy") as copies:
        for index, video in enumerate(videos, start=1):
            if should_cancel and should_cancel():
                break

            # Publish any copies that finished while we were analyzing others.
            _reap_pending(state, on_result=on_result, wait=False)

            if on_progress:
                on_progress(index, total, video.name)

            timing: dict[str, float] = {}
            t_clip = time.perf_counter()

            t0 = time.perf_counter()
            analysis = analyze_video(video, config)
            timing["analyze"] = time.perf_counter() - t0

            notes_path: str | None = None
            copied_to: str | None = None
            copy_future: Future | None = None
            copy_t0: float | None = None

            if analysis.keep:
                state.kept += 1

                if copy_videos:
                    dest = reserve_destination(output, video, state.reserved_names)
                    copied_to = str(dest)
                    copy_t0 = time.perf_counter()
                    copy_future = copies.submit(copy_to_destination, video, dest)
                    if on_progress:
                        on_progress(
                            index,
                            total,
                            f"Copying {video.name} (background)…",
                        )

                if describe and describer is not None:
                    if on_progress:
                        on_progress(index, total, f"Describing {video.name}")
                    try:
                        notes = describer.describe(video)
                        if notes.load_seconds > 0.05:
                            timing["model_load"] = notes.load_seconds
                        timing["keyframes"] = notes.keyframe_seconds
                        timing["inference"] = notes.inference_seconds
                        analysis.environment_tag = notes.environment_tag
                        analysis.people = notes.people
                        analysis.action = notes.action
                        analysis.description = notes.movement
                    except Exception as exc:  # noqa: BLE001
                        analysis.environment_tag = "unknown"
                        analysis.people = "none"
                        analysis.action = "unspecified"
                        analysis.description = f"Description failed: {exc}"

                    if copied_to:
                        txt_path = Path(copied_to).with_suffix(".txt")
                    else:
                        txt_path = Path(output) / f"{video.stem}.txt"
                    notes_file = write_notes_file(
                        analysis,
                        txt_path,
                        environment_tag=analysis.environment_tag,
                        people=analysis.people,
                        action=analysis.action,
                        movement=analysis.description,
                        video_path=Path(copied_to) if copied_to else video,
                    )
                    notes_path = str(notes_file)
                    analysis.notes_path = notes_path

                state.pending.append(
                    _PendingKeep(
                        analysis=analysis,
                        timing=timing,
                        t_clip=t_clip,
                        copied_to=copied_to,
                        notes_path=notes_path,
                        copy_future=copy_future,
                        copy_t0=copy_t0,
                        order=index,
                    )
                )
                # If copy already finished during describe, log it now.
                _reap_pending(state, on_result=on_result, wait=False)
            else:
                state.rejected += 1
                timing["total"] = time.perf_counter() - t_clip
                analysis.timing = {k: round(v, 3) for k, v in timing.items()}
                entry = analysis.to_dict()
                entry["copied_to"] = None
                entry["notes_path"] = None
                state.results.append(entry)
                if on_result:
                    on_result(analysis, None)

        if on_progress and state.pending:
            on_progress(total, total, "Finishing background copies…")
        _reap_pending(state, on_result=on_result, wait=True)

    # Refresh searchable catalog + embeddings for the output folder.
    if describe and state.kept > 0 and not (should_cancel and should_cancel()):
        try:
            from .catalog import rebuild_search_assets

            if on_progress:
                on_progress(total, total, "Building AI search index…")
            rebuild_search_assets(output)
        except Exception:
            # Search index is optional; triage results still stand.
            pass

    # Keep report order stable: rejects interleaved by scan order already;
    # pending keeps were appended when copies finished (still roughly ordered).
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(Path(source).resolve()),
        "output": str(Path(output).resolve()),
        "copy_videos": bool(copy_videos),
        "dry_run": not bool(copy_videos),
        "describe": bool(describe),
        "cancelled": bool(should_cancel and should_cancel()),
        "config": {
            "proof_flight_seconds": config.proof_flight_seconds,
            "remaining_after_proof": config.remaining_after_proof,
            "sample_fps": config.sample_fps,
            "motion_threshold": config.motion_threshold,
            "gap_seconds": config.gap_seconds,
            "model_id": (model_id or DEFAULT_MODEL_ID) if describe else None,
        },
        "summary": {
            "scanned": len(state.results),
            "kept": state.kept,
            "rejected": state.rejected,
        },
        "results": state.results,
    }

    out_report = Path(report_path) if report_path else Path(output) / "triage_report.json"
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return TriageSummary(
        scanned=len(state.results),
        kept=state.kept,
        rejected=state.rejected,
        report_path=out_report,
        results=state.results,
    )
