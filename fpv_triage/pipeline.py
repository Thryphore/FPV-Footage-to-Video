"""Shared triage pipeline used by the CLI and desktop app."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .analyzer import AnalyzerConfig, FlightAnalysis, analyze_video
from .copier import copy_keeper
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


def run_triage(
    source: Path,
    output: Path,
    config: AnalyzerConfig,
    *,
    dry_run: bool = False,
    describe: bool = True,
    model_id: str | None = None,
    extensions: set[str] | None = None,
    report_path: Path | None = None,
    should_cancel: Callable[[], bool] | None = None,
    on_progress: ProgressCallback | None = None,
    on_result: ResultCallback | None = None,
) -> TriageSummary:
    videos = find_videos(source, extensions)
    results: list[dict] = []
    kept = 0
    rejected = 0
    total = len(videos)
    describer: VideoDescriber | None = (
        VideoDescriber(model_id=model_id or DEFAULT_MODEL_ID) if describe else None
    )

    for index, video in enumerate(videos, start=1):
        if should_cancel and should_cancel():
            break

        if on_progress:
            on_progress(index, total, video.name)

        analysis = analyze_video(video, config)
        copied_to: str | None = None
        notes_path: str | None = None

        if analysis.keep:
            kept += 1
            if not dry_run:
                dest = copy_keeper(video, output)
                copied_to = str(dest)

            if describe and describer is not None:
                if on_progress:
                    on_progress(index, total, f"Describing {video.name}")
                try:
                    description = describer.describe(video)
                except Exception as exc:  # noqa: BLE001
                    description = f"Description failed: {exc}"
                analysis.description = description

                # Sidecar next to the copy when possible; otherwise under output.
                if copied_to:
                    txt_path = Path(copied_to).with_suffix(".txt")
                else:
                    txt_path = Path(output) / f"{video.stem}.txt"
                notes = write_notes_file(analysis, txt_path, description)
                notes_path = str(notes)
                analysis.notes_path = notes_path
        else:
            rejected += 1

        entry = analysis.to_dict()
        entry["copied_to"] = copied_to
        entry["notes_path"] = notes_path
        results.append(entry)

        if on_result:
            on_result(analysis, copied_to)

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(Path(source).resolve()),
        "output": str(Path(output).resolve()),
        "dry_run": bool(dry_run),
        "describe": bool(describe),
        "cancelled": bool(should_cancel and should_cancel()),
        "config": {
            "min_flight_seconds": config.min_flight_seconds,
            "sample_fps": config.sample_fps,
            "motion_threshold": config.motion_threshold,
            "gap_seconds": config.gap_seconds,
            "model_id": (model_id or DEFAULT_MODEL_ID) if describe else None,
        },
        "summary": {
            "scanned": len(results),
            "kept": kept,
            "rejected": rejected,
        },
        "results": results,
    }

    out_report = Path(report_path) if report_path else Path(output) / "triage_report.json"
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return TriageSummary(
        scanned=len(results),
        kept=kept,
        rejected=rejected,
        report_path=out_report,
        results=results,
    )
