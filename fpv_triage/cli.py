"""Command-line entry for FPV clip triage."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tqdm import tqdm

from .analyzer import VIDEO_EXTENSIONS, AnalyzerConfig
from .pipeline import run_triage


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fpv_triage",
        description=(
            "Scan FPV footage, keep clips with enough real flight, "
            "and copy keepers to a folder on your PC."
        ),
    )
    p.add_argument("--source", required=True, help="SD card or folder of videos")
    p.add_argument("--output", required=True, help="PC folder for keeper copies")
    p.add_argument(
        "--min-flight-seconds",
        type=float,
        default=10.0,
        help="Minimum seconds of real flight to keep (default: 10)",
    )
    p.add_argument(
        "--sample-fps",
        type=float,
        default=2.0,
        help="Analysis sample rate in frames/sec (default: 2)",
    )
    p.add_argument(
        "--motion-threshold",
        type=float,
        default=0.8,
        help="Optical-flow magnitude treated as flight (default: 0.8)",
    )
    p.add_argument(
        "--gap-seconds",
        type=float,
        default=1.5,
        help="Max quiet gap still counted inside one flight stretch",
    )
    p.add_argument(
        "--extensions",
        default=",".join(sorted(VIDEO_EXTENSIONS)),
        help="Comma-separated video extensions",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze and report without copying files",
    )
    p.add_argument(
        "--describe",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write AI description + speed scores to a .txt next to each keeper",
    )
    p.add_argument(
        "--model",
        default=None,
        help="Hugging Face vision model id (default: Qwen/Qwen3-VL-4B-Instruct)",
    )
    p.add_argument(
        "--report",
        default=None,
        help="Optional path for triage_report.json (default: <output>/triage_report.json)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source = Path(args.source)
    output = Path(args.output)
    extensions = {
        e.strip().lower() if e.strip().startswith(".") else f".{e.strip().lower()}"
        for e in args.extensions.split(",")
        if e.strip()
    }

    config = AnalyzerConfig(
        min_flight_seconds=args.min_flight_seconds,
        sample_fps=args.sample_fps,
        motion_threshold=args.motion_threshold,
        gap_seconds=args.gap_seconds,
    )

    print(f"Scanning videos from {source}")
    print(
        f"Keep rule: >= {config.min_flight_seconds:.0f}s real flight "
        f"(threshold={config.motion_threshold})"
    )
    if args.dry_run:
        print("Dry run: no files will be copied")

    bar: tqdm | None = None

    def on_progress(index: int, total: int, name: str) -> None:
        nonlocal bar
        if bar is None:
            bar = tqdm(total=total, desc="Analyzing", unit="clip")
        # Only advance on the main per-clip pass (not the "Describing …" update).
        if not name.startswith("Describing "):
            bar.n = index
            bar.total = total
            bar.refresh()
        bar.set_postfix_str(name)

    def on_result(analysis, copied_to: str | None) -> None:
        name = Path(analysis.path).name
        if analysis.keep:
            dest = Path(copied_to).name if copied_to else "(dry run)"
            msg = (
                f"KEEP  {name} -> {dest}  "
                f"(speed {analysis.overall_speed_score:.1f}/10, "
                f"top {analysis.top_speed_score:.1f}/10)"
            )
        else:
            msg = f"SKIP  {name}  ({analysis.reason})"
        if bar is not None:
            tqdm.write(msg)
        else:
            print(msg)
        if analysis.description:
            preview = analysis.description.replace("\n", " ")
            if len(preview) > 140:
                preview = preview[:137] + "..."
            line = f"      {preview}"
            if bar is not None:
                tqdm.write(line)
            else:
                print(line)

    try:
        summary = run_triage(
            source,
            output,
            config,
            dry_run=args.dry_run,
            describe=bool(args.describe),
            model_id=args.model,
            extensions=extensions,
            report_path=Path(args.report) if args.report else None,
            on_progress=on_progress,
            on_result=on_result,
        )
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    finally:
        if bar is not None:
            bar.close()

    if summary.scanned == 0:
        print(f"No videos found under {source}")
        return 1

    print()
    print(f"Done. Kept {summary.kept}/{summary.scanned}. Report: {summary.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
