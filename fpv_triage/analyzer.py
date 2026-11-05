"""Heuristic flight detection from optical-flow motion over time."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from .speed import speed_scores_from_motions


VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".mkv",
    ".avi",
    ".m4v",
    ".ts",
    ".mts",
    ".mxf",
    ".lrv",
}


@dataclass
class FlightAnalysis:
    path: str
    duration_seconds: float
    flight_seconds: float
    longest_flight_seconds: float
    sample_count: int
    mean_motion: float
    peak_motion: float
    overall_speed_score: float
    top_speed_score: float
    keep: bool
    reason: str
    description: str = ""
    notes_path: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AnalyzerConfig:
    min_flight_seconds: float = 10.0
    sample_fps: float = 2.0
    # Absolute mean optical-flow magnitude. Static bench footage is usually
    # ~0.05; sustained FPV flight is typically well above 1.0 at 2 sample-fps.
    motion_threshold: float = 0.8
    gap_seconds: float = 1.5
    analyze_width: int = 320


def _open_capture(path: Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    return cap


def _video_meta(cap: cv2.VideoCapture) -> tuple[float, float, int]:
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fps <= 1e-3:
        fps = 30.0
    duration = frame_count / fps if frame_count > 0 else 0.0
    return fps, duration, frame_count


def _resize_gray(frame: np.ndarray, width: int) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    if w <= width:
        return gray
    new_h = max(1, int(round(h * (width / w))))
    return cv2.resize(gray, (width, new_h), interpolation=cv2.INTER_AREA)


def _motion_series(
    path: Path,
    sample_fps: float,
    analyze_width: int,
) -> tuple[float, list[float], list[float]]:
    """Return duration, sample timestamps (sec), and per-step motion magnitudes."""
    cap = _open_capture(path)
    try:
        fps, duration, frame_count = _video_meta(cap)
        if duration <= 0 and frame_count <= 0:
            duration = 0.0

        step = max(1, int(round(fps / max(sample_fps, 0.1))))
        timestamps: list[float] = []
        motions: list[float] = []
        prev_gray: np.ndarray | None = None
        index = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index % step != 0:
                index += 1
                continue

            t = index / fps
            gray = _resize_gray(frame, analyze_width)
            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray,
                    gray,
                    None,
                    0.5,
                    3,
                    15,
                    3,
                    5,
                    1.2,
                    0,
                )
                mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
                motions.append(float(np.mean(mag)))
                timestamps.append(t)
            prev_gray = gray
            index += 1

        if duration <= 0:
            duration = index / fps if fps > 0 else 0.0
        return duration, timestamps, motions
    finally:
        cap.release()


def _flight_seconds_from_motion(
    timestamps: list[float],
    motions: list[float],
    motion_threshold: float,
    gap_seconds: float,
    sample_fps: float,
) -> tuple[float, float, float]:
    """Return (total_flight_seconds, longest_stretch, mean_motion)."""
    if not motions:
        return 0.0, 0.0, 0.0

    mean_motion = float(np.mean(motions))
    sample_dt = 1.0 / max(sample_fps, 0.1)
    flying = [m >= motion_threshold for m in motions]

    total = 0.0
    longest = 0.0
    stretch = 0.0
    gap_budget = 0.0

    for i, is_flying in enumerate(flying):
        dt = sample_dt
        if i > 0:
            dt = max(sample_dt * 0.5, timestamps[i] - timestamps[i - 1])

        if is_flying:
            stretch += dt + gap_budget
            gap_budget = 0.0
            total += dt
            longest = max(longest, stretch)
        elif stretch > 0:
            gap_budget += dt
            if gap_budget > gap_seconds:
                stretch = 0.0
                gap_budget = 0.0

    return total, longest, mean_motion


def _empty_analysis(path: Path, reason: str, **kwargs) -> FlightAnalysis:
    base = {
        "path": str(path),
        "duration_seconds": 0.0,
        "flight_seconds": 0.0,
        "longest_flight_seconds": 0.0,
        "sample_count": 0,
        "mean_motion": 0.0,
        "peak_motion": 0.0,
        "overall_speed_score": 1.0,
        "top_speed_score": 1.0,
        "keep": False,
        "reason": reason,
    }
    base.update(kwargs)
    return FlightAnalysis(**base)


def analyze_video(path: Path, config: AnalyzerConfig | None = None) -> FlightAnalysis:
    """Score one video for whether it contains enough real flight."""
    cfg = config or AnalyzerConfig()
    path = Path(path)

    try:
        duration, timestamps, motions = _motion_series(
            path, cfg.sample_fps, cfg.analyze_width
        )
    except Exception as exc:  # noqa: BLE001 - surface as reject reason
        return _empty_analysis(path, f"unreadable: {exc}")

    overall, top, mean_flight, peak = speed_scores_from_motions(
        motions, cfg.motion_threshold
    )
    mean_motion = float(np.mean(motions)) if motions else 0.0

    if duration < cfg.min_flight_seconds:
        return _empty_analysis(
            path,
            f"too short ({duration:.1f}s < {cfg.min_flight_seconds:.0f}s)",
            duration_seconds=round(duration, 2),
            sample_count=len(motions),
            mean_motion=round(mean_motion, 3),
            peak_motion=peak,
            overall_speed_score=overall,
            top_speed_score=top,
        )

    flight_seconds, longest, _ = _flight_seconds_from_motion(
        timestamps,
        motions,
        cfg.motion_threshold,
        cfg.gap_seconds,
        cfg.sample_fps,
    )

    keep = flight_seconds >= cfg.min_flight_seconds
    if keep:
        reason = f"keep: {flight_seconds:.1f}s flight (longest stretch {longest:.1f}s)"
    else:
        reason = (
            f"reject: only {flight_seconds:.1f}s flight "
            f"(need {cfg.min_flight_seconds:.0f}s)"
        )

    return FlightAnalysis(
        path=str(path),
        duration_seconds=round(duration, 2),
        flight_seconds=round(flight_seconds, 2),
        longest_flight_seconds=round(longest, 2),
        sample_count=len(motions),
        mean_motion=round(mean_motion, 3),
        peak_motion=peak,
        overall_speed_score=overall,
        top_speed_score=top,
        keep=keep,
        reason=reason,
    )


def extract_keyframes(
    path: Path,
    count: int = 6,
    max_width: int = 640,
) -> list:
    """Return evenly spaced RGB PIL images for the vision model."""
    from PIL import Image

    cap = _open_capture(path)
    try:
        fps, duration, frame_count = _video_meta(cap)
        if frame_count <= 0:
            frames_bgr = []
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frames_bgr.append(frame)
            frame_count = len(frames_bgr)
            if frame_count == 0:
                return []
            indexes = np.linspace(0, frame_count - 1, num=min(count, frame_count), dtype=int)
            picked = [frames_bgr[i] for i in indexes]
        else:
            indexes = np.linspace(0, frame_count - 1, num=min(count, frame_count), dtype=int)
            picked = []
            for idx in indexes:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ok, frame = cap.read()
                if ok:
                    picked.append(frame)

        images = []
        for frame in picked:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            if w > max_width:
                new_h = max(1, int(round(h * (max_width / w))))
                rgb = cv2.resize(rgb, (max_width, new_h), interpolation=cv2.INTER_AREA)
            images.append(Image.fromarray(rgb))
        return images
    finally:
        cap.release()
