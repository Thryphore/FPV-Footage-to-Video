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
    sample_count: int
    mean_motion: float
    peak_motion: float
    overall_speed_score: float
    top_speed_score: float
    keep: bool
    reason: str
    description: str = ""
    environment_tag: str = ""
    notes_path: str | None = None
    # Step timings in seconds — for app/CLI display only, not the notes file.
    timing: dict[str, float] | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AnalyzerConfig:
    # Keep once we have this much flight AND this much video left after that point.
    proof_flight_seconds: float = 5.0
    remaining_after_proof: float = 15.0
    sample_fps: float = 2.0
    # Absolute mean optical-flow magnitude. Static bench footage is usually
    # ~0.05; sustained FPV flight is typically well above 1.0 at 2 sample-fps.
    motion_threshold: float = 0.8
    gap_seconds: float = 1.5
    analyze_width: int = 320

    @property
    def min_duration_seconds(self) -> float:
        return self.proof_flight_seconds + self.remaining_after_proof


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


def probe_duration(path: Path) -> float:
    """Read duration from container metadata without decoding frames."""
    cap = _open_capture(path)
    try:
        _fps, duration, _frame_count = _video_meta(cap)
        if duration > 0:
            return duration
        msec = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
        return max(0.0, msec / 1000.0) if msec > 0 else 0.0
    finally:
        cap.release()


def _read_frame_at(cap: cv2.VideoCapture, frame_index: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_index))
    ok, frame = cap.read()
    return frame if ok else None


def _motion_series(
    path: Path,
    config: AnalyzerConfig,
) -> tuple[float, list[float], bool, str]:
    """
    Sample motion and decide keep/reject.

    Keep as soon as proof_flight_seconds of flight is reached AND
    remaining video after that timestamp is >= remaining_after_proof.
    """
    cap = _open_capture(path)
    try:
        fps, duration, frame_count = _video_meta(cap)
        if duration <= 0 and frame_count <= 0:
            return 0.0, [], False, "unreadable: no frames"

        step = max(1, int(round(fps / max(config.sample_fps, 0.1))))
        sample_dt = 1.0 / max(config.sample_fps, 0.1)
        motions: list[float] = []
        prev_gray: np.ndarray | None = None
        prev_t: float | None = None

        flight_total = 0.0
        stretch = 0.0
        gap_budget = 0.0

        if frame_count > 0:
            sample_indexes = list(range(0, frame_count, step))
        else:
            sample_indexes = []
            index = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if index % step == 0:
                    sample_indexes.append(index)
                index += 1
            if duration <= 0 and index > 0:
                duration = index / fps
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        for frame_index in sample_indexes:
            frame = _read_frame_at(cap, frame_index)
            if frame is None:
                continue
            t = frame_index / fps
            gray = _resize_gray(frame, config.analyze_width)
            if prev_gray is not None and prev_t is not None:
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
                motion = float(np.mean(mag))
                motions.append(motion)

                dt = max(sample_dt * 0.5, t - prev_t)
                if motion >= config.motion_threshold:
                    stretch += dt + gap_budget
                    gap_budget = 0.0
                    flight_total += dt
                    remaining = duration - t
                    if (
                        flight_total >= config.proof_flight_seconds
                        and remaining >= config.remaining_after_proof
                    ):
                        reason = (
                            f"keep: {config.proof_flight_seconds:.0f}s flight proven "
                            f"with {remaining:.0f}s left"
                        )
                        return duration, motions, True, reason
                elif stretch > 0:
                    gap_budget += dt
                    if gap_budget > config.gap_seconds:
                        stretch = 0.0
                        gap_budget = 0.0

            prev_gray = gray
            prev_t = t

        if duration <= 0 and prev_t is not None:
            duration = prev_t

        reason = (
            f"reject: never got {config.proof_flight_seconds:.0f}s flight "
            f"with {config.remaining_after_proof:.0f}s remaining"
        )
        return duration, motions, False, reason
    finally:
        cap.release()


def _empty_analysis(path: Path, reason: str, **kwargs) -> FlightAnalysis:
    base = {
        "path": str(path),
        "duration_seconds": 0.0,
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
        duration = probe_duration(path)
    except Exception as exc:  # noqa: BLE001
        return _empty_analysis(path, f"unreadable: {exc}")

    # Instant reject when the file cannot possibly satisfy proof + remaining.
    if 0 < duration < cfg.min_duration_seconds:
        return _empty_analysis(
            path,
            (
                f"too short ({duration:.1f}s < "
                f"{cfg.min_duration_seconds:.0f}s needed)"
            ),
            duration_seconds=round(duration, 2),
        )

    try:
        duration, motions, keep, reason = _motion_series(path, cfg)
    except Exception as exc:  # noqa: BLE001
        return _empty_analysis(path, f"unreadable: {exc}")

    overall, top, _mean_flight, peak = speed_scores_from_motions(
        motions, cfg.motion_threshold
    )
    mean_motion = float(np.mean(motions)) if motions else 0.0

    return FlightAnalysis(
        path=str(path),
        duration_seconds=round(duration, 2),
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
