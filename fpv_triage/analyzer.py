"""Heuristic flight detection from cheap frame-diff motion over time."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from .speed import speed_scores_from_motions


def _hidden_popen(cmd: list[str]) -> subprocess.Popen:
    """Start a process without flashing a console window on Windows."""
    kwargs: dict = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.DEVNULL,
        "stdin": subprocess.DEVNULL,
    }
    if sys.platform == "win32":
        # CREATE_NO_WINDOW = 0x08000000 — stops ffmpeg from popping a terminal.
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
    return subprocess.Popen(cmd, **kwargs)


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
    people: str = ""
    action: str = ""
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
    # Mean abs pixel change (0–255 gray, analyze_width). Bench static is usually
    # under ~2; sustained FPV flight is typically well above 6–8.
    motion_threshold: float = 6.0
    gap_seconds: float = 1.5
    analyze_width: int = 160

    @property
    def min_duration_seconds(self) -> float:
        return self.proof_flight_seconds + self.remaining_after_proof


def _open_capture(path: Path) -> cv2.VideoCapture:
    # Prefer FFMPEG; avoid MSMF quirks on some Windows DJI cards.
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    if not cap.isOpened():
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


def _frame_motion(prev_gray: np.ndarray, gray: np.ndarray) -> float:
    """Cheap motion score: mean absolute pixel difference (no optical flow)."""
    diff = cv2.absdiff(prev_gray, gray)
    return float(np.mean(diff))


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


def _ffmpeg_exe() -> str | None:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        return None


def _decide_from_motions(
    motions: list[float],
    sample_times: list[float],
    duration: float,
    config: AnalyzerConfig,
) -> tuple[bool, str, float]:
    """Walk sampled motions and return (keep, reason, flight_total)."""
    flight_total = 0.0
    stretch = 0.0
    gap_budget = 0.0
    prev_t: float | None = None

    for motion, t in zip(motions, sample_times):
        if duration > 0 and (duration - t) < config.remaining_after_proof:
            break
        dt = (t - prev_t) if prev_t is not None else (1.0 / max(config.sample_fps, 0.1))
        dt = max(dt, 0.5 / max(config.sample_fps, 0.1))
        prev_t = t

        if motion >= config.motion_threshold:
            stretch += dt + gap_budget
            gap_budget = 0.0
            flight_total += dt
            remaining = (duration - t) if duration > 0 else 0.0
            if (
                flight_total >= config.proof_flight_seconds
                and remaining >= config.remaining_after_proof
            ):
                reason = (
                    f"keep: {config.proof_flight_seconds:.0f}s flight proven "
                    f"with {remaining:.0f}s left"
                )
                return True, reason, flight_total
        elif stretch > 0:
            gap_budget += dt
            if gap_budget > config.gap_seconds:
                stretch = 0.0
                gap_budget = 0.0

    reason = (
        f"reject: never got {config.proof_flight_seconds:.0f}s flight "
        f"with {config.remaining_after_proof:.0f}s remaining"
    )
    return False, reason, flight_total


def _motion_series_ffmpeg(
    path: Path,
    config: AnalyzerConfig,
    duration: float,
    src_w: int,
    src_h: int,
) -> tuple[float, list[float], bool, str] | None:
    """
    Decode tiny gray frames via bundled ffmpeg (scale+fps in one pass).
    Much faster than OpenCV full-res decode on DJI 4K.
    """
    exe = _ffmpeg_exe()
    if not exe:
        return None

    width = max(32, int(config.analyze_width))
    if src_w > 0 and src_h > 0:
        height = max(18, int(round(src_h * (width / src_w))))
    else:
        height = max(18, int(round(width * 9 / 16)))
    height -= height % 2
    width -= width % 2
    frame_bytes = width * height
    sample_dt = 1.0 / max(config.sample_fps, 0.1)

    # Stop reading once keep is impossible: last useful timestamp is
    # duration - remaining_after_proof.
    max_t = None
    if duration > 0:
        max_t = max(0.0, duration - config.remaining_after_proof)

    vf = f"fps={config.sample_fps:.3f},scale={width}:{height}"
    cmd = [
        exe,
        "-hide_banner",
        "-loglevel",
        "error",
        "-hwaccel",
        "auto",
        "-i",
        str(path),
        "-an",
        "-sn",
        "-vf",
        vf,
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "pipe:1",
    ]

    try:
        proc = _hidden_popen(cmd)
    except OSError:
        return None

    assert proc.stdout is not None
    motions: list[float] = []
    sample_times: list[float] = []
    prev_gray: np.ndarray | None = None
    index = 0
    kept = False
    reason = ""

    try:
        while True:
            raw = proc.stdout.read(frame_bytes)
            if not raw or len(raw) < frame_bytes:
                break
            t = index * sample_dt
            if max_t is not None and t > max_t:
                break

            gray = np.frombuffer(raw, dtype=np.uint8).reshape((height, width))
            if prev_gray is not None:
                motion = _frame_motion(prev_gray, gray)
                motions.append(motion)
                sample_times.append(t)
                keep, reason, _flight = _decide_from_motions(
                    motions, sample_times, duration, config
                )
                if keep:
                    kept = True
                    break
            prev_gray = gray
            index += 1
    finally:
        try:
            proc.kill()
        except OSError:
            pass
        try:
            proc.wait(timeout=2)
        except Exception:  # noqa: BLE001
            pass

    if kept:
        return duration, motions, True, reason

    keep, reason, _ = _decide_from_motions(motions, sample_times, duration, config)
    return duration, motions, keep, reason


def _motion_series_opencv(
    path: Path,
    config: AnalyzerConfig,
) -> tuple[float, list[float], bool, str]:
    """Fallback: sequential OpenCV decode (no random seeks)."""
    cap = _open_capture(path)
    try:
        fps, duration, frame_count = _video_meta(cap)
        if duration <= 0 and frame_count <= 0:
            return 0.0, [], False, "unreadable: no frames"

        step = max(1, int(round(fps / max(config.sample_fps, 0.1))))
        sample_dt = step / fps
        motions: list[float] = []
        sample_times: list[float] = []
        prev_gray: np.ndarray | None = None
        frame_index = 0
        last_t = 0.0

        while True:
            if frame_index % step != 0:
                if not cap.grab():
                    break
                frame_index += 1
                continue

            ok, frame = cap.read()
            if not ok or frame is None:
                break

            t = frame_index / fps
            last_t = t
            if duration > 0 and (duration - t) < config.remaining_after_proof:
                break

            gray = _resize_gray(frame, config.analyze_width)
            if prev_gray is not None:
                motions.append(_frame_motion(prev_gray, gray))
                sample_times.append(t)
                keep, reason, _ = _decide_from_motions(
                    motions, sample_times, duration, config
                )
                if keep:
                    return duration, motions, True, reason

            prev_gray = gray
            frame_index += 1

        if duration <= 0 and last_t > 0:
            duration = last_t

        keep, reason, _ = _decide_from_motions(
            motions, sample_times, duration, config
        )
        return duration, motions, keep, reason
    finally:
        cap.release()


def _motion_series(
    path: Path,
    config: AnalyzerConfig,
) -> tuple[float, list[float], bool, str]:
    """
    Sample motion and decide keep/reject.

    Prefer ffmpeg downscale+fps decode (fast on DJI 4K). Fall back to
    sequential OpenCV if ffmpeg is unavailable.
    """
    src_w = src_h = 0
    duration = 0.0
    try:
        cap = _open_capture(path)
        try:
            fps, duration, frame_count = _video_meta(cap)
            src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            if duration <= 0 and frame_count <= 0:
                return 0.0, [], False, "unreadable: no frames"
        finally:
            cap.release()
    except Exception:  # noqa: BLE001
        return _motion_series_opencv(path, config)

    fast = _motion_series_ffmpeg(path, config, duration, src_w, src_h)
    if fast is not None:
        return fast
    return _motion_series_opencv(path, config)


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


def _pick_coverage_plus_motion(scores: list[float], count: int) -> list[int]:
    """
    Mostly even coverage (so people/playground aren't skipped), plus a few
    high-motion extras for rolls/flips.
    """
    n = len(scores)
    if n <= count:
        return list(range(n))

    even_n = max(3, count - max(2, count // 4))
    even = {
        int(i)
        for i in np.linspace(0, n - 1, num=min(even_n, n), dtype=int).tolist()
    }
    ranked = sorted(range(n), key=lambda i: scores[i], reverse=True)
    min_gap = max(1, n // (count * 3))
    chosen = set(even)
    for i in ranked:
        if len(chosen) >= count:
            break
        if any(abs(i - c) < min_gap for c in chosen):
            continue
        chosen.add(i)
    for i in ranked:
        if len(chosen) >= count:
            break
        chosen.add(i)
    return sorted(chosen)[:count]


def extract_keyframes(
    path: Path,
    count: int = 8,
    max_width: int = 768,
) -> list:
    """
    Return RGB PIL images for the vision model.

    Prioritizes even time coverage (people / playground), then adds a few
    high-motion frames (rolls / flips).
    """
    from PIL import Image

    path = Path(path)
    count = max(4, int(count))
    candidates = max(count * 2, 16)

    # Fast path: ffmpeg pulls evenly spaced RGB frames already scaled.
    exe = _ffmpeg_exe()
    duration = 0.0
    try:
        duration = probe_duration(path)
    except Exception:  # noqa: BLE001
        duration = 0.0

    if exe and duration > 0.5:
        width = max_width - (max_width % 2)
        height = int(round(width * 9 / 16))
        height -= height % 2
        fps = min(3.0, max(0.35, candidates / max(duration, 1.0)))
        frame_bytes = width * height * 3
        cmd = [
            exe,
            "-hide_banner",
            "-loglevel",
            "error",
            "-hwaccel",
            "auto",
            "-i",
            str(path),
            "-an",
            "-sn",
            "-vf",
            f"fps={fps:.4f},scale={width}:{height}:flags=lanczos",
            "-pix_fmt",
            "rgb24",
            "-f",
            "rawvideo",
            "pipe:1",
        ]
        try:
            proc = _hidden_popen(cmd)
        except OSError:
            proc = None
        if proc is not None and proc.stdout is not None:
            frames_rgb: list[np.ndarray] = []
            try:
                while len(frames_rgb) < candidates * 2:
                    raw = proc.stdout.read(frame_bytes)
                    if not raw or len(raw) < frame_bytes:
                        break
                    frames_rgb.append(
                        np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
                    )
            finally:
                try:
                    proc.kill()
                except OSError:
                    pass
                try:
                    proc.wait(timeout=2)
                except Exception:  # noqa: BLE001
                    pass

            if frames_rgb:
                scores = [0.0]
                for i in range(1, len(frames_rgb)):
                    a = cv2.cvtColor(frames_rgb[i - 1], cv2.COLOR_RGB2GRAY)
                    b = cv2.cvtColor(frames_rgb[i], cv2.COLOR_RGB2GRAY)
                    scores.append(float(np.mean(cv2.absdiff(a, b))))
                picks = _pick_coverage_plus_motion(
                    scores, min(count, len(frames_rgb))
                )
                return [Image.fromarray(frames_rgb[i]) for i in picks]

    # Fallback: OpenCV even spacing.
    cap = _open_capture(path)
    try:
        fps, _duration, frame_count = _video_meta(cap)
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
            indexes = np.linspace(
                0, frame_count - 1, num=min(count, frame_count), dtype=int
            )
            picked = [frames_bgr[i] for i in indexes]
        else:
            indexes = np.linspace(
                0, frame_count - 1, num=min(count, frame_count), dtype=int
            )
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
