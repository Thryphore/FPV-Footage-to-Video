"""
FPV text overlay: a readable 3D-looking sign the camera flies through.

Design (kept simple on purpose after more complex attempts failed):
- One phrase drawn as a single textured quad (not exploding letter soup).
- Depth is a smooth, reliable approach curve so every pass looks intentional.
- Optical flow only drives lateral/roll parallax so the sign drifts with the
  scene instead of sitting like a HUD sticker.
- ROI perspective warp — fast enough for 4K.
"""

from __future__ import annotations

import math
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .analyzer import (
    VIDEO_EXTENSIONS,
    AnalyzerConfig,
    _ffmpeg_exe,
    _open_capture,
    _video_meta,
    find_flight_proof_time,
)


PLACEHOLDER_TEXT = "This is text right here"
PLACEHOLDER_REPEAT = 15
DEFAULT_PASS_DURATION_S = 2.2
FLOW_WIDTH = 120
FLOW_EVERY_N = 2


@dataclass
class RenderResult:
    output_path: Path
    source_path: Path
    start_s: float
    duration_s: float
    frame_count: int
    cancelled: bool = False


def placeholder_block(
    text: str = PLACEHOLDER_TEXT,
    repeat: int = PLACEHOLDER_REPEAT,
) -> str:
    line = (text or PLACEHOLDER_TEXT).strip() or PLACEHOLDER_TEXT
    n = max(1, int(repeat))
    return f"{line}\n\n({n} fly-throughs)"


def list_keeper_videos(folder: Path) -> list[Path]:
    folder = Path(folder)
    if not folder.is_dir():
        return []
    files = [
        p
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    ]
    return sorted(files, key=lambda p: p.name.lower())


def _load_font(size: int) -> ImageFont.ImageFont:
    for path in (
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeuib.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
    ):
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _make_sign_bgra(text: str, *, base_width: int) -> np.ndarray:
    """One clear sign texture (BGRA) — readable over bright FPV sky."""
    line = " ".join((text or PLACEHOLDER_TEXT).split()) or PLACEHOLDER_TEXT
    target_w = max(700, min(1600, int(base_width * 0.72)))
    lo, hi = 40, 160
    best = _load_font(72)
    pad_x, pad_y = 44, 32
    for _ in range(12):
        mid = (lo + hi) // 2
        font = _load_font(mid)
        dummy = Image.new("RGBA", (8, 8))
        draw = ImageDraw.Draw(dummy)
        bbox = draw.textbbox((0, 0), line, font=font)
        if (bbox[2] - bbox[0] + pad_x * 2) <= target_w:
            best = font
            lo = mid + 1
        else:
            hi = mid - 1

    dummy = Image.new("RGBA", (8, 8))
    draw = ImageDraw.Draw(dummy)
    bbox = draw.textbbox((0, 0), line, font=best)
    tw = max(1, int(round(bbox[2] - bbox[0] + pad_x * 2)))
    th = max(1, int(round(bbox[3] - bbox[1] + pad_y * 2)))
    img = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    # Soft plate for contrast without looking like a UI widget.
    draw.rounded_rectangle(
        (4, 4, tw - 5, th - 5),
        radius=max(10, th // 5),
        fill=(6, 10, 18, 165),
        outline=(180, 220, 255, 210),
        width=max(2, th // 30),
    )
    xy = (float(pad_x - bbox[0]), float(pad_y - bbox[1]))
    for dx, dy in ((-4, 0), (4, 0), (0, -4), (0, 4), (-3, -3), (3, 3), (-3, 3), (3, -3)):
        draw.text((xy[0] + dx, xy[1] + dy), line, font=best, fill=(0, 0, 0, 255))
    draw.text(xy, line, font=best, fill=(255, 255, 255, 255))
    rgba = np.asarray(img.convert("RGBA"))
    return rgba[:, :, [2, 1, 0, 3]].copy()


def _flow_delta(prev: np.ndarray, cur: np.ndarray) -> tuple[float, float, float]:
    """Mean translation + roll proxy on a tiny gray pair."""
    flow = cv2.calcOpticalFlowFarneback(
        prev, cur, None, 0.5, 2, 9, 2, 5, 1.1, 0
    )
    fx, fy = flow[..., 0], flow[..., 1]
    dx = float(np.mean(fx))
    dy = float(np.mean(fy))
    h, w = prev.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = (w - 1) * 0.5, (h - 1) * 0.5
    rx, ry = xx - cx, yy - cy
    roll = float(np.mean((-fy * rx + fx * ry) / (rx * rx + ry * ry + 1e-3)))
    return dx, dy, roll


def _depth(u: float) -> float:
    """
    Smooth approach: hang at a readable distance, then commit through.
    Returns camera-relative depth (larger = farther).
    """
    u = float(np.clip(u, 0.0, 1.0))
    z_far, z_mid, z_near = 2.4, 1.05, 0.08
    if u < 0.55:
        t = u / 0.55
        t = t * t * (3.0 - 2.0 * t)
        return z_far * (1.0 - t) + z_mid * t
    t = (u - 0.55) / 0.45
    t = t * t
    return z_mid * (1.0 - t) + z_near * t


def _alpha(u: float, z: float) -> float:
    u = float(np.clip(u, 0.0, 1.0))
    if u < 0.08:
        return 0.4 + 0.6 * (u / 0.08)
    if z < 0.22:
        return float(np.clip((z - 0.06) / 0.16, 0.0, 1.0))
    if u > 0.92:
        return float(np.clip((1.0 - u) / 0.08, 0.0, 1.0))
    return 1.0


def _project_sign_quad(
    *,
    z: float,
    yaw: float,
    shift_x: float,
    shift_y: float,
    roll: float,
    frame_w: int,
    frame_h: int,
    half_w: float = 0.92,
    half_h: float = 0.20,
) -> np.ndarray | None:
    """Yawed vertical plane in front of the camera → screen quad."""
    z = max(0.06, float(z))
    f = float(frame_h) * 1.25
    cx = frame_w * 0.5 + shift_x
    cy = frame_h * 0.44 + shift_y
    cyaw, syaw = math.cos(yaw), math.sin(yaw)
    croll, sroll = math.cos(-roll), math.sin(-roll)
    pts = np.empty((4, 2), dtype=np.float32)
    for i, (u, v) in enumerate(
        ((-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h))
    ):
        x = u * cyaw
        y = v
        zz = z + u * syaw
        if zz <= 0.05:
            return None
        # Apply camera roll in view space.
        xr = x * croll - y * sroll
        yr = x * sroll + y * croll
        pts[i, 0] = cx + f * xr / zz
        pts[i, 1] = cy + f * yr / zz
    return pts


def _warp_roi(
    frame: np.ndarray, bgra: np.ndarray, quad: np.ndarray, alpha_scale: float
) -> None:
    if alpha_scale <= 0.02 or not np.isfinite(quad).all():
        return
    h, w = frame.shape[:2]
    xs, ys = quad[:, 0], quad[:, 1]
    x0 = max(0, int(math.floor(float(xs.min()) - 1)))
    y0 = max(0, int(math.floor(float(ys.min()) - 1)))
    x1 = min(w, int(math.ceil(float(xs.max()) + 1)))
    y1 = min(h, int(math.ceil(float(ys.max()) + 1)))
    rw, rh = x1 - x0, y1 - y0
    if rw < 2 or rh < 2:
        return
    # Through-phase quads often fill the frame — still warp (one ROI is fine).
    local = quad.copy()
    local[:, 0] -= x0
    local[:, 1] -= y0
    th, tw = bgra.shape[:2]
    src = np.asarray(
        [[0.0, 0.0], [tw - 1.0, 0.0], [tw - 1.0, th - 1.0], [0.0, th - 1.0]],
        dtype=np.float32,
    )
    try:
        mat = cv2.getPerspectiveTransform(src, local)
    except cv2.error:
        return
    warped = cv2.warpPerspective(
        bgra,
        mat,
        (rw, rh),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    a = warped[:, :, 3:4].astype(np.float32) * (alpha_scale / 255.0)
    if float(a.max()) < 1e-3:
        return
    bgr = warped[:, :, :3].astype(np.float32)
    roi = frame[y0:y1, x0:x1].astype(np.float32)
    frame[y0:y1, x0:x1] = np.clip(bgr * a + roi * (1.0 - a), 0, 255).astype(np.uint8)


def _open_ffmpeg_writer(
    exe: str, out_path: Path, width: int, height: int, fps: float
) -> subprocess.Popen:
    cmd = [
        exe,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps:.4f}",
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-threads",
        "0",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(out_path),
    ]
    kwargs: dict = {
        "stdin": subprocess.PIPE,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.PIPE,
        "bufsize": 10**7,
    }
    if sys.platform == "win32":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
    return subprocess.Popen(cmd, **kwargs)


def render_single_clip(
    source: Path,
    output_dir: Path,
    *,
    text: str | None = None,
    text_repeat: int = PLACEHOLDER_REPEAT,
    pass_duration_s: float = DEFAULT_PASS_DURATION_S,
    text_duration_s: float | None = None,
    config: AnalyzerConfig | None = None,
    should_cancel: Callable[[], bool] | None = None,
    on_progress: Callable[[float, str], None] | None = None,
) -> RenderResult:
    """
    After 5s of flight, fly through the phrase N times.

    Approach is a smooth curve (reliable). Flow adds parallax so the sign
    moves with the footage instead of sticking to screen center.
    """
    source = Path(source)
    output_dir = Path(output_dir)
    cfg = config or AnalyzerConfig()
    cancel = should_cancel or (lambda: False)
    progress = on_progress or (lambda _f, _m: None)
    n_passes = max(1, int(text_repeat))
    phrase = (text or PLACEHOLDER_TEXT).strip() or PLACEHOLDER_TEXT

    if not source.is_file():
        raise FileNotFoundError(f"Video not found: {source}")

    progress(0.02, "Detecting flight start…")
    start_s, _duration, reason = find_flight_proof_time(source, cfg)
    if start_s is None:
        raise RuntimeError(f"Cannot start render: {reason}")

    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"{source.stem}_text_{stamp}.mp4"

    progress(0.05, f"Flight proof at {start_s:.1f}s — rendering…")
    cap = _open_capture(source)
    fps = 30.0
    written = 0
    try:
        fps, _src_duration, _frame_count = _video_meta(cap)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if width <= 0 or height <= 0:
            raise RuntimeError("Could not read video dimensions")
        if fps <= 1e-3:
            fps = 30.0

        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, start_s) * 1000.0)
        sign = _make_sign_bgra(phrase, base_width=min(width, 1920))

        flow_w = FLOW_WIDTH
        flow_h = max(16, int(round(height * (flow_w / max(width, 1)))))
        flow_h -= flow_h % 2
        flow_w -= flow_w % 2

        exe = _ffmpeg_exe()
        if not exe:
            raise RuntimeError("ffmpeg not available (imageio-ffmpeg)")
        writer = _open_ffmpeg_writer(exe, out_path, width, height, fps)
        assert writer.stdin is not None

        if text_duration_s is not None:
            max_frames = max(1, int(round(float(text_duration_s) * fps)))
            frames_per_pass = max(1, max_frames // n_passes)
        else:
            frames_per_pass = max(10, int(round(float(pass_duration_s) * fps)))
            max_frames = frames_per_pass * n_passes

        # Parallax state (pixels) — follows the scene, reset lightly each pass.
        pan_x = 0.0
        pan_y = 0.0
        roll = 0.0
        vel_x = vel_y = vel_roll = 0.0
        prev_gray: np.ndarray | None = None

        try:
            while written < max_frames:
                if cancel():
                    break
                ok, frame = cap.read()
                if not ok or frame is None:
                    break

                pass_idx = written // frames_per_pass
                if pass_idx >= n_passes:
                    break
                frame_in_pass = written - pass_idx * frames_per_pass
                if frame_in_pass == 0:
                    pan_x *= 0.2
                    pan_y *= 0.2
                    roll *= 0.3
                    vel_x = vel_y = vel_roll = 0.0

                if written % FLOW_EVERY_N == 0:
                    small = cv2.resize(
                        frame, (flow_w, flow_h), interpolation=cv2.INTER_AREA
                    )
                    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                    if prev_gray is not None:
                        dx, dy, d_roll = _flow_delta(prev_gray, gray)
                        # Match scene motion: if background slides left, so does the sign.
                        scale = width / max(flow_w, 1)
                        vel_x = dx * scale * 0.85
                        vel_y = dy * scale * 0.85
                        vel_roll = d_roll * 0.5
                    prev_gray = gray

                pan_x = float(np.clip(pan_x + vel_x, -width * 0.22, width * 0.22))
                pan_y = float(np.clip(pan_y + vel_y, -height * 0.16, height * 0.16))
                roll = float(np.clip(roll + vel_roll, -0.4, 0.4))

                u = frame_in_pass / max(frames_per_pass - 1, 1)
                z = _depth(u)
                a = _alpha(u, z)
                # Alternate yaw so successive signs don't look identical.
                yaw = 0.38 * math.sin(pass_idx * 1.7 + 0.4)

                # Near the end, let the plane fill / rush past.
                near = float(np.clip((0.9 - z) / 0.9, 0.0, 1.0))
                half_w = 0.88 + 0.35 * near
                half_h = 0.19 + 0.08 * near

                quad = _project_sign_quad(
                    z=z,
                    yaw=yaw,
                    shift_x=pan_x,
                    shift_y=pan_y,
                    roll=roll,
                    frame_w=width,
                    frame_h=height,
                    half_w=half_w,
                    half_h=half_h,
                )
                if quad is not None and a > 0.03:
                    _warp_roi(frame, sign, quad, a)

                writer.stdin.write(frame.tobytes())
                written += 1
                if written % max(1, int(fps) or 1) == 0:
                    progress(
                        0.05 + 0.9 * (written / max_frames),
                        f"Pass {pass_idx + 1}/{n_passes} — {written}/{max_frames}",
                    )
        finally:
            try:
                writer.stdin.close()
            except Exception:  # noqa: BLE001
                pass
            err = b""
            try:
                err = writer.stderr.read() if writer.stderr else b""
            except Exception:  # noqa: BLE001
                pass
            ret = writer.wait(timeout=120)
            if ret != 0 and not cancel() and written > 0:
                detail = err.decode("utf-8", errors="replace").strip()
                raise RuntimeError(
                    f"ffmpeg encode failed (code {ret})"
                    + (f": {detail}" if detail else "")
                )
    finally:
        cap.release()

    if cancel():
        if out_path.exists() and written == 0:
            try:
                out_path.unlink()
            except OSError:
                pass
        progress(1.0, "Cancelled.")
        return RenderResult(
            output_path=out_path,
            source_path=source,
            start_s=float(start_s),
            duration_s=written / max(fps, 1e-3),
            frame_count=written,
            cancelled=True,
        )

    if written == 0:
        raise RuntimeError("No frames decoded after flight start")

    progress(1.0, f"Wrote {out_path.name}")
    return RenderResult(
        output_path=out_path,
        source_path=source,
        start_s=float(start_s),
        duration_s=written / max(fps, 1e-3),
        frame_count=written,
        cancelled=False,
    )


__all__ = [
    "PLACEHOLDER_TEXT",
    "PLACEHOLDER_REPEAT",
    "DEFAULT_PASS_DURATION_S",
    "RenderResult",
    "placeholder_block",
    "list_keeper_videos",
    "render_single_clip",
]
