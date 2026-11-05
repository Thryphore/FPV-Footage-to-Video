"""Generate synthetic clips to verify triage keep/reject behavior."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def write_clip(
    path: Path,
    seconds: float,
    fps: int,
    motion: str,
) -> None:
    """motion: static | short_burst | flight"""
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 640, 360
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    frames = int(seconds * fps)

    # Base landscape strip that can scroll under a virtual camera.
    canvas_w = width * 8
    rng = np.random.default_rng(42)
    ground = np.zeros((height, canvas_w, 3), dtype=np.uint8)
    ground[:, :] = (40, 90, 40)
    for _ in range(80):
        x = int(rng.integers(0, canvas_w - 40))
        y = int(rng.integers(height // 2, height - 20))
        cv2.rectangle(ground, (x, y), (x + 40, y + 18), (20, 60, 20), -1)
    sky = np.zeros((height, canvas_w, 3), dtype=np.uint8)
    sky[:, :] = (200, 160, 80)
    scene = sky.copy()
    scene[height // 2 :, :] = ground[height // 2 :, :]

    for i in range(frames):
        t = i / fps
        if motion == "static":
            x0 = 100
        elif motion == "short_burst":
            # A couple seconds of motion, then static (failed / aborted).
            x0 = int(80 + (t * 180 if 1.0 <= t <= 3.0 else 0))
        else:  # flight
            # Fast continuous pan + bob, closer to FPV look-around rates.
            x0 = int((t * 520) % (canvas_w - width))
            y_off = int(18 * np.sin(t * 5.0))
            y0 = max(0, min(height // 4, abs(y_off)))
            frame = scene[y0 : y0 + height, x0 : x0 + width]
            if frame.shape != (height, width, 3):
                pad = np.zeros((height, width, 3), dtype=np.uint8)
                h = min(height, frame.shape[0])
                w = min(width, frame.shape[1])
                pad[:h, :w] = frame[:h, :w]
                frame = pad
            noise = rng.integers(0, 18, frame.shape, dtype=np.uint8)
            frame = cv2.add(frame, noise)
            writer.write(frame)
            continue

        frame = scene[:height, x0 : x0 + width].copy()
        noise = rng.integers(0, 8, frame.shape, dtype=np.uint8)
        frame = cv2.add(frame, noise)
        writer.write(frame)

    writer.release()


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "test_data" / "sd_card"
    clips = [
        ("static_table.mp4", 12.0, "static"),
        ("abort_takeoff.mp4", 15.0, "short_burst"),
        ("good_flight.mp4", 18.0, "flight"),
        ("short_flight.mp4", 6.0, "flight"),
    ]
    for name, seconds, motion in clips:
        out = root / name
        write_clip(out, seconds=seconds, fps=24, motion=motion)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
