"""Detect FPV maneuvers from keyframe sequences (dives, power loops, flips, rolls)."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .edit_signals import MoveEvent


@dataclass
class MoveSignals:
    dive_score: float = 0.0
    power_loop_score: float = 0.0
    flip_score: float = 0.0
    roll_score: float = 0.0
    max_bank_deg: float = 0.0
    sky_range: float = 0.0
    ground_range: float = 0.0
    tags: list[str] = field(default_factory=list)
    events: list[MoveEvent] = field(default_factory=list)
    hint: str = ""


def _ground_sky_fraction(bgr: np.ndarray) -> tuple[float, float]:
    small = cv2.resize(bgr, (160, 90), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)
    hue = hsv[:, :, 0].astype(np.float32)
    sky = ((val > 140) & (sat < 60)) | (
        (hue > 90) & (hue < 130) & (val > 100) & (sat > 30)
    )
    rows = np.linspace(0, 1, small.shape[0])[:, None]
    ground = (~sky) & (rows > 0.35)
    return float(ground.mean()), float(sky.mean())


def _flow_fy_mag(prev_g: np.ndarray, cur_g: np.ndarray) -> tuple[float, float]:
    flow = cv2.calcOpticalFlowFarneback(
        prev_g, cur_g, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    fy = flow[..., 1]
    mag = np.hypot(flow[..., 0], fy)
    return float(np.mean(fy)), float(np.mean(mag))


def _bank_degrees(bgr: np.ndarray) -> float:
    # Local copy so moves.py does not depend on scene_signals import cycles.
    h, w = bgr.shape[:2]
    scale = 320 / max(w, 1)
    small = cv2.resize(
        bgr, (320, max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA
    )
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 60, 160)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=40,
        minLineLength=max(40, small.shape[1] // 5),
        maxLineGap=20,
    )
    if lines is None:
        return 0.0
    angles: list[float] = []
    for item in lines:
        x1, y1, x2, y2 = [float(v) for v in np.asarray(item).reshape(-1)[:4]]
        dx = x2 - x1
        dy = y2 - y1
        length = (dx * dx + dy * dy) ** 0.5
        if length < 30:
            continue
        ang = float(np.degrees(np.arctan2(dy, dx)))
        if ang > 90:
            ang -= 180
        if ang < -90:
            ang += 180
        if abs(ang) > 70:
            continue
        angles.append(abs(ang))
    if not angles:
        return 0.0
    return float(np.percentile(np.asarray(angles, dtype=np.float64), 75))


def _frame_time(times: list[float] | None, index: int, n: int, duration: float) -> float:
    if times and 0 <= index < len(times):
        return float(times[index])
    if n <= 1 or duration <= 0:
        return 0.0
    return float(index) / float(n - 1) * float(duration)


def _merge_events(events: list[MoveEvent], gap_s: float = 1.5) -> list[MoveEvent]:
    """Merge overlapping/nearby same-name events; keep strongest confidence."""
    if not events:
        return []
    by_name: dict[str, list[MoveEvent]] = {}
    for ev in events:
        by_name.setdefault(ev.name, []).append(ev)
    merged: list[MoveEvent] = []
    for name, group in by_name.items():
        group = sorted(group, key=lambda e: e.start_s)
        cur = group[0]
        for nxt in group[1:]:
            if nxt.start_s <= cur.end_s + gap_s:
                cur = MoveEvent(
                    name=name,
                    start_s=cur.start_s,
                    end_s=max(cur.end_s, nxt.end_s),
                    confidence=max(cur.confidence, nxt.confidence),
                )
            else:
                merged.append(cur)
                cur = nxt
        merged.append(cur)
    return sorted(merged, key=lambda e: e.start_s)


def detect_moves(
    images: list,
    *,
    times: list[float] | None = None,
    duration_seconds: float = 0.0,
) -> MoveSignals:
    """
    Classify maneuvers from ordered RGB PIL/keyframes.

    Tuned on DJIG0180 (dive) and DJIG0168 (power loop): sky/ground fill +
    vertical optical flow + bank, without relying on empty DJI SRT telemetry.
    """
    out = MoveSignals()
    if len(images) < 4:
        return out

    from .scene_signals import _pil_to_bgr

    banks: list[float] = []
    grounds: list[float] = []
    skys: list[float] = []
    fys: list[float] = []
    mags: list[float] = []
    prev = None
    for image in images:
        bgr = _pil_to_bgr(image)
        banks.append(_bank_degrees(bgr))
        g, s = _ground_sky_fraction(bgr)
        grounds.append(g)
        skys.append(s)
        gray = cv2.cvtColor(cv2.resize(bgr, (160, 90)), cv2.COLOR_BGR2GRAY)
        if prev is not None:
            fy, mag = _flow_fy_mag(prev, gray)
            fys.append(fy)
            mags.append(mag)
        prev = gray

    banks_a = np.asarray(banks, dtype=np.float64)
    grounds_a = np.asarray(grounds, dtype=np.float64)
    skys_a = np.asarray(skys, dtype=np.float64)
    fys_a = np.asarray(fys, dtype=np.float64) if fys else np.zeros(1)
    mags_a = np.asarray(mags, dtype=np.float64) if mags else np.zeros(1)

    out.max_bank_deg = float(banks_a.max()) if len(banks_a) else 0.0
    out.sky_range = float(skys_a.max() - skys_a.min()) if len(skys_a) else 0.0
    out.ground_range = (
        float(grounds_a.max() - grounds_a.min()) if len(grounds_a) else 0.0
    )

    n = len(grounds_a)
    duration = float(duration_seconds)
    if duration <= 0 and times:
        duration = float(max(times))

    raw_events: list[MoveEvent] = []

    for i in range(max(0, n - 5)):
        g0 = grounds_a[i]
        g1 = grounds_a[min(n - 1, i + 4)]
        s0 = skys_a[i]
        s1 = skys_a[min(n - 1, i + 4)]
        fy = float(fys_a[max(0, i) : min(len(fys_a), i + 4)].mean())
        mag = float(mags_a[max(0, i) : min(len(mags_a), i + 4)].mean())
        bank = float(banks_a[i : i + 5].max())
        t0 = _frame_time(times, i, n, duration)
        t1 = _frame_time(times, min(n - 1, i + 4), n, duration)

        # Dive: ground fills frame while pitching down (fy>0 in image coords).
        if (g1 - g0) >= 0.10 and fy >= 0.35 and mag >= 1.8 and s1 <= s0 + 0.08:
            out.dive_score += 1.0
            conf = float(
                np.clip(
                    0.35 * ((g1 - g0) / 0.2) + 0.35 * (fy / 1.0) + 0.30 * (mag / 4.0),
                    0.4,
                    0.98,
                )
            )
            raw_events.append(
                MoveEvent(name="dive", start_s=t0, end_s=max(t0 + 0.4, t1), confidence=conf)
            )

        # Power loop: sky fills (over the top), then ground returns quickly.
        if i + 4 < n:
            mid = i + 2
            end = min(n - 1, i + 4)
            mag_w = (
                float(mags_a[max(0, i) : min(len(mags_a), end + 1)].mean())
                if len(mags_a)
                else 0.0
            )
            sky_jump = float(skys_a[mid] - skys_a[i])
            ground_back = float(grounds_a[end] - grounds_a[mid])
            if (
                skys_a[mid] >= 0.55
                and sky_jump >= 0.30
                and grounds_a[mid] <= 0.35
                and ground_back >= 0.20
                and mag_w >= 1.4
            ):
                out.power_loop_score += 1.0
                t_end = _frame_time(times, end, n, duration)
                conf = float(
                    np.clip(
                        0.4 * (skys_a[mid] / 0.7)
                        + 0.3 * (sky_jump / 0.4)
                        + 0.3 * (ground_back / 0.3),
                        0.45,
                        0.98,
                    )
                )
                raw_events.append(
                    MoveEvent(
                        name="power loop",
                        start_s=t0,
                        end_s=max(t0 + 0.5, t_end),
                        confidence=conf,
                    )
                )

        # Flip: extreme bank + rapid sky/ground swap.
        if bank >= 50 and abs(s1 - s0) >= 0.25 and mag >= 2.0:
            out.flip_score += 1.0
            conf = float(np.clip(bank / 90.0, 0.5, 0.95))
            raw_events.append(
                MoveEvent(name="flips", start_s=t0, end_s=max(t0 + 0.3, t1), confidence=conf)
            )
        elif bank >= 35 and mag >= 2.0 and abs(s1 - s0) < 0.25:
            out.roll_score += 1.0
            conf = float(np.clip(bank / 80.0, 0.4, 0.9))
            raw_events.append(
                MoveEvent(name="rolls", start_s=t0, end_s=max(t0 + 0.3, t1), confidence=conf)
            )

    tags: list[str] = []
    has_loop = out.power_loop_score >= 1.0
    has_dive = out.dive_score >= 1.0
    if has_loop and has_dive and out.power_loop_score < 2.0 and out.sky_range < 0.50:
        has_loop = False
    if has_loop:
        tags.append("power loop")
    if has_dive:
        tags.append("dive")
    if out.flip_score >= 2.0 or (out.flip_score >= 1.0 and out.max_bank_deg >= 55):
        tags.append("flips")
    if out.roll_score >= 8.0 or (
        out.roll_score >= 4.0 and out.max_bank_deg >= 40 and "flips" not in tags
    ):
        tags.append("rolls")
    if not tags and out.max_bank_deg >= 22:
        tags.append("banking / aggressive turns")

    keep_names = set(tags)
    filtered = [e for e in raw_events if e.name in keep_names]
    # If we only have generic banking, don't emit timed roll spam.
    if tags == ["banking / aggressive turns"]:
        filtered = []
    out.events = _merge_events(filtered)
    out.tags = tags
    out.hint = ", ".join(tags)
    return out
