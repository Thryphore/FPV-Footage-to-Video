"""Edit-oriented signals: coolest window, pace tags, crash ending."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass
class TimeWindow:
    start_s: float
    end_s: float
    score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "start_s": round(self.start_s, 2),
            "end_s": round(self.end_s, 2),
            "score": round(self.score, 3),
        }

    def label(self) -> str:
        return f"{self.start_s:.1f}-{self.end_s:.1f}s"


@dataclass
class MoveEvent:
    name: str
    start_s: float
    end_s: float
    confidence: float

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "start_s": round(self.start_s, 2),
            "end_s": round(self.end_s, 2),
            "confidence": round(self.confidence, 3),
        }

    def label(self) -> str:
        return f"{self.name} @ {self.start_s:.1f}-{self.end_s:.1f}s ({self.confidence:.2f})"


@dataclass
class EditSignals:
    coolest_window: TimeWindow | None = None
    pace_tags: list[str] = field(default_factory=list)
    move_events: list[MoveEvent] = field(default_factory=list)
    ends_in_crash: bool = False
    crash_confidence: float = 0.0

    def to_dict(self) -> dict:
        return {
            "coolest_window": (
                self.coolest_window.to_dict() if self.coolest_window else None
            ),
            "pace_tags": list(self.pace_tags),
            "move_events": [e.to_dict() for e in self.move_events],
            "ends_in_crash": bool(self.ends_in_crash),
            "crash_confidence": round(self.crash_confidence, 3),
        }


def coolest_window(
    motions: list[float],
    times: list[float],
    duration: float,
    fraction: float = 0.10,
) -> TimeWindow | None:
    """
    Find the contiguous window covering ~fraction of the clip with the
    highest mean motion (the 'coolest' stretch for cutting).
    """
    if not motions or not times or duration <= 0.5:
        return None
    n = min(len(motions), len(times))
    if n < 4:
        return None

    arr = np.asarray(motions[:n], dtype=np.float64)
    t = np.asarray(times[:n], dtype=np.float64)
    target = max(0.5, float(duration) * max(0.05, min(fraction, 0.5)))

    best_score = -1.0
    best_i = 0
    best_j = n - 1
    j = 0
    for i in range(n):
        if j < i:
            j = i
        while j + 1 < n and (t[j] - t[i]) < target:
            j += 1
        # Expand one more sample when possible so window reaches ~target.
        while j + 1 < n and (t[j + 1] - t[i]) <= target * 1.15:
            j += 1
        span = float(t[j] - t[i])
        if span < target * 0.6 and j < n - 1:
            continue
        score = float(arr[i : j + 1].mean())
        if score > best_score:
            best_score = score
            best_i, best_j = i, j

    start = float(t[best_i])
    end = float(t[best_j])
    if end <= start:
        end = min(duration, start + target)
    return TimeWindow(start_s=start, end_s=min(duration, end), score=best_score)


def ends_in_crash(
    motions: list[float],
    times: list[float],
    duration: float,
    motion_threshold: float = 6.0,
) -> tuple[bool, float]:
    """
    Heuristic: late high-energy spike, then near-static tail (impact / tumble stop).
    Returns (flag, confidence 0..1).
    """
    if not motions or not times or duration < 6.0:
        return False, 0.0
    n = min(len(motions), len(times))
    if n < 10:
        return False, 0.0

    arr = np.asarray(motions[:n], dtype=np.float64)
    t = np.asarray(times[:n], dtype=np.float64)
    cutoff = max(0.0, duration - max(3.0, duration * 0.18))
    early = arr[t < cutoff]
    late_idx = np.where(t >= cutoff)[0]
    if early.size < 4 or late_idx.size < 4:
        return False, 0.0

    late = arr[late_idx]
    late_peak = float(late.max())
    tail_n = max(2, late.size // 3)
    late_tail = float(late[-tail_n:].mean())
    early_mean = float(early.mean())
    early_p90 = float(np.percentile(early, 90))

    # Peak near the end that exceeds typical flight, then dies off.
    peak_ok = late_peak >= max(motion_threshold * 1.8, early_p90 * 1.15)
    tail_dead = late_tail <= max(1.5, motion_threshold * 0.55)
    was_flying = early_mean >= motion_threshold * 0.75
    if not (peak_ok and tail_dead and was_flying):
        return False, 0.0

    # Confidence from how hard the peak was and how dead the tail is.
    peak_term = min(1.0, late_peak / max(early_p90, 1.0) / 2.0)
    dead_term = min(1.0, (motion_threshold - late_tail) / max(motion_threshold, 1.0))
    conf = float(np.clip(0.45 * peak_term + 0.55 * dead_term, 0.0, 1.0))
    return conf >= 0.45, conf


def pace_tags(
    *,
    overall_speed: float,
    top_speed: float,
    action: str,
    people: str,
    move_names: list[str] | None = None,
    ends_crash: bool = False,
) -> list[str]:
    """Music/edit energy tags for later assembly."""
    action_l = (action or "").lower()
    people_l = (people or "").lower()
    moves = {m.lower() for m in (move_names or [])}
    tags: list[str] = []

    hype_moves = {"dive", "power loop", "flips", "rolls"}
    is_hype = (
        top_speed >= 7.5
        or overall_speed >= 7.0
        or bool(moves & hype_moves)
        or any(k in action_l for k in ("dive", "power loop", "flip", "roll"))
    )
    is_slow = (
        overall_speed <= 4.5
        and top_speed <= 6.0
        and not is_hype
        and "proximity" not in action_l
    )
    if is_hype:
        tags.append("hype")
    if is_slow:
        tags.append("slow cruise")
    if "proximity" in action_l or "under-" in action_l or "under " in action_l:
        tags.append("proximity")
    if people_l not in {"", "none", "n/a", "unknown"}:
        tags.append("people")
    if ends_crash:
        tags.append("ends in crash")
    return tags
