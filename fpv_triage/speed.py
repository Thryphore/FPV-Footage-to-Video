"""Map optical-flow motion to overall / top speed scores (1-10)."""

from __future__ import annotations

import math

import numpy as np


def motion_to_score(motion: float) -> float:
    """
    Convert mean abs frame-diff (0–255 gray @ ~160px, 2 sample-fps) to 1–10.

    Rough guide for FPV:
      ~2    almost still / hover
      ~8    slow cruise / park flying
      ~18   normal freestyle pace
      ~35+  ripping / proximity
    """
    m = max(0.0, float(motion))
    score = 1.0 + 9.0 * (1.0 - math.exp(-m / 18.0))
    return round(min(10.0, max(1.0, score)), 1)


def speed_scores_from_motions(
    motions: list[float],
    motion_threshold: float,
) -> tuple[float, float, float, float]:
    """
    Return (overall_score, top_score, mean_flight_motion, peak_motion).

    Overall uses mean motion during flight-like samples when available.
    Top uses the 95th percentile so one noisy frame cannot dominate.
    """
    if not motions:
        return 1.0, 1.0, 0.0, 0.0

    arr = np.asarray(motions, dtype=np.float64)
    peak = float(np.percentile(arr, 95))
    flight = arr[arr >= motion_threshold]
    mean_flight = float(np.mean(flight)) if flight.size else float(np.mean(arr))

    return (
        motion_to_score(mean_flight),
        motion_to_score(peak),
        round(mean_flight, 3),
        round(peak, 3),
    )
