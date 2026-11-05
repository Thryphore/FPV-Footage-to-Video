"""Map optical-flow motion to overall / top speed scores (1-10)."""

from __future__ import annotations

import math

import numpy as np


def motion_to_score(motion: float) -> float:
    """
    Convert mean Farneback magnitude (at ~2 sample-fps, 320px wide) to 1-10.

    Rough guide for FPV:
      ~0.1  sitting / almost still
      ~0.8  takeoff / slow cruise
      ~2.0  normal freestyle pace
      ~4.0+ ripping / high-speed pass
    """
    m = max(0.0, float(motion))
    # Soft curve so mid speeds spread out and extremes compress toward 10.
    score = 1.0 + 9.0 * (1.0 - math.exp(-m / 2.2))
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
