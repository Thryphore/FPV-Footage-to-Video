"""Probe visual signatures for dives / power loops / flips on sample clips."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fpv_triage.analyzer import extract_keyframes, probe_duration  # noqa: E402
from fpv_triage.scene_signals import _bank_degrees, _pil_to_bgr  # noqa: E402


def ground_sky_fraction(bgr: np.ndarray) -> tuple[float, float]:
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


def flow_stats(prev_g: np.ndarray, cur_g: np.ndarray) -> tuple[float, float, float]:
    flow = cv2.calcOpticalFlowFarneback(
        prev_g, cur_g, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    fx, fy = flow[..., 0], flow[..., 1]
    return (
        float(np.mean(fx)),
        float(np.mean(fy)),
        float(np.mean(np.hypot(fx, fy))),
    )


def parse_srt_vs(path: Path) -> list[tuple[float, float, float, float]]:
    """Return list of (t_sec, height, hs, vs)."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    blocks = re.split(r"\n\s*\n", text.strip())
    out: list[tuple[float, float, float, float]] = []
    for block in blocks:
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if len(lines) < 3:
            continue
        m = re.search(
            r"(\d+):(\d+):(\d+),(\d+)\s*-->",
            lines[1],
        )
        if not m:
            continue
        t = (
            int(m.group(1)) * 3600
            + int(m.group(2)) * 60
            + int(m.group(3))
            + int(m.group(4)) / 1000.0
        )
        body = " ".join(lines[2:])
        hm = re.search(r"Height:(-?\d+(?:\.\d+)?)m", body)
        hsm = re.search(r"HS:(-?\d+(?:\.\d+)?)m/s", body)
        vsm = re.search(r"VS:(-?\d+(?:\.\d+)?)m/s", body)
        if not (hm and hsm and vsm):
            continue
        out.append((t, float(hm.group(1)), float(hsm.group(1)), float(vsm.group(1))))
    return out


def summarize_srt(path: Path) -> None:
    rows = parse_srt_vs(path)
    if not rows:
        print("SRT: no parseable rows")
        return
    vs = np.array([r[3] for r in rows], dtype=np.float64)
    hs = np.array([r[2] for r in rows], dtype=np.float64)
    h = np.array([r[1] for r in rows], dtype=np.float64)
    print(
        f"SRT samples={len(rows)} height {h.min():.1f}..{h.max():.1f} "
        f"HS max={hs.max():.1f} VS min/max={vs.min():.1f}/{vs.max():.1f}"
    )
    # Strong dive windows: VS strongly negative for a stretch.
    dive = 0
    loopish = 0
    for i in range(len(vs) - 20):
        window = vs[i : i + 20]
        if window.min() < -8 and (window < -5).sum() >= 8:
            dive += 1
        # climb then dive: positive then negative
        a, b = window[:10].mean(), window[10:].mean()
        if a > 3 and b < -3 and hs[i : i + 20].max() > 10:
            loopish += 1
    print(f"SRT dive_windows~{dive} climb-then-dive_windows~{loopish}")
    # Print extreme VS moments
    idx = int(np.argmin(vs))
    t, hh, hss, vss = rows[idx]
    print(f" deepest VS at t={t:.1f}s h={hh:.1f} HS={hss:.1f} VS={vss:.1f}")
    idx2 = int(np.argmax(vs))
    t, hh, hss, vss = rows[idx2]
    print(f" peak climb VS at t={t:.1f}s h={hh:.1f} HS={hss:.1f} VS={vss:.1f}")


def classify_moves(
    banks: np.ndarray,
    grounds: np.ndarray,
    skys: np.ndarray,
    fys: np.ndarray,
    mags: np.ndarray,
) -> dict[str, float]:
    n = len(grounds)
    dive = 0.0
    loop = 0.0
    flip = 0.0
    roll = 0.0
    for i in range(max(0, n - 5)):
        g0, g1 = grounds[i], grounds[min(n - 1, i + 4)]
        s0, s1 = skys[i], skys[min(n - 1, i + 4)]
        fy = float(fys[max(0, i) : min(len(fys), i + 4)].mean()) if len(fys) else 0.0
        mag = float(mags[max(0, i) : min(len(mags), i + 4)].mean()) if len(mags) else 0.0
        bank = float(banks[i : i + 5].max())
        # Dive: ground fills + downward pitch flow + energy.
        if (g1 - g0) >= 0.10 and fy >= 0.35 and mag >= 1.8 and s1 <= s0 + 0.05:
            dive += 1.0
        # Power loop: sky spikes then ground returns within ~6 frames, with energy.
        if i + 6 < n:
            mid = i + 3
            if (
                skys[mid] >= 0.45
                and skys[mid] - skys[i] >= 0.20
                and grounds[i + 6] - grounds[mid] >= 0.15
                and float(mags[max(0, i) : min(len(mags), i + 6)].mean() if len(mags) else 0)
                >= 1.5
            ):
                loop += 1.0
        # Flip-ish: large bank + sky/ground swap quickly.
        if bank >= 50 and abs(s1 - s0) >= 0.25 and mag >= 2.0:
            flip += 1.0
        elif bank >= 35 and mag >= 2.0 and abs(s1 - s0) < 0.25:
            roll += 1.0
    return {"dive": dive, "power_loop": loop, "flip": flip, "roll": roll}


def analyze(path: Path, label: str, n: int = 64) -> None:
    dur = probe_duration(path)
    print(f"\n=== {label} {path.name} dur={dur:.1f}s ===")
    imgs, _times = extract_keyframes(path, count=n, max_width=512)
    print(f"frames={len(imgs)}")
    banks: list[float] = []
    grounds: list[float] = []
    skys: list[float] = []
    fys: list[float] = []
    mags: list[float] = []
    prev = None
    for im in imgs:
        bgr = _pil_to_bgr(im)
        banks.append(_bank_degrees(bgr))
        g, s = ground_sky_fraction(bgr)
        grounds.append(g)
        skys.append(s)
        gray = cv2.cvtColor(cv2.resize(bgr, (160, 90)), cv2.COLOR_BGR2GRAY)
        if prev is not None:
            _mx, my, mag = flow_stats(prev, gray)
            fys.append(my)
            mags.append(mag)
        prev = gray

    banks_a = np.asarray(banks, dtype=np.float64)
    grounds_a = np.asarray(grounds, dtype=np.float64)
    skys_a = np.asarray(skys, dtype=np.float64)
    fys_a = np.asarray(fys, dtype=np.float64) if fys else np.zeros(1)
    mags_a = np.asarray(mags, dtype=np.float64) if mags else np.zeros(1)

    print(
        f"bank max={banks_a.max():.1f} p90={np.percentile(banks_a, 90):.1f} "
        f"mean={banks_a.mean():.1f}"
    )
    print(
        f"ground min/max/range="
        f"{grounds_a.min():.2f}/{grounds_a.max():.2f}/"
        f"{grounds_a.max() - grounds_a.min():.2f}"
    )
    print(
        f"sky min/max/range="
        f"{skys_a.min():.2f}/{skys_a.max():.2f}/"
        f"{skys_a.max() - skys_a.min():.2f}"
    )
    print(
        f"flow fy mean={fys_a.mean():.2f} min={fys_a.min():.2f} "
        f"max={fys_a.max():.2f} std={fys_a.std():.2f}"
    )
    print(f"flow mag mean={mags_a.mean():.2f} max={mags_a.max():.2f}")
    scores = classify_moves(banks_a, grounds_a, skys_a, fys_a, mags_a)
    print("scores", scores)


if __name__ == "__main__":
    analyze(Path(r"D:\testing\DJIG0180.mov"), "DIVE180", n=40)
    analyze(Path(r"D:\other thing drone\DJIG0168.mov"), "LOOP168", n=40)
    srt = Path(r"D:\other thing drone\DJIG0168.srt")
    if srt.exists():
        print("\n=== SRT 168 ===")
        summarize_srt(srt)
