"""Deterministic helpers that catch what a small VL model often misses."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


# COCO ids that matter for FPV proximity / landmarks.
_PROP_CLASSES = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    13: "bench",
    25: "umbrella",  # YOLO often labels bench/canopy overhangs as umbrella
    56: "chair",
    57: "couch",
    58: "potted plant",
    60: "table",
}

_OVERHANG_PROPS = {"bench", "table", "chair", "couch", "umbrella"}
_VEHICLE_PROPS = {"car", "truck", "bus", "motorcycle", "bicycle"}


@dataclass
class SceneSignals:
    person_hits: int = 0
    max_people_in_frame: int = 0
    frames_with_people: int = 0
    max_bank_deg: float = 0.0
    banked_frame_count: int = 0
    prop_hits: dict[str, int] | None = None
    under_object_frames: int = 0
    close_prop_frames: int = 0
    tree_frames: int = 0
    max_foliage_score: float = 0.0
    people_hint: str = "none"
    action_hint: str = ""
    props_hint: str = "none"
    landmarks_hint: str = "none"


_yolo = None


def _get_yolo():
    global _yolo
    if _yolo is not None:
        return _yolo
    try:
        from ultralytics import YOLO

        _yolo = YOLO("yolov8n.pt")
    except Exception:  # noqa: BLE001
        _yolo = False
    return _yolo


def _pil_to_bgr(image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _predict_boxes(bgr: np.ndarray, model, conf: float, imgsz: int, classes: list[int] | None):
    results = model.predict(
        bgr, verbose=False, conf=conf, classes=classes, imgsz=imgsz
    )
    if not results or results[0].boxes is None:
        return []
    boxes = results[0].boxes
    out = []
    xyxy = boxes.xyxy.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)
    for box, c in zip(xyxy, cls):
        out.append((c, box))
    return out


def _count_people(bgr: np.ndarray, model) -> int:
    """
    Count people, including small distant ones.

    Full-frame pass plus 2x2 tiles (helps playground/background kids).
    """
    total = len(_predict_boxes(bgr, model, conf=0.15, imgsz=960, classes=[0]))
    h, w = bgr.shape[:2]
    tiles = (
        bgr[0 : h // 2 + h // 10, 0 : w // 2 + w // 10],
        bgr[0 : h // 2 + h // 10, w // 2 - w // 10 : w],
        bgr[h // 2 - h // 10 : h, 0 : w // 2 + w // 10],
        bgr[h // 2 - h // 10 : h, w // 2 - w // 10 : w],
    )
    for tile in tiles:
        if tile.size == 0:
            continue
        total = max(
            total,
            len(_predict_boxes(tile, model, conf=0.12, imgsz=640, classes=[0])),
        )
    return total


def _detect_props(bgr: np.ndarray, model) -> list[tuple[str, float, tuple[float, float, float, float]]]:
    """Return (label, area_fraction, xyxy) for furniture-like props."""
    h, w = bgr.shape[:2]
    frame_area = float(max(h * w, 1))
    classes = list(_PROP_CLASSES.keys())
    found: list[tuple[str, float, tuple[float, float, float, float]]] = []
    for c, box in _predict_boxes(bgr, model, conf=0.12, imgsz=1280, classes=classes):
        label = _PROP_CLASSES.get(int(c))
        if not label:
            continue
        x1, y1, x2, y2 = map(float, box)
        area = max(0.0, (x2 - x1) * (y2 - y1)) / frame_area
        found.append((label, area, (x1, y1, x2, y2)))
    return found


def _foliage_score(bgr: np.ndarray) -> float:
    """
    Rough tree/bush score from green foliage in the mid/upper frame.

    Grass-only lawns score lower because foliage is concentrated lower and
    smoother; trees/bushes add textured green higher in the frame.
    """
    h, w = bgr.shape[:2]
    if h < 20 or w < 20:
        return 0.0
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # Green / yellow-green leaves (exclude very dark / gray).
    mask = cv2.inRange(hsv, (30, 40, 40), (95, 255, 255))
    upper = mask[: 2 * h // 3]
    lower = mask[2 * h // 3 :]
    upper_ratio = float(np.mean(upper > 0))
    lower_ratio = float(np.mean(lower > 0))
    # Texture: trees have more edges than flat grass.
    gray = cv2.cvtColor(bgr[: 2 * h // 3], cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 150)
    edge_ratio = float(np.mean(edges > 0))
    # Prefer green that isn't only a flat lawn carpet.
    score = (0.65 * upper_ratio) + (0.20 * max(0.0, upper_ratio - lower_ratio)) + (
        0.35 * edge_ratio
    )
    return float(score)


def _looks_under_overhang(bgr: np.ndarray) -> bool:
    """
    Heuristic for flying under a bench/table/shelter:
    upper third much darker/structured than the ground below.
    """
    h, w = bgr.shape[:2]
    if h < 40 or w < 40:
        return False
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    top = gray[: h // 3]
    mid = gray[h // 3 : 2 * h // 3]
    bottom = gray[2 * h // 3 :]
    top_mean = float(np.mean(top))
    bottom_mean = float(np.mean(bottom))
    mid_mean = float(np.mean(mid))
    # Under something: roof/seat above is darker; ground still visible.
    if top_mean < bottom_mean - 28 and bottom_mean > 40:
        return True
    # Strong horizontal structure in the upper half (bench seat / table edge).
    upper = gray[: h // 2]
    edges = cv2.Canny(cv2.GaussianBlur(upper, (5, 5), 0), 50, 140)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=30,
        minLineLength=max(30, w // 4),
        maxLineGap=12,
    )
    if lines is None:
        return False
    horiz = 0
    for item in lines:
        x1, y1, x2, y2 = [float(v) for v in np.asarray(item).reshape(-1)[:4]]
        ang = abs(float(np.degrees(np.arctan2(y2 - y1, x2 - x1))))
        if ang > 90:
            ang = abs(ang - 180)
        if ang < 18:
            horiz += 1
    return horiz >= 3 and top_mean <= mid_mean + 5


def _bank_degrees(bgr: np.ndarray) -> float:
    """
    Estimate camera bank from strong edges (horizon / ground lines).

    Returns absolute degrees from level; ~0 is upright, 30+ is clearly banked.
    """
    h, w = bgr.shape[:2]
    scale = 320 / max(w, 1)
    small = cv2.resize(bgr, (320, max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)
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
        # Angle from horizontal, folded into [-90, 90].
        ang = float(np.degrees(np.arctan2(dy, dx)))
        if ang > 90:
            ang -= 180
        if ang < -90:
            ang += 180
        # Ignore near-vertical lines (trees, poles).
        if abs(ang) > 70:
            continue
        angles.append(abs(ang))

    if not angles:
        return 0.0
    # Robust-ish: 75th percentile of |bank|.
    return float(np.percentile(np.asarray(angles, dtype=np.float64), 75))


def analyze_keyframes(images: list, bank_threshold_deg: float = 22.0) -> SceneSignals:
    """Run YOLO person/prop detect + bank/under-object heuristics."""
    signals = SceneSignals(prop_hits={})
    if not images:
        return signals

    model = _get_yolo()
    banks: list[float] = []
    assert signals.prop_hits is not None

    foliage_scores: list[float] = []
    for image in images:
        bgr = _pil_to_bgr(image)
        bank = _bank_degrees(bgr)
        banks.append(bank)
        if bank >= bank_threshold_deg:
            signals.banked_frame_count += 1

        foliage = _foliage_score(bgr)
        foliage_scores.append(foliage)
        # Tuned for park trees vs flat lawn; ~0.12+ usually means trees/bushes.
        if foliage >= 0.12:
            signals.tree_frames += 1

        under = _looks_under_overhang(bgr)
        if under:
            signals.under_object_frames += 1

        if model and model is not False:
            try:
                n = _count_people(bgr, model)
            except Exception:  # noqa: BLE001
                n = 0
            if n > 0:
                signals.frames_with_people += 1
                signals.person_hits += n
                signals.max_people_in_frame = max(signals.max_people_in_frame, n)

            try:
                props = _detect_props(bgr, model)
            except Exception:  # noqa: BLE001
                props = []
            close_in_frame = False
            for label, area, box in props:
                signals.prop_hits[label] = signals.prop_hits.get(label, 0) + 1
                _x1, y1, _x2, y2 = box
                h = bgr.shape[0]
                # Large box or object sitting in the upper half ≈ close / overhead.
                if area >= 0.08 or (area >= 0.03 and y1 < h * 0.45):
                    close_in_frame = True
                if under and label in _OVERHANG_PROPS and area >= 0.02:
                    close_in_frame = True
            if close_in_frame:
                signals.close_prop_frames += 1

    signals.max_bank_deg = float(max(banks) if banks else 0.0)
    signals.max_foliage_score = float(max(foliage_scores) if foliage_scores else 0.0)

    if signals.frames_with_people > 0:
        if signals.frames_with_people >= 2 or signals.max_people_in_frame >= 2:
            signals.people_hint = (
                f"people visible in {signals.frames_with_people} frame(s) "
                f"(up to {signals.max_people_in_frame} at once) — "
                "likely kids/spectators; check for chasing or playground"
            )
        else:
            signals.people_hint = (
                f"person visible in {signals.frames_with_people} frame(s)"
            )
    else:
        signals.people_hint = "none"

    action_bits: list[str] = []
    if signals.max_bank_deg >= 55 or signals.banked_frame_count >= 3:
        action_bits.extend(["rolls", "flips", "banking"])
    elif signals.max_bank_deg >= bank_threshold_deg or signals.banked_frame_count >= 1:
        action_bits.append("banking / aggressive turns")

    prop_bits: list[str] = []
    landmark_bits: list[str] = []
    for label, count in sorted(signals.prop_hits.items(), key=lambda kv: -kv[1]):
        if count <= 0:
            continue
        prop_bits.append(f"{label} x{count}")
        if label in _VEHICLE_PROPS or label == "potted plant":
            landmark_bits.append(f"{label} x{count}")

    if signals.tree_frames >= max(2, len(images) // 4) or signals.max_foliage_score >= 0.16:
        landmark_bits.insert(
            0,
            "trees/bushes visible — ENVIRONMENT and MOVEMENT must name trees or tree line",
        )
    elif signals.tree_frames >= 1:
        landmark_bits.insert(0, "some trees/foliage — mention trees if clear")

    overhang_props = sum(signals.prop_hits.get(k, 0) for k in _OVERHANG_PROPS)
    under_bench = signals.under_object_frames > 0 and (
        overhang_props > 0 or signals.close_prop_frames > 0
    )
    if under_bench or (
        signals.under_object_frames >= 2 and signals.close_prop_frames >= 1
    ):
        action_bits.append("under-bench / proximity pass")
        # Umbrella class is a common false label for bench seats / canopies.
        label = "bench"
        if signals.prop_hits.get("bench", 0):
            label = "bench"
        elif signals.prop_hits.get("table", 0):
            label = "table"
        elif signals.prop_hits.get("umbrella", 0):
            label = "bench"  # prefer editor-friendly wording
        prop_bits.insert(
            0,
            f"LIKELY UNDER a {label} — MOVEMENT must say under-{label} proximity pass",
        )
    elif signals.under_object_frames >= 2:
        action_bits.append("under-object / proximity pass")
        prop_bits.insert(
            0,
            "LIKELY flying UNDER a bench/table/pavilion — name under-bench if it fits",
        )
    elif signals.close_prop_frames >= 1 and signals.prop_hits:
        nearest = max(signals.prop_hits.items(), key=lambda kv: kv[1])[0]
        if nearest == "umbrella":
            nearest = "bench"
        action_bits.append("proximity pass")
        prop_bits.insert(0, f"CLOSE pass by {nearest}")

    # Near cars / trucks are useful edit landmarks too.
    vehicle_hits = sum(signals.prop_hits.get(k, 0) for k in _VEHICLE_PROPS)
    if vehicle_hits > 0 and "proximity" not in ", ".join(action_bits).lower():
        if any(
            signals.prop_hits.get(k, 0) > 0 and True
            for k in _VEHICLE_PROPS
        ):
            # Only tag proximity if a vehicle box was large / close.
            pass

    signals.action_hint = ", ".join(action_bits)
    signals.props_hint = "; ".join(prop_bits) if prop_bits else "none"
    signals.landmarks_hint = "; ".join(landmark_bits) if landmark_bits else "none"
    return signals
