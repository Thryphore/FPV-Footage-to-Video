"""Summarize what DJI/FPV .srt subtitle telemetry actually contains."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(r"D:\other thing drone")
SAMPLE_FILES = [
    "DJIG0168.srt",
    "DJIG0180.srt",
    "DJIG0061.srt",
    "DJIG0135.srt",
    "DJIG0150.srt",
    "DJIG0006.srt",
    "DJIG0048.srt",
]


def blocks(text: str) -> list[str]:
    return [b.strip() for b in re.split(r"\n\s*\n", text.strip()) if b.strip()]


def cue_body(block: str) -> str | None:
    lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
    if len(lines) < 3:
        return None
    return " ".join(lines[2:])


def field_tokens(body: str) -> list[tuple[str, str]]:
    # key:value tokens separated by spaces, values may be N/A or units.
    return re.findall(r"([A-Za-z][A-Za-z0-9_]*)\s*:\s*([^\s]+)", body)


def analyze_file(path: Path) -> dict:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    cues = blocks(raw)
    key_counts: Counter[str] = Counter()
    nonempty: Counter[str] = Counter()
    examples: dict[str, set[str]] = defaultdict(set)
    numeric_stats: dict[str, list[float]] = defaultdict(list)

    for block in cues:
        body = cue_body(block)
        if not body:
            continue
        for key, val in field_tokens(body):
            key_counts[key] += 1
            if val.upper() != "N/A":
                nonempty[key] += 1
                if len(examples[key]) < 6:
                    examples[key].add(val)
                m = re.match(r"^(-?\d+(?:\.\d+)?)([A-Za-z/%]*)$", val)
                if m:
                    numeric_stats[key].append(float(m.group(1)))

    return {
        "path": path,
        "bytes": path.stat().st_size,
        "cues": len(cues),
        "keys": key_counts,
        "nonempty": nonempty,
        "examples": {k: sorted(v) for k, v in examples.items()},
        "numeric": numeric_stats,
    }


def print_file_report(info: dict) -> None:
    path: Path = info["path"]
    print(f"\n=== {path.name} ({info['bytes']} bytes, {info['cues']} cues) ===")
    if info["cues"] == 0 or info["bytes"] == 0:
        print("  empty / unusable")
        return
    keys = sorted(info["keys"])
    print("  fields:")
    for key in keys:
        total = info["keys"][key]
        good = info["nonempty"].get(key, 0)
        nums = info["numeric"].get(key, [])
        extras = ""
        if nums:
            extras = (
                f"  num min/max/mean="
                f"{min(nums):.2f}/{max(nums):.2f}/{sum(nums)/len(nums):.2f}"
            )
            nonzero = sum(1 for n in nums if abs(n) > 1e-6)
            extras += f"  nonzero={nonzero}/{len(nums)}"
        print(
            f"    {key:12} nonempty {good}/{total} "
            f"({100*good/max(total,1):.0f}%)  eg={info['examples'].get(key, [])}"
            f"{extras}"
        )


def corpus_summary(files: list[Path]) -> None:
    print(f"\n=== CORPUS ({len(files)} srt files) ===")
    present = 0
    nonempty_files = Counter()
    useful_height = useful_vs = useful_hs = useful_gps = 0
    sizes = []
    for path in files:
        if path.stat().st_size <= 0:
            continue
        present += 1
        sizes.append(path.stat().st_size)
        try:
            info = analyze_file(path)
        except OSError:
            continue
        for key, good in info["nonempty"].items():
            if good > 0:
                nonempty_files[key] += 1
        nums = info["numeric"]
        if any(abs(v) > 0.05 for v in nums.get("Height", [])):
            useful_height += 1
        if any(abs(v) > 0.05 for v in nums.get("VS", [])):
            useful_vs += 1
        if any(abs(v) > 0.05 for v in nums.get("HS", [])):
            useful_hs += 1
        if any(v > 0 for v in nums.get("gpsNum", [])):
            useful_gps += 1

    print(f"non-empty files: {present}/{len(files)}")
    if sizes:
        print(
            f"size bytes min/median/max="
            f"{min(sizes)}/{sorted(sizes)[len(sizes)//2]}/{max(sizes)}"
        )
    print("files with at least one non-N/A value per field:")
    for key, count in nonempty_files.most_common():
        print(f"  {key:12} {count}/{present}")
    print(
        f"files with real Height>0: {useful_height}, HS>0: {useful_hs}, "
        f"VS!=0: {useful_vs}, gpsNum>0: {useful_gps}"
    )


if __name__ == "__main__":
    # Show a raw sample first.
    sample = ROOT / "DJIG0168.srt"
    text = sample.read_text(encoding="utf-8", errors="ignore")
    print("=== RAW SAMPLE DJIG0168 (first 2 cues) ===")
    print("\n\n".join(blocks(text)[:2]))
    sample2 = ROOT / "DJIG0180.srt"
    if sample2.exists():
        print("\n=== RAW SAMPLE DJIG0180 (cue ~middle) ===")
        b = blocks(sample2.read_text(encoding="utf-8", errors="ignore"))
        print(b[len(b) // 2] if b else "(empty)")

    for name in SAMPLE_FILES:
        path = ROOT / name
        if path.exists():
            print_file_report(analyze_file(path))

    all_srts = sorted(ROOT.glob("DJIG*.srt"))
    corpus_summary(all_srts)
