"""Searchable catalog + local semantic index over keeper notes."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


CATALOG_NAME = "clips_catalog.json"
INDEX_NAME = "clips_search_index.npz"
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
# Cosine floor for MiniLM; below this, results are usually unrelated nearest-neighbors.
DEFAULT_MIN_SCORE = 0.32


@dataclass
class ClipRecord:
    file: str
    video_path: str
    notes_path: str
    json_path: str
    duration_seconds: float
    overall_speed_score: float
    top_speed_score: float
    environment: str
    movement: str
    search_text: str
    people: str = "none"
    action: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _parse_notes_txt(text: str) -> dict:
    env = ""
    people = "none"
    action = ""
    movement = ""
    duration = 0.0
    overall = 0.0
    top = 0.0
    file_name = ""

    env_m = re.search(r"(?im)^\s*Environment\s*:\s*(.+)$", text)
    if env_m:
        env = env_m.group(1).strip()

    people_m = re.search(r"(?im)^\s*People\s*:\s*(.+)$", text)
    if people_m:
        people = people_m.group(1).strip() or "none"

    action_m = re.search(r"(?im)^\s*Action\s*:\s*(.+)$", text)
    if action_m:
        action = action_m.group(1).strip()

    file_m = re.search(r"(?im)^\s*File\s*:\s*(.+)$", text)
    if file_m:
        file_name = file_m.group(1).strip()

    dur_m = re.search(r"(?im)^\s*Duration\s*:\s*([0-9.]+)", text)
    if dur_m:
        duration = float(dur_m.group(1))

    overall_m = re.search(r"(?im)^\s*Overall speed score\s*:\s*([0-9.]+)", text)
    if overall_m:
        overall = float(overall_m.group(1))

    top_m = re.search(r"(?im)^\s*Top speed score\s*:\s*([0-9.]+)", text)
    if top_m:
        top = float(top_m.group(1))

    move_m = re.search(
        r"(?im)^\s*Movement\s*:\s*(.+?)(?=^\s*[A-Za-z][A-Za-z ]{0,24}:|\Z)",
        text,
        re.S,
    )
    if move_m:
        movement = " ".join(move_m.group(1).split()).strip()
    elif "Description:" in text:
        # Older notes format
        part = text.split("Description:", 1)[1].strip()
        movement = " ".join(part.split())

    return {
        "file": file_name,
        "duration_seconds": duration,
        "overall_speed_score": overall,
        "top_speed_score": top,
        "environment": env or "unknown",
        "people": people or "none",
        "action": action,
        "movement": movement or "",
    }


def write_clip_json(
    dest_json: Path,
    *,
    video_path: Path,
    notes_path: Path,
    duration_seconds: float,
    overall_speed_score: float,
    top_speed_score: float,
    environment: str,
    movement: str,
    people: str = "none",
    action: str = "",
) -> Path:
    """Machine-readable sidecar for AI search / agents."""
    from .describer import compose_search_blob

    people = (people or "none").strip()
    action = (action or "").strip()
    environment = (environment or "unknown").strip()
    movement = (movement or "").strip()
    record = ClipRecord(
        file=video_path.name,
        video_path=str(video_path),
        notes_path=str(notes_path),
        json_path=str(dest_json),
        duration_seconds=float(duration_seconds),
        overall_speed_score=float(overall_speed_score),
        top_speed_score=float(top_speed_score),
        environment=environment,
        people=people,
        action=action,
        movement=movement,
        search_text=compose_search_blob(environment, people, action, movement),
    )
    dest_json.parent.mkdir(parents=True, exist_ok=True)
    dest_json.write_text(json.dumps(record.to_dict(), indent=2), encoding="utf-8")
    return dest_json


def collect_records(output_dir: Path) -> list[ClipRecord]:
    """Load all clip records from sidecars or legacy .txt notes."""
    output_dir = Path(output_dir)
    records: list[ClipRecord] = []
    seen: set[str] = set()

    for json_path in sorted(output_dir.glob("*.json")):
        if json_path.name in {CATALOG_NAME, "triage_report.json"}:
            continue
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict) or (
            "search_text" not in data and "movement" not in data
        ):
            continue
        file_name = data.get("file") or json_path.with_suffix(".mp4").name
        video_path = Path(data.get("video_path") or (output_dir / file_name))
        notes_path = Path(data.get("notes_path") or video_path.with_suffix(".txt"))
        env = data.get("environment") or "unknown"
        people = data.get("people") or "none"
        action = data.get("action") or ""
        movement = data.get("movement") or data.get("description") or ""
        from .describer import compose_search_blob

        search_text = data.get("search_text") or compose_search_blob(
            env, people, action, movement
        )
        key = str(video_path.resolve()) if video_path.exists() else json_path.stem
        if key in seen:
            continue
        seen.add(key)
        records.append(
            ClipRecord(
                file=file_name,
                video_path=str(video_path),
                notes_path=str(notes_path),
                json_path=str(json_path),
                duration_seconds=float(data.get("duration_seconds") or 0.0),
                overall_speed_score=float(data.get("overall_speed_score") or 0.0),
                top_speed_score=float(data.get("top_speed_score") or 0.0),
                environment=env,
                people=people,
                action=action,
                movement=movement,
                search_text=search_text,
            )
        )

    for txt_path in sorted(output_dir.glob("*.txt")):
        try:
            parsed = _parse_notes_txt(txt_path.read_text(encoding="utf-8"))
        except OSError:
            continue
        # Prefer matching video beside the notes.
        video = None
        for ext in (".mp4", ".MP4", ".mov", ".MOV", ".mkv"):
            candidate = txt_path.with_suffix(ext)
            if candidate.exists():
                video = candidate
                break
        if video is None:
            video = output_dir / (parsed["file"] or txt_path.stem)
        key = str(video.resolve()) if video.exists() else txt_path.stem
        if key in seen:
            continue
        seen.add(key)
        env = parsed["environment"]
        people = parsed.get("people") or "none"
        action = parsed.get("action") or ""
        movement = parsed["movement"]
        json_path = txt_path.with_suffix(".json")
        from .describer import compose_search_blob

        records.append(
            ClipRecord(
                file=parsed["file"] or video.name,
                video_path=str(video),
                notes_path=str(txt_path),
                json_path=str(json_path),
                duration_seconds=parsed["duration_seconds"],
                overall_speed_score=parsed["overall_speed_score"],
                top_speed_score=parsed["top_speed_score"],
                environment=env,
                people=people,
                action=action,
                movement=movement,
                search_text=compose_search_blob(env, people, action, movement),
            )
        )

    return records


def write_catalog(output_dir: Path, records: list[ClipRecord] | None = None) -> Path:
    output_dir = Path(output_dir)
    records = records if records is not None else collect_records(output_dir)
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "count": len(records),
        "clips": [r.to_dict() for r in records],
    }
    path = output_dir / CATALOG_NAME
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


class NotesSearchIndex:
    """Local embedding search over keeper notes (CPU MiniLM)."""

    def __init__(self, model_id: str = EMBED_MODEL_ID) -> None:
        self.model_id = model_id
        self._model = None
        self.records: list[ClipRecord] = []
        self.embeddings: np.ndarray | None = None

    def _ensure_model(self):
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self.model_id)

    def build(self, output_dir: Path) -> int:
        self.records = collect_records(output_dir)
        write_catalog(output_dir, self.records)
        if not self.records:
            self.embeddings = None
            index_path = Path(output_dir) / INDEX_NAME
            if index_path.exists():
                index_path.unlink()
            return 0

        self._ensure_model()
        texts = [r.search_text for r in self.records]
        assert self._model is not None
        emb = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        self.embeddings = np.asarray(emb, dtype=np.float32)
        self.save(Path(output_dir) / INDEX_NAME)
        return len(self.records)

    def save(self, path: Path) -> None:
        if self.embeddings is None:
            return
        meta = json.dumps([r.to_dict() for r in self.records])
        np.savez_compressed(
            path,
            embeddings=self.embeddings,
            meta=np.asarray([meta], dtype=object),
            model_id=np.asarray([self.model_id]),
        )

    def load(self, output_dir: Path) -> bool:
        path = Path(output_dir) / INDEX_NAME
        if not path.exists():
            return False
        data = np.load(path, allow_pickle=True)
        self.embeddings = data["embeddings"]
        meta = json.loads(str(data["meta"][0]))
        self.records = [ClipRecord(**item) for item in meta]
        if "model_id" in data:
            self.model_id = str(data["model_id"][0])
        return True

    def search(
        self,
        query: str,
        top_k: int = 8,
        min_score: float = DEFAULT_MIN_SCORE,
    ) -> list[tuple[ClipRecord, float]]:
        """Hybrid keyword + semantic search. Drops weak nearest-neighbor noise."""
        query = (query or "").strip()
        if not query:
            return []
        if self.embeddings is None or not self.records:
            return []

        tokens = [
            t
            for t in re.findall(r"[a-z0-9]+", query.lower())
            if len(t) >= 2 and t not in {"a", "an", "the", "of", "to", "in", "on", "at", "for", "with"}
        ]

        def keyword_score(record: ClipRecord) -> float:
            if not tokens:
                return 0.0
            hay = (
                f"{record.search_text} {record.movement} {record.environment} "
                f"{record.people} {record.action} {record.file}"
            ).lower()
            hits = sum(1 for t in tokens if t in hay)
            return hits / len(tokens)

        self._ensure_model()
        assert self._model is not None
        q = self._model.encode([query], normalize_embeddings=True, show_progress_bar=False)
        q = np.asarray(q, dtype=np.float32)[0]
        semantic = self.embeddings @ q
        keyword = np.asarray(
            [keyword_score(r) for r in self.records], dtype=np.float32
        )
        # Exact note matches matter more for objects ("chair"); semantics help phrases.
        combined = (0.55 * semantic) + (0.45 * keyword)

        keep: list[tuple[int, float]] = []
        for i, (sem, kw, comb) in enumerate(zip(semantic, keyword, combined)):
            # Keep if notes literally mention most query terms, or semantics clear the floor.
            if kw >= 0.5 or float(sem) >= min_score:
                keep.append((i, float(comb)))

        keep.sort(key=lambda item: item[1], reverse=True)
        return [
            (self.records[i], score)
            for i, score in keep[: max(1, top_k)]
        ]


def rebuild_search_assets(output_dir: Path) -> tuple[Path, int]:
    """Refresh clips_catalog.json + embedding index for an output folder."""
    index = NotesSearchIndex()
    count = index.build(output_dir)
    return Path(output_dir) / CATALOG_NAME, count
