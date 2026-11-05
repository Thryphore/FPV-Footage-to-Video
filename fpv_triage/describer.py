"""Local vision-language descriptions for FPV clips."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path

from .analyzer import FlightAnalysis, extract_keyframes


DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"

DESCRIBE_PROMPT = (
    "These frames are from a real FPV / cinewhoop / DJI drone recording. "
    "Treat fisheye, flare, rolling shutter, and compression as normal. "
    "Respond with EXACTLY this format and nothing else:\n"
    "ENVIRONMENT: <2-4 word place tag, e.g. grassy park, forest trail, parking lot>\n"
    "MOVEMENT: <2-3 sentences focused on pilot/camera motion for editing: "
    "direction, altitude changes, turns/yaw, proximity passes, pace "
    "(hover / slow cruise / ripping), and useful cut points. "
    "Name 1-2 distinctive nearby props only if clearly visible "
    "(bench, chair, car, person, pavilion, fence) so editors can search later. "
    "Do NOT pad with scenery. Do NOT call the footage fake or simulated.>"
)


@dataclass
class ClipNotes:
    environment_tag: str
    movement: str
    raw: str
    keyframe_seconds: float = 0.0
    inference_seconds: float = 0.0
    load_seconds: float = 0.0


def parse_clip_notes(raw: str) -> tuple[str, str]:
    """Pull ENVIRONMENT / MOVEMENT lines out of model output."""
    text = (raw or "").strip()
    env = ""
    movement = ""

    env_match = re.search(
        r"(?im)^\s*ENVIRONMENT\s*:\s*(.+)$",
        text,
    )
    move_match = re.search(
        r"(?im)^\s*MOVEMENT\s*:\s*(.+?)(?=^\s*[A-Z]{3,}\s*:|\Z)",
        text,
        re.S,
    )
    if env_match:
        env = env_match.group(1).strip().strip('"').strip()
    if move_match:
        movement = " ".join(move_match.group(1).split()).strip().strip('"').strip()

    if not env and not movement:
        # Fallback: first short clause as tag-ish, rest as movement.
        movement = " ".join(text.split())
        env = "unknown"

    if env and len(env) > 48:
        env = env[:45].rstrip() + "..."
    return env or "unknown", movement or text or "(no description)"


class VideoDescriber:
    """Lazy-loaded Qwen3-VL captioner (4B fits well on 16GB VRAM)."""

    def __init__(self, model_id: str = DEFAULT_MODEL_ID) -> None:
        self.model_id = model_id
        self._model = None
        self._processor = None
        self._device = None
        self.last_load_seconds: float = 0.0

    def load(self) -> None:
        if self._model is not None:
            self.last_load_seconds = 0.0
            return

        import torch
        from transformers import AutoProcessor

        t0 = time.perf_counter()
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if self._device == "cuda" else torch.float32

        model_cls = self._model_class_for(self.model_id)
        self._processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        self._model = model_cls.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            device_map="auto" if self._device == "cuda" else None,
            trust_remote_code=True,
        )
        if self._device == "cpu":
            self._model = self._model.to(self._device)
        self._model.eval()
        self.last_load_seconds = time.perf_counter() - t0

    @staticmethod
    def _model_class_for(model_id: str):
        name = model_id.lower()
        if "qwen3-vl" in name or "qwen3_vl" in name:
            from transformers import Qwen3VLForConditionalGeneration

            return Qwen3VLForConditionalGeneration
        from transformers import Qwen2_5_VLForConditionalGeneration

        return Qwen2_5_VLForConditionalGeneration

    def describe(self, video_path: Path, keyframe_count: int = 4) -> ClipNotes:
        self.load()
        assert self._model is not None and self._processor is not None
        load_seconds = self.last_load_seconds

        import torch

        t_key = time.perf_counter()
        images = extract_keyframes(video_path, count=keyframe_count)
        keyframe_seconds = time.perf_counter() - t_key
        if not images:
            return ClipNotes(
                environment_tag="unknown",
                movement="Could not extract frames for description.",
                raw="",
                keyframe_seconds=keyframe_seconds,
                load_seconds=load_seconds,
            )

        content = [{"type": "image", "image": img} for img in images]
        content.append({"type": "text", "text": DESCRIBE_PROMPT})
        messages = [{"role": "user", "content": content}]

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(
            text=[text],
            images=images,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self._model.device)

        t_inf = time.perf_counter()
        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=140,
                do_sample=False,
            )
        inference_seconds = time.perf_counter() - t_inf

        trimmed = [
            out[len(inp) :] for inp, out in zip(inputs.input_ids, output_ids, strict=False)
        ]
        raw = self._processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        env, movement = parse_clip_notes(raw)
        return ClipNotes(
            environment_tag=env,
            movement=movement,
            raw=raw,
            keyframe_seconds=keyframe_seconds,
            inference_seconds=inference_seconds,
            load_seconds=load_seconds,
        )


def write_notes_file(
    analysis: FlightAnalysis,
    dest_txt: Path,
    *,
    environment_tag: str,
    movement: str,
    video_path: Path | None = None,
) -> Path:
    """Write human .txt + machine-readable .json sidecars for a keeper clip."""
    from .catalog import write_clip_json

    video_name = Path(analysis.path).name
    lines = [
        f"File: {video_name}",
        f"Duration: {analysis.duration_seconds:.1f}s",
        f"Overall speed score: {analysis.overall_speed_score:.1f} / 10",
        f"Top speed score: {analysis.top_speed_score:.1f} / 10",
        f"Environment: {environment_tag.strip() or 'unknown'}",
        "",
        "Movement:",
        movement.strip() or "(no description)",
        "",
    ]
    dest_txt.parent.mkdir(parents=True, exist_ok=True)
    dest_txt.write_text("\n".join(lines), encoding="utf-8")

    clip_video = Path(video_path) if video_path else dest_txt.with_suffix(".mp4")
    write_clip_json(
        dest_txt.with_suffix(".json"),
        video_path=clip_video,
        notes_path=dest_txt,
        duration_seconds=analysis.duration_seconds,
        overall_speed_score=analysis.overall_speed_score,
        top_speed_score=analysis.top_speed_score,
        environment=environment_tag,
        movement=movement,
    )
    return dest_txt


def format_timing_line(timing: dict[str, float] | None) -> str:
    if not timing:
        return ""
    parts = []
    order = (
        ("analyze", "analyze"),
        ("copy", "copy"),
        ("model_load", "model load"),
        ("keyframes", "keyframes"),
        ("inference", "AI"),
        ("total", "total"),
    )
    for key, label in order:
        if key in timing:
            parts.append(f"{label} {timing[key]:.1f}s")
    return " | ".join(parts)
