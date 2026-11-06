"""Local vision-language descriptions for FPV clips."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path

from .analyzer import FlightAnalysis, extract_keyframes


DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
DEFAULT_KEYFRAME_COUNT = 8

OVERHANG_PROMPT = (
    "These frames are from a low FPV proximity pass. "
    "Is the drone flying UNDER or THROUGH any of: bench, picnic table, "
    "pavilion/shelter, fence gap, playground equipment? "
    "Answer with EXACTLY one line:\n"
    "OVERHANG: <none OR under-bench OR under-table OR under-pavilion OR "
    "through-fence OR under-playground>"
)

DESCRIBE_PROMPT = (
    "These stills are in time order from one real FPV drone flight "
    "(fisheye/flare/rolling shutter are normal; never say fake).\n"
    "A detector already scanned the frames. Trust these hints — do not ignore them:\n"
    "DETECTED_PEOPLE: {people_hint}\n"
    "DETECTED_ACTION: {action_hint}\n"
    "DETECTED_PROPS: {props_hint}\n"
    "DETECTED_LANDMARKS: {landmarks_hint}\n"
    "Write editor notes. If DETECTED_PEOPLE is not none, PEOPLE must describe "
    "them (kids/spectators/playground/chasing when that fits) and MOVEMENT must "
    "mention them — never write 'no people'. "
    "If DETECTED_ACTION mentions banking/rolls/flips, ACTION must include those "
    "words — never replace them with only 'cruise'. "
    "If DETECTED_PROPS or DETECTED_ACTION mentions under-object, under-bench, "
    "bench, table, shelter, or proximity, ACTION and MOVEMENT MUST explicitly say "
    "the under-bench / under-table / under-shelter proximity pass. "
    "If DETECTED_LANDMARKS mentions trees/bushes/cars, ENVIRONMENT and MOVEMENT "
    "MUST name them (tree, tree line, cars, etc.) — do not omit scenery landmarks.\n"
    "Respond with EXACTLY:\n"
    "ENVIRONMENT: <2-6 word place tag including trees/cars when present>\n"
    "PEOPLE: <none OR specific: kids chasing drone, people on playground, etc.>\n"
    "ACTION: <comma tags>\n"
    "MOVEMENT: <3-5 sentences naming trees/landmarks, people, under-object "
    "proximity passes, maneuvers when present, camera path, pace, cut points>"
)


@dataclass
class ClipNotes:
    environment_tag: str
    movement: str
    raw: str
    people: str = "none"
    action: str = ""
    keyframe_seconds: float = 0.0
    inference_seconds: float = 0.0
    load_seconds: float = 0.0


def _field(text: str, name: str) -> str:
    match = re.search(
        rf"(?im)^\s*{name}\s*:\s*(.+?)(?=^\s*[A-Z]{{3,}}\s*:|\Z)",
        text,
        re.S,
    )
    if not match:
        return ""
    return " ".join(match.group(1).split()).strip().strip('"').strip()


def parse_clip_notes(raw: str) -> tuple[str, str, str, str]:
    """Pull ENVIRONMENT / PEOPLE / ACTION / MOVEMENT from model output."""
    text = (raw or "").strip()
    env = _field(text, "ENVIRONMENT")
    people = _field(text, "PEOPLE")
    action = _field(text, "ACTION")
    movement = _field(text, "MOVEMENT")

    if not env and not movement and not people and not action:
        movement = " ".join(text.split())
        env = "unknown"

    if env and len(env) > 48:
        env = env[:45].rstrip() + "..."
    if not people:
        people = "none"
    if people.lower() in {"n/a", "na", "no", "nobody", "no one", "no people"}:
        people = "none"
    return (
        env or "unknown",
        people,
        action or "unspecified",
        movement or text or "(no description)",
    )


def compose_search_blob(
    environment: str,
    people: str,
    action: str,
    movement: str,
) -> str:
    """Text used for semantic + keyword search."""
    parts = [
        (environment or "unknown").strip(),
        f"People: {(people or 'none').strip()}",
        f"Action: {(action or '').strip()}".rstrip(": ").strip(),
        (movement or "").strip(),
    ]
    return " ".join(p for p in parts if p)


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

    def _generate(self, images: list, prompt: str, max_new_tokens: int) -> str:
        assert self._model is not None and self._processor is not None
        import torch

        content = [{"type": "image", "image": img} for img in images]
        content.append({"type": "text", "text": prompt})
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
        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        trimmed = [
            out[len(inp) :] for inp, out in zip(inputs.input_ids, output_ids, strict=False)
        ]
        return self._processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

    def describe(
        self,
        video_path: Path,
        keyframe_count: int = DEFAULT_KEYFRAME_COUNT,
    ) -> ClipNotes:
        """
        YOLO + bank heuristics catch tiny people / rolls, then Qwen writes notes
        with those detections forced into the prompt.
        """
        from .scene_signals import analyze_keyframes

        self.load()
        load_seconds = self.last_load_seconds

        t_key = time.perf_counter()
        # Extra frames for YOLO/bank; fewer sent to the VL model for speed.
        signal_images = extract_keyframes(
            video_path,
            count=max(keyframe_count * 2, 16),
            max_width=768,
        )
        keyframe_seconds = time.perf_counter() - t_key
        if not signal_images:
            return ClipNotes(
                environment_tag="unknown",
                people="none",
                action="unspecified",
                movement="Could not extract frames for description.",
                raw="",
                keyframe_seconds=keyframe_seconds,
                load_seconds=load_seconds,
            )

        from .scene_signals import _looks_under_overhang, _pil_to_bgr

        signals = analyze_keyframes(signal_images)
        step = max(1, len(signal_images) // keyframe_count)
        images = signal_images[::step][:keyframe_count]
        if len(images) < min(4, len(signal_images)):
            images = signal_images[:keyframe_count]

        overhang_label = ""
        under_frames = [
            im
            for im in signal_images
            if _looks_under_overhang(_pil_to_bgr(im))
        ][:4]
        t_inf = time.perf_counter()
        if under_frames or "under" in (signals.props_hint or "").lower():
            sample = under_frames or images[:3]
            overhang_raw = self._generate(sample, OVERHANG_PROMPT, max_new_tokens=30)
            overhang_label = _field(overhang_raw, "OVERHANG") or overhang_raw.strip()
            overhang_label = overhang_label.split("\n")[0].strip().strip('"')
            if overhang_label.lower() in {"", "none", "n/a", "no"}:
                overhang_label = ""
            elif overhang_label:
                # Prefer concrete under-bench wording for editor search.
                low = overhang_label.lower()
                if "bench" in low:
                    overhang_label = "under-bench"
                signals.props_hint = (
                    f"{overhang_label} confirmed; " + (signals.props_hint or "none")
                )
                if "under-bench" not in (signals.action_hint or "").lower():
                    tip = overhang_label
                    signals.action_hint = (
                        f"{signals.action_hint}, {tip}".strip(", ")
                        if signals.action_hint
                        else tip
                    )

        prompt = DESCRIBE_PROMPT.format(
            people_hint=signals.people_hint or "none",
            action_hint=signals.action_hint or "none detected",
            props_hint=signals.props_hint or "none",
            landmarks_hint=signals.landmarks_hint or "none",
        )

        raw = self._generate(images, prompt, max_new_tokens=280)
        inference_seconds = time.perf_counter() - t_inf

        env, people, action, movement = parse_clip_notes(raw)
        if not people:
            people = _field(raw, "PEOPLE") or "none"
        if not action or action == "unspecified":
            action = _field(raw, "ACTION") or "unspecified"
        if not movement or movement.startswith("("):
            movement = _field(raw, "MOVEMENT") or raw or "(no description)"

        # Prefer detector truth when the VL model still says none.
        if people == "none" and signals.people_hint != "none":
            people = signals.people_hint
        if (
            action.lower() in {"none", "unspecified", "cruise", "cruise, yaw", "yaw"}
            and signals.action_hint
        ):
            action = signals.action_hint
        elif signals.action_hint:
            # Merge missing aerobatic / proximity tags.
            for tag in ("rolls", "flips", "banking", "under-object", "proximity"):
                if tag in signals.action_hint.lower() and tag not in action.lower():
                    action = f"{action}, {tag}".strip(", ")

        move_l = movement.lower()
        negative_people = any(
            p in move_l
            for p in (
                "no people",
                "nobody",
                "no person",
                "not visible",
                "none visible",
            )
        )
        if people == "none" and not negative_people and any(
            w in move_l
            for w in ("kid", "child", "children", "people", "person", "playground", "chase")
        ):
            people = "people visible (see movement)"

        # Strip contradictory "no people" lines when detector found humans.
        if signals.people_hint != "none" and negative_people:
            movement = re.sub(
                r"(?i)\s*no people are visible[^.]*\.?",
                "",
                movement,
            ).strip()
            if people == "none":
                people = signals.people_hint

        # Force under-bench / proximity wording when detectors saw it.
        props_l = (signals.props_hint or "").lower()
        action_l = (signals.action_hint or "").lower()
        overhang_l = (overhang_label or "").lower()
        needs_under = any(
            k in props_l or k in action_l or k in overhang_l
            for k in ("under", "bench", "proximity", "table", "shelter", "pavilion")
        )
        said_under = any(
            k in move_l
            for k in (
                "under-bench",
                "under bench",
                "under the bench",
                "beneath the bench",
                "under-table",
                "under the table",
                "under-pavilion",
                "under the pavilion",
                "proximity pass",
            )
        )
        if needs_under and not said_under:
            detail = "under-object proximity pass"
            if "bench" in overhang_l or "bench" in props_l or "bench" in action_l:
                detail = "under-bench proximity pass"
            elif "table" in overhang_l or "table" in props_l:
                detail = "under-table proximity pass"
            elif "pavilion" in overhang_l or "shelter" in props_l:
                detail = "under-pavilion proximity pass"
            elif overhang_label:
                detail = overhang_label.replace("_", " ")
            movement = f"{movement.rstrip('. ')}. Includes {detail}."
            if detail.lower() not in action.lower():
                action = f"{action}, {detail}".strip(", ")

        # Avoid false PEOPLE from phrases like "no visible people".
        if re.search(r"(?i)\bno visible people\b|\bno people\b", movement):
            if people.startswith("people visible"):
                people = "none" if signals.people_hint == "none" else signals.people_hint

        # Force trees / landmark wording when foliage or YOLO saw them.
        landmarks_l = (signals.landmarks_hint or "").lower()
        env_l = (env or "").lower()
        needs_trees = "tree" in landmarks_l or "bush" in landmarks_l or "foliage" in landmarks_l
        said_trees = any(
            k in env_l or k in move_l for k in ("tree", "trees", "bush", "bushes", "canopy")
        )
        if needs_trees and not said_trees:
            if "tree" not in env_l:
                env = f"{env} trees".strip() if env and env != "unknown" else "tree-lined park"
            movement = f"{movement.rstrip('. ')}. Trees/tree line visible in frame."
        vehicle_count = 0
        if signals.prop_hits:
            vehicle_count = sum(
                int(signals.prop_hits.get(k, 0) or 0)
                for k in ("car", "truck", "bus", "motorcycle", "bicycle")
            )
        needs_cars = vehicle_count >= 2 or (
            vehicle_count >= 1 and "car" in landmarks_l
        )
        said_cars = any(k in env_l or k in move_l for k in ("car", "truck", "vehicle", "parked"))
        if needs_cars and not said_cars:
            movement = f"{movement.rstrip('. ')}. Parked vehicles visible as landmarks."

        return ClipNotes(
            environment_tag=env or "unknown",
            people=people or "none",
            action=action or "unspecified",
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
    people: str = "none",
    action: str = "",
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
        f"People: {(people or 'none').strip()}",
        f"Action: {(action or 'unspecified').strip()}",
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
        people=people,
        action=action,
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
