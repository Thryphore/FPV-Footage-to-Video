"""Local vision-language descriptions for FPV clips."""

from __future__ import annotations

from pathlib import Path

from .analyzer import FlightAnalysis, extract_keyframes


DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"

DESCRIBE_PROMPT = (
    "These frames are from an FPV drone recording. Write a concise editing note "
    "in 3-5 sentences covering: setting/environment, what the pilot does, "
    "camera moves, and anything useful for picking this clip in a highlight reel. "
    "If it does not look like real flight, say so plainly. Do not mention scores."
)


class VideoDescriber:
    """Lazy-loaded Qwen3-VL captioner (4B fits well on 16GB VRAM)."""

    def __init__(self, model_id: str = DEFAULT_MODEL_ID) -> None:
        self.model_id = model_id
        self._model = None
        self._processor = None
        self._device = None

    def load(self) -> None:
        if self._model is not None:
            return

        import torch
        from transformers import AutoProcessor

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

    @staticmethod
    def _model_class_for(model_id: str):
        name = model_id.lower()
        if "qwen3-vl" in name or "qwen3_vl" in name:
            from transformers import Qwen3VLForConditionalGeneration

            return Qwen3VLForConditionalGeneration
        from transformers import Qwen2_5_VLForConditionalGeneration

        return Qwen2_5_VLForConditionalGeneration

    def describe(self, video_path: Path, keyframe_count: int = 6) -> str:
        self.load()
        assert self._model is not None and self._processor is not None

        import torch

        images = extract_keyframes(video_path, count=keyframe_count)
        if not images:
            return "Could not extract frames for description."

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

        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=220,
                do_sample=False,
            )

        trimmed = [
            out[len(inp) :] for inp, out in zip(inputs.input_ids, output_ids, strict=False)
        ]
        description = self._processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        return description or "No description generated."


def write_notes_file(
    analysis: FlightAnalysis,
    dest_txt: Path,
    description: str,
) -> Path:
    """Write a sidecar text file next to a keeper clip."""
    video_name = Path(analysis.path).name
    lines = [
        f"File: {video_name}",
        f"Duration: {analysis.duration_seconds:.1f}s",
        f"Flight time: {analysis.flight_seconds:.1f}s",
        f"Overall speed score: {analysis.overall_speed_score:.1f} / 10",
        f"Top speed score: {analysis.top_speed_score:.1f} / 10",
        "",
        "Description:",
        description.strip() or "(no description)",
        "",
    ]
    dest_txt.parent.mkdir(parents=True, exist_ok=True)
    dest_txt.write_text("\n".join(lines), encoding="utf-8")
    return dest_txt
