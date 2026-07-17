"""Qwen transcript repair adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jp_learning_platform.domain import Segment, Sentence
from jp_learning_platform.workflow.qwen_repair_stage import (
    QwenRepair,
    QwenRepairRequest,
)

DEFAULT_QWEN_CONTEXT = 4096
DEFAULT_QWEN_THREADS = 8
DEFAULT_QWEN_GPU_LAYERS = 0
DEFAULT_QWEN_MAX_TOKENS = 128
DEFAULT_QWEN_TEMPERATURE = 0.03
DEFAULT_QWEN_TOP_P = 0.9
DEFAULT_QWEN_REPEAT_PENALTY = 1.1


class QwenDependencyError(RuntimeError):
    """Raised when llama-cpp-python is not installed."""

    def __init__(self) -> None:
        super().__init__(
            "llama-cpp-python is required for Qwen repair. "
            "Install it with: python -m pip install -e '.[qwen]'"
        )


class QwenModelNotFoundError(RuntimeError):
    """Raised when the configured Qwen model file is missing."""

    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        super().__init__(f"Qwen model file not found: {model_path}")


@dataclass(frozen=True, slots=True)
class PassthroughQwenRepairer:
    """Keep aligned segments unchanged while still running the repair stage."""

    def repair(self, request: QwenRepairRequest) -> QwenRepair:
        if not isinstance(request, QwenRepairRequest):
            raise TypeError("request must be a QwenRepairRequest.")

        return QwenRepair(source_path=request.source_path, segments=request.segments)


@dataclass(slots=True)
class LlamaCppQwenRepairer:
    """Repair Japanese transcript text with a local Qwen GGUF model."""

    model_path: Path
    context_size: int = DEFAULT_QWEN_CONTEXT
    threads: int = DEFAULT_QWEN_THREADS
    gpu_layers: int = DEFAULT_QWEN_GPU_LAYERS
    max_tokens: int = DEFAULT_QWEN_MAX_TOKENS
    temperature: float = DEFAULT_QWEN_TEMPERATURE
    top_p: float = DEFAULT_QWEN_TOP_P
    repeat_penalty: float = DEFAULT_QWEN_REPEAT_PENALTY
    _model: Any | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.model_path = Path(self.model_path)

    def repair(self, request: QwenRepairRequest) -> QwenRepair:
        if not isinstance(request, QwenRepairRequest):
            raise TypeError("request must be a QwenRepairRequest.")

        texts = tuple(segment.text for segment in request.segments)
        repaired_segments = tuple(
            self._repair_segment(index, segment, texts)
            for index, segment in enumerate(request.segments)
        )
        return QwenRepair(
            source_path=request.source_path,
            segments=repaired_segments,
        )

    def _repair_segment(
        self,
        index: int,
        segment: Segment,
        texts: tuple[str, ...],
    ) -> Segment:
        previous_text = texts[index - 1] if index > 0 else ""
        next_text = texts[index + 1] if index + 1 < len(texts) else ""
        repaired_text = self._repair_text(previous_text, segment.text, next_text)
        words = tuple(
            word
            for sentence in segment.sentences
            for word in sentence.words
        )
        sentence = Sentence(
            text=repaired_text,
            time_range=segment.time_range,
            words=words,
        )
        return Segment(
            position=segment.position,
            text=repaired_text,
            time_range=segment.time_range,
            sentences=(sentence,),
        )

    def _repair_text(self, previous_text: str, current_text: str, next_text: str) -> str:
        normalized_text = current_text.strip()
        if not normalized_text:
            return current_text

        response = self._load_model()(
            self._build_prompt(previous_text, normalized_text, next_text),
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            repeat_penalty=self.repeat_penalty,
            stop=("\n\n", "PREV:", "NEXT:", "CURRENT:"),
        )
        choices = response.get("choices", ())
        if not choices:
            return normalized_text

        repaired_text = str(choices[0].get("text", "")).strip()
        return self._clean_output(repaired_text) or normalized_text

    def _load_model(self) -> Any:
        if not self.model_path.exists():
            raise QwenModelNotFoundError(self.model_path)

        if self._model is None:
            try:
                from llama_cpp import Llama
            except ImportError as error:
                raise QwenDependencyError() from error

            self._model = Llama(
                model_path=str(self.model_path),
                n_ctx=self.context_size,
                n_threads=self.threads,
                n_gpu_layers=self.gpu_layers,
                verbose=False,
            )

        return self._model

    def _build_prompt(
        self,
        previous_text: str,
        current_text: str,
        next_text: str,
    ) -> str:
        return f"""あなたは日本語字幕修正AIです。

Whisper音声認識の誤認識だけを修正してください。
CURRENT のみを自然な日本語に修正し、説明や補足を出力しないでください。

PREV:
{previous_text}

CURRENT:
{current_text}

NEXT:
{next_text}

出力:
"""

    def _clean_output(self, text: str) -> str:
        cleaned = text.strip()
        for marker in ("PREV:", "NEXT:", "CURRENT:", "出力:"):
            cleaned = cleaned.replace(marker, "")

        return " ".join(cleaned.split())


__all__ = [
    "DEFAULT_QWEN_CONTEXT",
    "DEFAULT_QWEN_GPU_LAYERS",
    "DEFAULT_QWEN_THREADS",
    "LlamaCppQwenRepairer",
    "PassthroughQwenRepairer",
    "QwenDependencyError",
    "QwenModelNotFoundError",
]
