"""Qwen transcript repair adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
from enum import Enum
import logging
from pathlib import Path
from typing import Any
import unicodedata

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
DEFAULT_QWEN_REPAIR_MAX_LENGTH_DELTA_RATIO = 0.2
DEFAULT_QWEN_REPAIR_MAX_CONTENT_CHANGE_RATIO = 0.2

_LOGGER = logging.getLogger(__name__)


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


class QwenRepairSafetyReason(Enum):
    ACCEPTED = "accepted"
    EMPTY_CANDIDATE = "empty_candidate"
    LENGTH_DELTA_EXCEEDED = "length_delta_exceeded"
    CONTENT_CHANGE_EXCEEDED = "content_change_exceeded"


def _normalize_ratio(value: float, field_name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be a number.")

    try:
        ratio = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{field_name} must be a number.") from error

    if ratio < 0.0:
        raise ValueError(f"{field_name} must be non-negative.")

    return ratio


@dataclass(frozen=True, slots=True)
class QwenRepairSafetyDecision:
    """Decision for accepting or rejecting one candidate repair."""

    original_text: str
    candidate_text: str
    accepted: bool
    reason: QwenRepairSafetyReason
    length_delta_ratio: float
    content_change_ratio: float

    def __post_init__(self) -> None:
        if not isinstance(self.original_text, str):
            raise TypeError("original_text must be a string.")

        if not isinstance(self.candidate_text, str):
            raise TypeError("candidate_text must be a string.")

        if not isinstance(self.accepted, bool):
            raise TypeError("accepted must be a bool.")

        if not isinstance(self.reason, QwenRepairSafetyReason):
            raise TypeError("reason must be a QwenRepairSafetyReason.")

        object.__setattr__(
            self,
            "length_delta_ratio",
            _normalize_ratio(self.length_delta_ratio, "length_delta_ratio"),
        )
        object.__setattr__(
            self,
            "content_change_ratio",
            _normalize_ratio(self.content_change_ratio, "content_change_ratio"),
        )

    @property
    def selected_text(self) -> str:
        if self.accepted:
            return self.candidate_text

        return self.original_text


@dataclass(frozen=True, slots=True)
class QwenRepairSafetyPolicy:
    """Reject Qwen repairs that likely add or remove spoken content."""

    max_length_delta_ratio: float = DEFAULT_QWEN_REPAIR_MAX_LENGTH_DELTA_RATIO
    max_content_change_ratio: float = DEFAULT_QWEN_REPAIR_MAX_CONTENT_CHANGE_RATIO

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "max_length_delta_ratio",
            _normalize_ratio(
                self.max_length_delta_ratio,
                "max_length_delta_ratio",
            ),
        )
        object.__setattr__(
            self,
            "max_content_change_ratio",
            _normalize_ratio(
                self.max_content_change_ratio,
                "max_content_change_ratio",
            ),
        )

    def decide(
        self,
        original_text: str,
        candidate_text: str,
    ) -> QwenRepairSafetyDecision:
        if not isinstance(original_text, str):
            raise TypeError("original_text must be a string.")

        if not isinstance(candidate_text, str):
            raise TypeError("candidate_text must be a string.")

        original = original_text.strip()
        candidate = candidate_text.strip()
        if not candidate:
            return QwenRepairSafetyDecision(
                original_text=original,
                candidate_text=candidate,
                accepted=False,
                reason=QwenRepairSafetyReason.EMPTY_CANDIDATE,
                length_delta_ratio=1.0,
                content_change_ratio=1.0,
            )

        original_core = _normalize_text_for_safety(original)
        candidate_core = _normalize_text_for_safety(candidate)
        if original_core == candidate_core:
            return QwenRepairSafetyDecision(
                original_text=original,
                candidate_text=candidate,
                accepted=True,
                reason=QwenRepairSafetyReason.ACCEPTED,
                length_delta_ratio=0.0,
                content_change_ratio=0.0,
            )

        length_delta_ratio = _length_delta_ratio(original_core, candidate_core)
        content_change_ratio = _content_change_ratio(original_core, candidate_core)
        if length_delta_ratio > self.max_length_delta_ratio:
            return QwenRepairSafetyDecision(
                original_text=original,
                candidate_text=candidate,
                accepted=False,
                reason=QwenRepairSafetyReason.LENGTH_DELTA_EXCEEDED,
                length_delta_ratio=length_delta_ratio,
                content_change_ratio=content_change_ratio,
            )

        if content_change_ratio > self.max_content_change_ratio:
            return QwenRepairSafetyDecision(
                original_text=original,
                candidate_text=candidate,
                accepted=False,
                reason=QwenRepairSafetyReason.CONTENT_CHANGE_EXCEEDED,
                length_delta_ratio=length_delta_ratio,
                content_change_ratio=content_change_ratio,
            )

        return QwenRepairSafetyDecision(
            original_text=original,
            candidate_text=candidate,
            accepted=True,
            reason=QwenRepairSafetyReason.ACCEPTED,
            length_delta_ratio=length_delta_ratio,
            content_change_ratio=content_change_ratio,
        )


def _normalize_text_for_safety(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    characters: list[str] = []
    for character in normalized:
        category = unicodedata.category(character)
        if character.isspace() or category.startswith("P"):
            continue

        characters.append(character.lower())

    return "".join(characters)


def _length_delta_ratio(original: str, candidate: str) -> float:
    baseline = max(len(original), 1)
    return abs(len(candidate) - len(original)) / baseline


def _content_change_ratio(original: str, candidate: str) -> float:
    baseline = max(len(original), len(candidate), 1)
    changed_units = 0
    for tag, original_start, original_end, candidate_start, candidate_end in (
        SequenceMatcher(None, original, candidate).get_opcodes()
    ):
        if tag == "equal":
            continue

        original_length = original_end - original_start
        candidate_length = candidate_end - candidate_start
        changed_units += max(original_length, candidate_length)

    return changed_units / baseline


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
    safety_policy: QwenRepairSafetyPolicy = QwenRepairSafetyPolicy()
    _model: Any | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.model_path = Path(self.model_path)
        if not isinstance(self.safety_policy, QwenRepairSafetyPolicy):
            raise TypeError("safety_policy must be a QwenRepairSafetyPolicy.")

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
        candidate_text = self._clean_output(repaired_text)
        decision = self.safety_policy.decide(normalized_text, candidate_text)
        if not decision.accepted:
            _LOGGER.info(
                "Rejected unsafe Qwen repair: reason=%s length_delta_ratio=%.3f "
                "content_change_ratio=%.3f",
                decision.reason.value,
                decision.length_delta_ratio,
                decision.content_change_ratio,
            )

        return decision.selected_text

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
音声にない語を追加したり、音声にある語を削除したりしないでください。
言い換え、要約、説明、補足は禁止です。
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
    "DEFAULT_QWEN_REPAIR_MAX_CONTENT_CHANGE_RATIO",
    "DEFAULT_QWEN_REPAIR_MAX_LENGTH_DELTA_RATIO",
    "DEFAULT_QWEN_THREADS",
    "LlamaCppQwenRepairer",
    "PassthroughQwenRepairer",
    "QwenDependencyError",
    "QwenModelNotFoundError",
    "QwenRepairSafetyDecision",
    "QwenRepairSafetyPolicy",
    "QwenRepairSafetyReason",
]
