from __future__ import annotations

from pathlib import Path
from typing import Any

from jp_learning_platform.domain import Segment, Sentence, TimeRange, Word
from jp_learning_platform.infrastructure import (
    LlamaCppQwenRepairer,
    QwenRepairSafetyPolicy,
    QwenRepairSafetyReason,
)
from jp_learning_platform.workflow import QwenRepairRequest


class FakeLlamaCppQwenRepairer(LlamaCppQwenRepairer):
    __slots__ = ("generated_text",)

    def __init__(self, generated_text: str) -> None:
        super().__init__(model_path=Path("unused.gguf"))
        self.generated_text = generated_text

    def _load_model(self) -> Any:
        def model(*args: object, **kwargs: object) -> dict[str, object]:
            return {"choices": ({"text": self.generated_text},)}

        return model


def _segment(text: str) -> Segment:
    words = (
        Word(text="今日", time_range=TimeRange(0.0, 0.4), confidence=0.9),
        Word(text="日本語", time_range=TimeRange(0.5, 1.1), confidence=0.9),
        Word(text="勉強", time_range=TimeRange(1.2, 1.8), confidence=0.9),
    )
    sentence = Sentence(
        text=text,
        time_range=TimeRange(0.0, 2.0),
        words=words,
    )
    return Segment(
        position=0,
        text=text,
        time_range=TimeRange(0.0, 2.0),
        sentences=(sentence,),
    )


def _request(segment: Segment) -> QwenRepairRequest:
    return QwenRepairRequest(
        source_path=Path("audio.mp3"),
        working_directory=Path("work"),
        run_id="run-001",
        segments=(segment,),
    )


def test_qwen_repair_safety_accepts_punctuation_only_changes() -> None:
    decision = QwenRepairSafetyPolicy().decide(
        original_text="今日は日本語を勉強します",
        candidate_text="今日は日本語を勉強します。",
    )

    assert decision.accepted
    assert decision.reason is QwenRepairSafetyReason.ACCEPTED
    assert decision.selected_text == "今日は日本語を勉強します。"


def test_qwen_repair_safety_rejects_inserted_spoken_content() -> None:
    decision = QwenRepairSafetyPolicy().decide(
        original_text="今日は日本語を勉強します",
        candidate_text="今日は一緒に日本語を勉強します",
    )

    assert not decision.accepted
    assert decision.reason is QwenRepairSafetyReason.LENGTH_DELTA_EXCEEDED
    assert decision.selected_text == "今日は日本語を勉強します"


def test_qwen_repair_safety_rejects_deleted_spoken_content() -> None:
    decision = QwenRepairSafetyPolicy().decide(
        original_text="今日は日本語を勉強します",
        candidate_text="今日は勉強します",
    )

    assert not decision.accepted
    assert decision.reason is QwenRepairSafetyReason.LENGTH_DELTA_EXCEEDED
    assert decision.selected_text == "今日は日本語を勉強します"


def test_llama_qwen_repairer_falls_back_when_safety_rejects_output() -> None:
    segment = _segment("今日は日本語を勉強します")
    repairer = FakeLlamaCppQwenRepairer(
        generated_text="今日は一緒に日本語を勉強します"
    )

    result = repairer.repair(_request(segment))

    repaired_segment = result.segments[0]
    assert repaired_segment.text == "今日は日本語を勉強します"
    assert repaired_segment.time_range == segment.time_range
    assert repaired_segment.sentences[0].words == segment.sentences[0].words


def test_llama_qwen_repairer_accepts_low_risk_typo_repair() -> None:
    segment = _segment("今日は日本語を便強します")
    repairer = FakeLlamaCppQwenRepairer(generated_text="今日は日本語を勉強します")

    result = repairer.repair(_request(segment))

    assert result.segments[0].text == "今日は日本語を勉強します"
