from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import unicodedata

from jp_learning_platform.domain import Segment, Sentence, TimeRange, Word
from jp_learning_platform.infrastructure import (
    DisabledQwenRepairer,
    LlamaCppQwenRepairer,
    PassthroughQwenRepairer,
    QwenRepairSafetyPolicy,
    QwenRepairSafetyReason,
)
from jp_learning_platform.workflow import QwenRepairRequest


class FakeLlamaCppQwenRepairer(LlamaCppQwenRepairer):
    __slots__ = ("generated_texts", "prompts")

    def __init__(self, generated_text: str | tuple[str, ...]) -> None:
        super().__init__(model_path=Path("unused.gguf"))
        self.generated_texts = (
            (generated_text,)
            if isinstance(generated_text, str)
            else tuple(generated_text)
        )
        self.prompts: list[str] = []

    def _load_model(self) -> Any:
        def model(*args: object, **kwargs: object) -> dict[str, object]:
            self.prompts.append(str(args[0]))
            index = min(len(self.prompts) - 1, len(self.generated_texts) - 1)
            return {"choices": ({"text": self.generated_texts[index]},)}

        return model


def _json_response(payload: dict[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _segment(text: str, position: int = 0) -> Segment:
    words = (
        Word(
            text="今日",
            time_range=TimeRange(0.0, 0.4),
            confidence=0.9,
            speaker_id="speaker-1",
        ),
        Word(
            text="日本語",
            time_range=TimeRange(0.5, 1.1),
            confidence=0.9,
            speaker_id="speaker-1",
        ),
        Word(
            text="勉強",
            time_range=TimeRange(1.2, 1.8),
            confidence=0.9,
            speaker_id="speaker-1",
        ),
    )
    sentence = Sentence(
        text=text,
        time_range=TimeRange(0.0, 2.0),
        words=words,
        speaker_id="speaker-1",
    )
    return Segment(
        position=position,
        text=text,
        time_range=TimeRange(0.0, 2.0),
        sentences=(sentence,),
        speaker_id="speaker-1",
    )


def _request(segment: Segment) -> QwenRepairRequest:
    return _request_for_segments((segment,))


def _request_for_segments(segments: tuple[Segment, ...]) -> QwenRepairRequest:
    return QwenRepairRequest(
        source_path=Path("audio.mp3"),
        working_directory=Path("work"),
        run_id="run-001",
        segments=segments,
    )


def _timed_segment(
    text: str,
    position: int,
    words: tuple[Word, ...],
) -> Segment:
    time_range = TimeRange(
        min(word.time_range.start_seconds for word in words),
        max(word.time_range.end_seconds for word in words),
    )
    sentence = Sentence(text=text, time_range=time_range, words=words)
    return Segment(
        position=position,
        text=text,
        time_range=time_range,
        sentences=(sentence,),
    )


def _content_core(text: str) -> str:
    return "".join(
        character
        for character in text
        if not character.isspace()
        and not unicodedata.category(character).startswith("P")
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


def test_qwen_repair_safety_rejects_short_meaningful_deletion() -> None:
    decision = QwenRepairSafetyPolicy().decide(
        original_text="2021年第2回日本語能力試験 懲戒N2",
        candidate_text="2021年第2回日本語能力試験 N2",
    )

    assert not decision.accepted
    assert decision.reason is QwenRepairSafetyReason.MEANINGFUL_CONTENT_DELETED
    assert decision.selected_text == "2021年第2回日本語能力試験 懲戒N2"


def test_qwen_repair_safety_accepts_phonetic_replacement_candidate() -> None:
    decision = QwenRepairSafetyPolicy().decide(
        original_text="2021年第2回日本語能力試験 懲戒N2",
        candidate_text="2021年第2回日本語能力試験 聴解N2",
    )

    assert decision.accepted
    assert decision.reason is QwenRepairSafetyReason.ACCEPTED
    assert decision.selected_text == "2021年第2回日本語能力試験 聴解N2"


def test_qwen_repair_safety_rejects_short_non_phonetic_replacement() -> None:
    decision = QwenRepairSafetyPolicy().decide(
        original_text="明日は雨です",
        candidate_text="明日は雪です",
    )

    assert not decision.accepted
    assert decision.reason is QwenRepairSafetyReason.PARAPHRASE_REWRITE
    assert decision.selected_text == "明日は雨です"


def test_qwen_repair_safety_rejects_alphanumeric_information_rewrite() -> None:
    decision = QwenRepairSafetyPolicy().decide(
        original_text="これから n 2の試験を始めます",
        candidate_text="これから 今の試験を始めます",
    )

    assert not decision.accepted
    assert decision.reason is QwenRepairSafetyReason.PROTECTED_INFORMATION_REWRITE
    assert decision.selected_text == "これから n 2の試験を始めます"


def test_qwen_repair_safety_rejects_paraphrased_expression() -> None:
    decision = QwenRepairSafetyPolicy().decide(
        original_text="問題がよく見えないときも手を挙げてください いつでもいいです",
        candidate_text="問題がよく見えないときも手を挙げてください。いつでも構いません。",
    )

    assert not decision.accepted
    assert decision.reason is QwenRepairSafetyReason.PARAPHRASE_REWRITE
    assert (
        decision.selected_text
        == "問題がよく見えないときも手を挙げてください いつでもいいです"
    )


def test_qwen_repair_safety_rejects_non_phonetic_synonym_rewrite() -> None:
    decision = QwenRepairSafetyPolicy().decide(
        original_text="問題1ではまず質問を聞いてくださいそれでは始めます",
        candidate_text="問題1ではまず質問を聞いてくださいそれでは開始します。",
    )

    assert not decision.accepted
    assert decision.reason is QwenRepairSafetyReason.PARAPHRASE_REWRITE
    assert decision.selected_text == "問題1ではまず質問を聞いてくださいそれでは始めます"


def test_qwen_repair_safety_rejects_polite_paraphrase() -> None:
    decision = QwenRepairSafetyPolicy().decide(
        original_text="問題がよく見えないときも手を挙げてくださいはい、いいですよ。",
        candidate_text="問題がよく見えないときも手を挙げてくださいはい、大丈夫です。",
    )

    assert not decision.accepted
    assert decision.reason is QwenRepairSafetyReason.PARAPHRASE_REWRITE
    assert decision.selected_text == "問題がよく見えないときも手を挙げてくださいはい、いいですよ。"


def test_llama_qwen_repairer_falls_back_when_safety_rejects_output() -> None:
    segment = _segment("今日は日本語を勉強します")
    generated_text = _json_response(
        {
            "edits": [
                {
                    "type": "replace",
                    "from": "日本語",
                    "to": "一緒に日本語",
                    "reason": "model_candidate",
                    "confidence": 0.7,
                }
            ],
            "punctuation": [],
        }
    )
    repairer = FakeLlamaCppQwenRepairer(generated_text=generated_text)

    result = repairer.repair(_request(segment))

    repaired_segment = result.segments[0]
    assert repaired_segment.text == "今日は日本語を勉強します"
    assert repaired_segment.time_range == segment.time_range
    assert repaired_segment.sentences[0].words == segment.sentences[0].words
    decision = result.decisions[0]
    assert decision.original_text == "今日は日本語を勉強します"
    assert decision.raw_text == generated_text
    assert decision.candidate_text == "今日は一緒に日本語を勉強します"
    assert decision.selected_text == "今日は日本語を勉強します"
    assert not decision.accepted
    assert decision.reason == QwenRepairSafetyReason.LENGTH_DELTA_EXCEEDED.value


def test_disabled_qwen_repairer_keeps_segments_exactly_unchanged() -> None:
    previous_segment = _timed_segment(
        text="いつでもいいです",
        position=0,
        words=(
            Word(text="いつ", time_range=TimeRange(65.359, 65.619)),
            Word(text="でも", time_range=TimeRange(65.619, 65.879)),
            Word(text="いい", time_range=TimeRange(65.879, 66.12)),
            Word(text="です", time_range=TimeRange(66.12, 66.18)),
        ),
    )
    current_segment = _timed_segment(
        text="です問題1問題1ではまず質問を聞いてください",
        position=1,
        words=(
            Word(text="です", time_range=TimeRange(66.18, 66.3)),
            Word(text="問題", time_range=TimeRange(66.3, 67.26)),
            Word(text="1", time_range=TimeRange(73.89, 74.51)),
            Word(text="問題", time_range=TimeRange(75.806, 76.286)),
        ),
    )

    result = DisabledQwenRepairer().repair(
        _request_for_segments((previous_segment, current_segment))
    )

    assert result.segments == (previous_segment, current_segment)
    assert tuple(decision.reason for decision in result.decisions) == (
        QwenRepairSafetyReason.DISABLED.value,
        QwenRepairSafetyReason.DISABLED.value,
    )
    assert all(decision.raw_text == "" for decision in result.decisions)
    assert all(decision.accepted for decision in result.decisions)


def test_passthrough_qwen_repairer_removes_leading_boundary_duplicate() -> None:
    previous_segment = _timed_segment(
        text="いつでもいいです",
        position=0,
        words=(
            Word(text="いつ", time_range=TimeRange(65.359, 65.619)),
            Word(text="でも", time_range=TimeRange(65.619, 65.879)),
            Word(text="いい", time_range=TimeRange(65.879, 66.12)),
            Word(text="です", time_range=TimeRange(66.12, 66.18)),
        ),
    )
    current_segment = _timed_segment(
        text="です問題1問題1ではまず質問を聞いてください",
        position=1,
        words=(
            Word(text="です", time_range=TimeRange(66.18, 66.3)),
            Word(text="問題", time_range=TimeRange(66.3, 67.26)),
            Word(text="1", time_range=TimeRange(73.89, 74.51)),
            Word(text="問題", time_range=TimeRange(75.806, 76.286)),
        ),
    )

    result = PassthroughQwenRepairer().repair(
        _request_for_segments((previous_segment, current_segment))
    )

    cleaned_segment = result.segments[1]
    assert cleaned_segment.text == "問題1問題1ではまず質問を聞いてください"
    assert cleaned_segment.time_range == TimeRange(66.3, 76.286)
    assert cleaned_segment.sentences[0].text == "問題1問題1ではまず質問を聞いてください"
    assert tuple(word.text for word in cleaned_segment.sentences[0].words) == (
        "問題",
        "1",
        "問題",
    )


def test_passthrough_qwen_repairer_keeps_repeated_content_word() -> None:
    previous_segment = _timed_segment(
        text="問題",
        position=0,
        words=(Word(text="問題", time_range=TimeRange(1.0, 1.5)),),
    )
    current_segment = _timed_segment(
        text="問題1ではまず質問を聞いてください",
        position=1,
        words=(
            Word(text="問題", time_range=TimeRange(1.5, 2.0)),
            Word(text="1", time_range=TimeRange(2.0, 2.2)),
        ),
    )

    result = PassthroughQwenRepairer().repair(
        _request_for_segments((previous_segment, current_segment))
    )

    assert result.segments[1] == current_segment


def test_passthrough_qwen_repairer_keeps_distant_function_word_repeat() -> None:
    previous_segment = _timed_segment(
        text="いいです",
        position=0,
        words=(
            Word(text="いい", time_range=TimeRange(1.0, 1.4)),
            Word(text="です", time_range=TimeRange(1.4, 1.5)),
        ),
    )
    current_segment = _timed_segment(
        text="です問題1ではまず質問を聞いてください",
        position=1,
        words=(
            Word(text="です", time_range=TimeRange(2.0, 2.1)),
            Word(text="問題", time_range=TimeRange(2.1, 2.5)),
        ),
    )

    result = PassthroughQwenRepairer().repair(
        _request_for_segments((previous_segment, current_segment))
    )

    assert result.segments[1] == current_segment


def test_llama_qwen_repairer_falls_back_when_model_paraphrases() -> None:
    segment = _segment("問題がよく見えないときも手を挙げてください いつでもいいです")
    generated_text = _json_response(
        {
            "edits": [
                {
                    "type": "replace",
                    "from": "いいです",
                    "to": "構いません",
                    "reason": "model_candidate",
                    "confidence": 0.8,
                }
            ],
            "punctuation": [
                {"after": "手を挙げてください", "mark": "。"},
                {"after": "いつでもいいです", "mark": "。"},
            ],
        }
    )
    repairer = FakeLlamaCppQwenRepairer(generated_text=generated_text)

    result = repairer.repair(_request(segment))

    assert (
        result.segments[0].text
        == "問題がよく見えないときも手を挙げてください。いつでもいいです。"
    )
    decision = result.decisions[0]
    assert decision.raw_text == generated_text
    assert decision.candidate_text == (
        "問題がよく見えないときも手を挙げてください。いつでもいいです。"
    )
    assert decision.selected_text == (
        "問題がよく見えないときも手を挙げてください。いつでもいいです。"
    )
    assert decision.accepted
    assert decision.reason == QwenRepairSafetyReason.PUNCTUATION_FALLBACK.value


def test_llama_qwen_repairer_rejects_terminal_punctuation_after_incomplete_clause() -> None:
    segment = _segment("音がよく聞こえないときは手を挙げてください")
    generated_text = _json_response(
        {
            "edits": [],
            "punctuation": [
                {"after": "音がよく聞こえないときは", "mark": "。"},
                {"after": "手を挙げてください", "mark": "。"},
            ],
        }
    )
    repairer = FakeLlamaCppQwenRepairer(generated_text=generated_text)

    result = repairer.repair(_request(segment))

    assert result.segments[0].text == "音がよく聞こえないときは手を挙げてください。"
    decision = result.decisions[0]
    assert decision.raw_text == generated_text
    assert decision.candidate_text == "音がよく聞こえないときは手を挙げてください。"
    assert decision.selected_text == "音がよく聞こえないときは手を挙げてください。"
    assert decision.accepted
    assert decision.reason == QwenRepairSafetyReason.PUNCTUATION_FALLBACK.value


def test_llama_qwen_repairer_extracts_only_safe_punctuation_from_rejected_candidate() -> None:
    original_text = "今日は日本語を勉強します明日は漢字を練習します"
    segment = _segment(original_text)
    generated_text = _json_response(
        {
            "edits": [
                {
                    "type": "replace",
                    "from": "漢字を練習",
                    "to": "練習",
                    "reason": "model_candidate",
                    "confidence": 0.6,
                }
            ],
            "punctuation": [
                {"after": "今日は日本語を勉強します", "mark": "。"},
                {"after": "明日は漢字を練習します", "mark": "。"},
            ],
        }
    )
    repairer = FakeLlamaCppQwenRepairer(generated_text=generated_text)

    result = repairer.repair(_request(segment))

    selected_text = result.segments[0].text
    assert _content_core(selected_text) == _content_core(original_text)
    assert "漢字を" in selected_text
    assert "追加" not in selected_text
    assert selected_text.startswith("今日は日本語を勉強します。")
    decision = result.decisions[0]
    assert decision.raw_text == generated_text
    assert decision.candidate_text == (
        "今日は日本語を勉強します。"
        "明日は漢字を練習します。"
    )
    assert decision.selected_text == selected_text
    assert decision.accepted
    assert decision.reason == QwenRepairSafetyReason.PUNCTUATION_FALLBACK.value


def test_llama_qwen_repairer_does_not_repair_deleted_term_without_model_candidate() -> None:
    segment = _segment("2021年第2回日本語能力試験 懲戒N2")
    generated_text = _json_response(
        {
            "edits": [
                {
                    "type": "delete",
                    "from": "懲戒",
                    "to": "",
                    "reason": "model_candidate",
                    "confidence": 0.5,
                }
            ],
            "punctuation": [],
        }
    )
    repairer = FakeLlamaCppQwenRepairer(generated_text=generated_text)

    result = repairer.repair(_request(segment))

    assert result.segments[0].text == "2021年第2回日本語能力試験 懲戒N2"
    decision = result.decisions[0]
    assert decision.raw_text == generated_text
    assert decision.candidate_text == "2021年第2回日本語能力試験 懲戒N2"
    assert decision.selected_text == "2021年第2回日本語能力試験 懲戒N2"
    assert not decision.accepted
    assert decision.reason == QwenRepairSafetyReason.INVALID_EDIT_CANDIDATE.value


def test_llama_qwen_repairer_accepts_model_proposed_phonetic_replacement() -> None:
    segment = _segment("2021年第2回日本語能力試験 懲戒N2")
    generated_text = _json_response(
        {
            "edits": [
                {
                    "type": "replace",
                    "from": "懲戒",
                    "from_reading": "チョウカイ",
                    "candidates": [
                        {
                            "to": "試験",
                            "to_reading": "シケン",
                            "reason": "model_candidate",
                            "confidence": 0.9,
                        },
                        {
                            "to": "聴解",
                            "to_reading": "チョウカイ",
                            "reason": "same_reading_candidate",
                            "confidence": 0.86,
                        },
                    ],
                }
            ],
            "punctuation": [],
        }
    )
    repairer = FakeLlamaCppQwenRepairer(generated_text=generated_text)

    result = repairer.repair(_request(segment))

    assert result.segments[0].text == "2021年第2回日本語能力試験 聴解N2"
    decision = result.decisions[0]
    assert decision.raw_text == generated_text
    assert decision.candidate_text == "2021年第2回日本語能力試験 聴解N2"
    assert decision.selected_text == "2021年第2回日本語能力試験 聴解N2"
    assert decision.accepted
    assert decision.reason == QwenRepairSafetyReason.ACCEPTED.value


def test_llama_qwen_repairer_uses_semantic_retry_after_bad_candidates() -> None:
    segment = _segment("2021年第2回日本語能力試験 懲戒N2")
    first_generated_text = _json_response(
        {
            "edits": [
                {
                    "type": "replace",
                    "from": "懲戒N2",
                    "from_reading": "テイコウエヌツー",
                    "candidates": [
                        {
                            "to": "平成N2",
                            "to_reading": "ヘイセイエヌツー",
                            "reason": "model_candidate",
                            "confidence": 0.9,
                        },
                        {
                            "to": "改正N2",
                            "to_reading": "カイセイエヌツー",
                            "reason": "model_candidate",
                            "confidence": 0.8,
                        },
                    ],
                }
            ],
            "punctuation": [],
        }
    )
    retry_generated_text = _json_response(
        {
            "edits": [
                {
                    "type": "replace",
                    "from": "懲戒",
                    "from_reading": "チョウカイ",
                    "candidates": [
                        {
                            "to": "聴解",
                            "to_reading": "チョウカイ",
                            "reason": "semantic_context_same_reading",
                            "confidence": 0.88,
                        }
                    ],
                }
            ],
            "punctuation": [],
        }
    )
    repairer = FakeLlamaCppQwenRepairer(
        generated_text=(first_generated_text, retry_generated_text)
    )

    result = repairer.repair(_request(segment))

    assert result.segments[0].text == "2021年第2回日本語能力試験 聴解N2"
    decision = result.decisions[0]
    assert decision.raw_text == retry_generated_text
    assert decision.candidate_text == "2021年第2回日本語能力試験 聴解N2"
    assert decision.selected_text == "2021年第2回日本語能力試験 聴解N2"
    assert decision.accepted
    assert decision.reason == QwenRepairSafetyReason.ACCEPTED.value
    assert len(repairer.prompts) == 2
    assert "日本語ASR修正レビューAI" in repairer.prompts[1]


def test_llama_qwen_repairer_keeps_original_when_model_deletes_meaningful_term() -> None:
    segment = _segment("懲戒処分を受けました")
    generated_text = _json_response(
        {
            "edits": [
                {
                    "type": "delete",
                    "from": "懲戒",
                    "to": "",
                    "reason": "model_candidate",
                    "confidence": 0.5,
                }
            ],
            "punctuation": [],
        }
    )
    repairer = FakeLlamaCppQwenRepairer(generated_text=generated_text)

    result = repairer.repair(_request(segment))

    assert result.segments[0].text == "懲戒処分を受けました"
    decision = result.decisions[0]
    assert decision.raw_text == generated_text
    assert decision.candidate_text == "懲戒処分を受けました"
    assert decision.selected_text == "懲戒処分を受けました"
    assert not decision.accepted
    assert decision.reason == QwenRepairSafetyReason.INVALID_EDIT_CANDIDATE.value


def test_llama_qwen_repairer_accepts_low_risk_typo_repair() -> None:
    segment = _segment("今日は日本語を便強します")
    generated_text = _json_response(
        {
            "edits": [
                {
                    "type": "replace",
                    "from": "便強",
                    "to": "勉強",
                    "reason": "same_reading_candidate",
                    "confidence": 0.9,
                }
            ],
            "punctuation": [],
        }
    )
    repairer = FakeLlamaCppQwenRepairer(generated_text=generated_text)

    result = repairer.repair(_request(segment))

    assert result.segments[0].text == "今日は日本語を勉強します"
    assert result.segments[0].speaker_id == "speaker-1"
    assert result.segments[0].sentences[0].speaker_id == "speaker-1"
    decision = result.decisions[0]
    assert decision.original_text == "今日は日本語を便強します"
    assert decision.raw_text == generated_text
    assert decision.candidate_text == "今日は日本語を勉強します"
    assert decision.selected_text == "今日は日本語を勉強します"
    assert decision.accepted
    assert decision.reason == QwenRepairSafetyReason.ACCEPTED.value


def test_llama_qwen_repairer_prompt_omits_neighbor_segment_text() -> None:
    previous_segment = _segment("前だけにある字幕です", position=0)
    current_segment = _segment("現在だけにある字幕です", position=1)
    next_segment = _segment("次だけにある字幕です", position=2)
    repairer = FakeLlamaCppQwenRepairer(
        generated_text=_json_response({"edits": [], "punctuation": []})
    )

    repairer.repair(
        _request_for_segments((previous_segment, current_segment, next_segment))
    )

    assert len(repairer.prompts) == 3
    current_prompt = repairer.prompts[1]
    assert "現在だけにある字幕です" in current_prompt
    assert "前だけにある字幕です" not in current_prompt
    assert "次だけにある字幕です" not in current_prompt
    assert "PREV:" not in current_prompt
    assert "NEXT:" not in current_prompt
    assert "CURRENT:" not in current_prompt
    assert "削除してはいけません" in current_prompt
    assert "同音または非常に近い発音" in current_prompt


def test_llama_qwen_repairer_prompt_does_not_include_term_specific_repair_hint() -> None:
    segment = _segment("2021年第2回日本語能力試験 懲戒N2")
    repairer = FakeLlamaCppQwenRepairer(
        generated_text=_json_response(
            {
                "edits": [
                    {
                        "type": "replace",
                        "from": "懲戒",
                        "to": "聴解",
                        "reason": "same_reading_candidate",
                        "confidence": 0.86,
                    }
                ],
                "punctuation": [],
            }
        )
    )

    repairer.repair(_request(segment))

    prompt = repairer.prompts[0]
    assert "修正ヒント" not in prompt
    assert "懲戒" in prompt
    assert "聴解" not in prompt
    assert "前後セグメント原文: omitted" in prompt
    assert '"edits"' in prompt
    assert '"candidates"' in prompt
    assert '"from_reading"' in prompt
    assert '"to_reading"' in prompt
    assert "修正済み全文は出力しないでください" in prompt


def test_llama_qwen_repairer_records_marker_like_output_without_stripping() -> None:
    segment = _segment("今日は日本語を勉強します")
    repairer = FakeLlamaCppQwenRepairer(
        generated_text="今日は日本語を勉強します NEXT: 余計な出力"
    )

    result = repairer.repair(_request(segment))

    decision = result.decisions[0]
    assert decision.raw_text == "今日は日本語を勉強します NEXT: 余計な出力"
    assert decision.candidate_text == "今日は日本語を勉強します NEXT: 余計な出力"
    assert decision.selected_text == "今日は日本語を勉強します"
    assert not decision.accepted
    assert decision.reason == QwenRepairSafetyReason.INVALID_EDIT_CANDIDATE.value
