from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from jp_learning_platform.domain import (
    Segment,
    Sentence,
    TimeRange,
    Word,
)
from jp_learning_platform.infrastructure.homophone_resolver import (
    BertHomophoneResolver,
    HomophoneLanguageModelCandidate,
    HomophoneTarget,
    _AnalyzedMorpheme,
    _unambiguous_confirmed_replacements,
)
from jp_learning_platform.workflow import (
    HomophoneCandidateScore,
    HomophoneResolutionDecision,
    HomophoneResolutionRequest,
)


_NOUN_POS = ("名詞", "普通名詞", "サ変可能", "*", "*", "*")
_GENERAL_NOUN_POS = ("名詞", "普通名詞", "一般", "*", "*", "*")
_PERSON_NAME_POS = ("名詞", "固有名詞", "人名", "姓", "*", "*")


@dataclass(slots=True)
class FakeAnalyzer:
    tokens: dict[str, tuple[tuple[str, str, tuple[str, ...]], ...]]
    single_tokens: dict[str, tuple[str, tuple[str, ...]]]

    def analyze(self, text: str) -> tuple[_AnalyzedMorpheme, ...]:
        result: list[_AnalyzedMorpheme] = []
        for surface, reading, part_of_speech in self.tokens.get(text, ()):
            start = text.index(surface)
            result.append(
                _AnalyzedMorpheme(
                    surface=surface,
                    reading=reading,
                    part_of_speech=part_of_speech,
                    start=start,
                    end=start + len(surface),
                )
            )
        return tuple(result)

    def analyze_single_token(self, text: str) -> _AnalyzedMorpheme | None:
        item = self.single_tokens.get(text)
        if item is None:
            return None

        reading, part_of_speech = item
        return _AnalyzedMorpheme(
            surface=text,
            reading=reading,
            part_of_speech=part_of_speech,
            start=0,
            end=len(text),
        )


@dataclass(slots=True)
class FakeCandidateGenerator:
    candidates: dict[str, tuple[HomophoneLanguageModelCandidate, ...]]
    scores: dict[str, float | None]
    seen_targets: list[HomophoneTarget] = field(default_factory=list)

    def candidates_for(
        self,
        sentence_text: str,
        target: HomophoneTarget,
    ) -> tuple[HomophoneLanguageModelCandidate, ...]:
        self.seen_targets.append(target)
        return self.candidates.get(target.text, ())

    def score_for(
        self,
        sentence_text: str,
        target: HomophoneTarget,
        replacement_text: str,
    ) -> float | None:
        return self.scores.get(replacement_text)


@dataclass(slots=True)
class PrefilterCandidateGenerator(FakeCandidateGenerator):
    original_scores: dict[str, float | None] = field(default_factory=dict)
    vocabulary_ranks: dict[str, float] = field(default_factory=dict)

    def lexical_candidates_for(self, target: HomophoneTarget) -> tuple[str, ...]:
        return tuple(candidate.text for candidate in self.candidates.get(target.text, ()))

    def original_scores_for(
        self,
        sentence_text: str,
        targets: tuple[HomophoneTarget, ...],
    ) -> tuple[float | None, ...]:
        return tuple(self.original_scores.get(target.text) for target in targets)

    def vocabulary_rank_for(self, text: str) -> float:
        return self.vocabulary_ranks.get(text, 0.0)


def _request(segment: Segment) -> HomophoneResolutionRequest:
    return HomophoneResolutionRequest(
        source_path=Path("lesson.mp3"),
        working_directory=Path("work"),
        run_id="run-001",
        segments=(segment,),
    )


def _segment(text: str) -> Segment:
    word = Word(
        text="懲戒",
        time_range=TimeRange(0.2, 0.8),
        confidence=0.7,
    )
    sentence = Sentence(
        text=text,
        time_range=TimeRange(0.0, 1.0),
        words=(word,),
    )
    return Segment(
        position=0,
        text=text,
        time_range=TimeRange(0.0, 1.0),
        sentences=(sentence,),
    )


def _resolver(
    candidates: tuple[HomophoneLanguageModelCandidate, ...],
    *,
    original_score: float | None = 0.1,
    candidate_score: float = 0.8,
    candidate_reading: str = "ちょうかい",
    candidate_pos: tuple[str, ...] = _GENERAL_NOUN_POS,
    require_original_score: bool = True,
) -> BertHomophoneResolver:
    analyzer = FakeAnalyzer(
        tokens={
            "2021年第2回日本語能力試験 懲戒N2": (
                ("懲戒", "ちょうかい", _NOUN_POS),
            ),
        },
        single_tokens={
            "聴解": (candidate_reading, candidate_pos),
            "試験": ("しけん", _GENERAL_NOUN_POS),
        },
    )
    generator = FakeCandidateGenerator(
        candidates={"懲戒": candidates},
        scores={"懲戒": original_score, "聴解": candidate_score},
    )
    return BertHomophoneResolver(
        candidate_generator=generator,
        analyzer=analyzer,
        require_original_score=require_original_score,
    )


def test_homophone_resolver_accepts_same_reading_candidate_with_better_context_score() -> None:
    resolver = _resolver(
        (HomophoneLanguageModelCandidate(text="聴解", score=0.8),),
        original_score=0.01,
    )

    result = resolver.resolve(
        _request(_segment("2021年第2回日本語能力試験 懲戒N2"))
    )

    assert result.segments[0].text == "2021年第2回日本語能力試験 聴解N2"
    assert result.segments[0].sentences[0].words[0].text == "聴解"
    decision = result.decisions[0]
    assert decision.original_text == "懲戒"
    assert decision.selected_text == "聴解"
    assert decision.reading == "ちょうかい"
    assert decision.accepted
    assert decision.reason == "accepted_same_reading_context"
    assert decision.original_score == 0.01
    assert decision.selected_score == 0.8


def test_homophone_resolver_propagates_strict_confirmation_within_document() -> None:
    texts = (
        "2021年第2回日本語能力試験 懲戒N2",
        "これからN2の懲戒試験を始めます",
        "これで懲戒試験を終わります",
    )
    analyzer = FakeAnalyzer(
        tokens={text: (("懲戒", "ちょうかい", _NOUN_POS),) for text in texts},
        single_tokens={"聴解": ("ちょうかい", _GENERAL_NOUN_POS)},
    )
    generator = FakeCandidateGenerator(
        candidates={
            "懲戒": (HomophoneLanguageModelCandidate("聴解", 0.8),),
        },
        scores={"懲戒": 0.001, "聴解": 0.8},
    )
    segments = tuple(
        Segment(
            position=index,
            text=text,
            time_range=TimeRange(index, index + 1),
            sentences=(
                Sentence(
                    text=text,
                    time_range=TimeRange(index, index + 1),
                    words=(
                        Word(
                            "懲戒",
                            TimeRange(index + 0.2, index + 0.8),
                            0.7 if index == 0 else 0.99,
                        ),
                    ),
                ),
            ),
        )
        for index, text in enumerate(texts)
    )

    result = BertHomophoneResolver(
        candidate_generator=generator,
        analyzer=analyzer,
    ).resolve(
        HomophoneResolutionRequest(
            source_path=Path("lesson.mp3"),
            working_directory=Path("work"),
            run_id="run-001",
            segments=segments,
        )
    )

    assert tuple("懲戒" not in segment.text for segment in result.segments) == (
        True,
        True,
        True,
    )
    assert [decision.reason for decision in result.decisions] == [
        "accepted_same_reading_context",
        "accepted_high_asr_confidence_with_strong_context",
        "accepted_high_asr_confidence_with_strong_context",
    ]


def test_document_confirmation_rejects_conflicting_mappings() -> None:
    def accepted(selected_text: str) -> HomophoneResolutionDecision:
        return HomophoneResolutionDecision(
            segment_position=0,
            sentence_index=0,
            original_text="回答",
            selected_text=selected_text,
            reading="かいとう",
            accepted=True,
            reason="accepted_same_reading_context",
            original_score=0.001,
            selected_score=0.8,
            candidates=(HomophoneCandidateScore(selected_text, "かいとう", 0.8),),
            target_start=0,
            target_end=2,
        )

    confirmed = _unambiguous_confirmed_replacements(
        (accepted("解答"), accepted("解糖"))
    )

    assert "回答" not in confirmed


def test_document_confirmation_requires_a_strong_seed_ratio() -> None:
    weak = HomophoneResolutionDecision(
        segment_position=0,
        sentence_index=0,
        original_text="回答",
        selected_text="解答",
        reading="かいとう",
        accepted=True,
        reason="accepted_same_reading_context",
        original_score=0.01,
        selected_score=0.5,
        candidates=(HomophoneCandidateScore("解答", "かいとう", 0.5),),
        target_start=0,
        target_end=2,
        score_ratio=50.0,
    )

    confirmed = _unambiguous_confirmed_replacements(
        (weak,),
        min_score_ratio=200.0,
    )

    assert confirmed == {}


def test_homophone_resolver_never_rewrites_person_name_without_external_evidence() -> None:
    text = "小野さんと話します"
    analyzer = FakeAnalyzer(
        tokens={text: (("小野", "おの", _PERSON_NAME_POS),)},
        single_tokens={"尾野": ("おの", _PERSON_NAME_POS)},
    )
    generator = FakeCandidateGenerator(
        candidates={
            "小野": (HomophoneLanguageModelCandidate("尾野", 0.9),),
        },
        scores={"小野": 0.000001, "尾野": 0.9},
    )
    word = Word("小野", TimeRange(0.0, 0.5), confidence=0.5)
    segment = Segment(
        position=0,
        text=text,
        time_range=TimeRange(0.0, 1.0),
        sentences=(Sentence(text, TimeRange(0.0, 1.0), words=(word,)),),
    )

    result = BertHomophoneResolver(
        candidate_generator=generator,
        analyzer=analyzer,
    ).resolve(_request(segment))

    assert result.segments[0].text == text
    assert not result.decisions[0].accepted
    assert result.decisions[0].reason == "person_name_requires_external_evidence"


def test_person_name_pos_does_not_block_a_place_name_component() -> None:
    text = "終点は川口湖です"
    analyzer = FakeAnalyzer(
        tokens={text: (("川口", "かわぐち", _PERSON_NAME_POS),)},
        single_tokens={"河口": ("かわぐち", _PERSON_NAME_POS)},
    )
    generator = FakeCandidateGenerator(
        candidates={
            "川口": (HomophoneLanguageModelCandidate("河口", 0.9),),
        },
        scores={"川口": 0.000001, "河口": 0.9},
    )
    word = Word("川口", TimeRange(0.0, 0.5), confidence=0.5)
    segment = Segment(
        position=0,
        text=text,
        time_range=TimeRange(0.0, 1.0),
        sentences=(Sentence(text, TimeRange(0.0, 1.0), words=(word,)),),
    )

    result = BertHomophoneResolver(
        candidate_generator=generator,
        analyzer=analyzer,
    ).resolve(_request(segment))

    assert result.segments[0].text == "終点は河口湖です"
    assert result.decisions[0].accepted


def test_document_confirmation_propagates_when_only_ratio_gate_failed() -> None:
    resolver = _resolver(())
    decision = HomophoneResolutionDecision(
        segment_position=356,
        sentence_index=0,
        original_text="懲戒",
        selected_text="懲戒",
        reading="ちょうかい",
        accepted=False,
        reason="candidate_score_ratio_too_low",
        original_score=0.000274,
        selected_score=0.000644,
        candidates=(HomophoneCandidateScore("聴解", "ちょうかい", 0.000644),),
        target_start=3,
        target_end=5,
    )

    propagated = resolver._propagate_decision(decision, {"懲戒": "聴解"})

    assert propagated.accepted
    assert propagated.selected_text == "聴解"
    assert propagated.reason == "accepted_document_consistency"


def test_document_confirmation_keeps_candidate_below_original() -> None:
    resolver = _resolver(())
    decision = HomophoneResolutionDecision(
        segment_position=356,
        sentence_index=0,
        original_text="懲戒",
        selected_text="懲戒",
        reading="ちょうかい",
        accepted=False,
        reason="candidate_score_ratio_too_low",
        original_score=0.0007,
        selected_score=0.0006,
        candidates=(HomophoneCandidateScore("聴解", "ちょうかい", 0.0006),),
        target_start=3,
        target_end=5,
    )

    propagated = resolver._propagate_decision(decision, {"懲戒": "聴解"})

    assert not propagated.accepted
    assert propagated.selected_text == "懲戒"


def test_homophone_resolver_rejects_different_reading_candidate() -> None:
    resolver = _resolver(
        (HomophoneLanguageModelCandidate(text="試験", score=0.9),),
        original_score=0.1,
    )

    result = resolver.resolve(
        _request(_segment("2021年第2回日本語能力試験 懲戒N2"))
    )

    assert result.segments[0].text == "2021年第2回日本語能力試験 懲戒N2"
    decision = result.decisions[0]
    assert not decision.accepted
    assert decision.reason == "no_same_reading_candidate"
    assert decision.candidates == ()


def test_homophone_resolver_rejects_candidate_that_is_not_better_than_original() -> None:
    resolver = _resolver(
        (HomophoneLanguageModelCandidate(text="聴解", score=0.2),),
        original_score=0.4,
        candidate_score=0.2,
    )

    result = resolver.resolve(
        _request(_segment("2021年第2回日本語能力試験 懲戒N2"))
    )

    assert result.segments[0].text == "2021年第2回日本語能力試験 懲戒N2"
    decision = result.decisions[0]
    assert not decision.accepted
    assert decision.reason == "candidate_not_better_than_original"
    assert decision.candidates[0].text == "聴解"


def test_homophone_resolver_requires_stronger_context_for_high_confidence_asr_word() -> None:
    resolver = _resolver(
        (HomophoneLanguageModelCandidate(text="聴解", score=0.3),),
        original_score=0.01,
    )
    segment = _segment("2021年第2回日本語能力試験 懲戒N2")
    sentence = segment.sentences[0]
    confident_word = Word(
        text="懲戒",
        time_range=sentence.words[0].time_range,
        confidence=0.99,
    )
    segment = Segment(
        position=segment.position,
        text=segment.text,
        time_range=segment.time_range,
        sentences=(
            Sentence(
                text=sentence.text,
                time_range=sentence.time_range,
                words=(confident_word,),
            ),
        ),
    )

    result = resolver.resolve(_request(segment))

    assert result.segments[0].text.endswith("懲戒N2")
    decision = result.decisions[0]
    assert decision.reason == "high_asr_confidence_requires_stronger_context"
    assert decision.asr_confidence == 0.99
    assert decision.score_ratio == pytest.approx(30.0)


def test_homophone_resolver_accepts_decisive_context_despite_high_asr_confidence() -> None:
    resolver = _resolver(
        (HomophoneLanguageModelCandidate(text="聴解", score=0.8),),
        original_score=0.001,
    )
    segment = _segment("2021年第2回日本語能力試験 懲戒N2")
    sentence = segment.sentences[0]
    segment = Segment(
        position=segment.position,
        text=segment.text,
        time_range=segment.time_range,
        sentences=(
            Sentence(
                text=sentence.text,
                time_range=sentence.time_range,
                words=(
                    Word(
                        "懲戒",
                        sentence.words[0].time_range,
                        confidence=0.99,
                    ),
                ),
            ),
        ),
    )

    result = resolver.resolve(_request(segment))

    assert result.segments[0].text.endswith("聴解N2")
    decision = result.decisions[0]
    assert decision.accepted
    assert decision.reason == "accepted_high_asr_confidence_with_strong_context"
    assert decision.score_ratio == pytest.approx(800.0)


def test_homophone_resolver_rejects_weak_contextual_ratio() -> None:
    resolver = _resolver(
        (HomophoneLanguageModelCandidate(text="聴解", score=0.15),),
        original_score=0.01,
    )

    result = resolver.resolve(
        _request(_segment("2021年第2回日本語能力試験 懲戒N2"))
    )

    assert result.segments[0].text.endswith("懲戒N2")
    assert result.decisions[0].reason == "candidate_score_ratio_too_low"


def test_homophone_resolver_always_enforces_minimum_candidate_score() -> None:
    resolver = _resolver(
        (HomophoneLanguageModelCandidate(text="聴解", score=0.00001),),
        original_score=0.00000001,
    )

    result = resolver.resolve(
        _request(_segment("2021年第2回日本語能力試験 懲戒N2"))
    )

    assert result.segments[0].text.endswith("懲戒N2")
    assert result.decisions[0].reason == "candidate_score_too_low"


def test_homophone_resolver_rejects_candidate_that_is_not_a_single_token() -> None:
    resolver = _resolver(
        (HomophoneLanguageModelCandidate(text="聴解N2", score=0.9),),
        original_score=0.1,
    )

    result = resolver.resolve(
        _request(_segment("2021年第2回日本語能力試験 懲戒N2"))
    )

    assert result.segments[0].text == "2021年第2回日本語能力試験 懲戒N2"
    assert result.decisions[0].reason == "no_same_reading_candidate"


def test_homophone_resolver_can_accept_when_original_score_is_unavailable_if_configured() -> None:
    resolver = _resolver(
        (HomophoneLanguageModelCandidate(text="聴解", score=0.8),),
        original_score=None,
        require_original_score=False,
    )

    result = resolver.resolve(
        _request(_segment("2021年第2回日本語能力試験 懲戒N2"))
    )

    assert result.segments[0].text == "2021年第2回日本語能力試験 聴解N2"
    assert result.decisions[0].reason == "accepted_same_reading_context"


def test_homophone_resolver_rejects_kana_only_replacement_for_kanji_word() -> None:
    analyzer = FakeAnalyzer(
        tokens={
            "手を挙げてください": (
                ("挙げ", "あげ", ("動詞", "一般", "*", "*", "*", "*")),
            ),
        },
        single_tokens={
            "あげ": ("あげ", ("動詞", "一般", "*", "*", "*", "*")),
        },
    )
    generator = FakeCandidateGenerator(
        candidates={
            "挙げ": (HomophoneLanguageModelCandidate(text="あげ", score=0.8),),
        },
        scores={"挙げ": 0.1, "あげ": 0.8},
    )
    resolver = BertHomophoneResolver(
        candidate_generator=generator,
        analyzer=analyzer,
    )
    word = Word(
        text="挙げ",
        time_range=TimeRange(0.2, 0.5),
        confidence=0.8,
    )
    sentence = Sentence(
        text="手を挙げてください",
        time_range=TimeRange(0.0, 1.0),
        words=(word,),
    )
    segment = Segment(
        position=0,
        text=sentence.text,
        time_range=sentence.time_range,
        sentences=(sentence,),
    )

    result = resolver.resolve(_request(segment))

    assert result.segments[0].text == "手を挙げてください"
    assert result.decisions[0].reason == "no_same_reading_candidate"


def test_homophone_resolver_rejects_kanji_replacement_for_kana_word() -> None:
    analyzer = FakeAnalyzer(
        tokens={
            "なかなか進まない": (
                ("なかなか", "なかなか", ("副詞", "*", "*", "*", "*", "*")),
            ),
        },
        single_tokens={
            "中中": ("なかなか", ("副詞", "*", "*", "*", "*", "*")),
        },
    )
    generator = FakeCandidateGenerator(
        candidates={
            "なかなか": (HomophoneLanguageModelCandidate(text="中中", score=0.8),),
        },
        scores={"なかなか": 0.001, "中中": 0.8},
    )
    word = Word("なかなか", TimeRange(0.0, 0.5), 0.2)
    sentence = Sentence("なかなか進まない", TimeRange(0.0, 1.0), (word,))
    segment = Segment(0, sentence.text, sentence.time_range, (sentence,))

    result = BertHomophoneResolver(
        candidate_generator=generator,
        analyzer=analyzer,
    ).resolve(_request(segment))

    assert result.segments[0].text == "なかなか進まない"
    assert result.decisions[0].reason == "no_same_reading_candidate"


def test_homophone_resolver_scores_at_most_three_suspicious_targets_per_sentence(
) -> None:
    surfaces = ("甲乙", "丙丁", "戊己", "庚辛")
    candidates = ("甲乙候", "丙丁候", "戊己候", "庚辛候")
    readings = ("こうおつ", "へいてい", "ぼき", "こうしん")
    analyzer = FakeAnalyzer(
        tokens={
            "".join(surfaces): tuple(
                (surface, reading, _GENERAL_NOUN_POS)
                for surface, reading in zip(surfaces, readings, strict=True)
            )
        },
        single_tokens={
            candidate: (reading, _GENERAL_NOUN_POS)
            for candidate, reading in zip(candidates, readings, strict=True)
        },
    )
    generator = PrefilterCandidateGenerator(
        candidates={
            surface: (HomophoneLanguageModelCandidate(candidate, 0.8),)
            for surface, candidate in zip(surfaces, candidates, strict=True)
        },
        scores={candidate: 0.8 for candidate in candidates},
        original_scores=dict(zip(surfaces, (0.9, 0.1, 0.2, 0.3), strict=True)),
        vocabulary_ranks=dict(zip(surfaces, (0.1, 0.2, 0.3, 0.4), strict=True)),
    )
    words = tuple(
        Word(
            text=surface,
            time_range=TimeRange(index, index + 0.5),
            confidence=0.9,
        )
        for index, surface in enumerate(surfaces)
    )
    sentence = Sentence(
        text="".join(surfaces),
        time_range=TimeRange(0.0, 4.0),
        words=words,
    )
    segment = Segment(
        position=0,
        text=sentence.text,
        time_range=sentence.time_range,
        sentences=(sentence,),
    )

    result = BertHomophoneResolver(
        candidate_generator=generator,
        analyzer=analyzer,
        max_targets_per_sentence=3,
    ).resolve(_request(segment))

    assert tuple(target.text for target in generator.seen_targets) == surfaces[1:]
    assert len(result.decisions) == 3
