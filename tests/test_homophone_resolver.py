from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

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
)
from jp_learning_platform.workflow import HomophoneResolutionRequest


_NOUN_POS = ("名詞", "普通名詞", "サ変可能", "*", "*", "*")
_GENERAL_NOUN_POS = ("名詞", "普通名詞", "一般", "*", "*", "*")


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
        original_score=0.1,
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
    assert decision.original_score == 0.1
    assert decision.selected_score == 0.8


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
