from __future__ import annotations

from pathlib import Path

from jp_learning_platform.domain import (
    Segment,
    Sentence,
    SentenceBoundaryCandidate,
    TimeRange,
    Word,
)
from jp_learning_platform.infrastructure import (
    AcousticSentenceBoundaryResolver,
    WordGapSentenceBoundaryDetector,
)
from jp_learning_platform.workflow import (
    SentenceBoundaryDetectionRequest,
    SentenceBoundaryResolutionRequest,
)


def _segment(text: str) -> Segment:
    words = (
        Word(text="これ", time_range=TimeRange(3.86, 4.3)),
        Word(text="から", time_range=TimeRange(4.3, 4.56)),
        Word(text="音", time_range=TimeRange(4.56, 5.18)),
        Word(text="を", time_range=TimeRange(5.18, 5.44)),
        Word(text="聞", time_range=TimeRange(5.44, 5.54)),
        Word(text="いて", time_range=TimeRange(5.54, 5.76)),
        Word(text="ください", time_range=TimeRange(5.76, 6.16)),
        Word(text="音", time_range=TimeRange(6.81, 7.67)),
        Word(text="が", time_range=TimeRange(7.67, 7.89)),
        Word(text="よく", time_range=TimeRange(7.89, 8.19)),
        Word(text="聞こえない", time_range=TimeRange(8.19, 8.81)),
        Word(text="ときは", time_range=TimeRange(8.81, 9.23)),
        Word(text="手", time_range=TimeRange(9.23, 9.93)),
        Word(text="を", time_range=TimeRange(9.93, 10.17)),
        Word(text="挙げて", time_range=TimeRange(10.17, 10.43)),
        Word(text="ください", time_range=TimeRange(10.43, 10.79)),
    )
    sentence = Sentence(
        text=text,
        time_range=TimeRange(3.86, 10.79),
        words=words,
    )
    return Segment(
        position=0,
        text=text,
        time_range=TimeRange(3.86, 10.79),
        sentences=(sentence,),
    )


def _segment_from_words(
    text: str,
    words: tuple[Word, ...],
    position: int = 0,
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


def test_word_gap_detector_records_pause_candidate_after_word_boundary() -> None:
    segment = _segment(
        "これから音を聞いてください 音がよく聞こえないときは手を挙げてください"
    )

    result = WordGapSentenceBoundaryDetector(min_pause_seconds=0.5).detect(
        SentenceBoundaryDetectionRequest(
            source_path=Path("audio.mp3"),
            working_directory=Path("work"),
            run_id="run-001",
            segments=(segment,),
        )
    )

    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    assert candidate.segment_position == 0
    assert candidate.after_word_index == 6
    assert candidate.pause_time_range == TimeRange(6.16, 6.81)
    assert candidate.source == "word-gap"


def test_sentence_boundary_resolver_splits_repaired_text_at_acoustic_candidate() -> None:
    repaired_segment = _segment(
        "これから音を聞いてください。音がよく聞こえないときは手を挙げてください。"
    )
    detection = WordGapSentenceBoundaryDetector(min_pause_seconds=0.5).detect(
        SentenceBoundaryDetectionRequest(
            source_path=Path("audio.mp3"),
            working_directory=Path("work"),
            run_id="run-001",
            segments=(repaired_segment,),
        )
    )

    result = AcousticSentenceBoundaryResolver().resolve(
        SentenceBoundaryResolutionRequest(
            source_path=Path("audio.mp3"),
            working_directory=Path("work"),
            run_id="run-001",
            segments=(repaired_segment,),
            candidates=detection.candidates,
        )
    )

    sentences = result.segments[0].sentences
    assert tuple(sentence.text for sentence in sentences) == (
        "これから音を聞いてください。",
        "音がよく聞こえないときは手を挙げてください。",
    )
    assert sentences[0].time_range == TimeRange(3.86, 6.16)
    assert sentences[1].time_range == TimeRange(6.81, 10.79)


def test_sentence_boundary_resolver_maps_stale_candidate_index_by_time() -> None:
    words = (
        Word(
            text="これから音を聞いてください",
            time_range=TimeRange(3.86, 6.16),
        ),
        Word(
            text="音がよく聞こえないときは手を挙げてください",
            time_range=TimeRange(6.81, 10.79),
        ),
    )
    sentence = Sentence(
        text="これから音を聞いてください。音がよく聞こえないときは手を挙げてください。",
        time_range=TimeRange(3.86, 10.79),
        words=words,
    )
    repaired_segment = Segment(
        position=0,
        text=sentence.text,
        time_range=sentence.time_range,
        sentences=(sentence,),
    )
    stale_candidate = SentenceBoundaryCandidate(
        segment_position=0,
        after_word_index=99,
        boundary_time_seconds=6.485,
        pause_time_range=TimeRange(6.16, 6.81),
        acoustic_score=0.9,
        source="test",
    )

    result = AcousticSentenceBoundaryResolver().resolve(
        SentenceBoundaryResolutionRequest(
            source_path=Path("audio.mp3"),
            working_directory=Path("work"),
            run_id="run-001",
            segments=(repaired_segment,),
            candidates=(stale_candidate,),
        )
    )

    assert tuple(sentence.text for sentence in result.segments[0].sentences) == (
        "これから音を聞いてください。",
        "音がよく聞こえないときは手を挙げてください。",
    )


def test_sentence_boundary_resolver_allows_overlapping_word_timings_in_chunk() -> None:
    words = (
        Word(text="これ", time_range=TimeRange(4.2, 4.461)),
        Word(text="から", time_range=TimeRange(4.461, 5.102)),
        Word(text="ください", time_range=TimeRange(5.76, 6.16)),
        Word(text="。", time_range=TimeRange(6.16, 6.16)),
        Word(text="音", time_range=TimeRange(6.81, 7.67)),
        Word(text="とき", time_range=TimeRange(8.987, 9.268)),
        Word(text="は", time_range=TimeRange(9.07, 9.23)),
        Word(text="ください", time_range=TimeRange(10.59, 10.79)),
        Word(text="。", time_range=TimeRange(10.79, 10.79)),
    )
    sentence = Sentence(
        text="これからください。音ときはください。",
        time_range=TimeRange(4.2, 10.79),
        words=words,
    )
    segment = Segment(
        position=0,
        text=sentence.text,
        time_range=sentence.time_range,
        sentences=(sentence,),
    )
    candidate = SentenceBoundaryCandidate(
        segment_position=0,
        after_word_index=99,
        boundary_time_seconds=6.485,
        pause_time_range=TimeRange(6.16, 6.81),
        acoustic_score=0.9,
        source="test",
    )

    result = AcousticSentenceBoundaryResolver().resolve(
        SentenceBoundaryResolutionRequest(
            source_path=Path("audio.mp3"),
            working_directory=Path("work"),
            run_id="run-001",
            segments=(segment,),
            candidates=(candidate,),
        )
    )

    sentences = result.segments[0].sentences
    assert len(sentences) == 2
    assert sentences[1].time_range == TimeRange(6.81, 10.79)


def test_sentence_boundary_resolver_falls_back_to_word_text_without_punctuation() -> None:
    segment = _segment(
        "これから音を聞いてください音がよく聞こえないときは手を挙げてください"
    )
    detection = WordGapSentenceBoundaryDetector(min_pause_seconds=0.5).detect(
        SentenceBoundaryDetectionRequest(
            source_path=Path("audio.mp3"),
            working_directory=Path("work"),
            run_id="run-001",
            segments=(segment,),
        )
    )

    result = AcousticSentenceBoundaryResolver().resolve(
        SentenceBoundaryResolutionRequest(
            source_path=Path("audio.mp3"),
            working_directory=Path("work"),
            run_id="run-001",
            segments=(segment,),
            candidates=detection.candidates,
        )
    )

    assert tuple(sentence.text for sentence in result.segments[0].sentences) == (
        "これから音を聞いてください",
        "音がよく聞こえないときは手を挙げてください",
    )


def test_sentence_boundary_resolver_keeps_conditional_clause_with_main_clause() -> None:
    words = (
        Word(text="問題用紙", time_range=TimeRange(51.62, 52.1)),
        Word(text="を", time_range=TimeRange(52.1, 52.25)),
        Word(text="開け", time_range=TimeRange(52.25, 52.65)),
        Word(text="て", time_range=TimeRange(52.65, 52.75)),
        Word(text="ください", time_range=TimeRange(52.75, 53.1)),
        Word(text="問題用紙", time_range=TimeRange(54.79, 55.3)),
        Word(text="の", time_range=TimeRange(55.3, 55.42)),
        Word(text="ページ", time_range=TimeRange(55.42, 56.0)),
        Word(text="が", time_range=TimeRange(56.0, 56.16)),
        Word(text="ない", time_range=TimeRange(56.16, 56.7)),
        Word(text="とき", time_range=TimeRange(56.7, 57.07)),
        Word(text="は", time_range=TimeRange(57.07, 57.248)),
        Word(text="手", time_range=TimeRange(57.929, 58.1)),
        Word(text="を", time_range=TimeRange(58.1, 58.22)),
        Word(text="挙げ", time_range=TimeRange(58.22, 58.5)),
        Word(text="て", time_range=TimeRange(58.5, 58.59)),
        Word(text="ください", time_range=TimeRange(58.59, 58.73)),
    )
    segment = _segment_from_words(
        "問題用紙を開けてください 問題用紙のページがないときは 手を挙げてください",
        words,
    )
    candidates = (
        SentenceBoundaryCandidate(
            segment_position=0,
            after_word_index=4,
            boundary_time_seconds=53.945,
            pause_time_range=TimeRange(53.1, 54.79),
            acoustic_score=1.0,
            source="test",
        ),
        SentenceBoundaryCandidate(
            segment_position=0,
            after_word_index=11,
            boundary_time_seconds=57.5885,
            pause_time_range=TimeRange(57.248, 57.929),
            acoustic_score=1.0,
            source="test",
        ),
    )

    result = AcousticSentenceBoundaryResolver().resolve(
        SentenceBoundaryResolutionRequest(
            source_path=Path("audio.mp3"),
            working_directory=Path("work"),
            run_id="run-001",
            segments=(segment,),
            candidates=candidates,
        )
    )

    assert tuple(sentence.text for sentence in result.segments[0].sentences) == (
        "問題用紙を開けてください",
        "問題用紙のページがないときは手を挙げてください",
    )


def test_sentence_boundary_resolver_keeps_selection_range_with_predicate() -> None:
    words = (
        Word(text="問題用紙", time_range=TimeRange(83.213, 83.7)),
        Word(text="の", time_range=TimeRange(83.7, 83.85)),
        Word(text="1", time_range=TimeRange(83.85, 84.0)),
        Word(text="から", time_range=TimeRange(84.0, 84.2)),
        Word(text="4", time_range=TimeRange(84.2, 84.35)),
        Word(text="の", time_range=TimeRange(84.35, 84.5)),
        Word(text="中", time_range=TimeRange(84.5, 84.75)),
        Word(text="から", time_range=TimeRange(84.75, 85.2)),
        Word(text="、", time_range=TimeRange(85.2, 85.2)),
        Word(text="最も", time_range=TimeRange(86.643, 87.0)),
        Word(text="良い", time_range=TimeRange(87.0, 87.35)),
        Word(text="もの", time_range=TimeRange(87.35, 87.7)),
        Word(text="を", time_range=TimeRange(87.7, 87.85)),
        Word(text="一つ", time_range=TimeRange(87.85, 88.2)),
        Word(text="選ん", time_range=TimeRange(88.2, 88.7)),
        Word(text="で", time_range=TimeRange(88.7, 88.84)),
        Word(text="ください", time_range=TimeRange(88.84, 89.07)),
        Word(text="。", time_range=TimeRange(89.07, 89.07)),
    )
    text = "問題用紙の1から4の中から、 最も良いものを一つ選んでください。"
    segment = _segment_from_words(text, words)
    candidate = SentenceBoundaryCandidate(
        segment_position=0,
        after_word_index=8,
        boundary_time_seconds=85.9215,
        pause_time_range=TimeRange(85.2, 86.643),
        acoustic_score=1.0,
        source="test",
    )

    result = AcousticSentenceBoundaryResolver().resolve(
        SentenceBoundaryResolutionRequest(
            source_path=Path("audio.mp3"),
            working_directory=Path("work"),
            run_id="run-001",
            segments=(segment,),
            candidates=(candidate,),
        )
    )

    assert tuple(sentence.text for sentence in result.segments[0].sentences) == (
        text,
    )


def test_sentence_boundary_resolver_keeps_connection_expression_unsplit() -> None:
    words = (
        Word(text="問題", time_range=TimeRange(66.3, 66.7)),
        Word(text="1", time_range=TimeRange(66.7, 66.9)),
        Word(text="では", time_range=TimeRange(66.9, 67.2)),
        Word(text="まず", time_range=TimeRange(67.2, 67.6)),
        Word(text="質問", time_range=TimeRange(67.6, 68.1)),
        Word(text="を", time_range=TimeRange(68.1, 68.24)),
        Word(text="聞いて", time_range=TimeRange(68.24, 68.65)),
        Word(text="ください", time_range=TimeRange(68.65, 69.0)),
        Word(text="それ", time_range=TimeRange(69.0, 69.25)),
        Word(text="から", time_range=TimeRange(70.0, 70.28)),
        Word(text="話", time_range=TimeRange(70.28, 70.7)),
        Word(text="を", time_range=TimeRange(70.7, 70.82)),
        Word(text="聞いて", time_range=TimeRange(70.82, 71.3)),
    )
    text = "問題1ではまず質問を聞いてくださいそれから話を聞いて"
    segment = _segment_from_words(text, words)
    candidate = SentenceBoundaryCandidate(
        segment_position=0,
        after_word_index=8,
        boundary_time_seconds=69.625,
        pause_time_range=TimeRange(69.25, 70.0),
        acoustic_score=1.0,
        source="test",
    )

    result = AcousticSentenceBoundaryResolver().resolve(
        SentenceBoundaryResolutionRequest(
            source_path=Path("audio.mp3"),
            working_directory=Path("work"),
            run_id="run-001",
            segments=(segment,),
            candidates=(candidate,),
        )
    )

    assert tuple(sentence.text for sentence in result.segments[0].sentences) == (
        text,
    )
