from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from jp_learning_platform.domain import (
    Document,
    PipelineContext,
    Segment,
    Sentence,
    Subtitle,
    TimeRange,
    Word,
)


def test_time_range_normalizes_values_and_reports_duration() -> None:
    time_range = TimeRange(start_seconds=1, end_seconds=3.5)

    assert time_range.start_seconds == 1.0
    assert time_range.end_seconds == 3.5
    assert time_range.duration_seconds == 2.5


def test_time_range_rejects_invalid_bounds() -> None:
    with pytest.raises(ValueError, match="end_seconds"):
        TimeRange(start_seconds=2.0, end_seconds=1.0)


def test_word_normalizes_text_and_confidence() -> None:
    word = Word(
        text="  nihongo  ",
        time_range=TimeRange(0.0, 0.5),
        confidence=0.9,
    )

    assert word.text == "nihongo"
    assert word.confidence == 0.9


def test_word_rejects_confidence_outside_probability_range() -> None:
    with pytest.raises(ValueError, match="confidence"):
        Word(text="nihongo", time_range=TimeRange(0.0, 0.5), confidence=1.1)


def test_sentence_requires_words_inside_time_range() -> None:
    word = Word(text="nihongo", time_range=TimeRange(1.0, 1.5))

    with pytest.raises(ValueError, match="words"):
        Sentence(text="nihongo", time_range=TimeRange(0.0, 1.0), words=(word,))


def test_segment_converts_sentence_collection_to_tuple() -> None:
    sentence = Sentence(text="nihongo desu", time_range=TimeRange(0.0, 2.0))
    segment = Segment(
        position=0,
        text="nihongo desu",
        time_range=TimeRange(0.0, 2.0),
        sentences=[sentence],
    )

    assert segment.sentences == (sentence,)


def test_subtitle_uses_one_based_index() -> None:
    with pytest.raises(ValueError, match="index"):
        Subtitle(index=0, text="nihongo desu", time_range=TimeRange(0.0, 2.0))


def test_document_converts_paths_and_collections() -> None:
    segment = Segment(
        position=0,
        text="nihongo desu",
        time_range=TimeRange(0.0, 2.0),
    )
    subtitle = Subtitle(
        index=1,
        text="nihongo desu",
        time_range=TimeRange(0.0, 2.0),
    )
    source_path = Path("audio/input.wav")
    document = Document(
        source_path=source_path,
        segments=[segment],
        subtitles=[subtitle],
    )

    assert document.source_path == source_path
    assert document.segments == (segment,)
    assert document.subtitles == (subtitle,)


def test_pipeline_context_identifies_document_and_workspace() -> None:
    document = Document(source_path=Path("audio/input.wav"))
    context = PipelineContext(
        run_id="run-001",
        document=document,
        working_directory=Path("work/run-001"),
    )

    assert context.run_id == "run-001"
    assert context.document == document
    assert context.working_directory == Path("work/run-001")


def test_domain_models_are_immutable() -> None:
    word = Word(text="nihongo", time_range=TimeRange(0.0, 0.5))

    with pytest.raises(FrozenInstanceError):
        word.text = "changed"
