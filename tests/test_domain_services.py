from __future__ import annotations

from pathlib import Path

import pytest

from jp_learning_platform.domain import (
    Document,
    DocumentRepository,
    DocumentValidator,
    DomainModelFactory,
    DomainValidationError,
    ValidationCode,
    ValidationIssue,
    LearningWord,
    Sentence,
    Segment,
    TimeRange,
    Word,
)


class InMemoryDocumentRepository:
    def __init__(self) -> None:
        self._documents: dict[Path, Document] = {}

    def save(self, document: Document) -> None:
        self._documents[document.source_path] = document

    def get(self, source_path: Path) -> Document | None:
        return self._documents.get(source_path)


def test_factory_builds_document_graph_from_primitive_values() -> None:
    factory = DomainModelFactory()
    word = factory.create_word("nihongo", 0.0, 0.5, confidence=0.8)
    sentence = factory.create_sentence("nihongo desu", 0.0, 1.0, words=(word,))
    segment = factory.create_segment(
        position=0,
        text="nihongo desu",
        start_seconds=0.0,
        end_seconds=1.0,
        sentences=(sentence,),
    )
    subtitle = factory.create_subtitle(1, "nihongo desu", 0.0, 1.0)
    document = factory.create_document(
        source_path=Path("audio/input.wav"),
        segments=(segment,),
        subtitles=(subtitle,),
    )

    assert document.segments == (segment,)
    assert document.subtitles == (subtitle,)
    assert document.segments[0].sentences[0].words == (word,)


def test_validator_accepts_consistent_document() -> None:
    factory = DomainModelFactory()
    document = factory.create_document(
        source_path=Path("audio/input.wav"),
        segments=(
            factory.create_segment(0, "first", 0.0, 1.0),
            factory.create_segment(1, "second", 1.0, 2.0),
        ),
        subtitles=(
            factory.create_subtitle(1, "first", 0.0, 1.0),
            factory.create_subtitle(2, "second", 1.0, 2.0),
        ),
    )

    result = DocumentValidator().validate(document)

    assert result.is_valid
    assert result.issues == ()


def test_validator_reports_segment_position_gaps() -> None:
    factory = DomainModelFactory()
    document = factory.create_document(
        source_path=Path("audio/input.wav"),
        segments=(
            factory.create_segment(0, "first", 0.0, 1.0),
            factory.create_segment(2, "third", 1.0, 2.0),
        ),
    )

    result = DocumentValidator().validate(document)

    assert not result.is_valid
    assert result.issues[0].code is ValidationCode.GAP_IN_SEGMENT_POSITIONS


def test_validator_reports_overlapping_subtitles() -> None:
    factory = DomainModelFactory()
    document = factory.create_document(
        source_path=Path("audio/input.wav"),
        subtitles=(
            factory.create_subtitle(1, "first", 0.0, 1.5),
            factory.create_subtitle(2, "second", 1.0, 2.0),
        ),
    )

    result = DocumentValidator().validate(document)

    assert result.issues[0].code is ValidationCode.OVERLAPPING_SUBTITLES


def test_validator_reports_incomplete_learning_word_coverage() -> None:
    time_range = TimeRange(0.0, 1.0)
    aligned = Word("日本語です", time_range)
    sentence = Sentence(
        "日本語です",
        time_range,
        (aligned,),
        learning_words=(
            LearningWord("日本語", 0, 3, (0,), time_range, True),
        ),
    )
    document = Document(
        Path("audio/input.wav"),
        (Segment(0, sentence.text, time_range, (sentence,)),),
    )

    result = DocumentValidator().validate(document)

    assert result.issues[0].code is ValidationCode.INVALID_LEARNING_WORD_COVERAGE


def test_validator_allows_punctuation_gaps_between_learning_words() -> None:
    time_range = TimeRange(0.0, 1.0)
    sentence = Sentence(
        "2、3冊",
        time_range,
        (Word("2、3冊", time_range),),
        learning_words=(
            LearningWord("2", 0, 1, (0,), time_range, True),
            LearningWord("3冊", 2, 4, (0,), time_range, True),
        ),
    )
    document = Document(
        Path("audio/input.wav"),
        (Segment(0, sentence.text, time_range, (sentence,)),),
    )

    assert DocumentValidator().validate(document).is_valid


def test_validator_rejects_lexical_gaps_between_learning_words() -> None:
    time_range = TimeRange(0.0, 1.0)
    sentence = Sentence(
        "日本語です",
        time_range,
        (Word("日本語です", time_range),),
        learning_words=(
            LearningWord("日本", 0, 2, (0,), time_range, True),
            LearningWord("です", 3, 5, (0,), time_range, True),
        ),
    )
    document = Document(
        Path("audio/input.wav"),
        (Segment(0, sentence.text, time_range, (sentence,)),),
    )

    result = DocumentValidator().validate(document)

    assert result.issues[0].code is ValidationCode.INVALID_LEARNING_WORD_COVERAGE


def test_validation_result_can_raise_for_errors() -> None:
    factory = DomainModelFactory()
    document = factory.create_document(
        source_path=Path("audio/input.wav"),
        subtitles=(factory.create_subtitle(2, "second", 0.0, 1.0),),
    )
    result = DocumentValidator().validate(document)

    with pytest.raises(DomainValidationError) as error:
        result.raise_for_errors()

    assert error.value.issues == result.issues


def test_validation_issue_rejects_invalid_message_type() -> None:
    with pytest.raises(TypeError, match="message"):
        ValidationIssue(
            code=ValidationCode.GAP_IN_SUBTITLE_INDEXES,
            message=1,
            location="document.subtitles",
        )


def test_document_repository_protocol_accepts_domain_implementations() -> None:
    repository = InMemoryDocumentRepository()
    document = Document(source_path=Path("audio/input.wav"))

    assert isinstance(repository, DocumentRepository)

    repository.save(document)

    assert repository.get(document.source_path) == document
