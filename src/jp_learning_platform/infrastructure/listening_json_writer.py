"""Structured JSON writer for intensive listening artifacts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import json
from pathlib import Path

from jp_learning_platform.domain import Segment, Sentence, Subtitle, TimeRange, Word
from jp_learning_platform.workflow.subtitle_writer_stage import (
    SubtitleWrite,
    SubtitleWriteRequest,
    SubtitleWriter,
)

LISTENING_JSON_SCHEMA_VERSION = "1.0"
DEFAULT_LISTENING_JSON_EXTENSION = ".json"


def _time_range_payload(time_range: TimeRange) -> Mapping[str, float]:
    return {
        "start_seconds": time_range.start_seconds,
        "end_seconds": time_range.end_seconds,
        "duration_seconds": time_range.duration_seconds,
    }


def _word_payload(word: Word) -> Mapping[str, object]:
    return {
        "text": word.text,
        "speaker_id": word.speaker_id,
        "confidence": word.confidence,
        **_time_range_payload(word.time_range),
    }


def _sentence_payload(sentence: Sentence) -> Mapping[str, object]:
    return {
        "text": sentence.text,
        "speaker_id": sentence.speaker_id,
        **_time_range_payload(sentence.time_range),
        "words": [_word_payload(word) for word in sentence.words],
    }


def _segment_payload(segment: Segment) -> Mapping[str, object]:
    return {
        "position": segment.position,
        "text": segment.text,
        "speaker_id": segment.speaker_id,
        **_time_range_payload(segment.time_range),
        "sentences": [
            _sentence_payload(sentence) for sentence in segment.sentences
        ],
    }


def _subtitle_payload(subtitle: Subtitle) -> Mapping[str, object]:
    return {
        "index": subtitle.index,
        "text": subtitle.text,
        "speaker_id": subtitle.speaker_id,
        **_time_range_payload(subtitle.time_range),
    }


def _validate_writer(writer: SubtitleWriter, field_name: str) -> None:
    if not callable(getattr(writer, "write", None)):
        raise TypeError(f"{field_name} must define a callable write method.")


def _tuple_of_writers(
    writers: Iterable[SubtitleWriter],
    field_name: str,
) -> tuple[SubtitleWriter, ...]:
    try:
        tuple_writers = tuple(writers)
    except TypeError as error:
        raise TypeError(f"{field_name} must be iterable.") from error

    for writer in tuple_writers:
        _validate_writer(writer, field_name)

    return tuple_writers


@dataclass(frozen=True, slots=True)
class ListeningJsonWriter:
    """Write transcript segments and subtitle cues to structured JSON."""

    output_directory: Path = Path("output")

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_directory", Path(self.output_directory))

    def write(self, request: SubtitleWriteRequest) -> SubtitleWrite:
        if not isinstance(request, SubtitleWriteRequest):
            raise TypeError("request must be a SubtitleWriteRequest.")

        output_path = self.output_path_for(request.source_path)
        payload = self._payload(request, output_path)
        self._write_json(output_path, payload)

        return SubtitleWrite(
            source_path=request.source_path,
            output_path=output_path,
        )

    def output_path_for(self, source_path: Path) -> Path:
        return self.output_directory / (
            f"{Path(source_path).stem}{DEFAULT_LISTENING_JSON_EXTENSION}"
        )

    def _payload(
        self,
        request: SubtitleWriteRequest,
        output_path: Path,
    ) -> Mapping[str, object]:
        return {
            "schema_version": LISTENING_JSON_SCHEMA_VERSION,
            "source_path": str(request.source_path),
            "output_path": str(output_path),
            "run_id": request.run_id,
            "segments": [_segment_payload(segment) for segment in request.segments],
            "subtitles": [
                _subtitle_payload(subtitle) for subtitle in request.subtitles
            ],
        }

    def _write_json(self, path: Path, payload: Mapping[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_name(f"{path.name}.tmp")
        encoded = json.dumps(payload, ensure_ascii=False, indent=2)
        temporary_path.write_text(f"{encoded}\n", encoding="utf-8")
        temporary_path.replace(path)


@dataclass(frozen=True, slots=True)
class CompositeSubtitleWriter:
    """Write one primary subtitle artifact and optional export artifacts."""

    primary_writer: SubtitleWriter
    export_writers: tuple[SubtitleWriter, ...] = ()

    def __post_init__(self) -> None:
        _validate_writer(self.primary_writer, "primary_writer")
        object.__setattr__(
            self,
            "export_writers",
            _tuple_of_writers(self.export_writers, "export_writers"),
        )

    def write(self, request: SubtitleWriteRequest) -> SubtitleWrite:
        if not isinstance(request, SubtitleWriteRequest):
            raise TypeError("request must be a SubtitleWriteRequest.")

        primary_write = self.primary_writer.write(request)
        if not isinstance(primary_write, SubtitleWrite):
            raise TypeError("primary_writer must return a SubtitleWrite.")

        if primary_write.source_path != request.source_path:
            raise ValueError("primary_writer source path must match the request.")

        for export_writer in self.export_writers:
            export_write = export_writer.write(request)
            if not isinstance(export_write, SubtitleWrite):
                raise TypeError("export_writers must return SubtitleWrite values.")

            if export_write.source_path != request.source_path:
                raise ValueError("export_writer source path must match the request.")

        return primary_write


__all__ = [
    "CompositeSubtitleWriter",
    "DEFAULT_LISTENING_JSON_EXTENSION",
    "LISTENING_JSON_SCHEMA_VERSION",
    "ListeningJsonWriter",
]
