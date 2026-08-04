"""Content-addressed pipeline context cache and FFmpeg audio normalization."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from typing import Protocol

from jp_learning_platform.domain import (
    Document,
    LearningWord,
    PipelineContext,
    Segment,
    Sentence,
    Subtitle,
    TimeRange,
    Word,
)

DEFAULT_NORMALIZED_SAMPLE_RATE = 16_000
DEFAULT_NORMALIZED_CHANNELS = 1
DEFAULT_NORMALIZATION_VERSION = "pcm-s16le-mono-16000-v1"
_HASH_CHUNK_BYTES = 1024 * 1024


class PipelineCacheError(RuntimeError):
    """Base error for local pipeline cache failures."""


class AudioNormalizationError(PipelineCacheError):
    """Raised when a source cannot be normalized to PCM WAV."""


class PipelineContextCache(Protocol):
    def audio_digest(self, source_path: Path) -> str: ...

    def load_context(
        self,
        audio_digest: str,
        stage_fingerprint: str,
    ) -> PipelineContext | None: ...

    def save_context(
        self,
        audio_digest: str,
        stage_fingerprint: str,
        context: PipelineContext,
    ) -> Path: ...

    def work_lock(self, audio_digest: str, pipeline_fingerprint: str) -> object: ...

    def normalized_audio_directory(self, audio_digest: str) -> Path: ...


class AudioNormalizer(Protocol):
    def normalize(
        self,
        source_path: Path,
        cache_directory: Path,
        audio_digest: str,
    ) -> Path: ...


@dataclass(frozen=True, slots=True)
class LocalPipelineContextCache:
    """Persist immutable stage contexts under content-addressed keys."""

    root_directory: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root_directory", Path(self.root_directory))

    def audio_digest(self, source_path: Path) -> str:
        digest = sha256()
        with Path(source_path).open("rb") as source:
            while chunk := source.read(_HASH_CHUNK_BYTES):
                digest.update(chunk)
        return digest.hexdigest()

    def audio_directory(self, audio_digest: str) -> Path:
        return self.root_directory / _validate_digest(audio_digest)

    def normalized_audio_directory(self, audio_digest: str) -> Path:
        return self.audio_directory(audio_digest) / "audio"

    def context_path(self, audio_digest: str, stage_fingerprint: str) -> Path:
        return (
            self.audio_directory(audio_digest)
            / "stages"
            / f"{_validate_digest(stage_fingerprint)}.json"
        )

    def load_context(
        self,
        audio_digest: str,
        stage_fingerprint: str,
    ) -> PipelineContext | None:
        path = self.context_path(audio_digest, stage_fingerprint)
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return _context_from_payload(payload)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise PipelineCacheError(f"Invalid cached pipeline context: {path}") from error

    def save_context(
        self,
        audio_digest: str,
        stage_fingerprint: str,
        context: PipelineContext,
    ) -> Path:
        path = self.context_path(audio_digest, stage_fingerprint)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(".json.tmp")
        encoded = json.dumps(
            _context_payload(context),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        temporary_path.write_text(f"{encoded}\n", encoding="utf-8")
        temporary_path.replace(path)
        return path

    @contextmanager
    def work_lock(
        self,
        audio_digest: str,
        pipeline_fingerprint: str,
    ) -> Iterator[None]:
        lock_path = (
            self.audio_directory(audio_digest)
            / "locks"
            / f"{_validate_digest(pipeline_fingerprint)}.lock"
        )
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+b") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


@dataclass(frozen=True, slots=True)
class FFmpegAudioNormalizer:
    """Create one reusable deterministic PCM WAV for incompatible inputs."""

    ffmpeg_executable: str = "ffmpeg"
    ffprobe_executable: str = "ffprobe"
    sample_rate: int = DEFAULT_NORMALIZED_SAMPLE_RATE
    channels: int = DEFAULT_NORMALIZED_CHANNELS
    version: str = DEFAULT_NORMALIZATION_VERSION

    def normalize(
        self,
        source_path: Path,
        cache_directory: Path,
        audio_digest: str,
    ) -> Path:
        source = Path(source_path)
        if self._is_compatible_wav(source):
            return source

        _validate_digest(audio_digest)
        directory = Path(cache_directory)
        output_path = directory / f"{self.version}.wav"
        lock_path = directory / f"{self.version}.lock"
        directory.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+b") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                if output_path.is_file() and output_path.stat().st_size > 0:
                    return output_path
                temporary_path = output_path.with_suffix(".wav.tmp")
                command = (
                    self.ffmpeg_executable,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(source),
                    "-map_metadata",
                    "-1",
                    "-vn",
                    "-ac",
                    str(self.channels),
                    "-ar",
                    str(self.sample_rate),
                    "-c:a",
                    "pcm_s16le",
                    "-f",
                    "wav",
                    str(temporary_path),
                )
                try:
                    subprocess.run(
                        command,
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                except (OSError, subprocess.CalledProcessError) as error:
                    temporary_path.unlink(missing_ok=True)
                    detail = getattr(error, "stderr", None) or str(error)
                    raise AudioNormalizationError(
                        f"FFmpeg audio normalization failed: {detail.strip()}"
                    ) from error
                if not temporary_path.is_file() or temporary_path.stat().st_size == 0:
                    temporary_path.unlink(missing_ok=True)
                    raise AudioNormalizationError(
                        "FFmpeg audio normalization produced an empty file."
                    )
                temporary_path.replace(output_path)
                return output_path
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _is_compatible_wav(self, source_path: Path) -> bool:
        if source_path.suffix.lower() not in {".wav", ".wave"}:
            return False
        command = (
            self.ffprobe_executable,
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_name,sample_rate,channels,start_time",
            "-of",
            "json",
            str(source_path),
        )
        try:
            completed = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
            )
            streams = json.loads(completed.stdout).get("streams", ())
        except (OSError, subprocess.CalledProcessError, json.JSONDecodeError):
            return False
        if len(streams) != 1:
            return False
        stream = streams[0]
        start_time = stream.get("start_time")
        return (
            stream.get("codec_name") == "pcm_s16le"
            and int(stream.get("sample_rate", 0)) == self.sample_rate
            and int(stream.get("channels", 0)) == self.channels
            and (start_time is None or abs(float(start_time)) < 1e-9)
        )


def _validate_digest(value: str) -> str:
    normalized = value.strip().lower()
    invalid_character = any(
        character not in "0123456789abcdef" for character in normalized
    )
    if len(normalized) != 64 or invalid_character:
        raise ValueError("cache digest must be a 64-character SHA-256 value.")
    return normalized


def _time_range_payload(value: TimeRange) -> Mapping[str, float]:
    return {"start_seconds": value.start_seconds, "end_seconds": value.end_seconds}


def _time_range_from_payload(value: Mapping[str, object]) -> TimeRange:
    return TimeRange(float(value["start_seconds"]), float(value["end_seconds"]))


def _word_payload(value: Word) -> Mapping[str, object]:
    return {
        "text": value.text,
        "time_range": _time_range_payload(value.time_range),
        "confidence": value.confidence,
    }


def _word_from_payload(value: Mapping[str, object]) -> Word:
    return Word(
        text=str(value["text"]),
        time_range=_time_range_from_payload(value["time_range"]),
        confidence=value.get("confidence"),
    )


def _learning_word_payload(value: LearningWord) -> Mapping[str, object]:
    return {
        "text": value.text,
        "start_char": value.start_char,
        "end_char": value.end_char,
        "aligned_word_indexes": list(value.aligned_word_indexes),
        "time_range": _time_range_payload(value.time_range),
        "timing_estimated": value.timing_estimated,
        "is_structure": value.is_structure,
    }


def _learning_word_from_payload(value: Mapping[str, object]) -> LearningWord:
    return LearningWord(
        text=str(value["text"]),
        start_char=int(value["start_char"]),
        end_char=int(value["end_char"]),
        aligned_word_indexes=tuple(
            int(index) for index in value.get("aligned_word_indexes", ())
        ),
        time_range=_time_range_from_payload(value["time_range"]),
        timing_estimated=bool(value.get("timing_estimated", False)),
        is_structure=bool(value.get("is_structure", False)),
    )


def _sentence_payload(value: Sentence) -> Mapping[str, object]:
    return {
        "text": value.text,
        "time_range": _time_range_payload(value.time_range),
        "words": [_word_payload(item) for item in value.words],
        "learning_words": [
            _learning_word_payload(item) for item in value.learning_words
        ],
        "is_question": value.is_question,
        "asr_boundary_word_indexes": list(value.asr_boundary_word_indexes),
    }


def _sentence_from_payload(value: Mapping[str, object]) -> Sentence:
    return Sentence(
        text=str(value["text"]),
        time_range=_time_range_from_payload(value["time_range"]),
        words=tuple(_word_from_payload(item) for item in value.get("words", ())),
        learning_words=tuple(
            _learning_word_from_payload(item)
            for item in value.get("learning_words", ())
        ),
        is_question=bool(value.get("is_question", False)),
        asr_boundary_word_indexes=tuple(
            int(index) for index in value.get("asr_boundary_word_indexes", ())
        ),
    )


def _segment_payload(value: Segment) -> Mapping[str, object]:
    return {
        "position": value.position,
        "text": value.text,
        "time_range": _time_range_payload(value.time_range),
        "sentences": [_sentence_payload(item) for item in value.sentences],
    }


def _segment_from_payload(value: Mapping[str, object]) -> Segment:
    return Segment(
        position=int(value["position"]),
        text=str(value["text"]),
        time_range=_time_range_from_payload(value["time_range"]),
        sentences=tuple(
            _sentence_from_payload(item) for item in value.get("sentences", ())
        ),
    )


def _subtitle_payload(value: Subtitle) -> Mapping[str, object]:
    return {
        "index": value.index,
        "text": value.text,
        "time_range": _time_range_payload(value.time_range),
        "source_sentence_index": value.source_sentence_index,
    }


def _subtitle_from_payload(value: Mapping[str, object]) -> Subtitle:
    return Subtitle(
        index=int(value["index"]),
        text=str(value["text"]),
        time_range=_time_range_from_payload(value["time_range"]),
        source_sentence_index=(
            int(value["source_sentence_index"])
            if value.get("source_sentence_index") is not None
            else None
        ),
    )


def _context_payload(value: PipelineContext) -> Mapping[str, object]:
    return {
        "run_id": value.run_id,
        "working_directory": str(value.working_directory),
        "document": {
            "source_path": str(value.document.source_path),
            "segments": [_segment_payload(item) for item in value.document.segments],
            "subtitles": [_subtitle_payload(item) for item in value.document.subtitles],
        },
    }


def _context_from_payload(value: Mapping[str, object]) -> PipelineContext:
    document = value["document"]
    if not isinstance(document, Mapping):
        raise TypeError("cached document must be a mapping.")
    return PipelineContext(
        run_id=str(value["run_id"]),
        working_directory=Path(str(value["working_directory"])),
        document=Document(
            source_path=Path(str(document["source_path"])),
            segments=tuple(
                _segment_from_payload(item) for item in document.get("segments", ())
            ),
            subtitles=tuple(
                _subtitle_from_payload(item)
                for item in document.get("subtitles", ())
            ),
        ),
    )


__all__ = [
    "AudioNormalizationError",
    "AudioNormalizer",
    "DEFAULT_NORMALIZATION_VERSION",
    "DEFAULT_NORMALIZED_CHANNELS",
    "DEFAULT_NORMALIZED_SAMPLE_RATE",
    "FFmpegAudioNormalizer",
    "LocalPipelineContextCache",
    "PipelineCacheError",
    "PipelineContextCache",
]
