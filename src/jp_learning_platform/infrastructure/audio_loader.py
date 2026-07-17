"""Audio loading boundary for subtitle pipeline inputs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Mapping


class AudioFormat(Enum):
    AAC = "aac"
    FLAC = "flac"
    M4A = "m4a"
    MP3 = "mp3"
    OGG = "ogg"
    OPUS = "opus"
    WAV = "wav"


_EXTENSION_FORMATS: Mapping[str, AudioFormat] = MappingProxyType(
    {
        ".aac": AudioFormat.AAC,
        ".flac": AudioFormat.FLAC,
        ".m4a": AudioFormat.M4A,
        ".mp3": AudioFormat.MP3,
        ".ogg": AudioFormat.OGG,
        ".opus": AudioFormat.OPUS,
        ".wav": AudioFormat.WAV,
        ".wave": AudioFormat.WAV,
    }
)

DEFAULT_AUDIO_FORMATS: tuple[AudioFormat, ...] = tuple(AudioFormat)


class AudioLoaderError(RuntimeError):
    """Base error for audio loading failures."""


class AudioFileNotFoundError(AudioLoaderError):
    """Raised when the audio source path does not exist."""

    def __init__(self, source_path: Path) -> None:
        self.source_path = source_path
        super().__init__(f"Audio file not found: {source_path}")


class UnsupportedAudioFormatError(AudioLoaderError):
    """Raised when an audio file extension is not supported."""

    def __init__(self, source_path: Path) -> None:
        self.source_path = source_path
        super().__init__(f"Unsupported audio format: {source_path.suffix}")


class EmptyAudioFileError(AudioLoaderError):
    """Raised when an audio file contains no bytes."""

    def __init__(self, source_path: Path) -> None:
        self.source_path = source_path
        super().__init__(f"Audio file is empty: {source_path}")


@dataclass(frozen=True, slots=True)
class LoadedAudio:
    source_path: Path
    format: AudioFormat
    data: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.format, AudioFormat):
            raise TypeError("format must be an AudioFormat.")

        if not isinstance(self.data, bytes):
            raise TypeError("data must be bytes.")

        if not self.data:
            raise EmptyAudioFileError(Path(self.source_path))

        object.__setattr__(self, "source_path", Path(self.source_path))

    @property
    def size_bytes(self) -> int:
        return len(self.data)


@dataclass(frozen=True, slots=True)
class AudioLoader:
    supported_formats: tuple[AudioFormat, ...] = DEFAULT_AUDIO_FORMATS

    def __post_init__(self) -> None:
        formats = tuple(self.supported_formats)
        if not formats:
            raise ValueError("supported_formats must not be empty.")

        for audio_format in formats:
            if not isinstance(audio_format, AudioFormat):
                raise TypeError("supported_formats must contain AudioFormat values.")

        object.__setattr__(self, "supported_formats", tuple(dict.fromkeys(formats)))

    def load(self, source_path: Path) -> LoadedAudio:
        path = Path(source_path)
        if not path.exists():
            raise AudioFileNotFoundError(path)

        if not path.is_file():
            raise AudioLoaderError(f"Audio source is not a file: {path}")

        audio_format = self.detect_format(path)
        if audio_format not in self.supported_formats:
            raise UnsupportedAudioFormatError(path)

        data = path.read_bytes()
        if not data:
            raise EmptyAudioFileError(path)

        return LoadedAudio(
            source_path=path,
            format=audio_format,
            data=data,
        )

    def detect_format(self, source_path: Path) -> AudioFormat:
        path = Path(source_path)
        audio_format = _EXTENSION_FORMATS.get(path.suffix.lower())
        if audio_format is None:
            raise UnsupportedAudioFormatError(path)

        return audio_format


__all__ = [
    "AudioFileNotFoundError",
    "AudioFormat",
    "AudioLoader",
    "AudioLoaderError",
    "EmptyAudioFileError",
    "LoadedAudio",
    "UnsupportedAudioFormatError",
]
