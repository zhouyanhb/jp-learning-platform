"""Application contracts for subtitle pipeline execution."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

DEFAULT_OUTPUT_DIRECTORY = Path("output")
SUPPORTED_AUDIO_EXTENSIONS = (
    ".aac",
    ".flac",
    ".m4a",
    ".mp3",
    ".ogg",
    ".opus",
    ".wav",
    ".wave",
)


def _normalize_path(value: Path, field_name: str) -> Path:
    try:
        return Path(value)
    except TypeError as error:
        raise TypeError(f"{field_name} must be a path-like value.") from error


def _normalize_extensions(values: Iterable[str]) -> tuple[str, ...]:
    extensions = tuple(dict.fromkeys(value.lower() for value in values))
    if not extensions:
        raise ValueError("supported_extensions must not be empty.")

    for extension in extensions:
        if not extension.startswith("."):
            raise ValueError("supported_extensions must include leading dots.")

    return extensions


class SubtitlePipelineInputError(ValueError):
    """Base error for invalid subtitle pipeline inputs."""


class InputPathNotFoundError(SubtitlePipelineInputError):
    """Raised when the requested input path does not exist."""

    def __init__(self, input_path: Path) -> None:
        self.input_path = input_path
        super().__init__(f"Input path not found: {input_path}")


class NoAudioInputsFoundError(SubtitlePipelineInputError):
    """Raised when no supported audio files can be discovered."""

    def __init__(self, input_path: Path) -> None:
        self.input_path = input_path
        super().__init__(f"No supported audio files found: {input_path}")


@dataclass(frozen=True, slots=True)
class SubtitlePipelineRequest:
    """Request to generate subtitle files from an audio file or folder."""

    input_path: Path
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_path", _normalize_path(self.input_path, "input_path"))
        object.__setattr__(
            self,
            "output_directory",
            _normalize_path(self.output_directory, "output_directory"),
        )


@dataclass(frozen=True, slots=True)
class SubtitlePipelineItemResult:
    """Result for one processed audio source."""

    source_path: Path
    output_path: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_path", _normalize_path(self.source_path, "source_path"))
        object.__setattr__(self, "output_path", _normalize_path(self.output_path, "output_path"))


@dataclass(frozen=True, slots=True)
class SubtitlePipelineResult:
    """Result for a subtitle pipeline request."""

    items: tuple[SubtitlePipelineItemResult, ...]

    def __post_init__(self) -> None:
        items = tuple(self.items)
        if not items:
            raise ValueError("items must not be empty.")

        for item in items:
            if not isinstance(item, SubtitlePipelineItemResult):
                raise TypeError("items must contain SubtitlePipelineItemResult values.")

        object.__setattr__(self, "items", items)

    @property
    def output_paths(self) -> tuple[Path, ...]:
        return tuple(item.output_path for item in self.items)


@dataclass(frozen=True, slots=True)
class AudioInputDiscovery:
    """Discover supported local audio inputs for a pipeline request."""

    supported_extensions: tuple[str, ...] = SUPPORTED_AUDIO_EXTENSIONS

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "supported_extensions",
            _normalize_extensions(self.supported_extensions),
        )

    def discover(self, input_path: Path) -> tuple[Path, ...]:
        path = _normalize_path(input_path, "input_path")
        if not path.exists():
            raise InputPathNotFoundError(path)

        if path.is_file():
            if self._is_supported_audio(path):
                return (path,)
            raise NoAudioInputsFoundError(path)

        if path.is_dir():
            audio_paths = tuple(
                sorted(
                    (
                        child
                        for child in path.iterdir()
                        if child.is_file() and self._is_supported_audio(child)
                    ),
                    key=lambda child: child.name,
                )
            )
            if audio_paths:
                return audio_paths

            raise NoAudioInputsFoundError(path)

        raise NoAudioInputsFoundError(path)

    def _is_supported_audio(self, path: Path) -> bool:
        return path.suffix.lower() in self.supported_extensions


__all__ = [
    "AudioInputDiscovery",
    "DEFAULT_OUTPUT_DIRECTORY",
    "InputPathNotFoundError",
    "NoAudioInputsFoundError",
    "SUPPORTED_AUDIO_EXTENSIONS",
    "SubtitlePipelineInputError",
    "SubtitlePipelineItemResult",
    "SubtitlePipelineRequest",
    "SubtitlePipelineResult",
]
