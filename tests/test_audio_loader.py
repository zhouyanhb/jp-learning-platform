from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from jp_learning_platform.infrastructure import (
    AudioFileNotFoundError,
    AudioFormat,
    AudioLoader,
    AudioLoaderError,
    EmptyAudioFileError,
    LoadedAudio,
    UnsupportedAudioFormatError,
)


def test_audio_loader_loads_supported_audio_file(tmp_path: Path) -> None:
    source_path = tmp_path / "input.wav"
    source_path.write_bytes(b"audio-bytes")

    loaded_audio = AudioLoader().load(source_path)

    assert loaded_audio.source_path == source_path
    assert loaded_audio.format is AudioFormat.WAV
    assert loaded_audio.data == b"audio-bytes"
    assert loaded_audio.size_bytes == len(b"audio-bytes")


def test_audio_loader_detects_formats_case_insensitively(tmp_path: Path) -> None:
    source_path = tmp_path / "input.MP3"
    source_path.write_bytes(b"audio-bytes")

    assert AudioLoader().load(source_path).format is AudioFormat.MP3


def test_audio_loader_rejects_missing_files(tmp_path: Path) -> None:
    source_path = tmp_path / "missing.wav"

    with pytest.raises(AudioFileNotFoundError) as error:
        AudioLoader().load(source_path)

    assert error.value.source_path == source_path


def test_audio_loader_rejects_directories(tmp_path: Path) -> None:
    with pytest.raises(AudioLoaderError, match="not a file"):
        AudioLoader().load(tmp_path)


def test_audio_loader_rejects_unsupported_formats(tmp_path: Path) -> None:
    source_path = tmp_path / "input.txt"
    source_path.write_bytes(b"not-audio")

    with pytest.raises(UnsupportedAudioFormatError) as error:
        AudioLoader().load(source_path)

    assert error.value.source_path == source_path


def test_audio_loader_rejects_empty_files(tmp_path: Path) -> None:
    source_path = tmp_path / "input.wav"
    source_path.write_bytes(b"")

    with pytest.raises(EmptyAudioFileError) as error:
        AudioLoader().load(source_path)

    assert error.value.source_path == source_path


def test_audio_loader_honors_configured_supported_formats(tmp_path: Path) -> None:
    source_path = tmp_path / "input.wav"
    source_path.write_bytes(b"audio-bytes")
    loader = AudioLoader(supported_formats=(AudioFormat.MP3,))

    with pytest.raises(UnsupportedAudioFormatError):
        loader.load(source_path)


def test_loaded_audio_is_immutable(tmp_path: Path) -> None:
    source_path = tmp_path / "input.wav"
    loaded_audio = LoadedAudio(
        source_path=source_path,
        format=AudioFormat.WAV,
        data=b"audio-bytes",
    )

    with pytest.raises(FrozenInstanceError):
        loaded_audio.data = b"changed"
