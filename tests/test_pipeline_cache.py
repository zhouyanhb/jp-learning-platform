from __future__ import annotations

from pathlib import Path
import subprocess

from jp_learning_platform.domain import (
    Document,
    PipelineContext,
    Segment,
    Sentence,
    Subtitle,
    TimeRange,
    Word,
)
from jp_learning_platform.infrastructure import (
    FFmpegAudioNormalizer,
    LocalPipelineContextCache,
)


def _context(source_path: Path, working_directory: Path) -> PipelineContext:
    time_range = TimeRange(0.0, 1.0)
    word = Word(
        text="聴解",
        time_range=time_range,
        confidence=0.8,
        speaker_id="speaker-1",
    )
    sentence = Sentence(
        text="聴解です。",
        time_range=time_range,
        words=(word,),
        speaker_id="speaker-1",
    )
    segment = Segment(
        position=0,
        text=sentence.text,
        time_range=time_range,
        sentences=(sentence,),
        speaker_id="speaker-1",
    )
    subtitle = Subtitle(
        index=1,
        text=sentence.text,
        time_range=time_range,
        speaker_id="speaker-1",
    )
    return PipelineContext(
        run_id="run-1",
        working_directory=working_directory,
        document=Document(
            source_path=source_path,
            segments=(segment,),
            subtitles=(subtitle,),
        ),
    )


def test_pipeline_cache_round_trips_complete_context(tmp_path: Path) -> None:
    audio_path = tmp_path / "lesson.mp3"
    audio_path.write_bytes(b"same audio")
    cache = LocalPipelineContextCache(tmp_path / "cache")
    audio_digest = cache.audio_digest(audio_path)
    stage_fingerprint = "a" * 64
    expected = _context(audio_path, tmp_path / "work")

    cache.save_context(audio_digest, stage_fingerprint, expected)

    assert cache.load_context(audio_digest, stage_fingerprint) == expected


def test_audio_digest_is_based_on_content_instead_of_filename(tmp_path: Path) -> None:
    first_path = tmp_path / "first.mp3"
    second_path = tmp_path / "second.mp3"
    first_path.write_bytes(b"same audio")
    second_path.write_bytes(b"same audio")
    cache = LocalPipelineContextCache(tmp_path / "cache")

    assert cache.audio_digest(first_path) == cache.audio_digest(second_path)


def test_ffmpeg_normalizer_reuses_cached_audio(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_path = tmp_path / "lesson.mp3"
    source_path.write_bytes(b"compressed audio")
    cache_directory = tmp_path / "cache"
    ffmpeg_commands: list[tuple[str, ...]] = []

    def fake_run(command, **kwargs):
        ffmpeg_commands.append(tuple(command))
        Path(command[-1]).write_bytes(b"normalized wav")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    normalizer = FFmpegAudioNormalizer()
    digest = "b" * 64

    first_result = normalizer.normalize(source_path, cache_directory, digest)
    second_result = normalizer.normalize(source_path, cache_directory, digest)

    assert first_result == second_result
    assert first_result.read_bytes() == b"normalized wav"
    assert len(ffmpeg_commands) == 1


def test_ffmpeg_normalizer_extracts_video_audio_without_video_stream(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_path = tmp_path / "lesson.mp4"
    source_path.write_bytes(b"video")
    commands: list[tuple[str, ...]] = []

    def fake_run(command, **kwargs):
        commands.append(tuple(command))
        Path(command[-1]).write_bytes(b"pcm audio")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = FFmpegAudioNormalizer().normalize(
        source_path,
        tmp_path / "cache",
        "c" * 64,
    )

    assert result.suffix == ".wav"
    assert result.read_bytes() == b"pcm audio"
    assert "-vn" in commands[0]
    assert commands[0][commands[0].index("-i") + 1] == str(source_path)
