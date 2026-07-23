from __future__ import annotations

from io import StringIO
import json
from pathlib import Path

from jp_learning_platform.domain import Document, PipelineContext
from jp_learning_platform.infrastructure import (
    ConsoleProgressReporter,
    StageArtifactStore,
)
from jp_learning_platform.workflow import (
    PipelineProgressEvent,
    PipelineProgressStatus,
    StageArtifactRecord,
)


def _context(source_path: Path, working_directory: Path) -> PipelineContext:
    return PipelineContext(
        run_id="transcribe-lesson",
        document=Document(source_path=source_path),
        working_directory=working_directory,
    )


def test_stage_artifact_store_writes_stage_artifact_and_manifest(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "lesson.mp3"
    output_path = tmp_path / "output" / "lesson.srt"
    store = StageArtifactStore(
        root_directory=tmp_path / "output" / ".work",
        run_name="run-001",
    )

    artifact_path = store.record(
        StageArtifactRecord(
            source_path=source_path,
            output_path=output_path,
            file_index=1,
            file_total=2,
            stage_name="audio-loader",
            status=PipelineProgressStatus.SUCCEEDED,
            context=_context(source_path, store.audio_directory(source_path)),
            elapsed_seconds=0.25,
            data={"raw": b"audio"},
        )
    )

    assert artifact_path == (
        tmp_path / "output" / ".work" / "run-001" / "lesson" / "00_audio_load.json"
    )
    artifact_payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact_payload["stage"] == "audio-loader"
    assert artifact_payload["status"] == "succeeded"
    assert artifact_payload["data"]["raw"] == {
        "type": "bytes",
        "size_bytes": 5,
    }
    assert artifact_payload["context"]["working_directory"] == str(
        store.audio_directory(source_path)
    )

    manifest_payload = json.loads(
        store.manifest_path(source_path).read_text(encoding="utf-8")
    )
    assert manifest_payload["current_stage"] == "audio-loader"
    assert manifest_payload["stage_artifact_path"] == str(artifact_path)


def test_stage_artifact_store_uses_ordered_stage_filenames(tmp_path: Path) -> None:
    store = StageArtifactStore(root_directory=tmp_path, run_name="run-001")
    source_path = Path("lesson.mp3")

    assert store.stage_path(source_path, "whisper").name == "01_whisper.json"
    assert store.stage_path(source_path, "whisperx-alignment").name == "02_align.json"
    assert store.stage_path(source_path, "qwen-repair").name == "03_repair.json"
    assert (
        store.stage_path(source_path, "homophone-resolution").name
        == "04_homophone_resolution.json"
    )
    assert (
        store.stage_path(source_path, "sentence-boundary-resolution").name
        == "05_sentence_boundary_resolution.json"
    )
    assert store.stage_path(source_path, "subtitle-writer").name == "10_write.json"


def test_console_progress_reporter_formats_stage_progress() -> None:
    output = StringIO()
    reporter = ConsoleProgressReporter(output=output)

    reporter.report(
        PipelineProgressEvent(
            source_path=Path("lesson.mp3"),
            output_path=Path("output/lesson.srt"),
            file_index=1,
            file_total=3,
            stage_name="whisper",
            status=PipelineProgressStatus.SUCCEEDED,
            elapsed_seconds=1.25,
            artifact_path=Path("output/.work/run/lesson/01_whisper.json"),
        )
    )

    assert output.getvalue() == (
        "[1/3] lesson.mp3 whisper done 1.25s -> "
        "output/.work/run/lesson/01_whisper.json\n"
    )


def test_console_progress_reporter_formats_pipeline_total() -> None:
    output = StringIO()
    reporter = ConsoleProgressReporter(output=output)

    reporter.report(
        PipelineProgressEvent(
            source_path=Path("lesson.mp3"),
            output_path=Path("output/lesson.srt"),
            file_index=1,
            file_total=1,
            stage_name="pipeline-total",
            status=PipelineProgressStatus.SUCCEEDED,
            elapsed_seconds=125.678,
        )
    )

    assert output.getvalue() == (
        "[1/1] lesson.mp3 pipeline-total done 125.68s\n"
    )
