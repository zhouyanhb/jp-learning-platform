"""JSON artifact store for local subtitle pipeline stages."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
import json
from pathlib import Path
import re

from jp_learning_platform.workflow.progress import (
    StageArtifactRecord,
)

STAGE_ARTIFACT_FILENAMES: Mapping[str, str] = {
    "audio-loader": "00_audio_load.json",
    "whisper": "01_whisper.json",
    "whisperx-alignment": "02_align.json",
    "qwen-repair": "03_repair.json",
    "homophone-resolution": "04_homophone_resolution.json",
    "sentence-boundary-resolution": "05_sentence_boundary_resolution.json",
    "subtitle-builder": "06_build.json",
    "subtitle-merger": "07_merge.json",
    "readability-optimizer": "08_readability.json",
    "subtitle-validator": "09_validate.json",
    "subtitle-writer": "10_write.json",
}

_SAFE_FRAGMENT_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")


def _default_run_name() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_path_fragment(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")

    normalized = _SAFE_FRAGMENT_PATTERN.sub("_", value.strip())
    normalized = normalized.strip("._")
    if not normalized:
        raise ValueError(f"{field_name} must not be empty.")

    return normalized


def _stage_filename(stage_name: str) -> str:
    filename = STAGE_ARTIFACT_FILENAMES.get(stage_name)
    if filename is not None:
        return filename

    return f"{_safe_path_fragment(stage_name.replace('-', '_'), 'stage_name')}.json"


class StageArtifactStoreError(RuntimeError):
    """Base error for stage artifact persistence failures."""


class InvalidStageArtifactRecordError(StageArtifactStoreError):
    """Raised when an invalid artifact record is supplied."""


class StageArtifactSerializationError(StageArtifactStoreError):
    """Raised when an artifact payload cannot be serialized."""


@dataclass(frozen=True, slots=True)
class StageArtifactStore:
    """Persist stage artifacts under one run directory."""

    root_directory: Path
    run_name: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "root_directory", Path(self.root_directory))
        object.__setattr__(
            self,
            "run_name",
            _safe_path_fragment(self.run_name or _default_run_name(), "run_name"),
        )

    @property
    def run_directory(self) -> Path:
        return self.root_directory / self.run_name

    def audio_directory(self, source_path: Path) -> Path:
        return self.run_directory / _safe_path_fragment(Path(source_path).stem, "source_path")

    def stage_path(self, source_path: Path, stage_name: str) -> Path:
        return self.audio_directory(source_path) / _stage_filename(stage_name)

    def manifest_path(self, source_path: Path) -> Path:
        return self.audio_directory(source_path) / "manifest.json"

    def record(self, record: StageArtifactRecord) -> Path:
        if not isinstance(record, StageArtifactRecord):
            raise InvalidStageArtifactRecordError(
                "record must be a StageArtifactRecord."
            )

        recorded_at = _utc_timestamp()
        artifact_path = self.stage_path(record.source_path, record.stage_name)
        artifact_payload = self._stage_payload(record, recorded_at)
        self._write_json(artifact_path, artifact_payload)
        self._write_json(
            self.manifest_path(record.source_path),
            self._manifest_payload(record, artifact_path, recorded_at),
        )
        return artifact_path

    def _stage_payload(
        self,
        record: StageArtifactRecord,
        recorded_at: str,
    ) -> Mapping[str, object]:
        return {
            "run_name": self.run_name,
            "source_path": str(record.source_path),
            "output_path": str(record.output_path),
            "file_index": record.file_index,
            "file_total": record.file_total,
            "stage": record.stage_name,
            "status": record.status.value,
            "elapsed_seconds": record.elapsed_seconds,
            "message": record.message,
            "recorded_at": recorded_at,
            "context": _to_json_value(record.context),
            "data": _to_json_value(record.data),
        }

    def _manifest_payload(
        self,
        record: StageArtifactRecord,
        artifact_path: Path,
        recorded_at: str,
    ) -> Mapping[str, object]:
        return {
            "run_name": self.run_name,
            "source_path": str(record.source_path),
            "output_path": str(record.output_path),
            "file_index": record.file_index,
            "file_total": record.file_total,
            "current_stage": record.stage_name,
            "status": record.status.value,
            "stage_artifact_path": str(artifact_path),
            "updated_at": recorded_at,
        }

    def _write_json(self, path: Path, payload: Mapping[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_name(f"{path.name}.tmp")
        try:
            encoded = json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        except TypeError as error:
            raise StageArtifactSerializationError(
                f"Stage artifact is not JSON serializable: {path}"
            ) from error

        temporary_path.write_text(f"{encoded}\n", encoding="utf-8")
        temporary_path.replace(path)


def _to_json_value(value: object) -> object:
    if value is None or isinstance(value, str | int | float | bool):
        return value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, Enum):
        return value.value

    if isinstance(value, bytes):
        return {
            "type": "bytes",
            "size_bytes": len(value),
        }

    if is_dataclass(value):
        return {
            field.name: _to_json_value(getattr(value, field.name))
            for field in fields(value)
        }

    if isinstance(value, Mapping):
        return {str(key): _to_json_value(item) for key, item in value.items()}

    if isinstance(value, tuple | list):
        return [_to_json_value(item) for item in value]

    if isinstance(value, set | frozenset):
        return [_to_json_value(item) for item in sorted(value, key=str)]

    return str(value)


__all__ = [
    "InvalidStageArtifactRecordError",
    "STAGE_ARTIFACT_FILENAMES",
    "StageArtifactSerializationError",
    "StageArtifactStore",
    "StageArtifactStoreError",
]
