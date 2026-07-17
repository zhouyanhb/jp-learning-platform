"""Subtitle validator workflow stage."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypeVar

from jp_learning_platform.domain import (
    PipelineContext,
    Segment,
    Subtitle,
    ValidationIssue,
    ValidationResult,
)
from jp_learning_platform.workflow.runtime import StageResult

SUBTITLE_VALIDATOR_STAGE_NAME = "subtitle-validator"

T = TypeVar("T")


def _normalize_name(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")

    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty.")

    return normalized


def _tuple_of_type(
    values: Iterable[T],
    item_type: type[T],
    field_name: str,
) -> tuple[T, ...]:
    try:
        tuple_values = tuple(values)
    except TypeError as error:
        raise TypeError(f"{field_name} must be iterable.") from error

    for value in tuple_values:
        if not isinstance(value, item_type):
            raise TypeError(f"{field_name} must contain {item_type.__name__} values.")

    return tuple_values


@dataclass(frozen=True, slots=True)
class SubtitleValidationRequest:
    """Input passed from the workflow stage to a subtitle validator."""

    source_path: Path
    working_directory: Path
    run_id: str
    segments: tuple[Segment, ...]
    subtitles: tuple[Subtitle, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_path", Path(self.source_path))
        object.__setattr__(
            self,
            "working_directory",
            Path(self.working_directory),
        )
        object.__setattr__(self, "run_id", _normalize_name(self.run_id, "run_id"))
        object.__setattr__(
            self,
            "segments",
            _tuple_of_type(self.segments, Segment, "segments"),
        )
        object.__setattr__(
            self,
            "subtitles",
            _tuple_of_type(self.subtitles, Subtitle, "subtitles"),
        )


@dataclass(frozen=True, slots=True)
class SubtitleValidation:
    """Normalized subtitle validator output."""

    source_path: Path
    result: ValidationResult

    def __post_init__(self) -> None:
        if not isinstance(self.result, ValidationResult):
            raise TypeError("result must be a ValidationResult.")

        object.__setattr__(self, "source_path", Path(self.source_path))


class SubtitleValidator(Protocol):
    """Contract implemented by subtitle validation adapters."""

    def validate(self, request: SubtitleValidationRequest) -> SubtitleValidation:
        """Validate optimized subtitle entries."""


class SubtitleValidatorStageError(RuntimeError):
    """Base error for subtitle validator stage failures."""


class InvalidSubtitleValidatorError(SubtitleValidatorStageError):
    """Raised when a configured validator does not satisfy the stage contract."""

    def __init__(self) -> None:
        super().__init__("Subtitle validator must define a callable validate method.")


class MissingSubtitlesToValidateError(SubtitleValidatorStageError):
    """Raised when the document has no subtitles to validate."""

    def __init__(self) -> None:
        super().__init__("Subtitle validation requires existing document subtitles.")


class InvalidSubtitleValidationError(SubtitleValidatorStageError):
    """Raised when a validator returns an invalid validation result."""


class SubtitleValidationFailedError(SubtitleValidatorStageError):
    """Raised when subtitle validation reports issues."""

    def __init__(self, issues: tuple[ValidationIssue, ...]) -> None:
        if not issues:
            raise ValueError("issues must not be empty.")

        self.issues = issues
        super().__init__("Subtitle validation failed.")


@dataclass(frozen=True, slots=True)
class SubtitleValidatorStage:
    """Workflow stage that coordinates subtitle validation."""

    validator: SubtitleValidator
    name: str = SUBTITLE_VALIDATOR_STAGE_NAME

    def __post_init__(self) -> None:
        if not callable(getattr(self.validator, "validate", None)):
            raise InvalidSubtitleValidatorError()

        object.__setattr__(self, "name", _normalize_name(self.name, "name"))

    def run(self, context: PipelineContext) -> StageResult:
        if not isinstance(context, PipelineContext):
            raise TypeError("context must be a PipelineContext.")

        if not context.document.subtitles:
            raise MissingSubtitlesToValidateError()

        request = SubtitleValidationRequest(
            source_path=context.document.source_path,
            working_directory=context.working_directory,
            run_id=context.run_id,
            segments=context.document.segments,
            subtitles=context.document.subtitles,
        )
        validation = self.validator.validate(request)
        if not isinstance(validation, SubtitleValidation):
            raise InvalidSubtitleValidationError(
                "Subtitle validator must return a SubtitleValidation."
            )

        if validation.source_path != request.source_path:
            raise InvalidSubtitleValidationError(
                "Subtitle validation source path must match the request source path."
            )

        if not validation.result.is_valid:
            raise SubtitleValidationFailedError(validation.result.issues)

        return StageResult(stage_name=self.name, context=context)


__all__ = [
    "InvalidSubtitleValidationError",
    "InvalidSubtitleValidatorError",
    "MissingSubtitlesToValidateError",
    "SUBTITLE_VALIDATOR_STAGE_NAME",
    "SubtitleValidation",
    "SubtitleValidationFailedError",
    "SubtitleValidationRequest",
    "SubtitleValidator",
    "SubtitleValidatorStage",
    "SubtitleValidatorStageError",
]
