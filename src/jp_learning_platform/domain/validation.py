"""Domain validation services for subtitle pipeline documents."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from jp_learning_platform.domain.models import Document, Segment, Subtitle

FIRST_SEGMENT_POSITION = 0
FIRST_SUBTITLE_INDEX = 1


class ValidationCode(Enum):
    DUPLICATE_SEGMENT_POSITION = "duplicate_segment_position"
    GAP_IN_SEGMENT_POSITIONS = "gap_in_segment_positions"
    OVERLAPPING_SEGMENTS = "overlapping_segments"
    DUPLICATE_SUBTITLE_INDEX = "duplicate_subtitle_index"
    GAP_IN_SUBTITLE_INDEXES = "gap_in_subtitle_indexes"
    OVERLAPPING_SUBTITLES = "overlapping_subtitles"


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    code: ValidationCode
    message: str
    location: str

    def __post_init__(self) -> None:
        if not isinstance(self.code, ValidationCode):
            raise TypeError("code must be a ValidationCode.")

        if not isinstance(self.message, str):
            raise TypeError("message must be a string.")

        if not self.message.strip():
            raise ValueError("message must not be empty.")

        if not isinstance(self.location, str):
            raise TypeError("location must be a string.")

        if not self.location.strip():
            raise ValueError("location must not be empty.")


class DomainValidationError(ValueError):
    """Raised when domain validation issues are promoted to an exception."""

    def __init__(self, issues: tuple[ValidationIssue, ...]) -> None:
        if not issues:
            raise ValueError("issues must not be empty.")

        self.issues = issues
        super().__init__("Domain validation failed.")


@dataclass(frozen=True, slots=True)
class ValidationResult:
    issues: tuple[ValidationIssue, ...] = ()

    def __post_init__(self) -> None:
        issues = tuple(self.issues)
        for issue in issues:
            if not isinstance(issue, ValidationIssue):
                raise TypeError("issues must contain ValidationIssue values.")

        object.__setattr__(self, "issues", issues)

    @property
    def is_valid(self) -> bool:
        return not self.issues

    def raise_for_errors(self) -> None:
        if self.issues:
            raise DomainValidationError(self.issues)


@dataclass(frozen=True, slots=True)
class DocumentValidator:
    """Validate cross-model document consistency."""

    def validate(self, document: Document) -> ValidationResult:
        if not isinstance(document, Document):
            raise TypeError("document must be a Document.")

        issues = (
            *_validate_segments(document.segments),
            *_validate_subtitles(document.subtitles),
        )
        return ValidationResult(issues=issues)


def _validate_segments(segments: tuple[Segment, ...]) -> tuple[ValidationIssue, ...]:
    issues: list[ValidationIssue] = []
    positions = [segment.position for segment in segments]
    duplicate_positions = _duplicates(positions)

    for position in duplicate_positions:
        issues.append(
            ValidationIssue(
                code=ValidationCode.DUPLICATE_SEGMENT_POSITION,
                message=f"Segment position {position} appears more than once.",
                location="document.segments",
            )
        )

    expected_positions = tuple(range(FIRST_SEGMENT_POSITION, len(segments)))
    if tuple(sorted(positions)) != expected_positions:
        issues.append(
            ValidationIssue(
                code=ValidationCode.GAP_IN_SEGMENT_POSITIONS,
                message="Segment positions must be contiguous from 0.",
                location="document.segments",
            )
        )

    for previous, current in zip(segments, segments[1:]):
        if current.time_range.start_seconds < previous.time_range.end_seconds:
            issues.append(
                ValidationIssue(
                    code=ValidationCode.OVERLAPPING_SEGMENTS,
                    message="Segment time ranges must not overlap.",
                    location=f"document.segments[{current.position}]",
                )
            )

    return tuple(issues)


def _validate_subtitles(
    subtitles: tuple[Subtitle, ...],
) -> tuple[ValidationIssue, ...]:
    issues: list[ValidationIssue] = []
    indexes = [subtitle.index for subtitle in subtitles]
    duplicate_indexes = _duplicates(indexes)

    for index in duplicate_indexes:
        issues.append(
            ValidationIssue(
                code=ValidationCode.DUPLICATE_SUBTITLE_INDEX,
                message=f"Subtitle index {index} appears more than once.",
                location="document.subtitles",
            )
        )

    expected_indexes = tuple(
        range(FIRST_SUBTITLE_INDEX, FIRST_SUBTITLE_INDEX + len(subtitles))
    )
    if tuple(sorted(indexes)) != expected_indexes:
        issues.append(
            ValidationIssue(
                code=ValidationCode.GAP_IN_SUBTITLE_INDEXES,
                message="Subtitle indexes must be contiguous from 1.",
                location="document.subtitles",
            )
        )

    for previous, current in zip(subtitles, subtitles[1:]):
        if current.time_range.start_seconds < previous.time_range.end_seconds:
            issues.append(
                ValidationIssue(
                    code=ValidationCode.OVERLAPPING_SUBTITLES,
                    message="Subtitle time ranges must not overlap.",
                    location=f"document.subtitles[{current.index}]",
                )
            )

    return tuple(issues)


def _duplicates(values: list[int]) -> tuple[int, ...]:
    seen: set[int] = set()
    duplicates: set[int] = set()

    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)

    return tuple(sorted(duplicates))


__all__ = [
    "DocumentValidator",
    "DomainValidationError",
    "ValidationCode",
    "ValidationIssue",
    "ValidationResult",
]
