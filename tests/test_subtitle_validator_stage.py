from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass
from pathlib import Path

import pytest

from jp_learning_platform.domain import (
    Document,
    PipelineContext,
    Segment,
    Subtitle,
    TimeRange,
    ValidationCode,
    ValidationIssue,
    ValidationResult,
)
from jp_learning_platform.workflow import (
    InvalidSubtitleValidationError,
    InvalidSubtitleValidatorError,
    MissingSubtitlesToValidateError,
    StageResult,
    SubtitleValidation,
    SubtitleValidationFailedError,
    SubtitleValidationRequest,
    SubtitleValidatorStage,
)


@dataclass(slots=True)
class FakeValidator:
    validation: SubtitleValidation
    requests: list[SubtitleValidationRequest]

    def validate(self, request: SubtitleValidationRequest) -> SubtitleValidation:
        self.requests.append(request)
        return self.validation


@dataclass(frozen=True, slots=True)
class InvalidValidationValidator:
    def validate(self, request: SubtitleValidationRequest) -> object:
        return request


def _segment() -> Segment:
    return Segment(
        position=0,
        text="Nihongo desu.",
        time_range=TimeRange(start_seconds=0.0, end_seconds=1.5),
    )


def _subtitle(index: int = 1, text: str = "Nihongo desu.") -> Subtitle:
    return Subtitle(
        index=index,
        text=text,
        time_range=TimeRange(start_seconds=0.0, end_seconds=1.5),
    )


def _context(
    source_path: Path,
    subtitles: tuple[Subtitle, ...],
    segments: tuple[Segment, ...] = (),
) -> PipelineContext:
    return PipelineContext(
        run_id="run-001",
        document=Document(
            source_path=source_path,
            segments=segments,
            subtitles=subtitles,
        ),
        working_directory=source_path.parent / "work",
    )


def _issue() -> ValidationIssue:
    return ValidationIssue(
        code=ValidationCode.OVERLAPPING_SUBTITLES,
        message="Subtitle time ranges must not overlap.",
        location="document.subtitles[2]",
    )


def test_subtitle_validator_stage_validates_existing_subtitles(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "input.wav"
    segment = _segment()
    subtitle = _subtitle()
    validator = FakeValidator(
        validation=SubtitleValidation(
            source_path=source_path,
            result=ValidationResult(),
        ),
        requests=[],
    )
    context = _context(source_path, (subtitle,), segments=(segment,))

    result = SubtitleValidatorStage(validator=validator).run(context)

    assert isinstance(result, StageResult)
    assert result.stage_name == "subtitle-validator"
    assert result.context is context
    assert validator.requests == [
        SubtitleValidationRequest(
            source_path=source_path,
            working_directory=tmp_path / "work",
            run_id="run-001",
            segments=(segment,),
            subtitles=(subtitle,),
        )
    ]


def test_subtitle_validator_stage_accepts_custom_stage_name(tmp_path: Path) -> None:
    source_path = tmp_path / "input.wav"
    subtitle = _subtitle()
    validator = FakeValidator(
        validation=SubtitleValidation(
            source_path=source_path,
            result=ValidationResult(),
        ),
        requests=[],
    )
    stage = SubtitleValidatorStage(validator=validator, name="  validator  ")

    result = stage.run(_context(source_path, (subtitle,)))

    assert stage.name == "validator"
    assert result.stage_name == "validator"


def test_subtitle_validator_stage_rejects_missing_subtitles(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "input.wav"
    validator = FakeValidator(
        validation=SubtitleValidation(
            source_path=source_path,
            result=ValidationResult(),
        ),
        requests=[],
    )
    stage = SubtitleValidatorStage(validator=validator)

    with pytest.raises(MissingSubtitlesToValidateError):
        stage.run(_context(source_path, ()))

    assert validator.requests == []


def test_subtitle_validator_stage_rejects_invalid_validator() -> None:
    with pytest.raises(InvalidSubtitleValidatorError):
        SubtitleValidatorStage(validator=object())


def test_subtitle_validator_stage_rejects_invalid_validation_return(
    tmp_path: Path,
) -> None:
    stage = SubtitleValidatorStage(validator=InvalidValidationValidator())

    with pytest.raises(InvalidSubtitleValidationError, match="SubtitleValidation"):
        stage.run(_context(tmp_path / "input.wav", (_subtitle(),)))


def test_subtitle_validator_stage_rejects_mismatched_source_path(
    tmp_path: Path,
) -> None:
    validator = FakeValidator(
        validation=SubtitleValidation(
            source_path=tmp_path / "other.wav",
            result=ValidationResult(),
        ),
        requests=[],
    )
    stage = SubtitleValidatorStage(validator=validator)

    with pytest.raises(InvalidSubtitleValidationError, match="source path"):
        stage.run(_context(tmp_path / "input.wav", (_subtitle(),)))


def test_subtitle_validator_stage_raises_for_validation_issues(
    tmp_path: Path,
) -> None:
    issue = _issue()
    validator = FakeValidator(
        validation=SubtitleValidation(
            source_path=tmp_path / "input.wav",
            result=ValidationResult(issues=(issue,)),
        ),
        requests=[],
    )
    stage = SubtitleValidatorStage(validator=validator)

    with pytest.raises(SubtitleValidationFailedError) as error:
        stage.run(_context(tmp_path / "input.wav", (_subtitle(),)))

    assert error.value.issues == (issue,)


def test_subtitle_validation_requires_validation_result() -> None:
    with pytest.raises(TypeError, match="ValidationResult"):
        SubtitleValidation(source_path=Path("input.wav"), result=object())


def test_subtitle_validation_is_immutable() -> None:
    validation = SubtitleValidation(
        source_path=Path("input.wav"),
        result=ValidationResult(),
    )

    with pytest.raises(FrozenInstanceError):
        validation.result = ValidationResult(issues=(_issue(),))
