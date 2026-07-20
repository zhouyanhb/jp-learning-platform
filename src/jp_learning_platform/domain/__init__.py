"""Domain layer for subtitle pipeline business rules."""

from jp_learning_platform.domain.factory import DomainModelFactory
from jp_learning_platform.domain.models import (
    Document,
    PipelineContext,
    Segment,
    Sentence,
    SentenceBoundaryCandidate,
    Subtitle,
    TimeRange,
    Word,
)
from jp_learning_platform.domain.repositories import DocumentRepository
from jp_learning_platform.domain.validation import (
    DocumentValidator,
    DomainValidationError,
    ValidationCode,
    ValidationIssue,
    ValidationResult,
)

__all__ = [
    "Document",
    "DocumentRepository",
    "DocumentValidator",
    "DomainModelFactory",
    "DomainValidationError",
    "PipelineContext",
    "Segment",
    "Sentence",
    "SentenceBoundaryCandidate",
    "Subtitle",
    "TimeRange",
    "ValidationCode",
    "ValidationIssue",
    "ValidationResult",
    "Word",
]
