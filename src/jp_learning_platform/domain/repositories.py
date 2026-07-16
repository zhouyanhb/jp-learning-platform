"""Domain repository interfaces."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from jp_learning_platform.domain.models import Document


@runtime_checkable
class DocumentRepository(Protocol):
    """Persistence boundary for subtitle pipeline documents."""

    def save(self, document: Document) -> None:
        """Persist a document."""

    def get(self, source_path: Path) -> Document | None:
        """Return a document by source path when one exists."""


__all__ = ["DocumentRepository"]
