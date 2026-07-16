# Coding Standard

Python 3.12

---

## Language

Use:

- dataclass(slots=True)
- Enum
- Path
- logging
- pytest
- type hints

Avoid:

- dict as business object
- print()
- global state
- magic numbers

---

## Domain

Business objects must be immutable whenever possible.

Use explicit Domain Models.

Core Models:

- Word
- Sentence
- Subtitle
- Segment
- Document
- PipelineContext

---

## Clean Architecture

Domain

↓

Application

↓

Workflow

↓

Infrastructure

↓

External SDK

Dependencies always point inward.

---

## Testing

Every feature requires:

- unit tests

Every bug fix requires:

- regression test

Tests must be deterministic.

---

## Documentation

Every code change requiring architectural understanding must update documentation.

Never let documentation fall behind implementation.