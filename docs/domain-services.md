# Domain Services

Domain services hold business operations that do not belong to a single model.
They remain independent from workflow orchestration, infrastructure adapters,
and external SDKs.

## Factory

`DomainModelFactory` builds the core immutable domain models from primitive
values. It centralizes creation of `TimeRange` objects so workflow and
infrastructure code do not duplicate model construction logic.

## Validation

`DocumentValidator` checks document-level consistency across model collections.
It verifies segment ordering, segment positions, subtitle ordering, and
one-based subtitle indexes. Validation returns a `ValidationResult` containing
structured `ValidationIssue` values. Callers that need fail-fast behavior can
use `raise_for_errors()`.

## Repository Interface

`DocumentRepository` is a domain protocol for persistence. Infrastructure may
implement the protocol later, but the domain layer owns the interface so outer
layers depend inward.
