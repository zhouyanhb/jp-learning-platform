# Repository Rules

Version: 1.0

This document defines how the AI must work inside this repository.

These rules are mandatory.

If any instruction conflicts with this document, this document takes precedence.

---

# Primary Goal

Maintain a production-quality repository.

The repository must remain:

- runnable
- reviewable
- mergeable
- releasable

at every commit.

Never sacrifice repository quality for development speed.

---

# Working Principle

The repository is the source of truth.

Never treat conversations as the source of truth.

Always inspect the current repository before making changes.

Before modifying any code:

1. Read `.ai/BOOTSTRAP.md`
2. Read `.ai/ROADMAP.md`
3. Read `.ai/ARCHITECTURE.md`
4. Read `.ai/CODING_STANDARD.md`
5. Inspect the existing implementation
6. Understand the dependency graph
7. Then make changes

Never assume repository state.

Always verify.

---

# Development Unit

The smallest development unit is one roadmap item.

One roadmap item

↓

One implementation

↓

One commit

Never implement multiple roadmap items in one commit.

Never mix unrelated changes.

---

# Repository First

Repository health is always more important than feature completion.

Do not introduce temporary implementations.

Do not introduce incomplete code.

Do not introduce placeholder implementations.

Do not introduce fake implementations.

Do not introduce TODO comments.

Do not comment out unfinished code.

Every implementation must be production-ready.

---

# Existing Code

Respect existing code.

Prefer extending existing implementations over replacing them.

Never rewrite large modules unless explicitly required.

Never change public APIs without necessity.

Avoid unnecessary file movement.

Avoid unnecessary renaming.

Minimize unrelated diffs.

---

# Architecture Discipline

Architecture is frozen.

Never redesign:

- DDD
- Clean Architecture
- Workflow Engine
- Tool Registry
- Plugin System

When adding new code:

Follow existing architecture.

Do not invent new architectural patterns.

---

# Domain Rules

Business logic belongs only to Domain.

Workflow coordinates.

Infrastructure integrates external systems.

Tools wrap third-party SDKs.

Never mix responsibilities.

---

# Code Quality

Every change must:

- compile
- pass static analysis (if configured)
- pass tests (if configured)

Use:

- type hints
- dataclass(slots=True)
- pathlib
- logging
- Enum

Avoid:

- duplicated logic
- dead code
- magic values
- global mutable state

---

# Documentation

Documentation evolves together with code.

Whenever architecture changes:

Update:

- Architecture documentation
- Changelog
- Related design documents

Never allow documentation to become outdated.

---

# Testing

Every new feature requires tests.

Every bug fix requires a regression test.

Tests must:

- be deterministic
- be isolated
- avoid network access unless explicitly intended
- avoid hidden dependencies

Do not remove tests to make builds pass.

---

# Dependency Management

Prefer standard library first.

Introduce new dependencies only when they provide clear value.

Avoid unnecessary libraries.

When adding a dependency:

- justify its purpose
- keep dependency count minimal

---

# Git Rules

Every commit must:

- build successfully (if build exists)
- pass tests (if configured)
- update documentation when needed
- contain a meaningful commit message

Use Conventional Commits.

# Repository State

PROJECT_STATE.yaml is the authoritative summary of the repository status.

Every completed roadmap item MUST update PROJECT_STATE.yaml before creating the Git commit.

PROJECT_STATE.yaml should always reflect the current repository state.

At minimum it must contain:

- current version
- completed roadmap items
- current roadmap item
- next roadmap item
- latest commit
- repository health
- current test status
- known architectural decisions

Never allow PROJECT_STATE.yaml to become outdated.

Never skip updating PROJECT_STATE.yaml.

# Session History

SESSION_LOG.md records repository evolution.

Every completed roadmap item MUST append one session entry.

Each session should include:

- date
- roadmap item
- summary
- affected modules
- tests added
- commit hash

Never rewrite history.

Always append.

SESSION_LOG.md is chronological.


Examples:

feat(workflow): add pipeline runtime

fix(builder): preserve subtitle timestamps

refactor(domain): simplify sentence validation

docs(architecture): clarify workflow lifecycle

test(validator): add regression cases

---

# Pull Request Mindset

Before considering work complete, review the changes as if preparing a Pull Request.

Verify:

- Scope is limited to the intended roadmap item.
- No unrelated files were modified.
- Public APIs remain consistent unless intentionally changed.
- Documentation reflects the implementation.
- Tests cover the new behavior.
- Code follows repository standards.

---

# Error Handling

Fail fast.

Provide meaningful exceptions.

Never silently ignore failures.

Never swallow exceptions.

Prefer explicit error propagation.

---

# Performance

Optimize only when justified.

Avoid premature optimization.

Prioritize:

1. correctness
2. readability
3. maintainability
4. performance

---

# Security

Never hard-code:

- API Keys
- Tokens
- Passwords
- Secrets

Use configuration files or environment variables.

Never commit sensitive information.

---

# Decision Making

When multiple implementations are possible:

Choose the one that:

- is simpler
- follows the existing architecture
- minimizes future maintenance
- minimizes code duplication

Document significant trade-offs.

---

# Completion Checklist

A roadmap item is complete only if:

- implementation finished
- tests added or updated
- documentation updated
- code reviewed
- repository remains runnable
- commit created

Only then proceed to the next roadmap item.
