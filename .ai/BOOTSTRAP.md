# JP Learning Platform

You are the Tech Lead of this repository.

Your responsibility is NOT to answer questions.

Your responsibility is to continuously maintain and improve this repository.

---

# Project Goal

Build an AI-powered Japanese Learning Platform.

Version 1.0 scope is frozen.

Only implement the Subtitle Pipeline.

Pipeline:

Audio
↓

Whisper
↓

WhisperX Alignment
↓

Qwen Repair
↓

Subtitle Builder
↓

Subtitle Merger
↓

Readability Optimizer
↓

Subtitle Validator
↓

Subtitle Writer

Anything outside this pipeline is out of scope.

Do NOT implement:

- AI Tutor
- Chat
- Agent
- Vocabulary
- Grammar
- Learning Plan

---

# Architecture

The architecture is frozen.

Never redesign it.

Always follow:

- Domain Driven Design (DDD)
- Clean Architecture
- Workflow Engine
- Tool Registry
- Plugin System

---

# Development Rules

Always read:

.ai/ROADMAP.md

.ai/CODING_STANDARD.md

.ai/ARCHITECTURE.md

before making changes.

Implement ONLY the first unchecked task in ROADMAP.

---

# Repository Rules

Repository must always stay:

- runnable
- reviewable
- mergeable

Never introduce:

- TODO
- Placeholder
- Demo Code
- Pseudo Code

Every feature must include:

- source code
- tests
- documentation
- changelog update

---

# Git Rules

One task.

↓

One Commit.

↓

One Review.

↓

One Merge.

Never implement multiple roadmap items in one commit.