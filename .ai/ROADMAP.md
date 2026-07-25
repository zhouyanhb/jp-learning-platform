# JP Learning Platform Roadmap

Version 1.0

---

## Phase 1 Foundation

- [x] Commit0001 Repository Initialize

Repository

Python

CI

README

Project Configuration

---

- [x] Commit0002 Architecture

Architecture Documentation

DDD

Clean Architecture

Workflow

Plugin

---

## Phase 2 Domain

- [x] Commit0003 Domain Models

Word

Sentence

Subtitle

Segment

Document

PipelineContext

---

- [x] Commit0004 Domain Services

Validation

Factory

Repository Interface

---

## Phase 3 Workflow

- [x] Commit0005 Workflow Runtime

Workflow

Stage

Pipeline

StageResult

Execution Engine

---

## Phase 4 Infrastructure

- [x] Commit0006 Tool Registry

---

- [x] Commit0007 Plugin System

---

## Phase 5 Subtitle Pipeline

- [x] Commit0008 Audio Loader

- [x] Commit0009 Whisper Stage

- [x] Commit0010 WhisperX Alignment

- [x] Commit0011 Qwen Repair

- [x] Commit0012 Subtitle Builder

- [x] Commit0013 Subtitle Merger

- [x] Commit0014 Readability Optimizer

- [x] Commit0015 Subtitle Validator

- [x] Commit0016 Subtitle Writer

---

## Phase 6 Release

- [x] Commit0017 Release Version 1.0

---

## Maintenance

- [x] Maintenance Homophone Candidate Prefiltering and Benchmark

Limit contextual language-model scoring to a small number of suspicious
same-reading targets per sentence and record a reproducible before/after
runtime comparison.

- [x] Maintenance Sentence-initial Discourse Marker Punctuation

Restore Japanese commas after sentence-initial discourse markers without
changing subtitle boundaries or timing.

- [x] Maintenance Cross-segment Dependent Continuation Merge

Merge adjacent same-speaker sentence fragments when the next segment begins
with a Japanese dependent continuation and timing is contiguous.

- [x] Maintenance Japanese Learning Word Normalization

Normalize aligned Japanese token fragments into learning-oriented word units
while preserving timing, confidence, and speaker metadata.

- [x] Maintenance Japanese Inflectional Learning Units

Merge inflectional auxiliaries and sahen verb constructions into complete
learning units using Sudachi morphology without sentence-specific replacements.

- [x] Maintenance Conservative Homophone Acceptance

Require independent ASR and contextual evidence before applying a same-reading
replacement so the resolver fails closed when semantic evidence is ambiguous.

- [x] Maintenance Document-consistent Homophone Propagation

Propagate an unambiguous strictly confirmed same-reading correction to matching
high-confidence occurrences within the same document.

- [x] Maintenance Weak-ratio Homophone Propagation

Allow an unambiguous document-confirmed correction to propagate when the local
candidate beats the original but only the strict score-ratio gate failed.

- [x] Maintenance Content-addressed Pipeline Cache and Audio Normalization

Deduplicate identical audio work by complete configuration, reuse compatible
stage contexts, and normalize unsupported media once through an atomic cache.

- [x] Maintenance Demand-driven Audio Normalization

Preserve compatible source audio for consumers that can decode it directly,
and normalize only before a stage that explicitly requires deterministic PCM.

- [x] Maintenance Cold-run Audio Compatibility

Keep required media compatibility conversion active when cache reuse is
disabled, while preserving cold-run timing and stage ordering.

- [x] Maintenance Local Nominal Reanalysis

Reanalyze short contiguous nominal morphology windows and merge only those
that Sudachi independently recognizes as one complete noun.

- [x] Maintenance Alignment Phase Timing

Measure and report forced alignment, pyannote diarization, and speaker
assignment separately while retaining the aggregate alignment duration.

- [x] Maintenance Robust Sentence Boundaries

Recover sentence boundaries from sentence-final particles and alignment-held
silence, preserve connective clauses, and prevent subtitle re-merging across
long semantic boundaries.

- [x] Maintenance Evidence-based Question Boundaries

Require independent timing evidence before treating an aligned question
particle as a sentence boundary, without lexical exclusion lists.

- [x] Maintenance Structural Learning Word Units

Normalize identifiers, numeric counters, katakana compounds, and compatible
auxiliary chains from morphology and character classes without lexical rules.
