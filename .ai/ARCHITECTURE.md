# Architecture

This repository uses:

- Domain Driven Design
- Clean Architecture
- Workflow Engine
- Tool Registry
- Plugin System

Architecture is frozen for Version 1.0.

---

# Layers

```
Application

↓

Workflow

↓

Domain

↓

Infrastructure

↓

SDK
```

---

# Workflow

Workflow orchestrates execution.

Workflow never contains business logic.

---

# Stage

One Stage.

One Responsibility.

Stages communicate only through PipelineContext.

---

# Domain

Domain owns all business rules.

Domain never depends on Infrastructure.

---

# Infrastructure

Infrastructure wraps third-party libraries.

Examples:

- Whisper
- WhisperX
- Qwen API
- FFmpeg

---

# Plugin

Plugins are optional capabilities.

Plugins register themselves.

Workflow depends on interfaces only.

---

# Tool Registry

All external tools are resolved through Tool Registry.

Workflow never instantiates SDK directly.

---

# Repository Layout

```
docs/

src/

tests/

scripts/

diagrams/
```

Repository structure is stable.

Avoid unnecessary restructuring.

---

# Architecture Specification

Detailed implementation documentation lives in:

```
docs/architecture.md
```

Layer boundary metadata lives in:

```
src/jp_learning_platform/architecture.py
```

Workflow runtime implementation lives in:

```
src/jp_learning_platform/workflow/runtime.py
```

Tool registry implementation lives in:

```
src/jp_learning_platform/infrastructure/tool_registry.py
```

Plugin system implementation lives in:

```
src/jp_learning_platform/plugins/system.py
```

Audio loader implementation lives in:

```
src/jp_learning_platform/infrastructure/audio_loader.py
```

Whisper stage implementation lives in:

```
src/jp_learning_platform/workflow/whisper_stage.py
```

WhisperX alignment stage implementation lives in:

```
src/jp_learning_platform/workflow/whisperx_alignment_stage.py
```

Qwen repair stage implementation lives in:

```
src/jp_learning_platform/workflow/qwen_repair_stage.py
```
