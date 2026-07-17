from __future__ import annotations

from dataclasses import dataclass

import pytest

from jp_learning_platform.infrastructure import (
    DuplicateToolError,
    ToolNotFoundError,
    ToolRegistry,
)


@dataclass(frozen=True, slots=True)
class FakeTool:
    name: str


def test_registry_resolves_registered_tool_by_name() -> None:
    registry = ToolRegistry()
    tool = FakeTool(name="whisper")

    registry.register(tool)

    assert registry.resolve("whisper") is tool
    assert registry.get("whisper") is tool
    assert registry.contains("whisper")


def test_registry_normalizes_tool_names() -> None:
    registry = ToolRegistry()
    tool = FakeTool(name="  ffmpeg  ")

    registry.register(tool)

    assert registry.names == ("ffmpeg",)
    assert registry.resolve(" ffmpeg ") is tool


def test_registry_rejects_duplicate_tool_names() -> None:
    registry = ToolRegistry()
    registry.register(FakeTool(name="whisper"))

    with pytest.raises(DuplicateToolError) as error:
        registry.register(FakeTool(name="whisper"))

    assert error.value.tool_name == "whisper"


def test_registry_raises_for_missing_tools() -> None:
    registry = ToolRegistry()

    with pytest.raises(ToolNotFoundError) as error:
        registry.resolve("qwen")

    assert error.value.tool_name == "qwen"


def test_registry_rejects_invalid_tool_names() -> None:
    registry = ToolRegistry()

    with pytest.raises(ValueError, match="tool.name"):
        registry.register(FakeTool(name=" "))


def test_registry_registers_many_tools_in_order() -> None:
    registry = ToolRegistry()
    whisper = FakeTool(name="whisper")
    ffmpeg = FakeTool(name="ffmpeg")

    registry.register_many((whisper, ffmpeg))

    assert registry.names == ("whisper", "ffmpeg")
    assert registry.tools == (whisper, ffmpeg)


def test_registry_registers_many_tools_atomically() -> None:
    registry = ToolRegistry()
    registry.register(FakeTool(name="whisper"))

    with pytest.raises(DuplicateToolError):
        registry.register_many((FakeTool(name="ffmpeg"), FakeTool(name="whisper")))

    assert registry.names == ("whisper",)
    assert not registry.contains("ffmpeg")
