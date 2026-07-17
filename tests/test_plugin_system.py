from __future__ import annotations

from dataclasses import dataclass

import pytest

from jp_learning_platform.infrastructure import ToolRegistry
from jp_learning_platform.plugins import (
    DuplicatePluginError,
    PluginContext,
    PluginMetadata,
    PluginNotFoundError,
    PluginRegistration,
    PluginRegistry,
)


@dataclass(frozen=True, slots=True)
class FakeTool:
    name: str


@dataclass(frozen=True, slots=True)
class ToolPlugin:
    metadata: PluginMetadata
    tool: FakeTool

    def register(self, context: PluginContext) -> PluginRegistration:
        context.register_tool(self.tool)
        return PluginRegistration(
            plugin_name=self.metadata.name,
            tool_names=(self.tool.name,),
        )


@dataclass(frozen=True, slots=True)
class InvalidRegistrationPlugin:
    metadata: PluginMetadata

    def register(self, context: PluginContext) -> object:
        return object()


@dataclass(frozen=True, slots=True)
class MismatchedRegistrationPlugin:
    metadata: PluginMetadata

    def register(self, context: PluginContext) -> PluginRegistration:
        return PluginRegistration(plugin_name="other")


def _plugin(name: str, tool_name: str) -> ToolPlugin:
    return ToolPlugin(
        metadata=PluginMetadata(
            name=name,
            version="1.0.0",
            capabilities=(tool_name,),
        ),
        tool=FakeTool(name=tool_name),
    )


def test_plugin_metadata_normalizes_values() -> None:
    metadata = PluginMetadata(
        name="  audio  ",
        version="  1.0.0  ",
        capabilities=(" whisper ",),
    )

    assert metadata.name == "audio"
    assert metadata.version == "1.0.0"
    assert metadata.capabilities == ("whisper",)


def test_plugin_registry_activates_plugin_and_registers_tool() -> None:
    tool_registry = ToolRegistry()
    context = PluginContext(tool_registry=tool_registry)
    plugin = _plugin("audio", "whisper")
    registry = PluginRegistry()
    registry.add(plugin)

    registration = registry.activate("audio", context)

    assert registration.plugin_name == "audio"
    assert registration.tool_names == ("whisper",)
    assert tool_registry.resolve("whisper") is plugin.tool


def test_plugin_registry_rejects_duplicate_plugins() -> None:
    registry = PluginRegistry()
    registry.add(_plugin("audio", "whisper"))

    with pytest.raises(DuplicatePluginError) as error:
        registry.add(_plugin("audio", "ffmpeg"))

    assert error.value.plugin_name == "audio"


def test_plugin_registry_resolves_missing_plugins_with_typed_error() -> None:
    registry = PluginRegistry()

    with pytest.raises(PluginNotFoundError) as error:
        registry.resolve("missing")

    assert error.value.plugin_name == "missing"


def test_plugin_registry_adds_many_plugins_atomically() -> None:
    registry = PluginRegistry()
    registry.add(_plugin("audio", "whisper"))

    with pytest.raises(DuplicatePluginError):
        registry.add_many((_plugin("align", "whisperx"), _plugin("audio", "ffmpeg")))

    assert registry.names == ("audio",)
    assert not registry.contains("align")


def test_plugin_registry_activates_all_in_registration_order() -> None:
    tool_registry = ToolRegistry()
    context = PluginContext(tool_registry=tool_registry)
    first = _plugin("audio", "whisper")
    second = _plugin("video", "ffmpeg")
    registry = PluginRegistry()
    registry.add_many((first, second))

    registrations = registry.activate_all(context)

    assert tuple(registration.plugin_name for registration in registrations) == (
        "audio",
        "video",
    )
    assert tool_registry.names == ("whisper", "ffmpeg")


def test_plugin_context_requires_tool_registry() -> None:
    with pytest.raises(TypeError, match="tool_registry"):
        PluginContext(tool_registry=object())


def test_plugin_activation_rejects_invalid_registration_result() -> None:
    registry = PluginRegistry()
    registry.add(
        InvalidRegistrationPlugin(
            metadata=PluginMetadata(name="invalid", version="1.0.0"),
        )
    )

    with pytest.raises(TypeError, match="PluginRegistration"):
        registry.activate("invalid", PluginContext(tool_registry=ToolRegistry()))


def test_plugin_activation_rejects_mismatched_registration_name() -> None:
    registry = PluginRegistry()
    registry.add(
        MismatchedRegistrationPlugin(
            metadata=PluginMetadata(name="actual", version="1.0.0"),
        )
    )

    with pytest.raises(ValueError, match="registration name"):
        registry.activate("actual", PluginContext(tool_registry=ToolRegistry()))
