"""Plugin system for optional project capabilities."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Protocol

from jp_learning_platform.infrastructure import RegisteredTool, ToolRegistry


def _normalize_name(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")

    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty.")

    return normalized


def _normalize_names(values: Iterable[str], field_name: str) -> tuple[str, ...]:
    try:
        names = tuple(values)
    except TypeError as error:
        raise TypeError(f"{field_name} must be iterable.") from error

    return tuple(
        _normalize_name(value, f"{field_name} item")
        for value in names
    )


@dataclass(frozen=True, slots=True)
class PluginMetadata:
    name: str
    version: str
    capabilities: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _normalize_name(self.name, "name"))
        object.__setattr__(
            self,
            "version",
            _normalize_name(self.version, "version"),
        )
        object.__setattr__(
            self,
            "capabilities",
            _normalize_names(self.capabilities, "capabilities"),
        )


@dataclass(frozen=True, slots=True)
class PluginRegistration:
    plugin_name: str
    tool_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "plugin_name",
            _normalize_name(self.plugin_name, "plugin_name"),
        )
        object.__setattr__(
            self,
            "tool_names",
            _normalize_names(self.tool_names, "tool_names"),
        )


@dataclass(frozen=True, slots=True)
class PluginContext:
    tool_registry: ToolRegistry

    def __post_init__(self) -> None:
        if not isinstance(self.tool_registry, ToolRegistry):
            raise TypeError("tool_registry must be a ToolRegistry.")

    def register_tool(self, tool: RegisteredTool) -> None:
        self.tool_registry.register(tool)

    def register_tools(self, tools: Iterable[RegisteredTool]) -> None:
        self.tool_registry.register_many(tools)

    def resolve_tool(self, tool_name: str) -> RegisteredTool:
        return self.tool_registry.resolve(tool_name)


class Plugin(Protocol):
    """Contract implemented by optional capability plugins."""

    metadata: PluginMetadata

    def register(self, context: PluginContext) -> PluginRegistration:
        """Register plugin capabilities into the provided context."""


class PluginRegistryError(RuntimeError):
    """Base error for plugin registry failures."""


class DuplicatePluginError(PluginRegistryError):
    """Raised when registering a plugin name that already exists."""

    def __init__(self, plugin_name: str) -> None:
        self.plugin_name = plugin_name
        super().__init__(f"Plugin already registered: {plugin_name}")


class PluginNotFoundError(PluginRegistryError):
    """Raised when resolving a plugin that has not been registered."""

    def __init__(self, plugin_name: str) -> None:
        self.plugin_name = plugin_name
        super().__init__(f"Plugin not registered: {plugin_name}")


@dataclass(slots=True)
class PluginRegistry:
    """Register and activate optional capability plugins."""

    _plugins: dict[str, Plugin] = field(default_factory=dict, init=False)

    def add(self, plugin: Plugin) -> None:
        plugin_name = _plugin_name(plugin)
        if plugin_name in self._plugins:
            raise DuplicatePluginError(plugin_name)

        _validate_plugin(plugin)
        self._plugins[plugin_name] = plugin

    def add_many(self, plugins: Iterable[Plugin]) -> None:
        plugin_tuple = tuple(plugins)
        plugin_names = tuple(_plugin_name(plugin) for plugin in plugin_tuple)
        names_seen: set[str] = set()

        for plugin, plugin_name in zip(plugin_tuple, plugin_names):
            if plugin_name in names_seen or plugin_name in self._plugins:
                raise DuplicatePluginError(plugin_name)

            _validate_plugin(plugin)
            names_seen.add(plugin_name)

        for plugin_name, plugin in zip(plugin_names, plugin_tuple):
            self._plugins[plugin_name] = plugin

    def resolve(self, plugin_name: str) -> Plugin:
        normalized_name = _normalize_name(plugin_name, "plugin_name")
        plugin = self._plugins.get(normalized_name)
        if plugin is None:
            raise PluginNotFoundError(normalized_name)

        return plugin

    def contains(self, plugin_name: str) -> bool:
        normalized_name = _normalize_name(plugin_name, "plugin_name")
        return normalized_name in self._plugins

    def activate(
        self,
        plugin_name: str,
        context: PluginContext,
    ) -> PluginRegistration:
        if not isinstance(context, PluginContext):
            raise TypeError("context must be a PluginContext.")

        plugin = self.resolve(plugin_name)
        registration = plugin.register(context)
        if not isinstance(registration, PluginRegistration):
            raise TypeError("plugin register must return a PluginRegistration.")

        expected_name = plugin.metadata.name
        if registration.plugin_name != expected_name:
            raise ValueError("plugin registration name must match plugin metadata.")

        return registration

    def activate_all(self, context: PluginContext) -> tuple[PluginRegistration, ...]:
        return tuple(self.activate(plugin_name, context) for plugin_name in self.names)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._plugins)

    @property
    def plugins(self) -> tuple[Plugin, ...]:
        return tuple(self._plugins.values())


def _plugin_metadata(plugin: Plugin) -> PluginMetadata:
    metadata = getattr(plugin, "metadata", None)
    if not isinstance(metadata, PluginMetadata):
        raise TypeError("plugin.metadata must be a PluginMetadata.")

    return metadata


def _plugin_name(plugin: Plugin) -> str:
    return _plugin_metadata(plugin).name


def _validate_plugin(plugin: Plugin) -> None:
    _plugin_metadata(plugin)
    if not callable(getattr(plugin, "register", None)):
        raise TypeError("plugin.register must be callable.")


__all__ = [
    "DuplicatePluginError",
    "Plugin",
    "PluginContext",
    "PluginMetadata",
    "PluginNotFoundError",
    "PluginRegistration",
    "PluginRegistry",
    "PluginRegistryError",
]
