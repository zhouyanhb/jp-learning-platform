"""Plugin registration layer for optional capabilities."""

from jp_learning_platform.plugins.system import (
    DuplicatePluginError,
    Plugin,
    PluginContext,
    PluginMetadata,
    PluginNotFoundError,
    PluginRegistration,
    PluginRegistry,
    PluginRegistryError,
)

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
