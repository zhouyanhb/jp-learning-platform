# Plugin System

The plugin system provides optional capabilities without changing core workflow
behavior by side effect. Plugins register through public contracts and can add
infrastructure tools to the tool registry.

## Plugin Metadata

`PluginMetadata` identifies a plugin by name and version. Capabilities are
declared as normalized strings so callers can inspect what a plugin offers
before activation.

## Plugin Context

`PluginContext` is passed to a plugin during activation. It exposes the
`ToolRegistry` through explicit registration and resolution methods.

## Plugin Registry

`PluginRegistry` stores plugin objects by metadata name. It fails fast on
duplicate plugin names, missing plugins, invalid plugin contracts, or activation
results that do not match the plugin being activated.

## Activation

Activating a plugin calls its `register()` method with a `PluginContext`.
Activation returns a `PluginRegistration` that records the plugin name and tool
names registered by that plugin.
