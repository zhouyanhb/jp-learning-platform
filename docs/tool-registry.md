# Tool Registry

The tool registry is the infrastructure boundary for resolving external tools.
Workflow code depends on contracts and asks for registered tools by name instead
of instantiating SDKs directly.

## Registered Tool

A registered tool is any object with a non-empty `name` attribute. The registry
does not define SDK behavior and does not call external systems. Concrete
adapters are introduced by later infrastructure and subtitle pipeline roadmap
items.

## Registry

`ToolRegistry` stores tools by normalized name. Registration fails when the name
already exists, and resolution fails with a typed exception when no tool is
registered for a requested name.

## Errors

- `DuplicateToolError` is raised when two tools use the same name.
- `ToolNotFoundError` is raised when a requested tool has not been registered.

These errors fail fast so misconfigured workflows are visible before external
SDK work begins.
