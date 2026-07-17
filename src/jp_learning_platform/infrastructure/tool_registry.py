"""Registry for resolving infrastructure tool adapters."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Protocol


def _normalize_name(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")

    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty.")

    return normalized


class RegisteredTool(Protocol):
    """Minimum contract for infrastructure tools resolved by the registry."""

    name: str


class ToolRegistryError(RuntimeError):
    """Base error for tool registry failures."""


class DuplicateToolError(ToolRegistryError):
    """Raised when registering a tool name that already exists."""

    def __init__(self, tool_name: str) -> None:
        self.tool_name = tool_name
        super().__init__(f"Tool already registered: {tool_name}")


class ToolNotFoundError(ToolRegistryError):
    """Raised when resolving a tool that has not been registered."""

    def __init__(self, tool_name: str) -> None:
        self.tool_name = tool_name
        super().__init__(f"Tool not registered: {tool_name}")


@dataclass(slots=True)
class ToolRegistry:
    """Resolve external tool adapters by name."""

    _tools: dict[str, RegisteredTool] = field(default_factory=dict, init=False)

    def register(self, tool: RegisteredTool) -> None:
        tool_name = _tool_name(tool)
        if tool_name in self._tools:
            raise DuplicateToolError(tool_name)

        self._tools[tool_name] = tool

    def register_many(self, tools: Iterable[RegisteredTool]) -> None:
        tool_tuple = tuple(tools)
        tool_names = tuple(_tool_name(tool) for tool in tool_tuple)
        names_seen: set[str] = set()

        for tool_name in tool_names:
            if tool_name in names_seen or tool_name in self._tools:
                raise DuplicateToolError(tool_name)

            names_seen.add(tool_name)

        for tool_name, tool in zip(tool_names, tool_tuple):
            self._tools[tool_name] = tool

    def resolve(self, tool_name: str) -> RegisteredTool:
        normalized_name = _normalize_name(tool_name, "tool_name")
        tool = self._tools.get(normalized_name)
        if tool is None:
            raise ToolNotFoundError(normalized_name)

        return tool

    def get(self, tool_name: str) -> RegisteredTool | None:
        normalized_name = _normalize_name(tool_name, "tool_name")
        return self._tools.get(normalized_name)

    def contains(self, tool_name: str) -> bool:
        normalized_name = _normalize_name(tool_name, "tool_name")
        return normalized_name in self._tools

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._tools)

    @property
    def tools(self) -> tuple[RegisteredTool, ...]:
        return tuple(self._tools.values())


def _tool_name(tool: RegisteredTool) -> str:
    return _normalize_name(getattr(tool, "name", None), "tool.name")


__all__ = [
    "DuplicateToolError",
    "RegisteredTool",
    "ToolNotFoundError",
    "ToolRegistry",
    "ToolRegistryError",
]
