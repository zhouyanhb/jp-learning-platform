"""Version 1.0 architecture boundary definitions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping


class ProjectLayer(Enum):
    """Internal architectural layers."""

    DOMAIN = "domain"
    APPLICATION = "application"
    WORKFLOW = "workflow"
    INFRASTRUCTURE = "infrastructure"
    PLUGINS = "plugins"


@dataclass(frozen=True, slots=True)
class LayerDefinition:
    layer: ProjectLayer
    package: str
    responsibility: str


LAYER_DEFINITIONS: tuple[LayerDefinition, ...] = (
    LayerDefinition(
        layer=ProjectLayer.DOMAIN,
        package="jp_learning_platform.domain",
        responsibility="Own subtitle pipeline business rules.",
    ),
    LayerDefinition(
        layer=ProjectLayer.APPLICATION,
        package="jp_learning_platform.application",
        responsibility="Expose use-case boundaries for application entrypoints.",
    ),
    LayerDefinition(
        layer=ProjectLayer.WORKFLOW,
        package="jp_learning_platform.workflow",
        responsibility="Coordinate stage execution without business rules.",
    ),
    LayerDefinition(
        layer=ProjectLayer.INFRASTRUCTURE,
        package="jp_learning_platform.infrastructure",
        responsibility="Adapt external tools to project contracts.",
    ),
    LayerDefinition(
        layer=ProjectLayer.PLUGINS,
        package="jp_learning_platform.plugins",
        responsibility="Register optional capabilities through public contracts.",
    ),
)

ALLOWED_LAYER_DEPENDENCIES: Mapping[ProjectLayer, frozenset[ProjectLayer]] = (
    MappingProxyType(
        {
            ProjectLayer.DOMAIN: frozenset(),
            ProjectLayer.APPLICATION: frozenset({ProjectLayer.DOMAIN}),
            ProjectLayer.WORKFLOW: frozenset(
                {
                    ProjectLayer.APPLICATION,
                    ProjectLayer.DOMAIN,
                }
            ),
            ProjectLayer.INFRASTRUCTURE: frozenset(
                {
                    ProjectLayer.WORKFLOW,
                    ProjectLayer.APPLICATION,
                    ProjectLayer.DOMAIN,
                }
            ),
            ProjectLayer.PLUGINS: frozenset(
                {
                    ProjectLayer.INFRASTRUCTURE,
                    ProjectLayer.WORKFLOW,
                    ProjectLayer.APPLICATION,
                    ProjectLayer.DOMAIN,
                }
            ),
        }
    )
)


def layer_definition(layer: ProjectLayer) -> LayerDefinition:
    for definition in LAYER_DEFINITIONS:
        if definition.layer is layer:
            return definition

    raise ValueError(f"Unknown architecture layer: {layer.value}")


def is_dependency_allowed(source: ProjectLayer, target: ProjectLayer) -> bool:
    if source is target:
        return True

    return target in ALLOWED_LAYER_DEPENDENCIES[source]


__all__ = [
    "ALLOWED_LAYER_DEPENDENCIES",
    "LAYER_DEFINITIONS",
    "LayerDefinition",
    "ProjectLayer",
    "is_dependency_allowed",
    "layer_definition",
]
