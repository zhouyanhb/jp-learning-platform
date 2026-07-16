from __future__ import annotations

import importlib

import pytest

from jp_learning_platform.architecture import (
    ALLOWED_LAYER_DEPENDENCIES,
    LAYER_DEFINITIONS,
    ProjectLayer,
    is_dependency_allowed,
    layer_definition,
)


def test_architecture_layer_packages_are_importable() -> None:
    for definition in LAYER_DEFINITIONS:
        module = importlib.import_module(definition.package)

        assert module.__name__ == definition.package


def test_every_layer_has_dependency_rule() -> None:
    assert set(ALLOWED_LAYER_DEPENDENCIES) == set(ProjectLayer)


@pytest.mark.parametrize("layer", list(ProjectLayer))
def test_layer_definitions_are_registered(layer: ProjectLayer) -> None:
    definition = layer_definition(layer)

    assert definition.layer is layer
    assert definition.package.startswith("jp_learning_platform.")
    assert definition.responsibility


def test_domain_has_no_outer_dependencies() -> None:
    outer_layers = set(ProjectLayer) - {ProjectLayer.DOMAIN}

    assert all(
        not is_dependency_allowed(ProjectLayer.DOMAIN, layer)
        for layer in outer_layers
    )


def test_clean_architecture_dependency_boundaries() -> None:
    assert is_dependency_allowed(ProjectLayer.APPLICATION, ProjectLayer.DOMAIN)
    assert is_dependency_allowed(ProjectLayer.WORKFLOW, ProjectLayer.APPLICATION)
    assert is_dependency_allowed(ProjectLayer.WORKFLOW, ProjectLayer.DOMAIN)
    assert is_dependency_allowed(ProjectLayer.INFRASTRUCTURE, ProjectLayer.WORKFLOW)
    assert is_dependency_allowed(ProjectLayer.PLUGINS, ProjectLayer.INFRASTRUCTURE)

    assert not is_dependency_allowed(ProjectLayer.DOMAIN, ProjectLayer.APPLICATION)
    assert not is_dependency_allowed(ProjectLayer.APPLICATION, ProjectLayer.WORKFLOW)
    assert not is_dependency_allowed(ProjectLayer.WORKFLOW, ProjectLayer.INFRASTRUCTURE)
