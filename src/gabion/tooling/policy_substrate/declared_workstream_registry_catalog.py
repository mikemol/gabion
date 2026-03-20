"""Declared workstream registry catalog.

This module is the canonical assembly seam for planning-substrate workstream
registries.  Keep provider ordering stable here so callers do not need to
search inside invariant_graph.py for the registry integration point.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from gabion.tooling.policy_substrate.boundary_ingress_convergence_registry import (
    boundary_ingress_convergence_workstream_registry,
)
from gabion.tooling.policy_substrate.connectivity_synergy_registry import (
    connectivity_synergy_workstream_registries,
)
from gabion.tooling.policy_substrate.dataflow_grammar_readiness_registry import (
    dataflow_grammar_readiness_workstream_registry,
)
from gabion.tooling.policy_substrate.delivery_flow_momentum_registry import (
    delivery_flow_momentum_workstream_registry,
)
from gabion.tooling.policy_substrate.delivery_flow_reliability_registry import (
    delivery_flow_reliability_workstream_registry,
)
from gabion.tooling.policy_substrate.local_ci_repro_viability_registry import (
    local_ci_repro_viability_workstream_registry,
)
from gabion.tooling.policy_substrate.policy_rule_frontmatter_migration_registry import (
    prf_workstream_registry,
)
from gabion.tooling.policy_substrate.public_surface_normalization_registry import (
    public_surface_normalization_workstream_registry,
)
from gabion.tooling.policy_substrate.projection_semantic_fragment_phase5_registry import (
    phase5_workstream_registry,
)
from gabion.tooling.policy_substrate.runtime_context_injection_registry import (
    runtime_context_injection_workstream_registry,
)
from gabion.tooling.policy_substrate.structural_anti_pattern_convergence_registry import (
    structural_anti_pattern_convergence_workstream_registry,
)
from gabion.tooling.policy_substrate.surface_contract_convergence_registry import (
    surface_contract_convergence_workstream_registry,
)
from gabion.tooling.policy_substrate.unit_test_readiness_registry import (
    unit_test_readiness_workstream_registry,
)
from gabion.tooling.policy_substrate.wrapper_retirement_drain_registry import (
    wrapper_retirement_drain_workstream_registry,
)
from gabion.tooling.policy_substrate.workstream_registry import WorkstreamRegistry


RegistryLoader = Callable[[], WorkstreamRegistry | None]
RegistryFamilyLoader = Callable[[], tuple[WorkstreamRegistry, ...]]


@dataclass(frozen=True)
class DeclaredWorkstreamRegistryProvider:
    provider_id: str
    loader: RegistryLoader


@dataclass(frozen=True)
class DeclaredWorkstreamRegistryFamilyProvider:
    provider_id: str
    loader: RegistryFamilyLoader


def declared_workstream_registry_catalog() -> tuple[
    DeclaredWorkstreamRegistryProvider | DeclaredWorkstreamRegistryFamilyProvider,
    ...,
]:
    return (
        DeclaredWorkstreamRegistryProvider(
            provider_id="phase5",
            loader=phase5_workstream_registry,
        ),
        DeclaredWorkstreamRegistryProvider(
            provider_id="prf",
            loader=prf_workstream_registry,
        ),
        DeclaredWorkstreamRegistryProvider(
            provider_id="scc",
            loader=surface_contract_convergence_workstream_registry,
        ),
        DeclaredWorkstreamRegistryProvider(
            provider_id="rci",
            loader=runtime_context_injection_workstream_registry,
        ),
        DeclaredWorkstreamRegistryProvider(
            provider_id="bic",
            loader=boundary_ingress_convergence_workstream_registry,
        ),
        DeclaredWorkstreamRegistryProvider(
            provider_id="utr",
            loader=unit_test_readiness_workstream_registry,
        ),
        DeclaredWorkstreamRegistryProvider(
            provider_id="dgr",
            loader=dataflow_grammar_readiness_workstream_registry,
        ),
        DeclaredWorkstreamRegistryProvider(
            provider_id="dfr",
            loader=delivery_flow_reliability_workstream_registry,
        ),
        DeclaredWorkstreamRegistryProvider(
            provider_id="lcr",
            loader=local_ci_repro_viability_workstream_registry,
        ),
        DeclaredWorkstreamRegistryProvider(
            provider_id="dfm",
            loader=delivery_flow_momentum_workstream_registry,
        ),
        DeclaredWorkstreamRegistryProvider(
            provider_id="sac",
            loader=structural_anti_pattern_convergence_workstream_registry,
        ),
        DeclaredWorkstreamRegistryProvider(
            provider_id="wrd",
            loader=wrapper_retirement_drain_workstream_registry,
        ),
        DeclaredWorkstreamRegistryProvider(
            provider_id="psn",
            loader=public_surface_normalization_workstream_registry,
        ),
        DeclaredWorkstreamRegistryFamilyProvider(
            provider_id="connectivity_synergy",
            loader=connectivity_synergy_workstream_registries,
        ),
    )


def declared_workstream_registries() -> tuple[WorkstreamRegistry, ...]:
    registries: list[WorkstreamRegistry] = []
    for provider in declared_workstream_registry_catalog():
        match provider:
            case DeclaredWorkstreamRegistryProvider(loader=loader):
                registry = loader()
                if registry is not None:
                    registries.append(registry)
            case DeclaredWorkstreamRegistryFamilyProvider(loader=loader):
                registries.extend(loader())
            case _:
                continue
    return tuple(registries)
