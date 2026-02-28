# gabion:decision_protocol_module
from __future__ import annotations

import os
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import Iterator

from gabion.analysis.derivation_cache import DerivationCacheConfig, derivation_cache_config_scope
from gabion.analysis.projection_registry import (
    ProjectionRegistryRuntimeConfig,
    projection_registry_runtime_config_scope,
)
from gabion.commands.transport_policy import TransportOverrideConfig, transport_override_scope
from gabion.invariants import ProofModeConfig, proof_mode_config_scope
from gabion.order_contract import OrderPolicy, OrderRuntimeConfig, order_runtime_config_scope


_STRICT_VALUES = {"1", "true", "yes", "on", "strict"}


@dataclass(frozen=True)
class RuntimePolicyConfig:
    proof_mode_enabled: bool = False
    order_policy: OrderPolicy | None = None
    order_caller_sorted: bool = False
    order_telemetry_enabled: bool = False
    order_enforce_canonical_allowlist: bool = False
    order_deadline_probe_enabled: bool = False
    transport_direct_requested: bool | None = None
    transport_override_record_json: str | None = None
    derivation_cache_max_entries: int = 4096
    projection_registry_gas_limit: int = 1_000_000


def _env_flag(name: str) -> bool:
    value = os.getenv(name, "")
    return value.strip().lower() in _STRICT_VALUES


# gabion:boundary_normalization
def _env_optional_policy(name: str) -> OrderPolicy | None:
    value = os.getenv(name)
    if value is None:
        return None
    normalized = value.strip().lower()
    if not normalized:
        return None
    if normalized in {"off", "false", "0"}:
        return OrderPolicy.SORT
    if normalized in {"on", "true", "1"}:
        return OrderPolicy.ENFORCE
    for candidate in OrderPolicy:
        if candidate.value == normalized:
            return candidate
    return None


def runtime_policy_from_env() -> RuntimePolicyConfig:
    direct_requested = _env_flag("GABION_DIRECT_RUN")
    override_record_json = os.getenv("GABION_OVERRIDE_RECORD_JSON")
    return RuntimePolicyConfig(
        proof_mode_enabled=_env_flag("GABION_PROOF_MODE"),
        order_policy=_env_optional_policy("GABION_ORDER_POLICY"),
        order_caller_sorted=os.getenv("GABION_CALLER_SORTED") == "1",
        order_telemetry_enabled=os.getenv("GABION_ORDER_TELEMETRY") == "1",
        order_enforce_canonical_allowlist=_env_flag(
            "GABION_ENFORCE_CANONICAL_SORT_ALLOWLIST"
        ),
        order_deadline_probe_enabled=os.getenv("GABION_ORDER_DEADLINE_PROBE") == "1",
        transport_direct_requested=(True if direct_requested else None),
        transport_override_record_json=(
            override_record_json.strip()
            if isinstance(override_record_json, str) and override_record_json.strip()
            else None
        ),
        derivation_cache_max_entries=max(
            1,
            int(os.getenv("GABION_DERIVATION_CACHE_MAX_ENTRIES", "4096") or "4096"),
        ),
        projection_registry_gas_limit=max(
            1,
            int(os.getenv("GABION_PROJECTION_REGISTRY_GAS_LIMIT", "1000000") or "1000000"),
        ),
    )




def apply_runtime_policy(config: RuntimePolicyConfig) -> None:
    from gabion.analysis.derivation_cache import set_derivation_cache_config
    from gabion.analysis.projection_registry import set_projection_registry_runtime_config
    from gabion.commands.transport_policy import set_transport_override
    from gabion.invariants import set_proof_mode_config
    from gabion.order_contract import set_order_runtime_config

    set_proof_mode_config(ProofModeConfig(enabled=config.proof_mode_enabled))
    set_order_runtime_config(
        OrderRuntimeConfig(
            default_policy=config.order_policy,
            legacy_caller_sorted=config.order_caller_sorted,
            telemetry_enabled=config.order_telemetry_enabled,
            enforce_canonical_allowlist=config.order_enforce_canonical_allowlist,
            deadline_probe_enabled=config.order_deadline_probe_enabled,
        )
    )
    set_transport_override(
        TransportOverrideConfig(
            direct_requested=config.transport_direct_requested,
            override_record_json=config.transport_override_record_json,
        )
    )
    set_derivation_cache_config(
        DerivationCacheConfig(max_entries=config.derivation_cache_max_entries)
    )
    set_projection_registry_runtime_config(
        ProjectionRegistryRuntimeConfig(gas_limit=config.projection_registry_gas_limit)
    )


def apply_runtime_policy_from_env() -> None:
    apply_runtime_policy(runtime_policy_from_env())


@contextmanager
def runtime_policy_scope(config: RuntimePolicyConfig) -> Iterator[None]:
    with ExitStack() as stack:
        stack.enter_context(
            proof_mode_config_scope(ProofModeConfig(enabled=config.proof_mode_enabled))
        )
        stack.enter_context(
            order_runtime_config_scope(
                OrderRuntimeConfig(
                    default_policy=config.order_policy,
                    legacy_caller_sorted=config.order_caller_sorted,
                    telemetry_enabled=config.order_telemetry_enabled,
                    enforce_canonical_allowlist=config.order_enforce_canonical_allowlist,
                    deadline_probe_enabled=config.order_deadline_probe_enabled,
                )
            )
        )
        stack.enter_context(
            transport_override_scope(
                TransportOverrideConfig(
                    direct_requested=config.transport_direct_requested,
                    override_record_json=config.transport_override_record_json,
                )
            )
        )
        stack.enter_context(
            derivation_cache_config_scope(
                DerivationCacheConfig(max_entries=config.derivation_cache_max_entries)
            )
        )
        stack.enter_context(
            projection_registry_runtime_config_scope(
                ProjectionRegistryRuntimeConfig(
                    gas_limit=config.projection_registry_gas_limit
                )
            )
        )
        yield
