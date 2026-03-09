from __future__ import annotations

from importlib import import_module

__all__ = [
    "aspf_normalization_idempotence_rule",
    "boundary_core_contract_rule",
    "branchless_rule",
    "defensive_fallback_rule",
    "fiber_filter_processor_contract_rule",
    "fiber_loop_structure_contract_rule",
    "fiber_noop_block_audit_rule",
    "fiber_normalization_contract_rule",
    "fiber_scalar_sentinel_contract_rule",
    "fiber_type_dispatch_contract_rule",
    "no_legacy_monolith_import_rule",
    "no_monkeypatch_rule",
    "orchestrator_primitive_barrel_rule",
    "runtime_narrowing_boundary_rule",
    "test_sleep_hygiene_rule",
    "test_subprocess_hygiene_rule",
    "typing_surface_rule",
]


def __getattr__(name: str) -> object:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f"{__name__}.{name}")
    globals()[name] = module
    return module

