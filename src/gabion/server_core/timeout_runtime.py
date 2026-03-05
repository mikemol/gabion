from __future__ import annotations

from dataclasses import dataclass

from gabion.server_core import command_orchestrator_primitives as legacy


@dataclass(frozen=True)
class TimeoutStageRuntime:
    analysis_timeout_budget_ns = staticmethod(legacy._analysis_timeout_budget_ns)
    analysis_timeout_total_ticks = staticmethod(legacy._analysis_timeout_total_ticks)
    deadline_profile_sample_interval = staticmethod(legacy._deadline_profile_sample_interval)
    timeout_context_payload = staticmethod(legacy._timeout_context_payload)


__all__ = ["TimeoutStageRuntime"]
