from __future__ import annotations

from dataclasses import dataclass

from gabion.server_core import command_orchestrator_primitives as legacy


@dataclass(frozen=True)
class TimeoutPrimitives:
    analysis_timeout_budget_ns = staticmethod(legacy._analysis_timeout_budget_ns)
    analysis_timeout_total_ticks = staticmethod(legacy._analysis_timeout_total_ticks)
    timeout_context_payload = staticmethod(legacy._timeout_context_payload)
    deadline_profile_sample_interval = staticmethod(legacy._deadline_profile_sample_interval)


def default_timeout_primitives() -> TimeoutPrimitives:
    return TimeoutPrimitives()
