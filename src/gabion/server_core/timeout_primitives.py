from __future__ import annotations

from dataclasses import dataclass

from gabion.server_core.timeout_runtime import TimeoutStageRuntime


@dataclass(frozen=True)
class TimeoutPrimitives:
    analysis_timeout_budget_ns = staticmethod(TimeoutStageRuntime.analysis_timeout_budget_ns)
    analysis_timeout_total_ticks = staticmethod(TimeoutStageRuntime.analysis_timeout_total_ticks)
    timeout_context_payload = staticmethod(TimeoutStageRuntime.timeout_context_payload)
    deadline_profile_sample_interval = staticmethod(TimeoutStageRuntime.deadline_profile_sample_interval)


def default_timeout_primitives() -> TimeoutPrimitives:
    return TimeoutPrimitives()
