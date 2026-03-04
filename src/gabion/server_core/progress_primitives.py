from __future__ import annotations

from dataclasses import dataclass

from gabion.server_core import command_orchestrator_primitives as legacy


@dataclass(frozen=True)
class ProgressPrimitives:
    lsp_progress_notification_method: str = legacy._LSP_PROGRESS_NOTIFICATION_METHOD
    lsp_progress_token_v2: str = legacy._LSP_PROGRESS_TOKEN_V2
    canonical_progress_event_schema_v2: str = legacy._CANONICAL_PROGRESS_EVENT_SCHEMA_V2
    progress_deadline_flush_margin_seconds: float = legacy._PROGRESS_DEADLINE_FLUSH_MARGIN_SECONDS
    progress_deadline_flush_seconds: float = legacy._PROGRESS_DEADLINE_FLUSH_SECONDS
    progress_deadline_watchdog_seconds: float = legacy._PROGRESS_DEADLINE_WATCHDOG_SECONDS
    progress_heartbeat_poll_seconds: float = legacy._PROGRESS_HEARTBEAT_POLL_SECONDS
    build_phase_progress_v2 = staticmethod(legacy._build_phase_progress_v2)
    incremental_progress_obligations = staticmethod(legacy._incremental_progress_obligations)
    progress_heartbeat_seconds = staticmethod(legacy._progress_heartbeat_seconds)


def default_progress_primitives() -> ProgressPrimitives:
    return ProgressPrimitives()
