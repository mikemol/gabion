from __future__ import annotations

from dataclasses import dataclass

from gabion.server_core.progress_contracts import ProgressStageContract


@dataclass(frozen=True)
class ProgressPrimitives:
    lsp_progress_notification_method: str = ProgressStageContract.lsp_progress_notification_method
    lsp_progress_token_v2: str = ProgressStageContract.lsp_progress_token_v2
    canonical_progress_event_schema_v2: str = ProgressStageContract.canonical_progress_event_schema_v2
    progress_deadline_flush_margin_seconds: float = ProgressStageContract.progress_deadline_flush_margin_seconds
    progress_deadline_flush_seconds: float = ProgressStageContract.progress_deadline_flush_seconds
    progress_deadline_watchdog_seconds: float = ProgressStageContract.progress_deadline_watchdog_seconds
    progress_heartbeat_poll_seconds: float = ProgressStageContract.progress_heartbeat_poll_seconds
    build_phase_progress_v2 = staticmethod(ProgressStageContract.build_phase_progress_v2)
    incremental_progress_obligations = staticmethod(ProgressStageContract.incremental_progress_obligations)
    progress_heartbeat_seconds = staticmethod(ProgressStageContract.progress_heartbeat_seconds)


def default_progress_primitives() -> ProgressPrimitives:
    return ProgressPrimitives()
