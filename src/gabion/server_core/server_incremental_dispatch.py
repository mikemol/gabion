from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from gabion.json_types import JSONObject, JSONValue
from gabion.server_core import command_orchestrator_primitives as orchestrator_primitives


def _write_bootstrap_incremental_artifacts(
    *,
    report_output_path: Path | None,
    report_section_journal_path: Path | None,
    report_phase_checkpoint_path: Path | None,
    witness_digest: str | None,
    root: Path,
    paths_requested: int,
    projection_rows: Sequence[Mapping[str, JSONValue]],
    phase_checkpoint_state: JSONObject,
) -> None:
    orchestrator_primitives._write_bootstrap_incremental_artifacts(
        report_output_path=report_output_path,
        report_section_journal_path=report_section_journal_path,
        report_phase_checkpoint_path=report_phase_checkpoint_path,
        witness_digest=witness_digest,
        root=root,
        paths_requested=paths_requested,
        projection_rows=projection_rows,
        phase_checkpoint_state=phase_checkpoint_state,
    )


def _render_incremental_report(
    *,
    analysis_state: str,
    progress_payload: Mapping[str, JSONValue] | None,
    projection_rows: Sequence[Mapping[str, JSONValue]],
    sections: Mapping[str, list[str]],
) -> tuple[str, dict[str, str]]:
    return orchestrator_primitives._render_incremental_report(
        analysis_state=analysis_state,
        progress_payload=progress_payload,
        projection_rows=projection_rows,
        sections=sections,
    )


def _collection_progress_intro_lines(
    *,
    collection_resume: Mapping[str, JSONValue],
    total_files: int,
    resume_state_intro: Mapping[str, JSONValue] | None = None,
) -> list[str]:
    return orchestrator_primitives._collection_progress_intro_lines(
        collection_resume=collection_resume,
        total_files=total_files,
        resume_state_intro=resume_state_intro,
    )


def _collection_components_preview_lines(
    *,
    collection_resume: Mapping[str, JSONValue],
) -> list[str]:
    return orchestrator_primitives._collection_components_preview_lines(
        collection_resume=collection_resume,
    )


def _groups_by_path_from_collection_resume(
    collection_resume: Mapping[str, JSONValue],
) -> dict[Path, dict[str, list[set[str]]]]:
    return orchestrator_primitives._groups_by_path_from_collection_resume(collection_resume)


def _split_incremental_obligations(
    obligations: Sequence[Mapping[str, JSONValue]],
) -> tuple[list[JSONObject], list[JSONObject]]:
    return orchestrator_primitives._split_incremental_obligations(obligations)


def _latest_report_phase(phases: Mapping[str, JSONValue] | None) -> str | None:
    return orchestrator_primitives._latest_report_phase(phases)


def _phase_progress_dimensions_summary(
    phase_progress_v2: Mapping[str, JSONValue] | None,
) -> str:
    return orchestrator_primitives._phase_progress_dimensions_summary(phase_progress_v2)
