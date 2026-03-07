# gabion:decision_protocol_module
from __future__ import annotations

from typing import Mapping

from gabion.analysis.foundation.timeout_context import deadline_loop_iter


def render_timeout_progress_markdown(
    *,
    analysis_state: str | None,
    progress: Mapping[str, object],
    deadline_profile: Mapping[str, object] | None = None,
) -> str:
    lines = ["# Timeout Progress", ""]
    if analysis_state:
        lines.append(f"- `analysis_state`: `{analysis_state}`")
    classification = progress.get("classification")
    if isinstance(classification, str):
        lines.append(f"- `classification`: `{classification}`")
    retry_recommended = progress.get("retry_recommended")
    if isinstance(retry_recommended, bool):
        lines.append(f"- `retry_recommended`: `{retry_recommended}`")
    resume_supported = progress.get("resume_supported")
    if isinstance(resume_supported, bool):
        lines.append(f"- `resume_supported`: `{resume_supported}`")
    ticks_consumed = progress.get("ticks_consumed")
    if isinstance(ticks_consumed, int):
        lines.append(f"- `ticks_consumed`: `{ticks_consumed}`")
    tick_limit = progress.get("tick_limit")
    if isinstance(tick_limit, int):
        lines.append(f"- `tick_limit`: `{tick_limit}`")
    ticks_remaining = progress.get("ticks_remaining")
    if isinstance(ticks_remaining, int):
        lines.append(f"- `ticks_remaining`: `{ticks_remaining}`")
    progress_ticks_per_ns = progress.get("ticks_per_ns")
    resolved_ticks_per_ns = (
        progress_ticks_per_ns
        if isinstance(progress_ticks_per_ns, (int, float))
        else (
            deadline_profile.get("ticks_per_ns")
            if isinstance(deadline_profile, Mapping)
            else None
        )
    )
    if isinstance(resolved_ticks_per_ns, (int, float)):
        lines.append(f"- `ticks_per_ns`: `{float(resolved_ticks_per_ns):.9f}`")
    resume = progress.get("resume")
    if isinstance(resume, Mapping):
        token = resume.get("resume_token")
        if isinstance(token, Mapping):
            lines.append("")
            lines.append("## Resume Token")
            lines.append("")
            for key in deadline_loop_iter(
                (
                    "phase",
                    "checkpoint_path",
                    "completed_files",
                    "remaining_files",
                    "total_files",
                    "witness_digest",
                )
            ):
                value = token.get(key)
                if value is None:
                    continue
                lines.append(f"- `{key}`: `{value}`")
    obligations = progress.get("incremental_obligations")
    if isinstance(obligations, list) and obligations:
        lines.append("")
        lines.append("## Incremental Obligations")
        lines.append("")
        for entry in deadline_loop_iter(obligations):
            if not isinstance(entry, Mapping):
                continue
            status = str(entry.get("status", "UNKNOWN") or "UNKNOWN")
            contract = str(entry.get("contract", "") or "")
            kind = str(entry.get("kind", "") or "")
            detail = str(entry.get("detail", "") or "")
            section_id = str(entry.get("section_id", "") or "")
            section_suffix = f" section={section_id}" if section_id else ""
            lines.append(
                f"- `{status}` `{contract}` `{kind}`{section_suffix}: {detail}"
            )
    return "\n".join(lines)
