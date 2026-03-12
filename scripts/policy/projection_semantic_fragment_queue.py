#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping

from gabion.order_contract import ordered_or_sorted
from gabion.tooling.runtime.projection_fiber_semantics_summary import (
    projection_fiber_semantics_summary_from_payload,
)

_FORMAT_VERSION = 1
_DEFAULT_SOURCE_ARTIFACT = "artifacts/out/policy_check_result.json"
_MAX_SEMANTIC_PREVIEW_SAMPLES = 20


def _sorted[T](values: list[T], *, key=None) -> list[T]:
    return ordered_or_sorted(
        values,
        source="scripts.policy.projection_semantic_fragment_queue",
        key=key,
    )


@dataclass(frozen=True)
class ProjectionSemanticFragmentCurrentState:
    decision: dict[str, Any]
    semantic_row_count: int
    compiled_projection_semantic_bundle_count: int
    compiled_projection_semantic_spec_names: tuple[str, ...]
    semantic_preview_count: int
    semantic_previews: tuple[dict[str, Any], ...]

    def as_payload(self) -> dict[str, object]:
        return {
            "decision": self.decision,
            "semantic_row_count": self.semantic_row_count,
            "compiled_projection_semantic_bundle_count": (
                self.compiled_projection_semantic_bundle_count
            ),
            "compiled_projection_semantic_spec_names": list(
                self.compiled_projection_semantic_spec_names
            ),
            "semantic_preview_count": self.semantic_preview_count,
            "semantic_previews": [item for item in self.semantic_previews],
        }


@dataclass(frozen=True)
class ProjectionSemanticFragmentQueueItem:
    queue_id: str
    phase: str
    status: str
    title: str
    summary: str
    next_action: str
    evidence_links: tuple[str, ...]

    def as_payload(self) -> dict[str, object]:
        return {
            "queue_id": self.queue_id,
            "phase": self.phase,
            "status": self.status,
            "title": self.title,
            "summary": self.summary,
            "next_action": self.next_action,
            "evidence_links": [item for item in self.evidence_links],
        }


@dataclass(frozen=True)
class ProjectionSemanticFragmentQueue:
    source_artifact: str
    current_state: ProjectionSemanticFragmentCurrentState
    next_queue_ids: tuple[str, ...]
    items: tuple[ProjectionSemanticFragmentQueueItem, ...]

    def as_payload(self) -> dict[str, object]:
        return {
            "format_version": _FORMAT_VERSION,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_artifact": self.source_artifact,
            "current_state": self.current_state.as_payload(),
            "next_queue_ids": [item for item in self.next_queue_ids],
            "items": [item.as_payload() for item in self.items],
        }


def analyze(
    *,
    payload: Mapping[str, object],
    source_artifact: str = _DEFAULT_SOURCE_ARTIFACT,
) -> ProjectionSemanticFragmentQueue:
    current_state = _current_state(payload)
    items = _queue_items(current_state)
    next_queue_ids = tuple(
        item.queue_id for item in items if item.status != "landed"
    )
    return ProjectionSemanticFragmentQueue(
        source_artifact=source_artifact,
        current_state=current_state,
        next_queue_ids=next_queue_ids,
        items=items,
    )


def _current_state(
    payload: Mapping[str, object],
) -> ProjectionSemanticFragmentCurrentState:
    summary = projection_fiber_semantics_summary_from_payload(payload)
    if summary is None:
        return ProjectionSemanticFragmentCurrentState(
            decision={},
            semantic_row_count=0,
            compiled_projection_semantic_bundle_count=0,
            compiled_projection_semantic_spec_names=(),
            semantic_preview_count=0,
            semantic_previews=(),
        )
    semantic_previews = tuple(
        _preview_payload(item.as_payload())
        for item in summary.semantic_previews
    )
    return ProjectionSemanticFragmentCurrentState(
        decision=dict(summary.decision.items()),
        semantic_row_count=summary.semantic_row_count,
        compiled_projection_semantic_bundle_count=(
            summary.compiled_projection_semantic_bundle_count
        ),
        compiled_projection_semantic_spec_names=(
            summary.compiled_projection_semantic_spec_names
        ),
        semantic_preview_count=len(semantic_previews),
        semantic_previews=semantic_previews[:_MAX_SEMANTIC_PREVIEW_SAMPLES],
    )


def _queue_items(
    current_state: ProjectionSemanticFragmentCurrentState,
) -> tuple[ProjectionSemanticFragmentQueueItem, ...]:
    spec_names = current_state.compiled_projection_semantic_spec_names
    spec_names_summary = ", ".join(spec_names) if spec_names else "<none>"
    row_count = current_state.semantic_row_count
    bundle_count = current_state.compiled_projection_semantic_bundle_count
    preview_count = current_state.semantic_preview_count
    semantic_lowering_landed = bundle_count > 0 and bool(spec_names)
    items = (
        ProjectionSemanticFragmentQueueItem(
            queue_id="PSF-001",
            phase="Phase 2",
            status="landed" if row_count > 0 else "queued",
            title="Carrier-first reflection over ASPF/fibration witnesses",
            summary=(
                f"{row_count} canonical semantic row(s) emitted into the policy surface."
                if row_count > 0
                else "Canonical semantic rows are not yet present in the source artifact payload."
            ),
            next_action=(
                "Keep structural identity / site identity continuity stable as new semantic surfaces land."
                if row_count > 0
                else "Land the canonical witnessed carrier on a real semantic path before expanding authoring surfaces."
            ),
            evidence_links=(
                "src/gabion/analysis/projection/semantic_fragment.py",
                "src/gabion/tooling/policy_substrate/lattice_convergence_semantic.py",
                "tests/gabion/tooling/runtime_policy/test_lattice_convergence_semantic.py",
            ),
        ),
        ProjectionSemanticFragmentQueueItem(
            queue_id="PSF-002",
            phase="Phase 3",
            status="landed" if semantic_lowering_landed else "queued",
            title="Deterministic SHACL/SPARQL lowering for declared quotient faces",
            summary=(
                f"{bundle_count} compiled semantic bundle(s) emitted for {spec_names_summary}."
                if semantic_lowering_landed
                else "Compiled semantic lowering is not yet present for declared quotient-face specs."
            ),
            next_action=(
                "Preserve lowering determinism and identity/witness trace continuity as additional faces are promoted."
                if semantic_lowering_landed
                else "Compile the first declared quotient-face authoring surface into SHACL/SPARQL plans."
            ),
            evidence_links=(
                "src/gabion/analysis/projection/projection_semantic_lowering.py",
                "src/gabion/analysis/projection/projection_semantic_lowering_compile.py",
                "src/gabion/analysis/projection/semantic_fragment_compile.py",
            ),
        ),
        ProjectionSemanticFragmentQueueItem(
            queue_id="PSF-003",
            phase="Phase 4",
            status="landed" if preview_count > 0 else "queued",
            title="Reporting-layer propagation of canonical semantic previews",
            summary=(
                f"{preview_count} semantic preview row(s) are now carried into queue/report artifacts."
                if preview_count > 0
                else "Reporting artifacts are not yet carrying canonical semantic previews."
            ),
            next_action=(
                "Use preview propagation as the continuity surface while broader carrier consumers are added."
                if preview_count > 0
                else "Thread canonical semantic previews into reporting surfaces that currently only see aggregate counts."
            ),
            evidence_links=(
                "src/gabion/tooling/runtime/policy_scanner_suite.py",
                "scripts/policy/hotspot_neighborhood_queue.py",
                "tests/gabion/tooling/policy/test_hotspot_neighborhood_queue.py",
            ),
        ),
        ProjectionSemanticFragmentQueueItem(
            queue_id="PSF-004",
            phase="Phase 4",
            status="in_progress" if semantic_lowering_landed else "queued",
            title="Friendly-surface convergence via typed ProjectionSpec lowering",
            summary=(
                f"Typed lowering exists for {spec_names_summary}, projection_exec remains the compatibility runtime, and the projection history ledger now records per-spec lowering status."
                if semantic_lowering_landed
                else "Friendly-surface lowering has not yet been anchored to a canonical semantic path."
            ),
            next_action="Promote additional declared semantic ops through lowering without adding new semantic behavior directly to projection_exec.",
            evidence_links=(
                "src/gabion/analysis/projection/projection_registry.py",
                "scripts/policy/build_projection_spec_history.py",
                "artifacts/out/projection_spec_history_ledger.json",
                "src/gabion/analysis/projection/projection_exec.py",
                "docs/projection_semantic_fragment_rfc.md",
            ),
        ),
        ProjectionSemanticFragmentQueueItem(
            queue_id="PSF-005",
            phase="Phase 4",
            status="queued",
            title="Expand the semantic op set beyond declared quotient-face slices",
            summary="The reflect + declared quotient_face projection_fiber slices are executable today; the remaining RFC ops are still design-only.",
            next_action="Add the next smallest lawful semantic op on top of the same carrier instead of widening generic row-shaping operators.",
            evidence_links=(
                "src/gabion/analysis/projection/semantic_fragment.py",
                "docs/projection_semantic_fragment_rfc.md",
                "docs/ttl_kernel_semantics.md",
            ),
        ),
        ProjectionSemanticFragmentQueueItem(
            queue_id="PSF-006",
            phase="Phase 4",
            status="queued",
            title="Move policy and authoring consumers toward direct canonical-carrier judgment",
            summary="The projection-fiber policy path is carrier-backed, but broader policy/authoring surfaces still depend on compatibility and summary bridges.",
            next_action="Shift the next consumer from row-shape inference to direct canonical-carrier reads, then preserve that path with policy tests.",
            evidence_links=(
                "scripts/policy/policy_check.py",
                "src/gabion/tooling/runtime/policy_scanner_suite.py",
                "docs/projection_semantic_fragment_rfc.md",
            ),
        ),
        ProjectionSemanticFragmentQueueItem(
            queue_id="PSF-007",
            phase="Phase 5",
            status="queued",
            title="Cut over legacy adapters and retire semantic_carrier_adapter boundaries",
            summary="The semantic fragment is still intentionally adapter-scoped while the canonical path converges.",
            next_action="Use the RFC cutover criteria and ratchet rules to remove temporary adapter status only after end-to-end semantic paths are stable.",
            evidence_links=(
                "src/gabion/analysis/projection/projection_exec.py",
                "src/gabion/analysis/projection/semantic_fragment.py",
                "docs/projection_semantic_fragment_rfc.md",
            ),
        ),
    )
    return items


def _preview_payload(value: Mapping[str, object]) -> dict[str, Any]:
    return {
        "spec_name": _string_value(value.get("spec_name")),
        "quotient_face": _string_value(value.get("quotient_face")),
        "source_structural_identity": _string_value(
            value.get("source_structural_identity")
        ),
        "path": _string_value(value.get("path")),
        "qualname": _string_value(value.get("qualname")),
        "structural_path": _string_value(value.get("structural_path")),
        "obligation_state": _string_value(value.get("obligation_state")),
        "complete": bool(value.get("complete")) if isinstance(value.get("complete"), bool) else False,
    }

def _string_value(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()


def _markdown_summary(queue: ProjectionSemanticFragmentQueue) -> str:
    current_state = queue.current_state
    spec_names = ", ".join(current_state.compiled_projection_semantic_spec_names) or "<none>"
    lines = [
        "# Projection Semantic Fragment Queue",
        "",
        f"- source_artifact: `{queue.source_artifact}`",
        f"- decision_rule: `{_string_value(current_state.decision.get('rule_id')) or '<none>'}`",
        f"- semantic_rows: `{current_state.semantic_row_count}`",
        (
            "- compiled_projection_semantic_bundles: "
            f"`{current_state.compiled_projection_semantic_bundle_count}`"
        ),
        f"- compiled_specs: `{spec_names}`",
        f"- semantic_preview_count: `{current_state.semantic_preview_count}`",
        (
            "- semantic_preview_samples: "
            f"`{len(current_state.semantic_previews)}`"
        ),
        "",
        "## Next Queue",
    ]
    if queue.next_queue_ids:
        lines.extend(f"- `{item}`" for item in queue.next_queue_ids)
    else:
        lines.append("- `<none>`")
    lines.extend(
        [
            "",
            "## Queue",
            "",
            "| id | phase | status | title |",
            "| --- | --- | --- | --- |",
        ]
    )
    for item in queue.items:
        lines.append(
            f"| {item.queue_id} | {item.phase} | {item.status} | {item.title} |"
        )
    if current_state.semantic_previews:
        lines.extend(
            [
                "",
                "## Semantic Previews",
                "",
                "| spec | quotient_face | path | qualname | structural_path |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for preview in current_state.semantic_previews:
            lines.append(
                "| {spec} | {face} | {path} | {qualname} | {structural_path} |".format(
                    spec=_string_value(preview.get("spec_name")),
                    face=_string_value(preview.get("quotient_face")),
                    path=_string_value(preview.get("path")),
                    qualname=_string_value(preview.get("qualname")),
                    structural_path=_string_value(preview.get("structural_path")),
                )
            )
    return "\n".join(lines) + "\n"


def run(
    *,
    source_artifact_path: Path,
    out_path: Path,
    markdown_out: Path | None = None,
) -> int:
    payload = json.loads(source_artifact_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("projection semantic fragment source payload must be a mapping")
    queue = analyze(
        payload=payload,
        source_artifact=str(source_artifact_path),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(queue.as_payload(), indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    if markdown_out is not None:
        markdown_out.parent.mkdir(parents=True, exist_ok=True)
        markdown_out.write_text(
            _markdown_summary(queue),
            encoding="utf-8",
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-artifact",
        default=_DEFAULT_SOURCE_ARTIFACT,
    )
    parser.add_argument(
        "--out",
        default="artifacts/out/projection_semantic_fragment_queue.json",
    )
    parser.add_argument(
        "--markdown-out",
        default="artifacts/out/projection_semantic_fragment_queue.md",
    )
    args = parser.parse_args(argv)
    return run(
        source_artifact_path=Path(args.source_artifact).resolve(),
        out_path=Path(args.out).resolve(),
        markdown_out=Path(args.markdown_out).resolve(),
    )


if __name__ == "__main__":
    raise SystemExit(main())
