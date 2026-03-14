from __future__ import annotations

from datetime import datetime, timezone
from contextlib import ExitStack
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TypedDict

from gabion.analysis.foundation.timeout_context import (
    Deadline,
    deadline_clock_scope,
    deadline_scope,
)
from gabion.deadline_clock import MonotonicClock
from gabion.order_contract import ordered_or_sorted
from gabion.tooling.policy_substrate.aspf_union_view import build_aspf_union_view
from gabion.tooling.policy_substrate.overlap_eval import evaluate_condition_overlaps
from gabion.tooling.policy_substrate.projection_lens import LensEvent
from gabion.tooling.policy_substrate.taint_intervals import build_taint_intervals
from gabion.tooling.runtime.policy_scan_batch import build_policy_scan_batch


class CrossOriginWitnessFieldCheck(TypedDict):
    field_name: str
    matches: bool
    left_value: str
    right_value: str


class CrossOriginWitnessRow(TypedDict):
    row_key: str
    row_kind: str
    left_origin_kind: str
    left_origin_key: str
    right_origin_kind: str
    right_origin_key: str
    remap_key: str
    summary: str


class CrossOriginWitnessCase(TypedDict):
    case_key: str
    case_kind: str
    title: str
    status: str
    summary: str
    left_label: str
    right_label: str
    evidence_paths: list[str]
    row_keys: list[str]
    field_checks: list[CrossOriginWitnessFieldCheck]


class CrossOriginWitnessSummary(TypedDict):
    case_count: int
    passing_case_count: int
    failing_case_count: int
    witness_row_count: int


class CrossOriginWitnessContractPayload(TypedDict):
    format_version: int
    schema_version: int
    artifact_kind: str
    producer: str
    generated_at_utc: str
    root: str
    summary: CrossOriginWitnessSummary
    cases: list[CrossOriginWitnessCase]
    witness_rows: list[CrossOriginWitnessRow]


_PRODUCER = (
    "gabion.tooling.runtime.cross_origin_witness_artifact."
    "build_cross_origin_witness_contract_artifact_payload"
)


def _sorted[T](values: list[T], *, key=None) -> list[T]:
    return ordered_or_sorted(
        values,
        source="tooling.runtime.cross_origin_witness_artifact",
        key=key,
    )


def _artifact_rel_path(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _render_boundary_value(value: object) -> str:
    try:
        rendered = json.dumps(
            value,
            sort_keys=False,
            separators=(",", ":"),
            ensure_ascii=True,
        )
    except TypeError:
        rendered = str(value)
    return rendered if len(rendered) <= 240 else rendered[:237] + "..."


def _field_check(
    *,
    field_name: str,
    left_value: object,
    right_value: object,
) -> CrossOriginWitnessFieldCheck:
    return CrossOriginWitnessFieldCheck(
        field_name=field_name,
        matches=left_value == right_value,
        left_value=_render_boundary_value(left_value),
        right_value=_render_boundary_value(right_value),
    )


def _sample_sources() -> dict[str, str]:
    return {
        "src/gabion/sample_alpha.py": (
            "def alpha(flag: bool) -> int:\n"
            "    if flag:\n"
            "        return 1\n"
            "    return 0\n"
        ),
        "src/gabion/sample_beta.py": (
            "def beta(items: list[int]) -> int:\n"
            "    total = 0\n"
            "    for item in items:\n"
            "        total += item\n"
            "    return total\n"
        ),
    }


def _write_sample_sources(root: Path) -> list[Path]:
    paths: list[Path] = []
    for rel_path, source in _sample_sources().items():
        path = root / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
        paths.append(path)
    return _sorted(paths, key=lambda item: item.as_posix())


def _analysis_union_path_remap_case(*, root: Path) -> tuple[CrossOriginWitnessCase, list[CrossOriginWitnessRow]]:
    import gabion.server as server

    with TemporaryDirectory(prefix="gabion-cross-origin-") as temp_dir:
        sample_root = Path(temp_dir)
        file_paths = _write_sample_sources(sample_root)
        reversed_paths = list(reversed(file_paths))
        with ExitStack() as scope:
            scope.enter_context(deadline_clock_scope(MonotonicClock()))
            scope.enter_context(deadline_scope(Deadline.from_timeout_ms(60_000)))
            witness = server._analysis_input_witness(
                root=sample_root,
                file_paths=reversed_paths,
                recursive=False,
                include_invariant_propositions=False,
                include_wl_refinement=False,
                config=server.AuditConfig(
                    project_root=sample_root,
                    exclude_dirs=set(),
                    ignore_params=set(),
                    external_filter=False,
                    strictness="high",
                ),
            )
            second_witness = server._analysis_input_witness(
                root=sample_root,
                file_paths=reversed_paths,
                recursive=False,
                include_invariant_propositions=False,
                include_wl_refinement=False,
                config=server.AuditConfig(
                    project_root=sample_root,
                    exclude_dirs=set(),
                    ignore_params=set(),
                    external_filter=False,
                    strictness="high",
                ),
            )
            first_digest = server._analysis_manifest_digest_from_witness(witness) or ""
            second_digest = (
                server._analysis_manifest_digest_from_witness(second_witness) or ""
            )
        batch = build_policy_scan_batch(
            root=sample_root,
            target_globs=(),
            files=file_paths,
        )
        union_view = build_aspf_union_view(batch=batch)
        raw_witness_files = witness.get("files", [])
        witness_paths = tuple(
            _artifact_rel_path(sample_root, Path(str(item.get("path", "")).strip()))
            for item in raw_witness_files
            if isinstance(item, dict) and str(item.get("path", "")).strip()
        )
        module_paths = tuple(module.rel_path for module in union_view.modules)
        overlap_paths = tuple(path for path in module_paths if path in set(witness_paths))
        missing_from_union = tuple(path for path in witness_paths if path not in set(module_paths))
        missing_from_witness = tuple(path for path in module_paths if path not in set(witness_paths))
        rows = _sorted(
            [
                CrossOriginWitnessRow(
                    row_key=f"path_remap:{path}",
                    row_kind="path_remap",
                    left_origin_kind="analysis_input_witness.file",
                    left_origin_key=path,
                    right_origin_kind="aspf_union_view.module",
                    right_origin_key=path,
                    remap_key=path,
                    summary=f"analysis witness file remapped to union module by rel_path={path}",
                )
                for path in overlap_paths
            ],
            key=lambda item: item["row_key"],
        )
        field_checks = [
            _field_check(
                field_name="manifest_digest_present",
                left_value=bool(first_digest),
                right_value=True,
            ),
            _field_check(
                field_name="manifest_digest_stable",
                left_value=first_digest,
                right_value=second_digest,
            ),
            _field_check(
                field_name="witness_order_differs_from_union_order",
                left_value=witness_paths != module_paths,
                right_value=True,
            ),
            _field_check(
                field_name="remap_row_count",
                left_value=len(rows),
                right_value=len(module_paths),
            ),
            _field_check(
                field_name="missing_from_union",
                left_value=missing_from_union,
                right_value=(),
            ),
            _field_check(
                field_name="missing_from_witness",
                left_value=missing_from_witness,
                right_value=(),
            ),
        ]
    mismatch_count = sum(1 for item in field_checks if not item["matches"])
    return (
        CrossOriginWitnessCase(
            case_key="analysis_union_path_remap",
            case_kind="cross_origin_path_remap",
            title="analysis witness to union-view path remap",
            status="pass" if mismatch_count == 0 else "fail",
            summary=(
                "analysis witness files remapped onto union-view modules with rows={rows} "
                "mismatches={mismatches}".format(
                    rows=len(rows),
                    mismatches=mismatch_count,
                )
            ),
            left_label="analysis_input_witness",
            right_label="aspf_union_view",
            evidence_paths=[
                _artifact_rel_path(root, root / "src/gabion/server.py"),
                _artifact_rel_path(root, root / "src/gabion/server_core/server_payload_dispatch.py"),
                _artifact_rel_path(root, root / "src/gabion/tooling/runtime/policy_scan_batch.py"),
                _artifact_rel_path(root, root / "src/gabion/tooling/policy_substrate/aspf_union_view.py"),
            ],
            row_keys=[item["row_key"] for item in rows],
            field_checks=field_checks,
        ),
        rows,
    )


def _condition_overlap_ledger_case(*, root: Path) -> tuple[CrossOriginWitnessCase, list[CrossOriginWitnessRow]]:
    events = (
        LensEvent(
            ordinal=1,
            site_id="site:intro",
            path="src/gabion/sample_alpha.py",
            qualname="sample.alpha",
            line=1,
            column=1,
            node_kind="name",
            surface="pyast",
            fiber_id="fiber:sample.alpha",
            event_kind="taint_intro",
            event_phase="taint_intro",
            input_slot="slot",
            taint_class="control",
            action="taint_intro",
        ),
        LensEvent(
            ordinal=2,
            site_id="site:condition:one",
            path="src/gabion/sample_alpha.py",
            qualname="sample.alpha",
            line=2,
            column=4,
            node_kind="if",
            surface="pyast",
            fiber_id="fiber:sample.alpha",
            event_kind="condition",
            event_phase="condition",
            input_slot="slot",
            taint_class="control",
            action="condition",
        ),
        LensEvent(
            ordinal=3,
            site_id="site:condition:two",
            path="src/gabion/sample_alpha.py",
            qualname="sample.alpha",
            line=3,
            column=4,
            node_kind="if",
            surface="pyast",
            fiber_id="fiber:sample.alpha",
            event_kind="condition",
            event_phase="condition",
            input_slot="slot",
            taint_class="control",
            action="condition",
        ),
        LensEvent(
            ordinal=4,
            site_id="site:erase",
            path="src/gabion/sample_alpha.py",
            qualname="sample.alpha",
            line=4,
            column=1,
            node_kind="return",
            surface="pyast",
            fiber_id="fiber:sample.alpha",
            event_kind="taint_erase",
            event_phase="taint_erase",
            input_slot="slot",
            taint_class="control",
            action="taint_erase",
        ),
    )
    intervals = tuple(build_taint_intervals(events=events))
    overlaps = tuple(
        evaluate_condition_overlaps(
            intervals=intervals,
            condition_events=(events[1], events[2]),
        )
    )
    permuted_overlaps = tuple(
        evaluate_condition_overlaps(
            intervals=intervals,
            condition_events=(events[2], events[1]),
        )
    )
    rows = _sorted(
        [
            CrossOriginWitnessRow(
                row_key=f"condition_overlap:{overlap.condition_overlap_id}",
                row_kind="condition_overlap",
                left_origin_kind="taint_interval",
                left_origin_key=overlap.taint_interval_id,
                right_origin_kind="condition_event",
                right_origin_key=f"{overlap.condition_event.fiber_id}:{overlap.condition_event.ordinal}",
                remap_key=overlap.condition_overlap_id,
                summary=(
                    "taint interval remapped to condition event overlap fiber={fiber} "
                    "ordinal={ordinal}".format(
                        fiber=overlap.fiber_id,
                        ordinal=overlap.condition_event.ordinal,
                    )
                ),
            )
            for overlap in overlaps
        ],
        key=lambda item: item["row_key"],
    )
    field_checks = [
        _field_check(
            field_name="interval_count",
            left_value=len(intervals),
            right_value=1,
        ),
        _field_check(
            field_name="overlap_count_stable",
            left_value=len(overlaps),
            right_value=len(permuted_overlaps),
        ),
        _field_check(
            field_name="overlap_ids_stable",
            left_value=tuple(item.condition_overlap_id for item in overlaps),
            right_value=tuple(item.condition_overlap_id for item in permuted_overlaps),
        ),
        _field_check(
            field_name="row_keys_cover_overlaps",
            left_value=tuple(sorted(item["row_key"] for item in rows)),
            right_value=tuple(
                sorted(f"condition_overlap:{item.condition_overlap_id}" for item in overlaps)
            ),
        ),
    ]
    mismatch_count = sum(1 for item in field_checks if not item["matches"])
    return (
        CrossOriginWitnessCase(
            case_key="condition_overlap_ledger",
            case_kind="condition_overlap_ledger",
            title="condition overlap ledger determinism",
            status="pass" if mismatch_count == 0 else "fail",
            summary=(
                "condition overlap rows={rows} mismatches={mismatches}".format(
                    rows=len(rows),
                    mismatches=mismatch_count,
                )
            ),
            left_label="intervals",
            right_label="condition_events",
            evidence_paths=[
                _artifact_rel_path(root, root / "src/gabion/tooling/policy_substrate/overlap_eval.py"),
            ],
            row_keys=[item["row_key"] for item in rows],
            field_checks=field_checks,
        ),
        rows,
    )


def build_cross_origin_witness_contract_artifact_payload(
    *,
    root: Path,
) -> CrossOriginWitnessContractPayload:
    root = root.resolve()
    remap_case, remap_rows = _analysis_union_path_remap_case(root=root)
    overlap_case, overlap_rows = _condition_overlap_ledger_case(root=root)
    cases = [remap_case, overlap_case]
    witness_rows = _sorted(
        [*remap_rows, *overlap_rows],
        key=lambda item: item["row_key"],
    )
    return CrossOriginWitnessContractPayload(
        format_version=1,
        schema_version=1,
        artifact_kind="cross_origin_witness_contract",
        producer=_PRODUCER,
        generated_at_utc=datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        root=str(root),
        summary=CrossOriginWitnessSummary(
            case_count=len(cases),
            passing_case_count=sum(1 for item in cases if item["status"] == "pass"),
            failing_case_count=sum(1 for item in cases if item["status"] != "pass"),
            witness_row_count=len(witness_rows),
        ),
        cases=cases,
        witness_rows=witness_rows,
    )


def write_cross_origin_witness_contract_artifact(*, path: Path, root: Path) -> Path:
    payload = build_cross_origin_witness_contract_artifact_payload(root=root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    return path
