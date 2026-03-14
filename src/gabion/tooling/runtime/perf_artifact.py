from __future__ import annotations

import ast
import cProfile
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from functools import lru_cache
from pathlib import Path
import pstats
from typing import TypedDict

from gabion.order_contract import ordered_or_sorted
from gabion.tooling.policy_substrate.invariant_graph import load_invariant_graph
from gabion.tooling.policy_substrate.policy_queue_identity import (
    ArtifactNodeId,
    PolicyQueueIdentitySpace,
)

class PerfArtifactNode(TypedDict):
    wire: str
    site_identity: str
    structural_identity: str
    rel_path: str
    qualname: str
    line: int
    column: int


class PerfArtifactSample(TypedDict, total=False):
    artifact_node: PerfArtifactNode
    rel_path: str
    qualname: str
    line: int
    inclusive_value: float


class PerfArtifactProfile(TypedDict):
    profiler: str
    metric_kind: str
    unit: str
    samples: list[PerfArtifactSample]


class PerfArtifactPayload(TypedDict):
    format_version: int
    generated_at_utc: str
    root: str
    command: list[str]
    requested_checks: list[str]
    returncode: int
    observation_count: int
    profiles: list[PerfArtifactProfile]


@dataclass(frozen=True)
class _FunctionLineRange:
    start_line: int
    end_line: int
    qualname: str


def _sorted[T](values: list[T], *, key=None) -> list[T]:
    return ordered_or_sorted(
        values,
        source="tooling.runtime.perf_artifact",
        key=key,
    )


def _normalized_rel_path(*, root: Path, path: str) -> str:
    try:
        resolved = Path(path).resolve()
    except (OSError, RuntimeError):
        return ""
    try:
        return str(resolved.relative_to(root)).replace("\\", "/")
    except ValueError:
        return ""


def _parse_function_line_ranges(path: Path) -> tuple[_FunctionLineRange, ...]:
    try:
        source = path.read_text(encoding="utf-8")
    except OSError:
        return ()
    try:
        module = ast.parse(source, filename=str(path))
    except SyntaxError:
        return ()

    class _Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.scope: list[str] = []
            self.entries: list[_FunctionLineRange] = []

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self.scope.append(node.name)
            self.generic_visit(node)
            self.scope.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self._visit_function(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self._visit_function(node)

        def _visit_function(
            self,
            node: ast.FunctionDef | ast.AsyncFunctionDef,
        ) -> None:
            self.scope.append(node.name)
            self.entries.append(
                _FunctionLineRange(
                    start_line=int(node.lineno),
                    end_line=int(getattr(node, "end_lineno", node.lineno) or node.lineno),
                    qualname=".".join(self.scope),
                )
            )
            self.generic_visit(node)
            self.scope.pop()

    visitor = _Visitor()
    visitor.visit(module)
    return tuple(visitor.entries)


@lru_cache(maxsize=None)
def _function_line_ranges(path: str) -> tuple[_FunctionLineRange, ...]:
    return _parse_function_line_ranges(Path(path))


def _qualname_for_line(*, path: Path, line: int, fallback_name: str) -> str:
    matching = [
        item
        for item in _function_line_ranges(str(path))
        if item.start_line <= line <= item.end_line
    ]
    if matching:
        return min(
            matching,
            key=lambda item: (item.end_line - item.start_line, item.start_line),
        ).qualname
    if fallback_name and not fallback_name.startswith("<"):
        return fallback_name
    return ""


def _node_identity_index(
    graph_artifact: Path | None,
) -> tuple[
    dict[tuple[str, int, str], ArtifactNodeId],
    dict[tuple[str, str], ArtifactNodeId],
]:
    if graph_artifact is None or not graph_artifact.exists():
        return ({}, {})
    graph = load_invariant_graph(graph_artifact)
    identity_space = PolicyQueueIdentitySpace()
    exact_index = {
        (node.rel_path, node.line, node.qualname): identity_space.artifact_node_id(
            site_identity=node.site_identity,
            structural_identity=node.structural_identity,
            rel_path=node.rel_path,
            qualname=node.qualname,
            line=node.line,
            column=node.column,
        )
        for node in graph.nodes
        if (
            node.rel_path
            and node.line > 0
            and node.qualname
            and node.site_identity
            and node.structural_identity
        )
    }
    qualname_candidates: dict[tuple[str, str], set[ArtifactNodeId]] = {}
    for node in graph.nodes:
        if not (
            node.rel_path
            and node.qualname
            and node.site_identity
            and node.structural_identity
        ):
            continue
        qualname_candidates.setdefault((node.rel_path, node.qualname), set()).add(
            identity_space.artifact_node_id(
                site_identity=node.site_identity,
                structural_identity=node.structural_identity,
                rel_path=node.rel_path,
                qualname=node.qualname,
                line=node.line,
                column=node.column,
            )
        )
    unique_qualname_index = {
        key: next(iter(values))
        for key, values in qualname_candidates.items()
        if len(values) == 1
    }
    return (exact_index, unique_qualname_index)


def _sample_sort_key(item: PerfArtifactSample) -> tuple[float, str, int, str]:
    artifact_node = item.get("artifact_node")
    if isinstance(artifact_node, dict):
        rel_path = str(artifact_node.get("rel_path", ""))
        line = int(artifact_node.get("line", 0) or 0)
        qualname = str(artifact_node.get("qualname", ""))
    else:
        rel_path = str(item.get("rel_path", ""))
        line = int(item.get("line", 0) or 0)
        qualname = str(item.get("qualname", ""))
    return (-float(item["inclusive_value"]), rel_path, line, qualname)


def build_cprofile_perf_artifact_payload(
    *,
    profile: cProfile.Profile,
    root: Path,
    command: tuple[str, ...],
    requested_checks: tuple[str, ...],
    returncode: int,
    graph_artifact: Path | None = None,
) -> PerfArtifactPayload:
    root = root.resolve()
    stats = pstats.Stats(profile)
    (
        node_identity_by_site,
        node_identity_by_qualname,
    ) = _node_identity_index(graph_artifact)
    samples: list[PerfArtifactSample] = []
    for raw_key, raw_stats in stats.stats.items():
        filename, line, funcname = raw_key
        if not isinstance(filename, str) or not isinstance(line, int):
            continue
        if line <= 0:
            continue
        rel_path = _normalized_rel_path(root=root, path=filename)
        if not rel_path:
            continue
        inclusive_value = float(raw_stats[3])
        if inclusive_value <= 0:
            continue
        source_path = root / rel_path
        qualname = _qualname_for_line(path=source_path, line=line, fallback_name=str(funcname))
        if not qualname:
            continue
        artifact_node = node_identity_by_site.get(
            (rel_path, line, qualname),
            node_identity_by_qualname.get(
                (rel_path, qualname),
                None,
            ),
        )
        sample: PerfArtifactSample = {
            "inclusive_value": inclusive_value,
        }
        if artifact_node is None:
            sample.update(
                {
                    "rel_path": rel_path,
                    "qualname": qualname,
                    "line": line,
                }
            )
        else:
            sample["artifact_node"] = artifact_node.as_payload()
        samples.append(sample)
    sorted_samples = _sorted(
        samples,
        key=_sample_sort_key,
    )
    return {
        "format_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "command": list(command),
        "requested_checks": list(requested_checks),
        "returncode": returncode,
        "observation_count": len(sorted_samples),
        "profiles": [
            {
                "profiler": "cProfile",
                "metric_kind": "cpu_time",
                "unit": "seconds",
                "samples": sorted_samples,
            }
        ],
    }


def write_cprofile_perf_artifact(
    *,
    path: Path,
    profile: cProfile.Profile,
    root: Path,
    command: tuple[str, ...],
    requested_checks: tuple[str, ...],
    returncode: int,
    graph_artifact: Path | None = None,
) -> None:
    payload = build_cprofile_perf_artifact_payload(
        profile=profile,
        root=root,
        command=command,
        requested_checks=requested_checks,
        returncode=returncode,
        graph_artifact=graph_artifact,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


__all__ = [
    "build_cprofile_perf_artifact_payload",
    "write_cprofile_perf_artifact",
]
