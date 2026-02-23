#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable


SOURCE_GLOB = "src/gabion/**/*.py"
TEST_GLOB = "tests/**/*.py"

_PRIVATE_REF_PATTERNS: dict[str, re.Pattern[str]] = {
    "server": re.compile(r"\bserver\._[A-Za-z0-9_]+\b"),
    "dataflow_audit": re.compile(r"\bdataflow_audit\._[A-Za-z0-9_]+\b"),
    "cli": re.compile(r"\bcli\._[A-Za-z0-9_]+\b"),
}

_BRANCH_NODE_TYPES = (
    ast.If,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.Try,
    ast.TryStar,
    ast.Match,
    ast.IfExp,
    ast.BoolOp,
    ast.ExceptHandler,
    ast.With,
    ast.AsyncWith,
    ast.comprehension,
)

_FOCUS_FUNCTIONS = frozenset(
    {
        "src/gabion/server.py::_execute_command_total",
        "src/gabion/analysis/dataflow_audit.py::analyze_paths",
        "src/gabion/analysis/dataflow_audit.py::_collect_deadline_obligations",
        "src/gabion/analysis/dataflow_audit.py::_emit_report",
    }
)


@dataclass(frozen=True)
class FunctionMetric:
    path: str
    qualname: str
    line_start: int
    line_end: int
    line_count: int
    branch_count: int

    @property
    def key(self) -> str:
        return f"{self.path}::{self.qualname}"


@dataclass(frozen=True)
class TestFileMetric:
    path: str
    test_case_count: int
    private_ref_counts: dict[str, int]


def _iter_python_files(root: Path, pattern: str) -> list[Path]:
    return sorted(
        path
        for path in root.glob(pattern)
        if path.is_file() and not any(part == "__pycache__" for part in path.parts)
    )


def _safe_parse(path: Path) -> ast.Module:
    source = path.read_text()
    return ast.parse(source, filename=str(path))


def _collect_function_metrics(path: Path, rel_path: str) -> list[FunctionMetric]:
    tree = _safe_parse(path)
    metrics: list[FunctionMetric] = []

    def _walk(node: ast.AST, scope: list[str]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                _walk(child, [*scope, child.name])
                continue
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                line_start = getattr(child, "lineno", 1)
                line_end = getattr(child, "end_lineno", line_start)
                line_count = max(1, line_end - line_start + 1)
                branch_count = sum(
                    1 for branch in ast.walk(child) if isinstance(branch, _BRANCH_NODE_TYPES)
                )
                qualname = ".".join([*scope, child.name]) if scope else child.name
                metrics.append(
                    FunctionMetric(
                        path=rel_path,
                        qualname=qualname,
                        line_start=line_start,
                        line_end=line_end,
                        line_count=line_count,
                        branch_count=branch_count,
                    )
                )
                _walk(child, [*scope, child.name])
                continue
            _walk(child, scope)

    _walk(tree, [])
    return metrics


def _collect_test_file_metric(path: Path, rel_path: str) -> TestFileMetric:
    tree = _safe_parse(path)
    test_case_count = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith(
            "test_"
        ):
            test_case_count += 1
    source = path.read_text()
    private_ref_counts = {
        module: len(pattern.findall(source))
        for module, pattern in _PRIVATE_REF_PATTERNS.items()
    }
    return TestFileMetric(
        path=rel_path,
        test_case_count=test_case_count,
        private_ref_counts=private_ref_counts,
    )


def _top_items[T](values: Iterable[T], key: str, limit: int) -> list[T]:
    return sorted(values, key=lambda value: getattr(value, key), reverse=True)[:limit]


def _focus_metrics(function_metrics: list[FunctionMetric]) -> dict[str, dict[str, int]]:
    indexed = {metric.key: metric for metric in function_metrics}
    out: dict[str, dict[str, int]] = {}
    for key in sorted(_FOCUS_FUNCTIONS):
        metric = indexed.get(key)
        if metric is None:
            continue
        out[key] = {"line_count": metric.line_count, "branch_count": metric.branch_count}
    return out


def _private_ref_totals(test_metrics: list[TestFileMetric]) -> dict[str, int]:
    totals = {module: 0 for module in _PRIVATE_REF_PATTERNS}
    for metric in test_metrics:
        for module, count in metric.private_ref_counts.items():
            totals[module] = totals.get(module, 0) + int(count)
    return totals


def _top_modules_by_private_refs(test_metrics: list[TestFileMetric]) -> list[dict[str, int]]:
    totals = _private_ref_totals(test_metrics)
    return [
        {"module": module, "private_ref_count": count}
        for module, count in sorted(totals.items(), key=lambda item: item[1], reverse=True)
    ]


def _build_report(root: Path, *, top_n: int) -> dict[str, object]:
    source_files = _iter_python_files(root, SOURCE_GLOB)
    test_files = _iter_python_files(root, TEST_GLOB)

    function_metrics: list[FunctionMetric] = []
    for path in source_files:
        rel_path = str(path.relative_to(root))
        function_metrics.extend(_collect_function_metrics(path, rel_path))

    test_metrics: list[TestFileMetric] = []
    for path in test_files:
        rel_path = str(path.relative_to(root))
        test_metrics.append(_collect_test_file_metric(path, rel_path))

    top_by_lines = _top_items(function_metrics, "line_count", top_n)
    top_by_branches = _top_items(function_metrics, "branch_count", top_n)
    top_test_files = _top_items(test_metrics, "test_case_count", top_n)
    top4_test_total = sum(metric.test_case_count for metric in top_test_files[:4])

    max_line_count = top_by_lines[0].line_count if top_by_lines else 0
    max_branch_count = top_by_branches[0].branch_count if top_by_branches else 0

    payload: dict[str, object] = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "root": str(root),
        "scan": {
            "source_glob": SOURCE_GLOB,
            "test_glob": TEST_GLOB,
            "source_file_count": len(source_files),
            "test_file_count": len(test_files),
            "function_count": len(function_metrics),
        },
        "summary": {
            "max_function_line_count": max_line_count,
            "max_function_branch_count": max_branch_count,
            "top4_test_case_total": top4_test_total,
            "private_ref_counts": _private_ref_totals(test_metrics),
        },
        "focus_functions": _focus_metrics(function_metrics),
        "top_functions_by_lines": [asdict(metric) for metric in top_by_lines],
        "top_functions_by_branch_count": [asdict(metric) for metric in top_by_branches],
        "top_test_files_by_case_count": [asdict(metric) for metric in top_test_files],
        "top_modules_by_private_test_refs": _top_modules_by_private_refs(test_metrics),
    }
    return payload


def _load_json(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"baseline at {path} must be a JSON object")
    return data


def _int_from_path(data: dict[str, object], *path: str) -> int | None:
    node: object = data
    for key in path:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return int(node) if isinstance(node, int) else None


def _dict_from_path(data: dict[str, object], *path: str) -> dict[str, object] | None:
    node: object = data
    for key in path:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node if isinstance(node, dict) else None


def _regression_messages(
    *,
    current: dict[str, object],
    baseline: dict[str, object],
) -> list[str]:
    # dataflow-bundle: baseline, current
    messages: list[str] = []

    comparisons = (
        ("max_function_line_count", ("summary", "max_function_line_count")),
        ("max_function_branch_count", ("summary", "max_function_branch_count")),
        ("top4_test_case_total", ("summary", "top4_test_case_total")),
    )
    for label, keys in comparisons:
        current_value = _int_from_path(current, *keys)
        baseline_value = _int_from_path(baseline, *keys)
        if current_value is None or baseline_value is None:
            continue
        if current_value > baseline_value:
            messages.append(
                f"{label} regressed: current={current_value} baseline={baseline_value}"
            )

    current_private = _dict_from_path(current, "summary", "private_ref_counts") or {}
    baseline_private = _dict_from_path(baseline, "summary", "private_ref_counts") or {}
    for module in sorted(_PRIVATE_REF_PATTERNS):
        current_value = current_private.get(module)
        baseline_value = baseline_private.get(module)
        if not isinstance(current_value, int) or not isinstance(baseline_value, int):
            continue
        if current_value > baseline_value:
            messages.append(
                f"private_ref_counts[{module}] regressed: current={current_value} baseline={baseline_value}"
            )

    current_focus = _dict_from_path(current, "focus_functions") or {}
    baseline_focus = _dict_from_path(baseline, "focus_functions") or {}
    for key in sorted(_FOCUS_FUNCTIONS):
        current_metric = current_focus.get(key)
        baseline_metric = baseline_focus.get(key)
        if not isinstance(current_metric, dict) or not isinstance(baseline_metric, dict):
            continue
        current_line = current_metric.get("line_count")
        baseline_line = baseline_metric.get("line_count")
        if isinstance(current_line, int) and isinstance(baseline_line, int):
            if current_line > baseline_line:
                messages.append(
                    f"{key} line_count regressed: current={current_line} baseline={baseline_line}"
                )
        current_branch = current_metric.get("branch_count")
        baseline_branch = baseline_metric.get("branch_count")
        if isinstance(current_branch, int) and isinstance(baseline_branch, int):
            if current_branch > baseline_branch:
                messages.append(
                    f"{key} branch_count regressed: current={current_branch} baseline={baseline_branch}"
                )

    return messages


def _default_baseline_path(root: Path) -> Path:
    return root / "artifacts" / "audit_reports" / "complexity_baseline.json"


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".", help="Repository root")
    parser.add_argument(
        "--emit",
        default=None,
        help="Write computed report JSON to this path",
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help="Baseline report path for --fail-on-regression",
    )
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit non-zero if complexity/test-surface metrics regress against baseline",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of top items to keep for ranked sections",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    root = Path(args.root).resolve()
    report = _build_report(root, top_n=max(1, int(args.top_n)))

    emit_path = Path(args.emit) if args.emit else None
    if emit_path is not None:
        if not emit_path.is_absolute():
            emit_path = (root / emit_path).resolve()
        emit_path.parent.mkdir(parents=True, exist_ok=True)
        emit_path.write_text(json.dumps(report, indent=2) + "\n")
        print(f"[complexity-audit] wrote report: {emit_path}")

    summary = report.get("summary", {})
    print(
        "[complexity-audit] summary "
        f"max_line_count={summary.get('max_function_line_count')} "
        f"max_branch_count={summary.get('max_function_branch_count')} "
        f"top4_test_case_total={summary.get('top4_test_case_total')}"
    )

    if not args.fail_on_regression:
        return 0

    baseline_path = Path(args.baseline) if args.baseline else _default_baseline_path(root)
    if not baseline_path.is_absolute():
        baseline_path = (root / baseline_path).resolve()
    if not baseline_path.exists():
        print(
            f"[complexity-audit] baseline not found for regression check: {baseline_path}",
            file=sys.stderr,
        )
        return 1

    baseline = _load_json(baseline_path)
    regressions = _regression_messages(current=report, baseline=baseline)
    if regressions:
        print("[complexity-audit] regression(s) detected:", file=sys.stderr)
        for message in regressions:
            print(f"- {message}", file=sys.stderr)
        return 1
    print(f"[complexity-audit] no regressions against baseline: {baseline_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
