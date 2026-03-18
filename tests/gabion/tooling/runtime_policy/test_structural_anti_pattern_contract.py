from __future__ import annotations

import json
from pathlib import Path

from gabion.tooling.policy_substrate import structural_anti_pattern_contract as contract


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# gabion:behavior primary=desired
def test_collect_findings_flags_targeted_structural_anti_patterns(tmp_path: Path) -> None:
    sample = tmp_path / "src" / "pkg" / "sample.py"
    _write(
        sample,
        """
import ast

def walk_tree(module):
    rows = []
    for node in ast.walk(module):
        if not isinstance(node, ast.FunctionDef):
            continue
        if not node.name.startswith("_"):
            continue
        rows.append(node.name)
    return rows

def iter_pairs(entries):
    items = []
    for entry in entries:
        match_list = entry.get("matches", [])
        if len(match_list) < 2:
            continue
        items.append(match_list)
    return tuple(items)

def choose(value):
    match value:
        case 1:
            return "one"
        case _:
            pass
""".strip()
        + "\n",
    )

    findings = contract.collect_findings_for_path(sample, root=tmp_path)
    codes = {finding.code for finding in findings}

    assert "ast_walk_prefilter_in_loop" in codes
    assert "len_guard_continue" in codes
    assert "eager_tuple_materialization" in codes
    assert "wildcard_soft_fallthrough" in codes


# gabion:behavior primary=desired
def test_collect_findings_does_not_flag_filter_first_streaming_shapes(tmp_path: Path) -> None:
    sample = tmp_path / "src" / "pkg" / "clean.py"
    _write(
        sample,
        """
import ast
from itertools import chain

def _is_private_function(node):
    return isinstance(node, ast.FunctionDef) and node.name.startswith("_")

def walk_tree(module):
    return [node.name for node in filter(_is_private_function, ast.walk(module))]

def iter_pairs(entries):
    return tuple(
        match_list
        for match_list in (
            entry.get("matches", [])
            for entry in filter(lambda item: len(item.get("matches", [])) >= 2, entries)
        )
    )

def expand(targets):
    return chain.from_iterable(iter((target,)) for target in targets)
""".strip()
        + "\n",
    )

    assert contract.collect_findings_for_path(sample, root=tmp_path) == []


# gabion:behavior primary=desired
def test_run_writes_artifact_and_fails_in_check_mode_when_findings_exist(tmp_path: Path) -> None:
    sample = tmp_path / "src" / "pkg" / "sample.py"
    _write(
        sample,
        """
import ast

def walk_tree(module):
    for node in ast.walk(module):
        if not isinstance(node, ast.FunctionDef):
            continue
        return node
    return None
""".strip()
        + "\n",
    )
    artifact = tmp_path / "artifacts" / "out" / "structural_anti_pattern_contract.json"

    rc = contract.run(root=tmp_path, out_path=artifact, check=True)

    assert rc == 1
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert payload["artifact_kind"] == "structural_anti_pattern_contract"
    assert payload["counts"]["total"] >= 1
    assert payload["findings"][0]["rel_path"] == "src/pkg/sample.py"
