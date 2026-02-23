from __future__ import annotations

import json
from pathlib import Path

from scripts import order_lifetime_check


def _write_module(root: Path, rel_path: str, content: str) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


# gabion:evidence E:call_footprint::tests/test_order_lifetime_lint.py::test_order_lifetime_check_rejects_sort_keys_true_serializer::order_lifetime_check.py::scripts.order_lifetime_check.collect_violations
def test_order_lifetime_check_rejects_sort_keys_true_serializer(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "src/gabion/lsp_client.py",
        "import json\n"
        "def f(payload):\n"
        "    return json.dumps(payload, sort_keys=True)\n",
    )
    violations = order_lifetime_check.collect_violations(root=tmp_path)
    assert violations
    assert any("sort_keys=True" in violation.message for violation in violations)


# gabion:evidence E:call_footprint::tests/test_order_lifetime_lint.py::test_order_lifetime_check_rejects_sort_keys_true_everywhere::order_lifetime_check.py::scripts.order_lifetime_check.collect_violations
def test_order_lifetime_check_rejects_sort_keys_true_everywhere(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "src/gabion/commands/boundary_order.py",
        "import json\n"
        "def _stable_text(value):\n"
        "    return json.dumps(value, sort_keys=True)\n",
    )
    violations = order_lifetime_check.collect_violations(root=tmp_path)
    assert any("sort_keys=True" in violation.message for violation in violations)


# gabion:evidence E:call_footprint::tests/test_order_lifetime_lint.py::test_order_lifetime_check_rejects_active_ordered_or_sorted_calls::order_lifetime_check.py::scripts.order_lifetime_check.collect_violations
def test_order_lifetime_check_rejects_active_ordered_or_sorted_calls(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "src/gabion/runtime/json_io.py",
        "from gabion.order_contract import ordered_or_sorted\n"
        "def f(values):\n"
        "    return ordered_or_sorted(values, source='x')\n",
    )
    violations = order_lifetime_check.collect_violations(root=tmp_path)
    assert any("direct active-sort mode via ordered_or_sorted" in violation.message for violation in violations)


# gabion:evidence E:call_footprint::tests/test_order_lifetime_lint.py::test_order_lifetime_check_rejects_raw_sorted_in_strict_module::order_lifetime_check.py::scripts.order_lifetime_check.collect_violations
def test_order_lifetime_check_rejects_raw_sorted_in_strict_module(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "src/gabion/runtime/json_io.py",
        "def f(values):\n"
        "    return sorted(values)\n",
    )
    violations = order_lifetime_check.collect_violations(root=tmp_path)
    assert any("raw sorted" in violation.message for violation in violations)


# gabion:evidence E:call_footprint::tests/test_order_lifetime_lint.py::test_order_lifetime_check_requires_sort_once_source::order_lifetime_check.py::scripts.order_lifetime_check.collect_violations
def test_order_lifetime_check_requires_sort_once_source(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "src/gabion/runtime/json_io.py",
        "from gabion.order_contract import sort_once\n"
        "def f(values):\n"
        "    return sort_once(values)\n",
    )
    violations = order_lifetime_check.collect_violations(root=tmp_path)
    assert any("requires source" in violation.message for violation in violations)


# gabion:evidence E:call_footprint::tests/test_order_lifetime_lint.py::test_order_lifetime_check_rejects_dynamic_sort_once_source::order_lifetime_check.py::scripts.order_lifetime_check.collect_violations
def test_order_lifetime_check_rejects_dynamic_sort_once_source(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "src/gabion/runtime/json_io.py",
        "from gabion.order_contract import sort_once\n"
        "def f(values):\n"
        "    source_tag = 'runtime.dynamic'\n"
        "    return sort_once(values, source=source_tag)\n",
    )
    violations = order_lifetime_check.collect_violations(root=tmp_path)
    assert any(
        "must be string literal or f-string" in violation.message
        for violation in violations
    )


# gabion:evidence E:call_footprint::tests/test_order_lifetime_lint.py::test_order_lifetime_check_emits_inventory::order_lifetime_check.py::scripts.order_lifetime_check.collect_violations_and_inventory
def test_order_lifetime_check_emits_inventory(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "src/gabion/analysis/sample.py",
        "from gabion.order_contract import sort_once\n"
        "def f(values):\n"
        "    return sort_once(values, source='sample.sort')\n",
    )
    inventory_path = tmp_path / "artifacts" / "audit_reports" / "inventory.json"
    # Call helpers directly to avoid subprocess overhead.
    violations, inventories = order_lifetime_check.collect_violations_and_inventory(
        root=tmp_path
    )
    assert violations == []
    payload = {
        "target_glob": order_lifetime_check.TARGET_GLOB,
        "totals": {
            "sorted_calls": sum(item.sorted_calls for item in inventories),
            "dot_sort_calls": sum(item.dot_sort_calls for item in inventories),
            "ordered_or_sorted_calls": sum(item.ordered_or_sorted_calls for item in inventories),
            "sort_once_calls": sum(item.sort_once_calls for item in inventories),
            "json_dumps_sort_keys_true_calls": sum(
                item.json_dumps_sort_keys_true_calls for item in inventories
            ),
        },
        "files": [item.as_dict() for item in inventories],
    }
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    inventory_path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    persisted = json.loads(inventory_path.read_text(encoding="utf-8"))
    assert persisted["totals"]["sort_once_calls"] == 1
