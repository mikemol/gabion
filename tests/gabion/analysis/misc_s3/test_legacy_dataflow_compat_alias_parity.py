from __future__ import annotations

import ast
import importlib
from pathlib import Path


def _load(module_path: str):
    return importlib.import_module(module_path)


def _assert_alias_parity(*, owner_module_path: str, canonical_module_path: str) -> None:
    owner = _load(owner_module_path)
    canonical = _load(canonical_module_path)

    owner_all = getattr(owner, "__all__", ())
    assert owner_all, f"{owner_module_path} must expose __all__"

    missing = [name for name in owner_all if not hasattr(canonical, name)]
    assert not missing, (
        f"{owner_module_path} exports symbols missing from canonical owner "
        f"{canonical_module_path}: {missing}"
    )

    mismatched = [
        name
        for name in owner_all
        if getattr(owner, name) is not getattr(canonical, name)
    ]
    assert not mismatched, (
        f"{owner_module_path} exports non-alias symbols vs {canonical_module_path}: "
        f"{mismatched}"
    )


def _assert_symbol_alias(
    *,
    module_path: str,
    canonical_module_path: str,
    symbol: str,
) -> None:
    module = _load(module_path)
    canonical = _load(canonical_module_path)
    assert hasattr(module, symbol)
    assert hasattr(canonical, symbol)
    assert getattr(module, symbol) is getattr(canonical, symbol)


def _monolith_post_phase_aliases() -> tuple[str, ...]:
    return _monolith_aliases_for(
        "gabion.analysis.dataflow.engine.dataflow_post_phase_analyses"
    )


def _monolith_projection_aliases() -> tuple[str, ...]:
    return _monolith_aliases_for(
        "gabion.analysis.dataflow.engine.dataflow_projection_materialization"
    )


def _monolith_resume_aliases() -> tuple[str, ...]:
    return _monolith_aliases_for(
        "gabion.analysis.dataflow.engine.dataflow_resume_serialization"
    )


def _monolith_analysis_index_aliases() -> tuple[str, ...]:
    return _monolith_aliases_for(
        "gabion.analysis.dataflow.engine.dataflow_analysis_index"
    )


def _monolith_aliases_for(module_path: str) -> tuple[str, ...]:
    return tuple(bound for bound, _ in _monolith_alias_bindings_for(module_path))


def _monolith_alias_bindings_for(module_path: str) -> tuple[tuple[str, str], ...]:
    return _module_alias_bindings_for(
        source_path=(
            Path(__file__).resolve().parents[4]
            / "src/gabion/analysis/dataflow/engine/dataflow_indexed_file_scan.py"
        ),
        module_path=module_path,
    )


def _module_alias_bindings_for(
    *,
    source_path: Path,
    module_path: str,
) -> tuple[tuple[str, str], ...]:
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    bindings: list[tuple[str, str]] = []
    seen_bound_names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == module_path:
            for alias in node.names:
                bound_name = alias.asname or alias.name
                if bound_name in seen_bound_names:
                    continue
                seen_bound_names.add(bound_name)
                bindings.append((bound_name, alias.name))
    if not bindings:
        raise AssertionError(
            f"import surface not found for {module_path} in {source_path}"
        )
    return tuple(bindings)


def _module_alias_bindings_by_module(*, source_path: Path) -> dict[str, tuple[tuple[str, str], ...]]:
    repo_root = Path(__file__).resolve().parents[4]
    if not source_path.is_absolute():
        source_path = repo_root / source_path
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    by_module: dict[str, list[tuple[str, str]]] = {}
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom):
            continue
        module = node.module or ""
        if not module:
            continue
        by_module.setdefault(module, [])
        for alias in node.names:
            if alias.name != "*":
                by_module[module].append((alias.asname or alias.name, alias.name))
                continue
            canonical = _load(module)
            canonical_all = getattr(canonical, "__all__", ())
            for symbol in canonical_all:
                by_module[module].append((symbol, symbol))

    normalized: dict[str, tuple[tuple[str, str], ...]] = {}
    for module, bindings in by_module.items():
        seen_bound_names: set[str] = set()
        deduped: list[tuple[str, str]] = []
        for bound_name, source_name in bindings:
            if bound_name in seen_bound_names:
                continue
            seen_bound_names.add(bound_name)
            deduped.append((bound_name, source_name))
        normalized[module] = tuple(deduped)
    return normalized


def test_legacy_owner_modules_preserve_alias_parity() -> None:
    _assert_alias_parity(
        owner_module_path=(
            "gabion.analysis.dataflow.engine.dataflow_analysis_index_owner"
        ),
        canonical_module_path="gabion.analysis.dataflow.engine.dataflow_analysis_index",
    )
    _assert_alias_parity(
        owner_module_path=(
            "gabion.analysis.dataflow.engine.dataflow_deadline_runtime_owner"
        ),
        canonical_module_path=(
            "gabion.analysis.dataflow.engine.dataflow_deadline_runtime"
        ),
    )
    _assert_alias_parity(
        owner_module_path=(
            "gabion.analysis.dataflow.engine.dataflow_runtime_reporting_owner"
        ),
        canonical_module_path=(
            "gabion.analysis.dataflow.engine.dataflow_runtime_reporting"
        ),
    )
    _assert_alias_parity(
        owner_module_path=(
            "gabion.analysis.dataflow.engine.dataflow_deadline_summary_owner"
        ),
        canonical_module_path=(
            "gabion.analysis.dataflow.engine.dataflow_deadline_summary"
        ),
    )


def test_legacy_owner_modules_match_canonical_all_surface_exactly() -> None:
    owner_to_canonical = {
        "gabion.analysis.dataflow.engine.dataflow_analysis_index_owner": (
            "gabion.analysis.dataflow.engine.dataflow_analysis_index"
        ),
        "gabion.analysis.dataflow.engine.dataflow_deadline_runtime_owner": (
            "gabion.analysis.dataflow.engine.dataflow_deadline_runtime"
        ),
        "gabion.analysis.dataflow.engine.dataflow_runtime_reporting_owner": (
            "gabion.analysis.dataflow.engine.dataflow_runtime_reporting"
        ),
        "gabion.analysis.dataflow.engine.dataflow_deadline_summary_owner": (
            "gabion.analysis.dataflow.engine.dataflow_deadline_summary"
        ),
    }
    for owner_module_path, canonical_module_path in owner_to_canonical.items():
        owner = _load(owner_module_path)
        canonical = _load(canonical_module_path)
        owner_all = tuple(getattr(owner, "__all__", ()))
        canonical_all = tuple(getattr(canonical, "__all__", ()))
        assert owner_all == canonical_all, (
            "legacy owner compatibility module must match canonical __all__ "
            f"surface exactly; owner={owner_module_path} "
            f"canonical={canonical_module_path}"
        )


def test_legacy_monolith_and_facade_selected_symbol_alias_parity() -> None:
    # Monolith compatibility aliases by phase owner.
    _assert_symbol_alias(
        module_path="gabion.analysis.dataflow.engine.dataflow_indexed_file_scan",
        canonical_module_path="gabion.analysis.dataflow.engine.dataflow_analysis_index",
        symbol="_build_analysis_index",
    )
    _assert_symbol_alias(
        module_path="gabion.analysis.dataflow.engine.dataflow_indexed_file_scan",
        canonical_module_path="gabion.analysis.dataflow.engine.dataflow_deadline_helpers",
        symbol="_DeadlineFunctionCollector",
    )
    _assert_symbol_alias(
        module_path="gabion.analysis.dataflow.engine.dataflow_indexed_file_scan",
        canonical_module_path=(
            "gabion.analysis.dataflow.engine.dataflow_post_phase_analyses"
        ),
        symbol="analyze_deadness_flow_repo",
    )
    _assert_symbol_alias(
        module_path="gabion.analysis.dataflow.engine.dataflow_indexed_file_scan",
        canonical_module_path=(
            "gabion.analysis.dataflow.engine.dataflow_projection_materialization"
        ),
        symbol="_materialize_projection_spec_rows",
    )
    _assert_symbol_alias(
        module_path="gabion.analysis.dataflow.engine.dataflow_indexed_file_scan",
        canonical_module_path=(
            "gabion.analysis.dataflow.engine.dataflow_resume_serialization"
        ),
        symbol="_serialize_file_scan_resume_state",
    )

    # Facade compatibility aliases across canonical owners.
    _assert_symbol_alias(
        module_path="gabion.analysis.dataflow.engine.dataflow_facade",
        canonical_module_path="gabion.analysis.dataflow.engine.dataflow_analysis_index",
        symbol="_build_analysis_index",
    )
    _assert_symbol_alias(
        module_path="gabion.analysis.dataflow.engine.dataflow_facade",
        canonical_module_path="gabion.analysis.dataflow.engine.dataflow_deadline_helpers",
        symbol="_resolve_callee",
    )
    _assert_symbol_alias(
        module_path="gabion.analysis.dataflow.engine.dataflow_facade",
        canonical_module_path=(
            "gabion.analysis.dataflow.engine.dataflow_runtime_reporting"
        ),
        symbol="_report_section_spec",
    )


def test_facade_covers_monolith_post_phase_alias_surface() -> None:
    facade = _load("gabion.analysis.dataflow.engine.dataflow_facade")
    canonical = _load("gabion.analysis.dataflow.engine.dataflow_post_phase_analyses")
    for symbol in _monolith_post_phase_aliases():
        assert hasattr(facade, symbol), (
            "facade must carry full monolith post-phase compatibility surface; "
            f"missing={symbol}"
        )
        assert getattr(facade, symbol) is getattr(canonical, symbol), (
            "facade post-phase symbol must remain an alias to canonical owner; "
            f"symbol={symbol}"
        )


def test_facade_covers_monolith_projection_alias_surface() -> None:
    facade = _load("gabion.analysis.dataflow.engine.dataflow_facade")
    canonical = _load(
        "gabion.analysis.dataflow.engine.dataflow_projection_materialization"
    )
    for symbol in _monolith_projection_aliases():
        assert hasattr(facade, symbol), (
            "facade must carry full monolith projection compatibility surface; "
            f"missing={symbol}"
        )
        assert getattr(facade, symbol) is getattr(canonical, symbol), (
            "facade projection symbol must remain an alias to canonical owner; "
            f"symbol={symbol}"
        )


def test_facade_covers_monolith_resume_alias_surface() -> None:
    facade = _load("gabion.analysis.dataflow.engine.dataflow_facade")
    canonical = _load("gabion.analysis.dataflow.engine.dataflow_resume_serialization")
    for symbol in _monolith_resume_aliases():
        assert hasattr(facade, symbol), (
            "facade must carry full monolith resume compatibility surface; "
            f"missing={symbol}"
        )
        assert getattr(facade, symbol) is getattr(canonical, symbol), (
            "facade resume symbol must remain an alias to canonical owner; "
            f"symbol={symbol}"
        )


def test_facade_covers_monolith_analysis_index_alias_surface() -> None:
    facade = _load("gabion.analysis.dataflow.engine.dataflow_facade")
    canonical = _load("gabion.analysis.dataflow.engine.dataflow_analysis_index")
    for symbol in _monolith_analysis_index_aliases():
        assert hasattr(facade, symbol), (
            "facade must carry full monolith analysis-index compatibility surface; "
            f"missing={symbol}"
        )
        assert getattr(facade, symbol) is getattr(canonical, symbol), (
            "facade analysis-index symbol must remain an alias to canonical owner; "
            f"symbol={symbol}"
        )


def test_facade_covers_monolith_analysis_support_alias_surfaces() -> None:
    facade = _load("gabion.analysis.dataflow.engine.dataflow_facade")
    module_paths = (
        "gabion.analysis.dataflow.engine.dataflow_call_graph_algorithms",
        "gabion.analysis.dataflow.engine.dataflow_callee_resolution_support",
        "gabion.analysis.dataflow.engine.dataflow_deadline_contracts",
        "gabion.analysis.dataflow.engine.dataflow_deadline_helpers",
        "gabion.analysis.dataflow.engine.dataflow_evidence_helpers",
        "gabion.analysis.dataflow.engine.dataflow_function_index_decision_support",
        "gabion.analysis.dataflow.engine.dataflow_function_index_runtime_support",
        "gabion.analysis.dataflow.engine.dataflow_function_semantics",
        "gabion.analysis.dataflow.engine.dataflow_ingest_helpers",
        "gabion.analysis.dataflow.engine.dataflow_lint_helpers",
    )
    for module_path in module_paths:
        canonical = _load(module_path)
        for symbol in _monolith_aliases_for(module_path):
            assert hasattr(facade, symbol), (
                "facade must carry full monolith analysis-support compatibility "
                f"surface; module={module_path} missing={symbol}"
            )
            assert getattr(facade, symbol) is getattr(canonical, symbol), (
                "facade analysis-support symbol must remain an alias to canonical "
                f"owner; module={module_path} symbol={symbol}"
            )


def test_facade_covers_monolith_external_support_alias_surfaces() -> None:
    facade = _load("gabion.analysis.dataflow.engine.dataflow_facade")
    module_paths = (
        "gabion.analysis.aspf.aspf",
        "gabion.analysis.core.visitors",
        "gabion.analysis.foundation.timeout_context",
        "gabion.analysis.projection.projection_registry",
    )
    for module_path in module_paths:
        canonical = _load(module_path)
        for symbol in _monolith_aliases_for(module_path):
            assert hasattr(facade, symbol), (
                "facade must carry full monolith external-support compatibility "
                f"surface; module={module_path} missing={symbol}"
            )
            assert getattr(facade, symbol) is getattr(canonical, symbol), (
                "facade external-support symbol must remain an alias to canonical "
                f"owner; module={module_path} symbol={symbol}"
            )


def test_facade_covers_monolith_reporting_alias_surface() -> None:
    facade = _load("gabion.analysis.dataflow.engine.dataflow_facade")
    canonical = _load("gabion.analysis.dataflow.io.dataflow_reporting")
    for bound_name, source_name in _monolith_alias_bindings_for(
        "gabion.analysis.dataflow.io.dataflow_reporting"
    ):
        assert hasattr(facade, bound_name), (
            "facade must carry full monolith reporting compatibility surface; "
            f"missing={bound_name}"
        )
        assert getattr(facade, bound_name) is getattr(canonical, source_name), (
            "facade reporting symbol must remain an alias to canonical owner; "
            f"bound={bound_name} source={source_name}"
        )


def test_facade_covers_all_common_monolith_import_surfaces() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    monolith_path = (
        repo_root / "src/gabion/analysis/dataflow/engine/dataflow_indexed_file_scan.py"
    )
    facade_path = repo_root / "src/gabion/analysis/dataflow/engine/dataflow_facade.py"
    facade = _load("gabion.analysis.dataflow.engine.dataflow_facade")

    monolith_by_module = _module_alias_bindings_by_module(source_path=monolith_path)
    facade_by_module = _module_alias_bindings_by_module(source_path=facade_path)

    for module_path in sorted(set(monolith_by_module) & set(facade_by_module)):
        canonical = _load(module_path)
        facade_bound_names = {bound for bound, _ in facade_by_module[module_path]}
        for bound_name, source_name in monolith_by_module[module_path]:
            assert bound_name in facade_bound_names, (
                "facade must cover all common monolith import surfaces; "
                f"module={module_path} missing={bound_name}"
            )
            assert getattr(facade, bound_name) is getattr(canonical, source_name), (
                "facade common-surface symbol must alias canonical source symbol; "
                f"module={module_path} bound={bound_name} source={source_name}"
            )
