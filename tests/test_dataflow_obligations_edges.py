from __future__ import annotations

from gabion.analysis import dataflow_obligations


class _FakeNode:
    def __init__(self, meta: dict[str, object]) -> None:
        self.meta = meta


class _FakeForest:
    def __init__(self, *, suite_id_meta: object) -> None:
        self._suite_id_meta = suite_id_meta
        self.nodes: dict[object, _FakeNode] = {}
        self.added_alts: list[tuple[str, tuple[object, object], dict[str, object]]] = []

    def add_suite_site(
        self,
        path: str,
        function_name: str,
        suite_kind: str,
        *,
        span: tuple[int, int, int, int],
    ) -> object:
        node_id = ("suite", path, function_name, suite_kind, span)
        self.nodes[node_id] = _FakeNode({"suite_id": self._suite_id_meta})
        return node_id

    def add_paramset(self, bundle: list[str]) -> object:
        return ("paramset", tuple(bundle))

    def add_alt(
        self,
        kind: str,
        nodes: tuple[object, object],
        *,
        evidence: dict[str, object],
    ) -> None:
        self.added_alts.append((kind, nodes, dict(evidence)))


# gabion:evidence E:call_footprint::tests/test_dataflow_obligations_edges.py::test_deadline_obligation_builder_skips_suite_id_when_meta_not_nonempty_string::dataflow_obligations.py::gabion.analysis.dataflow_obligations._DeadlineObligationBuilder.add_obligation
def test_deadline_obligation_builder_skips_suite_id_when_meta_not_nonempty_string() -> None:
    dataflow_obligations._bind_audit_symbols()
    forest = _FakeForest(suite_id_meta=0)
    builder = dataflow_obligations._DeadlineObligationBuilder(
        by_qual={},
        facts_by_qual={},
        forest=forest,  # type: ignore[arg-type]
        project_root=None,
    )
    builder.add_obligation(
        path="src/example.py",
        function="pkg.fn",
        param="deadline",
        status="SATISFIED",
        kind="resolved",
        detail="ok",
        span=(1, 0, 1, 8),
        suite_kind="function",
    )
    assert len(builder.obligations) == 1
    site_payload = builder.obligations[0]["site"]
    assert isinstance(site_payload, dict)
    assert "suite_id" not in site_payload
    assert site_payload["suite_kind"] == "function"
    assert len(forest.added_alts) == 1
