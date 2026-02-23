from __future__ import annotations

from pathlib import Path

from gabion import server


_TIMEOUT_PAYLOAD = {
    "analysis_timeout_ticks": 50_000,
    "analysis_timeout_tick_ns": 1_000_000,
}


class DummyWorkspace:
    def __init__(self, root_path: str) -> None:
        self.root_path = root_path


class DummyServer:
    def __init__(self, root_path: str) -> None:
        self.workspace = DummyWorkspace(root_path)


def with_timeout(payload: dict[str, object]) -> dict[str, object]:
    merged = {**_TIMEOUT_PAYLOAD, **payload}
    if (
        "analysis_timeout_ticks" not in payload
        and (
            "analysis_timeout_ms" in payload
            or "analysis_timeout_seconds" in payload
        )
    ):
        merged.pop("analysis_timeout_ticks", None)
        merged.pop("analysis_timeout_tick_ns", None)
    return merged


def write_minimal_module(path: Path) -> None:
    path.write_text(
        "def _fixture_target(x: int) -> int:\n"
        "    return x + 1\n",
        encoding="utf-8",
    )


def empty_analysis_result() -> server.AnalysisResult:
    return server.AnalysisResult(
        groups_by_path={},
        param_spans_by_path={},
        bundle_sites_by_path={},
        type_suggestions=[],
        type_ambiguities=[],
        type_callsite_evidence=[],
        constant_smells=[],
        unused_arg_smells=[],
        forest=server.Forest(),
    )
