from __future__ import annotations

from pathlib import Path
import re
import textwrap

from tests.path_helpers import REPO_ROOT


def _load():
    repo_root = REPO_ROOT
    from gabion.refactor.engine import RefactorEngine
    from gabion.refactor.model import LoopGeneratorRequest, RefactorPlanOutcome

    return RefactorEngine, LoopGeneratorRequest, RefactorPlanOutcome


# gabion:evidence E:call_footprint::tests/test_refactor_engine.py::test_loop_generator_rewrite_emits_ops_and_filter::engine.py::gabion.refactor.engine.RefactorEngine.plan_loop_generator_rewrite
# gabion:behavior primary=desired
def test_loop_generator_rewrite_emits_ops_and_filter(tmp_path: Path) -> None:
    RefactorEngine, LoopGeneratorRequest, RefactorPlanOutcome = _load()
    target = tmp_path / "sample.py"
    target.write_text(
        textwrap.dedent(
            """
            def apply(xs, out, seen, mapping):
                for item in xs:
                    if item < 0:
                        continue
                    out.append(item * 2)
                    seen.add(item)
                    mapping[item] = item + 1
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    plan = RefactorEngine(project_root=tmp_path).plan_loop_generator_rewrite(
        LoopGeneratorRequest(
            target_path=target,
            target_functions=["apply"],
        )
    )
    assert plan.outcome == RefactorPlanOutcome.APPLIED
    assert not plan.errors
    assert plan.edits
    replacement = plan.edits[0].replacement
    assert "def _iter_apply_loop_" in replacement
    assert "filter(lambda item: not " in replacement
    assert "yield _LoopListAppendOp" in replacement
    assert "yield _LoopSetAddOp" in replacement
    assert "yield _LoopDictSetOp" in replacement
    assert "return _iter_apply_loop_" in replacement
    assert plan.rewrite_plans
    assert plan.rewrite_plans[0].kind == "LOOP_GENERATOR"


# gabion:evidence E:call_footprint::tests/test_refactor_engine.py::test_loop_generator_rewrite_supports_binary_reducer_form::engine.py::gabion.refactor.engine.RefactorEngine.plan_loop_generator_rewrite
# gabion:behavior primary=desired
def test_loop_generator_rewrite_supports_binary_reducer_form(tmp_path: Path) -> None:
    RefactorEngine, LoopGeneratorRequest, RefactorPlanOutcome = _load()
    target = tmp_path / "sample.py"
    target.write_text(
        textwrap.dedent(
            """
            def reduce_only(xs, acc):
                for item in xs:
                    acc = acc + item
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    plan = RefactorEngine(project_root=tmp_path).plan_loop_generator_rewrite(
        LoopGeneratorRequest(
            target_path=target,
            target_functions=["reduce_only"],
        )
    )
    assert plan.outcome == RefactorPlanOutcome.APPLIED
    replacement = plan.edits[0].replacement
    assert "yield _LoopReduceOp" in replacement
    assert re.search(r"operator\s*=\s*['\"]\+['\"]", replacement)


# gabion:evidence E:call_footprint::tests/test_refactor_engine.py::test_loop_generator_rewrite_rejects_break_without_edits::engine.py::gabion.refactor.engine.RefactorEngine.plan_loop_generator_rewrite
# gabion:behavior primary=desired
def test_loop_generator_rewrite_rejects_break_without_edits(tmp_path: Path) -> None:
    RefactorEngine, LoopGeneratorRequest, RefactorPlanOutcome = _load()
    target = tmp_path / "sample.py"
    target.write_text(
        textwrap.dedent(
            """
            def bad(xs, out):
                for item in xs:
                    if item > 10:
                        break
                    out.append(item)
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    plan = RefactorEngine(project_root=tmp_path).plan_loop_generator_rewrite(
        LoopGeneratorRequest(
            target_path=target,
            target_functions=["bad"],
        )
    )
    assert plan.outcome == RefactorPlanOutcome.ERROR
    assert not plan.edits
    assert plan.errors
    assert plan.rewrite_plans
    assert plan.rewrite_plans[0].status == "ABSTAINED"


# gabion:evidence E:call_footprint::tests/test_refactor_engine.py::test_loop_generator_rewrite_is_idempotent_on_second_pass::engine.py::gabion.refactor.engine.RefactorEngine.plan_loop_generator_rewrite
# gabion:behavior primary=desired
def test_loop_generator_rewrite_is_idempotent_on_second_pass(tmp_path: Path) -> None:
    RefactorEngine, LoopGeneratorRequest, RefactorPlanOutcome = _load()
    target = tmp_path / "sample.py"
    target.write_text(
        textwrap.dedent(
            """
            def apply(xs, out):
                for item in xs:
                    out.append(item)
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    engine = RefactorEngine(project_root=tmp_path)
    request = LoopGeneratorRequest(target_path=target, target_functions=["apply"])
    first = engine.plan_loop_generator_rewrite(request)
    assert first.outcome == RefactorPlanOutcome.APPLIED
    target.write_text(first.edits[0].replacement, encoding="utf-8")
    second = engine.plan_loop_generator_rewrite(request)
    assert second.outcome == RefactorPlanOutcome.NO_CHANGES
    assert not second.edits
    assert second.rewrite_plans
    assert any("helper chase from: apply" in entry.summary for entry in second.rewrite_plans)


# gabion:evidence E:call_footprint::tests/test_refactor_engine.py::test_loop_generator_reference_equivalence_collection_ops::engine.py::gabion.refactor.engine.RefactorEngine.plan_loop_generator_rewrite
# gabion:behavior primary=desired
def test_loop_generator_reference_equivalence_collection_ops(tmp_path: Path) -> None:
    RefactorEngine, LoopGeneratorRequest, RefactorPlanOutcome = _load()
    target = tmp_path / "sample.py"
    source = textwrap.dedent(
        """
        def apply(xs, out, seen, mapping):
            for item in xs:
                if item < 0:
                    continue
                out.append(item * 2)
                seen.add(item)
                mapping[item] = item + 1
        """
    ).strip() + "\n"
    target.write_text(source, encoding="utf-8")

    original_globals: dict[str, object] = {}
    exec(source, original_globals)
    original_apply = original_globals["apply"]
    assert callable(original_apply)

    xs = [-2, -1, 0, 1, 2, 3]
    out_expected: list[object] = []
    seen_expected: set[object] = set()
    mapping_expected: dict[object, object] = {}
    original_apply(xs, out_expected, seen_expected, mapping_expected)

    plan = RefactorEngine(project_root=tmp_path).plan_loop_generator_rewrite(
        LoopGeneratorRequest(
            target_path=target,
            target_functions=["apply"],
        )
    )
    assert plan.outcome == RefactorPlanOutcome.APPLIED

    rewritten_globals: dict[str, object] = {}
    exec(plan.edits[0].replacement, rewritten_globals)
    rewritten_apply = rewritten_globals["apply"]
    assert callable(rewritten_apply)

    out_actual: list[object] = []
    seen_actual: set[object] = set()
    mapping_actual: dict[object, object] = {}
    events = list(rewritten_apply(xs, out_actual, seen_actual, mapping_actual))
    for event in events:
        kind = type(event).__name__
        if kind == "_LoopListAppendOp":
            out_actual.append(event.value)
        elif kind == "_LoopSetAddOp":
            seen_actual.add(event.value)
        elif kind == "_LoopDictSetOp":
            mapping_actual[event.key] = event.value
        else:  # pragma: no cover - unsupported event kinds are intentionally absent here
            raise AssertionError(f"unexpected event kind: {kind}")

    assert out_actual == out_expected
    assert seen_actual == seen_expected
    assert mapping_actual == mapping_expected


# gabion:behavior primary=desired
def test_loop_generator_nested_peeling_with_repeated_invocation(tmp_path: Path) -> None:
    RefactorEngine, LoopGeneratorRequest, RefactorPlanOutcome = _load()
    target = tmp_path / "sample.py"
    target.write_text(
        textwrap.dedent(
            """
            def nested(xs, ys, out):
                for x in xs:
                    for y in ys:
                        out.append(y)
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    engine = RefactorEngine(project_root=tmp_path)
    request = LoopGeneratorRequest(target_path=target, target_functions=["nested"])

    first = engine.plan_loop_generator_rewrite(request)
    assert first.outcome == RefactorPlanOutcome.APPLIED
    assert first.edits
    target.write_text(first.edits[0].replacement, encoding="utf-8")

    second = engine.plan_loop_generator_rewrite(request)
    assert second.outcome == RefactorPlanOutcome.NO_CHANGES
    assert second.rewrite_plans
    assert any("helper chase from: nested" in entry.summary for entry in second.rewrite_plans)


# gabion:behavior primary=desired
def test_loop_generator_partial_apply_mixed_success_and_failure(tmp_path: Path) -> None:
    RefactorEngine, LoopGeneratorRequest, RefactorPlanOutcome = _load()
    target = tmp_path / "sample.py"
    target.write_text(
        textwrap.dedent(
            """
            def good(xs, out):
                for x in xs:
                    out.append(x)

            def bad(xs, out):
                while xs:
                    out.append(xs.pop())
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    plan = RefactorEngine(project_root=tmp_path).plan_loop_generator_rewrite(
        LoopGeneratorRequest(
            target_path=target,
            target_functions=["good", "bad"],
        )
    )
    assert plan.outcome == RefactorPlanOutcome.APPLIED
    assert plan.edits
    assert plan.errors
    assert any("bad" in reason for reason in plan.errors)
    assert any(entry.status == "applied" for entry in plan.rewrite_plans)
    assert any(entry.status == "ABSTAINED" for entry in plan.rewrite_plans)


# gabion:behavior primary=desired
def test_loop_generator_nested_target_loop_line_selection(tmp_path: Path) -> None:
    RefactorEngine, LoopGeneratorRequest, RefactorPlanOutcome = _load()
    target = tmp_path / "sample.py"
    source = textwrap.dedent(
        """
        def nested(xs, ys, out):
            for x in xs:
                for y in ys:
                    out.append(y)
        """
    ).strip() + "\n"
    target.write_text(source, encoding="utf-8")
    engine = RefactorEngine(project_root=tmp_path)

    select_inner = engine.plan_loop_generator_rewrite(
        LoopGeneratorRequest(
            target_path=target,
            target_functions=["nested"],
            target_loop_lines=[3],
        )
    )
    assert select_inner.outcome == RefactorPlanOutcome.APPLIED

    select_outer = engine.plan_loop_generator_rewrite(
        LoopGeneratorRequest(
            target_path=target,
            target_functions=["nested"],
            target_loop_lines=[2],
        )
    )
    assert select_outer.outcome == RefactorPlanOutcome.ERROR
    assert any("nested loops are not supported" in reason for reason in select_outer.errors)

    select_missing = engine.plan_loop_generator_rewrite(
        LoopGeneratorRequest(
            target_path=target,
            target_functions=["nested"],
            target_loop_lines=[999],
        )
    )
    assert select_missing.outcome == RefactorPlanOutcome.ERROR
    assert any("no loop matched requested target_loop_lines" in reason for reason in select_missing.errors)
