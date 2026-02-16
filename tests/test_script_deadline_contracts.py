from __future__ import annotations

from pathlib import Path


def test_scripts_using_deadlines_require_clock_and_forest_scope() -> None:
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    offenders: list[str] = []
    for script_path in sorted(scripts_dir.glob("*.py")):
        text = script_path.read_text(encoding="utf-8")
        uses_deadline_checks = (
            "check_deadline(" in text
            or "deadline_scope(" in text
            or "deadline_loop_iter(" in text
        )
        if not uses_deadline_checks:
            continue
        has_local_scope = "deadline_clock_scope(" in text and "forest_scope(" in text
        has_shared_scope = "deadline_scope_from_" in text
        if not (has_local_scope or has_shared_scope):
            offenders.append(str(script_path))
    assert not offenders, (
        "Scripts using deadline checks must establish deadline scope directly "
        f"or via scripts/deadline_runtime.py: {offenders}"
    )


def test_deadline_runtime_provides_scope_guards() -> None:
    path = Path(__file__).resolve().parents[1] / "scripts" / "deadline_runtime.py"
    text = path.read_text(encoding="utf-8")
    assert "forest_scope(" in text
    assert "deadline_scope(" in text
    assert "deadline_clock_scope(" in text
