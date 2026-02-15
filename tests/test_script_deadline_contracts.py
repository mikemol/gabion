from __future__ import annotations

from pathlib import Path


def test_scripts_using_deadlines_require_clock_and_forest_scope() -> None:
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    offenders: list[str] = []
    for script_path in sorted(scripts_dir.glob("*.py")):
        text = script_path.read_text(encoding="utf-8")
        uses_deadline_checks = "check_deadline(" in text or "deadline_scope(" in text
        if not uses_deadline_checks:
            continue
        if "deadline_clock_scope(" not in text or "forest_scope(" not in text:
            offenders.append(str(script_path))
    assert not offenders, (
        "Scripts using deadline checks must establish deadline_clock_scope "
        f"and forest_scope: {offenders}"
    )
