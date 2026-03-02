from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TESTS_ROOT = Path(__file__).resolve().parent
SCRIPTS_ROOT = REPO_ROOT / "scripts"
FIXTURES_ROOT = TESTS_ROOT / "fixtures"


def fixture_path(*parts: str) -> Path:
    return FIXTURES_ROOT.joinpath(*parts)
