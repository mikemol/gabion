from __future__ import annotations

from gabion.refactor import engine, model, rewrite_plan


# gabion:behavior primary=desired
def test_refactor_directory_integration_imports() -> None:
    assert engine is not None
    assert model is not None
    assert rewrite_plan is not None
