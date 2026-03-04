from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SppfGraphResult:
    graph: dict[str, object]


@dataclass(frozen=True)
class SppfStatusConsistencyResult:
    violations: list[str]
    warnings: list[str]

    @property
    def payload(self) -> dict[str, object]:
        return {
            "violations": self.violations,
            "warnings": self.warnings,
            "summary": {
                "violation_count": len(self.violations),
                "warning_count": len(self.warnings),
            },
        }
