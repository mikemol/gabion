from __future__ import annotations

from pathlib import Path


def write_sample_module(path: Path) -> None:
    path.write_text(
        "from __future__ import annotations\n\n"
        "def alpha(value: int) -> int:\n"
        "    return value + 1\n\n"
        "def beta(value: int) -> int:\n"
        "    return alpha(value)\n",
        encoding="utf-8",
    )
