from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
import time

from gabion.invariants import never


class DeadlineClock(Protocol):
    def consume(self, ticks: int = 1) -> None:
        """Consume logical progress units."""

    def get_mark(self) -> int:
        """Return the current monotonic mark for profiling/deltas."""


class DeadlineClockExhausted(RuntimeError):
    """Raised by logical clocks when available ticks are exhausted."""


@dataclass(frozen=True)
class MonotonicClock:
    """Default wall-clock implementation used when no logical clock is injected."""

    def consume(self, ticks: int = 1) -> None:
        # Wall-clock mode does not consume logical gas.
        return

    def get_mark(self) -> int:
        return time.monotonic_ns()


@dataclass
class GasMeter:
    """Deterministic logical clock driven by consumed ticks."""

    limit: int
    current: int = 0

    def __post_init__(self) -> None:
        if int(self.limit) <= 0:
            never("invalid gas meter limit", limit=self.limit)
        self.limit = int(self.limit)
        self.current = int(self.current)
        if self.current < 0:
            never("invalid gas meter current", current=self.current)

    def consume(self, ticks: int = 1) -> None:
        ticks_value = int(ticks)
        if ticks_value <= 0:
            never("invalid gas meter ticks", ticks=ticks)
        self.current += ticks_value
        if self.current >= self.limit:
            raise DeadlineClockExhausted(
                f"Gas exhausted: {self.current}/{self.limit}"
            )

    def get_mark(self) -> int:
        return self.current

