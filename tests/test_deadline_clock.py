from __future__ import annotations

import pytest

from gabion.deadline_clock import DeadlineClockExhausted, GasMeter, MonotonicClock
from gabion.exceptions import NeverThrown


# gabion:evidence E:call_footprint::tests/test_deadline_clock.py::test_monotonic_clock_mark_increases::deadline_clock.py::gabion.deadline_clock.MonotonicClock
def test_monotonic_clock_mark_increases() -> None:
    clock = MonotonicClock()
    first = clock.get_mark()
    second = clock.get_mark()
    assert second >= first
    clock.consume(1)


# gabion:evidence E:call_footprint::tests/test_deadline_clock.py::test_gas_meter_exhausts_at_limit::deadline_clock.py::gabion.deadline_clock.DeadlineClockExhausted::deadline_clock.py::gabion.deadline_clock.GasMeter
def test_gas_meter_exhausts_at_limit() -> None:
    meter = GasMeter(limit=3)
    meter.consume()
    meter.consume()
    with pytest.raises(DeadlineClockExhausted):
        meter.consume()


# gabion:evidence E:call_footprint::tests/test_deadline_clock.py::test_gas_meter_rejects_invalid_inputs::deadline_clock.py::gabion.deadline_clock.GasMeter::exceptions.py::gabion.exceptions.NeverThrown
def test_gas_meter_rejects_invalid_inputs() -> None:
    with pytest.raises(NeverThrown):
        GasMeter(limit=0)
    with pytest.raises(NeverThrown):
        GasMeter(limit=2, current=-1)
    meter = GasMeter(limit=2)
    with pytest.raises(NeverThrown):
        meter.consume(0)
