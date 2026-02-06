"""Exception protocol markers for Gabion analysis."""


class NeverRaise(RuntimeError):
    """Sentinel exception that should be statically unreachable.

    Raising this exception is a signal to Gabion that the code path is expected
    to be proven unreachable by analysis. If it is reachable, Gabion should
    treat it as a violation.
    """


class NeverThrown(NeverRaise):
    """Alias for NeverRaise used by the explicit never() marker."""
