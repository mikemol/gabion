"""Gabion package root."""

from gabion.exceptions import NeverRaise, NeverThrown
from gabion.invariants import never

__all__ = ["__version__", "NeverRaise", "NeverThrown", "never"]

__version__ = "0.1.5"
