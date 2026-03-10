from __future__ import annotations

"""ASPF visitors facade.

Implementation is hosted in foundation to keep ASPF neighborhood surfaces thin
under force-majeure strictification.
"""

from gabion.analysis.foundation.aspf_visitors_impl import *  # noqa: F401,F403
from gabion.analysis.foundation.aspf_visitors_impl import (
    _normalize_two_cell_witness_for_replay,
)
