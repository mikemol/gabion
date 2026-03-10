from __future__ import annotations

"""ASPF mutation-log facade.

Implementation is hosted in foundation to keep ASPF neighborhood surfaces thin
under force-majeure strictification.
"""

from gabion.analysis.foundation.aspf_mutation_log_impl import *  # noqa: F401,F403
from gabion.analysis.foundation.aspf_mutation_log_impl import (
    _decode_varint,
    _encode_length_delimited,
    _encode_uint64,
    _encode_varint,
    _parse_wire_fields,
    _wire_bytes_to_object,
)

