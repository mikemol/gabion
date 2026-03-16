from __future__ import annotations

"""Compatibility facade for legacy indexed dataflow runtime symbols."""

from gabion.analysis.dataflow.engine.dataflow_indexed_file_scan_alias_inventory import (
    BOUNDARY_ADAPTER_LIFECYCLE as _BOUNDARY_ADAPTER_LIFECYCLE,
)
from gabion.analysis.dataflow.engine.dataflow_indexed_file_scan_alias_inventory import (
    materialize_alias_boundary_surface as _materialize_alias_boundary_surface,
)


_ALIAS_SURFACE = _materialize_alias_boundary_surface()
globals().update(_ALIAS_SURFACE.exports)
DATAFLOW_INDEXED_FILE_SCAN_ALIAS_SURFACE_INVENTORY = _ALIAS_SURFACE.inventory
DATAFLOW_INDEXED_FILE_SCAN_RETIREMENT_TELEMETRY = _ALIAS_SURFACE.telemetry
__all__ = tuple(DATAFLOW_INDEXED_FILE_SCAN_ALIAS_SURFACE_INVENTORY["star_export_names"])
