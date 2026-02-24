from __future__ import annotations

# Canonical semantic command identifiers shared by CLI, LSP client, and server.
CHECK_COMMAND = "gabion.check"
DATAFLOW_COMMAND = "gabion.dataflowAudit"
DECISION_DIFF_COMMAND = "gabion.decisionDiff"
IMPACT_COMMAND = "gabion.impactQuery"
LSP_PARITY_GATE_COMMAND = "gabion.lspParityGate"
REFACTOR_COMMAND = "gabion.refactorProtocol"
STRUCTURE_DIFF_COMMAND = "gabion.structureDiff"
STRUCTURE_REUSE_COMMAND = "gabion.structureReuse"
SYNTHESIS_COMMAND = "gabion.synthesisPlan"

# Deterministic canonical order is part of the command-boundary contract.
# Sort key is lexical command-id text.
SEMANTIC_COMMAND_IDS: tuple[str, ...] = (
    CHECK_COMMAND,
    DATAFLOW_COMMAND,
    DECISION_DIFF_COMMAND,
    IMPACT_COMMAND,
    LSP_PARITY_GATE_COMMAND,
    REFACTOR_COMMAND,
    STRUCTURE_DIFF_COMMAND,
    STRUCTURE_REUSE_COMMAND,
    SYNTHESIS_COMMAND,
)
