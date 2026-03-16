# gabion:ambiguity_boundary_module
# gabion:grade_boundary kind=semantic_carrier_adapter name=dataflow_indexed_file_scan_alias_adapter_projection
from __future__ import annotations

"""Projection/reporting alias groups for the legacy monolith path."""

from gabion.analysis.dataflow.engine.dataflow_indexed_file_scan_alias_contract import (
    AliasGroupSpec,
    alias_group,
    module_alias,
)

PROJECTION_ALIAS_GROUPS: tuple[AliasGroupSpec, ...] = (
    alias_group(
        'projection_materialization',
        'Projection Materialization',
        module_alias(
            'gabion.analysis.dataflow.engine.dataflow_projection_materialization',
            'CallAmbiguity',
            '_ambiguity_suite_relation',
            '_ambiguity_suite_row_to_suite',
            '_ambiguity_virtual_count_gt_1',
            '_collect_call_ambiguities',
            '_collect_call_ambiguities_indexed',
            '_dedupe_call_ambiguities',
            '_emit_call_ambiguities',
            '_format_span_fields',
            '_lint_lines_from_call_ambiguities',
            '_materialize_ambiguity_suite_agg_spec',
            '_materialize_ambiguity_virtual_set_spec',
            '_materialize_projection_spec_rows',
            '_materialize_suite_order_spec',
            '_populate_bundle_forest',
            '_spec_row_span',
            '_summarize_call_ambiguities',
            '_suite_order_relation',
            '_suite_order_row_to_site',
        ),
    ),
    alias_group(
        'reporting_io',
        'Reporting And Projection IO',
        module_alias(
            'gabion.analysis.dataflow.io.dataflow_parse_helpers',
            ('_parse_module_tree_optional', '_parse_module_tree'),
        ),
        module_alias(
            'gabion.analysis.dataflow.engine.dataflow_runtime_reporting',
            '_report_section_spec',
        ),
        module_alias(
            'gabion.analysis.dataflow.io.dataflow_projection_helpers',
            '_topologically_order_report_projection_specs',
        ),
        module_alias(
            'gabion.analysis.dataflow.engine.dataflow_contracts',
            'AuditConfig',
            'CallArgs',
            'ClassInfo',
            'FunctionInfo',
            'InvariantProposition',
            'ParamUse',
            'SymbolTable',
        ),
        module_alias(
            'gabion.analysis.dataflow.io.dataflow_reporting',
            ('emit_report', '_emit_report'),
            'render_report',
        ),
        module_alias(
            'gabion.analysis.dataflow.io.dataflow_reporting_helpers',
            ('render_mermaid_component', '_render_mermaid_component'),
        ),
        module_alias(
            'gabion.analysis.dataflow.io.dataflow_parse_helpers',
            '_ParseModuleStage',
            '_forbid_adhoc_bundle_discovery',
        ),
        module_alias(
            'gabion.analysis.indexed_scan.scanners.report_sections',
            'extract_report_sections',
        ),
    ),
)
