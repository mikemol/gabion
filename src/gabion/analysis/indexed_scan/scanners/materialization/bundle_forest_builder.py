# gabion:boundary_normalization_module
# gabion:decision_protocol_module
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from gabion.analysis.foundation.json_types import JSONObject


@dataclass(frozen=True)
class BundleForestBuildDeps:
    check_deadline_fn: Callable[[], None]
    iter_monotonic_paths_fn: Callable[..., list[Path]]
    build_analysis_index_fn: Callable[..., object]
    sort_once_fn: Callable[..., list[object]]
    is_test_path_fn: Callable[[Path], bool]
    materialize_structured_suite_sites_fn: Callable[..., None]
    add_interned_alt_fn: Callable[..., None]
    collect_config_bundles_fn: Callable[..., dict[Path, dict[str, list[str]]]]
    collect_dataclass_registry_fn: Callable[..., dict[str, list[str]]]
    build_symbol_table_fn: Callable[..., object]
    iter_documented_bundles_fn: Callable[..., Iterable[tuple[str, ...]]]
    iter_dataclass_call_bundles_fn: Callable[..., Iterable[tuple[str, ...]]]
    progress_emit_min_interval_seconds: float


def populate_bundle_forest(
    forest: object,
    *,
    groups_by_path: dict[Path, dict[str, list[set[str]]]],
    file_paths: list[Path],
    project_root=None,
    include_all_sites: bool = True,
    ignore_params=None,
    strictness: str = "high",
    transparent_decorators=None,
    parse_failure_witnesses: list[JSONObject],
    analysis_index=None,
    on_progress=None,
    deps: BundleForestBuildDeps,
) -> None:
    deps.check_deadline_fn()
    if not groups_by_path:
        return
    ordered_file_paths = deps.iter_monotonic_paths_fn(
        file_paths,
        source="_populate_bundle_forest.file_paths",
    )
    index = analysis_index
    if include_all_sites and index is None:
        index = deps.build_analysis_index_fn(
            ordered_file_paths,
            project_root=project_root,
            ignore_params=ignore_params or set(),
            strictness=strictness,
            external_filter=True,
            transparent_decorators=transparent_decorators,
            parse_failure_witnesses=parse_failure_witnesses,
        )
    callsite_inventory_total = 0
    if index is not None:
        callsite_inventory_total = sum(len(info.calls) for info in index.by_qual.values())
    site_materialization_total = 0
    site_materialization_done = 0
    structured_suite_total = 0
    structured_suite_done = 0
    group_paths_total = len(groups_by_path)
    group_paths_done = 0
    config_paths_total = 0
    config_paths_done = 0
    dataclass_quals_total = 0
    dataclass_quals_done = 0
    marker_paths_total = len(ordered_file_paths)
    marker_paths_done = 0
    progress_accepts_payload = None
    last_progress_emit_monotonic = None

    def _forest_progress_snapshot(*, marker: str) -> JSONObject:
        mutable_done = (
            site_materialization_done
            + structured_suite_done
            + group_paths_done
            + config_paths_done
            + dataclass_quals_done
            + marker_paths_done
        )
        mutable_total = (
            site_materialization_total
            + structured_suite_total
            + group_paths_total
            + config_paths_total
            + dataclass_quals_total
            + marker_paths_total
        )
        return {
            "format_version": 1,
            "schema": "gabion/forest_progress_v2",
            "primary_unit": "forest_mutable_steps",
            "primary_done": mutable_done,
            "primary_total": mutable_total,
            "dimensions": {
                "site_materialization": {
                    "done": site_materialization_done,
                    "total": site_materialization_total,
                },
                "structured_suite_materialization": {
                    "done": structured_suite_done,
                    "total": structured_suite_total,
                },
                "group_paths": {
                    "done": group_paths_done,
                    "total": group_paths_total,
                },
                "config_paths": {
                    "done": config_paths_done,
                    "total": config_paths_total,
                },
                "dataclass_quals": {
                    "done": dataclass_quals_done,
                    "total": dataclass_quals_total,
                },
                "marker_paths": {
                    "done": marker_paths_done,
                    "total": marker_paths_total,
                },
            },
            "inventory": {
                "callsites_total": callsite_inventory_total,
                "input_file_paths_total": len(ordered_file_paths),
            },
            "marker": marker,
        }

    def _notify_progress(progress_delta: int, *, marker: str) -> None:
        nonlocal progress_accepts_payload
        if on_progress is not None:
            snapshot = _forest_progress_snapshot(marker=marker)
            normalized_delta = max(int(progress_delta), 0)
            if progress_accepts_payload is True:
                on_progress(snapshot)
            elif progress_accepts_payload is False:
                try:
                    on_progress(normalized_delta)
                except TypeError:
                    on_progress()
            else:
                try:
                    on_progress(snapshot)
                    progress_accepts_payload = True
                except TypeError:
                    progress_accepts_payload = False
                    try:
                        on_progress(normalized_delta)
                    except TypeError:
                        on_progress()

    def _emit_progress(*, force: bool = False, marker: str) -> None:
        nonlocal last_progress_emit_monotonic
        if on_progress is not None:
            now = time.monotonic()
            min_interval_elapsed = (
                last_progress_emit_monotonic is None
                or now - last_progress_emit_monotonic
                >= deps.progress_emit_min_interval_seconds
            )
            if force or min_interval_elapsed:
                last_progress_emit_monotonic = now
                _notify_progress(1, marker=marker)

    _notify_progress(0, marker="start")
    if include_all_sites:
        non_test_quals = [
            qual
            for qual in deps.sort_once_fn(
                index.by_qual,
                source="_populate_bundle_forest.index.by_qual",
            )
            if not deps.is_test_path_fn(index.by_qual[qual].path)
        ]
        site_materialization_total = len(non_test_quals)
        for qual in non_test_quals:
            deps.check_deadline_fn()
            info = index.by_qual[qual]
            forest.add_site(info.path.name, info.qual)
            site_materialization_done += 1
            _emit_progress(marker="site_materialization")
        non_test_file_paths = [
            path for path in ordered_file_paths if not deps.is_test_path_fn(path)
        ]
        structured_suite_total = 1
        deps.materialize_structured_suite_sites_fn(
            forest=forest,
            file_paths=non_test_file_paths,
            project_root=project_root,
            parse_failure_witnesses=parse_failure_witnesses,
            analysis_index=index,
        )
        structured_suite_done = 1
        _emit_progress(force=True, marker="structured_suite_materialization")

    def _add_alt(
        kind: str,
        inputs,
        evidence=None,
    ) -> None:
        deps.add_interned_alt_fn(
            forest=forest,
            kind=kind,
            inputs=inputs,
            evidence=evidence,
        )

    for path in deps.sort_once_fn(
        groups_by_path,
        source="_populate_bundle_forest.groups_by_path",
        key=lambda candidate: str(candidate),
    ):
        deps.check_deadline_fn()
        groups = groups_by_path[path]
        for fn_name in deps.sort_once_fn(
            groups,
            source="gabion.analysis.dataflow_indexed_file_scan._populate_bundle_forest.site_1",
        ):
            deps.check_deadline_fn()
            site_id = forest.add_site(path.name, fn_name)
            for bundle in groups[fn_name]:
                deps.check_deadline_fn()
                paramset_id = forest.add_paramset(bundle)
                _add_alt(
                    "SignatureBundle",
                    (site_id, paramset_id),
                    evidence={"path": path.name, "qual": fn_name},
                )
        group_paths_done += 1
        _emit_progress(marker="group_paths")

    config_bundles_by_path = deps.collect_config_bundles_fn(
        ordered_file_paths,
        parse_failure_witnesses=parse_failure_witnesses,
        analysis_index=index,
    )
    config_paths_total = len(config_bundles_by_path)
    _emit_progress(force=True, marker="config_paths_discovered")
    for path in deps.iter_monotonic_paths_fn(
        config_bundles_by_path,
        source="_populate_bundle_forest.config_bundles_by_path",
    ):
        deps.check_deadline_fn()
        bundles = config_bundles_by_path[path]
        for name in deps.sort_once_fn(
            bundles,
            source="gabion.analysis.dataflow_indexed_file_scan._populate_bundle_forest.site_2",
        ):
            deps.check_deadline_fn()
            paramset_id = forest.add_paramset(bundles[name])
            _add_alt(
                "ConfigBundle",
                (paramset_id,),
                evidence={"path": path.name, "name": name},
            )
        config_paths_done += 1
        _emit_progress(marker="config_paths")

    dataclass_registry = deps.collect_dataclass_registry_fn(
        ordered_file_paths,
        project_root=project_root,
        parse_failure_witnesses=parse_failure_witnesses,
        analysis_index=index,
    )
    dataclass_quals_total = len(dataclass_registry)
    _emit_progress(force=True, marker="dataclass_quals_discovered")
    for qual_name in deps.sort_once_fn(
        dataclass_registry,
        source="gabion.analysis.dataflow_indexed_file_scan._populate_bundle_forest.site_3",
    ):
        deps.check_deadline_fn()
        paramset_id = forest.add_paramset(dataclass_registry[qual_name])
        _add_alt(
            "DataclassBundle",
            (paramset_id,),
            evidence={"qual": qual_name},
        )
        dataclass_quals_done += 1
        _emit_progress(marker="dataclass_quals")

    if index is None or not index.symbol_table.external_filter:
        symbol_table = deps.build_symbol_table_fn(
            ordered_file_paths,
            project_root,
            external_filter=True,
            parse_failure_witnesses=parse_failure_witnesses,
        )
    else:
        symbol_table = index.symbol_table
    for path in ordered_file_paths:
        deps.check_deadline_fn()
        for bundle in deps.sort_once_fn(
            deps.iter_documented_bundles_fn(path),
            source="gabion.analysis.dataflow_indexed_file_scan._populate_bundle_forest.site_4",
        ):
            deps.check_deadline_fn()
            paramset_id = forest.add_paramset(bundle)
            _add_alt("MarkerBundle", (paramset_id,), evidence={"path": path.name})
        for bundle in deps.sort_once_fn(
            deps.iter_dataclass_call_bundles_fn(
                path,
                project_root=project_root,
                symbol_table=symbol_table,
                dataclass_registry=dataclass_registry,
                parse_failure_witnesses=parse_failure_witnesses,
            ),
            source="gabion.analysis.dataflow_indexed_file_scan._populate_bundle_forest.site_5",
        ):
            deps.check_deadline_fn()
            paramset_id = forest.add_paramset(bundle)
            _add_alt(
                "DataclassCallBundle",
                (paramset_id,),
                evidence={"path": path.name},
            )
        marker_paths_done += 1
        _emit_progress(marker="marker_paths")
    _emit_progress(force=True, marker="complete")

