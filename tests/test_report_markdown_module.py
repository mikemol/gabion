from __future__ import annotations

import random

from gabion.analysis.report_markdown import render_report_markdown


def test_render_report_markdown_is_byte_stable_for_shuffled_scope() -> None:
    baseline = None
    scope_entries = ["artifacts", "repo", "analysis"]
    for seed in range(12):
        rng = random.Random(seed)
        shuffled = list(scope_entries)
        rng.shuffle(shuffled)
        rendered = render_report_markdown(
            "demo_report",
            ["## Demo", "", "body"],
            doc_scope=shuffled,
        )
        if baseline is None:
            baseline = rendered
            continue
        assert rendered == baseline
