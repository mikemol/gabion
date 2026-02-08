from __future__ import annotations

from pathlib import Path
import sys
import textwrap


def _load_config_module():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from gabion.config import dataflow_defaults, merge_payload

    return dataflow_defaults, merge_payload


# gabion:evidence E:decision_surface/direct::config.py::gabion.config.load_config::config_path,root
def test_dataflow_defaults_reads_toml(tmp_path: Path) -> None:
    config_path = tmp_path / "gabion.toml"
    config_path.write_text(
        textwrap.dedent(
            """
            [dataflow]
            exclude = ["in", "extra"]
            ignore_params = ["self", "cls"]
            strictness = "low"
            allow_external = true
            type_audit = true
            fail_on_type_ambiguities = true
            """
        ).strip()
        + "\n"
    )
    dataflow_defaults, _ = _load_config_module()
    defaults = dataflow_defaults(root=tmp_path, config_path=config_path)
    assert defaults["exclude"] == ["in", "extra"]
    assert defaults["ignore_params"] == ["self", "cls"]
    assert defaults["strictness"] == "low"
    assert defaults["allow_external"] is True
    assert defaults["type_audit"] is True
    assert defaults["fail_on_type_ambiguities"] is True


# gabion:evidence E:function_site::config.py::gabion.config.merge_payload
def test_merge_payload_prefers_explicit_values(tmp_path: Path) -> None:
    dataflow_defaults, merge_payload = _load_config_module()
    defaults = {
        "exclude": ["in"],
        "ignore_params": ["self"],
        "strictness": "high",
        "allow_external": False,
    }
    payload = {
        "exclude": None,
        "ignore_params": ["cls"],
        "strictness": None,
        "allow_external": True,
    }
    merged = merge_payload(payload, defaults)
    assert merged["exclude"] == ["in"]
    assert merged["ignore_params"] == ["cls"]
    assert merged["strictness"] == "high"
    assert merged["allow_external"] is True
