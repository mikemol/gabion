from __future__ import annotations

from pathlib import Path
import sys


def _load():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from gabion.synthesis.model import NamingContext, SynthesisConfig
    from gabion.synthesis.protocols import Synthesizer

    return NamingContext, SynthesisConfig, Synthesizer


def test_synthesizer_filters_and_builds_specs() -> None:
    NamingContext, SynthesisConfig, Synthesizer = _load()
    synth = Synthesizer(config=SynthesisConfig(max_tier=2, min_bundle_size=2))
    bundle_tiers = {
        frozenset({"a", "b"}): 2,
        frozenset({"x"}): 2,
    }
    plan = synth.plan(bundle_tiers, field_types={"a": "int"})
    assert len(plan.protocols) == 1
    spec = plan.protocols[0]
    assert spec.name == "ABundle"
    assert spec.tier == 2
    assert {f.name for f in spec.fields} == {"a", "b"}
    assert next(f for f in spec.fields if f.name == "a").type_hint == "int"


def test_synthesizer_uses_existing_names() -> None:
    NamingContext, SynthesisConfig, Synthesizer = _load()
    synth = Synthesizer(
        config=SynthesisConfig(max_tier=2, min_bundle_size=1, allow_singletons=True)
    )
    bundle_tiers = {frozenset({"ctx"}): 2}
    context = NamingContext(existing_names={"CtxBundle"})
    plan = synth.plan(bundle_tiers, naming_context=context)
    assert plan.protocols[0].name == "CtxBundle2"


def test_synthesizer_warns_on_empty() -> None:
    NamingContext, SynthesisConfig, Synthesizer = _load()
    synth = Synthesizer(config=SynthesisConfig(max_tier=1, min_bundle_size=2))
    plan = synth.plan({})
    assert plan.protocols == []
    assert plan.warnings


def test_synthesizer_skips_empty_bundle() -> None:
    NamingContext, SynthesisConfig, Synthesizer = _load()
    synth = Synthesizer(config=SynthesisConfig(max_tier=1, min_bundle_size=1))
    plan = synth.plan({frozenset(): 1})
    assert plan.protocols == []
