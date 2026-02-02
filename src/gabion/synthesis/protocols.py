from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Set

from gabion.synthesis.model import (
    FieldSpec,
    NamingContext,
    ProtocolSpec,
    SynthesisConfig,
    SynthesisPlan,
)
from gabion.synthesis.naming import suggest_name


@dataclass
class Synthesizer:
    config: SynthesisConfig = field(default_factory=SynthesisConfig)

    def plan(
        self,
        bundle_tiers: Mapping[frozenset[str], int],
        field_types: Mapping[str, str] | None = None,
        naming_context: NamingContext | None = None,
    ) -> SynthesisPlan:
        field_types = field_types or {}
        naming_context = naming_context or NamingContext()
        protocols: List[ProtocolSpec] = []
        warnings: List[str] = []

        existing_names = set(naming_context.existing_names)
        context = NamingContext(
            existing_names=existing_names,
            frequency=naming_context.frequency,
            fallback_prefix=naming_context.fallback_prefix,
        )

        for bundle, tier in bundle_tiers.items():
            bundle_set = set(bundle)
            if not self._bundle_allowed(bundle_set, tier):
                continue
            name = suggest_name(bundle_set, context)
            context.existing_names.add(name)
            fields = self._build_fields(bundle_set, field_types)
            protocols.append(
                ProtocolSpec(
                    name=name,
                    fields=fields,
                    bundle=bundle_set,
                    tier=tier,
                    rationale=f"Tier-{tier} bundle",
                )
            )

        if not protocols:
            warnings.append("No bundles qualified for synthesis.")

        return SynthesisPlan(protocols=protocols, warnings=warnings, errors=[])

    def _bundle_allowed(self, bundle: Set[str], tier: int) -> bool:
        if not bundle:
            return False
        if len(bundle) < self.config.min_bundle_size and not self.config.allow_singletons:
            return False
        return tier <= self.config.max_tier

    def _build_fields(
        self, bundle: Iterable[str], field_types: Dict[str, str]
    ) -> List[FieldSpec]:
        fields: List[FieldSpec] = []
        for name in sorted(bundle):
            type_hint = field_types.get(name)
            fields.append(FieldSpec(name=name, type_hint=type_hint, source_params={name}))
        return fields
