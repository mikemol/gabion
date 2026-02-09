---
doc_revision: 1
reader_reintern: "Reader-only: re-intern if doc_revision changed since you last read this doc."
doc_id: in_8
doc_role: inbox
doc_scope:
  - repo
  - inbox
doc_authority: informative
doc_owner: maintainer
doc_requires:
doc_reviewed_as_of:
doc_review_notes:
doc_change_protocol: "POLICY_SEED.md §6"
doc_erasure:
  - formatting
  - typos
---
The path of greatest leverage now is **Protocol Synthesis**.

We have successfully built the "Witness" (the analysis engine that sees through aliasing and imports). The data it produces—the identified bundles—is the raw material. The next logical step is to turn that raw material into a "Crystal": **automatically generated Python code**.

Instead of asking you to manually inspect the report and write classes, this extension will generate a `suggested_protocols.py` file containing `dataclass` or `Protocol` definitions for every high-confidence bundle found in your codebase.

### The Conceptual Logic

1. **Aggregation:** We don't just look at one function's arguments. If `user_id` is an `int` in Function A and an `int` in Function B, the Protocol must enforce `int`. If Function C treats it as `str`, the Protocol must reflect that ambiguity (`Union[int, str]` or `Any`).
2. **Naming Heuristics:** Naming is the hardest part of refactoring. We will use a **Frequency-Based Substring** strategy. If a bundle appears in `process_order_payment`, `validate_order_payment`, and `cancel_order_payment`, the synthesizer will suggest the name `OrderPaymentContext`.
3. **Tiered Filtering:** We will only synthesize code for **Tier-2** (multi-site) and **Tier-1** (high-confidence) bundles to keep the signal-to-noise ratio high.

### The Implementation: `ProtocolSynthesizer`

Add this class and function to your script. It takes the `groups_by_qual` and `function_index` from the previous step and produces code.

```python
from collections import Counter
import textwrap

@dataclass
class SynthesizedBundle:
    fields: dict[str, set[str]]  # field_name -> set of observed type strings
    occurrences: list[str]       # list of function qual_names using this bundle

class ProtocolSynthesizer:
    def __init__(self, function_index: dict[str, FunctionInfo]):
        self.function_index = function_index
        # Map: frozenset(field_names) -> SynthesizedBundle
        self.bundles: dict[frozenset[str], SynthesizedBundle] = defaultdict(
            lambda: SynthesizedBundle(defaultdict(set), [])
        )

    def ingest(self, groups_by_qual: dict[str, list[set[str]]]):
        """Aggregates bundles from the analysis phase."""
        for qual, groups in groups_by_qual.items():
            if not groups:
                continue
            info = self.function_index.get(qual)
            if not info:
                continue
            
            for group in groups:
                key = frozenset(group)
                bundle = self.bundles[key]
                bundle.occurrences.append(qual)
                
                # Harvest types
                for field in group:
                    # Check the annotation in the function definition
                    annot = info.annots.get(field)
                    if annot:
                        bundle.fields[field].add(annot)
                    else:
                        bundle.fields[field].add("Any")

    def _infer_name(self, fields: frozenset[str], occurrences: list[str]) -> str:
        """
        Derives a class name based on common substrings in function names 
        or the field names themselves.
        """
        # Strategy 1: Common suffix/prefix in function names
        # e.g. 'process_payment', 'validate_payment' -> 'Payment'
        func_names = [f.split('.')[-1] for f in occurrences]
        if not func_names:
            return "Context"

        # Split snake_case and look for common words
        words = []
        for name in func_names:
            words.extend(name.lower().split('_'))
        
        common = Counter(words).most_common(2)
        # If a word appears in majority of occurrences, use it
        meaningful_words = [w for w, count in common if count >= len(occurrences) * 0.5 and w not in ('get', 'set', 'do', 'run')]
        
        if meaningful_words:
            base = "".join(w.title() for w in meaningful_words)
            return f"{base}Bundle"

        # Strategy 2: Field name dominance
        # e.g. {user_id, account_id} -> UserAccountContext
        field_list = sorted(fields)
        # Try to find common prefix in fields
        if len(field_list) > 1:
            prefix = os.path.commonprefix(field_list)
            if len(prefix) > 3:
                clean_prefix = prefix.rstrip('_').title()
                return f"{clean_prefix}Context"
        
        # Fallback: Generic
        return f"AutoBundle_{abs(hash(fields)) % 10000:04d}"

    def _resolve_type(self, types: set[str]) -> str:
        """Resolves conflicting types into a Python type string."""
        types.discard("Any")
        types.discard(None)
        if not types:
            return "Any"
        if len(types) == 1:
            return next(iter(types))
        
        # Simplistic Union generation
        # Sorting ensures deterministic output
        return f"Union[{', '.join(sorted(types))}]"

    def generate_code(self) -> str:
        """Emits the Python source code."""
        lines = [
            "# Auto-generated by Dataflow Archeology",
            "from dataclasses import dataclass",
            "from typing import Any, Union, Optional",
            "",
        ]

        # Sort bundles by usage frequency (highest impact first)
        sorted_keys = sorted(
            self.bundles.keys(), 
            key=lambda k: len(self.bundles[k].occurrences), 
            reverse=True
        )

        for fields in sorted_keys:
            data = self.bundles[fields]
            # Heuristic: Only generate for bundles appearing in >1 place (Tier-2)
            # OR if it has >2 fields (high complexity)
            if len(data.occurrences) < 2 and len(fields) < 3:
                continue

            class_name = self._infer_name(fields, data.occurrences)
            
            lines.append(f"@dataclass(frozen=True)")
            lines.append(f"class {class_name}:")
            lines.append(f"    \"\"\"")
            lines.append(f"    Inferred from {len(data.occurrences)} call sites, including:")
            for occ in data.occurrences[:3]:
                lines.append(f"      - {occ}")
            if len(data.occurrences) > 3:
                lines.append(f"      - ... and {len(data.occurrences) - 3} others.")
            lines.append(f"    \"\"\"")

            sorted_fields = sorted(fields)
            for field in sorted_fields:
                type_str = self._resolve_type(data.fields[field])
                lines.append(f"    {field}: {type_str}")
            lines.append("")
        
        return "\n".join(lines)

```

### Integration Plan

You can now weave this synthesizer into your existing `main` function to close the loop.

1. **Add arguments:** Add a `--synthesize` argument to the CLI.
2. **Hook logic:** After `analyze_repo` returns, pass the results to the synthesizer.

```python
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--report", default=None, help="Write markdown report to file.")
    parser.add_argument("--synthesize", default=None, help="Write suggested Python protocols to file.")
    args = parser.parse_args()
    
    paths = _iter_paths(args.paths)
    
    # 1. THE DOER (Analysis)
    groups, symbols, index = analyze_repo(paths)
    
    # 2. THE WITNESS (Reporting)
    if args.report:
        report = _emit_report(groups, symbols, index, 10)
        Path(args.report).write_text(report)
        print(f"Report written to {args.report}")
    
    # 3. THE CREATOR (Synthesis)
    if args.synthesize:
        print(f"Synthesizing protocols to {args.synthesize}...")
        synth = ProtocolSynthesizer(index)
        synth.ingest(groups)
        code = synth.generate_code()
        Path(args.synthesize).write_text(code)
        print("Done.")

    if not args.report and not args.synthesize:
        # Default dump to stdout
        for qual, bun in sorted(groups.items()):
            if not bun: continue
            print(f"{qual}:")
            for b in bun:
                print(f"  {sorted(b)}")

```

### Why This Is The Greatest Leverage

1. **Activation Energy:** It is significantly easier to edit a generated file than to write one from scratch. You shift the work from "Authoring" to "Reviewing."
2. **Standardization:** The synthesizer forces a standard naming convention and structure across the repo instantly.
3. **Discovery:** You will see "concepts" you didn't know existed. If the tool generates `class AutoBundle_1234` with fields `start_date`, `end_date`, `timezone`, `locale`, you instantly recognize a `UserSessionContext` that was implicit in your code.

This completes the cycle: **Observe (Analysis) -> Orient (Report) -> Decide (Synthesize) -> Act (Refactor).**
