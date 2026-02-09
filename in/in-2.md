---
doc_revision: 1
reader_reintern: "Reader-only: re-intern if doc_revision changed since you last read this doc."
doc_id: in_2
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
This is a highly specialized static analysis tool that sits somewhere between a linter and a compiler optimizer. It attempts to reverse-engineer architectural intent (specifically, data grouping) from implementation details.

Here is the evaluation based on your requested dimensions:

### 1. What is Novel?

* **The "Dataflow Grammar" Abstraction:** Most static analysis tools look for errors (bugs) or style violations (pep8). This tool looks for *latent structure*. Treating parameter bundles as a "grammar" that flows through the system is a strong conceptual leap. It implies that data clumps aren't just a smell, but a hidden type system waiting to be reified.
* **Fixed-Point Bundle Propagation (`_propagate_groups`):** The implementation doesn't just look at one function; it iteratively propagates bundles up the call stack until a fixed point is reached. If `C` takes `(x, y)`, and `B` calls `C(x, y)`, and `A` calls `B(x, y)`, the tool correctly identifies that `A` is implicitly dealing with the `{x, y}` bundle, even if `A` never uses them directly.
* **"Refactoring-First" Reporting:** The output isn't a list of errors; it’s a strategic roadmap. By generating Mermaid diagrams of "Components" (clusters of functions sharing data bundles), it visualizes the refactoring blast radius. This is rare in standard tooling.
* **Tiered Confidence Levels:** Distinguishing between **Tier 1** (declared in Config), **Tier 2** (observed multiple times), and **Tier 3** (observed once) is a pragmatic feature that prioritizes developer attention effectively.

### 2. What is Not Novel?

* **AST-Based Call Graph Heuristics:** The method of resolving callees (`_resolve_callee`)—matching by name, then by module suffix, then by uniqueness—is a standard (and somewhat fragile) workaround in dynamic languages lacking full symbol tables.
* **"Data Clump" Detection:** The core concept is the detection of "Data Clumps" (a Martin Fowler refactoring pattern). Tools like JDeodorant (for Java) have done this for years.
* **Constant Propagation (`analyze_constant_flow_repo`):** Finding parameters that are always constant is a standard optimization pass in compilers, though using it to detect "dead configuration knobs" in Python is a useful application of the concept.

### 3. What is Well-Supported?

* **Local Parameter Usage Analysis (`_analyze_function`):** The logic for determining if a variable is "passed through" vs. "used" is strict and conservative. It correctly handles positional and keyword mapping, and it disqualifies variables that are modified or used in logic (`info.non_forward`). This ensures false positives are minimized—if the tool says it's a bundle, it’s likely a true passthrough.
* **Graph Theory Implementation:** The connected components logic (`_connected_components`) and the graph visualization (`_emit_dot`, `_render_mermaid_component`) are mathematically sound and correctly implemented.
* **Component Isolation:** The script effectively isolates independent clusters of dataflow. This prevents the analysis from becoming a "giant blob" of dependencies, which is crucial for making the report actionable in a large codebase.

### 4. What is Not Well-Supported?

* **Import Aliasing and Re-exports:** The `_resolve_callee` function relies heavily on string matching of names. It appears to lack a true symbol resolution engine.
* *Failure Mode:* If `from utils import runner as run_job` is used, and the call is `run_job(...)`, the tool likely won't link it back to `utils.runner` unless the name is unique across the entire repo.


* **Complex Control Flow & Variable Aliasing:** The tool tracks `Name` nodes directly.
* *Failure Mode:*
```python
def wrapper(a, b):
    args = (a, b)
    target(*args) # The link is broken here

```


The tool cannot trace the bundle through intermediate variable assignments or unpacking.


* **Polymorphism/Inheritance:** It treats `self.foo` by looking for `foo` in the same file. It does not understand class hierarchies. If `self.foo` is defined in a parent class in a different file, the link is severed.
* **Dynamic Dispatch:** Any use of `getattr`, decorators that change signatures, or `*args` / `**kwargs` forwarding without explicit naming will result in "invisible" edges in the graph.

### 5. What is Valuable?

* **Architecture Mining in Legacy Code:** For a large, untyped Python codebase (the "Big Ball of Mud"), this tool is incredibly high-value. It automates the discovery of "Context Objects" or "Config Objects" that *should* exist but don't.
* **Type Tightening (`analyze_type_flow_repo`):** The ability to say "Function A takes `Any`, but every function it calls expects `int`, so Function A actually takes `int`" is a massive win for gradual typing adoption.
* **Dead Code Detection:** The "Constant Smell" detector is a subtle but powerful way to find features that were built but never actually configured dynamically.

### Summary

This is a **strategic refactoring tool**. It trades perfect accuracy (symbol resolution) for high-level insight (dataflow clustering).

* **Use it to:** Plan a migration to `dataclasses` or configuration objects in a messy repo.
* **Do not use it to:** Perform safety-critical analysis or automated refactoring without human review, as the resolution heuristics will miss edge cases.
