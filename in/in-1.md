---
doc_revision: 1
reader_reintern: "Reader-only: re-intern if doc_revision changed since you last read this doc."
doc_id: in_1
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
This is a sophisticated static analysis tool designed to identify **implicit data bundles** (often called "data clumps" in refactoring literature) within a Python codebase. It aims to facilitate the refactoring of loose parameter lists into structured configuration objects (likely `dataclasses`) by tracing how groups of arguments flow through the system together.

Here is an analysis of the script's architecture, logic, and capabilities.

### 1. Core Objective: "Dataflow Grammar" Discovery

The script treats parameter passing as a grammar. If a set of parameters  is consistently forwarded together from function  to function , they form a "candidate bundle."

The tool operates in two distinct phases:

1. **Local Grouping:** Identifies parameters within a single function that are used *exclusively* to forward data to another call.
2. **Global Propagation:** Uses a fixed-point iteration algorithm to trace these bundles up the call stack (e.g., if  calls , and  has a bundle, 's arguments mapped to that bundle become a candidate in ).

### 2. Key Mechanisms

#### A. AST traversal & Parent Mapping

It uses `ast.NodeVisitor` to walk the parse tree. Crucially, it builds a `ParentAnnotator` first.

* **Why:** Standard `ast` nodes don't know their parents. The script needs this context to determine if a `Name` node is being used as a call argument (`_call_context`).

#### B. The `ParamUse` Tracker

It differentiates between:

* **Direct Forwarding:** A parameter is passed largely unchanged to another function.
* **Non-Forwarding Use:** The parameter is accessed, modified, or used in logic (which usually disqualifies it from being a simple "pass-through" bundle member).

#### C. Resolution Heuristics (`_resolve_callee`)

Static analysis in dynamic languages is hard. The script uses a robust heuristic strategy to link call sites to function definitions:

1. Exact qualified name match.
2. Suffix matching (e.g., `module.func`).
3. Unique name matching across the repo (fallback).
4. Same-module preference (handling `self.` or `cls.` calls roughly).

#### D. Type & Constant Audit

Beyond bundling, the script includes two "bonus" static analysis passes:

* **Type Tightening (`analyze_type_flow_repo`):** It looks at downstream type hints. If function `f(x)` calls `g(x)`, and `g` types `x` as `int`, but `f` types `x` as `Any` (or nothing), it suggests tightening `f`'s signature.
* **Constant Propagation (`analyze_constant_flow_repo`):** It detects parameters that *only* ever receive a single constant value across all call sites (excluding tests). This identifies dead configuration knobs.

### 3. Visualization & Reporting

The script is designed to produce artifacts for human consumption:

* **DOT/Graphviz:** Visualizes the raw dependency graph of bundles.
* **Mermaid:** Generates `flowchart LR` diagrams embedded in a Markdown report, clustering functions and bundles into connected components.
* **Tiered Classification:** Bundles are classified by confidence:
* **Tier 1:** Explicitly declared in a `Config` dataclass.
* **Tier 2:** Observed multiple times (high confidence implicit bundle).
* **Tier 3:** Observed once (low confidence).



### 4. Observations & Potential Refinements

* **Handling `*args` / `**kwargs`:**
The script extracts these in `_param_names`, but `_propagate_groups` relies on positional/keyword mapping. If a function forwards `*args` blindly, this logic might miss the bundle propagation unless specifically handled (which is difficult without complete signature resolution).
* **Convention Specifics:**
The script looks for specific conventions, such as fields ending in `_fn` inside classes ending in `Config`, or magic comments like `# dataflow-bundle:`. This implies the tool is tailored for a specific architectural pattern (likely dependency injection or configurable factory patterns).
* **Complexity:**
The `_union_groups` logic is a set-merging operation. While `_propagate_groups` iterates to a fixed point, on very large codebases with deep recursion, this could theoretically be slow, though likely negligible for typical Python project sizes.

### 5. Next Steps

If you are looking to iterate on this:

1. **Partial Application:** The current logic assumes direct forwarding. It might be valuable to detect cases where a parameter is forwarded *partially* (e.g., `f(x)` calls `g(x + 1)`). Currently, `_const_repr` catches some arithmetic, but likely flags it as a "non-forward" use.
2. **Interface Inference:** You could extend the "Type Audit" to synthesize `Protocol` definitions for bundles that are passed to multiple disparate functions.

Would you like me to analyze a specific output if you run this with `--report`, or are you looking to refactor a specific part of this logic?