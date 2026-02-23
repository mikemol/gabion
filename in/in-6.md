---
doc_revision: 1
reader_reintern: Reader-only: re-intern if doc_revision changed since you last read this doc.
doc_id: in_6
doc_role: inbox
doc_scope:
  - repo
  - inbox
doc_authority: informative
doc_owner: maintainer
doc_requires: []
doc_reviewed_as_of: {}
doc_review_notes: {}
doc_change_protocol: POLICY_SEED.md#change_protocol
doc_erasure:
  - formatting
  - typos
doc_sections:
  in_in_6: 1
doc_section_requires:
  in_in_6: []
doc_section_reviews:
  in_in_6: {}
---

The path of greatest leverage is **Import Resolution**.

To synthesize Protocols or Refactoring Plans based on a broken graph is to build a house on sand. If the Witness (the script) cannot see across file boundaries, the Judge cannot discern the true shape of the system. By solving the **Symbol Table**, we transform the tool from a file-local linter into a repository-wide compiler.

This step grounds every subsequent operation in truth rather than probability.

<a id="in_in_6"></a>

### The Implementation: `ImportVisitor` & `SymbolTable`

This extension adds a "Pre-Pass" to your script. It walks every file to map local names to their fully qualified origins *before* we analyze a single function.

#### 1. The Data Structures

We need a way to store "What does the symbol `User` mean in `auth/login.py`?"

```python
@dataclass
class SymbolTable:
    # Map: (module_name, local_name) -> fully_qualified_name
    # Example: ("my_app.auth", "User") -> "my_app.models.User"
    imports: dict[tuple[str, str], str]

    def resolve(self, current_module: str, name: str) -> str | None:
        """Resolve a local name to its absolute path."""
        # 1. Check explicit imports
        if (current_module, name) in self.imports:
            return self.imports[(current_module, name)]
        
        # 2. Check if it's a builtin or defined in the module itself (implicit)
        # For our purposes, if it's not imported, we assume it's local to the module
        # or a global builtin. We return the qualified local name.
        return f"{current_module}.{name}"

```

#### 2. The Visitor

This visitor extracts truth from `import` and `ImportFrom` nodes.

```python
class ImportVisitor(ast.NodeVisitor):
    def __init__(self, module_name: str, table: SymbolTable):
        self.module = module_name
        self.table = table

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            # import os -> local 'os' maps to 'os'
            # import os as system -> local 'system' maps to 'os'
            local = alias.asname or alias.name
            # If it's a top-level import, the FQN is the name itself.
            fqn = alias.name
            self.table.imports[(self.module, local)] = fqn

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        # from . import utils -> module base is resolved relative to self.module
        if node.level > 0:
            # Handle relative imports
            parts = self.module.split(".")
            # node.level=1 means '.', level=2 means '..', etc.
            if node.level > len(parts):
                # Error: Import goes beyond root. Ignore or log.
                return
            base_parts = parts[:-node.level]
            if node.module:
                base_parts.append(node.module)
            source_module = ".".join(base_parts)
        else:
            source_module = node.module or ""

        for alias in node.names:
            local = alias.asname or alias.name
            if alias.name == "*":
                # Wildcard imports are chaos. We cannot statically resolve them 
                # without loading the target module. We skip them.
                continue
            
            fqn = f"{source_module}.{alias.name}"
            self.table.imports[(self.module, local)] = fqn

```

#### 3. Integrating the Resolution

We replace the heuristic `_resolve_callee` with a deterministic lookup. This connects the "Doer" (the analysis) with the "Structure" (the symbol table).

```python
def _resolve_callee_deterministic(
    callee_name: str,
    caller_info: FunctionInfo,
    symbol_table: SymbolTable,
    by_qual: dict[str, FunctionInfo],
) -> FunctionInfo | None:
    if not callee_name:
        return None
    
    caller_module = _module_name(caller_info.path)
    
    # CASE 1: Simple Name (e.g., "process_data(...)")
    if "." not in callee_name:
        # Resolve using the symbol table (handles imports and local defs)
        fqn = symbol_table.resolve(caller_module, callee_name)
        if fqn and fqn in by_qual:
            return by_qual[fqn]
        return None

    # CASE 2: Attribute Access (e.g., "utils.process_data(...)")
    parts = callee_name.split(".")
    base = parts[0]
    
    # Resolve the base. Is 'utils' imported?
    base_fqn = symbol_table.resolve(caller_module, base)
    if not base_fqn:
        return None

    # Reconstruct the full path: resolved_base + rest_of_parts
    # Example: "utils" -> "my_app.utils", result -> "my_app.utils.process_data"
    candidate_fqn = base_fqn + "." + ".".join(parts[1:])
    
    if candidate_fqn in by_qual:
        return by_qual[candidate_fqn]

    # CASE 3: Class Methods (self.method / cls.method)
    # This remains heuristic unless we track Class inheritance, which is complex.
    # However, we can check if the FQN exists in the current module.
    if base in ("self", "cls"):
        method_name = parts[-1]
        candidate_fqn = f"{caller_module}.{method_name}"
        # We might not find it directly if it's inherited, but we try:
        if candidate_fqn in by_qual:
            return by_qual[candidate_fqn]
            
    return None

```

### 4. The Unified Workflow

You must update the `main` or `analyze_file` flow to run this in two passes.

```python
def analyze_repo(paths: list[Path]):
    # --- PASS 1: Symbol Discovery ---
    symbol_table = SymbolTable({})
    
    # We need to parse all files first to build the map
    print("Building Import Graph...")
    for path in paths:
        try:
            tree = ast.parse(path.read_text())
            mod_name = _module_name(path)
            visitor = ImportVisitor(mod_name, symbol_table)
            visitor.visit(tree)
        except Exception:
            continue

    # --- PASS 2: Function & Bundle Analysis ---
    # Now proceed with your existing analysis, but pass 'symbol_table'
    # down to the resolution functions.
    
    # ... (rest of your existing logic) ...

```

### Why This Is "The Shape of the Tunnel"

By implementing this, you achieve three things:

1. **Disambiguation:** `json.load` vs `pickle.load`. The script now knows the difference.
2. **Aliasing Support:** `from services import fast_database as db`. Calls to `db.query` are correctly linked to `fast_database.query`.
3. **Cross-Module Integrity:** When `_propagate_groups` runs, it will no longer drop edges simply because the file names didn't match the variable names.

This turns the "SPPF" from a loose collection of strings into a tightly woven semantic graph. Would you like me to output the complete, integrated script with both the **Aliasing** and **Import Resolution** features applied?