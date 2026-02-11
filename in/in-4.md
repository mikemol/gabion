---
doc_revision: 1
reader_reintern: Reader-only: re-intern if doc_revision changed since you last read this doc.
doc_id: in_4
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
  in_in_4: 1
doc_section_requires:
  in_in_4: []
doc_section_reviews:
  in_in_4: {}
---

This is the correct path. To refuse to see `y` as `x` when `y = x` is to deny the transitivity of identity, which creates a "gap" in your structural lattice. If the Witness (the script) blinks when the data changes names, the Judge cannot accurately sentence the code.

However, we must apply the **Strict Necessity Test** to this extension. We are not just "adding features"; we are **expanding the definition of a valid morphism**.

<a id="in_in_4"></a>

### The Conceptual Shift: Identity vs. Symbol

Currently, your script equates **Identity** (the bundle) with **Symbol** (the parameter name).

* *Current Axiom:* `Identity(p) <=> Symbol(p)`
* *Corrected Axiom:* `Identity(p)` is a set of symbols `{s_0, s_1, ...}` such that  and is immutable.

To implement this with "meticulousness," we must track the **Chain of Custody**.

### The Plan

I will rewrite the `_analyze_function` logic (and its dependencies) to introduce a **Symbol Table for Identity Preservation**.

1. **Transitive Aliasing:** If `local = param`, then `local` inherits `param`'s bundle identity. If `other = local`, `other` inherits it too.
2. **Assignment as Preservation:** The act of assignment `y = x` is no longer a "use" (which usually triggers `non_forward`); it is now a "preservation" event, *provided* `x` is the sole value being assigned.
3. **Taint Analysis (The Knot):** If a parameter is reassigned (`param = new_value`), its identity is severed. We must stop tracking it to avoid "Safety" failures.

### The Refined Code

Replace the `ParamUse`, `CallArgs`, and `_analyze_function` sections of your script with the following "Identity-Aware" implementation.

```python
@dataclass
class ParamUse:
    """Tracks how a parameter (or its aliases) flows through the function."""
    direct_forward: set[tuple[str, str]]
    non_forward: bool
    # Track which local names currently hold this parameter's identity
    current_aliases: set[str]


def _analyze_function(fn, parents, *, is_test: bool):
    params = _param_names(fn)
    # Initialize usage map.
    # We add 'aliases' to track local variables that are effectively the parameter.
    use_map: dict[str, ParamUse] = {
        p: ParamUse(set(), False, {p}) for p in params
    }
    
    # Reverse lookup: distinct local alias -> original parameter name
    # Invariant: A local variable maps to AT MOST one parameter.
    alias_to_param: dict[str, str] = {p: p for p in params}

    call_args: list[CallArgs] = []

    class UseVisitor(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:
            callee = _callee_name(node)
            pos_map = {}
            kw_map = {}
            const_pos: dict[str, str] = {}
            const_kw: dict[str, str] = {}
            non_const_pos: set[str] = set()
            non_const_kw: set[str] = set()

            for idx, arg in enumerate(node.args):
                const = _const_repr(arg)
                if const is not None:
                    const_pos[str(idx)] = const
                    continue
                
                # Check if the argument is a known alias to a parameter
                if isinstance(arg, ast.Name) and arg.id in alias_to_param:
                    # It's a forward! Map it back to the original parameter.
                    origin_param = alias_to_param[arg.id]
                    pos_map[str(idx)] = origin_param
                else:
                    non_const_pos.add(str(idx))

            for kw in node.keywords:
                if kw.arg is None:
                    # TODO: Handle **kwargs forwarding if valuable later.
                    continue
                
                const = _const_repr(kw.value)
                if const is not None:
                    const_kw[kw.arg] = const
                    continue

                # Check if keyword value is a known alias
                if isinstance(kw.value, ast.Name) and kw.value.id in alias_to_param:
                    origin_param = alias_to_param[kw.value.id]
                    kw_map[kw.arg] = origin_param
                else:
                    non_const_kw.add(kw.arg)

            call_args.append(
                CallArgs(
                    callee=callee,
                    pos_map=pos_map,
                    kw_map=kw_map,
                    const_pos=const_pos,
                    const_kw=const_kw,
                    non_const_pos=non_const_pos,
                    non_const_kw=non_const_kw,
                    is_test=is_test,
                )
            )
            # Continue visiting children (e.g. nested calls)
            self.generic_visit(node)

        def visit_Assign(self, node: ast.Assign) -> None:
            # Handle aliasing: y = x
            # We only support simple assignment: single target, Name value.
            
            # 1. Analyze the value being assigned (RHS)
            rhs_param = None
            if isinstance(node.value, ast.Name) and node.value.id in alias_to_param:
                rhs_param = alias_to_param[node.value.id]

            # 2. Process targets (LHS)
            for target in node.targets:
                if not isinstance(target, ast.Name):
                    # Complex assignment (e.g. unpacking, attribute set).
                    # If we are writing TO a parameter, we must sever its identity.
                    self._check_write(target)
                    continue

                lhs_name = target.id

                if rhs_param:
                    # Case: y = x (where x is param or alias).
                    # Grant 'y' the identity of 'x'.
                    alias_to_param[lhs_name] = rhs_param
                    use_map[rhs_param].current_aliases.add(lhs_name)
                else:
                    # Case: y = <something else>.
                    # If 'y' was previously an alias, it is no longer.
                    if lhs_name in alias_to_param:
                        old_param = alias_to_param.pop(lhs_name)
                        if old_param in use_map:
                            use_map[old_param].current_aliases.discard(lhs_name)

            # Visit children (RHS might be a Call we need to track)
            self.visit(node.value)
            # Do NOT visit targets in Load context to avoid flagging them as 'uses'
            
        def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
            # Handle annotated assignment: y: int = x
            if not isinstance(node.target, ast.Name):
                if node.value: self.visit(node.value)
                return

            lhs_name = node.target.id
            rhs_param = None
            
            if node.value and isinstance(node.value, ast.Name) and node.value.id in alias_to_param:
                rhs_param = alias_to_param[node.value.id]
            
            if rhs_param:
                alias_to_param[lhs_name] = rhs_param
                use_map[rhs_param].current_aliases.add(lhs_name)
            else:
                if lhs_name in alias_to_param:
                    old_param = alias_to_param.pop(lhs_name)
                    if old_param in use_map:
                        use_map[old_param].current_aliases.discard(lhs_name)

            if node.value:
                self.visit(node.value)

        def _check_write(self, node: ast.AST):
            """If a node is being written to, and it's a parameter, it's tainted."""
            if isinstance(node, ast.Name) and node.id in alias_to_param:
                # We are writing to a variable that holds a parameter identity.
                # Is it the parameter itself?
                param = alias_to_param[node.id]
                if node.id == param:
                     # Reassigning the parameter itself! 'x = 5'.
                     # Mark as non-forward because it's modified.
                     use_map[param].non_forward = True
                     # It loses its own identity.
                     alias_to_param.pop(param, None)
                else:
                    # Reassigning an alias: 'y = x; y = 5'. 
                    # Just remove 'y' from aliases. 'x' is safe.
                    alias_to_param.pop(node.id, None)
                    use_map[param].current_aliases.discard(node.id)

        def visit_Name(self, node: ast.Name) -> None:
            if not isinstance(node.ctx, ast.Load):
                return
            
            # If this name isn't tracking a parameter, we don't care.
            if node.id not in alias_to_param:
                return

            origin_param = alias_to_param[node.id]

            # Context check:
            # 1. Is it a Call argument? (Handled in visit_Call, usually)
            #    Wait, visit_Call visits children. So we will end up here.
            #    We need to check if the parent is the Call we just processed.
            # 2. Is it the RHS of an assignment? (Handled in visit_Assign)
            
            call, is_arg = _call_context(node, parents)
            if call:
                if is_arg:
                    # It is an argument. 
                    # We record the forwarding bundle entry here.
                    callee = _callee_name(call)
                    
                    # Determine slot (similar to old logic, but aware of alias)
                    slot = None
                    for idx, arg in enumerate(call.args):
                        if arg is node:
                            slot = f"arg[{idx}]"
                            break
                    if slot is None:
                        for kw in call.keywords:
                            if kw.value is node:
                                slot = f"kw[{kw.arg}]"
                                break
                    
                    if slot:
                        use_map[origin_param].direct_forward.add((callee, slot))
                    else:
                         # Argument in a weird spot (e.g. *args), count as non-forward for safety
                         use_map[origin_param].non_forward = True
                    return
                else:
                    # It's inside a call but NOT as an argument (e.g. func(x + 1))
                    # Wait, _call_context handles this. 
                    # If is_arg is False, it's part of an expression inside the call.
                    use_map[origin_param].non_forward = True
                    return

            # If we are here, we are in a Load context, NOT in a call.
            # We must check if we are the RHS of an assignment.
            parent = parents.get(node)
            if isinstance(parent, (ast.Assign, ast.AnnAssign)):
                # If we are the value, this is an Identity Transfer, not a 'use'.
                if parent.value is node:
                    return

            # If here, it's a random read (e.g. 'if x:', 'y = x + 1').
            # This counts as usage (non-forward).
            use_map[origin_param].non_forward = True

    UseVisitor().visit(fn)
    return use_map, call_args

```

### Analysis of the Interplay

1. **Completeness vs. Correctness:**
* *Previously:* Correct but incomplete. If `y=x` occurred, `x` was dead.
* *Now:* We accept the alias.
* *Risk:* The logic inside `visit_Name` is complex. Specifically, `visit_Call` calls `generic_visit`, which triggers `visit_Name`.
* *Resolution:* In `visit_Name`, we check `_call_context`. If it confirms we are an argument, we log the forward. This is redundant with the pre-scan in `visit_Call` in the previous implementation, but necessary here because `visit_Call` doesn't know if `arg` is `node`.
* *Refinement:* The provided code splits the logic. `visit_Call` handles the `CallArgs` struct construction (for global propagation), while `visit_Name` handles the `ParamUse` struct (for local bundle detection). This separation of concerns is cleaner.


2. **Safety (The Knot):**
* The `_check_write` logic acts as the "Knot." If the variable is mutated (reassigned), the knot slips, and the truth (the bundle identity) is no longer held. The system "cools" the thought by discarding the alias immediately.


3. **Structure (The Lattice):**
* This change effectively allows the "Dataflow Grammar" to handle **Rename Morphisms**. A bundle `{a, b}` passed as `func(x=a, y=b)` is structurally identical to `a2=a; b2=b; func(x=a2, y=b2)`.



### Next Action

To fully integrate this, you would replace the `_analyze_function` in your script with the version above. You would also need to ensure `ParamUse` is updated to include the `current_aliases` field (as shown in the dataclass definition above).

This change moves the tool from "Pattern Matching" to "Flow Analysis." It is a higher order of operational truth.