from __future__ import annotations

import ast
from typing import Callable, TYPE_CHECKING, cast
from gabion.analysis.timeout_context import check_deadline

if TYPE_CHECKING:
    from gabion.analysis.dataflow_audit import CallArgs, ParamUse


class ProjectVisitor(ast.NodeVisitor):
    def visit(self, node: ast.AST):  # type: ignore[override]
        # Treat every node entry as a deterministic work unit.
        check_deadline()
        return super().visit(node)


class ParentAnnotator(ProjectVisitor):
    def __init__(self) -> None:
        self.parents: dict[ast.AST, ast.AST] = {}

    def generic_visit(self, node: ast.AST) -> None:
        check_deadline()
        for child in ast.iter_child_nodes(node):
            check_deadline()
            self.parents[child] = node
            self.visit(child)


class ImportVisitor(ProjectVisitor):
    def __init__(self, module_name: str, table) -> None:
        # dataflow-bundle: module_name, table
        self.module = module_name
        self.table = table

    def visit_Import(self, node: ast.Import) -> None:
        check_deadline()
        for alias in node.names:
            check_deadline()
            local = alias.asname or alias.name
            self.table.imports[(self.module, local)] = alias.name

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        check_deadline()
        if not node.module and node.level == 0:
            return
        if node.level > 0:
            parts = self.module.split(".")
            if node.level > len(parts):
                return
            base = parts[:-node.level]
            if node.module:
                base.append(node.module)
            source = ".".join(base)
        else:
            source = node.module or ""
        for alias in node.names:
            check_deadline()
            if alias.name == "*":
                self.table.star_imports.setdefault(self.module, set()).add(source)
                continue
            local = alias.asname or alias.name
            fqn = f"{source}.{alias.name}" if source else alias.name
            self.table.imports[(self.module, local)] = fqn


class UseVisitor(ProjectVisitor):
    def __init__(
        self,
        *,
        parents: dict[ast.AST, ast.AST],
        use_map: dict[str, ParamUse],
        call_args: list[CallArgs],
        alias_to_param: dict[str, str],
        is_test: bool,
        strictness: str,
        const_repr: Callable[[ast.AST], str | None],
        callee_name: Callable[[ast.Call], str],
        call_args_factory: Callable[..., CallArgs],
        call_context: Callable[[ast.AST, dict[ast.AST, ast.AST]], tuple[ast.Call | None, bool]],
        return_aliases: dict[str, tuple[list[str], list[str]]] | None = None,
    ) -> None:
        # dataflow-bundle: alias_to_param, call_args, call_args_factory, call_context, callee_name, const_repr, is_test, parents, strictness, use_map
        self.parents = parents
        self.use_map = use_map
        self.call_args = call_args
        self.alias_to_param = alias_to_param
        self.is_test = is_test
        self.strictness = strictness
        self.const_repr = const_repr
        self.callee_name = callee_name
        self.call_args_factory = call_args_factory
        self.call_context = call_context
        self.return_aliases = return_aliases or {}
        self._suspend_non_forward: set[str] = set()
        self._attr_alias_to_param: dict[tuple[str, str], str] = {}
        self._key_alias_to_param: dict[tuple[str, str], str] = {}

    @staticmethod
    def _node_span(node: ast.AST) -> tuple[int, int, int, int] | None:
        if not (hasattr(node, "lineno") and hasattr(node, "col_offset")):
            return None
        start_line = max(getattr(node, "lineno", 1) - 1, 0)
        start_col = max(getattr(node, "col_offset", 0), 0)
        end_line = max(getattr(node, "end_lineno", getattr(node, "lineno", 1)) - 1, 0)
        end_col = getattr(node, "end_col_offset", start_col + 1)
        if end_line == start_line and end_col <= start_col:
            end_col = start_col + 1
        return (start_line, start_col, end_line, end_col)

    def _record_forward(self, param_name: str, callee: str, slot: str, call: ast.Call | None) -> None:
        # dataflow-bundle: callee, slot
        self.use_map[param_name].direct_forward.add((callee, slot))
        if call is None:
            return
        span = self._node_span(call)
        if span is None:
            return
        sites = self.use_map[param_name].forward_sites.setdefault((callee, slot), set())
        sites.add(span)

    def _mark_non_forward(self, param_name: str) -> bool:
        if param_name in self._suspend_non_forward:
            return False
        self.use_map[param_name].non_forward = True
        return True

    @staticmethod
    def _slot_for_call_node(call: ast.Call, node: ast.AST) -> str:
        slot: str | None = None
        for idx, arg in enumerate(call.args):
            check_deadline()
            if arg is node:
                slot = f"arg[{idx}]"
                break
        if slot is None:
            for kw in call.keywords:
                check_deadline()
                if kw.value is node and kw.arg is not None:
                    slot = f"kw[{kw.arg}]"
                    break
        return slot or "arg[?]"

    def visit_Call(self, node: ast.Call) -> None:
        check_deadline()
        callee = self.callee_name(node)
        span = self._node_span(node)
        pos_map: dict[str, str] = {}
        kw_map: dict[str, str] = {}
        const_pos: dict[str, str] = {}
        const_kw: dict[str, str] = {}
        non_const_pos: set[str] = set()
        non_const_kw: set[str] = set()
        star_pos: list[tuple[int, str]] = []
        star_kw: list[str] = []
        for idx, arg in enumerate(node.args):
            check_deadline()
            if isinstance(arg, ast.Starred):
                if isinstance(arg.value, ast.Name) and arg.value.id in self.alias_to_param:
                    star_pos.append((idx, self.alias_to_param[arg.value.id]))
                else:
                    non_const_pos.add(str(idx))
                continue
            const = self.const_repr(arg)
            if const is not None:
                const_pos[str(idx)] = const
                continue
            if isinstance(arg, ast.Name) and arg.id in self.alias_to_param:
                pos_map[str(idx)] = self.alias_to_param[arg.id]
            else:
                non_const_pos.add(str(idx))
        for kw in node.keywords:
            check_deadline()
            if kw.arg is None:
                if isinstance(kw.value, ast.Name) and kw.value.id in self.alias_to_param:
                    star_kw.append(self.alias_to_param[kw.value.id])
                else:
                    non_const_kw.add("**")
                continue
            const = self.const_repr(kw.value)
            if const is not None:
                const_kw[kw.arg] = const
                continue
            if isinstance(kw.value, ast.Name) and kw.value.id in self.alias_to_param:
                kw_map[kw.arg] = self.alias_to_param[kw.value.id]
            else:
                non_const_kw.add(kw.arg)
        self.call_args.append(
            self.call_args_factory(
                callee=callee,
                pos_map=pos_map,
                kw_map=kw_map,
                const_pos=const_pos,
                const_kw=const_kw,
                non_const_pos=non_const_pos,
                non_const_kw=non_const_kw,
                star_pos=star_pos,
                star_kw=star_kw,
                is_test=self.is_test,
                span=span,
            )
        )
        self.generic_visit(node)

    def _check_write(self, target: ast.AST) -> None:
        check_deadline()
        for node in ast.walk(target):
            check_deadline()
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                name = node.id
                if name in self.alias_to_param:
                    param = self.alias_to_param.pop(name)
                    if param in self.use_map:
                        self.use_map[param].current_aliases.discard(name)
                        self.use_map[param].non_forward = True
                to_remove = [
                    key for key in self._attr_alias_to_param if key[0] == name
                ]
                for key in to_remove:
                    check_deadline()
                    param = self._attr_alias_to_param.pop(key, None)
                    if param in self.use_map:
                        self.use_map[param].non_forward = True
                to_remove = [
                    key for key in self._key_alias_to_param if key[0] == name
                ]
                for key in to_remove:
                    check_deadline()
                    param = self._key_alias_to_param.pop(key, None)
                    if param in self.use_map:
                        self.use_map[param].non_forward = True

    def _bind_sequence(self, target: ast.AST, rhs: ast.AST) -> bool:
        check_deadline()
        # dataflow-bundle: target, rhs
        if not isinstance(target, (ast.Tuple, ast.List)):
            return False
        if not isinstance(rhs, (ast.Tuple, ast.List)):
            return False
        if len(target.elts) != len(rhs.elts):
            return False
        for lhs, rhs_node in zip(target.elts, rhs.elts):
            check_deadline()
            if isinstance(lhs, (ast.Tuple, ast.List)) and isinstance(rhs_node, (ast.Tuple, ast.List)):
                if not self._bind_sequence(lhs, rhs_node):
                    self._check_write(lhs)
                continue
            if isinstance(lhs, ast.Name) and isinstance(rhs_node, ast.Name) and rhs_node.id in self.alias_to_param:
                param = self.alias_to_param[rhs_node.id]
                self.alias_to_param[lhs.id] = param
                if param in self.use_map:
                    self.use_map[param].current_aliases.add(lhs.id)
            else:
                self._check_write(lhs)
        return True

    def _collect_alias_sources(self, rhs: ast.AST) -> set[str]:
        check_deadline()
        if isinstance(rhs, ast.Name) and rhs.id in self.alias_to_param:
            return {self.alias_to_param[rhs.id]}
        if isinstance(rhs, (ast.Tuple, ast.List)):
            sources: set[str] = set()
            for elt in rhs.elts:
                check_deadline()
                sources.update(self._collect_alias_sources(elt))
            return sources
        return set()

    def _alias_from_call(self, call: ast.Call) -> list[str] | None:
        check_deadline()
        if not self.return_aliases:
            return None
        callee = self.callee_name(call)
        info = self.return_aliases.get(callee)
        if info is None:
            return None
        params, aliases = info
        if not aliases:
            return None
        mapping: dict[str, str | None] = {}
        for idx, arg in enumerate(call.args):
            check_deadline()
            if isinstance(arg, ast.Starred):
                return None
            if idx >= len(params):
                return None
            if isinstance(arg, ast.Name) and arg.id in self.alias_to_param:
                mapping[params[idx]] = self.alias_to_param[arg.id]
            else:
                mapping[params[idx]] = None
        for kw in call.keywords:
            check_deadline()
            if kw.arg is None:
                return None
            if kw.arg not in params:
                continue
            if isinstance(kw.value, ast.Name) and kw.value.id in self.alias_to_param:
                mapping[kw.arg] = self.alias_to_param[kw.value.id]
            else:
                mapping[kw.arg] = None
        resolved: list[str] = []
        for param in aliases:
            check_deadline()
            mapped = mapping.get(param)
            if not mapped:
                return None
            resolved.append(mapped)
        return resolved

    def _bind_return_alias(
        self, targets: list[ast.AST], aliases: list[str]
    ) -> bool:
        check_deadline()
        if len(targets) != 1:
            return False
        target = targets[0]
        if isinstance(target, ast.Name):
            if len(aliases) != 1:
                return False
            param = aliases[0]
            self.alias_to_param[target.id] = param
            if param in self.use_map:
                self.use_map[param].current_aliases.add(target.id)
            return True
        if isinstance(target, (ast.Tuple, ast.List)):
            if len(target.elts) != len(aliases):
                return False
            if not all(isinstance(elt, ast.Name) for elt in target.elts):
                return False
            named_targets = cast(list[ast.Name], target.elts)
            for elt, param in zip(named_targets, aliases):
                check_deadline()
                self.alias_to_param[elt.id] = param
                if param in self.use_map:
                    self.use_map[param].current_aliases.add(elt.id)
            return True
        return False

    def visit_Assign(self, node: ast.Assign) -> None:
        check_deadline()
        if isinstance(node.value, ast.Call):
            aliases = self._alias_from_call(node.value)
            if aliases and self._bind_return_alias(node.targets, aliases):
                self.visit(node.value)
                return
        rhs_param = None
        if isinstance(node.value, ast.Name) and node.value.id in self.alias_to_param:
            rhs_param = self.alias_to_param[node.value.id]

        handled_alias = False
        for target in node.targets:
            check_deadline()
            if self._bind_sequence(target, node.value):
                handled_alias = True
                continue
            if rhs_param and isinstance(target, ast.Name):
                self.alias_to_param[target.id] = rhs_param
                self.use_map[rhs_param].current_aliases.add(target.id)
                handled_alias = True
            elif rhs_param and isinstance(target, ast.Attribute):
                if isinstance(target.value, ast.Name):
                    self._attr_alias_to_param[(target.value.id, target.attr)] = rhs_param
                    handled_alias = True
            elif rhs_param and isinstance(target, ast.Subscript):
                if (
                    isinstance(target.value, ast.Name)
                    and isinstance(target.slice, ast.Constant)
                    and isinstance(target.slice.value, str)
                ):
                    self._key_alias_to_param[
                        (target.value.id, target.slice.value)
                    ] = rhs_param
                    handled_alias = True
            else:
                self._check_write(target)

        if handled_alias:
            sources = self._collect_alias_sources(node.value)
            self._suspend_non_forward.update(sources)
            self.visit(node.value)
            self._suspend_non_forward.difference_update(sources)
        else:
            self.visit(node.value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is None:
            return
        if isinstance(node.value, ast.Call) and isinstance(node.target, ast.Name):
            aliases = self._alias_from_call(node.value)
            if aliases and len(aliases) == 1:
                param = aliases[0]
                self.alias_to_param[node.target.id] = param
                if param in self.use_map:
                    self.use_map[param].current_aliases.add(node.target.id)
                self.visit(node.value)
                return
        rhs_param = None
        if isinstance(node.value, ast.Name) and node.value.id in self.alias_to_param:
            rhs_param = self.alias_to_param[node.value.id]
        handled_alias = False
        if isinstance(node.target, ast.Name) and rhs_param:
            self.alias_to_param[node.target.id] = rhs_param
            self.use_map[rhs_param].current_aliases.add(node.target.id)
            handled_alias = True
        else:
            self._check_write(node.target)
        if handled_alias:
            sources = self._collect_alias_sources(node.value)
            self._suspend_non_forward.update(sources)
            self.visit(node.value)
            self._suspend_non_forward.difference_update(sources)
        else:
            self.visit(node.value)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self._check_write(node.target)
        self.visit(node.value)

    def visit_Name(self, node: ast.Name) -> None:
        check_deadline()
        if not isinstance(node.ctx, ast.Load):
            return
        if node.id not in self.alias_to_param:
            return
        parent = self.parents.get(node)
        if isinstance(parent, ast.Starred):
            param_name = self.alias_to_param[node.id]
            if self.strictness == "high":
                self.use_map[param_name].non_forward = True
                return
            call, _ = self.call_context(node, self.parents)
            self._record_forward(param_name, "args[*]", "arg[*]", call)
            return
        if isinstance(parent, ast.keyword) and parent.arg is None:
            param_name = self.alias_to_param[node.id]
            if self.strictness == "high":
                self.use_map[param_name].non_forward = True
                return
            call, _ = self.call_context(node, self.parents)
            self._record_forward(param_name, "kwargs[*]", "kw[*]", call)
            return
        param_name = self.alias_to_param[node.id]
        if param_name in self._suspend_non_forward:
            return
        call, direct = self.call_context(node, self.parents)
        if call is None or not direct:
            self.use_map[param_name].non_forward = True
            return
        callee = self.callee_name(call)
        slot = self._slot_for_call_node(call, node)
        self._record_forward(param_name, callee, slot, call)

    def _root_name(self, node: ast.AST) -> str | None:
        check_deadline()
        current = node
        while isinstance(current, (ast.Attribute, ast.Subscript)):
            check_deadline()
            current = current.value
        if isinstance(current, ast.Name):
            return current.id
        return None

    def visit_Attribute(self, node: ast.Attribute) -> None:
        check_deadline()
        if not isinstance(node.ctx, ast.Load):
            return
        if not isinstance(node.value, ast.Name):
            root_name = self._root_name(node)
            if root_name and root_name in self.alias_to_param:
                param_name = self.alias_to_param[root_name]
                self._mark_non_forward(param_name)
            self.generic_visit(node)
            return
        key = (node.value.id, node.attr)
        if key not in self._attr_alias_to_param:
            if node.value.id in self.alias_to_param:
                param_name = self.alias_to_param[node.value.id]
                self._mark_non_forward(param_name)
            return
        param_name = self._attr_alias_to_param[key]
        if param_name in self._suspend_non_forward:
            return
        call, direct = self.call_context(node, self.parents)
        if call is None or not direct:
            self.use_map[param_name].non_forward = True
            return
        callee = self.callee_name(call)
        slot = self._slot_for_call_node(call, node)
        self._record_forward(param_name, callee, slot, call)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        check_deadline()
        if not isinstance(node.ctx, ast.Load):
            return
        if not isinstance(node.value, ast.Name):
            root_name = self._root_name(node)
            if root_name and root_name in self.alias_to_param:
                param_name = self.alias_to_param[root_name]
                self._mark_non_forward(param_name)
            self.generic_visit(node)
            return
        key_value = None
        if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
            key_value = node.slice.value
        if key_value is None:
            if node.value.id in self.alias_to_param:
                param_name = self.alias_to_param[node.value.id]
                self._mark_non_forward(param_name)
            self.visit(node.slice)
            return
        key = (node.value.id, key_value)
        if key not in self._key_alias_to_param:
            if node.value.id in self.alias_to_param:
                param_name = self.alias_to_param[node.value.id]
                self._mark_non_forward(param_name)
            self.visit(node.slice)
            return
        param_name = self._key_alias_to_param[key]
        if param_name in self._suspend_non_forward:
            return
        call, direct = self.call_context(node, self.parents)
        if call is None or not direct:
            self.use_map[param_name].non_forward = True
            return
        callee = self.callee_name(call)
        slot = self._slot_for_call_node(call, node)
        self._record_forward(param_name, callee, slot, call)
