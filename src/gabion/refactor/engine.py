from __future__ import annotations

from pathlib import Path

import libcst as cst

from gabion.refactor.model import FieldSpec, RefactorPlan, RefactorRequest, TextEdit
from gabion.analysis.timeout_context import check_deadline


class RefactorEngine:
    def __init__(self, project_root: Path | None = None) -> None:
        self.project_root = project_root

    def plan_protocol_extraction(self, request: RefactorRequest) -> RefactorPlan:
        check_deadline()
        path = Path(request.target_path)
        if self.project_root and not path.is_absolute():
            path = self.project_root / path
        try:
            source = path.read_text()
        except Exception as exc:
            return RefactorPlan(errors=[f"Failed to read {path}: {exc}"])
        try:
            module = cst.parse_module(source)
        except Exception as exc:
            return RefactorPlan(errors=[f"LibCST parse failed for {path}: {exc}"])
        protocol = (request.protocol_name or "").strip()
        if not protocol:
            return RefactorPlan(errors=["Protocol name is required for extraction."])
        bundle = [name.strip() for name in request.bundle or [] if name.strip()]
        field_specs: list[FieldSpec] = []
        seen_fields: set[str] = set()
        for spec in request.fields or []:
            name = (spec.name or "").strip()
            if not name or name in seen_fields:
                continue
            seen_fields.add(name)
            field_specs.append(spec)
        if bundle:
            for name in bundle:
                if name in seen_fields:
                    continue
                seen_fields.add(name)
                field_specs.append(FieldSpec(name=name))
        elif field_specs:
            bundle = [spec.name for spec in field_specs]
        if not bundle:
            return RefactorPlan(errors=["Bundle fields are required for extraction."])

        body = list(module.body)

        insert_idx = _find_import_insert_index(body)

        protocol_base: cst.CSTNode
        import_stmt: cst.SimpleStatementLine | None = None
        if _has_typing_import(body):
            protocol_base = cst.Attribute(cst.Name("typing"), cst.Name("Protocol"))
        elif _has_typing_protocol_import(body):
            protocol_base = cst.Name("Protocol")
        else:
            protocol_base = cst.Name("Protocol")
            import_stmt = cst.SimpleStatementLine(
                [cst.ImportFrom(module=cst.Name("typing"), names=[cst.ImportAlias(name=cst.Name("Protocol"))])]
            )

        doc_lines = []
        if request.rationale:
            doc_lines.append(f"Rationale: {request.rationale}")
        doc_lines.append(f"Bundle: {', '.join(bundle)}")
        docstring = cst.SimpleStatementLine(
            [cst.Expr(cst.SimpleString('"""' + "\\n".join(doc_lines) + '"""'))]
        )
        warnings: list[str] = []

        def _annotation_for(hint: str | None) -> cst.BaseExpression:
            if not hint:
                return cst.Name("object")
            try:
                return cst.parse_expression(hint)
            except Exception as exc:
                warnings.append(f"Failed to parse type hint '{hint}': {exc}")
                return cst.Name("object")

        field_lines = []
        for spec in field_specs:
            field_lines.append(
                cst.SimpleStatementLine(
                    [
                        cst.AnnAssign(
                            target=cst.Name(spec.name),
                            annotation=cst.Annotation(_annotation_for(spec.type_hint)),
                            value=None,
                        )
                    ]
                )
            )
        class_body = [docstring] + field_lines
        class_def = cst.ClassDef(
            name=cst.Name(protocol),
            bases=[cst.Arg(protocol_base)],
            body=cst.IndentedBlock(body=class_body),
        )

        new_body = list(body)
        if import_stmt is not None:
            new_body.insert(insert_idx, import_stmt)
            insert_idx += 1
        if insert_idx > 0 and not isinstance(new_body[insert_idx - 1], cst.EmptyLine):
            new_body.insert(insert_idx, cst.EmptyLine())
            insert_idx += 1
        new_body.insert(insert_idx, class_def)

        new_module = module.with_changes(body=new_body)

        targets = {name.strip() for name in request.target_functions or [] if name.strip()}
        bundle_fields = [spec.name for spec in field_specs]
        protocol_hint = protocol
        if targets:
            compat_shim = bool(request.compatibility_shim)
            if compat_shim:
                new_module = _ensure_compat_imports(new_module)
            transformer = _RefactorTransformer(
                targets=targets,
                bundle_fields=bundle_fields,
                protocol_hint=protocol_hint,
                compat_shim=compat_shim,
            )
            new_module = new_module.visit(transformer)
            warnings.extend(transformer.warnings)
        if targets:
            target_module = _module_name(path, self.project_root)
            call_warnings, call_edits = _rewrite_call_sites(
                new_module,
                file_path=path,
                target_path=path,
                target_module=target_module,
                protocol_name=protocol,
                bundle_fields=bundle_fields,
                targets=targets,
            )
            warnings.extend(call_warnings)
            if call_edits is None:
                new_source = new_module.code
            else:
                new_module = call_edits
                new_source = new_module.code
        else:
            new_source = new_module.code
        if new_source == source:  # pragma: no cover
            warnings.append("No changes generated for protocol extraction.")  # pragma: no cover
            return RefactorPlan(warnings=warnings)  # pragma: no cover
        end_line = len(source.splitlines())
        edits = [
            TextEdit(
                path=str(path),
                start=(0, 0),
                end=(end_line, 0),
                replacement=new_source,
            )
        ]

        if targets and self.project_root:
            extra_edits, extra_warnings = _rewrite_call_sites_in_project(
                project_root=self.project_root,
                target_path=path,
                target_module=_module_name(path, self.project_root),
                protocol_name=protocol,
                bundle_fields=bundle_fields,
                targets=targets,
            )
            edits.extend(extra_edits)
            warnings.extend(extra_warnings)
        return RefactorPlan(edits=edits, warnings=warnings)


def _module_name(path: Path, project_root: Path | None) -> str:
    rel = path.with_suffix("")
    if project_root is not None:
        try:
            rel = rel.relative_to(project_root)
        except ValueError:
            pass
    parts = list(rel.parts)
    if parts and parts[0] == "src":
        parts = parts[1:]
    return ".".join(parts)


def _is_docstring(stmt: cst.CSTNode) -> bool:
    if not isinstance(stmt, cst.SimpleStatementLine) or not stmt.body:
        return False
    expr = stmt.body[0]
    return isinstance(expr, cst.Expr) and isinstance(expr.value, cst.SimpleString)


def _is_import(stmt: cst.CSTNode) -> bool:
    if not isinstance(stmt, cst.SimpleStatementLine):
        return False
    return any(isinstance(item, (cst.Import, cst.ImportFrom)) for item in stmt.body)


def _find_import_insert_index(body: list[cst.CSTNode]) -> int:
    check_deadline()
    insert_idx = 0
    if body and _is_docstring(body[0]):
        insert_idx = 1
    while insert_idx < len(body) and _is_import(body[insert_idx]):
        insert_idx += 1
    return insert_idx


def _module_expr_to_str(expr: cst.BaseExpression | None) -> str | None:
    check_deadline()
    if expr is None:
        return None
    if isinstance(expr, cst.Name):
        return expr.value
    if isinstance(expr, cst.Attribute):
        parts = []
        current: cst.BaseExpression | None = expr
        while isinstance(current, cst.Attribute):
            if isinstance(current.attr, cst.Name):
                parts.append(current.attr.value)
            current = current.value
        if isinstance(current, cst.Name):
            parts.append(current.value)
        if parts:
            return ".".join(reversed(parts))
    return None  # pragma: no cover


def _has_typing_import(body: list[cst.CSTNode]) -> bool:
    check_deadline()
    for stmt in body:
        if not isinstance(stmt, cst.SimpleStatementLine):
            continue
        for item in stmt.body:
            if isinstance(item, cst.Import):
                for alias in item.names:
                    if isinstance(alias, cst.ImportAlias) and isinstance(alias.name, cst.Name):
                        if alias.name.value == "typing":
                            return True
                    if isinstance(alias, cst.ImportAlias) and isinstance(
                        alias.name, cst.Attribute
                    ):
                        if _module_expr_to_str(alias.name) == "typing":
                            return True  # pragma: no cover
    return False


def _has_typing_protocol_import(body: list[cst.CSTNode]) -> bool:
    check_deadline()
    for stmt in body:
        if not isinstance(stmt, cst.SimpleStatementLine):
            continue
        for item in stmt.body:
            if not isinstance(item, cst.ImportFrom):
                continue
            module = _module_expr_to_str(item.module)
            if module != "typing":
                continue
            for alias in item.names:
                if isinstance(alias, cst.ImportAlias) and isinstance(alias.name, cst.Name):
                    if alias.name.value == "Protocol":
                        return True
    return False


def _has_typing_overload_import(body: list[cst.CSTNode]) -> bool:
    check_deadline()
    for stmt in body:
        if not isinstance(stmt, cst.SimpleStatementLine):
            continue
        for item in stmt.body:
            if not isinstance(item, cst.ImportFrom):
                continue
            module = _module_expr_to_str(item.module)
            if module != "typing":
                continue
            for alias in item.names:
                if isinstance(alias, cst.ImportAlias) and isinstance(alias.name, cst.Name):
                    if alias.name.value == "overload":
                        return True
    return False


def _has_warnings_import(body: list[cst.CSTNode]) -> bool:
    check_deadline()
    for stmt in body:
        if not isinstance(stmt, cst.SimpleStatementLine):
            continue
        for item in stmt.body:
            if isinstance(item, cst.Import):
                for alias in item.names:
                    if isinstance(alias, cst.ImportAlias) and isinstance(alias.name, cst.Name):
                        if alias.name.value == "warnings":
                            return True
    return False


def _ensure_compat_imports(module: cst.Module) -> cst.Module:
    body = list(module.body)
    insert_idx = _find_import_insert_index(body)
    if not _has_warnings_import(body):
        body.insert(
            insert_idx,
            cst.SimpleStatementLine(
                [cst.Import(names=[cst.ImportAlias(name=cst.Name("warnings"))])]
            ),
        )
        insert_idx += 1
    if not _has_typing_overload_import(body):
        body.insert(
            insert_idx,
            cst.SimpleStatementLine(
                [
                    cst.ImportFrom(
                        module=cst.Name("typing"),
                        names=[cst.ImportAlias(name=cst.Name("overload"))],
                    )
                ]
            ),
        )
        insert_idx += 1
    return module.with_changes(body=body)


def _collect_import_context(
    module: cst.Module,
    *,
    target_module: str,
    protocol_name: str,
) -> tuple[dict[str, str], dict[str, str], str | None]:
    check_deadline()
    module_aliases: dict[str, str] = {}
    imported_targets: dict[str, str] = {}
    protocol_alias: str | None = None
    for stmt in module.body:
        if not isinstance(stmt, cst.SimpleStatementLine):
            continue
        for item in stmt.body:
            if isinstance(item, cst.Import):
                for alias in item.names:
                    if not isinstance(alias, cst.ImportAlias):  # pragma: no cover
                        continue
                    module_name = _module_expr_to_str(alias.name)
                    if not module_name:  # pragma: no cover
                        continue
                    if module_name != target_module:
                        continue
                    local = alias.asname.name.value if alias.asname else module_name
                    module_aliases[local] = module_name
            elif isinstance(item, cst.ImportFrom):
                module_name = _module_expr_to_str(item.module)
                if module_name != target_module:
                    continue
                for alias in item.names:
                    if not isinstance(alias, cst.ImportAlias):
                        continue
                    if not isinstance(alias.name, cst.Name):
                        continue
                    local = alias.asname.name.value if alias.asname else alias.name.value
                    imported_targets[local] = alias.name.value
                    if alias.name.value == protocol_name:
                        protocol_alias = local
    return module_aliases, imported_targets, protocol_alias


def _rewrite_call_sites(
    module: cst.Module,
    *,
    file_path: Path,
    target_path: Path,
    target_module: str,
    protocol_name: str,
    bundle_fields: list[str],
    targets: set[str],
) -> tuple[list[str], cst.Module | None]:
    check_deadline()
    warnings: list[str] = []
    file_is_target = file_path == target_path
    if not targets:
        return warnings, None
    target_simple = {name for name in targets if "." not in name}
    target_methods: dict[str, set[str]] = {}
    for name in targets:
        if "." not in name:
            continue
        parts = name.split(".")
        class_name = ".".join(parts[:-1])
        method = parts[-1]
        target_methods.setdefault(class_name, set()).add(method)
    module_aliases: dict[str, str] = {}
    imported_targets: dict[str, str] = {}
    protocol_alias: str | None = None
    if not file_is_target and target_module:
        module_aliases, imported_targets, protocol_alias = _collect_import_context(
            module, target_module=target_module, protocol_name=protocol_name
        )

    constructor_expr: cst.BaseExpression
    needs_import = False
    if file_is_target:
        constructor_expr = cst.Name(protocol_name)
    else:
        if protocol_alias:
            constructor_expr = cst.Name(protocol_alias)
        elif module_aliases:
            alias = sorted(module_aliases.keys())[0]
            constructor_expr = cst.Attribute(cst.Name(alias), cst.Name(protocol_name))
        else:
            constructor_expr = cst.Name(protocol_name)
            needs_import = True

    transformer = _CallSiteTransformer(
        file_is_target=file_is_target,
        target_simple=target_simple,
        target_methods=target_methods,
        module_aliases=set(module_aliases.keys()),
        imported_targets=set(
            name for name, original in imported_targets.items() if original in target_simple
        ),
        bundle_fields=bundle_fields,
        constructor_expr=constructor_expr,
    )
    new_module = module.visit(transformer)
    warnings.extend(transformer.warnings)
    if not transformer.changed:
        return warnings, None

    if not file_is_target and needs_import and target_module:
        body = list(new_module.body)
        insert_idx = _find_import_insert_index(body)
        try:
            module_expr = cst.parse_expression(target_module)
        except Exception:  # pragma: no cover
            module_expr = cst.Name(target_module.split(".")[0])  # pragma: no cover
        if not isinstance(module_expr, (cst.Name, cst.Attribute)):
            module_expr = cst.Name(target_module.split(".")[0])  # pragma: no cover
        import_stmt = cst.SimpleStatementLine(
            [
                cst.ImportFrom(
                    module=module_expr,
                    names=[cst.ImportAlias(name=cst.Name(protocol_name))],
                )
            ]
        )
        body.insert(insert_idx, import_stmt)
        new_module = new_module.with_changes(body=body)
    return warnings, new_module


def _rewrite_call_sites_in_project(
    *,
    project_root: Path,
    target_path: Path,
    target_module: str,
    protocol_name: str,
    bundle_fields: list[str],
    targets: set[str],
) -> tuple[list[TextEdit], list[str]]:
    check_deadline()
    edits: list[TextEdit] = []
    warnings: list[str] = []
    scan_root = project_root / "src"
    if not scan_root.exists():
        scan_root = project_root
    for path in sorted(scan_root.rglob("*.py")):
        if path == target_path:
            continue
        try:
            source = path.read_text()
        except Exception as exc:
            warnings.append(f"Failed to read {path}: {exc}")
            continue
        try:
            module = cst.parse_module(source)
        except Exception as exc:
            warnings.append(f"LibCST parse failed for {path}: {exc}")
            continue
        call_warnings, updated_module = _rewrite_call_sites(
            module,
            file_path=path,
            target_path=target_path,
            target_module=target_module,
            protocol_name=protocol_name,
            bundle_fields=bundle_fields,
            targets=targets,
        )
        warnings.extend(call_warnings)
        if updated_module is None:
            continue
        new_source = updated_module.code
        if new_source == source:  # pragma: no cover
            continue
        end_line = len(source.splitlines())
        edits.append(
            TextEdit(
                path=str(path),
                start=(0, 0),
                end=(end_line, 0),
                replacement=new_source,
            )
        )
    return edits, warnings


class _RefactorTransformer(cst.CSTTransformer):
    def __init__(
        self,
        *,
        targets: set[str],
        bundle_fields: list[str],
        protocol_hint: str,
        compat_shim: bool = False,
    ) -> None:
        # dataflow-bundle: bundle_fields, compat_shim, protocol_hint, targets
        self.targets = targets
        self.bundle_fields = bundle_fields
        self.protocol_hint = protocol_hint
        self.compat_shim = compat_shim
        self.warnings: list[str] = []
        self._stack: list[str] = []

    def visit_ClassDef(self, node: cst.ClassDef) -> bool:
        self._stack.append(node.name.value)
        return True

    def leave_ClassDef(
        self, original_node: cst.ClassDef, updated_node: cst.ClassDef
    ) -> cst.CSTNode:
        if self._stack:
            self._stack.pop()
        return updated_node

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
        self._stack.append(node.name.value)
        return True

    def leave_FunctionDef(
        self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef
    ) -> cst.CSTNode:
        updated = self._maybe_rewrite_function(original_node, updated_node)
        if self._stack:
            self._stack.pop()
        return updated

    def visit_AsyncFunctionDef(self, node: cst.AsyncFunctionDef) -> bool:  # pragma: no cover
        self._stack.append(node.name.value)  # pragma: no cover
        return True  # pragma: no cover

    def leave_AsyncFunctionDef(  # pragma: no cover
        self, original_node: cst.AsyncFunctionDef, updated_node: cst.AsyncFunctionDef
    ) -> cst.CSTNode:
        updated = self._maybe_rewrite_function(original_node, updated_node)  # pragma: no cover
        if self._stack:  # pragma: no cover
            self._stack.pop()  # pragma: no cover
        return updated  # pragma: no cover

    def _maybe_rewrite_function(
        self,
        original_node: cst.FunctionDef,
        updated_node: cst.FunctionDef | cst.AsyncFunctionDef,
    ) -> cst.CSTNode:
        qualname = ".".join(self._stack)
        name = original_node.name.value
        if name not in self.targets and qualname not in self.targets:
            return updated_node

        original_params = self._ordered_param_names(original_node.params)
        if not original_params:
            return updated_node

        keep_self = original_params[0] in {"self", "cls"}
        self_param = None
        if keep_self:
            self_param = self._find_self_param(original_node.params, original_params[0])

        bundle_param_name = self._choose_bundle_name(original_params)
        new_params = self._build_parameters(self_param, bundle_param_name)

        bundle_set = set(self.bundle_fields)
        target_fields = [
            name
            for name in original_params
            if name in bundle_set and name not in {"self", "cls"}
        ]
        new_body = self._inject_preamble(
            updated_node.body, bundle_param_name, target_fields
        )
        updated_impl = updated_node.with_changes(params=new_params, body=new_body)
        if not self.compat_shim:
            return updated_impl
        impl_name = f"_{name}_bundle"
        impl_node = updated_impl.with_changes(
            name=cst.Name(impl_name),
            decorators=[],
        )
        shim_nodes = self._build_compat_shim(
            original_node=original_node,
            impl_name=impl_name,
            bundle_param=bundle_param_name,
            self_param=self_param,
        )
        return cst.FlattenSentinel([*shim_nodes, impl_node])

    def _build_compat_shim(
        self,
        *,
        original_node: cst.FunctionDef | cst.AsyncFunctionDef,
        impl_name: str,
        bundle_param: str,
        self_param: cst.Param | None,
    ) -> list[cst.CSTNode]:
        decorators = list(original_node.decorators)
        overload_decorators = [cst.Decorator(cst.Name("overload")), *decorators]
        bundle_params = self._build_parameters(self_param, bundle_param)
        legacy_params = original_node.params
        bundle_stub = self._build_overload_stub(
            original_node=original_node,
            params=bundle_params,
            decorators=overload_decorators,
        )
        legacy_stub = self._build_overload_stub(
            original_node=original_node,
            params=legacy_params,
            decorators=overload_decorators,
        )
        wrapper_params = self._build_shim_parameters(self_param)
        wrapper_body = self._build_shim_body(
            impl_name=impl_name,
            bundle_type=self.protocol_hint,
            self_param=self_param,
            is_async=bool(original_node.asynchronous),
            public_name=original_node.name.value,
        )
        wrapper = self._build_wrapper(
            original_node=original_node,
            params=wrapper_params,
            body=wrapper_body,
            decorators=decorators,
        )
        return [bundle_stub, legacy_stub, wrapper]

    def _build_overload_stub(
        self,
        *,
        original_node: cst.FunctionDef,
        params: cst.Parameters,
        decorators: list[cst.Decorator],
    ) -> cst.CSTNode:
        body = cst.IndentedBlock(
            body=[cst.SimpleStatementLine([cst.Expr(cst.Ellipsis())])]
        )
        return original_node.with_changes(
            params=params,
            body=body,
            decorators=decorators,
        )

    def _build_wrapper(
        self,
        *,
        original_node: cst.FunctionDef,
        params: cst.Parameters,
        body: cst.BaseSuite,
        decorators: list[cst.Decorator],
    ) -> cst.CSTNode:
        return original_node.with_changes(
            params=params,
            body=body,
            decorators=decorators,
        )

    def _build_shim_parameters(self, self_param: cst.Param | None) -> cst.Parameters:
        params: list[cst.Param] = []
        if self_param is not None:
            params.append(self_param)
        return cst.Parameters(
            params=params,
            star_arg=cst.Param(name=cst.Name("args")),
            kwonly_params=[],
            star_kwarg=cst.Param(name=cst.Name("kwargs")),
            posonly_params=[],
            posonly_ind=cst.MaybeSentinel.DEFAULT,
        )

    def _build_shim_body(
        self,
        *,
        impl_name: str,
        bundle_type: str,
        self_param: cst.Param | None,
        is_async: bool,
        public_name: str,
    ) -> cst.BaseSuite:
        receiver = self_param.name.value if self_param is not None else None
        impl_call = f"{receiver}.{impl_name}" if receiver else impl_name
        return_prefix = "return await" if is_async else "return"
        guard = (
            f"if args and isinstance(args[0], {bundle_type}):\n"
            f"    if len(args) != 1 or kwargs:\n"
            f"        raise TypeError(\"{public_name}() bundle call expects a single {bundle_type} argument\")\n"
            f"    {return_prefix} {impl_call}(args[0])\n"
        )
        warn = (
            f"warnings.warn(\"{public_name}() is deprecated; use {public_name}({bundle_type}(...))\", "
            "DeprecationWarning, stacklevel=2)"
        )
        build = f"bundle = {bundle_type}(*args, **kwargs)"
        tail = f"{return_prefix} {impl_call}(bundle)"
        return cst.IndentedBlock(
            body=[
                cst.parse_statement(guard),
                cst.parse_statement(warn),
                cst.parse_statement(build),
                cst.parse_statement(tail),
            ]
        )

    def _ordered_param_names(self, params: cst.Parameters) -> list[str]:
        check_deadline()
        names: list[str] = []
        for param in params.posonly_params:
            names.append(param.name.value)
        for param in params.params:
            names.append(param.name.value)
        for param in params.kwonly_params:
            names.append(param.name.value)
        return names

    def _find_self_param(
        self, params: cst.Parameters, name: str
    ) -> cst.Param | None:
        check_deadline()
        for param in params.posonly_params:
            if param.name.value == name:
                return param
        for param in params.params:
            if param.name.value == name:
                return param
        return None

    def _choose_bundle_name(self, existing: list[str]) -> str:
        check_deadline()
        candidate = "bundle"
        if candidate not in existing:
            return candidate
        idx = 1
        while f"bundle_{idx}" in existing:
            idx += 1
        return f"bundle_{idx}"

    def _build_parameters(
        self, self_param: cst.Param | None, bundle_name: str
    ) -> cst.Parameters:
        params: list[cst.Param] = []
        if self_param is not None:
            params.append(self_param)
        annotation = None
        if self.protocol_hint:
            try:
                annotation = cst.Annotation(cst.parse_expression(self.protocol_hint))
            except Exception as exc:
                self.warnings.append(
                    f"Failed to parse protocol type hint '{self.protocol_hint}': {exc}"
                )
        params.append(
            cst.Param(
                name=cst.Name(bundle_name),
                annotation=annotation,
            )
        )
        return cst.Parameters(
            params=params,
            star_arg=cst.MaybeSentinel.DEFAULT,
            kwonly_params=[],
            star_kwarg=None,
            posonly_params=[],
            posonly_ind=cst.MaybeSentinel.DEFAULT,
        )

    def _inject_preamble(
        self, body: cst.BaseSuite, bundle_name: str, fields: list[str]
    ) -> cst.BaseSuite:
        if not fields:
            return body
        if not isinstance(body, cst.IndentedBlock):
            return body
        assign_lines = [
            cst.SimpleStatementLine(
                [
                    cst.Assign(
                        targets=[cst.AssignTarget(cst.Name(name))],
                        value=cst.Attribute(
                            value=cst.Name(bundle_name),
                            attr=cst.Name(name),
                        ),
                    )
                ]
            )
            for name in fields
        ]
        existing = list(body.body)
        insert_at = 0
        if existing:
            first = existing[0]
            if isinstance(first, cst.SimpleStatementLine) and first.body:
                expr = first.body[0]
                if isinstance(expr, cst.Expr) and isinstance(expr.value, cst.SimpleString):
                    insert_at = 1
        new_body = existing[:insert_at] + assign_lines + existing[insert_at:]
        return body.with_changes(body=new_body)


class _CallSiteTransformer(cst.CSTTransformer):
    def __init__(
        self,
        *,
        file_is_target: bool,
        target_simple: set[str],
        target_methods: dict[str, set[str]],
        module_aliases: set[str],
        imported_targets: set[str],
        bundle_fields: list[str],
        constructor_expr: cst.BaseExpression,
    ) -> None:
        # dataflow-bundle: bundle_fields, constructor_expr, file_is_target, imported_targets, module_aliases, target_methods, target_simple
        self.file_is_target = file_is_target
        self.target_simple = target_simple
        self.target_methods = target_methods
        self.module_aliases = module_aliases
        self.imported_targets = imported_targets
        self.bundle_fields = bundle_fields
        self.constructor_expr = constructor_expr
        self.changed = False
        self.warnings: list[str] = []
        self._class_stack: list[str] = []

    def visit_ClassDef(self, node: cst.ClassDef) -> bool:
        self._class_stack.append(node.name.value)
        return True

    def leave_ClassDef(
        self, original_node: cst.ClassDef, updated_node: cst.ClassDef
    ) -> cst.CSTNode:
        if self._class_stack:
            self._class_stack.pop()
        return updated_node

    def leave_Call(self, original_node: cst.Call, updated_node: cst.Call) -> cst.CSTNode:
        if not self._is_target_call(updated_node.func):
            return updated_node
        if self._already_wrapped(updated_node):
            return updated_node
        bundle_args = self._build_bundle_args(updated_node)
        if bundle_args is None:
            return updated_node
        bundle_call = cst.Call(func=self.constructor_expr, args=bundle_args)
        self.changed = True
        return updated_node.with_changes(args=[cst.Arg(value=bundle_call)])

    def _is_target_call(self, func: cst.BaseExpression) -> bool:
        if isinstance(func, cst.Name):
            if self.file_is_target and func.value in self.target_simple:
                return True
            if not self.file_is_target and func.value in self.imported_targets:
                return True
            return False
        if isinstance(func, cst.Attribute):
            if not isinstance(func.attr, cst.Name):  # pragma: no cover
                return False
            attr = func.attr.value
            if self.file_is_target and self._class_stack:
                class_name = ".".join(self._class_stack)
                methods = self.target_methods.get(class_name, set())
                if attr in methods and isinstance(func.value, cst.Name):
                    if func.value.value in {"self", "cls", self._class_stack[-1]}:
                        return True
            if not self.file_is_target and isinstance(func.value, cst.Name):
                if func.value.value in self.module_aliases and attr in self.target_simple:
                    return True
        return False

    def _already_wrapped(self, call: cst.Call) -> bool:
        if len(call.args) != 1:
            return False
        arg = call.args[0]
        if arg.star:
            return False
        value = arg.value
        if not isinstance(value, cst.Call):
            return False
        if isinstance(value.func, cst.Name) and isinstance(self.constructor_expr, cst.Name):
            return value.func.value == self.constructor_expr.value
        if isinstance(value.func, cst.Attribute) and isinstance(self.constructor_expr, cst.Attribute):
            return value.func.attr.value == self.constructor_expr.attr.value
        return False

    def _build_bundle_args(self, call: cst.Call) -> list[cst.Arg] | None:
        check_deadline()
        if any(arg.star in {"*", "**"} for arg in call.args):
            self.warnings.append("Skipped call with star args/kwargs during refactor.")
            return None
        positional = [arg for arg in call.args if arg.keyword is None]
        keyword_args = {
            arg.keyword.value: arg.value
            for arg in call.args
            if arg.keyword is not None and isinstance(arg.keyword, cst.Name)
        }
        for key in keyword_args:
            if key not in self.bundle_fields:
                self.warnings.append(
                    f"Skipped call with unknown keyword '{key}' during refactor."
                )
                return None
        mapping: dict[str, cst.BaseExpression] = {}
        mapping.update(keyword_args)
        remaining = [field for field in self.bundle_fields if field not in mapping]
        if len(positional) > len(remaining):
            self.warnings.append("Skipped call with extra positional args during refactor.")
            return None
        for field, arg in zip(remaining, positional):
            mapping[field] = arg.value
        if len(mapping) != len(self.bundle_fields):
            self.warnings.append("Skipped call with missing bundle fields during refactor.")
            return None
        return [
            cst.Arg(keyword=cst.Name(field), value=mapping[field])
            for field in self.bundle_fields
        ]
