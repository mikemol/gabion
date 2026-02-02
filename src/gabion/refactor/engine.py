from __future__ import annotations

from pathlib import Path

import libcst as cst

from gabion.refactor.model import FieldSpec, RefactorPlan, RefactorRequest, TextEdit


class RefactorEngine:
    def __init__(self, project_root: Path | None = None) -> None:
        self.project_root = project_root

    def plan_protocol_extraction(self, request: RefactorRequest) -> RefactorPlan:
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

        def _is_docstring(stmt: cst.CSTNode) -> bool:
            if not isinstance(stmt, cst.SimpleStatementLine) or not stmt.body:
                return False
            expr = stmt.body[0]
            return isinstance(expr, cst.Expr) and isinstance(expr.value, cst.SimpleString)

        def _is_import(stmt: cst.CSTNode) -> bool:
            if not isinstance(stmt, cst.SimpleStatementLine):
                return False
            return any(isinstance(item, (cst.Import, cst.ImportFrom)) for item in stmt.body)

        def _has_import_from_typing(name: str) -> bool:
            for stmt in body:
                if not isinstance(stmt, cst.SimpleStatementLine):
                    continue
                for item in stmt.body:
                    if not isinstance(item, cst.ImportFrom):
                        continue
                    module_name = item.module
                    if not isinstance(module_name, cst.Name):
                        continue
                    if module_name.value != "typing":
                        continue
                    for alias in item.names:
                        if isinstance(alias, cst.ImportAlias) and isinstance(alias.name, cst.Name):
                            if alias.name.value == name:
                                return True
            return False

        def _has_typing_import() -> bool:
            for stmt in body:
                if not isinstance(stmt, cst.SimpleStatementLine):
                    continue
                for item in stmt.body:
                    if isinstance(item, cst.Import):
                        for alias in item.names:
                            if isinstance(alias, cst.ImportAlias) and isinstance(alias.name, cst.Name):
                                if alias.name.value == "typing":
                                    return True
            return False

        insert_idx = 0
        if body and _is_docstring(body[0]):
            insert_idx = 1
        while insert_idx < len(body) and _is_import(body[insert_idx]):
            insert_idx += 1

        protocol_base: cst.CSTNode
        import_stmt: cst.SimpleStatementLine | None = None
        if _has_typing_import():
            protocol_base = cst.Attribute(cst.Name("typing"), cst.Name("Protocol"))
        elif _has_import_from_typing("Protocol"):
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
            transformer = _RefactorTransformer(
                targets=targets,
                bundle_fields=bundle_fields,
                protocol_hint=protocol_hint,
            )
            new_module = new_module.visit(transformer)
            warnings.extend(transformer.warnings)

        new_source = new_module.code
        if new_source == source:
            warnings.append("No changes generated for protocol extraction.")
            return RefactorPlan(warnings=warnings)
        end_line = len(source.splitlines())
        return RefactorPlan(
            edits=[
                TextEdit(
                    path=str(path),
                    start=(0, 0),
                    end=(end_line, 0),
                    replacement=new_source,
                )
            ],
            warnings=warnings,
        )


class _RefactorTransformer(cst.CSTTransformer):
    def __init__(
        self,
        *,
        targets: set[str],
        bundle_fields: list[str],
        protocol_hint: str,
    ) -> None:
        self.targets = targets
        self.bundle_fields = bundle_fields
        self.protocol_hint = protocol_hint
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

    def visit_AsyncFunctionDef(self, node: cst.AsyncFunctionDef) -> bool:
        self._stack.append(node.name.value)
        return True

    def leave_AsyncFunctionDef(
        self, original_node: cst.AsyncFunctionDef, updated_node: cst.AsyncFunctionDef
    ) -> cst.CSTNode:
        updated = self._maybe_rewrite_function(original_node, updated_node)
        if self._stack:
            self._stack.pop()
        return updated

    def _maybe_rewrite_function(
        self,
        original_node: cst.FunctionDef | cst.AsyncFunctionDef,
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
        return updated_node.with_changes(params=new_params, body=new_body)

    def _ordered_param_names(self, params: cst.Parameters) -> list[str]:
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
        for param in params.posonly_params:
            if param.name.value == name:
                return param
        for param in params.params:
            if param.name.value == name:
                return param
        return None

    def _choose_bundle_name(self, existing: list[str]) -> str:
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
