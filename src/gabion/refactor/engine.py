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
