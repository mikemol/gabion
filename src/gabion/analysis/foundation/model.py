from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class SymbolTable:
    imports: Dict[Tuple[str, str], str] = field(default_factory=dict)
    # Map: (module_name, local_name) -> fully_qualified_name


@dataclass
class ClassInfo:
    qual: str
    bases: List[str]
    methods: Set[str]


@dataclass
class DispatchTable:
    name: str
    targets: Set[str]


@dataclass(frozen=True)
class CallArgs:
    callee_expr: str
    pos_map: Dict[str, str]
    kw_map: Dict[str, str]
    resolved_targets: List[str] = field(default_factory=list)


@dataclass
class FunctionInfo:
    qual: str
    params: List[str]
    calls: List[CallArgs]


@dataclass
class ParamUse:
    direct_forward: Set[Tuple[str, str]]
    non_forward: bool
    current_aliases: Set[str]
