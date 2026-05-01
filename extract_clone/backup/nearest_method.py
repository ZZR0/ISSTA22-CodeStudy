# nearest_method.py
"""
Utilities for locating Java methods in a source tree.

This module parses Java files with Tree-sitter, gathers method metadata,
and exposes helpers (e.g., ``find_fully_qualified_method``) that map code
span locations to the corresponding fully qualified method descriptors.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from tree_sitter import Node
from java_treesitter_parser import JavaTreeSitterParser


@dataclass(frozen=True)
class MethodInfo:
    qualified_name: str
    signature: str
    class_name: Optional[str]
    package_name: Optional[str]
    start_line: int
    end_line: int
    span: int
    source_code: str


def _slice(source_bytes: bytes, node: Node) -> str:
    # Robust decode: never crash on odd bytes
    return source_bytes[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _collect_package(root: Node, source_bytes: bytes) -> Optional[str]:
    for child in root.children:
        if child.type == "package_declaration":
            for grand in child.children:
                if grand.type in {"scoped_identifier", "identifier"}:
                    return _slice(source_bytes, grand)
    return None


def _is_class_like(node: Node) -> bool:
    return node.type in {
        "class_declaration",
        "interface_declaration",
        "enum_declaration",
        "annotation_type_declaration",
        "record_declaration",
    }


def _param_type_strings(param_node: Node, source_bytes: bytes) -> List[str]:
    param_types: List[str] = []
    for child in param_node.children:
        if child.type in {
            "formal_parameter",
            "receiver_parameter",
            "spread_parameter",
        }:
            for pchild in child.children:
                if pchild.type in {
                    "type_identifier",
                    "generic_type",
                    "integral_type",
                    "floating_point_type",
                    "boolean_type",
                    "void_type",
                    "array_type",
                    "primitive_type",
                }:
                    param_types.append(_slice(source_bytes, pchild))
                    break
        elif child.type == "inferred_parameters":
            param_types.append(_slice(source_bytes, child))
    return param_types


def _method_signature(node: Node, source_bytes: bytes) -> Tuple[str, str]:
    name = None
    params: List[str] = []
    for child in node.children:
        if child.type == "identifier":
            name = _slice(source_bytes, child)
        elif child.type == "formal_parameters":
            params = _param_type_strings(child, source_bytes)
    if name is None:
        name = "<anonymous>"
    signature = f"{name}({', '.join(params)})"
    return name, signature


def _walk_methods(
    node: Node,
    source_bytes: bytes,
    pkg: Optional[str],
    class_stack: List[str],
    out: List[MethodInfo],
) -> None:
    """
    Iterative DFS walk to avoid Python recursion depth limits.

    Semantics match the old recursive implementation:
      - On entering a class-like node: push class name (or <anonymous>)
      - On leaving a class-like node: pop
      - Record method_declaration and constructor_declaration nodes
    """
    class_stack = list(class_stack)  # defensive copy

    # (node, entering=True/False)
    stack: List[Tuple[Node, bool]] = [(node, True)]

    while stack:
        cur, entering = stack.pop()

        if entering:
            if _is_class_like(cur):
                cls_name = None
                for child in cur.children:
                    if child.type == "identifier":
                        cls_name = _slice(source_bytes, child)
                        break
                class_stack.append(cls_name if cls_name else "<anonymous>")

                # schedule exit to pop class_stack after processing children
                stack.append((cur, False))

            if cur.type in {"method_declaration", "constructor_declaration"}:
                _, signature = _method_signature(cur, source_bytes)

                qualified_parts: List[str] = []
                if pkg:
                    qualified_parts.append(pkg)
                if class_stack:
                    qualified_parts.append(".".join(class_stack))
                qualified_parts.append(signature)

                out.append(
                    MethodInfo(
                        qualified_name=".".join(part for part in qualified_parts if part),
                        signature=signature,
                        class_name=class_stack[-1] if class_stack else None,
                        package_name=pkg,
                        start_line=cur.start_point[0] + 1,
                        end_line=cur.end_point[0] + 1,
                        span=(cur.end_point[0] - cur.start_point[0] + 1),
                        source_code=_slice(source_bytes, cur),
                    )
                )

            # push children in reverse to preserve traversal order similar to recursion
            for child in reversed(cur.children):
                stack.append((child, True))

        else:
            if _is_class_like(cur) and class_stack:
                class_stack.pop()


@lru_cache(maxsize=256)
def _parse_methods(java_path: Path) -> Tuple[MethodInfo, ...]:
    """
    Parse methods from a Java file. Robust to encoding/pathological files.
    Returns empty tuple on failure so batch pipelines don't crash.
    """
    try:
        parser = JavaTreeSitterParser.from_file(java_path)
        source_bytes = parser.source_bytes
        package_name = _collect_package(parser.root_node, source_bytes)
        methods: List[MethodInfo] = []
        _walk_methods(parser.root_node, source_bytes, package_name, [], methods)
        return tuple(methods)
    except Exception:
        return tuple()


def _overlaps(start: int, end: int, method: MethodInfo) -> bool:
    return not (end < method.start_line or start > method.end_line)


def _nearest(line_start: int, line_end: int, methods: Sequence[MethodInfo]) -> MethodInfo:
    def distance(method: MethodInfo) -> int:
        if line_start > method.end_line:
            return line_start - method.end_line
        if line_end < method.start_line:
            return method.start_line - line_end
        return 0

    return min(methods, key=distance)


def _method_dict(method: MethodInfo) -> Dict[str, Union[str, int]]:
    return {
        "qualified_name": method.qualified_name,
        "start_line": method.start_line,
        "end_line": method.end_line,
        "span": method.span,
        "func": method.source_code,
    }


def find_fully_qualified_method(
    *,
    source_root: Union[str, Path],
    input_source_path: Union[str, Path],
    start_line: int,
    end_line: int,
) -> Dict[str, Union[bool, List[Dict[str, Union[str, int]]]]]:
    source_root = Path(source_root)
    rel_path = Path(input_source_path)
    java_path = source_root / rel_path
    if not java_path.is_file():
        raise FileNotFoundError(f"Missing source file: {java_path}")

    methods = list(_parse_methods(java_path))
    if not methods:
        return {"is_function": False, "methods": []}

    exact_methods = [
        _method_dict(m) for m in methods if m.start_line == start_line and m.end_line == end_line
    ]
    if exact_methods:
        return {"is_function": True, "methods": exact_methods}

    overlapping_methods = [_method_dict(m) for m in methods if _overlaps(start_line, end_line, m)]
    if overlapping_methods:
        overlapping_methods.sort(key=lambda m: (m["start_line"], m["end_line"]))
        return {"is_function": False, "methods": overlapping_methods}

    nearest_method = _method_dict(_nearest(start_line, end_line, methods))
    return {"is_function": False, "methods": [nearest_method]}
