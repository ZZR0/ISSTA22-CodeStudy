# AST_Clone_Extractability/hazards.py
from util_ast import iter_descendants_cursor

_HAZARD_ALWAYS = {"break_statement", "continue_statement", "throw_statement", "yield_statement"}
_RETURN = "return_statement"

_NESTED_FUNCTION_LIKE = {
    "method_declaration",
    "constructor_declaration",
    "compact_constructor_declaration",
    "lambda_expression",
}

def _in_range(node, start, end) -> bool:
    line = node.start_point[0] + 1
    return start <= line <= end

def _has_nested_function_ancestor(node, outer_method_node) -> bool:
    cur = node.parent
    while cur is not None:
        if cur == outer_method_node:
            return False
        if cur.type in _NESTED_FUNCTION_LIKE:
            return True
        cur = cur.parent
    return False

def _tail_return_is_safe(method_node, clone_start, clone_end, ret_node) -> bool:
    ret_line = ret_node.start_point[0] + 1
    for n in iter_descendants_cursor(method_node):
        if not _in_range(n, clone_start, clone_end):
            continue
        if _has_nested_function_ancestor(n, method_node):
            continue
        if (n.start_point[0] + 1) > ret_line and n.type.endswith("_statement"):
            return False
    return True

def detect_cf_hazard_detail(method_node, clone_start: int, clone_end: int):
    """
    Return (hazard: bool, detail: dict|None)
    detail = {"type": <node_type>, "line": <1-based file line>}
    """
    for n in iter_descendants_cursor(method_node):
        if not _in_range(n, clone_start, clone_end):
            continue
        if _has_nested_function_ancestor(n, method_node):
            continue

        if n.type in _HAZARD_ALWAYS:
            return True, {"type": n.type, "line": n.start_point[0] + 1}

        if n.type == _RETURN:
            if not _tail_return_is_safe(method_node, clone_start, clone_end, n):
                return True, {"type": n.type, "line": n.start_point[0] + 1}

    return False, None

def detect_cf_hazard(method_node, clone_start: int, clone_end: int) -> bool:
    hazard, _ = detect_cf_hazard_detail(method_node, clone_start, clone_end)
    return hazard
