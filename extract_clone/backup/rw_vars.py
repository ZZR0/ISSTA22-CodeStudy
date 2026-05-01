# AST_Clone_Extractability/rw_vars.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Set, Tuple

from util_ast import iter_descendants_cursor


REG_PRE = "CloneRegion_pre"
REG_WITHIN = "CloneRegion_within"
REG_POST = "CloneRegion_post"


def _region_for_line(line: int, start: int, end: int) -> str:
    if line < start:
        return REG_PRE
    if line > end:
        return REG_POST
    return REG_WITHIN


def collect_parameters(parser, method_node) -> Set[str]:
    """Return names of formal parameters of the enclosing method/constructor."""
    params: Set[str] = set()
    param_list = method_node.child_by_field_name("parameters")
    if not param_list:
        return params

    for n in iter_descendants_cursor(param_list):
        if n.type in {"formal_parameter", "spread_parameter"}:
            nm = n.child_by_field_name("name")
            if nm and nm.type == "identifier":
                params.add(parser.text_of(nm))
    return params


def collect_locals(parser, method_node) -> Set[str]:
    """Return names of local variables declared in the method body."""
    locals_: Set[str] = set()
    body = method_node.child_by_field_name("body")
    if not body:
        return locals_

    for n in iter_descendants_cursor(body):
        if n.type == "local_variable_declaration":
            for d in iter_descendants_cursor(n):
                if d.type == "variable_declarator":
                    nm = d.child_by_field_name("name")
                    if nm and nm.type == "identifier":
                        locals_.add(parser.text_of(nm))
    return locals_


def _is_descendant(node, ancestor) -> bool:
    cur = node
    while cur is not None:
        if cur == ancestor:
            return True
        cur = cur.parent
    return False


def classify_identifier_rw(ident_node) -> Tuple[bool, bool]:
    """
    Classify a Java identifier occurrence as (read, write) using syntactic context.

    This is a conservative approximation intended for clone refactoring feasibility
    screening (Extract Method). Definitions-at-entry (e.g., parameters) are not
    counted as writes within the clone region. Specifically: Simple assignments (LHS) are NOT reads.

    Key rules:
      - variable_declarator name            => write
      - formal_parameter / catch parameter  => neither (definition at entry)
      - assignment LHS                      => write (compound => read + write)
      - assignment RHS                      => read
      - update_expression (i++, ++i)        => read + write
      - method_invocation name              => neither (not a variable)
      - field_access field                  => neither (member name)
      - default                             => read
    """
    p = ident_node.parent
    if p is None:
        return True, False

    # Fix 1: Simple Assignment LHS is NOT a read
    if p.type == "assignment_expression":
        left = p.child_by_field_name("left")
        if left and (ident_node == left or _is_descendant(ident_node, left)):
            op = p.child_by_field_name("operator")
            op_txt = op.text.decode("utf-8", "ignore") if op else "="
            # Only read if it's compound like +=, -=, etc.
            is_r = (op_txt != "=")
            return (is_r, True)
        return True, False  # RHS is always a read

    # Fix 2: Variable Declarators (e.g., FileStatus status = ...)
    if p.type == "variable_declarator":
        nm = p.child_by_field_name("name")
        if nm == ident_node:
            return False, True

    # Fix 3: Method calls and Fields are not variable dependencies
    if p.type == "method_invocation":
        nm = p.child_by_field_name("name")
        if nm and (ident_node == nm or _is_descendant(ident_node, nm)):
            return False, False
            
    if p.type == "field_access":
        fld = p.child_by_field_name("field")
        if fld and (ident_node == fld or _is_descendant(ident_node, fld)):
            return False, False

    return True, False


@dataclass
class RWRegions:
    locals_in_method: Set[str]
    params_in_method: Set[str]
    vr: Dict[str, Set[str]]  # region -> variable names
    vw: Dict[str, Set[str]]  # region -> variable names


def extract_rw_by_region(
    parser,
    method_node,
    clone_start: int,
    clone_end: int,
    only_method_scope: bool = True,
) -> RWRegions:
    """
    Compute variable read/write sets (V_r, V_w) for pre/within/post clone regions.

    Args:
        parser: JavaTreeSitterParser
        method_node: enclosing method node
        clone_start, clone_end: clone line range in the file (inclusive)
        only_method_scope: if True, restrict to method-scoped vars (params + locals)

    Returns:
        RWRegions with vr/vw maps keyed by:
          - CloneRegion_pre
          - CloneRegion_within
          - CloneRegion_post
    """
    params = collect_parameters(parser, method_node)
    locals_ = collect_locals(parser, method_node)
    scope_vars = params | locals_

    vr = {REG_PRE: set(), REG_WITHIN: set(), REG_POST: set()}
    vw = {REG_PRE: set(), REG_WITHIN: set(), REG_POST: set()}
    
    # Track variables written inside the clone to identify local definitions
    locally_defined_within: Set[str] = set()

    # The traversal MUST be in document order for this logic to work
    for n in iter_descendants_cursor(method_node):
        if n.type != "identifier":
            continue

        name = parser.text_of(n)
        if only_method_scope and name not in scope_vars:
            continue

        line = n.start_point[0] + 1
        region = _region_for_line(line, clone_start, clone_end)
        is_r, is_w = classify_identifier_rw(n)

        if region == REG_WITHIN:
            # Paper Logic: A variable is only an external 'Use' if it 
            # hasn't been defined/overwritten earlier in this same region.
            if is_r and name not in locally_defined_within:
                vr[region].add(name)
            
            if is_w:
                vw[region].add(name)
                locally_defined_within.add(name)
        else:
            # Pre and Post regions collect all occurrences normally
            if is_r: vr[region].add(name)
            if is_w: vw[region].add(name)

    return RWRegions(
        locals_in_method=locals_,
        params_in_method=params,
        vr=vr,
        vw=vw,
    )