from typing import Iterable, Optional, List
from typing import Set, Dict, Tuple
from dataclasses import dataclass
from java_treesitter_parser import JavaTreeSitterParser
import re

_METHOD_NODES = {
    "method_declaration", "constructor_declaration", "compact_constructor_declaration",
}

_CLASS_NODES = {
    "class_declaration", "interface_declaration", "enum_declaration", "record_declaration",
}

def enclosing_class_node(node):
    """
    Walk upward from `node` to find the nearest enclosing class-like node.
    Returns the node or None if not inside a class/interface/enum/record.
    """
    cur = node
    while cur is not None:
        if cur.type in _CLASS_NODES:
            return cur
        cur = cur.parent
    return None

def class_simple_name(parser, class_like_node) -> str | None:
    """
    Return the simple name of a class/interface/enum/record node, if present.
    """
    nm = class_like_node.child_by_field_name("name")
    return parser.text_of(nm) if nm else None

def enclosing_class_name(parser, node) -> str | None:
    """
    Find the nearest enclosing class-like node for `node` and return its name.
    If no class is found, return None (or change to '<toplevel>' if you prefer).
    """
    cls = enclosing_class_node(node)
    return class_simple_name(parser, cls) if cls else None

def same_node(a, b) -> bool:
    # Tree-sitter Node equality isn’t guaranteed by `is`, so compare by span+type.
    return (a is b) or (a and b and a.type == b.type
                        and a.start_byte == b.start_byte
                        and a.end_byte == b.end_byte)

def classes_directly_under(cls_node) -> Iterable:
    """
    Yield only class-like nodes (nested classes/interfaces/enums/records)
    whose nearest enclosing class is exactly `cls_node`.
    """
    for n in iter_descendants(cls_node):
        if n.type in _CLASS_NODES:
            owner = enclosing_class_node(n)
            if same_node(owner, cls_node):
                yield n

def methods_directly_under(cls_node) -> Iterable:
    """
    Yield only the method-like nodes whose nearest enclosing class
    is exactly `cls_node` (not a nested class).
    """
    for n in iter_descendants(cls_node):
        if n.type in _METHOD_NODES:
            owner = enclosing_class_node(n)
            if same_node(owner, cls_node):
                yield n

def class_name(parser: JavaTreeSitterParser, class_node) -> Optional[str]:
    """Return the simple name of a class/interface/enum/record node, if present."""
    nm = class_node.child_by_field_name("name")
    return parser.text_of(nm) if nm else None


def method_name(parser: JavaTreeSitterParser, decl) -> Optional[str]:
    """Return the name of a method/constructor/compact-constructor declaration node."""
    nm = decl.child_by_field_name("name")
    if nm:
        return parser.text_of(nm)
    for ch in decl.children:
        if ch.type == "method_declarator":
            x = ch.child_by_field_name("name")
            if x:
                return parser.text_of(x)
        if ch.type == "identifier":
            return parser.text_of(ch)
    return None


def collect_modifiers(parser: JavaTreeSitterParser, decl) -> List[str]:
    """Collect all modifiers (public, private, static, etc.) for a method or class node."""
    mods = decl.child_by_field_name("modifiers")
    if not mods:
        return []
    return [parser.text_of(ch).strip() for ch in mods.children if parser.text_of(ch).strip()]


def collect_params(parser: JavaTreeSitterParser, decl) -> List[str]:
    """Extract parameter list for a method or constructor as strings."""
    params_parent = decl.child_by_field_name("parameters")
    if not params_parent:
        return []
    items = []
    for ch in params_parent.children:
        if ch.type in {"formal_parameter", "spread_parameter", "receiver_parameter"}:
            items.append(parser.text_of(ch).strip())
    if not items:
        # Fallback: manually parse the parameter string
        body = parser.text_of(params_parent).strip()
        if body.startswith("(") and body.endswith(")"):
            body = body[1:-1].strip()
        if body:
            items = [p.strip() for p in body.split(",")]
    return items


def collect_local_vars(parser, method_node) -> List[Tuple[str, str, int]]:
    """
    Return a list of (type, name, line_number) for all local variables 
    declared in the method body.
    """
    locals_list: List[Tuple[str, str, int]] = []
    body = method_node.child_by_field_name("body")
    
    if not body:
        return locals_list

    scope_boundaries = {
        "class_declaration", "interface_declaration", 
        "object_creation_expression", "lambda_expression"
    }

    stack = [body]
    while stack:
        current = stack.pop()
        
        # Get the 1-based line number of this declaration
        line_num = current.start_point[0] + 1

        if current.type == "local_variable_declaration":
            type_node = current.child_by_field_name("type")
            if type_node:
                var_type = parser.text_of(type_node).strip()
                for child in current.children:
                    if child.type == "variable_declarator":
                        name_node = child.child_by_field_name("name")
                        if name_node:
                            var_name = parser.text_of(name_node).strip()
                            locals_list.append((var_type, var_name, line_num))

        elif current.type in {"enhanced_for_statement", "catch_formal_parameter"}:
            type_node = current.child_by_field_name("type")
            name_node = current.child_by_field_name("name")
            if type_node and name_node:
                var_type = parser.text_of(type_node).strip()
                var_name = parser.text_of(name_node).strip()
                locals_list.append((var_type, var_name, line_num))

        for child in reversed(current.children):
            if child.type not in scope_boundaries:
                stack.append(child)

    return locals_list

def collect_class_fields(parser, class_node) -> List[Tuple[str, str]]:
    """
    Extract all class-level field types and names,
    ignoring loggers and standard ALL_CAPS constants.
    """
    fields = []
    if not class_node:
        return fields
        
    body = class_node.child_by_field_name("body")
    if not body:
        return fields
        
    for child in body.children:
        if child.type == "field_declaration":
            # Extract the type (e.g., "Queue<Transformer>", "Connector", "Map<String, Object>")
            type_node = child.child_by_field_name("type")
            field_type = parser.text_of(type_node).strip() if type_node else "Object"
            
            for sub in child.children:
                if sub.type == "variable_declarator":
                    name_node = sub.child_by_field_name("name")
                    if name_node:
                        field_name = parser.text_of(name_node).strip()
                        
                        # --- Heuristic Filters ---
                        if field_name.upper() in ("LOG", "LOGGER"):
                            continue
                        if re.fullmatch(r"[A-Z0-9_]+", field_name):
                            continue
                            
                        # Append both type and name
                        fields.append((field_type, field_name))
                        
    return fields

# --- Constants for Data-Flow Regions ---
REG_PRE = "PRE"
REG_WITHIN = "WITHIN"
REG_POST = "POST"

@dataclass
class RWRegions:
    """Stores the read (vr) and write (vw) sets for local variables across regions."""
    locals_in_method: Set[str]
    params_in_method: Set[str]
    fields_in_class: Set[str]  # <--- NEW: Track class fields as well
    field_types: Dict[str, str]  # <--- ADD THIS
    vr: Dict[str, Set[str]]
    vw: Dict[str, Set[str]]

def _region_for_line(line: int, clone_start: int, clone_end: int) -> str:
    """Classifies a line number into Pre, Within, or Post clone regions."""
    if line < clone_start:
        return REG_PRE
    elif line > clone_end:
        return REG_POST
    else:
        return REG_WITHIN

def classify_identifier_rw(node) -> Tuple[bool, bool]:
    """
    Determines if an identifier node is being Read (is_r) and/or Written (is_w).
    Handles direct assignments, arrays (arr[i]=1), and fields (obj.x=1).
    """
    is_r = True
    is_w = False
    
    parent = node.parent
    if not parent:
        return (is_r, is_w)

    # 1. Fast checks for declarations and special bindings (Pure Writes)
    if parent.type == "variable_declarator" and parent.child_by_field_name("name") == node:
        return (False, True) 
    if parent.type == "enhanced_for_statement" and parent.child_by_field_name("name") == node:
        return (False, True)
    if parent.type == "catch_formal_parameter" and parent.child_by_field_name("name") == node:
        return (False, True)

    # 2. Traverse upward through accessors to see if this node is the target of a mutation
    cur = node
    is_pure_assign = False
    
    while cur.parent and cur.parent.type in ("field_access", "array_access"):
        p = cur.parent
        if p.type == "array_access" and p.child_by_field_name("index") == cur:
            break  # The index itself is just being read, stop traversing
        if p.type == "field_access" and p.child_by_field_name("object") == cur:
            break  # The base object is just being read, stop traversing
        cur = p

    # 3. Check if the top-level accessor we found is being mutated
    if cur.parent:
        if cur.parent.type == "assignment_expression" and cur.parent.child_by_field_name("left") == cur:
            is_w = True
            op = cur.parent.child_by_field_name("operator")
            if op and op.type == "=":
                is_pure_assign = True
        elif cur.parent.type == "update_expression":
            is_w = True

    # 4. Final Read/Write resolution
    if is_w:
        # If it is a pure assignment directly to the identifier (e.g., x = 5), it's a pure write.
        # If it is an array/field mutation (e.g., arr[i] = 5), the base identifier is technically 
        # read to resolve the reference in memory, then mutated.
        if is_pure_assign and cur == node:
            is_r = False
        else:
            is_r = True

    return (is_r, is_w)

def extract_rw_by_region(
    parser,
    method_node,
    clone_start: int,
    clone_end: int,
    only_method_scope: bool = True,
) -> RWRegions:
    """
    Compute variable read/write sets (V_r, V_w) for pre/within/post clone regions.
    """
    # 1. Clean Method Parameters
    raw_params = collect_params(parser, method_node)
    params = set()
    for p in raw_params:
        parts = str(p).strip().rsplit(" ", 1)
        if len(parts) == 2:
            params.add(parts[1])
        elif len(parts) == 1:
            params.add(parts[0])

    # 2. Local Variables
    raw_locals = collect_local_vars(parser, method_node)
    locals_ = {var_name for _, var_name, _ in raw_locals}

    # 3. Class Fields
    cls_node = enclosing_class_node(method_node)
    raw_fields = collect_class_fields(parser, cls_node)
    
    fields = {f_name for f_type, f_name in raw_fields}
    field_type_map = {f_name: f_type for f_type, f_name in raw_fields}
    
    # 4. Build the master scope set
    scope_vars = params | locals_ | fields

    vr = {REG_PRE: set(), REG_WITHIN: set(), REG_POST: set()}
    vw = {REG_PRE: set(), REG_WITHIN: set(), REG_POST: set()}
    
    locally_defined_within: Set[str] = set()

    for n in iter_descendants(method_node):
        # NEW: Must include field_identifier to catch object properties and "this.x"
        if n.type not in ("identifier", "field_identifier"):
            continue

        name = parser.text_of(n)
        if only_method_scope and name not in scope_vars:
            continue

        line = n.start_point[0] + 1
        region = _region_for_line(line, clone_start, clone_end)
        is_r, is_w = classify_identifier_rw(n)
        
        var_with_line = f"{name} (Line {line})"

        if region == REG_WITHIN:
            if is_r and name not in locally_defined_within:
                vr[region].add(var_with_line)
            
            if is_w:
                vw[region].add(var_with_line)
                locally_defined_within.add(name)
        else:
            if is_r: vr[region].add(var_with_line)
            if is_w: vw[region].add(var_with_line)

    return RWRegions(
        locals_in_method=locals_,
        params_in_method=params,
        fields_in_class=fields,    # <--- ADD THIS
        field_types=field_type_map, # <--- ADD THIS
        vr=vr,
        vw=vw,
    )

def return_type(parser: JavaTreeSitterParser, decl) -> Optional[str]:
    """Return the return type of a method, or None if it is a constructor."""
    if decl.type in {"constructor_declaration", "compact_constructor_declaration"}:
        return None
    t = decl.child_by_field_name("type")
    return parser.text_of(t).strip() if t else None


# def iter_descendants(node) -> Iterable:
#     """
#     Depth-first traversal generator over an AST node and all its descendants.
#     Yields nodes in natural left-to-right source order.
#     """
#     stack = [node]
#     while stack:
#         n = stack.pop()
#         yield n
#         # Important: reversed() keeps children visited left-to-right
#         stack.extend(reversed(n.children))

def iter_descendants(node) -> Iterable:
    return iter_descendants_cursor(node)


def iter_descendants_cursor(node) -> Iterable:
    """
    Depth-first traversal generator over an AST node and all its descendants,
    implemented with a TreeCursor. Yields nodes in natural left-to-right order.
    """
    cursor = node.walk()
    done = False

    while not done:
        # Pre-order visit: yield the node we’re currently on
        yield cursor.node

        # Try to go deeper first (first child), then sideways (next sibling),
        # and if neither exists, backtrack up until a sibling is found (or root is done).
        if cursor.goto_first_child():
            continue
        if cursor.goto_next_sibling():
            continue

        while True:
            if not cursor.goto_parent():
                done = True  # backtracked past root; traversal complete
                break
            if cursor.goto_next_sibling():
                break
