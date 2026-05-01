from typing import Iterable, Optional, List
from typing import Set, Dict, Tuple
from dataclasses import dataclass
from python_treesitter_parser import PythonTreeSitterParser
import re

_METHOD_NODES = {
    "function_definition",
}

_CLASS_NODES = {
    "class_definition",
}

def enclosing_class_node(node):
    """
    Walk upward from `node` to find the nearest enclosing class definition.
    Returns the node or None if not inside a class.
    """
    cur = node
    while cur is not None:
        if cur.type in _CLASS_NODES:
            return cur
        cur = cur.parent
    return None

def class_simple_name(parser: PythonTreeSitterParser, class_like_node) -> str | None:
    """Return the simple name of a class definition node, if present."""
    nm = class_like_node.child_by_field_name("name")
    return parser.text_of(nm) if nm else None

def enclosing_class_name(parser: PythonTreeSitterParser, node) -> str | None:
    """Find the nearest enclosing class node for `node` and return its name."""
    cls = enclosing_class_node(node)
    return class_simple_name(parser, cls) if cls else None

def same_node(a, b) -> bool:
    # Tree-sitter Node equality isn’t guaranteed by `is`, so compare by span+type.
    return (a is b) or (a and b and a.type == b.type
                        and a.start_byte == b.start_byte
                        and a.end_byte == b.end_byte)

def classes_directly_under(cls_node) -> Iterable:
    """Yield only class nodes whose nearest enclosing class is exactly `cls_node`."""
    for n in iter_descendants(cls_node):
        if n.type in _CLASS_NODES:
            owner = enclosing_class_node(n)
            if same_node(owner, cls_node):
                yield n

def methods_directly_under(cls_node) -> Iterable:
    """Yield only the method nodes whose nearest enclosing class is exactly `cls_node`."""
    for n in iter_descendants(cls_node):
        if n.type in _METHOD_NODES:
            owner = enclosing_class_node(n)
            if same_node(owner, cls_node):
                yield n

def class_name(parser: PythonTreeSitterParser, class_node) -> Optional[str]:
    """Return the simple name of a class definition node, if present."""
    nm = class_node.child_by_field_name("name")
    return parser.text_of(nm) if nm else None


def method_name(parser: PythonTreeSitterParser, decl) -> Optional[str]:
    """Return the name of a function/method definition node."""
    nm = decl.child_by_field_name("name")
    return parser.text_of(nm) if nm else None


def collect_modifiers(parser: PythonTreeSitterParser, decl) -> List[str]:
    """
    Collect all modifiers. Python doesn't have public/private keywords, 
    but we can treat decorators (e.g., @staticmethod, @property) as modifiers.
    """
    modifiers = []
    if decl.parent and decl.parent.type == "decorated_definition":
        for child in decl.parent.children:
            if child.type == "decorator":
                modifiers.append(parser.text_of(child).strip())
    return modifiers


def collect_params(parser: PythonTreeSitterParser, decl) -> List[str]:
    """Extract parameter list for a function definition as strings."""
    params_parent = decl.child_by_field_name("parameters")
    if not params_parent:
        return []
    
    items = []
    param_types = {
        "identifier", "typed_parameter", "default_parameter", 
        "list_splat_pattern", "dictionary_splat_pattern", "typed_default_parameter"
    }
    
    for ch in params_parent.children:
        if ch.type in param_types:
            items.append(parser.text_of(ch).strip())
            
    return items


def collect_local_vars(parser: PythonTreeSitterParser, method_node) -> List[Tuple[str, str, int]]:
    """
    Return a list of (type, name, line_number) for local variables.
    Python is dynamically typed, so type defaults to "Any" unless annotated.
    """
    locals_list: List[Tuple[str, str, int]] = []
    body = method_node.child_by_field_name("body")
    
    if not body:
        return locals_list

    scope_boundaries = {"function_definition", "class_definition", "lambda"}

    stack = [body]
    while stack:
        current = stack.pop()
        line_num = current.start_point[0] + 1

        # Standard assignment: x = 5
        if current.type == "assignment":
            left_node = current.child_by_field_name("left")
            if left_node and left_node.type == "identifier":
                var_name = parser.text_of(left_node).strip()
                locals_list.append(("Any", var_name, line_num))
                
        # Typed assignment: x: int = 5
        elif current.type == "typed_assignment":
            left_node = current.child_by_field_name("left")
            type_node = current.child_by_field_name("type")
            if left_node and type_node and left_node.type == "identifier":
                var_name = parser.text_of(left_node).strip()
                var_type = parser.text_of(type_node).strip()
                locals_list.append((var_type, var_name, line_num))
                
        # Loop variables: for x in iterable:
        elif current.type == "for_statement":
            left_node = current.child_by_field_name("left")
            if left_node and left_node.type == "identifier":
                var_name = parser.text_of(left_node).strip()
                locals_list.append(("Any", var_name, line_num))
                
        # Exception binding: except Exception as e:
        elif current.type == "except_clause":
            name_node = current.child_by_field_name("name")
            type_node = current.child_by_field_name("type")
            if name_node:
                var_name = parser.text_of(name_node).strip()
                var_type = parser.text_of(type_node).strip() if type_node else "Exception"
                locals_list.append((var_type, var_name, line_num))

        for child in reversed(current.children):
            if child.type not in scope_boundaries:
                stack.append(child)

    return locals_list

def collect_class_fields(parser: PythonTreeSitterParser, class_node) -> List[Tuple[str, str]]:
    """
    Extract class-level attributes.
    Python doesn't have strict 'fields' like Java, so we look for assignments
    directly under the class body.
    """
    fields = []
    if not class_node:
        return fields
        
    body = class_node.child_by_field_name("body")
    if not body:
        return fields
        
    for child in body.children:
        # Check for class-level assignments like: x = 5 or x: int = 5
        if child.type == "expression_statement":
            expr = child.children[0]
            if expr.type in {"assignment", "typed_assignment"}:
                left_node = expr.child_by_field_name("left")
                if left_node and left_node.type == "identifier":
                    field_name = parser.text_of(left_node).strip()
                    
                    # Extract type if available
                    field_type = "Any"
                    if expr.type == "typed_assignment":
                        type_node = expr.child_by_field_name("type")
                        if type_node:
                            field_type = parser.text_of(type_node).strip()
                    
                    # --- Heuristic Filters ---
                    if field_name.upper() in ("LOG", "LOGGER"):
                        continue
                    if re.fullmatch(r"[A-Z0-9_]+", field_name): # Ignore ALL_CAPS constants
                        continue
                        
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
    fields_in_class: Set[str]
    field_types: Dict[str, str]
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
    Adapted for Python's AST (assignments, augmented assignments, etc.).
    """
    is_r = True
    is_w = False
    
    parent = node.parent
    if not parent:
        return (is_r, is_w)

    cur = node
    is_pure_assign = False
    
    # Traverse upward through attributes/subscripts (e.g., arr[i] or obj.x)
    while cur.parent and cur.parent.type in ("attribute", "subscript"):
        p = cur.parent
        if p.type == "subscript" and p.children[-2] == cur: # Roughly the index/slice
            break
        if p.type == "attribute" and p.child_by_field_name("object") == cur:
            break
        cur = p

    if cur.parent:
        # Standard Assignment: x = 1
        if cur.parent.type in ("assignment", "typed_assignment"):
            left = cur.parent.child_by_field_name("left")
            # If the current node is within the left-hand side pattern
            if left == cur or (left.type in ("pattern_list", "tuple_pattern") and cur in left.children):
                is_w = True
                is_pure_assign = True
                
        # Augmented Assignment: x += 1
        elif cur.parent.type == "augmented_assignment" and cur.parent.child_by_field_name("left") == cur:
            is_w = True
            is_r = True # Augmented assignment requires reading the old value
            
        # Loop iteration variable: for x in ...
        elif cur.parent.type == "for_statement" and cur.parent.child_by_field_name("left") == cur:
            is_w = True
            is_pure_assign = True

    # Final Read/Write resolution
    if is_w:
        if is_pure_assign and cur == node:
            is_r = False # Pure overwrite, no read needed
        else:
            is_r = True # e.g., mutating an attribute self.x = 1 (reads 'self')

    return (is_r, is_w)

def extract_rw_by_region(
    parser: PythonTreeSitterParser,
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
        # Strip type hints and defaults to get just the param name
        clean_name = p.split(':')[0].split('=')[0].strip()
        clean_name = clean_name.replace('*', '') # remove splat operators
        if clean_name:
            params.add(clean_name)

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
        # In Python tree-sitter, we look for 'identifier'
        if n.type != "identifier":
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
        fields_in_class=fields,
        field_types=field_type_map,
        vr=vr,
        vw=vw,
    )

def return_type(parser: PythonTreeSitterParser, decl) -> Optional[str]:
    """Return the return type hint of a function, if annotated."""
    rt = decl.child_by_field_name("return_type")
    return parser.text_of(rt).strip() if rt else None


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
        yield cursor.node

        if cursor.goto_first_child():
            continue
        if cursor.goto_next_sibling():
            continue

        while True:
            if not cursor.goto_parent():
                done = True 
                break
            if cursor.goto_next_sibling():
                break