from typing import Iterable, Optional, List
from java_treesitter_parser import JavaTreeSitterParser

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
