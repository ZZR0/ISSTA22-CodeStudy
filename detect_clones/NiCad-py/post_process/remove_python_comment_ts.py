#!/usr/bin/env python3
"""
remove_python_comment_ts.py

Purpose:
  - Remove Python docstrings and comments using Tree-sitter (AST-based).
  - Robustly handle multi-line arguments in __init__ that confuse standard tokenizer.
  - Identify "boilerplate" functions that lack meaningful control flow logic.

Requirements:
  pip install tree-sitter tree-sitter-python
"""

import tree_sitter_python as tspython
from tree_sitter import Language, Parser

# Initialize Tree-sitter for Python
try:
    PY_LANGUAGE = Language(tspython.language())
    parser = Parser(PY_LANGUAGE)
except Exception as e:
    print(f"[ERROR] Tree-sitter initialization failed: {e}")
    print("Please run: pip install tree-sitter tree-sitter-python")
    raise e

def remove_python_comments_ts(source_code: str) -> str:
    """
    Parses code into an AST and surgically removes:
    1. Docstrings (expression statements with string as first child of block)
    2. Comments (native comment nodes)
    
    Returns:
        Cleaned source code string.
    """
    if not source_code:
        return ""

    # Tree-sitter requires bytes
    try:
        source_bytes = bytearray(bytes(source_code, "utf8"))
        tree = parser.parse(source_bytes)
    except Exception:
        # Fallback if encoding fails
        return source_code

    root_node = tree.root_node

    # 1. Query for Docstrings
    # Matches a string that is the immediate first child of a function/class block
    # This works regardless of complex function signatures.
    docstring_query = PY_LANGUAGE.query("""
        (function_definition
            body: (block . (expression_statement (string)) @docstring))
    """)
    
    # 2. Query for Comments
    comment_query = PY_LANGUAGE.query("""
        (comment) @comment
    """)

    # Collect ranges to remove
    ranges_to_remove = []
    
    for node, _ in docstring_query.captures(root_node):
        ranges_to_remove.append((node.start_byte, node.end_byte))
        
    for node, _ in comment_query.captures(root_node):
        ranges_to_remove.append((node.start_byte, node.end_byte))

    # Sort ranges in reverse order to delete safely without shifting offsets
    ranges_to_remove.sort(key=lambda x: x[0], reverse=True)

    for start, end in ranges_to_remove:
        # Simple deletion. 
        # (Optional: Advanced logic could trim adjacent newlines, but usually not strictly necessary for analysis)
        del source_bytes[start:end]

    return source_bytes.decode("utf8")


def is_boilerplate_ts(source_code: str) -> bool:
    """
    Returns True if the function contains NO meaningful control flow or operation logic.
    
    Criteria for "Boilerplate":
    - Contains ONLY assignments, super() calls, return (literals), or pass.
    - Does NOT contain: if, for, while, try, with, or binary operators (+, -, *, etc).
    """
    if not source_code:
        return True

    try:
        tree = parser.parse(bytes(source_code, "utf8"))
    except Exception:
        return False # Assume meaningful if we can't parse

    # Query for "Real Logic" nodes
    logic_query = PY_LANGUAGE.query("""
        (if_statement) @logic
        (for_statement) @logic
        (while_statement) @logic
        (try_statement) @logic
        (with_statement) @logic
        (binary_operator) @logic
        (call 
            function: (attribute) @attr 
            (#not-eq? @attr "super")) @logic_call ; Calls other than super()
    """)
    
    captures = logic_query.captures(tree.root_node)
    
    # If 0 captures, it implies the code is purely structural (assignments/init).
    return len(captures) == 0