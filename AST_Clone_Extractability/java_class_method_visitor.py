from typing import Iterable, Optional, List, Dict
from java_treesitter_parser import JavaTreeSitterParser
from util_ast import (
    class_name, enclosing_class_name, method_name, collect_modifiers, collect_params, return_type, iter_descendants, methods_directly_under,
)

# Node types in the Java AST that represent "class-like" constructs
# These are the top-level entities we care about visiting.
_CLASS_NODES = {
    "class_declaration",        # e.g., "class MyClass { ... }"
    "interface_declaration",    # e.g., "interface MyInterface { ... }"
    "enum_declaration",         # e.g., "enum MyEnum { ... }"
    "record_declaration",       # e.g., "record MyRecord(...) { ... }"
}


class JavaClassMethodVisitor:
    """
    A base visitor for traversing Java classes and their methods
    using Tree-sitter AST nodes.

    Subclasses are expected to override three "hook" methods:
      - enter_class(class_name, node): called when a class is entered
      - visit_method(class_name, method_info, node): called for each method
      - exit_class(class_name, node): called when leaving a class

    This structure allows developers to focus on *what to do* with
    class/method information, without rewriting the traversal logic.
    """

    def __init__(self, parser: JavaTreeSitterParser):
        """
        Initialize with a JavaTreeSitterParser that provides:
          - parser.root: the root AST node
          - text_of(node): extract source text for a node
          - char_offset(byte_offset): map byte offset → character offset
        """
        self.parser = parser

    def run(self) -> None:
        """
        Main entry point.
        Walk through all AST nodes looking for class-like constructs,
        then visit their directly owned methods.
        """
        # 1) Traverse the AST in depth-first order
        for cls in iter_descendants(self.parser.root):

            # Only process nodes that are classes/interfaces/enums/records
            if cls.type not in _CLASS_NODES:
                continue

            # Extract the class name (may be None if anonymous)
            cname = class_name(self.parser, cls)

            # Hook: allow subclasses to handle "entering" a class
            self.enter_class(cname, cls)

            # 2) For this class, iterate over methods defined *directly* under it.
            #    (Avoids accidentally picking up nested methods from inner classes.)
            for decl in methods_directly_under(cls):
                self._handle_method(decl)

            # Hook: allow subclasses to handle "exiting" a class
            self.exit_class(cname, cls)

    def _handle_method(self, decl) -> None:
        """
        Internal helper.
        Given a method declaration node, extract all relevant info
        (name, modifiers, return type, parameters, position, etc.)
        and forward it to the visit_method hook.
        """
        parser = self.parser

        # Extract method/constructor name, or mark as "<anonymous>" if missing
        name = method_name(parser, decl) or "<anonymous>"

        # Determine what kind of declaration this is
        kind = (
            "compact_constructor" if decl.type == "compact_constructor_declaration"
            else "constructor" if decl.type == "constructor_declaration"
            else "method"
        )

        # Collect modifiers like 'public', 'static', etc.
        mods = collect_modifiers(parser, decl)

        # Determine return type (None if constructor)
        rtype = return_type(parser, decl)

        # Collect parameter list as strings
        params = collect_params(parser, decl)

        # Compute source positions (both offsets and line numbers)
        start_char = parser.char_offset(decl.start_byte)
        end_char = parser.char_offset(decl.end_byte)
        start_line = decl.start_point[0] + 1  # Tree-sitter uses 0-based line numbers
        end_line = decl.end_point[0] + 1

        # Resolve the enclosing class name, to build a qualified method name
        cls_name = enclosing_class_name(parser, decl)
        qualified = f"{cls_name}.{name}" if cls_name else name

        # Package all extracted info into a dictionary for convenience
        info = {
            "name": name,                # Simple method name
            "qualified": qualified,      # Class-qualified name, e.g., MyClass.foo
            "kind": kind,                # "method", "constructor", or "compact_constructor"
            "modifiers": mods,           # List of modifiers
            "return_type": rtype,        # Return type (None for constructors)
            "parameters": params,        # Parameter list
            "start_line": start_line,    # Start line number
            "end_line": end_line,        # End line number
            "start_offset": start_char,  # Start character offset in file
            "end_offset": end_char,      # End character offset in file
        }

        # Hook: let subclass decide what to do with the method info
        self.visit_method(cls_name, info, decl)

    # ---------- hooks ----------
    def enter_class(self, class_name: Optional[str], node):  # override
        pass

    def visit_method(self, class_name: Optional[str], method_info: Dict, node):  # override
        pass

    def exit_class(self, class_name: Optional[str], node):  # override
        pass
