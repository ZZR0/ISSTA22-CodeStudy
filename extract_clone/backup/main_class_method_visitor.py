import argparse
from pathlib import Path
from typing import Optional
import re 

from java_treesitter_parser import JavaTreeSitterParser
from java_class_method_visitor import JavaClassMethodVisitor
from util_ast import iter_descendants


def _sanitize_filename(name: str) -> str:
    # Keep letters, digits, dot, dash, underscore; replace others with '_'
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)

class MethodInfoExtractor(JavaClassMethodVisitor):
    """
    Concrete visitor that prints information about methods found in a Java file.

    This class relies on the base visitor (JavaClassMethodVisitor) to:
      - Traverse the syntax tree and find class-like nodes.
      - For each class, compute method metadata (method_info) and call visit_method.

    Responsibilities here:
      - Count how many classes and methods were visited.
      - Format and print a human-readable method signature.
      - Count method invocations within each method body.
    """

    def __init__(self, parser, extract_source: bool = False, save_dir: Optional[Path] = None, debug: bool = False):
        """
        Store the parser wrapper (JavaTreeSitterParser) and initialize counters.

        Parameters
        ----------
        parser : JavaTreeSitterParser
            Holds the parsed syntax tree and helpers like text_of(), char_offset(), etc.
        """
        super().__init__(parser)
        self.debug = debug
        self._class_count = 0   # incremented in enter_class
        self._method_count = 0  # incremented in visit_method

        self.extract_source = extract_source
        self.save_dir = Path(save_dir) if save_dir is not None else None
        if self.save_dir is not None:
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def enter_class(self, class_name, node):
        """
        Hook called by the base visitor before visiting methods of a class.
        """
        self._class_count += 1

    def visit_method(self, class_name, method_info, node):
        """
        Hook called once per method/constructor directly owned by the current class.

        Parameters
        ----------
        class_name : str | None
            The enclosing class name for the current method.
        method_info : dict
            Method metadata prepared by the base visitor. Expected keys include:
              - "name": simple method name
              - "qualified": "ClassName.methodName"
              - "kind": "method" | "constructor" | "compact_constructor"
              - "modifiers": list of modifiers (e.g., ["public", "static"])
              - "return_type": string or None (constructors have no return type)
              - "parameters": list of parameter strings
              - "start_line": 1-based starting line number
              - "start_offset": character offset of the start in the source
              - "end_offset": character offset of the end in the source
        node : Node
            Tree-sitter node for the method/constructor declaration.

        Behavior
        --------
          walking its subtree and matching nodes of type "method_invocation".
        """
        self._method_count += 1
        print(f"== Enclosing Class: {class_name} ==")

        # Build and print a readable signature (e.g., "void InputSample.foo(String str)")
        sig = self._format_signature(method_info)
        print(f"{sig}  [line {method_info['start_line']}, off {method_info['start_offset']}]")

        # Traverse this method's subtree and count call sites.
        # iter_descendants() yields nodes in pre-order, left-to-right.
        calls = sum(1 for ch in iter_descendants(node) if ch.type == "method_invocation")
        if calls:
            print(f"  calls: {calls}")

        self._extract_method(method_info, node)

    def exit_class(self, class_name, node):
        print()

    def _extract_method(self, method_info: dict, node) -> None:
        source_text = self.parser.text_of(node)
        method_info["source"] = source_text
        qualified = method_info.get("qualified") or method_info.get("name") or "method"
        start_line = method_info.get("start_line")
        end_line = method_info.get("end_line")
        base = _sanitize_filename(f"{qualified}_{start_line}-{end_line}")
        out_path = self.save_dir / f"{base}.java.txt"

        try:
            out_path.write_text(source_text, encoding="utf-8")
            method_info["saved_to"] = str(out_path)
        except Exception as e:
            method_info["saved_to"] = None

    def _format_signature(self, info):
        mods = " ".join(info["modifiers"]) + (" " if info["modifiers"] else "")
        rtype = (info["return_type"] + " ") if (info["return_type"] and info["kind"] == "method") else ""
        params = ", ".join(info["parameters"])
        return f"{mods}{rtype}{info['qualified']}({params})"


def main():
    """
    Entry point for the CLI.

    Steps
    -----
    1) Parse arguments: --java-file points to the Java source to analyze.
    2) Read the file contents and build a JavaTreeSitterParser (parses the AST).
    3) Instantiate the MethodInfoExtractor visitor and run it over the tree.
    4) Print a debug summary of how many classes/methods were visited.
    """
    ap = argparse.ArgumentParser(
        description="Visit all methods in each class and apply a customizable handler."
    )
    ap.add_argument("--java-file", default="input/InputSample.java", help="Path to a Java source file")
    ap.add_argument("--debug", action="store_true", help="Print parser debug info")
    args = ap.parse_args()

    # Load source text (UTF-8) and parse into a syntax tree wrapper.
    src = Path(args.java_file).read_text(encoding="utf-8")
    parser = JavaTreeSitterParser(src)

    # Run the visitor. The base class orchestrates traversal and method_info creation.
    method_visitor = MethodInfoExtractor(parser, extract_source=True, save_dir='output', debug=args.debug)
    method_visitor.run()

    # Final debug summary for quick sanity checks.
    print(f"[DBG] Visited classes: {method_visitor._class_count}, methods: {method_visitor._method_count}")


if __name__ == "__main__":
    main()
