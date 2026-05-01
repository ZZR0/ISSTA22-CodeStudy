from pathlib import Path
from tree_sitter import Language, Parser
import tree_sitter_java as tsjava 
# import warnings

# Suppress specific FutureWarnings raised internally by tree_sitter packages.
# This keeps runtime output clean by hiding non-critical warnings.
# warnings.filterwarnings("ignore", category=FutureWarning, module=r"^tree_sitter(\.|$)")

# Load the Java grammar
JAVA = Language(tsjava.language())


class JavaTreeSitterParser:
    """
    A lightweight wrapper around the Tree-sitter parser for Java.
      - Initialize a parser with Java grammar.
      - Parse raw Java source code into an Abstract Syntax Tree (AST).
    """

    def __init__(self, src_text: str):
        """
        Initialize the parser with Java source code.
        Args:
            src_text (str): Full source code of a Java file as a string.
        """
        self.src_text = src_text
        # Tree-sitter works on byte sequences, not plain strings.
        # Encode the source into UTF-8 so Tree-sitter can parse it correctly.
        self.src_bytes = src_text.encode("utf-8")
        # Create a parser bound to the Java grammar.
        self.parser = Parser(JAVA)
        # Parse the source code (in bytes) into an AST (syntax tree).
        self.tree = self.parser.parse(self.src_bytes)
        # Store the root node of the AST for traversal.
        self.root = self.tree.root_node

    def text_of(self, node) -> str:
        """
        Extract the exact source substring corresponding to a given AST node.
        """
        return self.src_bytes[node.start_byte:node.end_byte].decode("utf-8", "replace")

    def char_offset(self, byte_index: int) -> int:
        """
        Convert a byte index (from Tree-sitter) into a character offset.
        """
        return len(self.src_bytes[:byte_index].decode("utf-8", "ignore"))

    def line_number(self, node) -> int:
        """
        Get the 1-based line number where a node starts in the source code.
        """
        return node.start_point[0] + 1

    # -------------------------------------------------------------------------
    # Convenience constructor (commented out for now):
    # This alternative constructor would allow parsing a file directly
    # without needing to manually read it into a string first.
    #
    # @classmethod
    # def from_file(cls, path: str | Path) -> "JavaTreeSitterParser":
    #     text = Path(path).read_text(encoding="utf-8")
    #     return cls(text)
