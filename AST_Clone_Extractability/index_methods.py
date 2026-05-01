# AST_Clone_Extractability/index_methods.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

from java_treesitter_parser import JavaTreeSitterParser
from java_class_method_visitor import JavaClassMethodVisitor


@dataclass(frozen=True)
class MethodRecord:
    class_name: Optional[str]
    method_info: dict
    node: object  # tree_sitter.Node


class _MethodIndexer(JavaClassMethodVisitor):
    def __init__(self, parser: JavaTreeSitterParser):
        super().__init__(parser)
        self.by_qualified: Dict[str, MethodRecord] = {}

    def visit_method(self, class_name, method_info, node):
        q = method_info.get("qualified")
        # start_line = method_info.get("start_line")
        # end_line = method_info.get("end_line")
        # print(f"\nstart line: {start_line}, end line: {end_line}")
        if q:
            self.by_qualified[q] = MethodRecord(class_name=class_name, method_info=method_info, node=node)


class FileMethodIndex:
    """
    Cache of parsed Java files and their method nodes indexed by qualified name.
    """
    def __init__(self):
        self._cache: Dict[str, Tuple[JavaTreeSitterParser, Dict[str, MethodRecord]]] = {}

    def get(self, file_path: str) -> Tuple[JavaTreeSitterParser, Dict[str, MethodRecord]]:
        if file_path in self._cache:
            return self._cache[file_path]

        src = Path(file_path).read_text(encoding="utf-8", errors="replace")
        parser = JavaTreeSitterParser(src)
        idx = _MethodIndexer(parser)
        idx.run()

        self._cache[file_path] = (parser, idx.by_qualified)
        return self._cache[file_path]
