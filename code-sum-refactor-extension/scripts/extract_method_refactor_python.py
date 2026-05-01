from __future__ import annotations

import argparse
import json
import keyword
import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from python_treesitter_parser import PythonTreeSitterParser
from util_ast_python import enclosing_class_node, iter_descendants


# ============================================================
# Data model
# ============================================================


@dataclass(frozen=True)
class CloneVariable:
    name: str
    type_name: str


@dataclass(frozen=True)
class EnclosingFunction:
    qualified_name: str
    fun_range: str
    fun_nlines: int
    func_code: str


@dataclass(frozen=True)
class CloneSource:
    file: str
    clone_range: str
    nlines: int
    code: str
    func_id: str
    in_vars: List[CloneVariable]
    out_vars: List[CloneVariable]
    defbefore_vars: List[str]
    enclosing_function: EnclosingFunction
    extracted_signature_hint: Optional[str] = None
    return_type_hint: Optional[str] = None
    ground_truth_after_vscode_ref: Optional[dict] = None


@dataclass(frozen=True)
class CloneClassCase:
    classid: str
    nclones: int
    project: str
    same_file: int
    actual_label: int
    clone_predict: int
    clone_predict_after_adapt: int
    refactorable: int
    inspection_case: str
    sources: List[CloneSource]


@dataclass(frozen=True)
class LineSpan:
    start_line: int
    end_line: int


@dataclass(frozen=True)
class ExtractionSignature:
    method_name: str
    is_static: bool
    parameters: List[Tuple[str, str]]
    out_variables: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class CloneRewrite:
    func_id: str
    file: str
    clone_range: str
    replacement_code: str


@dataclass(frozen=True)
class RewriteArtifact:
    file: str
    output_path: str
    inserted_extracted_method: bool
    rewritten_source: str


@dataclass(frozen=True)
class RefactorResult:
    extracted_method_code: str
    extracted_signature: ExtractionSignature
    replacements: List[CloneRewrite]
    updated_files: List[RewriteArtifact]
    diagnostics: List[str]


# ============================================================
# Generic helpers
# ============================================================


RANGE_RE = re.compile(r"^(\d+)-(\d+)$")

METHOD_NODE_TYPES = {"function_definition"}
CLASS_NODE_TYPES = {"class_definition"}


def parse_range(range_text: str) -> LineSpan:
    m = RANGE_RE.match(range_text.strip())
    if not m:
        raise ValueError(f"Invalid range: {range_text}")
    start_line, end_line = int(m.group(1)), int(m.group(2))
    if start_line > end_line:
        raise ValueError(f"Invalid line range order: {range_text}")
    return LineSpan(start_line, end_line)


def line_offsets(text: str) -> List[int]:
    offsets = [0]
    running = 0
    for line in text.splitlines(keepends=True):
        running += len(line)
        offsets.append(running)
    return offsets


def line_span_to_offsets(text: str, span: LineSpan) -> Tuple[int, int]:
    offsets = line_offsets(text)
    if span.end_line > len(offsets) - 1:
        raise ValueError(f"Range {span.start_line}-{span.end_line} exceeds file length")
    return offsets[span.start_line - 1], offsets[span.end_line]


def indent_block(text: str, indent: str) -> str:
    return "\n".join((indent + line if line.strip() else line) for line in text.splitlines())


def apply_replacements(source: str, replacements: Sequence[Tuple[Tuple[int, int], str]]) -> str:
    out = source
    for (start_offset, end_offset), replacement in sorted(replacements, key=lambda x: x[0][0], reverse=True):
        out = out[:start_offset] + replacement + out[end_offset:]
    return out


def sanitize_identifier(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", name)
    if not cleaned:
        cleaned = "extracted"
    if cleaned[0].isdigit():
        cleaned = f"extracted_{cleaned}"
    if keyword.iskeyword(cleaned):
        cleaned += "_"
    return cleaned


def choose_unique_method_name(classid: str) -> str:
    return "extracted"


def normalize_type(type_name: str) -> str:
    t = (type_name or "Any").strip()
    return t if t else "Any"


def infer_is_static(func_code: str) -> bool:
    lines = [line.strip() for line in func_code.splitlines() if line.strip()]
    for i, line in enumerate(lines[:3]):
        if line.startswith("@staticmethod"):
            return True
        if line.startswith("def "):
            break
    return False


def has_self_parameter(func_code: str) -> bool:
    m = re.search(r"def\s+\w+\s*\((.*?)\)\s*:", func_code, flags=re.DOTALL)
    if not m:
        return False
    params_blob = m.group(1).strip()
    if not params_blob:
        return False
    first = params_blob.split(",")[0].strip()
    first = first.split(":")[0].split("=")[0].strip()
    return first == "self"


# ============================================================
# Input loading
# ============================================================


class CloneCaseLoader:
    @staticmethod
    def _parse_variables(entry: dict, key_name: str) -> List[CloneVariable]:
        if not entry:
            return []
            
        names = []
        if isinstance(entry, dict):
            names = entry.get(key_name, [])
            if not names:
                for k, v in entry.items():
                    if isinstance(v, list) and v:
                        names = v
                        break
        elif isinstance(entry, list):
            names = entry
        elif isinstance(entry, str):
            names = [entry]
            
        result = []
        for name in names:
            if isinstance(name, str):
                result.append(CloneVariable(name=name, type_name="Any"))
        return result

    @classmethod
    def from_json(cls, payload: dict) -> CloneClassCase:
        sources: List[CloneSource] = []
        for source in payload["sources"]:
            enclosing = source["enclosing_function"]
            sources.append(
                CloneSource(
                    file=source["file"],
                    clone_range=source["range"],
                    nlines=int(source.get("nlines", 0)),
                    code=source["code"],
                    func_id=source["func_id"],
                    in_vars=cls._parse_variables(source.get("In", {}), "In(i)"),
                    out_vars=cls._parse_variables(source.get("Out", {}), "Out(i)"),
                    defbefore_vars=source.get("In", {}).get("InType", {}).get("Defbefore(i)", []),
                    enclosing_function=EnclosingFunction(
                        qualified_name=enclosing["qualified_name"],
                        fun_range=enclosing["fun_range"],
                        fun_nlines=int(enclosing.get("fun_nlines", 0)),
                        func_code=enclosing["func_code"],
                    ),
                    extracted_signature_hint=source.get("Extracted Signature"),
                    return_type_hint=source.get("ReturnType"),
                    ground_truth_after_vscode_ref=source.get("ground_truth_after_VSCode_ref"),
                )
            )
        return CloneClassCase(
            classid=payload["classid"],
            nclones=int(payload["nclones"]),
            project=payload.get("project", ""),
            same_file=int(payload.get("same_file", 0)),
            actual_label=int(payload.get("actual_label", 0)),
            clone_predict=int(payload.get("clone_predict", 0)),
            clone_predict_after_adapt=int(payload.get("clone_predict_after_adapt", 0)),
            refactorable=int(payload.get("Refactorable", 0)),
            inspection_case=payload.get("inspection_case", ""),
            sources=sources,
        )


# ============================================================
# AST location services
# ============================================================


class CloneLocator:
    def __init__(self, source_root: Path):
        self.source_root = source_root

    def absolute_file(self, relative_file: str) -> Path:
        return self.source_root / relative_file

    def load_source(self, relative_file: str) -> str:
        return self.absolute_file(relative_file).read_text(encoding="utf-8", errors="replace")

    def find_method_node(self, source: CloneSource):
        file_path = self.absolute_file(source.file)
        parser = PythonTreeSitterParser(file_path.read_text(encoding="utf-8", errors="replace"))
        method_span = parse_range(source.enclosing_function.fun_range)

        for node in iter_descendants(parser.root):
            if node.type not in METHOD_NODE_TYPES:
                continue
            start_line = node.start_point[0] + 1
            if start_line != method_span.start_line:
                continue

            name_node = node.child_by_field_name("name")
            qname = parser.text_of(name_node) if name_node else ""
            if source.enclosing_function.qualified_name.endswith(qname):
                return parser, node

        raise KeyError(f"Method not found for {source.enclosing_function.qualified_name} @ line {method_span.start_line}")

    def find_enclosing_class_node(self, source: CloneSource):
        parser, method_node = self.find_method_node(source)
        class_node = enclosing_class_node(method_node)
        return parser, class_node

    def clone_offsets(self, file_text: str, clone_range: str) -> Tuple[int, int]:
        return line_span_to_offsets(file_text, parse_range(clone_range))


# ============================================================
# Signature synthesis
# ============================================================


class SignatureSynthesizer:
    def __init__(self, case: CloneClassCase):
        self.case = case

    def synthesize(self) -> ExtractionSignature:
        if not self.case.sources:
            raise ValueError("Empty clone class")

        first_src = self.case.sources[0]
        is_static = infer_is_static(first_src.enclosing_function.func_code)

        parameters: List[Tuple[str, str]] = []
        for var in first_src.in_vars:
            if not is_static and var.name == "self":
                continue
            parameters.append((normalize_type(var.type_name), sanitize_identifier(var.name)))

        out_vars = [sanitize_identifier(v.name) for v in first_src.out_vars]

        return ExtractionSignature(
            method_name=choose_unique_method_name(self.case.classid),
            is_static=is_static,
            parameters=parameters,
            out_variables=out_vars,
        )


# ============================================================
# Clone normalization and method generation
# ============================================================


class CloneNormalizer:
    def __init__(self, case: CloneClassCase, locator: CloneLocator):
        self.case = case
        self.locator = locator

    def normalized_body(self) -> str:
        first = self.case.sources[0]
        file_text = self.locator.load_source(first.file)
        span = parse_range(first.clone_range)
        
        lines = file_text.splitlines(keepends=True)
        raw_lines = lines[span.start_line - 1 : span.end_line]
        
        raw_text = "".join(raw_lines).expandtabs(4)
        lines_text = raw_text.splitlines()
        
        first_indent = ""
        for line in lines_text:
            if line.strip():
                first_indent = re.match(r"^[ \t]*", line).group(0)
                break
        
        dedented = []
        for line in lines_text:
            if line.startswith(first_indent):
                dedented.append(line[len(first_indent):])
            elif not line.strip():
                dedented.append("")
            else:
                dedented.append(line)
                
        return "\n".join(dedented).strip("\n")


class PythonMethodRenderer:
    def __init__(self, signature: ExtractionSignature, normalized_body: str, in_class_scope: bool):
        self.signature = signature
        self.normalized_body = normalized_body
        self.in_class_scope = in_class_scope

    def render(self) -> str:
        lines: List[str] = []

        if self.in_class_scope and self.signature.is_static:
            lines.append("@staticmethod")

        params = [n for _, n in self.signature.parameters]
        if self.in_class_scope and not self.signature.is_static:
            params = ["self"] + params

        header = f"def {self.signature.method_name}({', '.join(params)}):"
        lines.append(header)

        body = self.normalized_body.strip()
        if body:
            lines.append(indent_block(body, "    "))
        else:
            lines.append("    pass")
            
        if self.signature.out_variables:
            returns = ", ".join(self.signature.out_variables)
            # Smart Check: Only append return __return__ if the code doesn't natively return already
            if not ("return " in body and "__return__" in self.signature.out_variables):
                lines.append(indent_block(f"return {returns}", "    "))

        return "\n".join(lines)


# ============================================================
# Rewriting engine
# ============================================================


class Rewriter:
    def __init__(
        self,
        source_root: Path,
        case: CloneClassCase,
        signature: ExtractionSignature,
        in_class_scope: bool
    ):
        self.source_root = source_root
        self.case = case
        self.signature = signature
        self.in_class_scope = in_class_scope
        self.locator = CloneLocator(source_root)

    def _call_args_for_source(self, source: CloneSource) -> List[str]:
        source_arg_map = {sanitize_identifier(v.name): v.name for v in source.in_vars}
        args: List[str] = []
        for _type_name, param_name in self.signature.parameters:
            args.append(source_arg_map.get(param_name, param_name))
        return args

    def _build_callsite(self, source: CloneSource, file_text: str) -> str:
        args = ", ".join(self._call_args_for_source(source))
        
        prefix = ""
        source_out_names = [v.name for v in source.out_vars]
        if not source_out_names and self.signature.out_variables:
            source_out_names = self.signature.out_variables
            
        if source_out_names:
            prefix = f"{', '.join(source_out_names)} = "
            
        caller = "self." if (self.in_class_scope and not self.signature.is_static) else ""
        call_line = f"{prefix}{caller}{self.signature.method_name}({args})"

        span = parse_range(source.clone_range)
        lines = file_text.splitlines()
        first_line = lines[span.start_line - 1].expandtabs(4)
        base_indent_match = re.match(r"^[ \t]*", first_line)
        base_indent = base_indent_match.group(0) if base_indent_match else ""

        return f"{base_indent}{call_line}\n"

    def rewrite(self, extracted_method_code: str) -> Tuple[List[CloneRewrite], List[RewriteArtifact]]:
        by_file: Dict[str, List[CloneSource]] = {}
        for src in self.case.sources:
            by_file.setdefault(src.file, []).append(src)

        replacements_out: List[CloneRewrite] = []
        artifacts: List[RewriteArtifact] = []
        first_file = self.case.sources[0].file if self.case.sources else None

        for relative_file, clone_sources in by_file.items():
            file_text = self.locator.load_source(relative_file)
            
            first_source = clone_sources[0]
            parser, class_node = self.locator.find_enclosing_class_node(first_source)
            _, func_node = self.locator.find_method_node(first_source)
            
            target_node = class_node if class_node else func_node
            insert_at_original = target_node.end_byte

            # Safety check: If TreeSitter choked on bad syntax, it truncates the AST.
            # We force it to the end of the file so we don't inject inside a try/catch block.
            max_clone_end = max(self.locator.clone_offsets(file_text, src.clone_range)[1] for src in clone_sources)
            if insert_at_original < max_clone_end:
                insert_at_original = len(file_text)

            replacements: List[Tuple[Tuple[int, int], str]] = []
            for src in clone_sources:
                start_offset, end_offset = self.locator.clone_offsets(file_text, src.clone_range)
                call_site = self._build_callsite(src, file_text)
                replacements_out.append(
                    CloneRewrite(
                        func_id=src.func_id,
                        file=src.file,
                        clone_range=src.clone_range,
                        replacement_code=call_site,
                    )
                )
                replacements.append(((start_offset, end_offset), call_site))

            # Apply call site replacements first
            rewritten = apply_replacements(file_text, replacements)

            # Mathematical Shift: Calculate where insert_at shifted to
            inserted = False
            if relative_file == first_file:
                delta = 0
                for (start, end), rep_str in replacements:
                    if end <= insert_at_original:
                        delta += len(rep_str) - (end - start)
                    elif start < insert_at_original < end:
                        delta += len(rep_str) - (insert_at_original - start)

                new_insert_at = insert_at_original + delta

                lines = file_text.splitlines(keepends=True)
                start_line_idx = target_node.start_point[0]
                base_indent = re.match(r"^[ \t]*", lines[start_line_idx].expandtabs(4)).group(0)
                method_indent = base_indent + "    " if class_node else base_indent
                indented_method = "\n\n" + indent_block(extracted_method_code, method_indent) + "\n"

                # Apply isolated method insertion
                rewritten = rewritten[:new_insert_at] + indented_method + rewritten[new_insert_at:]
                inserted = True

            artifacts.append(
                RewriteArtifact(
                    file=relative_file,
                    output_path="",
                    inserted_extracted_method=inserted,
                    rewritten_source=rewritten,
                )
            )

        return replacements_out, artifacts


# ============================================================
# Orchestrator
# ============================================================


class ExtractMethodRefactorer:
    def __init__(self, source_root: Path):
        self.source_root = source_root
        self.locator = CloneLocator(source_root)

    def refactor(self, case: CloneClassCase) -> RefactorResult:
        first_func_code = case.sources[0].enclosing_function.func_code
        in_class_scope = has_self_parameter(first_func_code) or ("." in case.sources[0].enclosing_function.qualified_name)

        signature = SignatureSynthesizer(case).synthesize()
        
        normalized_body = CloneNormalizer(case, self.locator).normalized_body()

        extracted_method_code = PythonMethodRenderer(
            signature, normalized_body, in_class_scope=in_class_scope
        ).render()

        replacements, updated_files = Rewriter(
            self.source_root, case, signature, in_class_scope
        ).rewrite(extracted_method_code)

        return RefactorResult(
            extracted_method_code=extracted_method_code,
            extracted_signature=signature,
            replacements=replacements,
            updated_files=updated_files,
            diagnostics=[],
        )


# ============================================================
# Output formatting
# ============================================================


class ResultFormatter:
    @staticmethod
    def to_json(case: CloneClassCase, result: RefactorResult, concise: bool = True) -> dict:
        payload = {
            "classid": case.classid,
            "extracted_method": {
                "method_name": result.extracted_signature.method_name,
                "is_static": result.extracted_signature.is_static,
                "parameters": [f"{t} {n}" for t, n in result.extracted_signature.parameters],
                "code": result.extracted_method_code,
            },
            "sources": [
                {
                    "func_id": src.func_id,
                    "file": src.file,
                    "range": src.clone_range,
                    "nlines": src.nlines,
                    "code": src.code,
                    "enclosing_function": {
                        "qualified_name": src.enclosing_function.qualified_name,
                        "fun_range": src.enclosing_function.fun_range,
                        "fun_nlines": src.enclosing_function.fun_nlines,
                        "func_code": src.enclosing_function.func_code,
                    },
                    "replacement_code": next(
                        (repl.replacement_code for repl in result.replacements if repl.func_id == src.func_id),
                        "",
                    ),
                    "ground_truth_after_vscode_ref": src.ground_truth_after_vscode_ref,
                }
                for src in case.sources
            ],
            "updated_files": [
                {
                    "file": artifact.file,
                    "inserted_extracted_method": artifact.inserted_extracted_method,
                    "rewritten_file_path": artifact.output_path,
                }
                for artifact in result.updated_files
            ],
        }

        if not concise:
            payload.update(
                {
                    "project": case.project,
                    "inspection_case": case.inspection_case,
                    "refactoring_type": "extract_method",
                    "nclones": case.nclones,
                    "same_file": case.same_file,
                    "clone_predict": case.clone_predict,
                    "clone_predict_after_adapt": case.clone_predict_after_adapt,
                    "Refactorable": case.refactorable,
                    "diagnostics": result.diagnostics,
                }
            )

        return payload


# ============================================================
# Direct rewrite (tree-sitter driven, drop-line aware)
# ============================================================


def _find_func_node_at_line(parser: PythonTreeSitterParser, insert_line: int):
    """Return the innermost function_definition node that contains *insert_line* (1-indexed)."""
    best = None
    for node in iter_descendants(parser.root):
        if node.type != "function_definition":
            continue
        if node.start_point[0] + 1 <= insert_line <= node.end_point[0] + 1:
            if best is None or node.start_byte > best.start_byte:
                best = node
    return best


def _method_indent_for_insert(file_text: str, func_node, class_node) -> str:
    """Compute the indentation string for the extracted method being inserted."""
    target = class_node if class_node else func_node
    if target is None:
        return ""
    lines = file_text.splitlines(keepends=True)
    idx = target.start_point[0]
    if idx >= len(lines):
        return ""
    base = re.match(r"^[ \t]*", lines[idx].expandtabs(4)).group(0)
    return base + "    " if class_node else base


def _signature_to_call_expr(signature_str: str) -> str:
    """Convert ``'def extracted(a, b, c):'`` to ``'extracted(a, b, c)'``."""
    m = re.match(r"def\s+(\w+)\s*\((.*?)\)\s*(?:->.*?)?:", signature_str.strip(), re.DOTALL)
    if not m:
        return "extracted()"
    name = m.group(1)
    args = [
        p.strip().split(":")[0].split("=")[0].strip().lstrip("*")
        for p in m.group(2).split(",")
    ]
    args = [a for a in args if a and a != "self"]
    return f"{name}({', '.join(args)})"


def direct_rewrite(
    file_text: str,
    signature_str: str,
    body: str,
    call_ranges: List[str],
    insert_line: int,
    replacement_codes: Optional[List[str]] = None,
) -> str:
    """Extract-method refactoring using tree-sitter, driven by the user's drop location.

    Uses tree-sitter to:
      1. Locate the enclosing function/class at *insert_line*.
      2. Determine the correct insertion byte (end of that scope).
      3. Compute proper indentation from the AST context.

    Args:
        file_text:         Full source of the file to rewrite.
        signature_str:     ``'def extracted(...):'`` line of the new method.
        body:              The clone code that forms the new method's body.
        call_ranges:       ``['start-end', ...]`` ranges to replace with call sites.
        insert_line:       1-indexed line where the user dropped (insertion hint).
        replacement_codes: Pre-computed call strings per range; derived from *signature_str*
                           when absent.

    Returns:
        The fully rewritten source as a string.
    """
    parser = PythonTreeSitterParser(file_text)

    # 1. Use tree-sitter to locate the enclosing function and class at the drop line.
    func_node = _find_func_node_at_line(parser, insert_line)
    class_node = enclosing_class_node(func_node) if func_node else None
    target_node = class_node if class_node else func_node

    if target_node is not None:
        insert_at = target_node.end_byte
    else:
        # Drop line is outside all functions — insert at start of that line.
        offsets = line_offsets(file_text)
        idx = min(insert_line - 1, len(offsets) - 1)
        insert_at = offsets[idx]

    # Safety: if tree-sitter truncated (bad syntax), fall back to EOF.
    max_clone_end = 0
    for r in call_ranges:
        try:
            _, e = line_span_to_offsets(file_text, parse_range(r.strip()))
            if e > max_clone_end:
                max_clone_end = e
        except ValueError:
            pass
    if insert_at < max_clone_end:
        insert_at = len(file_text)

    # 2. Build extracted method code from signature + body.
    sig = signature_str.strip()
    if not sig.endswith(":"):
        sig += ":"
    normalized_body = textwrap.dedent(body.strip("\n"))
    method_code = sig + "\n" + indent_block(normalized_body, "    ")

    # 3. Build call-site replacements.
    replacements: List[Tuple[Tuple[int, int], str]] = []
    for i, range_str in enumerate(call_ranges):
        span = parse_range(range_str.strip())
        start_off, end_off = line_span_to_offsets(file_text, span)

        if replacement_codes and i < len(replacement_codes) and replacement_codes[i]:
            call_text = replacement_codes[i]
        else:
            lines = file_text.splitlines()
            first_line = lines[span.start_line - 1].expandtabs(4)
            base_indent = re.match(r"^[ \t]*", first_line).group(0)
            call_text = f"{base_indent}{_signature_to_call_expr(sig)}\n"

        replacements.append(((start_off, end_off), call_text))

    # 4. Apply call-site replacements (reverse offset order avoids index drift).
    rewritten = apply_replacements(file_text, replacements)

    # 5. Shift insertion point to account for replacements that precede it.
    delta = 0
    for (start, end), rep_str in sorted(replacements, key=lambda x: x[0][0]):
        if end <= insert_at:
            delta += len(rep_str) - (end - start)
    new_insert_at = insert_at + delta

    # 6. Determine indentation for the extracted method from the AST context.
    method_indent = _method_indent_for_insert(file_text, func_node, class_node)
    indented_method = "\n\n" + indent_block(method_code, method_indent) + "\n"

    # 7. Splice the extracted method into the rewritten source.
    return rewritten[:new_insert_at] + indented_method + rewritten[new_insert_at:]


def _main_direct() -> None:
    """Entry point for the ``direct`` subcommand.

    Reads the function body from **stdin** and all other parameters from CLI flags.
    Prints a JSON object ``{"rewritten_source": "..."}`` to stdout.
    """
    import sys as _sys

    ap = argparse.ArgumentParser(
        prog="extract_method_refactor_python.py direct",
        description="Extract-method: tree-sitter driven, given signature, body (stdin), and drop line.",
    )
    ap.add_argument("--file", required=True,
                    help="Path to the Python source file (absolute or relative to --source-root).")
    ap.add_argument("--source-root", type=Path, default=Path("."),
                    help="Root directory; used only when --file is relative.")
    ap.add_argument("--signature", required=True,
                    help="Signature line of the extracted method, e.g. 'def extracted(a, b):'.")
    ap.add_argument("--call-ranges", required=True,
                    help="Comma-separated line ranges to replace with calls, e.g. '412-425,355-368'.")
    ap.add_argument("--insert-line", type=int, required=True,
                    help="1-indexed line where the user dropped (tree-sitter uses it to find insertion point).")
    ap.add_argument("--replacement-codes", default="",
                    help="JSON array of pre-computed call strings per range. Derived automatically if omitted.")
    args = ap.parse_args(_sys.argv[2:])

    file_path = Path(args.file)
    if not file_path.is_absolute():
        file_path = args.source_root / file_path
    file_text = file_path.read_text(encoding="utf-8", errors="replace")

    body = _sys.stdin.read()

    call_ranges = [r.strip() for r in args.call_ranges.split(",") if r.strip()]
    replacement_codes: Optional[List[str]] = json.loads(args.replacement_codes) if args.replacement_codes.strip() else None

    rewritten = direct_rewrite(
        file_text=file_text,
        signature_str=args.signature,
        body=body,
        call_ranges=call_ranges,
        insert_line=args.insert_line,
        replacement_codes=replacement_codes,
    )

    print(json.dumps({"rewritten_source": rewritten}))


# ============================================================
# CLI
# ============================================================


def main() -> None:
    import sys as _sys
    if len(_sys.argv) > 1 and _sys.argv[1] == "direct":
        _main_direct()
        return

    parser = argparse.ArgumentParser(description="Extract Method refactoring for Python clone classes.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("refactor_out"))
    parser.add_argument("--full-report", action="store_true")
    parser.add_argument("--merged-report-name", type=str, default="all_refactor_results.json")
    args = parser.parse_args()

    text = args.input.read_text(encoding="utf-8").strip()

    if text.startswith("[") and text.endswith("]"):
        try:
            payloads = json.loads(text)
        except json.JSONDecodeError:
            payloads = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        try:
            payload = json.loads(text)
            payloads = [payload] if isinstance(payload, dict) else payload
        except json.JSONDecodeError:
            payloads = [json.loads(line) for line in text.splitlines() if line.strip()]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_reports = []
    merged_reports = []

    for idx, case_payload in enumerate(payloads, start=1):
        case = CloneCaseLoader.from_json(case_payload)
        refactorer = ExtractMethodRefactorer(args.source_root)
        result = refactorer.refactor(case)

        case_output_dir = args.output_dir / case.classid
        case_output_dir.mkdir(parents=True, exist_ok=True)

        materialized_artifacts: List[RewriteArtifact] = []
        for artifact in result.updated_files:
            relative_path = Path(artifact.file)
            out_path = case_output_dir / relative_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(artifact.rewritten_source, encoding="utf-8")

            materialized_artifacts.append(
                RewriteArtifact(
                    file=artifact.file,
                    output_path=str(out_path),
                    inserted_extracted_method=artifact.inserted_extracted_method,
                    rewritten_source=artifact.rewritten_source,
                )
            )

        final_result = RefactorResult(
            extracted_method_code=result.extracted_method_code,
            extracted_signature=result.extracted_signature,
            replacements=result.replacements,
            updated_files=materialized_artifacts,
            diagnostics=result.diagnostics,
        )

        report = ResultFormatter.to_json(case, final_result, concise=not args.full_report)
        merged_reports.append(report)

        summary_reports.append(
            {
                "classid": case.classid,
                "rewritten_files": [a.output_path for a in materialized_artifacts],
                "diagnostics": final_result.diagnostics,
            }
        )

        print(f"[{idx}/{len(payloads)}] Finished: {case.classid}")
        for artifact in materialized_artifacts:
            print(f"Rewritten file written to: {artifact.output_path}")

    merged_report_path = args.output_dir / args.merged_report_name
    merged_report_path.write_text(json.dumps(merged_reports, indent=2), encoding="utf-8")

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary_reports, indent=2), encoding="utf-8")

    print(f"Merged report written to: {merged_report_path}")
    print(f"Summary written to: {summary_path}")


if __name__ == "__main__":
    main()