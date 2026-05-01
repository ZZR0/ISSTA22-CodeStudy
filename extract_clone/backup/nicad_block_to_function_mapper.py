#!/usr/bin/env python3
"""
NiCad block-to-function mapper (Java)

Goal
----
Annotate NiCad clone JSONL records with enclosing function metadata, and (optionally)
the enclosing method code.

Why this version works better
-----------------------------
1) First tries Tree-sitter via nearest_method.find_fully_qualified_method().
2) If Tree-sitter returns no methods (common if parser init fails or file is pathological),
   falls back to a lightweight text-based method boundary heuristic.
3) For non-function clones, finds the *smallest enclosing* method when possible.
4) Adds:
   - function.{qualified_name, fun_range, fun_nlines, func, fun_code, clone_ratio}
   - is_function_clone
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

from nearest_method import find_fully_qualified_method

# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------


def parse_range(range_str: str) -> Tuple[int, int]:
    """Parse a 'start-end' line range string into (start, end) inclusive."""
    start, end = range_str.split("-", 1)
    return int(start), int(end)


def normalize_relative_path(rel_path: Union[str, Path]) -> Path:
    """
    Normalize a NiCad-relative path.

    Your NiCad JSONL contains file paths like:
        systems/ant-java/src/main/...

    With --source-root pointing to your project root (that contains 'systems/'),
    we should keep this as-is.
    """
    return Path(rel_path)


def resolve_source_path(rel_path: Union[str, Path], roots: Sequence[Path]) -> Tuple[Path, Path]:
    """
    Resolve a relative path against a list of source roots.

    Returns:
        (root, full_path)
    """
    rel = normalize_relative_path(rel_path)
    for root in roots:
        candidate = root / rel
        if candidate.is_file():
            return root, candidate
    raise FileNotFoundError(f"Missing source file under roots {roots}: {rel_path}")


def ranges_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    """Return True if two inclusive ranges [a_start, a_end] and [b_start, b_end] overlap."""
    return not (a_end < b_start or b_end < a_start)


def compute_clone_ratio(sub_lines: int, fun_span: Optional[int]) -> Optional[float]:
    if not sub_lines or not fun_span or fun_span <= 0:
        return None
    return round(min(sub_lines / fun_span, 1.0), 2)


def _read_file_lines(full_path: Path) -> List[str]:
    try:
        text = full_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        logging.warning("Non-UTF8 source file, decoding with replacement: %s", full_path)
        text = full_path.read_text(encoding="utf-8", errors="replace")
    return text.splitlines(keepends=True)


def _slice_lines(lines: List[str], start_line: int, end_line: int) -> str:
    """1-based inclusive line slicing."""
    if start_line <= 0 or end_line <= 0 or end_line < start_line:
        return ""
    return "".join(lines[start_line - 1 : end_line])


# ---------------------------------------------------------------------------
# Choose enclosing method from tree-sitter results
# ---------------------------------------------------------------------------


def choose_enclosing_method(
    methods: Iterable[Mapping[str, Any]],
    clone_start: int,
    clone_end: int,
) -> Tuple[Optional[Mapping[str, Any]], bool]:
    """
    Pick the best method for this clone range.

    Returns:
      (chosen_method, is_function_clone)

    Strategy:
      1) If any method exactly matches clone range => function clone.
      2) Else choose the *smallest* method that fully contains the clone range (enclosing).
      3) Else fall back to an overlapping method with the smallest span.
    """
    ms: List[Mapping[str, Any]] = []
    for m in methods:
        try:
            m_start = int(m.get("start_line", 0))
            m_end = int(m.get("end_line", 0))
        except Exception:
            continue
        if m_start <= 0 or m_end <= 0:
            continue
        ms.append(m)

    if not ms:
        return None, False

    # 1) exact match => function clone
    for m in ms:
        if int(m["start_line"]) == clone_start and int(m["end_line"]) == clone_end:
            return m, True

    # 2) enclosing candidates (prefer smallest span)
    enclosing = [
        m for m in ms if int(m["start_line"]) <= clone_start and int(m["end_line"]) >= clone_end
    ]
    if enclosing:
        enclosing.sort(
            key=lambda m: (
                int(m.get("span") or (int(m["end_line"]) - int(m["start_line"]) + 1)),
                int(m["start_line"]),
            )
        )
        return enclosing[0], False

    # 3) overlapping fallback (prefer smallest span)
    overlapping = [
        m
        for m in ms
        if ranges_overlap(int(m["start_line"]), int(m["end_line"]), clone_start, clone_end)
    ]
    if overlapping:
        overlapping.sort(
            key=lambda m: (
                int(m.get("span") or (int(m["end_line"]) - int(m["start_line"]) + 1)),
                int(m["start_line"]),
            )
        )
        return overlapping[0], False

    return None, False


# ---------------------------------------------------------------------------
# Fallback method boundary heuristic (no tree-sitter)
# ---------------------------------------------------------------------------

# A pragmatic method header detector: good enough for most Java in Ant-like codebases.
# Requires "{" on the same line as the signature (common in many projects).
METHOD_HEADER_RE = re.compile(
    r"""^\s*
        (?:@\w+(?:\([^)]*\))?\s*)*                                   # annotations
        (?:(?:public|protected|private|static|final|abstract|native|
            synchronized|strictfp)\s+)*                               # modifiers
        (?:<[^>]+>\s+)?                                               # generics
        (?:[\w\.\[\]<>]+\s+)+                                         # return type(s)
        (?P<name>[A-Za-z_]\w*)\s*                                     # method name
        \((?P<params>[^)]*)\)\s*                                      # params
        (?:throws\s+[^{]+)?                                           # throws
        \{                                                            # body starts
        \s*$
    """,
    re.VERBOSE,
)

CONTROL_STMT_RE = re.compile(r"^\s*(if|for|while|switch|catch|do|try)\b")


def _find_package(lines: List[str]) -> Optional[str]:
    for ln in lines[:400]:
        m = re.match(r"^\s*package\s+([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)\s*;", ln)
        if m:
            return m.group(1)
    return None


def _find_outer_class(lines: List[str], upto_line: int) -> Optional[str]:
    class_re = re.compile(
        r"^\s*(?:public|protected|private)?\s*(?:final|abstract)?\s*"
        r"(class|interface|enum|record)\s+([A-Za-z_]\w*)\b"
    )
    last = None
    for ln in lines[: max(1, upto_line)]:
        m = class_re.match(ln)
        if m:
            last = m.group(2)
    return last


def _brace_balance_delta(line: str) -> int:
    # Remove strings/chars roughly so braces in literals don't ruin balance
    line = re.sub(r'"(?:\\.|[^"\\])*"', '""', line)
    line = re.sub(r"'(?:\\.|[^'\\])*'", "''", line)
    return line.count("{") - line.count("}")


def _fallback_enclosing_method(
    lines: List[str],
    clone_start: int,
    clone_end: int,
) -> Optional[Dict[str, Any]]:
    """
    Heuristic fallback when tree-sitter fails:
      - scan upward from clone_start to find a probable method header line
      - then scan forward counting braces to find method end
    """
    header_idx = None
    name = None
    params = ""
    for i in range(clone_start, 0, -1):
        ln = lines[i - 1].rstrip("\n")
        if CONTROL_STMT_RE.match(ln):
            continue
        m = METHOD_HEADER_RE.match(ln)
        if m:
            header_idx = i
            name = m.group("name")
            params = (m.group("params") or "").strip()
            break

    if not header_idx or not name:
        return None

    bal = 0
    end_idx = None
    for j in range(header_idx, len(lines) + 1):
        bal += _brace_balance_delta(lines[j - 1])
        if bal == 0 and j > header_idx:
            end_idx = j
            break
    if not end_idx:
        return None

    pkg = _find_package(lines)
    cls = _find_outer_class(lines, header_idx)
    sig = f"{name}({params})" if params else f"{name}()"
    qualified_parts = [p for p in [pkg, cls, sig] if p]
    qualified_name = ".".join(qualified_parts) if qualified_parts else sig

    fun_code = _slice_lines(lines, header_idx, end_idx)
    span = end_idx - header_idx + 1
    return {
        "qualified_name": qualified_name,
        "start_line": header_idx,
        "end_line": end_idx,
        "span": span,
        "func": fun_code,
    }


# ---------------------------------------------------------------------------
# Core annotator
# ---------------------------------------------------------------------------


def annotate_source_with_function(
    src: MutableMapping[str, Any],
    source_roots: Sequence[Path],
) -> MutableMapping[str, Any]:
    """
    Annotate a single NiCad source entry with enclosing function info + enclosing method code.

    Adds:
      src["function"] : dict or None
      src["is_function_clone"] : bool
    """
    if "range" not in src:
        return src

    clone_start, clone_end = parse_range(src["range"])
    clone_nlines = int(src.get("nlines") or (clone_end - clone_start + 1))

    try:
        root, full_path = resolve_source_path(src["file"], source_roots)
    except FileNotFoundError:
        src["function"] = None
        src["is_function_clone"] = False
        return src

    try:
        lines = _read_file_lines(full_path)
    except OSError:
        src["function"] = None
        src["is_function_clone"] = False
        return src

    rel_from_root = full_path.relative_to(root)

    chosen: Optional[Mapping[str, Any]] = None
    is_fun_clone = False

    # 1) Try tree-sitter helper
    try:
        result = find_fully_qualified_method(
            source_root=root,
            input_source_path=rel_from_root,
            start_line=clone_start,
            end_line=clone_end,
        )
        methods: Iterable[Mapping[str, Any]] = result.get("methods") or []
        chosen, is_fun_clone = choose_enclosing_method(methods, clone_start, clone_end)

        # if tree-sitter silently failed inside nearest_method, methods==[]
        if not methods:
            logging.debug("Tree-sitter returned 0 methods for %s", full_path)

    except Exception as e:
        logging.warning("nearest_method failed for %s:%s (%s)", src.get("file"), src.get("range"), e)

    # 2) Fallback heuristic if nothing found
    if not chosen:
        fallback = _fallback_enclosing_method(lines, clone_start, clone_end)
        if fallback:
            chosen = fallback
            is_fun_clone = (fallback["start_line"] == clone_start and fallback["end_line"] == clone_end)

    if not chosen:
        src["function"] = None
        src["is_function_clone"] = False
        return src

    fun_start = int(chosen.get("start_line"))
    fun_end = int(chosen.get("end_line"))
    fun_span = int(chosen.get("span") or (fun_end - fun_start + 1))

    fun_code = chosen.get("func") or _slice_lines(lines, fun_start, fun_end)

    src["function"] = {
        "qualified_name": chosen.get("qualified_name"),
        "fun_range": f"{fun_start}-{fun_end}",
        "fun_nlines": fun_span,
        "func": fun_code,         # keep compatibility with nearest_method
        "fun_code": fun_code,     # explicit: enclosing method code (your request)
        "clone_ratio": compute_clone_ratio(clone_nlines, fun_span),
    }
    src["is_function_clone"] = bool(is_fun_clone)

    return src


# ---------------------------------------------------------------------------
# Processing JSONL
# ---------------------------------------------------------------------------


def process(input_jsonl: Path, output_jsonl: Path, source_roots: Sequence[Path]) -> Dict[str, Any]:
    clusters = 0
    total_sources = 0
    annotated = 0
    missing_fun = 0
    fun_clone = 0

    with input_jsonl.open("r", encoding="utf-8") as fin, output_jsonl.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            clusters += 1
            sources = record.get("sources") or []
            new_sources: List[MutableMapping[str, Any]] = []

            for src in sources:
                if not isinstance(src, dict):
                    continue
                src2 = dict(src)
                src2.pop("function", None)
                src2.pop("is_function_clone", None)

                src2 = annotate_source_with_function(src2, source_roots)
                new_sources.append(src2)

                total_sources += 1
                if src2.get("function"):
                    annotated += 1
                else:
                    missing_fun += 1
                if src2.get("is_function_clone"):
                    fun_clone += 1

            record["sources"] = new_sources
            record["nclones"] = len(new_sources)

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {
        "clusters": clusters,
        "sources": total_sources,
        "annotated_sources": annotated,
        "missing_function": missing_fun,
        "function_clones": fun_clone,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Annotate NiCad JSONL clone sources with enclosing function metadata (tree-sitter + fallback)."
    )
    parser.add_argument("--input", required=True, type=Path, help="NiCad JSONL file.")
    parser.add_argument("--output", required=True, type=Path, help="Output JSONL with function annotations.")
    parser.add_argument(
        "--source-root",
        required=True,
        action="append",
        type=Path,
        help="Repo/source root. Pass multiple times if needed.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s [%(levelname)s] %(message)s")

    stats = process(args.input, args.output, list(args.source_root))
    logging.info(
        "Clusters=%d Sources=%d Annotated=%d MissingFunction=%d FunctionClones=%d",
        stats["clusters"],
        stats["sources"],
        stats["annotated_sources"],
        stats["missing_function"],
        stats["function_clones"],
    )


if __name__ == "__main__":
    main()