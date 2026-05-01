#!/usr/bin/env python3
import argparse, json, re, sys
from pathlib import Path
from typing import Optional, Dict, Any

PACKAGE_PATTERN = re.compile(r'^\s*package\s+([\w\.]+)\s*;', re.MULTILINE)
IDENT_BEFORE_PAREN = re.compile(r'([A-Za-z_]\w*)\s*\(')
CTRL_WORDS = {
    "if", "for", "while", "switch", "catch", "new", "return", "throw",
    "case", "do", "try", "assert", "synchronized", "else"
}

# cache file -> source text
SOURCE_CACHE: Dict[str, Optional[str]] = {}

def load_source(file_path: str, projects_root: Optional[Path]) -> Optional[str]:
    """Try absolute, then projects_root/file_path."""
    if not file_path:
        return None
    if file_path in SOURCE_CACHE:
        return SOURCE_CACHE[file_path]

    text: Optional[str] = None
    p = Path(file_path)
    if p.is_file():
        text = p.read_text(encoding="utf-8", errors="ignore")
    elif projects_root is not None:
        cand = projects_root / file_path
        if cand.is_file():
            text = cand.read_text(encoding="utf-8", errors="ignore")

    SOURCE_CACHE[file_path] = text
    return text

def package_name_from_file(file_path: str, projects_root: Optional[Path]) -> Optional[str]:
    src = load_source(file_path, projects_root)
    if not src:
        return None
    m = PACKAGE_PATTERN.search(src)
    return m.group(1) if m else None

def class_name_from_path(file_path: str) -> str:
    return Path(file_path).stem or "UnknownClass"

def _match_paren_span(src: str, open_idx: int) -> Optional[int]:
    depth = 0
    for pos in range(open_idx, len(src)):
        ch = src[pos]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return pos
    return None

def method_signature_from_code(code: str, file_path: str) -> str:
    """
    Very lightweight: take the last identifier before '(' in the header,
    skipping control words. Works well for typical Java method declarations.
    """
    cls = class_name_from_path(file_path)
    if not code:
        return f"{cls}.<unknown>()"

    header = code.split("{", 1)[0]
    name = None
    paren_idx = -1

    for m in IDENT_BEFORE_PAREN.finditer(header):
        ident = m.group(1)
        if ident not in CTRL_WORDS:
            name = ident
            paren_idx = m.end() - 1

    if not name:
        return f"{cls}.<unknown>()"

    params = ""
    if paren_idx != -1:
        close = _match_paren_span(header, paren_idx)
        if close is not None:
            params = header[paren_idx + 1 : close].strip()
            params = re.sub(r"\s+", " ", params)
            params = re.sub(r"\s*,\s*", ", ", params)

    return f"{cls}.{name}({params})" if params else f"{cls}.{name}()"

def build_qualified_name(source_obj: Dict[str, Any], projects_root: Optional[Path]) -> str:
    file_path = source_obj.get("file", "")
    code = source_obj.get("code", "") or ""
    pkg = package_name_from_file(file_path, projects_root)
    sig = method_signature_from_code(code, file_path)
    return f"{pkg}.{sig}" if pkg else sig

def annotate_line(obj: Dict[str, Any], projects_root: Optional[Path]) -> Dict[str, Any]:
    sources = obj.get("sources")
    if isinstance(sources, list):
        for s in sources:
            if isinstance(s, dict):
                s["qualified_name"] = build_qualified_name(s, projects_root)
    return obj

def main():
    ap = argparse.ArgumentParser(
        description="Add sources[].qualified_name = package + class + methodSignature to your NiCad JSONL."
    )
    ap.add_argument("--in", dest="in_path", default="-", help="Input JSONL (or - for stdin)")
    ap.add_argument("--out", dest="out_path", default="-", help="Output JSONL (or - for stdout)")
    ap.add_argument("--projects-root", default=None,
                    help="Base dir to resolve relative paths like systems/... (recommended)")
    args = ap.parse_args()

    projects_root = Path(args.projects_root).resolve() if args.projects_root else None

    fin = sys.stdin if args.in_path == "-" else open(args.in_path, "r", encoding="utf-8")
    fout = sys.stdout if args.out_path == "-" else open(args.out_path, "w", encoding="utf-8")

    # ---- stats ----
    class_count = 0
    total_nclones = 0
    nonzero_count = 0
    nonzero_sum = 0

    try:
        for raw in fin:
            line = raw.strip()
            if not line:
                continue

            obj = json.loads(line)

            # count only "class" objects that have sources list
            if isinstance(obj, dict) and isinstance(obj.get("sources"), list):
                class_count += 1

                # prefer declared nclones, fallback to len(sources)
                ncl = obj.get("nclones")
                if not isinstance(ncl, (int, float)):
                    ncl = len(obj["sources"])
                ncl_int = int(ncl)

                total_nclones += ncl_int
                if ncl_int > 0:
                    nonzero_count += 1
                    nonzero_sum += ncl_int

                obj = annotate_line(obj, projects_root)

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
        print(f"[OK] wrote {class_count} clone groups → {args.out_path}")

    finally:
        if fin is not sys.stdin:
            fin.close()
        if fout is not sys.stdout:
            fout.close()

    # finalize stats (print to stderr so JSONL output is clean)
    avg_all = (total_nclones / class_count) if class_count else 0.0
    avg_nonzero = (nonzero_sum / nonzero_count) if nonzero_count else 0.0

    print(f"[stats] classes parsed = {class_count}", file=sys.stderr)
    print(f"[stats] nclones total  = {total_nclones}", file=sys.stderr)
    print(f"[stats] nclones avg (all classes)          = {avg_all:.3f}", file=sys.stderr)
    print(f"[stats] nclones avg (classes with nclones) = {avg_nonzero:.3f}", file=sys.stderr)

if __name__ == "__main__":
    main()
