#!/usr/bin/env python3
import argparse
import json
import os
import random
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

# python inspect_clones.py --jsonl ./data/nicad_camel_clone_func.jsonl --classid 2520
# 
# Random sampling
# 
# python inspect_clones.py --jsonl ./data/nicad_camel_clone_func.jsonl --random 5
# 
# Filter by similarity
# 
# python inspect_clones.py --jsonl ./data/nicad_camel_clone_func.jsonl --random 10 --min_similarity 80
# 
# Generate HTML report
# 
# python inspect_clones.py --jsonl ./data/nicad_camel_clone_func.jsonl --classid 2520 --html b.html

# ----------------------------
# Utilities
# ----------------------------

TOKEN_RE = re.compile(r"[A-Za-z_]\w+|\d+|==|!=|<=|>=|&&|\|\||[{}()[\].,;:+\-*/%<>=!&|^~?]")

JAVA_KEYWORDS = {
    "abstract","assert","boolean","break","byte","case","catch","char","class","const","continue",
    "default","do","double","else","enum","extends","final","finally","float","for","goto","if",
    "implements","import","instanceof","int","interface","long","native","new","package","private",
    "protected","public","return","short","static","strictfp","super","switch","synchronized","this",
    "throw","throws","transient","try","void","volatile","while","var","record","sealed","permits",
    "non-sealed","yield"
}

def tokenize(code: str) -> List[str]:
    return TOKEN_RE.findall(code)

def normalize_tokens(tokens: List[str]) -> List[str]:
    """
    Heuristic normalization: replace identifiers with ID, numbers with NUM,
    keep keywords and punctuation.
    """
    out = []
    for t in tokens:
        if t.isdigit():
            out.append("NUM")
        elif re.match(r"^[A-Za-z_]\w+$", t):
            if t in JAVA_KEYWORDS:
                out.append(t)
            else:
                out.append("ID")
        else:
            out.append(t)
    return out

def jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(1, len(sa | sb))

def trigram_jaccard(a: List[str], b: List[str]) -> float:
    def trigrams(x: List[str]) -> set:
        if len(x) < 3:
            return {tuple(x)} if x else set()
        return {tuple(x[i:i+3]) for i in range(len(x)-2)}
    ta, tb = trigrams(a), trigrams(b)
    if not ta and not tb:
        return 1.0
    return len(ta & tb) / max(1, len(ta | tb))

def format_pct(x: float) -> str:
    return f"{x*100:5.1f}%"

# ----------------------------
# Data model
# ----------------------------

@dataclass
class SourceSnippet:
    file: str
    range: str
    nlines: int
    pcid: str
    code: str
    qualified_name: str

@dataclass
class CloneGroup:
    classid: int
    nclones: int
    similarity: float
    sources: List[SourceSnippet]

def load_jsonl(path: str) -> List[CloneGroup]:
    groups: List[CloneGroup] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                sources = []
                for s in obj.get("sources", []):
                    sources.append(SourceSnippet(
                        file=s.get("file", ""),
                        range=s.get("range", ""),
                        nlines=int(s.get("nlines", 0)),
                        pcid=str(s.get("pcid", "")),
                        code=s.get("code", ""),
                        qualified_name=s.get("qualified_name", ""),
                    ))
                groups.append(CloneGroup(
                    classid=int(obj.get("classid")),
                    nclones=int(obj.get("nclones", len(sources))),
                    similarity=float(obj.get("similarity", 0.0)),
                    sources=sources
                ))
            except Exception as e:
                raise RuntimeError(f"Failed parsing JSONL at line {line_no}: {e}") from e
    return groups

# ----------------------------
# Printing / HTML
# ----------------------------

def print_group(g: CloneGroup, show_matrix: bool = True, norm: bool = True) -> None:
    print("=" * 90)
    print(f"CLASSID: {g.classid}  |  nclones: {g.nclones}  |  dataset_similarity: {g.similarity}")
    print("-" * 90)

    token_sets = []
    tri_sets = []

    for idx, s in enumerate(g.sources, 1):
        print(f"[{idx}] {s.qualified_name}")
        print(f"    file: {s.file}:{s.range}  |  nlines: {s.nlines}  |  pcid: {s.pcid}")
        print("    code:")
        code_indented = "\n".join("      " + line.rstrip("\n") for line in s.code.splitlines())
        print(code_indented)
        print()

        toks = tokenize(s.code)
        toks = normalize_tokens(toks) if norm else toks
        token_sets.append(toks)
        tri_sets.append(toks)

    if show_matrix and len(g.sources) >= 2:
        print("Pairwise similarity (Jaccard on normalized tokens) + (Trigram Jaccard):")
        n = len(g.sources)

        # header
        hdr = "      " + " ".join([f"{i:>8}" for i in range(1, n+1)])
        print(hdr)

        for i in range(n):
            row = [f"{i+1:>4}  "]
            for j in range(n):
                if i == j:
                    row.append("   (self)")
                else:
                    jac = jaccard(token_sets[i], token_sets[j])
                    tri = trigram_jaccard(tri_sets[i], tri_sets[j])
                    row.append(f"{format_pct(jac)}/{format_pct(tri)}")
            print(" ".join(row))
        print()

def html_escape(s: str) -> str:
    return (s.replace("&", "&amp;")
             .replace("<", "&lt;")
             .replace(">", "&gt;")
             .replace('"', "&quot;")
             .replace("'", "&#39;"))

def write_html(groups: List[CloneGroup], out_path: str, norm: bool = True) -> None:
    parts = []
    parts.append("""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Clone Group Inspector</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 16px; }
    .group { border: 1px solid #ddd; border-radius: 10px; padding: 12px; margin-bottom: 14px; }
    .meta { color: #444; margin-bottom: 8px; }
    details { border: 1px solid #eee; border-radius: 10px; padding: 10px; margin: 10px 0; background: #fafafa; }
    summary { cursor: pointer; font-weight: 600; }
    pre { white-space: pre-wrap; background: #111; color: #f1f1f1; padding: 10px; border-radius: 10px; overflow-x: auto; }
    table { border-collapse: collapse; margin-top: 8px; font-size: 14px; }
    td, th { border: 1px solid #ddd; padding: 6px 8px; text-align: center; }
    th { background: #f3f3f3; }
    .small { font-size: 13px; color: #555; }
  </style>
</head>
<body>
<h2>Clone Group Inspector</h2>
<p class="small">Each group shows snippets and a pairwise similarity matrix (token Jaccard / trigram Jaccard).</p>
""")

    for g in groups:
        parts.append(f'<div class="group">')
        parts.append(f'<div class="meta"><b>CLASSID</b>: {g.classid} &nbsp; | &nbsp; '
                     f'<b>nclones</b>: {g.nclones} &nbsp; | &nbsp; '
                     f'<b>dataset_similarity</b>: {g.similarity}</div>')

        tokens = []
        for s in g.sources:
            toks = tokenize(s.code)
            toks = normalize_tokens(toks) if norm else toks
            tokens.append(toks)

        for i, s in enumerate(g.sources, 1):
            parts.append("<details>")
            parts.append(f"<summary>[{i}] {html_escape(s.qualified_name)}</summary>")
            parts.append(f'<div class="small">file: {html_escape(s.file)}:{html_escape(s.range)} '
                         f'| nlines: {s.nlines} | pcid: {html_escape(s.pcid)}</div>')
            parts.append(f"<pre>{html_escape(s.code)}</pre>")
            parts.append("</details>")

        if len(g.sources) >= 2:
            n = len(g.sources)
            parts.append("<div class='small'><b>Pairwise similarity</b> (token Jaccard / trigram Jaccard)</div>")
            parts.append("<table>")
            parts.append("<tr><th></th>" + "".join([f"<th>{i}</th>" for i in range(1, n+1)]) + "</tr>")
            for i in range(n):
                parts.append(f"<tr><th>{i+1}</th>")
                for j in range(n):
                    if i == j:
                        parts.append("<td>(self)</td>")
                    else:
                        jac = jaccard(tokens[i], tokens[j])
                        tri = trigram_jaccard(tokens[i], tokens[j])
                        parts.append(f"<td>{format_pct(jac)}<br/>{format_pct(tri)}</td>")
                parts.append("</tr>")
            parts.append("</table>")

        parts.append("</div>")

    parts.append("</body></html>")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))

# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser(description="Inspect clone groups in JSONL dataset.")
    ap.add_argument("--jsonl", required=True, help="Path to data.jsonl")
    ap.add_argument("--classid", type=int, default=None, help="Inspect a specific classid")
    ap.add_argument("--random", type=int, default=0, help="Inspect N random groups")
    ap.add_argument("--min_similarity", type=float, default=None, help="Filter groups by dataset similarity >= X")
    ap.add_argument("--limit", type=int, default=1, help="How many groups to print when not using --random")
    ap.add_argument("--no_matrix", action="store_true", help="Don't print pairwise similarity matrix")
    ap.add_argument("--no_norm", action="store_true", help="Don't normalize identifiers/numbers")
    ap.add_argument("--html", default=None, help="Write inspected groups to an HTML file")
    args = ap.parse_args()

    groups = load_jsonl(args.jsonl)
    by_id: Dict[int, CloneGroup] = {g.classid: g for g in groups}

    # filter
    filtered = groups
    if args.min_similarity is not None:
        filtered = [g for g in filtered if g.similarity >= args.min_similarity]

    selected: List[CloneGroup] = []

    if args.classid is not None:
        if args.classid not in by_id:
            print(f"classid {args.classid} not found in dataset.", file=sys.stderr)
            sys.exit(2)
        selected = [by_id[args.classid]]

    elif args.random and args.random > 0:
        if len(filtered) == 0:
            print("No groups match filter.", file=sys.stderr)
            sys.exit(2)
        k = min(args.random, len(filtered))
        selected = random.sample(filtered, k)

    else:
        selected = filtered[: max(1, args.limit)]

    for g in selected:
        print_group(g, show_matrix=not args.no_matrix, norm=not args.no_norm)

    if args.html:
        write_html(selected, args.html, norm=not args.no_norm)
        print(f"Wrote HTML report: {args.html}")

if __name__ == "__main__":
    main()