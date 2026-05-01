#!/usr/bin/env python3
import re
import json
import argparse
from pathlib import Path
from html import unescape as html_unescape
from typing import Dict

# --- Regexes ---
CLASS_RE  = re.compile(r"<class\b([^>]*)>(.*?)</class>",  re.DOTALL | re.IGNORECASE)
SOURCE_RE = re.compile(r"<source\b([^>]*)>(.*?)</source>", re.DOTALL | re.IGNORECASE)
ATTR_RE   = re.compile(r'([A-Za-z_:][\w:.-]*)\s*=\s*"([^"]*)"')

# --- Helpers ---
def parse_attrs(attr_text: str) -> Dict[str, str]:
    return {k: v for k, v in ATTR_RE.findall(attr_text or "")}

def to_int(s, default=0):
    try:
        return int(str(s).strip())
    except Exception:
        return default

def to_float(s, default=0.0):
    try:
        return float(str(s).strip())
    except Exception:
        return default

_ILLEGAL_CTRL = re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F]')

def clean_code_text(code: str, strip_blank_edges: bool = True, sanitize_ctrl: bool = True) -> str:
    if not code:
        return code
    s = code
    if sanitize_ctrl:
        s = _ILLEGAL_CTRL.sub('', s)
    s = s.replace('\r\n', '\n').replace('\r', '\n')
    if strip_blank_edges:
        lines = s.splitlines()
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
        s = "\n".join(lines)
    return s

# --- Core ---
def nicad_xml_to_jsonl_regex(xml_path: str, out_path: str, mode: str = "class",
                             unescape_entities: bool = False, keep_ctrl: bool = False):
    text = Path(xml_path).read_text(encoding="utf-8", errors="replace")

    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    n_rows = 0
    wrote_warning = False

    # Global aggregations for class mode
    total_nclones = 0
    class_count = 0
    class_count_with_nclones = 0

    with out_p.open("w", encoding="utf-8") as f:
        class_iter = CLASS_RE.finditer(text)
        first_class = next(class_iter, None)

        if mode == "class" and first_class is not None:
            print("[info] parsing <class> blocks → one row per class with sources[]")

            # Handle the first found class, then the rest from the same iterator
            pending_classes = [first_class]
            pending_classes_iter = (m for m in class_iter)

            for m in (*pending_classes, *pending_classes_iter):
                class_attrs_raw = m.group(1) or ""
                class_body = m.group(2) or ""
                a = parse_attrs(class_attrs_raw)

                cls_obj = {
                    "classid": to_int(a.get("classid", a.get("id", "0")), 0),
                    "nclones": to_int(a.get("nclones", "0"), 0),
                    "similarity": to_float(a.get("similarity", "0"), 0.0),
                    "sources": []
                }

                # --- accumulate global stats ---
                total_nclones += cls_obj["nclones"]
                class_count += 1
                if cls_obj["nclones"] > 0:
                    class_count_with_nclones += 1

                parsed_sources = 0
                for sm in SOURCE_RE.finditer(class_body):
                    src_attrs_raw = sm.group(1) or ""
                    code_text = sm.group(2) or ""
                    sa = parse_attrs(src_attrs_raw)

                    file_attr = sa.get("file") or sa.get("srcfile") or ""
                    start = to_int(sa.get("startline", "0"), 0)
                    end   = to_int(sa.get("endline", "0"), 0)
                    rng = f"{start}-{end}" if start and end and end >= start else ""
                    nlines_src = (end - start + 1) if rng else 0
                    pcid = sa.get("pcid")

                    if unescape_entities:
                        code_text = html_unescape(code_text)
                    code_text = clean_code_text(code_text, sanitize_ctrl=not keep_ctrl)

                    # NOTE: removed file_base, startline, endline
                    cls_obj["sources"].append({
                        "file": file_attr,
                        "range": rng,
                        "nlines": nlines_src,
                        "pcid": pcid,
                        "code": code_text,
                    })
                    parsed_sources += 1

                if cls_obj["nclones"] and parsed_sources != cls_obj["nclones"] and not wrote_warning:
                    print(f"[warn] some classes report nclones != parsed count (first seen at class {cls_obj['classid']})")
                    wrote_warning = True

                f.write(json.dumps(cls_obj, ensure_ascii=False) + "\n")
                n_rows += 1

            # --- print global stats at the end of class mode ---
            avg_all = (total_nclones / class_count) if class_count else 0.0
            avg_nonzero = (total_nclones / class_count_with_nclones) if class_count_with_nclones else 0.0
            print(f"[stats] classes parsed = {class_count}")
            print(f"[stats] nclones total  = {total_nclones}")
            print(f"[stats] nclones avg (all classes)          = {avg_all:.3f}")
            print(f"[stats] nclones avg (classes with nclones) = {avg_nonzero:.3f}")

        else:
            if mode == "class":
                print("[info] no <class> blocks detected; falling back to --mode source")
            print("[info] parsing <source> blocks → one row per source")

            for sm in SOURCE_RE.finditer(text):
                src_attrs_raw = sm.group(1) or ""
                code_text = sm.group(2) or ""
                sa = parse_attrs(src_attrs_raw)

                file_attr = sa.get("file") or sa.get("srcfile") or ""
                start = to_int(sa.get("startline", "0"), 0)
                end   = to_int(sa.get("endline", "0"), 0)
                rng = f"{start}-{end}" if start and end and end >= start else ""
                nlines_src = (end - start + 1) if rng else 0
                pcid = sa.get("pcid")

                if unescape_entities:
                    code_text = html_unescape(code_text)
                code_text = clean_code_text(code_text, sanitize_ctrl=not keep_ctrl)

                # NOTE: removed file_base, startline, endline
                row = {
                    "file": file_attr,
                    "range": rng,
                    "nlines": nlines_src,
                    "pcid": pcid,
                    "code": code_text
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_rows += 1

    print(f"[OK] wrote {n_rows} row(s) → {out_p}")

# --- CLI ---
def main():
    ap = argparse.ArgumentParser(
        description="Regex-based NiCad classes-withsource XML → JSONL (keeps code)."
    )
    ap.add_argument("--xml", required=True, help="Path to *-classes-withsource.xml (or fragment with <source> blocks)")
    ap.add_argument("--out", required=True, help="Output JSONL")
    ap.add_argument("--mode", choices=["class", "source"], default="class",
                    help="class: one row per <class> (with sources[]); source: one row per <source>")
    ap.add_argument("--unescape", action="store_true",
                    help="HTML-unescape code bodies (e.g., &lt; → <).")
    ap.add_argument("--keep-ctrl", action="store_true",
                    help="Keep illegal ASCII control characters (default strips them).")
    args = ap.parse_args()

    nicad_xml_to_jsonl_regex(
        args.xml,
        args.out,
        mode=args.mode,
        unescape_entities=args.unescape,
        keep_ctrl=args.keep_ctrl
    )

if __name__ == "__main__":
    main()
