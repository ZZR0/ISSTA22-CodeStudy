#!/usr/bin/env python3
"""
6_gen_pos_clone_sample.py

Generate positive (clone) function pairs from a NiCad clone-group JSONL.
- Input JSONL must contain func_id inside each source (from Step 4).
- Output:
  * JSONL pairs (label=1)
  * TXT pairs: func_id1<TAB>func_id2<TAB>1
  * Optional HTML / Markdown reports
  * Optional verification on the TXT file

Example:
python 6_gen_pos_clone_sample.py \
  --input step4_nicad_camel_sim0.7_fqn_filtered_with_func_id.jsonl \
  --out_txt step6_nicad_camel_pos_samples.txt \
  --out_jsonl step6_nicad_camel_pos_samples.jsonl \
  --out_html step6_display_pos_sample.html \
  --out_md step6_display_pos_sample.md \
  --seed 42 \
  --verify \
  --cleanup
"""

import argparse
import html
import itertools
import json
import os
import random
from typing import Any, Dict, List, Tuple


def _makedirs_for_file(path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


# -------------------------
# Reports
# -------------------------
def generate_html_report(pairs: List[Dict[str, Any]], output_file: str) -> None:
    """Side-by-side HTML report for positive (clone) pairs."""
    html_content = [
        "<html><head><style>",
        "body { font-family: sans-serif; margin: 20px; background-color: #f9f9f9; }",
        "h2 { color: #333; }",
        ".pair-container { background: white; border: 1px solid #ddd; margin-bottom: 30px; padding: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }",
        ".pair-header { font-weight: bold; margin-bottom: 10px; padding-bottom: 5px; border-bottom: 2px solid #eee; color: #555; }",
        ".code-row { display: flex; gap: 20px; }",
        ".code-col { flex: 1; min-width: 0; }",
        ".meta-info { font-size: 0.85em; color: #666; margin-bottom: 5px; background: #f0f0f0; padding: 5px; border-radius: 4px; }",
        "pre { background-color: #f4f4f4; padding: 10px; border: 1px solid #ccc; overflow-x: auto; font-size: 0.8em; font-family: Consolas, monospace; height: 300px; }",
        ".label-badge { background-color: #2ecc71; color: white; padding: 2px 6px; border-radius: 3px; font-size: 0.8em; vertical-align: middle; }",
        "</style></head><body>",
        f"<h2>Generated Positive Samples (Label 1) - Total {len(pairs)} Pairs</h2>",
        "<p>These pairs represent functions from the <b>same</b> clone group.</p>",
    ]

    for i, pair in enumerate(pairs, 1):
        f1 = pair["func1"]
        f2 = pair["func2"]
        meta = pair.get("meta", {})

        info1 = (
            f"<b>ID:</b> {html.escape(str(f1.get('func_id', 'N/A')))}<br>"
            f"<b>ClassID:</b> {html.escape(str(meta.get('classid', 'N/A')))}<br>"
            f"<b>File:</b> {html.escape(str(f1.get('file', 'N/A')))}<br>"
            f"<b>Name:</b> {html.escape(str(f1.get('qualified_name', 'Unknown')))}"
        )
        info2 = (
            f"<b>ID:</b> {html.escape(str(f2.get('func_id', 'N/A')))}<br>"
            f"<b>ClassID:</b> {html.escape(str(meta.get('classid', 'N/A')))}<br>"
            f"<b>File:</b> {html.escape(str(f2.get('file', 'N/A')))}<br>"
            f"<b>Name:</b> {html.escape(str(f2.get('qualified_name', 'Unknown')))}"
        )

        code1 = html.escape(f1.get("code", "") or "")
        code2 = html.escape(f2.get("code", "") or "")

        html_content.append(
            f"""
        <div class="pair-container">
            <div class="pair-header">
                Pair #{i} <span class="label-badge">Positive (Clone)</span>
            </div>
            <div class="code-row">
                <div class="code-col">
                    <div class="meta-info">{info1}</div>
                    <pre>{code1}</pre>
                </div>
                <div class="code-col">
                    <div class="meta-info">{info2}</div>
                    <pre>{code2}</pre>
                </div>
            </div>
        </div>
        """
        )

    html_content.append("</body></html>")

    _makedirs_for_file(output_file)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(html_content))
    print(f"HTML report generated: {output_file}")


def generate_markdown_report(pairs: List[Dict[str, Any]], output_file: str) -> None:
    """Markdown table report (compact preview: first 5 lines)."""
    md_lines = [
        "### 🔍 Positive Sample Inspection (generated)",
        "",
        "| Pair ID | Function A (Source) | Function B (Target) |",
        "| :--- | :--- | :--- |",
    ]

    for i, pair in enumerate(pairs, 1):
        f1 = pair["func1"]
        f2 = pair["func2"]

        code1_snippet = "<br>".join((f1.get("code", "") or "").split("\n")[:5]).replace("|", "&#124;") + "..."
        code2_snippet = "<br>".join((f2.get("code", "") or "").split("\n")[:5]).replace("|", "&#124;") + "..."

        meta1 = f"<b>ID:</b> `{f1.get('func_id')}`<br>`{f1.get('qualified_name')}`"
        meta2 = f"<b>ID:</b> `{f2.get('func_id')}`<br>`{f2.get('qualified_name')}`"

        row = f"| **#{i}** | {meta1}<br><pre>{code1_snippet}</pre> | {meta2}<br><pre>{code2_snippet}</pre> |"
        md_lines.append(row)

    _makedirs_for_file(output_file)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"Markdown report generated: {output_file}")


# -------------------------
# Core generation
# -------------------------
def _iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def load_groups(input_file: str) -> Tuple[Dict[Any, List[Dict[str, Any]]], int, int]:
    """
    Loads JSONL clone groups.
    Returns:
      groups: dict[classid] -> list[func]
      total_funcs: int
      total_positive_pairs: int (sum of nC2)
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    groups: Dict[Any, List[Dict[str, Any]]] = {}
    total_funcs = 0
    total_positive_pairs = 0

    for obj in _iter_jsonl(input_file):
        classid = obj.get("classid")
        sources = obj.get("sources", [])
        if classid is None or not sources:
            continue

        # Keep original behavior: last occurrence wins if classid repeats
        groups[classid] = sources
        n = len(sources)
        total_funcs += n
        if n > 1:
            total_positive_pairs += (n * (n - 1)) // 2  # nC2

    return groups, total_funcs, total_positive_pairs


def pair_key(func_a: Dict[str, Any], func_b: Dict[str, Any]) -> Tuple[str, str]:
    """Uniqueness key: prefer func_id; fallback to qualified_name+range."""
    k1 = func_a.get("func_id") or (str(func_a.get("qualified_name", "")) + str(func_a.get("range", "")))
    k2 = func_b.get("func_id") or (str(func_b.get("qualified_name", "")) + str(func_b.get("range", "")))
    return tuple(sorted((str(k1), str(k2))))


def generate_positive_pairs(input_file: str, max_pairs: int, seed: int) -> List[Dict[str, Any]]:
    """
    Generate positive (clone) pairs from within each group.
    max_pairs = 0 => generate ALL possible pairs
    max_pairs > 0 => sample up to max_pairs from the full set
    """
    random.seed(seed)

    groups, total_funcs, total_pos_capacity = load_groups(input_file)

    print(f"--- Loading data from: {input_file} ---")
    print(f"Loaded {len(groups)} groups with {total_funcs} total functions.")
    print(f"Max possible POSITIVE samples (sum of nC2): {total_pos_capacity:,}")

    if total_pos_capacity == 0:
        print("Error: No positive pairs possible (all groups have size < 2).")
        return []

    all_pairs: List[Tuple[Any, Dict[str, Any], Dict[str, Any]]] = []
    for classid, funcs in groups.items():
        if len(funcs) < 2:
            continue
        for a, b in itertools.combinations(funcs, 2):
            all_pairs.append((classid, a, b))

    if max_pairs and max_pairs > 0:
        if max_pairs > len(all_pairs):
            print(f"Warning: max_pairs={max_pairs} > capacity={len(all_pairs)}. Using capacity instead.")
            max_pairs = len(all_pairs)
        all_pairs = random.sample(all_pairs, max_pairs)

    seen = set()
    pairs_out: List[Dict[str, Any]] = []
    for classid, a, b in all_pairs:
        k = pair_key(a, b)
        if k in seen:
            continue
        seen.add(k)
        pairs_out.append({"label": 1, "func1": a, "func2": b, "meta": {"classid": classid}})

    print(f"Done. Generated {len(pairs_out)} unique positive pairs.")
    return pairs_out


# -------------------------
# Save + verify
# -------------------------
def save_outputs(pairs: List[Dict[str, Any]], out_jsonl: str, out_txt: str) -> None:
    _makedirs_for_file(out_jsonl)
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    print(f"JSONL saved to: {out_jsonl}")

    _makedirs_for_file(out_txt)
    count = 0
    with open(out_txt, "w", encoding="utf-8") as f:
        for p in pairs:
            fid1 = p["func1"].get("func_id")
            fid2 = p["func2"].get("func_id")
            if fid1 and fid2:
                f.write(f"{fid1}\t{fid2}\t1\n")
                count += 1
    print(f"TXT saved to: {out_txt} ({count} lines)")


def verify_positive_samples(txt_path: str) -> bool:
    """
    Verify:
      - 3 tab-separated columns
      - label == '1'
      - func_id prefix (before '_') is same for both IDs
    """
    if not os.path.exists(txt_path):
        print(f"Error: File '{txt_path}' not found.")
        return False

    print(f"\n--- Verifying Positive Samples: {txt_path} ---")

    total = 0
    errors = 0
    valid = 0

    with open(txt_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            total += 1
            parts = line.split("\t")
            if len(parts) != 3:
                print(f"[Line {line_num}] Format Error: Expected 3 columns, got {len(parts)}")
                errors += 1
                continue

            id1, id2, label = parts
            if label != "1":
                print(f"[Line {line_num}] Label Error: Expected '1', got '{label}'")
                errors += 1
                continue

            if "_" not in id1 or "_" not in id2:
                print(f"[Line {line_num}] ID Format Error: func_id must contain '_'. Got ({id1}, {id2})")
                errors += 1
                continue

            g1 = id1.split("_", 1)[0]
            g2 = id2.split("_", 1)[0]

            if g1 != g2:
                print(f"[Line {line_num}] ❌ LOGIC FAILURE: Different groups ({g1} vs {g2}) labeled as 1")
                print(f"   -> {id1} vs {id2}")
                errors += 1
            else:
                valid += 1

    print(f"Total Lines Checked: {total}")
    print(f"Valid Positive Pairs: {valid}")

    if errors == 0:
        print("✅ SUCCESS: All pairs are valid positive samples (same clone groups).")
        return True

    print(f"❌ FAILED: Found {errors} invalid pairs.")
    return False


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser(description="Generate positive (clone) pairs from NiCad clone groups.")
    ap.add_argument("--input", required=True, help="Input JSONL clone groups (must include func_id)")
    ap.add_argument("--out_txt", required=True, help="Output TXT path (func_id1 func_id2 label)")
    ap.add_argument("--out_jsonl", required=True, help="Output JSONL path (pairs)")
    ap.add_argument("--out_html", default=None, help="Optional HTML report path")
    ap.add_argument("--out_md", default=None, help="Optional Markdown report path")
    ap.add_argument("--max_pairs", type=int, default=0, help="0=ALL positives; else sample up to this many")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--cleanup", action="store_true", help="Remove existing outputs before running")
    ap.add_argument("--verify", action="store_true", help="Verify the generated TXT file")

    args = ap.parse_args()

    if args.cleanup:
        for fpath in [args.out_jsonl, args.out_txt, args.out_html, args.out_md]:
            if fpath and os.path.exists(fpath):
                os.remove(fpath)

    pairs = generate_positive_pairs(args.input, max_pairs=args.max_pairs, seed=args.seed)
    if not pairs:
        return

    save_outputs(pairs, args.out_jsonl, args.out_txt)

    if args.out_html:
        generate_html_report(pairs, args.out_html)
    if args.out_md:
        generate_markdown_report(pairs, args.out_md)

    if args.verify:
        verify_positive_samples(args.out_txt)


if __name__ == "__main__":
    main()
