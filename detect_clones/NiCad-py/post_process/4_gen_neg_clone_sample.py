#!/usr/bin/env python3
"""
5_gen_neg_clone_sample.py

Generate negative (non-clone) function pairs from a NiCad clone-group JSONL.
- Input JSONL must contain func_id inside each source (from Step 4).
- Output:
  * JSONL pairs (label=0)
  * TXT pairs: func_id1<TAB>func_id2<TAB>0
  * Optional HTML / Markdown reports
  * Optional verification on the TXT file

Example:
python 5_gen_neg_clone_sample.py \
  --input nicad_camel_clone_data.jsonl \
  --out_txt nicad_camel_neg_samples.txt \
  --out_jsonl nicad_camel_neg_samples.jsonl \
  --out_html display_neg_sample.html \
  --out_md display_neg_sample.md \
  --seed 42 \
  --verify
"""

import argparse
import html
import json
import os
import random
import sys
from typing import Any, Dict, List, Tuple


def _makedirs_for_file(path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def generate_html_report(pairs: List[Dict[str, Any]], output_file: str) -> None:
    """Generates an HTML report to visualize generated negative pairs side-by-side."""
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
        ".label-badge { background-color: #e74c3c; color: white; padding: 2px 6px; border-radius: 3px; font-size: 0.8em; vertical-align: middle; }",
        "</style></head><body>",
        f"<h2>Generated Negative Samples (Label 0) - Total {len(pairs)} Pairs</h2>",
        "<p>These pairs represent functions from <b>different</b> clone groups (different logic).</p>",
    ]

    for i, pair in enumerate(pairs, 1):
        f1 = pair["func1"]
        f2 = pair["func2"]
        meta = pair["meta"]

        info1 = (
            f"<b>ID:</b> {html.escape(str(f1.get('func_id', 'N/A')))}<br>"
            f"<b>ClassID:</b> {html.escape(str(meta.get('classid1')))}<br>"
            f"<b>File:</b> {html.escape(str(f1.get('file', 'N/A')))}<br>"
            f"<b>Name:</b> {html.escape(str(f1.get('qualified_name', 'Unknown')))}"
        )
        info2 = (
            f"<b>ID:</b> {html.escape(str(f2.get('func_id', 'N/A')))}<br>"
            f"<b>ClassID:</b> {html.escape(str(meta.get('classid2')))}<br>"
            f"<b>File:</b> {html.escape(str(f2.get('file', 'N/A')))}<br>"
            f"<b>Name:</b> {html.escape(str(f2.get('qualified_name', 'Unknown')))}"
        )

        code1 = html.escape(f1.get("code", "") or "")
        code2 = html.escape(f2.get("code", "") or "")

        html_content.append(
            f"""
        <div class="pair-container">
            <div class="pair-header">
                Pair #{i} <span class="label-badge">Negative (Non-Clone)</span>
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
    """Generates a Markdown table report for GitHub README embedding."""
    md_lines = [
        "### 🔍 Negative Sample Inspection (generated)",
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

def extract_clone_group(fid: str) -> str:
    # Expected format: repo_classid_counter (e.g., activemq_125_228)
    parts = fid.split("_")
    if len(parts) < 3:
        return "unknown"
    return parts[1]   # classid

def verify_negative_samples(txt_path: str) -> bool:
    """Verifies that generated negative samples truly belong to different clone groups."""
    if not os.path.exists(txt_path):
        print(f"Error: File '{txt_path}' not found.")
        return False

    print(f"\n--- Verifying Negative Samples: {txt_path} ---")

    total_lines = 0
    errors = 0
    valid_pairs = 0

    with open(txt_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            total_lines += 1
            parts = line.split("\t")

            if len(parts) != 3:
                print(f"[Line {line_num}] Format Error: Expected 3 columns, found {len(parts)}.")
                errors += 1
                continue

            id1, id2, label = parts

            if label != "0":
                print(f"[Line {line_num}] Label Error: Expected '0', found '{label}'.")
                errors += 1
                continue

            if "_" not in id1 or "_" not in id2:
                print(f"[Line {line_num}] ID Format Error: IDs must contain '_'. Got ({id1}, {id2})")
                errors += 1
                continue
            # Handle the updated format: activemq_125_228, activemq_60_51
            group1 = extract_clone_group(id1)
            group2 = extract_clone_group(id2)
            # group1 = id1.split("_", 1)[0]
            # group2 = id2.split("_", 1)[0]

            if group1 == group2:
                print(
                    f"[Line {line_num}] ❌ LOGIC FAILURE: Pair is from the SAME group ({group1}). "
                    f"This is a POSITIVE sample labeled as 0!"
                )
                print(f"   -> {id1} vs {id2}")
                errors += 1
            else:
                valid_pairs += 1

    print(f"Total Lines Checked: {total_lines}")
    print(f"Valid Negative Pairs: {valid_pairs}")

    if errors == 0:
        print("✅ SUCCESS: All pairs are valid negative samples (different clone groups).")
        return True

    print(f"❌ FAILED: Found {errors} invalid pairs.")
    return False


def generate_negative_samples(
    input_file: str,
    seed: int,
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Returns: (pairs_list, total_positive_pairs)
    pairs_list entries look like:
      {"label":0, "func1": {...}, "func2": {...}, "meta":{"classid1":..., "classid2":...}}
    """
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        print("Tip: Make sure Step 4 generated func_id first.")
        sys.exit(1)

    random.seed(seed)

    print(f"--- Loading data from: {input_file} ---")

    groups: Dict[Any, List[Dict[str, Any]]] = {}
    flat_funcs: List[Tuple[Any, Dict[str, Any]]] = []
    total_positive_pairs = 0

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            class_id = data.get("classid")
            sources = data.get("sources", [])

            if class_id is None or not sources:
                continue

            groups[class_id] = sources

            n = len(sources)
            if n > 1:
                total_positive_pairs += (n * (n - 1)) // 2

            for src in sources:
                flat_funcs.append((class_id, src))

    total_funcs = len(flat_funcs)
    if total_funcs < 2 or len(groups) < 2:
        print("Error: Need at least 2 clone groups and 2 total functions to generate negatives.")
        sys.exit(1)

    total_possible_combinations = (total_funcs * (total_funcs - 1)) // 2
    max_negative_pairs = total_possible_combinations - total_positive_pairs

    print(f"Loaded {len(groups)} groups with {total_funcs} total functions.")

    print(f"\n=== Detailed Calculation of Dataset Capacity ===")
    print(f"1. Total Functions (N): {total_funcs}")
    print(f"   - We have {total_funcs} unique functions in the dataset.")
    print(f"   - Total possible pairs (Universe) = N * (N - 1) / 2")
    print(f"   - {total_funcs} * {total_funcs - 1} / 2 = {total_possible_combinations:,} pairs")

    print(f"\n2. Max Possible Positive Samples (Clones): {total_positive_pairs:,}")
    print(f"   - Calculation: Sum of pairs within each clone group.")
    print(f"   - Formula: Sum( size * (size - 1) / 2 ) for all groups.")
    print(f"   - This represents all pairs where Label = 1.")

    print(f"\n3. Max Possible Negative Samples (Non-Clones): {max_negative_pairs:,}")
    print(f"   - Calculation: Universe - Positives")
    print(f"   - Formula: {total_possible_combinations:,} - {total_positive_pairs:,} = {max_negative_pairs:,}")
    print(f"   - This represents all pairs where Label = 0.")
    print(f"================================================\n")

    target_neg_count = total_positive_pairs if total_positive_pairs > 0 else 10
    if target_neg_count > max_negative_pairs:
        print(f"Warning: Target {target_neg_count} exceeds max possible negatives. Adjusting to {max_negative_pairs}.")
        target_neg_count = max_negative_pairs

    print(f"--- Generating {target_neg_count} Negative Samples (Balanced) ---")

    generated_pairs_list: List[Dict[str, Any]] = []
    seen_pairs = set()
    attempts = 0
    max_attempts = max(1000, target_neg_count * 50)

    while len(generated_pairs_list) < target_neg_count and attempts < max_attempts:
        attempts += 1
        id1, data1 = random.choice(flat_funcs)
        id2, data2 = random.choice(flat_funcs)

        if id1 == id2:
            continue

        k1 = data1.get("func_id") or (data1.get("qualified_name", "") + str(data1.get("range", "")))
        k2 = data2.get("func_id") or (data2.get("qualified_name", "") + str(data2.get("range", "")))
        pair_key = tuple(sorted((str(k1), str(k2))))

        if pair_key in seen_pairs:
            continue

        seen_pairs.add(pair_key)
        generated_pairs_list.append(
            {
                "label": 0,
                "func1": data1,
                "func2": data2,
                "meta": {"classid1": id1, "classid2": id2},
            }
        )

    print(f"Done. Generated {len(generated_pairs_list)} unique negative pairs.")
    return generated_pairs_list, total_positive_pairs


def save_outputs(
    pairs: List[Dict[str, Any]],
    out_jsonl: str,
    out_txt: str,
) -> None:
    _makedirs_for_file(out_jsonl)
    with open(out_jsonl, "w", encoding="utf-8") as out_f:
        for p in pairs:
            out_f.write(json.dumps(p) + "\n")
    print(f"JSONL saved to: {out_jsonl}")

    _makedirs_for_file(out_txt)
    count = 0
    with open(out_txt, "w", encoding="utf-8") as txt_f:
        for p in pairs:
            fid1 = p["func1"].get("func_id")
            fid2 = p["func2"].get("func_id")
            if fid1 and fid2:
                txt_f.write(f"{fid1}\t{fid2}\t0\n")
                count += 1
    print(f"TXT saved to: {out_txt} ({count} lines)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate negative (non-clone) function pairs from NiCad clone groups.")
    ap.add_argument("--input", required=True, help="Input clone-group JSONL (must include func_id)")
    ap.add_argument("--out_txt", required=True, help="Output TXT path (func_id1 func_id2 label)")
    ap.add_argument("--out_jsonl", required=True, help="Output JSONL path (pairs)")
    ap.add_argument("--out_html", default=None, help="Optional HTML report path")
    ap.add_argument("--out_md", default=None, help="Optional Markdown report path")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--verify", action="store_true", help="Verify the generated TXT file")
    ap.add_argument("--cleanup", action="store_true", help="Remove existing outputs before running")

    args = ap.parse_args()

    if args.cleanup:
        for f in [args.out_jsonl, args.out_txt, args.out_html, args.out_md]:
            if f and os.path.exists(f):
                os.remove(f)

    pairs, _ = generate_negative_samples(args.input, seed=args.seed)
    save_outputs(pairs, args.out_jsonl, args.out_txt)

    if args.out_html:
        generate_html_report(pairs, args.out_html)
    if args.out_md:
        generate_markdown_report(pairs, args.out_md)

    if args.verify:
        verify_negative_samples(args.out_txt)


if __name__ == "__main__":
    main()
