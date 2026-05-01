#!/usr/bin/env python3
import os
import glob
import json
import argparse
import html
import pandas as pd
import re

def _as_sorted_unique_list(data):
    if isinstance(data, list):
        return sorted(list(set(str(v).strip() for v in data if v)))
    return []

def format_code_html(func_info):
    """Generates the HTML block with highlighted clones for a given function."""
    if not func_info:
        return "No information available."
    
    clone_range = func_info.get("range", "")
    enclosing = func_info.get("enclosing_function", {})
    fun_range = enclosing.get("fun_range", "")
    func_code = enclosing.get("func_code", func_info.get("code", ""))
    
    use_set = func_info.get("In", {}).get("InType", {}).get("Use(i)", [])
    
    if not fun_range or not clone_range:
        return html.escape(func_code)
    
    try:
        c_start, c_end = map(int, clone_range.split("-"))
        f_start, f_end = map(int, fun_range.split("-"))
    except:
        return html.escape(func_code)

    lines = func_code.splitlines()
    out_html = ""
    in_signature = True
    
    for i, line in enumerate(lines):
        line_num = f_start + i
        escaped_line = html.escape(line)
        if not escaped_line: escaped_line = " " 
            
        if c_start <= line_num <= c_end:
            cls = "in-clone"
            in_signature = False
            for var in sorted(use_set, key=len, reverse=True):
                escaped_line = re.sub(rf"\b({re.escape(var)})\b", r'<span style="background-color: #ffeb3b; color: #b30000; border-radius: 2px; padding: 0 2px;">\1</span>', escaped_line)
        elif line_num < c_start:
            if in_signature:
                cls = "method-signature"
                if "{" in line:
                    in_signature = False
            else:
                cls = "before-clone"
        else:
            cls = "after-clone"
        
        out_html += f'<span class="line-number">{line_num}</span><span class="{cls}">{escaped_line}</span>\n'
    return out_html

def generate_html_report(records, out_html_path):
    """Generates a professional standalone HTML report for the sampled pairs."""
    html_out = ["""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Refactoring Manual Inspection Report</title>
<style>
    body { font-family: sans-serif; background-color: #f4f4f9; padding: 20px; }
    h1 { text-align: center; }
    .clone-class { background: #fff; border: 1px solid #ddd; margin-bottom: 30px; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    .clone-instance { margin-top: 15px; border-top: 2px dashed #eee; padding-top: 15px; }
    .meta-info { font-size: 0.9em; color: #555; margin-bottom: 10px; }
    pre { background-color: #fafafa; border: 1px solid #ccc; padding: 15px; overflow-x: auto; border-radius: 4px; line-height: 1.4; }
    
    /* Coloring rules */
    .method-signature { color: #8b008b; font-weight: bold; } /* Dark Magenta */
    .before-clone { color: blue; }
    .in-clone { color: red; font-weight: bold; }
    .after-clone { color: black; }
    .line-number {
        display: inline-block;
        width: 45px;
        color: #aaa;
        border-right: 1px solid #ddd;
        padding-right: 10px;
        margin-right: 10px;
        text-align: right;
        user-select: none;
        font-size: 0.9em;
    }
    .badge { display: inline-block; padding: 5px 10px; border-radius: 12px; font-weight: bold; color: #fff; font-size: 12px; margin-right: 10px; margin-bottom: 10px;}
    .case1 { background: #4caf50; }
    .case2 { background: #2196f3; }
    .case3 { background: #f44336; }
</style>
</head>
<body>
<h1>Refactoring Manual Inspection Report</h1>
<p style="text-align: center;">Total Sampled Pairs: """ + str(len(records)) + "</p>"]

    def generate_meta_box(func_info):
        ext_sig = html.escape(func_info.get("Extracted Signature", ""))
        
        # Simple heuristic to colorize the signature like main.py
        sig_parts = ext_sig.split(" extractedClone(")
        if len(sig_parts) == 2:
            ret_type = sig_parts[0]
            args = sig_parts[1].rstrip(")")
            styled_sig = f'<span style="color: #0066cc;">{ret_type}</span> <span style="color: #6f42c1;">extractedClone</span>({args})'
        else:
            styled_sig = ext_sig
            
        in_dict = func_info.get("In", {})
        out_dict = func_info.get("Out", {})
        
        in_i = in_dict.get("In(i)", [])
        use_i = in_dict.get("InType", {}).get("Use(i)", [])
        def_before_i = in_dict.get("InType", {}).get("Defbefore(i)", [])
        
        out_i = out_dict.get("Out(i)", [])
        def_within_i = out_dict.get("Defwithin(i)", [])
        use_after_i = out_dict.get("Useafter(i)", [])
        
        def fmt_set(s): return "{" + html.escape(", ".join(sorted(s))) + "}" if s else "{}"
        def fmt_bold_set(s): return "<strong>{" + html.escape(", ".join(sorted(s))) + "}</strong>" if s else "<strong>{}</strong>"
        
        box = f"""
        <div style="background-color: #e8f4f8; border: 1px solid #b3d4fc; padding: 8px; margin-top: 10px; border-radius: 4px; font-family: monospace;">
            <strong>Extracted Signature:</strong> <br>
            {styled_sig}
            
            <div style="margin-top: 8px; font-size: 0.9em; color: #444; border-top: 1px dashed #b3d4fc; padding-top: 5px;">
                <strong>Parameters Derivation:</strong> <br>
                In(i) = Use(i) &cap; Def<sub>before</sub>(i) <br>
                In(i) = {fmt_set(use_i)} &cap; {fmt_set(def_before_i)} = {fmt_bold_set(in_i)}
            </div>

            <div style="margin-top: 8px; font-size: 0.9em; color: #444; border-top: 1px dashed #b3d4fc; padding-top: 5px;">
                <strong>Return Derivation:</strong> <br>
                Out(i) = Def<sub>within</sub>(i) &cap; Use<sub>after</sub>(i) <br>
                Out(i) = {fmt_set(def_within_i)} &cap; {fmt_set(use_after_i)} = {fmt_bold_set(out_i)}
            </div>
        </div>"""
        return box

    for r in records:
        case_name = r.get("inspection_case", "")
        badge_cls = "case1" if "Case 1" in case_name else "case2" if "Case 2" in case_name else "case3"
        
        sources = r.get("sources", [])
        if len(sources) < 2: continue
        
        class_id = r.get('classid', 'Unknown')
        
        html_out.append(f"""<div class="clone-class">
    <h2>Clone Class ID: {html.escape(class_id)} (Instances: 2)</h2>
    <div style="margin-bottom: 10px;">
        <span class="badge {badge_cls}">{html.escape(case_name)}</span>
        <strong>Project:</strong> {html.escape(str(r.get('project', '')))} | 
        <strong>Refactorable:</strong> {r.get('Refactorable', 0)}
    </div>""")

        for func_info in sources:
            html_out.append(f"""<div class="clone-instance">
    <div class="meta-info">
        <strong>Func ID:</strong> {html.escape(func_info.get("func_id", ""))} <br>
        <strong>File:</strong> {html.escape(func_info.get("file", ""))} <br>
        <strong>Clone Range:</strong> {html.escape(func_info.get("range", ""))} <br>
        <strong>Enclosing Method:</strong> {html.escape(func_info.get("enclosing_function", {}).get("qualified_name", ""))} (Lines {html.escape(func_info.get("enclosing_function", {}).get("fun_range", ""))}) <br>
        {generate_meta_box(func_info)}
    </div>
    <pre><code>{format_code_html(func_info)}</code></pre>
</div>""")
        
        html_out.append("</div>")
    html_out.append("</body></html>")
    
    with open(out_html_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html_out))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl_dir", required=True, help="Directory with extractable *_clone_analysis.jsonl files")
    parser.add_argument("--gt_dir", required=True, help="Base directory for Ground Truth text files")
    parser.add_argument("--pred_dir", required=True, help="Directory for BEFORE adaptation predictions")
    parser.add_argument("--pred_adapt_dir", required=True, help="Directory for AFTER adaptation predictions")
    parser.add_argument("--out_all_jsonl", required=True, help="Output path for ALL evaluated pairs")
    parser.add_argument("--out_sampled_jsonl", required=True, help="Output path for the filtered 100 samples")
    parser.add_argument("--out_html", required=True, help="Output path for HTML visualization")
    args = parser.parse_args()

    # Create output directories if they don't exist
    os.makedirs(os.path.dirname(args.out_all_jsonl), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_sampled_jsonl), exist_ok=True)

    all_pairs = []
    
    # 1. Loop through all available projects
    jsonl_files = glob.glob(os.path.join(args.jsonl_dir, "augmented_*.jsonl"))
    print(f"Found {len(jsonl_files)} projects to process.")
    
    for jsonl_file in jsonl_files:
        base = os.path.basename(jsonl_file)
        project_name = base.replace("augmented_", "").replace(".jsonl", "")
        
        gt_file = os.path.join(args.gt_dir, project_name, "test.txt")
        pred_before = os.path.join(args.pred_dir, f"predictions_{project_name}_test.txt")
        pred_after = os.path.join(args.pred_adapt_dir, f"predictions_{project_name}_test.txt")
        
        if not (os.path.exists(gt_file) and os.path.exists(pred_before) and os.path.exists(pred_after)):
            continue # Skip projects that don't have predictions yet
            
        print(f"Processing {project_name}...")
        
        # Load Raw JSONL Function Data
        func_map = {}
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    for s in obj.get("sources", []):
                        if "func_id" in s: func_map[s["func_id"]] = s
                except: continue
                
        # Load Ground Truth and Adapted Predictions mappings
        gt_map, adapt_map = {}, {}
        with open(gt_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3: gt_map[(parts[0], parts[1])] = parts[2]
        with open(pred_after, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3: adapt_map[(parts[0], parts[1])] = parts[2]

        # Evaluate Predictions
        with open(pred_before, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3: continue
                left_id, right_id, prediction = parts[0], parts[1], parts[2]
                
                if left_id not in func_map or right_id not in func_map: continue
                
                meta_l, meta_r = func_map[left_id], func_map[right_id]
                l_file, r_file = meta_l.get("file", "UNKNOWN"), meta_r.get("file", "UNKNOWN")
                
                l_in, r_in = meta_l.get("In", {}).get("In(i)", []), meta_r.get("In", {}).get("In(i)", [])
                l_out, r_out = meta_l.get("Out", {}).get("Out(i)", []), meta_r.get("Out", {}).get("Out(i)", [])
                l_in_t, r_in_t = _as_sorted_unique_list(meta_l.get("In", {}).get("InType", {}).get("InType", [])), _as_sorted_unique_list(meta_r.get("In", {}).get("InType", {}).get("InType", []))
                l_out_t, r_out_t = _as_sorted_unique_list(meta_l.get("Out", {}).get("OutType", [])), _as_sorted_unique_list(meta_r.get("Out", {}).get("OutType", []))

                is_same_file = 1 if (l_file == r_file and l_file != "UNKNOWN") else 0
                is_refactorable = 1 if (len(l_in) == len(r_in) and len(l_out) == len(r_out) and l_in_t == r_in_t and l_out_t == r_out_t) else 0
                actual_label = gt_map.get((left_id, right_id), "N/A")
                
                # Match main.py class format representation
                pair_record = {
                    "classid": f"{left_id}_vs_{right_id}",
                    "nclones": 2,
                    "similarity": None,
                    "project": project_name,
                    "same_file": is_same_file,
                    "actual_label": int(actual_label) if actual_label != "N/A" else -1,
                    "clone_predict": int(prediction),
                    "clone_predict_after_adapt": int(adapt_map.get((left_id, right_id), 0)),
                    "Refactorable": is_refactorable,
                    "sources": [meta_l, meta_r]
                }
                all_pairs.append(pair_record)

    # Output Requirement 1: Save ALL evaluated pairs into ONE massive JSONL file
    with open(args.out_all_jsonl, "w", encoding="utf-8") as f:
        for p in all_pairs:
            f.write(json.dumps(p) + "\n")
    print(f"\n[1/3] Saved {len(all_pairs)} total pairs to {args.out_all_jsonl}")

    # Output Requirement 2: Apply the Pandas Filters constraints
    df = pd.DataFrame(all_pairs)
    df = df[df['same_file'] == 1]

    mask_c1 = (df['actual_label'] == 1) & (df['clone_predict'] == 1) & (df['clone_predict_after_adapt'] == 1) & (df['Refactorable'] == 1)
    mask_c2 = (df['actual_label'] == 1) & (df['clone_predict'] == 0) & (df['clone_predict_after_adapt'] == 1) & (df['Refactorable'] == 1)
    mask_c3 = (df['actual_label'] == 0) & (df['clone_predict'] == 1) & (df['clone_predict_after_adapt'] == 0)

    df_c1 = df[mask_c1].copy(); df_c1['inspection_case'] = 'Case 1 (Both Found - Refactorable)'
    df_c2 = df[mask_c2].copy(); df_c2['inspection_case'] = 'Case 2 (Adaptation Fixed Miss - Refactorable)'
    df_c3 = df[mask_c3].copy(); df_c3['inspection_case'] = 'Case 3 (Adaptation Rejected False Alarm)'

    s1 = df_c1.sample(n=min(40, len(df_c1)), random_state=42)
    s2 = df_c2.sample(n=min(30, len(df_c2)), random_state=42)
    s3 = df_c3.sample(n=min(30, len(df_c3)), random_state=42)
    
    sampled_df = pd.concat([s1, s2, s3])
    sampled_records = sampled_df.to_dict('records')

    # CHANGED: Now saving as a standard, human-readable JSON array instead of JSON Lines
    with open(args.out_sampled_jsonl, "w", encoding="utf-8") as f:
        json.dump(sampled_records, f, indent=4)
            
    print(f"\n[2/3] Saved {len(sampled_records)} strictly sampled constraints pairs to {args.out_sampled_jsonl}")
    print("      === SAMPLING BREAKDOWN ===")
    print(f"      Case 1 (Requested 40): Extracted {len(s1)} (out of {len(df_c1)} available in pool)")
    print(f"      Case 2 (Requested 30): Extracted {len(s2)} (out of {len(df_c2)} available in pool)")
    print(f"      Case 3 (Requested 30): Extracted {len(s3)} (out of {len(df_c3)} available in pool)\n")

    # Output Requirement 3: Generate the HTML file
    generate_html_report(sampled_records, args.out_html)
    print(f"[3/3] Generated standalone HTML visualization report at {args.out_html}\n")

if __name__ == "__main__":
    main()