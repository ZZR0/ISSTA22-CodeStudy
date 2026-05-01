import json
import html
import re

def calculate_metrics(orig_code, gt_code, func_info, gt_sig):
    # 1. LOC Metric
    orig_lines = [l for l in orig_code.splitlines() if l.strip()]
    gt_lines = [l for l in gt_code.splitlines() if l.strip()]
    loc_red = ((len(orig_lines) - len(gt_lines)) / len(orig_lines)) * 100 if orig_lines else 0
    
    # 2. Interface Metrics
    # Only count In(i) for the expected signature parameters
    in_params = func_info.get("In", {}).get("In(i)", [])
    
    # Extract signature parameters from string
    sig_match = re.search(r'\((.*?)\)', gt_sig)
    sig_params = []
    if sig_match and sig_match.group(1).strip():
        parts = [p.strip() for p in sig_match.group(1).split(',') if p.strip()]
        sig_params = [p.split()[-1] for p in parts]
    
    # Use len(in_params) as the "Expected" count
    expected_count = len(in_params)
    param_score = "Balanced" if len(sig_params) == expected_count else "Mismatched"
    
    return {
        "loc_reduction": f"{loc_red:.1f}%",
        "param_score": param_score,
        "sig_params_len": len(sig_params),
        "expected_params_len": expected_count
    }

def format_code_html(func_info):
    """Generates the HTML block with highlighted clones."""
    if not func_info: return "No information available."
    
    clone_range = func_info.get("range", "")
    enclosing = func_info.get("enclosing_function", {})
    fun_range = enclosing.get("fun_range", "")
    func_code = enclosing.get("func_code", func_info.get("code", ""))
    use_set = func_info.get("In", {}).get("InType", {}).get("Use(i)", [])
    
    try:
        c_start, c_end = map(int, clone_range.split("-"))
        f_start, f_end = map(int, fun_range.split("-"))
    except: return html.escape(func_code)

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
                if "{" in line: in_signature = False
            else: cls = "before-clone"
        else: cls = "after-clone"
        
        out_html += f'<span class="line-number">{line_num}</span><span class="{cls}">{escaped_line}</span>\n'
    return out_html

def generate_meta_box(func_info):
    """Generates the parameter and return derivation metadata box."""
    ext_sig = html.escape(func_info.get("Extracted Signature", "N/A"))
    in_dict = func_info.get("In", {})
    out_dict = func_info.get("Out", {})
    
    def fmt_bold_set(s): return f"<strong>{html.escape(str(s))}</strong>"
    
    return f"""
    <div style="background-color: #e8f4f8; border: 1px solid #b3d4fc; padding: 8px; margin-top: 10px; border-radius: 4px; font-family: monospace; font-size: 0.9em;">
        <strong>Extracted Signature:</strong><br>{ext_sig}<br><br>
        <strong>Parameters Derivation (In(i)):</strong><br>
        {html.escape(str(in_dict.get("In(i)", "N/A")))}<br><br>
        <strong>Return Derivation (Out(i)):</strong><br>
        {html.escape(str(out_dict.get("Out(i)", "N/A")))}
    </div>"""

def generate_html_report(json_path, out_html_path="./output/sampling_clones/refactor_diff_report.html"):
    with open(json_path, 'r', encoding='utf-8') as f:
        records = json.load(f)

    # Calculate global stats
    total_pairs, refactored, failed = 0, 0, 0
    for r in records:
        for f_info in r.get("sources", []):
            total_pairs += 1
            if f_info.get("ground_truth_after_VSCode_ref", {}).get("extracted_method_code"):
                refactored += 1
            else:
                failed += 1

    html_out = [f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Refactoring Diff Report</title>
<style>
    body {{ font-family: sans-serif; background-color: #f4f4f9; padding: 20px; }}
    .stats-box {{ display: flex; gap: 20px; justify-content: center; margin-bottom: 20px; }}
    .stat-card {{ background: #fff; padding: 15px; border-radius: 8px; border: 1px solid #ddd; text-align: center; }}
    .clone-class {{ background: #fff; border: 1px solid #ddd; margin-bottom: 30px; padding: 20px; border-radius: 8px; }}
    .clone-instance {{ margin-top: 15px; border-top: 2px dashed #eee; padding-top: 15px; }}
    .meta-info {{ font-size: 0.9em; color: #555; margin-bottom: 10px; }}
    .diff-container {{ display: flex; gap: 20px; align-items: flex-start; margin-top: 10px; }}
    .diff-col {{ flex: 1; min-width: 0; }}
    pre {{ background-color: #fafafa; border: 1px solid #ccc; padding: 10px; overflow-x: auto; font-size: 12px; line-height: 1.4; }}
    .method-signature {{ color: #8b008b; font-weight: bold; }}
    .before-clone {{ color: blue; }}
    .in-clone {{ color: red; font-weight: bold; }}
    .after-clone {{ color: black; }}
    .line-number {{ display: inline-block; width: 45px; color: #aaa; border-right: 1px solid #ddd; padding-right: 10px; margin-right: 10px; text-align: right; user-select: none; }}
</style>
</head>
<body>
<h1>Refactoring Diff Report</h1>
<div class="stats-box">
    <div class="stat-card"><strong>Total</strong><br>{total_pairs}</div>
    <div class="stat-card" style="color: green;"><strong>Success</strong><br>{refactored}</div>
    <div class="stat-card" style="color: orange;"><strong>Failed</strong><br>{failed}</div>
</div>"""]

    for r in records:
        html_out.append(f'<div class="clone-class"><h2>Clone Class ID: {html.escape(str(r.get("classid", "")))}</h2>')
        for func_info in r.get("sources", []):
            # 1. RESTORED Full Meta-Info
            func_id = html.escape(func_info.get("func_id", "N/A"))
            file_path = html.escape(func_info.get("file", "N/A"))
            clone_range = html.escape(func_info.get("range", "N/A"))
            enclosing = func_info.get("enclosing_function", {})
            enclosing_name = html.escape(enclosing.get("qualified_name", "N/A"))
            enclosing_range = html.escape(enclosing.get("fun_range", "N/A"))
            
            # 2. Extract Ground Truth data
            gt = func_info.get("ground_truth_after_VSCode_ref", {})
            gt_code = html.escape(str(gt.get("extracted_method_code", "N/A")))
            gt_sig = html.escape(str(gt.get("extracted_method_signature", "N/A")))
            
            # 3. Calculate metrics
            orig_code = func_info.get("code") or enclosing.get("func_code", "")
            metrics = calculate_metrics(orig_code, gt_code, func_info, gt_sig)
            
            html_out.append(f"""
<div class="clone-instance">
    <div class="meta-info">
        <strong>Func ID:</strong> {func_id} <br>
        <strong>File:</strong> {file_path} <br>
        <strong>Clone Range:</strong> {clone_range} <br>
        <strong>Enclosing Method:</strong> {enclosing_name} (Lines {enclosing_range})
        {generate_meta_box(func_info)}
    </div>
    <div class="diff-container">
        <div class="diff-col">
            <strong>Original Clone:</strong>
            <pre><code>{format_code_html(func_info)}</code></pre>
        </div>
        <div class="diff-col">
            <strong>Ground Truth(Refatoring with VS Code):</strong>
            <pre><code>{gt_code}</code></pre>
            <div style="margin-top: 10px; padding: 10px; background: #fff3e0; border: 1px solid #ffe0b2; border-radius: 4px; font-size: 0.9em;">
                <strong>Refactor Metrics:</strong><br>
                • LOC Reduction: <strong>{metrics['loc_reduction']}</strong><br>
                • Interface Quality: <strong>{metrics['param_score']}</strong><br>
                (Signature Params: {metrics['sig_params_len']} | Expected: {metrics['expected_params_len']})
            </div>
            <p><strong>Signature:</strong> <code>{gt_sig}</code></p>
        </div>
    </div>
</div>""")
        html_out.append("</div>")
    
    html_out.append("</body></html>")
    with open(out_html_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html_out))
    print(f"Report generated at {out_html_path}")

if __name__ == "__main__":
    generate_html_report('/home/user1-system11/research_dream/llm-clone/extract_clone/output/sampling_clones/sampled_70pairs_after_ref.json')