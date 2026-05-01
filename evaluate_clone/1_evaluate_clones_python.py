import argparse
import pandas as pd
import sys
import json
from pathlib import Path
from string import Template

def get_ground_truth(row):
    try:
        left_class = str(row['pair_left_func']).split('_')[1]
        right_class = str(row['pair_right_func']).split('_')[1]
        return 1 if left_class == right_class else 0
    except IndexError:
        return 0

def compute_penalties(row):
    p_detect = 0
    p_in = 0
    p_out = 0
    p_intype = 0
    p_outtype = 0

    if row['ground_truth'] != row['clone_predict']:
        p_detect = -1
        
    if row['clone_predict'] == 1:
        if str(row['pair_left_in']) != str(row['pair_right_in']):
            p_in = -1
        if str(row['pair_left_out']) != str(row['pair_right_out']):
            p_out = -1
        if str(row['pair_inType_ok']).strip().lower() == 'false':
            p_intype = -1
        if str(row['pair_outType_ok']).strip().lower() == 'false':
            p_outtype = -1
            
    total = p_detect + p_in + p_out + p_intype + p_outtype
    return pd.Series([p_detect, p_in, p_out, p_intype, p_outtype, total])

def load_template(template_name: str) -> str:
    path = Path("template") / template_name
    if not path.is_file():
        print(f"Error: Required template file '{path}' is missing.")
        sys.exit(1)
    return path.read_text(encoding="utf-8")

def process_file(input_path: Path, template: Template, base_output_dir: Path):
    """
    Processes a CSV file, generates HTML in the model's subfolder, and returns a list of RCP metrics per pair.
    """
    try:
        df = pd.read_csv(input_path, sep=None, engine='python')
    except Exception as e:
        print(f"Error reading file {input_path.name}: {e}")
        return None

    df['ground_truth'] = df.apply(get_ground_truth, axis=1)
    penalty_cols = ['P_Detect', 'P_In', 'P_Out', 'P_InType', 'P_OutType', 'Total_Penalty']
    df[penalty_cols] = df.apply(compute_penalties, axis=1)

    TP = len(df[(df['ground_truth'] == 1) & (df['clone_predict'] == 1)])
    FP_df = df[(df['ground_truth'] == 0) & (df['clone_predict'] == 1)]
    FN_df = df[(df['ground_truth'] == 1) & (df['clone_predict'] == 0)]
    TN = len(df[(df['ground_truth'] == 0) & (df['clone_predict'] == 0)])

    FP = len(FP_df)
    FN = len(FN_df)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    overall_penalty = df['Total_Penalty'].sum()
    total_pairs = len(df)
    
    actual_clones = len(df[df['ground_truth'] == 1])
    avg_penalty = overall_penalty / actual_clones if actual_clones > 0 else 0.0

    print(f"\n=== Evaluation Results for: {input_path.name} ===")
    print(f"Dataset Size: {total_pairs} pairs evaluated")
    print(f"TP: {TP} | FP: {FP} | TN: {TN} | FN: {FN}")
    print(f"Total Penalty: {overall_penalty} (Average per true clone: {avg_penalty:.2f})")

    stage = "unknown"
    if "before" in str(input_path).lower(): stage = "before"
    elif "after" in str(input_path).lower(): stage = "after"
        
    model = input_path.parent.name
    model_output_dir = base_output_dir / model
    model_output_dir.mkdir(parents=True, exist_ok=True)
    
    file_stem = input_path.stem
    if file_stem.startswith("pred_"):
        subject = file_stem[5:].split('_')[0]
        file_stem_clean = file_stem[5:]
    else:
        subject = file_stem.split('_')[0]
        file_stem_clean = file_stem
        
    html_report_path = model_output_dir / f"report_{stage}_{model}_{file_stem_clean}.html"
    
    display_cols = [
        'pair_left_func', 'pair_right_func', 'clone_predict', 'ground_truth', 
        'P_Detect', 'P_In', 'P_Out', 'P_InType', 'P_OutType', 'Total_Penalty'
    ]
    
    fp_html = FP_df[display_cols].to_html(index=False, border=1, classes="data-table") if not FP_df.empty else "<p><i>No False Positives found!</i></p>"
    fn_html = FN_df[display_cols].to_html(index=False, border=1, classes="data-table") if not FN_df.empty else "<p><i>No False Negatives found!</i></p>"
    
    html_content = template.safe_substitute(
        stage=stage.capitalize(),
        model=model,
        input_filename=input_path.name,
        total_pairs=total_pairs,
        tp=TP, fp=FP, tn=TN, fn=FN,
        precision=f"{precision:.4f}",
        recall=f"{recall:.4f}",
        f1_score=f"{f1_score:.4f}",
        total_penalty=overall_penalty,
        avg_penalty=f"{avg_penalty:.2f}",
        fp_html=fp_html,
        fn_html=fn_html
    )
    
    with open(html_report_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"--> Saved RCP Report to: {html_report_path}")

    # --- MODIFIED SECTION: Return individual row scores instead of aggregate file score ---
    records = []
    for index, row in df.iterrows():
        # Fallback to row index if 'id' column isn't present in your CSV
        pair_id = row['id'] if 'id' in df.columns else index
        
        records.append({
            "llm": model,
            "subject": f"{subject}-{pair_id}",
            "adapt": stage,
            "score": int(row['Total_Penalty'])
        })

    return records

def main():
    parser = argparse.ArgumentParser(description="Calculate metrics and generate HTML error reports.")
    parser.add_argument("--input", type=str, help="Path to a single CSV/TSV file.")
    parser.add_argument("--input-dir", type=str, help="Base data directory (e.g., ./data).")
    parser.add_argument("--input-model", type=str, help="Comma-separated list of models (e.g., codegpt,codebert).")
    parser.add_argument("--output", type=str, required=True, help="Base directory to save outputs.")
    
    args = parser.parse_args()

    if not args.input and not args.input_dir:
        print("Error: You must provide either --input or --input-dir.")
        sys.exit(1)

    base_output_dir = Path(args.output)
    base_output_dir.mkdir(parents=True, exist_ok=True)

    html_template_str = load_template("report_template.html")
    template = Template(html_template_str)

    results = []

    # 1. Single File Override
    if args.input:
        input_path = Path(args.input)
        if input_path.is_file():
            res = process_file(input_path, template, base_output_dir)
            if res: results.extend(res) # Changed from append to extend

    # 2. Directory & Model Traversal
    if args.input_dir:
        base_dir = Path(args.input_dir)
        
        # If specific models are provided, traverse the strict structure
        if args.input_model:
            models = [m.strip() for m in args.input_model.split(",")]
            stages = ["before", "after"]
            
            # Pre-clear old JSONL files so data doesn't duplicate on re-runs
            for m in models:
                jsonl_file = base_output_dir / m / "rcp_scores.jsonl"
                if jsonl_file.exists():
                    jsonl_file.unlink()

            for stage in stages:
                for model in models:
                    target_dir = base_dir / f"pred_{stage}_adapt_refactorability" / model
                    
                    if target_dir.is_dir():
                        csv_files = list(target_dir.glob("*.csv"))
                        if not csv_files:
                            print(f"Warning: No CSV files found in {target_dir}")
                            
                        for csv_file in csv_files:
                            res = process_file(csv_file, template, base_output_dir)
                            if res: results.extend(res) # Changed from append to extend
                    else:
                        print(f"Warning: Directory not found: {target_dir}")
                        
        # Fallback for simple flat-folder processing
        else:
            if base_dir.is_dir():
                for csv_file in base_dir.glob("*.csv"):
                    res = process_file(csv_file, template, base_output_dir)
                    if res: results.extend(res) # Changed from append to extend

    # 3. Save JSONL data grouped by model
    if results:
        models_processed = set(r["llm"] for r in results)
        
        for m in models_processed:
            model_results = [r for r in results if r["llm"] == m]
            model_dir = base_output_dir / m
            model_dir.mkdir(parents=True, exist_ok=True)
            
            jsonl_output_path = model_dir / "rcp_scores.jsonl"
            
            # Safe to append because we deleted the old file at the start of the script
            with open(jsonl_output_path, "a", encoding="utf-8") as f:
                for r in model_results:
                    f.write(json.dumps(r) + "\n")
                    
            print(f"\n--> Saved {len(model_results)} individual pair scores to: {jsonl_output_path}")

if __name__ == "__main__":
    main()