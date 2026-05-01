import json
import csv
import re

def count_params(signature_str):
    """Helper to count parameters in a method signature string."""
    if not signature_str or signature_str == "N/A":
        return 0
    match = re.search(r'\((.*?)\)', signature_str)
    if match and match.group(1).strip():
        # Split by comma and filter empty entries
        params = [p.strip() for p in match.group(1).split(',') if p.strip()]
        return len(params)
    return 0

def export_to_flattened_csv(json_path, output_csv="refactoring_analysis_flat.csv"):
    with open(json_path, 'r', encoding='utf-8') as f:
        records = json.load(f)
        
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # Updated Header
        writer.writerow([
            "Sample N", "Clone Class ID", "Instance ID", "IsRefactoredOne", 
            "IsRefactoredAll", "Similarity", "Actual Label", "Clone Predict", 
            "Clone Predict After", "Refactorable", "Extracted Sig", "Ground Truth Sig", "Param Count Diff"
        ])
        
        for i, r in enumerate(records, 1):
            sources = r.get("sources", [])
            success_list = [
                s for s in sources 
                if s.get("ground_truth_after_VSCode_ref", {}).get("extracted_method_code")
            ]
            
            is_one = "Yes" if len(success_list) >= 1 else "No"
            is_all = "Yes" if len(success_list) == len(sources) else "No"
            
            for s in sources:
                gt = s.get("ground_truth_after_VSCode_ref", {})
                
                # Extract signatures
                ext_sig = s.get("Extracted Signature", "N/A")
                gt_sig = gt.get("extracted_method_signature", "N/A")
                
                # Calculate Metric: len(GT_Params) - len(Extracted_Params)
                param_diff = count_params(gt_sig) - count_params(ext_sig)
                
                writer.writerow([
                    i, 
                    r.get("classid"), 
                    s.get("func_id"),
                    is_one, 
                    is_all, 
                    r.get("similarity"), 
                    r.get("actual_label"), 
                    r.get("clone_predict"), 
                    r.get("clone_predict_after_adapt"), 
                    r.get("Refactorable"),
                    ext_sig, 
                    gt_sig,
                    param_diff
                ])
    print(f"Full analysis exported to {output_csv}")

if __name__ == "__main__":
    export_to_flattened_csv('/home/user1-system11/research_dream/llm-clone/extract_clone/output/sampling_clones/sampled_70pairs_after_ref.json')