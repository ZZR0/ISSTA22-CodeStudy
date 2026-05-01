import pandas as pd
import argparse
from pathlib import Path

def get_ground_truth(row):
    try:
        left_class = str(row['pair_left_func']).split('_')[1]
        right_class = str(row['pair_right_func']).split('_')[1]
        return 1 if left_class == right_class else 0
    except IndexError:
        return 0

def compute_penalties(row):
    p_detect = p_in = p_out = p_intype = p_outtype = 0

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

def analyze_subject(model: str, subject: str, base_dir: Path):
    print(f"\n{'='*60}")
    print(f"🔍 Analyzing Outlier: {subject} (Model: {model})")
    print(f"{'='*60}")

    stages = ['before', 'after']
    metrics = {}

    for stage in stages:
        file_path = base_dir / f"pred_{stage}_adapt_refactorability" / model / f"pred_{subject}_with_refactorability.csv"
        
        if not file_path.exists():
            print(f"❌ Error: Could not find file {file_path}")
            return

        df = pd.read_csv(file_path, sep=None, engine='python')
        df['ground_truth'] = df.apply(get_ground_truth, axis=1)
        
        penalty_cols = ['P_Detect', 'P_In', 'P_Out', 'P_InType', 'P_OutType', 'Total_Penalty']
        df[penalty_cols] = df.apply(compute_penalties, axis=1)

        actual_clones = len(df[df['ground_truth'] == 1])
        fp_count = len(df[(df['ground_truth'] == 0) & (df['clone_predict'] == 1)])

        if actual_clones == 0:
            print(f"No true clones found in {stage} dataset.")
            continue

        # Corrected: Sum ALL penalties (including FP hallucinations) to match your main pipeline
        metrics[stage] = {
            'Avg_P_Detect': df['P_Detect'].sum() / actual_clones,
            'Avg_P_In': df['P_In'].sum() / actual_clones,
            'Avg_P_Out': df['P_Out'].sum() / actual_clones,
            'Avg_P_InType': df['P_InType'].sum() / actual_clones,
            'Avg_P_OutType': df['P_OutType'].sum() / actual_clones,
            'Total_Avg': df['Total_Penalty'].sum() / actual_clones,
            'FP_Count': fp_count
        }

    print(f"{'Penalty Type':<20} | {'Before Adapt':<15} | {'After Adapt':<15} | {'Delta (Improvement)'}")
    print("-" * 80)
    
    categories = ['Avg_P_Detect', 'Avg_P_In', 'Avg_P_Out', 'Avg_P_InType', 'Avg_P_OutType', 'Total_Avg']
    labels = ['Detection Error', 'Input Mapping', 'Output Mapping', 'Input Types', 'Output Types', 'OVERALL PENALTY']

    for cat, label in zip(categories, labels):
        val_before = metrics['before'][cat]
        val_after = metrics['after'][cat]
        delta = val_after - val_before
        
        trend = "📉 Worse" if delta < 0 else ("📈 Better" if delta > 0 else "➖ Same")
        print(f"{label:<20} | {val_before:<15.3f} | {val_after:<15.3f} | {delta:+.3f} ({trend})")

    print("-" * 80)
    fp_before = metrics['before']['FP_Count']
    fp_after = metrics['after']['FP_Count']
    fp_trend = "📉 Worse (Spike!)" if fp_after > fp_before else ("📈 Better" if fp_after < fp_before else "➖ Same")
    
    print(f"{'False Positives':<20} | {fp_before:<15} | {fp_after:<15} | {fp_after - fp_before:+} ({fp_trend})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze penalty breakdown for specific outlier projects.")
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., codet5)")
    parser.add_argument("--subject", type=str, required=True, help="Subject project (e.g., commons-codec)")
    parser.add_argument("--dir", type=str, default="./data", help="Base data directory")
    
    args = parser.parse_args()
    analyze_subject(args.model, args.subject, Path(args.dir))