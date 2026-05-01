#!/usr/bin/env python3
import argparse
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

def calc_metrics(y_true, y_pred):
    if len(y_true) == 0:
        return 0, 0, 0
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f = f1_score(y_true, y_pred, zero_division=0)
    return p, r, f

def main():
    parser = argparse.ArgumentParser(description="Analyze Model Adaptation Impact on Refactorability")
    parser.add_argument("--csv", required=True, help="Path to the refactor clone eval CSV")
    parser.add_argument("--log", required=False, help="Optional: Path to save the output text log")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    # We will store all print statements in a list to write to the log later
    output = []
    
    output.append(f"========== ANALYSIS REPORT: {args.csv} ==========\n")

    # 1. Overall Performance
    output.append("=== 1. OVERALL PERFORMANCE ===")
    p_b, r_b, f_b = calc_metrics(df['actual_label'], df['clone_predict'])
    p_a, r_a, f_a = calc_metrics(df['actual_label'], df['clone_predict_after_adapt'])
    output.append(f"Before Adapt : Precision={p_b:.4f}, Recall={r_b:.4f}, F1={f_b:.4f}")
    output.append(f"After Adapt  : Precision={p_a:.4f}, Recall={r_a:.4f}, F1={f_a:.4f}\n")

    # 2. Performance on Refactorable Candidates
    output.append("=== 2. REFACTORABILITY-AWARE PERFORMANCE (Refactorable == 1) ===")
    df_ref1 = df[df['Refactorable'] == 1]
    p_b, r_b, f_b = calc_metrics(df_ref1['actual_label'], df_ref1['clone_predict'])
    p_a, r_a, f_a = calc_metrics(df_ref1['actual_label'], df_ref1['clone_predict_after_adapt'])
    output.append(f"Total Refactorable Pairs: {len(df_ref1)}")
    output.append(f"Before Adapt : Precision={p_b:.4f}, Recall={r_b:.4f}, F1={f_b:.4f}")
    output.append(f"After Adapt  : Precision={p_a:.4f}, Recall={r_a:.4f}, F1={f_a:.4f}\n")

    # 3. Transition Analysis (The "Fix" Rate)
    output.append("=== 3. ADAPTATION IMPACT ON REFACTORABLE CLONES ===")
    missed_before = df_ref1[(df_ref1['actual_label'] == 1) & (df_ref1['clone_predict'] == 0)]
    output.append(f"Missed refactorable clones BEFORE adapt (False Negatives): {len(missed_before)}")
    
    fixed = missed_before[missed_before['clone_predict_after_adapt'] == 1]
    fix_rate = (len(fixed) / len(missed_before)) * 100 if len(missed_before) > 0 else 0
    output.append(f"  -> Successfully recovered by adaptation: {len(fixed)} ({fix_rate:.1f}% recovery rate)")
    
    newly_missed = df_ref1[(df_ref1['actual_label'] == 1) & (df_ref1['clone_predict'] == 1) & (df_ref1['clone_predict_after_adapt'] == 0)]
    output.append(f"Newly missed by adaptation (Regressions): {len(newly_missed)}\n")

    output.append("=====================================================")

    # Combine all lines into a single string
    final_output_text = "\n".join(output)

    # Print to console
    print(final_output_text)

    # Save to log file if the argument was provided
    if args.log:
        with open(args.log, "w", encoding="utf-8") as f_log:
            f_log.write(final_output_text)
        print(f"\n[INFO] Report successfully saved to: {args.log}")

if __name__ == "__main__":
    main()