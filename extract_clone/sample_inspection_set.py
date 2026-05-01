#!/usr/bin/env python3
"""
sample_manual_inspection.py

Goal (100 pairs total, same_file == True always):
- Case 1 (40): both are refactorable (stability on good clones)
- Case 2 (30): baseline missed (0) -> adapt found (1) AND adapt made it refactorable
- Case 3 (30): baseline found (1) but was bad (non-refactorable) -> adapt flagged non-refactorable

Works per-repo CSV produced by clone_refactorability_evaluation.py
Example columns:
pair_left_func,pair_right_func,same_file,actual_label,clone_predict,clone_predict_after_adapt,Refactorable,...
"""

import argparse
import sys
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Sample clone pairs for manual inspection (3 cases).")
    p.add_argument("--csv", required=True, help="Path to *_refactor_clone_eval.csv")
    p.add_argument("--out", required=True, help="Path to save the sampled dataset CSV")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument("--n_case1", type=int, default=40, help="Samples for Case 1 (default: 40)")
    p.add_argument("--n_case2", type=int, default=30, help="Samples for Case 2 (default: 30)")
    p.add_argument("--n_case3", type=int, default=30, help="Samples for Case 3 (default: 30)")
    return p.parse_args()


def ensure_columns(df: pd.DataFrame, required):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def recompute_same_file(df: pd.DataFrame) -> pd.Series:
    """
    Robust same-file check using pair_*_func format: <repo>_<fileid>_<funcid>
    Example: camel_3492_3312 -> fileid=3492
    """
    def file_id(func_name: str):
        parts = str(func_name).split("_")
        return parts[1] if len(parts) >= 3 else None

    left = df["pair_left_func"].map(file_id)
    right = df["pair_right_func"].map(file_id)
    return (left.notna()) & (right.notna()) & (left == right)


def safe_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n <= 0 or len(df) == 0:
        return df.head(0).copy()
    return df.sample(n=min(n, len(df)), random_state=seed)


def main():
    args = parse_args()

    df = pd.read_csv(args.csv)

    required_cols = [
        "pair_left_func", "pair_right_func",
        "actual_label", "clone_predict", "clone_predict_after_adapt",
        "Refactorable",
    ]
    ensure_columns(df, required_cols)

    # Normalize to numeric (some CSVs may contain strings)
    for c in ["actual_label", "clone_predict", "clone_predict_after_adapt", "Refactorable"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["actual_label", "clone_predict", "clone_predict_after_adapt", "Refactorable"]).copy()
    df[["actual_label", "clone_predict", "clone_predict_after_adapt", "Refactorable"]] = df[
        ["actual_label", "clone_predict", "clone_predict_after_adapt", "Refactorable"]
    ].astype(int)

    # Rule 1: same_file == True (Always)
    # Prefer recomputing from func ids (more reliable). If "same_file" exists, we keep it for reference.
    df["same_file_rule"] = recompute_same_file(df)
    df = df[df["same_file_rule"] == True].copy()

    # IMPORTANT: Your requested cases now use "Refactorable" directly (not the old masks).
    #
    # Interpretation based on your text:
    # - Case 1: good clones where refactorable == 1 (stability set)
    # - Case 2: baseline missed clone, adapt found it, AND refactorable == 1 (success story)
    # - Case 3: baseline found clone, but refactorable == 0, adapt flags non-refactorable (filters garbage)
    #
    # Note: Case 1 overlaps with Case 2 if you define it too broadly.
    # To keep cases disjoint, define Case 1 as: refactorable == 1 AND prediction did NOT change (stable)
    # (You said "model is stable on good clones". This means adapt == baseline, and both are correct-ish.)
    #
    # We'll implement Case 1 as:
    #   Refactorable==1 AND clone_predict == clone_predict_after_adapt
    # If you want ONLY positives, add: actual_label==1 and both preds==1.

    mask_case2 = (
        (df["Refactorable"] == 1) &
        (df["actual_label"] == 1) &
        (df["clone_predict"] == 0) &
        (df["clone_predict_after_adapt"] == 1)
    )

    mask_case3 = (
        (df["Refactorable"] == 0) &
        (df["clone_predict"] == 1) &
        (df["clone_predict_after_adapt"] == 0)
    )

    # Case 1 = stability on good clones, disjoint from Case 2:
    mask_case1 = (
        (df["Refactorable"] == 1) &
        (~mask_case2) &
        (df["clone_predict"] == df["clone_predict_after_adapt"])
    )

    df_case1 = df[mask_case1].copy()
    df_case2 = df[mask_case2].copy()
    df_case3 = df[mask_case3].copy()

    df_case1["inspection_case"] = "Case 1 (Stable on Refactorable)"
    df_case2["inspection_case"] = "Case 2 (FN Fixed + Refactorable)"
    df_case3["inspection_case"] = "Case 3 (FP Filtered as Non-Refactorable)"

    # Sample
    s1 = safe_sample(df_case1, args.n_case1, args.seed)
    s2 = safe_sample(df_case2, args.n_case2, args.seed)
    s3 = safe_sample(df_case3, args.n_case3, args.seed)

    final = pd.concat([s1, s2, s3], ignore_index=True)

    # Output columns (keep extra if present)
    preferred_cols = [
        "inspection_case",
        "pair_left_func", "pair_right_func",
        "same_file_rule",
        "actual_label", "clone_predict", "clone_predict_after_adapt",
        "Refactorable",
        "InCount_L", "InCount_R", "OutCount_L", "OutCount_R",
        "InType_L", "InType_R", "OutType_L", "OutType_R",
    ]
    cols_to_write = [c for c in preferred_cols if c in final.columns]
    final[cols_to_write].to_csv(args.out, index=False)

    # Summary
    print(f"\n[SUCCESS] Manual inspection dataset created: {args.out}")
    print("=== FILTER RULES ===")
    print(f"same_file == True (enforced via same_file_rule): remaining {len(df)} rows")
    print("=== CASE POOLS (BEFORE SAMPLING) ===")
    print(f"Case 1 pool: {len(df_case1)}")
    print(f"Case 2 pool: {len(df_case2)}")
    print(f"Case 3 pool: {len(df_case3)}")
    print("=== SAMPLING BREAKDOWN ===")
    print(f"Case 1 requested {args.n_case1}: sampled {len(s1)}")
    print(f"Case 2 requested {args.n_case2}: sampled {len(s2)}")
    print(f"Case 3 requested {args.n_case3}: sampled {len(s3)}")
    print(f"Total pairs extracted: {len(final)}\n")

    # Exit non-zero if you didn't get enough samples (useful for batch scripts)
    missing = (args.n_case1 - len(s1)) + (args.n_case2 - len(s2)) + (args.n_case3 - len(s3))
    if missing > 0:
        print(f"[WARNING] Not enough samples to reach requested total. Missing: {missing}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()