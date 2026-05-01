#!/usr/bin/env python3
from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Meta: (in_count, out_count, in_types, out_types, file_path)
FuncMeta = Tuple[int, int, List[str], List[str], str]

def _as_sorted_unique_list(data: Any) -> List[str]:
    """Helper to ensure consistent type-list comparisons."""
    if isinstance(data, list):
        return sorted(list(set(str(v).strip() for v in data if v)))
    return []

def build_func_map(jsonl_path: Path) -> Dict[str, FuncMeta]:
    """Parses the JSONL to extract function metadata AND file paths."""
    func_map = {}
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                for s in obj.get("sources", []):
                    fid = s.get("func_id")
                    if not fid: continue
                    
                    ins = s.get("In", {}).get("In(i)", [])
                    outs = s.get("Out", {}).get("Out(i)", [])
                    in_types = _as_sorted_unique_list(s.get("In", {}).get("InType", {}).get("InType", []))
                    out_types = _as_sorted_unique_list(s.get("Out", {}).get("OutType", []))
                    
                    # Extract the absolute/relative file path
                    file_path = s.get("file", "UNKNOWN_FILE")
                    
                    # Store 5 elements in the tuple now
                    func_map[fid] = (len(ins), len(outs), in_types, out_types, file_path)
            except json.JSONDecodeError: continue
    return func_map

def main() -> int:
    ap = argparse.ArgumentParser(description="RefactorParity: Align Clone Predictions with Ground Truth and Type Parity.")
    ap.add_argument("--jsonl", required=True, type=Path)
    ap.add_argument("--gt", required=True, type=Path, help="Path to test.txt ground truth")
    ap.add_argument("--pred", required=True, type=Path, help="Path to predictions BEFORE adaptation")
    ap.add_argument("--pred_adapt", required=True, type=Path, help="Path to predictions AFTER adaptation")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    # 1. Load Ground Truth mapping
    gt_map = {}
    with args.gt.open("r") as f_gt:
        for line in f_gt:
            parts = line.strip().split()
            if len(parts) >= 3:
                gt_map[(parts[0], parts[1])] = parts[2]

    # 2. Load Adapted Predictions mapping
    adapt_map = {}
    with args.pred_adapt.open("r") as f_adapt:
        for line in f_adapt:
            parts = line.strip().split()
            if len(parts) >= 3:
                adapt_map[(parts[0], parts[1])] = parts[2]

    # 3. Build metadata map
    func_map = build_func_map(args.jsonl)
    
    # 4. Process Predictions and Join Data
    with args.pred.open("r") as fin, args.out.open("w", newline="") as fout:
        writer = csv.writer(fout)
        
        # ADDED 'same_file' AND 'source_filename' to the header
        writer.writerow([
            "pair_left_func", "pair_right_func", "same_file", "source_filename", "actual_label", 
            "clone_predict", "clone_predict_after_adapt", "Refactorable",
            "InCount_L", "InCount_R", "OutCount_L", "OutCount_R",
            "InType_L", "InType_R", "OutType_L", "OutType_R"
        ])
        
        for line in fin:
            parts = line.strip().split()
            if len(parts) < 3: continue
            left_id, right_id, prediction = parts[0], parts[1], parts[2]
            
            actual_label = gt_map.get((left_id, right_id), "N/A")
            prediction_adapt = adapt_map.get((left_id, right_id), "N/A")
            
            meta_l, meta_r = func_map.get(left_id), func_map.get(right_id)
            if not (meta_l and meta_r): continue
            
            # Unpack the 5 variables, including file_path
            l_in_cnt, l_out_cnt, l_in_types, l_out_types, l_file = meta_l
            r_in_cnt, r_out_cnt, r_in_types, r_out_types, r_file = meta_r
            
            # Logic 1: Are they in the same Java file?
            is_same_file = 1 if (l_file == r_file and l_file != "UNKNOWN_FILE") else 0
            
            # Extract the actual filename if they are the same file
            source_filename = l_file if is_same_file == 1 else "N/A"
            
            # Logic 2: Structural AND Type parity
            is_refactorable = (
                (l_in_cnt == r_in_cnt) and
                (l_out_cnt == r_out_cnt) and
                (l_in_types == r_in_types) and
                (l_out_types == r_out_types)
            )
            
            # Helper to format lists for CSV, replacing empty lists with "N/A"
            def fmt_types(t_list: List[str]) -> str:
                return ";".join(t_list) if t_list else "N/A"

            writer.writerow([
                left_id, 
                right_id, 
                is_same_file,          # 1 or 0
                source_filename,       # The path to the java file (or N/A)
                actual_label, 
                prediction, 
                prediction_adapt, 
                1 if is_refactorable else 0,
                l_in_cnt, 
                r_in_cnt, 
                l_out_cnt, 
                r_out_cnt,
                fmt_types(l_in_types), 
                fmt_types(r_in_types),
                fmt_types(l_out_types), 
                fmt_types(r_out_types)
            ])

    print(f"Analysis Complete: Results saved to {args.out}")
    return 0

if __name__ == "__main__":
    sys.exit(main())