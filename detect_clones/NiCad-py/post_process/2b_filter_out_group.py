#!/usr/bin/env python3
import json
import argparse
import sys
import statistics
from collections import Counter

def process_and_filter_groups(input_path, output_path, min_size=None, max_size=None):
    group_sizes = []
    kept_count = 0
    excluded_count = 0
    
    print(f"[INFO] Reading file: {input_path} ...", file=sys.stderr)
    print(f"[INFO] Filtering criteria: Min Size={min_size}, Max Size={max_size}", file=sys.stderr)
    
    try:
        with open(input_path, 'r', encoding='utf-8') as fin, \
             open(output_path, 'w', encoding='utf-8') as fout:
            
            for line_num, line in enumerate(fin, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    
                    # Determine group size
                    size = 0
                    if "sources" in data and isinstance(data["sources"], list):
                        size = len(data["sources"])
                    elif "nclones" in data:
                        size = int(data["nclones"])
                    else:
                        # If size cannot be determined, we default to keeping it (or skipping)
                        # Here assuming valid clone groups have sources/nclones
                        continue

                    group_sizes.append(size)

                    # Filtering Logic
                    # If min_size is set and size < min_size -> Exclude
                    if min_size is not None and size < min_size:
                        excluded_count += 1
                        continue
                    
                    # If max_size is set and size > max_size -> Exclude
                    if max_size is not None and size > max_size:
                        excluded_count += 1
                        continue

                    # Keep the group
                    fout.write(line + "\n")
                    kept_count += 1
                        
                except json.JSONDecodeError:
                    print(f"[WARN] Line {line_num} is not valid JSON. Skipping.", file=sys.stderr)
                    continue
                    
    except FileNotFoundError:
        print(f"[ERROR] File not found: {input_path}", file=sys.stderr)
        return

    if not group_sizes:
        print("[WARN] No clone groups found.", file=sys.stderr)
        return

    # --- Statistics Report ---
    total_groups = len(group_sizes)
    avg_size = statistics.mean(group_sizes)
    try:
        median_size = statistics.median(group_sizes)
    except:
        median_size = 0
    max_group_size = max(group_sizes)
    
    counts = Counter(group_sizes)

    print("\n" + "="*65)
    print(f"📊 CLONE GROUP SIZE DISTRIBUTION (Input Data)")
    print("="*65)
    print(f"Total Clone Groups: {total_groups}")
    print(f"Mean Group Size:    {avg_size:.2f}")
    print(f"Median Group Size:  {median_size}")
    print(f"Max Group Size:     {max_group_size}")
    print("-" * 65)
    print(f"{'# Clones (Size)':<15} | {'Count (Groups)':<15} | {'% of Total':<10} | {'Distribution'}")
    print("-" * 65)
    
    sorted_sizes = sorted(counts.keys())
    for size in sorted_sizes:
        count = counts[size]
        percentage = (count / total_groups) * 100
        bar_len = int(percentage / 5) 
        bar = "█" * bar_len
        if percentage > 0 and bar_len == 0:
            bar = "▏"
        
        # Mark filtered rows visually
        status = ""
        if (min_size is not None and size < min_size) or \
           (max_size is not None and size > max_size):
            status = " [EXCLUDED]"
            
        print(f"{size:<15} | {count:<15} | {percentage:6.2f}%    | {bar}{status}")
    
    print("-" * 65)
    print(f"[RESULT] Total Groups Processed: {total_groups}")
    print(f"[RESULT] Groups Kept:            {kept_count}")
    print(f"[RESULT] Groups Excluded:        {excluded_count}")
    print(f"[RESULT] Output saved to:        {output_path}")
    print("="*65)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter clone groups by size (arity) and analyze distribution.")
    
    parser.add_argument("--input", default="./data/step2_nicad_azure_sim0.7.jsonl", 
                        help="Path to the input .jsonl file")
    parser.add_argument("--output", default="./data/step3_nicad_azure_sim0.7.jsonl", 
                        help="Path to the output .jsonl file")
    
    parser.add_argument("--min-size", type=int, default=None,
                        help="Minimum number of clones in a group to keep (inclusive).")
    parser.add_argument("--max-size", type=int, default=None,
                        help="Maximum number of clones in a group to keep (inclusive).")
    
    args = parser.parse_args()
    
    process_and_filter_groups(args.input, args.output, args.min_size, args.max_size)