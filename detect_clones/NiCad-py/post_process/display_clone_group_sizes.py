#!/usr/bin/env python3
import json
import argparse
import sys
import statistics
from collections import Counter

def analyze_group_sizes(file_path):
    group_sizes = []
    
    print(f"[INFO] Reading file: {file_path} ...", file=sys.stderr)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    
                    # Heuristic to find the list of clones
                    # Priority 1: 'sources' list
                    if "sources" in data and isinstance(data["sources"], list):
                        size = len(data["sources"])
                        group_sizes.append(size)
                    # Priority 2: 'nclones' field (if sources is missing, though less reliable for data availability)
                    elif "nclones" in data:
                        size = int(data["nclones"])
                        group_sizes.append(size)
                        
                except json.JSONDecodeError:
                    print(f"[WARN] Line {line_num} is not valid JSON. Skipping.", file=sys.stderr)
                    continue
    except FileNotFoundError:
        print(f"[ERROR] File not found: {file_path}", file=sys.stderr)
        return

    if not group_sizes:
        print("[WARN] No clone groups found.", file=sys.stderr)
        return

    # --- Statistics ---
    total_groups = len(group_sizes)
    avg_size = statistics.mean(group_sizes)
    try:
        median_size = statistics.median(group_sizes)
    except:
        median_size = 0
    max_size = max(group_sizes)
    
    # Count frequencies
    counts = Counter(group_sizes)

    # --- Output Report ---
    print("\n" + "="*65)
    print(f"📊 CLONE GROUP SIZE DISTRIBUTION (Arity)")
    print("="*65)
    print(f"Total Clone Groups: {total_groups}")
    print(f"Mean Group Size:    {avg_size:.2f}")
    print(f"Median Group Size:  {median_size}")
    print(f"Max Group Size:     {max_size}")
    print("-" * 65)
    print(f"{'# Clones (Size)':<15} | {'Count (Groups)':<15} | {'% of Total':<10} | {'Distribution'}")
    print("-" * 65)
    
    # Sort by group size (X-axis)
    sorted_sizes = sorted(counts.keys())
    
    for size in sorted_sizes:
        count = counts[size]
        percentage = (count / total_groups) * 100
        
        # Visualization Bar (Max width 20 chars)
        # Scaled so that 100% = 20 chars
        bar_len = int(percentage / 5) 
        bar = "█" * bar_len
        if percentage > 0 and bar_len == 0:
            bar = "▏" # Small indicator for < 5%
            
        print(f"{size:<15} | {count:<15} | {percentage:6.2f}%    | {bar}")
    print("-" * 65)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze distribution of clone group sizes.")
    # Default changed to step2 file
    parser.add_argument("--input", default="./data/step2_nicad_azure_sim0.7.jsonl", help="Path to the .jsonl input file")
    args = parser.parse_args()
    
    analyze_group_sizes(args.input)