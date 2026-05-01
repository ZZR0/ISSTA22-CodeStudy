import argparse
import json
import re
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import defaultdict

# Default input file
DEFAULT_INPUT = "../data/java/nicad_camel_clone_func.jsonl"
OUTPUT_IMAGE = "token_distribution_combined.png"

def java_tokenize_standard(code):
    """
    Standard lexical tokenizer for Java.
    Counts: Keywords, Identifiers, Literals, Operators.
    Ignores: Whitespace, Comments.
    """
    token_pattern = re.compile(
        r'''
        "(?:\\.|[^\\"])*"         |  # String Literal
        '(?:\\.|[^\\'])*'         |  # Char Literal
        //.*?$                    |  # Line Comment
        /\*.*?\*/                 |  # Block Comment
        \b0[xX][0-9a-fA-F]+\b     |  # Hex Number
        \b[0-9]+\.?[0-9]*(?:[eE][-+]?[0-9]+)?\b | # Decimal Number
        @[a-zA-Z_$][a-zA-Z0-9_$]* |  # Annotation
        [a-zA-Z_$][a-zA-Z0-9_$]* |  # Identifier / Keyword
        [(){}\[\],.;:?!~+\-*/%&|^=<>]+ # Operators & Punctuation
        ''',
        re.VERBOSE | re.MULTILINE | re.DOTALL
    )
    
    raw_matches = token_pattern.findall(code)
    clean_tokens = [t for t in raw_matches if not (t.startswith('//') or t.startswith('/*'))]
    return clean_tokens

def main():
    parser = argparse.ArgumentParser(description="Plot combined token distribution and list duplicates.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input JSONL file")
    parser.add_argument("--output", default=OUTPUT_IMAGE, help="Output image filename")
    args = parser.parse_args()

    print(f"--- Reading Data: {args.input} ---")
    
    if not os.path.exists(args.input):
        print(f"Error: File {args.input} not found.")
        return

    all_token_counts = []           # List for ALL instances
    unique_token_map = {}           # Dict for UNIQUE instances {qualified_name: count}
    frequency_map = defaultdict(int) # Dict to track frequency {qualified_name: occurrences}
    total_groups = 0

    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    data = json.loads(line)
                    total_groups += 1
                    
                    sources = data.get("sources", [])
                    for src in sources:
                        code = src.get("code", "")
                        q_name = src.get("qualified_name")
                        
                        if code:
                            tokens = java_tokenize_standard(code)
                            cnt = len(tokens)
                            
                            # 1. Track All Instances
                            all_token_counts.append(cnt)
                            
                            # 2. Track Unique Functions & Frequency
                            if q_name:
                                unique_token_map[q_name] = cnt
                                frequency_map[q_name] += 1
                            
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    if not all_token_counts:
        print("No data found.")
        return

    # Convert to numpy arrays
    all_counts = np.array(all_token_counts)
    unique_counts = np.array(list(unique_token_map.values()))
    
    # Identify Duplicates
    duplicates = [(name, count) for name, count in frequency_map.items() if count > 1]
    # Sort duplicates by frequency (descending)
    duplicates.sort(key=lambda x: x[1], reverse=True)

    # --- Statistics Report ---
    print(f"\n=== Analysis Report ===")
    print(f"Total Clone Groups: {total_groups}")
    
    print(f"\n[1] All Instances (Volume)")
    print(f"  Count:   {len(all_counts)}")
    print(f"  Avg:     {np.mean(all_counts):.2f}")
    print(f"  Max:     {np.max(all_counts)}")
    
    print(f"\n[2] Unique Functions (Diversity)")
    print(f"  Count:   {len(unique_counts)}")
    print(f"  Avg:     {np.mean(unique_counts):.2f}")
    print(f"  Max:     {np.max(unique_counts)}")

    print(f"\n[3] Duplication Impact")
    print(f"  Duplicated Instances: {len(all_counts) - len(unique_counts)}")
    print(f"  Duplication Ratio:    {1.0 - (len(unique_counts) / len(all_counts)):.2%}")

    print(f"\n[4] List of Duplicated Functions ({len(duplicates)} distinct functions found multiple times)")
    if duplicates:
        print(f"{'Count':<6} | {'Qualified Name'}")
        print("-" * 80)
        for name, count in duplicates:
            print(f"{count:<6} | {name}")
    else:
        print("  None found.")

    # --- Plotting ---
    plt.figure(figsize=(12, 6))

    # Plot Total Instances first (Background, Lighter)
    plt.hist(all_counts, bins=100, color='silver', edgecolor='gray', alpha=0.5, log=True, label='All Instances (Includes Duplicates)')
    
    # Plot Unique Functions on top (Foreground, Darker)
    plt.hist(unique_counts, bins=100, color='teal', edgecolor='black', alpha=0.7, log=True, label='Unique Functions (Distinct Logic)')
    
    plt.title(f'Standard Token Distribution: All vs. Unique', fontsize=14)
    plt.xlabel('Number of Standard Tokens (Logical Size)', fontsize=12)
    plt.ylabel('Frequency (Log Scale)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.legend(loc='upper right', fontsize=10)
    plt.tight_layout()

    # Save
    plt.savefig(args.output)
    print(f"\nGraph saved to: {args.output}")

if __name__ == "__main__":
    main()