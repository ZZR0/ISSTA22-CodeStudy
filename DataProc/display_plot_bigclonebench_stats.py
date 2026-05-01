import argparse
import json
import re
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_INPUT = "data/data.jsonl"
OUTPUT_IMAGE = "token_distribution_bigclonebench.png"

def java_tokenize_standard(code):
    """
    Standard lexical tokenizer for Java (Refined Version).
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
    parser = argparse.ArgumentParser(description="Plot Standard Token distribution.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input JSONL file")
    parser.add_argument("--output", default=OUTPUT_IMAGE, help="Output image filename")
    args = parser.parse_args()

    print(f"--- Reading Data: {args.input} ---")
    
    token_counts = []

    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    data = json.loads(line)
                    code = data.get("func", "")
                    if code:
                        tokens = java_tokenize_standard(code)
                        token_counts.append(len(tokens))
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        print(f"Error: File {args.input} not found.")
        return

    if not token_counts:
        print("No data found.")
        return

    counts = np.array(token_counts)
    total_funcs = len(counts)
    
    # Calculate Statistics based purely on Standard Tokens
    avg_len = np.mean(counts)
    p95 = np.percentile(counts, 95)
    p99 = np.percentile(counts, 99)
    max_len = np.max(counts)

    print(f"Analyzed {total_funcs} functions.")
    print(f"  Avg: {avg_len:.2f}")
    print(f"  95th %: {p95:.2f}")
    print(f"  99th %: {p99:.2f}")
    print(f"  Max: {max_len}")

    # --- Plotting ---
    plt.figure(figsize=(12, 6))

    # Histogram (Standard Tokens)
    n, bins, patches = plt.hist(counts, bins=100, color='royalblue', edgecolor='black', alpha=0.7, log=True)
    
    plt.title(f'Standard Token Distribution (BigCloneBench) - Total {total_funcs} functions', fontsize=14)
    plt.xlabel('Number of Standard Tokens (Keywords, Identifiers, Operators)', fontsize=12)
    plt.ylabel('Frequency (Log Scale)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # Reference Lines: Percentiles (Facts about the data, NOT Transformer limits)
    plt.axvline(x=p95, color='orange', linestyle='--', linewidth=2, label=f'95th Percentile ({int(p95)} tokens)')
    plt.axvline(x=p99, color='red', linestyle='--', linewidth=2, label=f'99th Percentile ({int(p99)} tokens)')
    plt.axvline(x=avg_len, color='green', linestyle='-', linewidth=2, label=f'Average ({int(avg_len)} tokens)')

    plt.legend()
    plt.tight_layout()

    # Save
    plt.savefig(args.output)
    print(f"\nGraph saved to: {args.output}")

if __name__ == "__main__":
    main()