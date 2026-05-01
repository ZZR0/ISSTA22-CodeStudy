import argparse
import json
import sys
import re
import numpy as np

# Default input file (Change this to your actual BigCloneBench data path)
DEFAULT_INPUT = "data/data.jsonl"

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
        //.*?$                    |  # Line Comment (matches but we ignore)
        /\*.*?\*/                 |  # Block Comment (matches but we ignore)
        \b0[xX][0-9a-fA-F]+\b     |  # Hex Number
        \b[0-9]+\.?[0-9]*(?:[eE][-+]?[0-9]+)?\b | # Decimal Number (Non-capturing)
        @[a-zA-Z_$][a-zA-Z0-9_$]* |  # Annotation
        [a-zA-Z_$][a-zA-Z0-9_$]* |  # Identifier / Keyword
        [(){}\[\],.;:?!~+\-*/%&|^=<>]+ # Operators & Punctuation
        ''',
        re.VERBOSE | re.MULTILINE | re.DOTALL
    )
    
    # Find all matches
    raw_matches = token_pattern.findall(code)
    
    # Filter out comments (start with // or /*)
    clean_tokens = [t for t in raw_matches if not (t.startswith('//') or t.startswith('/*'))]
    
    return clean_tokens

def main():
    parser = argparse.ArgumentParser(description="Count tokens for BigCloneBench format data.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input JSONL file (BigCloneBench format)")
    args = parser.parse_args()

    print(f"--- Reading Data: {args.input} ---")
    
    token_counts = []
    function_details = [] 

    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line: continue
                try:
                    data = json.loads(line)
                    
                    # --- BigCloneBench Format Parsing ---
                    # Format: {"func": "code...", "idx": "12345"}
                    code = data.get("func", "")
                    func_id = data.get("idx", "Unknown")
                    
                    if not code:
                        continue
                        
                    # Use Standard Tokenizer
                    tokens = java_tokenize_standard(code)
                    count = len(tokens)
                    
                    token_counts.append(count)
                    function_details.append((count, func_id))

                    if i % 5000 == 0:
                        sys.stdout.write(f"\rProcessed {i} lines...")
                        sys.stdout.flush()

                except json.JSONDecodeError:
                    continue

    except FileNotFoundError:
        print(f"Error: File {args.input} not found.")
        return

    print(f"\n\n=== BigCloneBench Token Analysis ===")
    
    if not token_counts:
        print("No valid data found.")
        return

    counts = np.array(token_counts)
    
    print(f"[1] Statistics (Whitespace Ignored)")
    print(f"  Count:   {len(counts)}")
    print(f"  Min:     {np.min(counts)}")
    print(f"  Max:     {np.max(counts)}")
    print(f"  Avg:     {np.mean(counts):.2f}")
    print(f"  Median:  {np.median(counts):.2f}")
    
    # Show Top 5 Largest Functions (Standard Count)
    # Sorting by count descending
    function_details.sort(key=lambda x: x[0], reverse=True)
    
    print(f"\n[2] Top 5 Largest Functions (Standard Count)")
    for rank, (cnt, idx) in enumerate(function_details[:5], 1):
        print(f"  #{rank}: {cnt} tokens - ID: {idx}")

    # Recommendation for block_size based on BigCloneBench stats
    max_tokens = np.max(counts)
    # Rough estimate: 1 Standard Token ~= 2.5 BPE Tokens (conservative)
    estimated_bpe_max = max_tokens * 2.5 
    
    print(f"\n[3] Configuration Suggestion")
    print(f"  Max Standard Tokens: {max_tokens}")
    print(f"  Est. BPE Max Tokens: ~{int(estimated_bpe_max)}")
    
    if estimated_bpe_max < 512:
        print("  >>> Recommended --block_size: 512 (Safe)")
    else:
        print("  >>> Warning: Some functions might be truncated with block_size 512.")
        print("      Consider checking the truncation ratio.")

if __name__ == "__main__":
    main()