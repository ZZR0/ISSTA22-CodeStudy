import argparse
import json
import sys
import re
import numpy as np

DEFAULT_INPUT = "data/nicad_camel_clone_func.jsonl"

def java_tokenize(code):
    """
    A standard lexical tokenizer for Java.
    It splits code into meaningful tokens:
    1. String Literals ("hello")
    2. Char Literals ('c')
    3. Identifiers & Keywords (public, void, myVar)
    4. Numbers (123, 0.45)
    5. Operators & Punctuation (+, =, {, }, ;)
    6. Annotations (@Override)
    
    It ignores:
    - Whitespace (Space, Tab, Newline)
    - Comments (Assumed stripped, but regex handles them naturally by not matching)
    """
    token_pattern = re.compile(
        r'''
        "(?:\\.|[^\\"])*"         |  # String Literal
        '(?:\\.|[^\\'])*'         |  # Char Literal
        //.*?$                    |  # Line Comment (matches but we ignore in count)
        /\*.*?\*/                 |  # Block Comment (matches but we ignore in count)
        \b0[xX][0-9a-fA-F]+\b     |  # Hex Number
        \b[0-9]+\.?[0-9]*(?:[eE][-+]?[0-9]+)?\b | # Decimal Number (FIXED: Non-capturing)
        @[a-zA-Z_$][a-zA-Z0-9_$]* |  # Annotation
        [a-zA-Z_$][a-zA-Z0-9_$]* |  # Identifier / Keyword
        [(){}\[\],.;:?!~+\-*/%&|^=<>]+ # Operators & Punctuation
        ''',
        re.VERBOSE | re.MULTILINE | re.DOTALL
    )
    
    # Find all matches
    raw_matches = token_pattern.findall(code)
    
    # Filter out comments if they were captured (just in case)
    # A simple match for comment start
    clean_tokens = [t for t in raw_matches if not (t.startswith('//') or t.startswith('/*'))]
    
    return clean_tokens

def main():
    parser = argparse.ArgumentParser(description="Count tokens using standard lexical rules (ignoring whitespace).")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input JSONL file")
    args = parser.parse_args()

    print(f"--- Reading Data: {args.input} ---")
    
    token_counts = []
    function_details = [] # Store (count, name) for ranking

    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line: continue
                try:
                    data = json.loads(line)
                    sources = data.get("sources", [])
                    
                    for src in sources:
                        code = src.get("code", "")
                        name = src.get("qualified_name", src.get("file", "Unknown"))
                        
                        # Use Standard Tokenizer
                        tokens = java_tokenize(code)
                        count = len(tokens)
                        
                        token_counts.append(count)
                        function_details.append((count, name))

                    if i % 1000 == 0:
                        sys.stdout.write(f"\rProcessed {i} lines...")
                        sys.stdout.flush()

                except json.JSONDecodeError:
                    continue

    except FileNotFoundError:
        print(f"Error: File {args.input} not found.")
        return

    print(f"\n\n=== Standard Lexical Token Analysis ===")
    
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
    
    print(f"\n[2] Comparison Perspective")
    print(f"  * This count represents the 'logical size' of the code.")
    print(f"  * GPT-2 Tokenizer count will be ~2x to 3x higher than this due to whitespace.")

    # Show Top 5 Smallest Functions
    function_details.sort(key=lambda x: x[0])
    print(f"\n[3] Top 5 Smallest Functions (Standard Count)")
    for rank, (cnt, name) in enumerate(function_details[:5], 1):
        print(f"  #{rank}: {cnt} tokens - {name}")

    # Show Top 5 Largest Functions
    print(f"\n[4] Top 5 Largest Functions (Standard Count)")
    for rank, (cnt, name) in enumerate(function_details[-5:], 1):
        print(f"  #{rank}: {cnt} tokens - {name}")

if __name__ == "__main__":
    main()