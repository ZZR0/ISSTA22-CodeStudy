import argparse
import json
import sys
import numpy as np
from transformers import AutoTokenizer

# Default model
DEFAULT_MODEL = "microsoft/CodeGPT-small-java-adaptedGPT2"
DEFAULT_INPUT = "data/nicad_camel_clone_func.jsonl"

def main():
    parser = argparse.ArgumentParser(description="Count exact tokens to determine optimal block_size for run.py")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input JSONL file")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HuggingFace model name")
    args = parser.parse_args()

    print(f"--- Loading Tokenizer: {args.model} ---")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    except Exception as e:
        print(f"Error: {e}")
        return

    print(f"--- Reading Data: {args.input} ---")
    
    single_function_lengths = []

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
                        # Count tokens for SINGLE function
                        tokens = tokenizer.tokenize(code)
                        single_function_lengths.append(len(tokens))

                    if i % 500 == 0:
                        sys.stdout.write(f"\rProcessed {i} lines...")
                        sys.stdout.flush()

                except json.JSONDecodeError:
                    continue

    except FileNotFoundError:
        print(f"Error: File {args.input} not found.")
        return

    print(f"\n\n=== Token Analysis for run.py Config ===")
    
    if not single_function_lengths:
        print("No valid data found.")
        return

    singles = np.array(single_function_lengths)
    max_single = np.max(singles)
    avg_single = np.mean(singles)

    print(f"[1] Single Function Statistics")
    print(f"  Count:   {len(singles)}")
    print(f"  Min:     {np.min(singles)}")
    print(f"  Max:     {max_single}  <-- Critical for block_size")
    print(f"  Avg:     {avg_single:.2f}")
    print(f"  99th %:  {np.percentile(singles, 99):.2f}")

    # --- Recommendation Logic Updated based on run.py structure ---
    print(f"\n[2] Recommendation for --block_size")
    print(f"  (Logic: run.py concatenates two blocks. Total Input = 2 * block_size)")
    print(f"  (Constraint: 2 * block_size <= 1024, so block_size MUST be <= 512)")
    
    rec_block = 0
    warning = ""

    if max_single <= 128:
        rec_block = 128
    elif max_single <= 256:
        rec_block = 256
    elif max_single <= 400:
        rec_block = 400
    elif max_single <= 512:
        rec_block = 512
    else:
        rec_block = 512
        warning = f"  [WARNING] Max function length ({max_single}) exceeds hard limit (512).\n  Truncation WILL occur for functions longer than 512 tokens."

    print(f"\n  >>> Recommended --block_size: {rec_block}")
    if warning:
        print(warning)
    else:
        print(f"  >>> This creates a total input length of {rec_block * 2} (Safe within 1024)")

if __name__ == "__main__":
    main()