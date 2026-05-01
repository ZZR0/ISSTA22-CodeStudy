#!/usr/bin/env python3
import json
import sys
import argparse
from typing import List, Dict, Any

def is_init_function(code: str) -> bool:
    """
    Checks if the function is a Python constructor (__init__).
    """
    if not code:
        return False
    # Simple check for definition line
    for line in code.splitlines():
        # strict check for def __init__
        if line.strip().startswith("def __init__"):
            return True
    return False

def main():
    parser = argparse.ArgumentParser(description="Filter out __init__ methods from JSONL data.")
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    args = parser.parse_args()

    total_count = 0
    kept_count = 0
    excluded_count = 0

    print(f"Processing {args.input} -> {args.output}...", file=sys.stderr)

    with open(args.input, "r", encoding="utf-8") as fin, open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            
            total_count += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(f"[WARN] Skipping invalid JSON at line {total_count}", file=sys.stderr)
                continue

            # Handle sources list or legacy flat format
            sources = obj.get("sources")
            if isinstance(sources, list):
                items_to_check = sources
            else:
                items_to_check = [obj]

            should_exclude_group = False

            for item in items_to_check:
                code = item.get("code", "") or item.get("func", "")
                
                # Check: Is it __init__?
                if is_init_function(code):
                    should_exclude_group = True
                    break

            if should_exclude_group:
                excluded_count += 1
            else:
                kept_count += 1
                fout.write(line + "\n")

    print(f"Done.", file=sys.stderr)
    print(f"Total processed: {total_count}", file=sys.stderr)
    print(f"Kept:            {kept_count}", file=sys.stderr)
    print(f"Excluded:        {excluded_count} (__init__ functions)", file=sys.stderr)

if __name__ == "__main__":
    main()