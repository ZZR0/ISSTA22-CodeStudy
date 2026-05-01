#!/usr/bin/env python3
"""
3_verify_data.py

Verify that every (id1, id2) in pair files exists in the mapping jsonl keys.

Usage Example:
  python3 3_verify_data.py \
    --mapping_jsonl ../dataset/mix_azure/data.jsonl \
    --train_pairs ../dataset/train_mix_azure.txt \
    --valid_pairs ../dataset/valid_mix_azure.txt \
    --test_pairs ../dataset/test.txt

Exit code:
  - 0 if pass (or non-strict fail)
  - 1 if --strict and any missing found
  - 2 if input files not found
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def load_keys(jsonl_path: Path, key_field: str = "idx") -> set:
    keys = set()
    bad = 0
    total = 0
    
    print(f"[LOAD] Loading keys from: {jsonl_path}")
    
    try:
        with jsonl_path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                total += 1
                try:
                    js = json.loads(line)
                except Exception:
                    bad += 1
                    continue
                
                if key_field in js:
                    keys.add(str(js[key_field]))
                else:
                    bad += 1
    except FileNotFoundError:
        print(f"[ERR] JSONL file not found: {jsonl_path}", file=sys.stderr)
        sys.exit(2)

    print(f"       -> Found {len(keys)} unique keys (lines={total}, bad/missing_key={bad})")
    return keys


def verify_pairs(pair_path: Path, keys: set, max_examples: int = 10) -> Dict:
    total_lines = 0
    parsed = 0
    bad_format = 0

    missing_pairs = 0
    missing_id_occ = 0
    examples: List[Tuple[int, str, str, str, str]] = []

    with pair_path.open("r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f, start=1):
            total_lines += 1
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            
            # Expecting: id1 id2 label (at least 3 columns)
            if len(parts) < 3:
                bad_format += 1
                continue

            parsed += 1
            a, b, y = parts[0], parts[1], parts[2]
            
            in_a = a in keys
            in_b = b in keys

            if not (in_a and in_b):
                missing_pairs += 1
                if not in_a:
                    missing_id_occ += 1
                if not in_b:
                    missing_id_occ += 1

                if len(examples) < max_examples:
                    which = []
                    if not in_a:
                        which.append("id1")
                    if not in_b:
                        which.append("id2")
                    examples.append((i, a, b, y, ",".join(which)))

    return {
        "file": str(pair_path),
        "total_lines": total_lines,
        "parsed_pairs": parsed,
        "bad_format": bad_format,
        "missing_pairs": missing_pairs,
        "missing_id_occ": missing_id_occ,
        "examples": examples,
    }


def print_report(rep: Dict):
    print(f"\n[CHECK] {rep['file']}")
    print(f"  - total lines     : {rep['total_lines']}")
    print(f"  - parsed pairs    : {rep['parsed_pairs']}")
    print(f"  - bad format lines: {rep['bad_format']}")
    print(f"  - missing pairs   : {rep['missing_pairs']}")
    print(f"  - missing ID occ  : {rep['missing_id_occ']}")
    if rep["examples"]:
        print("  - examples (line_no, id1, id2, label, missing):")
        for ex in rep["examples"]:
            print(f"    * {ex[0]}  {ex[1]}  {ex[2]}  {ex[3]}   missing={ex[4]}")
    else:
        print("  - examples: none (all resolvable)")


def main():
    ap = argparse.ArgumentParser(description="Verify pair files against mix/data.jsonl mapping keys.")
    
    # Mapping file
    ap.add_argument("--mapping_jsonl", type=str, required=True,
                    help="Path to mapping JSONL file (e.g., ../dataset/mix_azure/data.jsonl)")
    
    ap.add_argument("--key_field", type=str, default="idx",
                    help="Key field in JSONL objects (default: idx)")

    # Pair files
    ap.add_argument("--train_pairs", type=str, required=True,
                    help="Train pairs file (e.g., ../dataset/train_mix_azure.txt)")
    
    ap.add_argument("--valid_pairs", type=str, default=None,
                    help="Valid pairs file (optional)")
    
    ap.add_argument("--test_pairs", type=str, default=None,
                    help="Test pairs file (optional)")

    # Legacy support / Helper for default file naming
    ap.add_argument("--otherdomain_name", type=str, default=None,
                    help="If provided and --test_pairs is missing, infers test file as ../dataset/test_<name>.txt")

    ap.add_argument("--max_examples", type=int, default=10,
                    help="Max number of missing examples to print per file (default: 10)")
    
    ap.add_argument("--strict", action="store_true",
                    help="Exit with code 1 if any missing pairs are found.")

    args = ap.parse_args()

    # 1. Resolve Mapping Path
    mapping_path = Path(args.mapping_jsonl)
    if not mapping_path.exists():
        print(f"[ERR] mapping_jsonl not found: {mapping_path}", file=sys.stderr)
        sys.exit(2)

    # 2. Resolve Pair Files
    pair_files = []

    # Train (Required)
    pair_files.append(("train", Path(args.train_pairs)))

    # Valid (Optional but recommended)
    if args.valid_pairs:
        pair_files.append(("valid", Path(args.valid_pairs)))

    # Test (Flexible logic)
    test_path = None
    if args.test_pairs:
        test_path = Path(args.test_pairs)
    elif args.otherdomain_name:
        # Legacy fallback logic: assume it is in ../dataset/
        test_path = Path("../dataset") / f"test_{args.otherdomain_name}.txt"
    
    if test_path:
        pair_files.append(("test", test_path))

    # 3. Check existence of all pair files before processing
    for name, p in pair_files:
        if not p.exists():
            print(f"[ERR] {name} pair file not found: {p}", file=sys.stderr)
            sys.exit(2)

    # 4. Load Keys
    keys = load_keys(mapping_path, key_field=args.key_field)

    # 5. Verify Files
    overall_missing = 0
    for _, pf in pair_files:
        rep = verify_pairs(pf, keys, max_examples=args.max_examples)
        print_report(rep)
        overall_missing += rep["missing_pairs"]

    print("\n[SUMMARY]")
    if overall_missing == 0:
        print("  ✅ PASS: All pairs are resolvable in the mapping JSONL.")
        sys.exit(0)
    else:
        print(f"  ❌ FAIL: Found missing pairs total = {overall_missing}")
        if args.strict:
            sys.exit(1)
        else:
            sys.exit(0)


if __name__ == "__main__":
    main()