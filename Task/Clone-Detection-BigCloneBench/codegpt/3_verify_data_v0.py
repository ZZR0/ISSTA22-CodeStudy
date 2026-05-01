#!/usr/bin/env python3
"""
3_verify_data.py

Verify that every (id1, id2) in pair files exists in the mapping jsonl keys.

Defaults:
  mapping: ../dataset/mix/data.jsonl
  train  : ../dataset/train_mix.txt
  valid  : ../dataset/valid_mix.txt
  test   : ../dataset/test_<otherdomain>.txt  (requires --otherdomain_name)

Pair format (whitespace-separated, typically TSV):
  <id1> <id2> <label>

JSONL format:
  each line is a JSON object with key "idx" (string)

Exit code:
  - 0 if pass
  - 1 if --strict and any missing found
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def load_keys(jsonl_path: Path, key_field: str = "idx") -> set:
    keys = set()
    bad = 0
    total = 0
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
    print(f"[MAP] {jsonl_path} -> keys={len(keys)} lines={total} bad_or_missing_field={bad}")
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
            if len(parts) != 3:
                bad_format += 1
                continue

            parsed += 1
            a, b, y = parts
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
    ap.add_argument("--mapping_jsonl", type=str, default="../dataset/mix/data.jsonl",
                    help="Path to mapping JSONL (default: ../dataset/mix/data.jsonl)")
    ap.add_argument("--key_field", type=str, default="idx",
                    help="Key field in JSONL objects (default: idx)")

    ap.add_argument("--train_pairs", type=str, default="../dataset/train_mix.txt",
                    help="Train pairs file (default: ../dataset/train_mix.txt)")
    ap.add_argument("--valid_pairs", type=str, default="../dataset/valid_mix.txt",
                    help="Valid pairs file (default: ../dataset/valid_mix.txt)")

    # New: otherdomain_name determines default test file name
    ap.add_argument("--otherdomain_name", type=str, default=None,
                    help="Other domain name used to infer default test file (../dataset/test_<name>.txt). Example: camel")

    # test_pairs can override the inferred default
    ap.add_argument("--test_pairs", type=str, default=None,
                    help="Test pairs file. If not set, uses ../dataset/test_<otherdomain_name>.txt")

    ap.add_argument("--max_examples", type=int, default=10,
                    help="Max number of missing examples to print per file (default: 10)")
    ap.add_argument("--strict", action="store_true",
                    help="Exit with code 1 if any missing pairs are found.")

    args = ap.parse_args()

    mapping_path = Path(args.mapping_jsonl)
    if not mapping_path.exists():
        print(f"[ERR] mapping_jsonl not found: {mapping_path}", file=sys.stderr)
        sys.exit(2)

    test_path = None
    if args.test_pairs:
        test_path = Path(args.test_pairs)
    else:
        if not args.otherdomain_name:
            print("[ERR] Provide --otherdomain_name (to infer ../dataset/test_<name>.txt) or set --test_pairs explicitly.",
                  file=sys.stderr)
            sys.exit(2)
        test_path = Path("../dataset") / f"test_{args.otherdomain_name}.txt"

    pair_files = [
        ("train", Path(args.train_pairs)),
        ("valid", Path(args.valid_pairs)),
        ("test",  test_path),
    ]
    for name, p in pair_files:
        if not p.exists():
            print(f"[ERR] {name} pair file not found: {p}", file=sys.stderr)
            sys.exit(2)

    keys = load_keys(mapping_path, key_field=args.key_field)

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
