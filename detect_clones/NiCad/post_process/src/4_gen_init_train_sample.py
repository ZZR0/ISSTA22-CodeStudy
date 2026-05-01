#!/usr/bin/env python3
"""
4_gen_init_train_sample.py

Purpose
-------
Assign a globally unique func_id to every function in each clone group.

Input:
  JSONL clone groups with structure:
    {
      "classid": <int|str>,
      "sources": [ { ... }, { ... }, ... ]
    }

Output:
  Same JSONL structure, but each source has:
    func_id = {classid}_{global_counter}

This step prepares the dataset for:
  - negative sampling
  - positive sampling
  - clone benchmarks
"""

import argparse
import json
import os
import sys
import re

def infer_repo(src_file: str) -> str:
    # Example path: systems/activemq-java/...  -> activemq
    m = re.search(r"^systems/([^/]+?)-java/", (src_file or "").replace("\\", "/"))
    return m.group(1) if m else "unknown"

def add_unique_ids(input_file: str, output_file: str) -> None:
    if not os.path.exists(input_file):
        print(f"[ERROR] Input file not found: {input_file}")
        sys.exit(1)

    print(f"--- Step 4: Assigning func_id ---")
    print(f"Input : {input_file}")
    print(f"Output: {output_file}")

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    global_func_counter = 0
    total_groups = 0

    with open(input_file, "r", encoding="utf-8") as fin, \
         open(output_file, "w", encoding="utf-8") as fout:

        for line_no, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                print(f"[WARN] Skipping invalid JSON at line {line_no}")
                continue

            class_id = data.get("classid", "unknown")
            sources = data.get("sources", [])

            for src in sources:
                repo = infer_repo(src.get("file", ""))
                src["func_id"] = f"{repo}_{class_id}_{global_func_counter}"
                #src["func_id"] = f"{class_id}_{global_func_counter}"
                global_func_counter += 1

            fout.write(json.dumps(data, ensure_ascii=False) + "\n")
            total_groups += 1

    print("--- Completed ---")
    print(f"Clone groups processed : {total_groups}")
    print(f"Total functions tagged : {global_func_counter}")
    print(f"Saved to               : {output_file}")

    _verify_sample(output_file)


def _verify_sample(file_path: str) -> None:
    """Print a small verification sample from the first line."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            line = f.readline()
            if not line:
                return
            obj = json.loads(line)
            if obj.get("sources"):
                src = obj["sources"][0]
                print("\n[Verification Sample]")
                print(f"  classid : {obj.get('classid')}")
                print(f"  func_id : {src.get('func_id')}  (classid_globalIndex)")
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(
        description="Add globally unique func_id to each function in clone groups"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input JSONL clone groups (from Step 3)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSONL with func_id added (Step 4 output)",
    )

    args = parser.parse_args()
    add_unique_ids(args.input, args.output)


if __name__ == "__main__":
    main()
