#!/usr/bin/env python3
# gen_funcs_from_clone_groups.py

import argparse
import json
import os

def main():
    ap = argparse.ArgumentParser(
        description="Convert NiCad clone-group JSONL (classid + sources[]) into function JSONL: {'func','idx'}"
    )
    ap.add_argument("--input", required=True, help="Path to nicad_camel_clone_data.jsonl")
    ap.add_argument("--output", required=True, help="Output JSONL path (one function per line)")
    ap.add_argument("--skip_empty_code", action="store_true", help="Skip entries with empty code")
    ap.add_argument("--dedup", action="store_true", help="Deduplicate by func_id (recommended)")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    seen = set()
    groups_read = 0
    funcs_seen = 0
    funcs_written = 0

    with open(args.input, "r", encoding="utf-8") as fin, open(args.output, "w", encoding="utf-8") as fout:
        for line_num, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            sources = obj.get("sources", [])
            if not isinstance(sources, list) or not sources:
                continue

            groups_read += 1

            for src in sources:
                if not isinstance(src, dict):
                    continue

                funcs_seen += 1

                func_id = src.get("func_id")
                code = src.get("code", "")

                # Must have func_id, because you want idx=func_id
                if not func_id:
                    continue

                if args.skip_empty_code and not code:
                    continue

                if args.dedup:
                    if func_id in seen:
                        continue
                    seen.add(func_id)

                fout.write(json.dumps({"func": code, "idx": str(func_id)}, ensure_ascii=False) + "\n")
                funcs_written += 1

    print("=== Done ===")
    print(f"Groups read      : {groups_read}")
    print(f"Functions scanned: {funcs_seen}")
    print(f"Functions written: {funcs_written}")
    print(f"Output           : {args.output}")

if __name__ == "__main__":
    main()
