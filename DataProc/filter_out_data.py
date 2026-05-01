#!/usr/bin/env python3
"""
filter_out_data.py

Purpose
-------
Post-process a NiCad-generated clone dataset (JSONL) by:
  1) Filtering out clone groups that contain test code (configurable mode),
  2) Discarding clone groups that are too large (nclones >= threshold),
  3) Optionally stripping Java comments from source code to reduce token size.

This script is intended to produce a cleaner, smaller dataset for downstream
analysis and/or LLM-based processing.

Input / Output
--------------
Input  (default):
  data/step2_nicad_camel-java_sim0.7_classes_fqn.jsonl

Output (default):
  data/nicad_camel_clone_func.jsonl

Expected JSONL schema (per line)
-------------------------------
Each line is a JSON object representing one clone group, typically containing:
  - "nclones": int
  - "sources": list[dict], where each source may include:
      - "file": str   (path to the source file)
      - "code": str   (Java source code snippet)

Key Processing Rules
--------------------
1) Size filtering:
   - Drop clone groups where nclones >= --max_clones (default: 20).

2) Test filtering:
   - Default mode: drop_group_if_any_test
   - If ANY source path matches common test directory or test filename patterns,
     the entire clone group is discarded.

3) Comment stripping:
   - Enabled by default (disable via --keep_comments)
   - Removes Java line comments (//...) and block comments (/*...*/)
   - Preserves string literals to avoid corrupting code.

Usage Examples
--------------
Run with defaults:
  python filter_out_data.py

Specify input/output:
  python filter_out_data.py --input <path/to/input.jsonl> --output <path/to/output.jsonl>

Disable comment stripping:
  python filter_out_data.py --keep_comments

Change clone group size threshold:
  python filter_out_data.py --max_clones 15
"""
import argparse
import json
import os
import re
import sys
from typing import Dict, Any, List

# ---- Default Paths ----
DEFAULT_INPUT = "data/step2_nicad_camel-java_sim0.7_classes_fqn.jsonl"
DEFAULT_OUTPUT = "data/nicad_camel_clone_func.jsonl"

# ---- Test detection rules ----
TEST_DIR_PATTERNS = [
    r"(^|/)src/test(/|/java/|/resources/|$)",
    r"(^|/)src/it(/|$)",
    r"(^|/)src/integration-test(/|$)",
    r"(^|/)test(/|$)",
    r"(^|/)tests(/|$)",
]

TEST_FILENAME_PATTERNS = [
    r"Test\.java$",
    r"Tests\.java$",
    r"TestCase\.java$",
    r"IT\.java$",
    r"ITCase\.java$",
    r"IntegrationTest\.java$",
]

TEST_DIR_RES = [re.compile(p) for p in TEST_DIR_PATTERNS]
TEST_FILE_RES = [re.compile(p) for p in TEST_FILENAME_PATTERNS]

# ---- Comment Stripping Regex ----
# This pattern matches:
# 1. Line comments (//...)
# 2. Block comments (/*...*/)
# 3. Strings ("..." or '...') -> to ignore slashes inside strings
JAVA_COMMENT_PATTERN = re.compile(
    r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
    re.DOTALL | re.MULTILINE
)

def remove_java_comments(text: str) -> str:
    """
    Remove Java block comments and line comments while preserving string literals.
    """
    def replacer(match):
        s = match.group(0)
        if s.startswith('/'):
            return " "  # Replace comment with a single space
        else:
            return s  # Return string literal as is
            
    return JAVA_COMMENT_PATTERN.sub(replacer, text)

def process_group_comments(obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Iterate over all sources in the clone group and strip comments from the code.
    """
    for src in obj.get("sources", []):
        if "code" in src:
            src["code"] = remove_java_comments(src["code"])
    return obj

def is_test_path(path: str) -> bool:
    """Check if the file path matches any test directory or filename patterns."""
    norm = path.replace("\\", "/")
    for r in TEST_DIR_RES:
        if r.search(norm):
            return True
    base = os.path.basename(norm)
    for r in TEST_FILE_RES:
        if r.search(base):
            return True
    return False

def group_has_any_test_source(obj: Dict[str, Any]) -> bool:
    """Return True if any source within the group is identified as a test file."""
    for s in obj.get("sources", []):
        if is_test_path(s.get("file", "")):
            return True
    return False

def main():
    ap = argparse.ArgumentParser(
        description="Filter out test functions, limit clone group size, and strip comments from a NiCad clone JSONL file."
    )
    # Set default values to allow execution without explicit command line arguments
    ap.add_argument("--input", default=DEFAULT_INPUT, help=f"Input JSONL file (default: {DEFAULT_INPUT})")
    ap.add_argument("--output", default=DEFAULT_OUTPUT, help=f"Output JSONL file (default: {DEFAULT_OUTPUT})")
    ap.add_argument(
        "--mode",
        choices=["drop_group_if_any_test", "drop_only_test_sources"],
        default="drop_group_if_any_test",
        help="Filtering mode for test files."
    )
    # Max clones limit based on Professor's directive: nclones < 20
    ap.add_argument(
        "--max_clones",
        type=int,
        default=20,
        help="Discard groups where nclones is equal to or greater than this value (default: 20)."
    )
    # Option to disable comment stripping if needed
    ap.add_argument(
        "--keep_comments",
        action="store_true",
        help="If set, comments will NOT be removed from the source code."
    )

    args = ap.parse_args()

    in_path = args.input
    out_path = args.output

    kept_groups = 0
    dropped_groups_test = 0
    dropped_groups_size = 0
    total_groups = 0

    if not os.path.exists(os.path.dirname(out_path)) and os.path.dirname(out_path) != "":
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print(f"Reading from: {in_path}")
    print(f"Writing to:   {out_path}")
    if not args.keep_comments:
        print("Feature:      Java Comment Stripping ENABLED (Optimization)")

    with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            total_groups += 1
            try:
                obj = json.loads(line)
            except Exception as e:
                print(f"[ERROR] JSON parse failed at line {line_no}: {e}", file=sys.stderr)
                continue

            # ---- SIZE FILTERING LOGIC ----
            # Objective: Include only groups where nclones < 20.
            # Decision: If nclones is 20 or more, the group is discarded to prevent overfitting.
            current_nclones = obj.get("nclones", 0)
            if current_nclones >= args.max_clones:
                dropped_groups_size += 1
                continue

            # ---- TEST FILTERING LOGIC ----
            # We determine whether to keep the group based on test files.
            should_write = False
            
            if args.mode == "drop_group_if_any_test":
                if group_has_any_test_source(obj):
                    dropped_groups_test += 1
                    continue
                should_write = True
            
            # (If needed, other modes like 'drop_only_test_sources' can be handled here)

            # ---- FINAL PROCESSING & WRITING ----
            if should_write:
                # OPTIMIZATION: Strip Comments
                # Unless the user explicitly requested to keep them, we remove comments to save token space.
                if not args.keep_comments:
                    obj = process_group_comments(obj)

                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                kept_groups += 1

    print("--- Processing Complete ---")
    print(f"  Input groups:           {total_groups}")
    print(f"  Dropped (Size >= {args.max_clones}): {dropped_groups_size}")
    print(f"  Dropped (Test related): {dropped_groups_test}")
    print(f"  Kept groups:            {kept_groups}")
    print(f"  Output written:         {out_path}")

if __name__ == "__main__":
    main()