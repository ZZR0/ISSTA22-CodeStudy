#!/usr/bin/env python3
import json
import sys
import argparse
import io
import tokenize
from pathlib import Path
from typing import List, Union

# ---------------------------------------------------------
# Filtering Logic (Extracted)
# ---------------------------------------------------------

def is_test_filename(file_path: str) -> bool:
    if not file_path:
        return False
    p = file_path.lower()
    name = Path(p).name

    # Check for common test directory patterns
    if "/test/" in p or "/tests/" in p or "/testing/" in p:
        return True
    # Check for engineering/build scripts often containing tests
    if "/eng/" in p or p.startswith("eng/"):
        return True
    # Check standard Python test file naming conventions
    return name.startswith("test_") or name.endswith("_test.py")

def is_test_function_name(code: str) -> bool:
    if not code:
        return False
    for line in code.splitlines():
        if line.strip().startswith("def test_"):
            return True
    return False

def has_assertions_tokenwise(code: str) -> bool:
    if not code:
        return False
    if "assert" not in code:
        return False
    src = code if code.endswith("\n") else (code + "\n")
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type == tokenize.NAME and tok.string == "assert":
                return True
        return False
    except (tokenize.TokenError, IndentationError, SyntaxError):
        # Fallback if tokenization fails
        return "assert " in code

def has_test_framework_imports_tokenwise(code: str) -> bool:
    if not code:
        return False
    lowered = code.lower()
    if "pytest" not in lowered and "unittest" not in lowered:
        return False

    src = code if code.endswith("\n") else (code + "\n")
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return (
            "import pytest" in lowered or "from pytest" in lowered or
            "import unittest" in lowered or "from unittest" in lowered
        )

    n = len(toks)
    for i in range(n):
        t = toks[i]
        if t.type != tokenize.NAME:
            continue
        if t.string in ("import", "from"):
            j = i + 1
            while j < n and toks[j].type in (
                tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT
            ):
                j += 1
            if j < n and toks[j].type == tokenize.NAME and toks[j].string in ("pytest", "unittest"):
                return True
    return False

def test_reasons(file_path: str, code_clean: str) -> List[str]:
    """Returns a list of reasons why a file/code snippet is considered a test."""
    reasons: List[str] = []
    if is_test_filename(file_path):
        reasons.append("FILENAME")
    if is_test_function_name(code_clean):
        reasons.append("FUNCNAME")
    if has_assertions_tokenwise(code_clean):
        reasons.append("ASSERT")
    if has_test_framework_imports_tokenwise(code_clean):
        reasons.append("IMPORT")
    return reasons

# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Filter out Python test functions from JSONL data.")
    parser.add_argument("--input", required=True, help="Path to input JSONL file")
    parser.add_argument("--output", required=True, help="Path to output JSONL file")
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

            # Handle both 'sources' list and legacy flat format
            sources = obj.get("sources")
            items_to_check = []

            if isinstance(sources, list):
                items_to_check = sources
            else:
                # Flat format (legacy)
                items_to_check = [obj]

            # Check if ANY item in this group is a test
            is_test_group = False
            found_reasons = []

            for item in items_to_check:
                # Extract code and filename, handling variations in keys
                code = item.get("code", "") or item.get("func", "")
                file_path = item.get("file", "") or item.get("file_path", "")
                
                # Run the filters
                reasons = test_reasons(file_path, code)
                if reasons:
                    is_test_group = True
                    found_reasons = reasons
                    break
            
            if is_test_group:
                excluded_count += 1
                # Optional: print exclusion reason for debugging
                # print(f"[EXCLUDE] Line {total_count}: {found_reasons}", file=sys.stderr)
            else:
                kept_count += 1
                fout.write(line + "\n")

    print(f"Done.", file=sys.stderr)
    print(f"Total lines processed: {total_count}", file=sys.stderr)
    print(f"Kept: {kept_count}", file=sys.stderr)
    print(f"Excluded (Tests): {excluded_count}", file=sys.stderr)

if __name__ == "__main__":
    main()