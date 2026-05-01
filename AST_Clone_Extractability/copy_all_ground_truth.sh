#!/usr/bin/env bash
set -euo pipefail

SRC_ROOT="/home/user1-system11/research_dream/llm-clone/detect_clones/NiCad/post_process/data/java"
DEST_DIR="/home/user1-system11/research_dream/llm-clone/AST_Clone_Extractability/input/ground_truth_test"

mkdir -p "$DEST_DIR"

echo "Copying all test.txt files..."
echo

for dir in "$SRC_ROOT"/*-sim0.7; do
    base="$(basename "$dir")"           # e.g., activemq-sim0.7
    repo="${base%-sim0.7}"              # e.g., activemq

    src_file="$dir/$repo/test.txt"

    if [[ -f "$src_file" ]]; then
        dest_file="$DEST_DIR/${repo}_test.txt"
        cp "$src_file" "$dest_file"
        echo "✓ Copied $repo → ${repo}_test.txt"
    else
        echo "⚠ Skipped $repo (test.txt not found)"
    fi
done

echo
echo "Done."