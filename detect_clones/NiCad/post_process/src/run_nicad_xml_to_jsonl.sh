#!/usr/bin/env bash
set -Eeuo pipefail
shopt -s nullglob

IN_DIR="/home/user1-system11/research_dream/clone_research/postprocess-clone-results/java/Nicad/input"
OUT_DIR="/home/user1-system11/research_dream/clone_research/postprocess-clone-results/java/Nicad/output"

mkdir -p "$OUT_DIR"

for xml in "$IN_DIR"/*-classes-withsource.xml; do
    # Example:
    # activemq-java_functions-blind-clones-0.30-classes-withsource.xml
    fname="$(basename "$xml")"

    # Extract project name (activemq-java)
    project="${fname%%_functions-blind-clones-*}"

    out_json="$OUT_DIR/nicad_${project}_sim0.7_classes.jsonl"
    out_log="$OUT_DIR/nicad_${project}_sim0.7_classes.log"

    echo "Processing $project"

    python 1_nicad_xml_to_jsonl.py \
        --xml "$xml" \
        --out "$out_json" \
        --mode class \
        > "$out_log" 2>&1
done
