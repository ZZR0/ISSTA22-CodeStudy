#!/bin/bash

# Define paths and models
INPUT_DIR="./data-py"
OUTPUT_DIR="./output-py"
MODELS="codegpt,codebert,codet5,graphcodebert"

echo "========================================="
echo "== Evaluating Models: $MODELS"
echo "========================================="

# 1. Evaluate all models and stages in one single Python call
python 1_evaluate_clones_python.py \
    --input-dir "$INPUT_DIR" \
    --input-model "$MODELS" \
    --output "$OUTPUT_DIR"

