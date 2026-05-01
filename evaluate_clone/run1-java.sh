#!/bin/bash

# Define paths and models
INPUT_DIR="./data"
OUTPUT_DIR="./output-java"
MODELS="codegpt,codebert,codet5,graphcodebert"

echo "========================================="
echo "== Evaluating Models: $MODELS"
echo "========================================="

# 1. Evaluate all models and stages in one single Python call
python 1_evaluate_clones_java.py \
    --input-dir "$INPUT_DIR" \
    --input-model "$MODELS" \
    --output "$OUTPUT_DIR"

