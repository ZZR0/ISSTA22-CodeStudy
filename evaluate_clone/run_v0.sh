#!/bin/bash

MODELS="codegpt"
OUTPUT_DIR="output-py"
INPUT_DIR="./data-sample-py"

python 1_evaluate_clones.py \
    --input-dir "$INPUT_DIR" \
    --input-model "$MODELS" \
    --output "$OUTPUT_DIR"



# ADAPT_STAGE="after"
# # REPO="camel"
# REPO="azure-sdk-for-python"

# MODELS="codegpt"
# OUTPUT_DIR="output-py"
# INPUT_DIR="./data-sample-py"

# # INPUT_PATH="./data/pred_${ADAPT_STAGE}_adapt_refactorability/${MODEL}/pred_${REPO}_with_refactorability.csv"
# # INPUT_PATH="./data/pred_${ADAPT_STAGE}_adapt_refactorability/${MODEL}/pred_${REPO}_with_refactorability_sample.csv"
# INPUT_PATH="./data/pred_${ADAPT_STAGE}_adapt_refactorability/${MODEL}/pred_${REPO}_with_refactorability-sample.csv"

# python 1_evaluate_clones.py --input "$INPUT_PATH" --output "$OUTPUT_DIR" --input-model "$MODELS"

# echo ' ';echo '-----------------------------';echo ' '

# ADAPT_STAGE="before"

# # INPUT_PATH="./data/pred_${ADAPT_STAGE}_adapt_refactorability/${MODEL}/pred_${REPO}_with_refactorability.csv"
# # INPUT_PATH="./data/pred_${ADAPT_STAGE}_adapt_refactorability/${MODEL}/pred_${REPO}_with_refactorability_sample.csv"
# INPUT_PATH="./data/pred_${ADAPT_STAGE}_adapt_refactorability/${MODEL}/pred_${REPO}_with_refactorability-sample.csv"

# python 1_evaluate_clones.py --input "$INPUT_PATH" --output "$OUTPUT_DIR" --input-model "$MODELS"

