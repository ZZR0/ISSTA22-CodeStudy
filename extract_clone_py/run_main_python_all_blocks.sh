#!/bin/bash

# Define the base directories
INPUT_DIR="/home/user1-system11/research_dream/llm-clone/extract_clone_py/data/py_blocks_step1_2a_2b_2d_3"
NICAD_BASE_DIR="/home/user1-system11/research_dream/llm-clone/detect_clones/NiCad-py-block"
OUTPUT_DIR="/home/user1-system11/research_dream/llm-clone/extract_clone_py/data/augmented_post_nicad_blocks"

# Create a timestamped log file in the input directory to avoid duplicate file names
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/full_pipeline_steps_1_2a_2b_2d_3_${TIMESTAMP}.log"

# Create the output directory if it doesn't already exist
mkdir -p "$OUTPUT_DIR"

# Redirect all subsequent output (stdout and stderr) to both the terminal and the log file
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Starting clone analysis pipeline..."
echo "Log file created at: $LOG_FILE"
echo "==================================================="

# Iterate over all directories in the input path that end with -sim0.7
for dir_path in "$INPUT_DIR"/*-sim0.7; do
    if [ -d "$dir_path" ]; then
        dir_name=$(basename "$dir_path")
        repo_name="${dir_name%-sim0.7}"
        
        JSONL_FILE="$dir_path/step3_nicad_${repo_name}_sim0.7.jsonl"
        OUT_HTML="$OUTPUT_DIR/${repo_name}_clone_visualization.html"
        OUT_JSONL="$OUTPUT_DIR/augmented_${repo_name}.jsonl"
        
        if [ -f "$JSONL_FILE" ]; then
            echo "[*] Processing repository: $repo_name..."
            
            python main_python.py \
                --jsonl "$JSONL_FILE" \
                --base-dir "$NICAD_BASE_DIR" \
                --output "$OUT_HTML" \
                --out-jsonl "$OUT_JSONL"
                
            echo "[+] Finished $repo_name"
            echo "---------------------------------------------------"
        else
            echo "[-] Warning: Input file not found for $repo_name at $JSONL_FILE. Skipping."
            echo "---------------------------------------------------"
        fi
    fi
done

echo "All repositories processed successfully!"
echo "Full log saved to: $LOG_FILE"