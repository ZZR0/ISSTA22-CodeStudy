#!/usr/bin/env bash
set -uo pipefail

INPUT_DIR="/home/user1-system11/research_dream/llm-clone/extract_clone/data/extractable_nicad_block_clones"
OUTPUT_ROOT="./output/ground_truth"

SIM_TAG="sim0.7"
SEED=42
LOG_FILE="./output/full_pipeline_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$(dirname "$LOG_FILE")"

echo "Starting full pipeline at $(date)" | tee "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "=========================================" | tee -a "$LOG_FILE"

# Track failures
FAIL_LIST="./output/failed_systems_$(date +%Y%m%d_%H%M%S).txt"
: > "$FAIL_LIST"

for input_file in "$INPUT_DIR"/*.jsonl; do
  base_name=$(basename "$input_file" .jsonl)
  repo_name=${base_name%_clone_analysis}

  # Folder layout:
  #   ./output/ground_truth/<repo>_sim0.7/<repo>/
  TOP_DIR="$OUTPUT_ROOT/${repo_name}_${SIM_TAG}"
  OUT_DIR="$TOP_DIR/$repo_name"
  mkdir -p "$OUT_DIR"

  echo | tee -a "$LOG_FILE"
  echo "=========================================" | tee -a "$LOG_FILE"
  echo "Processing repo: $repo_name" | tee -a "$LOG_FILE"
  echo "Input: $input_file" | tee -a "$LOG_FILE"
  echo "Output: $OUT_DIR" | tee -a "$LOG_FILE"
  echo "Start time: $(date)" | tee -a "$LOG_FILE"

  # Run each repo in a subshell so failures don't kill the whole loop
  (
    set -e

    # Step 5: negatives
    python 5_gen_neg_clone_sample.py \
      --input "$input_file" \
      --out_txt "$TOP_DIR/step5_nicad_${repo_name}_${SIM_TAG}_neg_pairs.txt" \
      --out_jsonl "$TOP_DIR/step5_nicad_${repo_name}_${SIM_TAG}_neg_pairs.jsonl" \
      --out_html "$TOP_DIR/step5_display_nicad_${repo_name}_${SIM_TAG}_neg_pairs.html" \
      --out_md "$TOP_DIR/step5_display_nicad_${repo_name}_${SIM_TAG}_neg_pairs.md" \
      --seed $SEED \
      --verify \
      --cleanup

    # Step 6: positives
    python 6_gen_pos_clone_sample.py \
      --input "$input_file" \
      --out_txt "$TOP_DIR/step6_nicad_${repo_name}_${SIM_TAG}_pos_pairs.txt" \
      --out_jsonl "$TOP_DIR/step6_nicad_${repo_name}_${SIM_TAG}_pos_pairs.jsonl" \
      --out_html "$TOP_DIR/step6_display_nicad_${repo_name}_${SIM_TAG}_pos_pairs.html" \
      --out_md "$TOP_DIR/step6_display_nicad_${repo_name}_${SIM_TAG}_pos_pairs.md" \
      --seed $SEED \
      --verify \
      --cleanup

    # Step 7: corpus (SAVE INTO repo folder under <repo>_sim0.7/<repo>/)
    python 7_gen_func_index_corpus.py \
      --input "$input_file" \
      --output "$OUT_DIR/data.jsonl" \
      --dedup

    # Step 8: combine (SAVE INTO repo folder under <repo>_sim0.7/<repo>/)
    python 8_combine_neg_pos_pairs.py \
      --neg "$TOP_DIR/step5_nicad_${repo_name}_${SIM_TAG}_neg_pairs.txt" \
      --pos "$TOP_DIR/step6_nicad_${repo_name}_${SIM_TAG}_pos_pairs.txt" \
      --out "$OUT_DIR/test.txt"
  ) >>"$LOG_FILE" 2>&1

  rc=$?
  if [ $rc -ne 0 ]; then
    echo "❌ FAILED repo: $repo_name (exit=$rc). Skipping." | tee -a "$LOG_FILE"
    echo "${repo_name}    ${input_file}" >> "$FAIL_LIST"
    continue
  fi

  echo "✅ Finished repo: $repo_name at $(date)" | tee -a "$LOG_FILE"
done

echo | tee -a "$LOG_FILE"
echo "=========================================" | tee -a "$LOG_FILE"
echo "All repos processed at $(date)" | tee -a "$LOG_FILE"
echo "Failures (if any) saved to: $FAIL_LIST" | tee -a "$LOG_FILE"