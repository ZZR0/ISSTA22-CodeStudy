#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/user1-system11/research_dream/llm-clone/AST_Clone_Extractability"
IN_BASE="$ROOT/input/java-functions"
OUT_BASE="$ROOT/output/java-functions"

LOG_FILE="$OUT_BASE/ALL_repos_Extractability.log"

mkdir -p "$OUT_BASE"

echo "==============================================================" > "$LOG_FILE"
echo "Batch Extractability Run Started: $(date)" >> "$LOG_FILE"
echo "ROOT    : $ROOT" >> "$LOG_FILE"
echo "IN_BASE : $IN_BASE" >> "$LOG_FILE"
echo "OUT_BASE: $OUT_BASE" >> "$LOG_FILE"
echo "==============================================================" >> "$LOG_FILE"

shopt -s nullglob
found_any=0

for repo_dir in "$IN_BASE"/*-sim0.7; do
  [[ -d "$repo_dir" ]] || continue
  found_any=1

  repo="$(basename "$repo_dir")"

  matches=("$repo_dir"/step4_nicad_*_filtered_with_func_id.jsonl)
  in_file="${matches[0]:-}"

  if [[ -z "${in_file}" || ! -f "$in_file" ]]; then
    echo "[SKIP] $repo (no matching input file)" | tee -a "$LOG_FILE"
    continue
  fi

  out_file="$OUT_BASE/${repo}_extractability.jsonl"

  echo "" | tee -a "$LOG_FILE"
  echo "==============================================================" | tee -a "$LOG_FILE"
  echo "[RUN ] Repo : $repo" | tee -a "$LOG_FILE"
  echo "Input : $in_file" | tee -a "$LOG_FILE"
  echo "Output: $out_file" | tee -a "$LOG_FILE"
  echo "Time  : $(date)" | tee -a "$LOG_FILE"
  echo "==============================================================" | tee -a "$LOG_FILE"

  (
    cd "$ROOT"
    PYTHONPATH=.. python main.py \
      --input "$in_file" \
      --output "$out_file" \
      --jsonl
  ) 2>&1 | tee -a "$LOG_FILE"

done

if [[ "$found_any" -eq 0 ]]; then
  echo "[WARN] No repo directories matched: $IN_BASE/*-sim0.7" | tee -a "$LOG_FILE"
fi

echo "" | tee -a "$LOG_FILE"
echo "Batch Finished: $(date)" | tee -a "$LOG_FILE"
echo "==============================================================" | tee -a "$LOG_FILE"