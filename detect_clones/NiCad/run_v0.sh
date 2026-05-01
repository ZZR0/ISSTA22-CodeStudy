#!/usr/bin/env bash

# Exit immediately if a command fails
set -e

# Make globs that don't match expand to nothing (instead of literal text)
shopt -s nullglob

if [ ! -x ./nicad6 ]; then
  echo "ERROR: ./nicad6 not found or not executable. Run from the NiCad directory and ensure NiCad is built."
  exit 1
fi

LOG_DIR="nicad_logs"
mkdir -p "$LOG_DIR"

TOTAL_START=$(date +%s)

echo "======================================"
echo "NiCad batch run started at $(date)"
echo "======================================"

datasets=(systems/*-java)
if ((${#datasets[@]} == 0)); then
  echo "No Java datasets found under systems/ (expected directories matching systems/*-java)."
  echo "Nothing to do."
  exit 0
fi

for dataset_path in "${datasets[@]}"; do
  d=$(basename "$dataset_path")
  if [ -d "$dataset_path" ] && [[ "$d" != *_functions* ]]; then
    echo "--------------------------------------"
    echo "Running NiCad on $d"

    START_TIME=$(date +%s)

    ./nicad6 functions java "$dataset_path" default-report \
      > "$LOG_DIR/${d}_nicad.log" 2>&1

    END_TIME=$(date +%s)
    RUNTIME=$((END_TIME - START_TIME))

    R_HOURS=$((RUNTIME / 3600))
    R_MINUTES=$(((RUNTIME % 3600) / 60))
    R_SECONDS=$((RUNTIME % 60))

    printf "Finished %s in %02dh %02dm %02ds\n" \
      "$d" "$R_HOURS" "$R_MINUTES" "$R_SECONDS"

    printf "Runtime: %02dh %02dm %02ds\n" \
      "$R_HOURS" "$R_MINUTES" "$R_SECONDS" \
      >> "$LOG_DIR/${d}_nicad.log"
  fi
done

TOTAL_END=$(date +%s)
TOTAL_RUNTIME=$((TOTAL_END - TOTAL_START))

T_HOURS=$((TOTAL_RUNTIME / 3600))
T_MINUTES=$(((TOTAL_RUNTIME % 3600) / 60))
T_SECONDS=$((TOTAL_RUNTIME % 60))

echo "======================================"
echo "All NiCad runs completed at $(date)"
printf "Total runtime: %02dh %02dm %02ds\n" \
  "$T_HOURS" "$T_MINUTES" "$T_SECONDS"
echo "======================================"
