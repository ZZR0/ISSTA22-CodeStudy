#!/usr/bin/env bash
set -Eeuo pipefail
shopt -s nullglob

LOG_ROOT="/home/user1-system11/research_dream/clone_research/postprocess-clone-results/java/Nicad/output"
OUT_CSV="nicad_stats.csv"

echo "project,classes_parsed,nclones_total,nclones_avg_all,log_path" > "$OUT_CSV"

# find all step2 logs (one per project file run)
# example: .../output/ant-java/step2_nicad_ant-java_sim0.7_classes_fqn.log
while IFS= read -r log; do
  base="$(basename "$log")"

  # project name from filename: step2_nicad_<project>_sim0.7_...
  project="$base"
  project="${project#step2_nicad_}"
  project="${project%_sim0.7_classes_fqn.log}"

  # robust parse:
  # [stats] classes parsed = 1216
  classes="$(awk -F'=' '/\[stats\][[:space:]]+classes parsed/ {gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2); print $2; exit}' "$log")"
  total="$(awk -F'=' '/\[stats\][[:space:]]+nclones total/  {gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2); print $2; exit}' "$log")"
  avg_all="$(awk -F'=' '/\[stats\][[:space:]]+nclones avg \(all classes\)/ {gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2); print $2; exit}' "$log")"
 
  # defaults if missing
  classes="${classes:-NA}"
  total="${total:-NA}"
  avg_all="${avg_all:-NA}"
 
  echo "$project,$classes,$total,$avg_all,$log" >> "$OUT_CSV"
done < <(find "$LOG_ROOT" -type f -name 'step2_nicad_*_sim0.7_classes_fqn.log' | sort)

echo "Wrote: $OUT_CSV"
