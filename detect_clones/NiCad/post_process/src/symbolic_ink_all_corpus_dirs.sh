#!/usr/bin/env bash
set -u -o pipefail
shopt -s nullglob

SIM_TAG="${1:-sim0.7}"

OUTPUT_ROOT="/home/user1-system11/research_dream/llm-clone/detect_clones/NiCad/post_process/data/java"
TARGET_ROOT="/home/user1-system11/research_dream/llm-clone/Task/Clone-Detection-BigCloneBench/dataset/java"

mkdir -p "$TARGET_ROOT"

echo "Linking corpus directories"
echo "FROM: $OUTPUT_ROOT"
echo "TO:   $TARGET_ROOT"
echo "SIM_TAG: $SIM_TAG"
echo

sys_dirs=( "$OUTPUT_ROOT"/*-"$SIM_TAG" )
echo "Found ${#sys_dirs[@]} system folders matching *-${SIM_TAG}"

linked=0 skipped=0 warned=0 failed=0

for sys_dir in "${sys_dirs[@]}"; do
  system="$(basename "$sys_dir")"
  system="${system%-${SIM_TAG}}"

  corpus_dir="$sys_dir/$system"
  target="$TARGET_ROOT/$system"

  if [[ ! -d "$corpus_dir" ]]; then
    echo "[WARN] Missing corpus dir: $corpus_dir"
    warned=$((warned+1))
    continue
  fi

  if [[ -L "$target" ]]; then
    echo "[SKIP] Symlink exists: $target"
    skipped=$((skipped+1))
    continue
  fi

  if [[ -e "$target" ]]; then
    echo "[SKIP] Exists (not symlink): $target"
    skipped=$((skipped+1))
    continue
  fi

  echo "[LINK] $target -> $corpus_dir"
  if ln -s "$corpus_dir" "$target"; then
    linked=$((linked+1))
  else
    echo "[FAIL] ln failed for: $system"
    failed=$((failed+1))
  fi
done

echo
echo "Done."
echo "  linked : $linked"
echo "  skipped: $skipped"
echo "  warned : $warned"
echo "  failed : $failed"
