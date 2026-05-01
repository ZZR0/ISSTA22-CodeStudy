#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT="/home/user1-system11/research_dream/llm-clone/extract_clone/output/ground_truth"
TARGET_ROOT="/home/user1-system11/research_dream/llm-clone/Task/Clone-Detection-BigCloneBench/dataset/nicad_block_java"

mkdir -p "$TARGET_ROOT"

for sim_dir in "$SOURCE_ROOT"/*_sim0.7; do
    repo_name=$(basename "$sim_dir" _sim0.7)
    repo_folder="$sim_dir/$repo_name"

    if [ -d "$repo_folder" ]; then
        link_path="$TARGET_ROOT/$repo_name"

        # Remove old link if exists
        if [ -L "$link_path" ] || [ -e "$link_path" ]; then
            rm -rf "$link_path"
        fi

        ln -s "$repo_folder" "$link_path"
        echo "Created symlink: $link_path -> $repo_folder"
    fi
done

echo "All symbolic links created successfully."