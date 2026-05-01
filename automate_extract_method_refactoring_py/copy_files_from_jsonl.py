import json
import shutil
from pathlib import Path

# Input paths
source_root = Path("/home/user1-system11/research_dream/llm-clone/detect_clones/NiCad-py-block/systems")
jsonl_file = Path("/home/user1-system11/research_dream/llm-clone/automate_extract_method_refactoring_py/data/sampled_pairs_refactored_true_20.jsonl")
target_root = Path("/home/user1-system11/research_dream/llm-clone/automate_extract_method_refactoring_py/systems")

target_root.mkdir(parents=True, exist_ok=True)

copied = set()
missing = []

with jsonl_file.open("r", encoding="utf-8") as f:
    for line_num, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue

        try:
            item = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"[Line {line_num}] JSON decode error: {e}")
            continue

        for src in item.get("sources", []):
            rel_path_str = src.get("file")
            if not rel_path_str:
                continue

            rel_path = Path(rel_path_str)

            # If JSONL path starts with "systems/", remove it because source_root already points to .../systems
            if rel_path.parts and rel_path.parts[0] == "systems":
                rel_path = Path(*rel_path.parts[1:])

            src_file = source_root / rel_path
            dst_file = target_root / rel_path

            if src_file.exists() and src_file.is_file():
                dst_file.parent.mkdir(parents=True, exist_ok=True)
                if src_file not in copied:
                    shutil.copy2(src_file, dst_file)
                    copied.add(src_file)
                    print(f"Copied: {src_file} -> {dst_file}")
            else:
                missing.append(str(src_file))

print("\nDone.")
print(f"Total copied files: {len(copied)}")

if missing:
    print(f"Missing files: {len(missing)}")
    for m in sorted(set(missing)):
        print("Missing:", m)