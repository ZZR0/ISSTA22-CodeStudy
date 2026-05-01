import json, os, shutil

json_path = "/home/user1-system11/research_dream/llm-clone/automate_extract_method_refactoring/data/sampled_java_pairs_refactored_true_20.json"
src_root = "/home/user1-system11/research_dream/llm-clone/detect_clones/NiCad/systems"
dst_root = "/home/user1-system11/research_dream/llm-clone/automate_extract_method_refactoring/systems"

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

paths = []

def walk(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "sources" and isinstance(v, list):
                for item in v:
                    if isinstance(item, dict) and "file" in item:
                        paths.append(item["file"])
            walk(v)
    elif isinstance(obj, list):
        for item in obj:
            walk(item)

walk(data)

unique_paths = sorted(set(paths))
copied = 0
missing = []

for rel in unique_paths:
    rel_from_systems = rel[len("systems/"):] if rel.startswith("systems/") else rel
    src = os.path.join(src_root, rel_from_systems)
    dst = os.path.join(dst_root, rel_from_systems)

    os.makedirs(os.path.dirname(dst), exist_ok=True)

    if os.path.exists(src):
        shutil.copy2(src, dst)
        copied += 1
    else:
        missing.append(src)

print(f"Found {len(unique_paths)} unique source files")
print(f"Copied {copied} files")
print(f"Missing {len(missing)} files")

if missing:
    print("\nMissing files:")
    for m in missing:
        print(m)