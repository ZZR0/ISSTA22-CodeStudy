import json
from pathlib import Path

input_path = Path("/home/user1-system11/research_dream/llm-clone/extract_clone_py/output/sampling_clones/sampled_100_pairs.jsonl")
output_path = Path("/home/user1-system11/research_dream/llm-clone/extract_clone_py/output/sampling_clones/sampled_pairs_refactored_true.jsonl")

with input_path.open("r", encoding="utf-8") as f:
    data = json.load(f)

def has_refactored_false(record: dict) -> bool:
    sources = record.get("sources", [])
    for src in sources:
        gt = src.get("ground_truth_after_VSCode_ref") or {}
        if gt.get("refactored") is False:
            return True
    return False

if isinstance(data, list):
    filtered = [
        item for item in data
        if isinstance(item, dict) and not has_refactored_false(item)
    ]
elif isinstance(data, dict):
    filtered = [] if has_refactored_false(data) else [data]
else:
    raise ValueError("Input JSON must be a list or dict.")

with output_path.open("w", encoding="utf-8") as f:
    json.dump(filtered, f, indent=2, ensure_ascii=False)

print(f"Input records: {len(data) if isinstance(data, list) else 1}")
print(f"Output records: {len(filtered)}")
print(f"Written to: {output_path}")