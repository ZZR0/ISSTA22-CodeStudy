#!/usr/bin/env python3
import json
from pathlib import Path

PAIR_IDS = {
    "salt_45_90_vs_salt_45_91",
    "simplejson_2_0_vs_simplejson_2_1",
    "salt_45_89_vs_salt_45_93",
    "keras_25_25_vs_keras_25_26",
    "salt_101_251_vs_salt_101_252",
    "keystone_26_65_vs_keystone_26_66",
    "salt_114_281_vs_salt_114_283",
    "salt_39_77_vs_salt_39_78",
    "web2py_31_40_vs_web2py_31_46",
    "salt_146_356_vs_salt_146_357",
    "salt_152_369_vs_salt_152_370",
    "web2py_31_40_vs_web2py_31_45",
    "web2py_48_66_vs_web2py_48_67",
    "sympy_114_137_vs_sympy_114_138",
    "salt_7_12_vs_salt_7_13",
    "salt_118_290_vs_salt_118_291",
    "ansible_11_23_vs_ansible_11_24",
    "salt_88_208_vs_salt_88_209",
    "salt_97_228_vs_salt_97_230",
    "salt_105_260_vs_salt_105_261"
}

INPUT_PATH = Path("/home/user1-system11/research_dream/llm-clone/automate_extract_method_refactoring_py/data/sampled_pairs_refactored_true.jsonl")
OUTPUT_PATH = Path("/home/user1-system11/research_dream/llm-clone/automate_extract_method_refactoring_py/data/sampled_pairs_refactored_true_20.jsonl")


def load_records(path: Path):
    text = path.read_text(encoding="utf-8").strip()

    # Try full JSON first
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
        elif isinstance(parsed, dict):
            return [parsed]
    except json.JSONDecodeError:
        pass

    # Fallback: JSONL
    records = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        if line in {"[", "]"}:
            continue
        if line.endswith(","):
            line = line[:-1].strip()
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"Skipping bad JSON at line {lineno}: {e}")
    return records


records = load_records(INPUT_PATH)

kept = []
missing = set(PAIR_IDS)

for obj in records:
    pair_id = obj.get("pair_id") or obj.get("classid")
    if pair_id in PAIR_IDS:
        kept.append(obj)
        missing.discard(pair_id)

with OUTPUT_PATH.open("w", encoding="utf-8") as f:
    for obj in kept:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

print(f"Loaded {len(records)} records")
print(f"Saved {len(kept)} records to: {OUTPUT_PATH}")

if missing:
    print("\nMissing pair_ids:")
    for x in sorted(missing):
        print(f"  {x}")