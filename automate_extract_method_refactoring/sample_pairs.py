#!/usr/bin/env python3
import json
from pathlib import Path

PAIR_IDS = {
    "hive_1516_1388_vs_hive_1516_1389",
    "tomcat_427_360_vs_tomcat_427_362",
    "pig_775_1153_vs_pig_775_1154",
    "maven_265_433_vs_maven_265_434",
    "hive_1516_1383_vs_hive_1516_1389",
    "hive_1516_1383_vs_hive_1516_1391",
    "cxf_976_1492_vs_cxf_976_1493",
    "pdfbox_137_205_vs_pdfbox_137_207",
    "tomcat_428_379_vs_tomcat_428_380",
    "camel_3715_3797_vs_camel_3715_3799",
    "logging-log4j2_273_375_vs_logging-log4j2_273_376",
    "pdfbox_137_202_vs_pdfbox_137_204",
    "derby_33_63_vs_derby_33_64",
    "jhotdraw7_23_85_vs_jhotdraw7_23_86",
    "pig_774_1150_vs_pig_774_1151",
    "flink_457_559_vs_flink_457_561",
    "jackrabbit_773_1280_vs_jackrabbit_773_1282",
    "hive_1756_2123_vs_hive_1756_2127",
    "cassandra_1099_228_vs_cassandra_1099_229",
    "hive_1756_2117_vs_hive_1756_2128",
    "camel_2720_2658_vs_camel_2720_2660"
}

INPUT_PATH = Path("/home/user1-system11/research_dream/llm-clone/automate_extract_method_refactoring/data/sampled_42pairs_refactored_ inconsistently.json")
OUTPUT_PATH = Path("/home/user1-system11/research_dream/llm-clone/automate_extract_method_refactoring/data/sampled_java_pairs_refactored_true_20.json")


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
    json.dump(kept, f, ensure_ascii=False, indent=2)

print(f"Loaded {len(records)} records")
print(f"Saved {len(kept)} records to: {OUTPUT_PATH}")

if missing:
    print("\nMissing pair_ids:")
    for x in sorted(missing):
        print(f"  {x}")