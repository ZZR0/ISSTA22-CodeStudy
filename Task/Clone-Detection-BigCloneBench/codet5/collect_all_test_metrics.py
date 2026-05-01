import re
import csv
from pathlib import Path

# Path configuration
LOG_PATH = Path("/home/user1-system11/research_dream/llm-clone/Task/Clone-Detection-BigCloneBench/codet5/saved_models_combined/nicad_block_java/test_all_systems_codet5_bcb_java_block_function.log")
OUT_CSV = Path("/home/user1-system11/research_dream/llm-clone/Task/Clone-Detection-BigCloneBench/codet5/saved_models_combined/repo_metrics_codet5_bcb_java_block_function_after_adaption.csv")

# Regex patterns
RE_REPO = re.compile(r"^-{5,}\s*\[([^\]]+)\]\s+.*-+\s*$")
RE_EVAL_METRIC = re.compile(r"\beval_(f1|precision|recall)\s*=\s*([0-9]*\.?[0-9]+)\b", re.IGNORECASE)
RE_PAIRS = re.compile(r"Num examples\s*=\s*(\d+)")

def parse_log(text: str, debug: bool = False):
    results_by_repo = {}
    repo = None
    # Initialize with None
    data = {"pairs": None, "precision": None, "recall": None, "f1": None}

    def flush():
        nonlocal repo, data
        if repo and any(v is not None for v in data.values()):
            results_by_repo[repo] = {
                "Repo_name": repo,
                "Pairs_loaded": data["pairs"],
                "Precision": data["precision"],
                "Recall": data["recall"],
                "F1": data["f1"],
            }
        repo = None
        data = {"pairs": None, "precision": None, "recall": None, "f1": None}

    for line in text.splitlines():
        # 1. Look for new repo header
        m_repo = RE_REPO.match(line.strip())
        if m_repo:
            flush()
            repo = m_repo.group(1).strip()
            continue

        if repo:
            # 2. Extract Num examples
            m_pairs = RE_PAIRS.search(line)
            if m_pairs:
                data["pairs"] = int(m_pairs.group(1))
            
            # 3. Extract metrics
            m_metric = RE_EVAL_METRIC.search(line)
            if m_metric:
                k = m_metric.group(1).lower()
                data[k] = float(m_metric.group(2))

    flush()
    rows = list(results_by_repo.values())
    rows.sort(key=lambda r: r["Repo_name"].lower())
    return rows

def main():
    text = LOG_PATH.read_text(errors="replace")
    rows = parse_log(text)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["Repo_name", "Pairs_loaded", "Precision", "Recall", "F1"]

    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {OUT_CSV.resolve()} ({len(rows)} repos)")

if __name__ == "__main__":
    main()