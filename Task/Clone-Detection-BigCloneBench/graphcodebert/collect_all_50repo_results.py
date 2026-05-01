import re
from pathlib import Path
import csv

LOG_PATH = "/home/user1-system11/research_dream/llm-clone/Task/Clone-Detection-BigCloneBench/graphcodebert/saved_models_combined_py/logs/test_all_systems_graphcodebert_bcb_python_combined.log"  
OUT_CSV  = "/home/user1-system11/research_dream/llm-clone/Task/Clone-Detection-BigCloneBench/graphcodebert/saved_models_combined_py/repo_metrics_graphcodebert_bcb_python_combined.csv"

# Extract repo from:
# train: ../dataset/ant-ivy/train.txt
RE_TRAIN = re.compile(r"train:\s+.*?/dataset/python/([^/]+)/train\.txt")

RE_METRIC = re.compile(
    r"(precision|recall|f1)\s*[:=]\s*([0-9]*\.?[0-9]+)",
    re.I
)

def parse_log(text, debug=True):
    rows = []
    repo = None
    metrics = {"precision": None, "recall": None, "f1": None}

    seen_repo = 0
    seen_metric = 0

    def flush():
        nonlocal repo, metrics
        if repo and any(v is not None for v in metrics.values()):
            rows.append({
                "Repo_name": repo,
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "F1": metrics["f1"],
            })
        repo = None
        metrics = {"precision": None, "recall": None, "f1": None}

    for i, line in enumerate(text.splitlines(), 1):

        # repo from train path
        m = RE_TRAIN.search(line)
        if m:
            flush()
            repo = m.group(1)
            seen_repo += 1
            if debug:
                print(f"[DEBUG] Line {i}: repo = {repo}")
            continue

        if repo:
            m2 = RE_METRIC.search(line)
            if m2:
                k = m2.group(1).lower()
                v = float(m2.group(2))
                metrics[k] = v
                seen_metric += 1
                if debug:
                    print(f"[DEBUG] Line {i}: {repo} {k} = {v}")

    flush()

    if debug:
        print(f"\n[DEBUG] repos found: {seen_repo}")
        print(f"[DEBUG] metric lines: {seen_metric}")
        print(f"[DEBUG] rows produced: {len(rows)}\n")

    return rows

def main():
    log_path = Path(LOG_PATH)
    text = log_path.read_text(errors="replace")

    rows = parse_log(text)

    out = Path(OUT_CSV)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Repo_name","Precision","Recall","F1"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {out.resolve()} ({len(rows)} rows)")

if __name__ == "__main__":
    main()
