import re
from pathlib import Path
import csv

LOG_PATH = "/home/user1-system11/research_dream/llm-clone/Task/Clone-Detection-BigCloneBench/codet5/saved_models_combined_py/python/test_all_systems_codet5_bcb_python_combined.log"
OUT_CSV  = "/home/user1-system11/research_dream/llm-clone/Task/Clone-Detection-BigCloneBench/codet5/saved_models_combined_py/python/repo_metrics_codeT5_bcb_python.csv"

# Repo header from bash loop:
# -------------------- [activemq] Sun Feb  8 01:23:38 PM CST 2026 --------------------
RE_REPO = re.compile(r"^-{5,}\s*\[([^\]]+)\]\s+.*-+\s*$")

# Metrics from CodeT5 logging (NO threshold)
RE_EVAL_METRIC = re.compile(
    r"\beval_(f1|precision|recall)\s*=\s*([0-9]*\.?[0-9]+)\b",
    re.IGNORECASE
)

def parse_log(text: str, debug: bool = False):
    results_by_repo = {}
    repo = None
    metrics = {"precision": None, "recall": None, "f1": None}

    seen_repo = 0
    seen_metric = 0

    def flush():
        nonlocal repo, metrics
        if repo and any(v is not None for v in metrics.values()):
            results_by_repo[repo] = {
                "Repo_name": repo,
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "F1": metrics["f1"],
            }
        repo = None
        metrics = {"precision": None, "recall": None, "f1": None}

    for i, line in enumerate(text.splitlines(), 1):
        m = RE_REPO.match(line.strip())
        if m:
            flush()
            repo = m.group(1).strip()
            seen_repo += 1
            if debug:
                print(f"[DEBUG] Line {i}: repo = {repo}")
            continue

        if repo:
            m2 = RE_EVAL_METRIC.search(line)
            if m2:
                k = m2.group(1).lower()
                v = float(m2.group(2))
                metrics[k] = v
                seen_metric += 1
                if debug:
                    print(f"[DEBUG] Line {i}: {repo} eval_{k} = {v}")

    flush()

    rows = list(results_by_repo.values())

    if debug:
        print(f"\n[DEBUG] repo blocks found: {seen_repo}")
        print(f"[DEBUG] metric lines found: {seen_metric}")
        print(f"[DEBUG] unique repos with metrics: {len(rows)}\n")

    rows.sort(key=lambda r: r["Repo_name"].lower())
    return rows

def main():
    text = Path(LOG_PATH).read_text(errors="replace")
    rows = parse_log(text, debug=False)

    out = Path(OUT_CSV)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["Repo_name", "Precision", "Recall", "F1"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {out.resolve()} ({len(rows)} repos)")

if __name__ == "__main__":
    main()
