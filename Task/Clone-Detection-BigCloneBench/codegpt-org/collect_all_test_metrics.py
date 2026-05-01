import re
import csv
from pathlib import Path

# Path configuration
LOG_PATH = Path("/home/user1-system11/research_dream/llm-clone/Task/Clone-Detection-BigCloneBench/codegpt-org/saved_models_combined/logs/test_all_systems_codegpt_bcb_java_block_function.log")
OUT_CSV = Path("/home/user1-system11/research_dream/llm-clone/Task/Clone-Detection-BigCloneBench/codegpt-org/saved_models_combined/repo_metrics_codegpt_bcb_java_block_function_after_adaption.csv")

# Regex: Matches the repository name from test or train lines
# Regex patterns
RE_REPO = re.compile(r"TESTING\s+(\S+)\s+={2,}")
RE_METRIC = re.compile(r"eval_(precision|recall|f1)\s*=\s*([0-9]*\.?[0-9]+)")
RE_PAIRS = re.compile(r"Num examples\s*=\s*(\d+)")

def parse_log(log_file):
    rows = []
    current_repo = None
    data = {}

    with log_file.open("r", errors="replace") as f:
        for line in f:
            # 1. New repo found: save previous data and start fresh
            repo_match = RE_REPO.search(line)
            if repo_match:
                if current_repo and data:
                    rows.append({"Repo_name": current_repo, **data})
                current_repo = repo_match.group(1)
                data = {"Pairs_loaded": None, "Precision": None, "Recall": None, "F1": None}
                continue

            # 2. Extract pairs_loaded
            pairs_match = RE_PAIRS.search(line)
            if pairs_match and current_repo:
                data["Pairs_loaded"] = pairs_match.group(1)

            # 3. Extract metrics
            metric_match = RE_METRIC.search(line)
            if metric_match and current_repo:
                key = metric_match.group(1).capitalize()
                data[key] = metric_match.group(2)

    # Append the last repository
    if current_repo and data:
        rows.append({"Repo_name": current_repo, **data})
        
    return rows

def main():
    rows = parse_log(LOG_PATH)
    
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["Repo_name", "Pairs_loaded", "Precision", "Recall", "F1"]
    
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Parsed {len(rows)} repositories. Saved to: {OUT_CSV}")

if __name__ == "__main__":
    main()