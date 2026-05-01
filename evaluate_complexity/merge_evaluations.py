import os
import json
import argparse

def normalize_metric(value: float, min_val: float, max_val: float) -> float:
    """Normalizes a value to a 0.0 - 1.0 scale, capping at the boundaries."""
    if value <= min_val:
        return 0.0
    if value >= max_val:
        return 1.0
    return (value - min_val) / (max_val - min_val)

def compute_curriculum_tier(metrics: dict) -> dict:
    """
    Synthesizes the 5 orthogonal dimensions into a single Curriculum Difficulty Score.
    Higher score = harder for the LLM = later in the curriculum.
    """
    
    # 1. Control-Flow (Unbounded, typical range 1 to 15+)
    # Normalize: Assume a score of 1 is easy (0.0), >= 15 is max difficulty (1.0)
    cf_raw = metrics["control_flow"]["pair_complexity_score"]
    cf_norm = normalize_metric(cf_raw, 1.0, 15.0) if cf_raw != -1 else 1.0
    
    # 2. Similarity (Bounded 0.0 to 1.0, where 1.0 is identical/EASY)
    # We INVERT this so 1.0 means completely different/HARD
    sim_raw = metrics["similarity"]["composite_similarity_score"]
    sim_norm = 1.0 - sim_raw
    
    # 3. Semantic Divergence (Bounded 1 to 4)
    # Normalize: Type-1 (0.0), Type-2 (0.33), Type-3 (0.66), Type-4 (1.0)
    sem_raw = metrics["semantic"]["divergence_score"]
    sem_norm = normalize_metric(sem_raw, 1.0, 4.0)
    
    # 4. Data-Flow Coupling (Unbounded Euclidean magnitude, typical range 0 to 8+)
    # Normalize: 0 external vars is easy (0.0), >= 8 is max difficulty (1.0)
    df_raw = metrics["dataflow"]["pair_coupling_score"]
    df_norm = normalize_metric(df_raw, 0.0, 8.0) if df_raw != -1 else 1.0
    
    # 5. Architectural Distance (Bounded 1 to 4)
    arch_raw = metrics["architecture"]["architectural_tier"]
    arch_norm = normalize_metric(arch_raw, 1.0, 4.0)

    # Calculate Weighted Composite Score (0.0 to 1.0)
    # You can adjust these weights based on your paper's domain-adaptive priorities!
    composite_score = (
        (cf_norm * 0.15) +   # Internal logic
        (sim_norm * 0.20) +  # Surface pattern matching
        (sem_norm * 0.25) +  # Structural semantics
        (df_norm * 0.25) +   # Extraction/Refactoring difficulty (Highly weighted)
        (arch_norm * 0.15)   # Contextual isolation
    )
    
    # Map back to a strict Curriculum Tier (1 to 4) for batching
    if composite_score <= 0.25:
        tier = 1
    elif composite_score <= 0.50:
        tier = 2
    elif composite_score <= 0.75:
        tier = 3
    else:
        tier = 4

    return {
        "composite_difficulty_score": round(composite_score, 4),
        "curriculum_tier": tier
    }

def main():
    parser = argparse.ArgumentParser(description="Merge 5 complexity dimensions into a final curriculum dataset.")
    parser.add_argument("--cf", required=True, help="Path to Control Flow JSONL")
    parser.add_argument("--sim", required=True, help="Path to Similarity JSONL")
    parser.add_argument("--sem", required=True, help="Path to Semantic JSONL")
    parser.add_argument("--df", required=True, help="Path to Dataflow JSONL")
    parser.add_argument("--arch", required=True, help="Path to Architecture JSONL")
    parser.add_argument("--output", required=True, help="Path to save merged JSONL")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    # Use a dictionary to join records safely on `pair_id`
    merged_data = {}

    # Helper function to load a file and inject its metric
    def load_and_merge(filepath, metric_key):
        print(f"Loading {metric_key} from {filepath}...")
        with open(filepath, 'r') as f:
            for line in f:
                if not line.strip(): continue
                record = json.loads(line.strip())
                pid = record["pair_id"]
                
                if pid not in merged_data:
                    merged_data[pid] = {
                        "pair_id": pid,
                        "clone_class_id": record["clone_class_id"],
                        "metrics": {}
                    }
                
                # Extract the specific evaluation payload
                # E.g., "complexity_evaluation", "similarity_evaluation", etc.
                eval_payload = next(v for k, v in record.items() if k.endswith("_evaluation"))
                merged_data[pid]["metrics"][metric_key] = eval_payload

    # 1. Load all 5 dimensions
    load_and_merge(args.cf, "control_flow")
    load_and_merge(args.sim, "similarity")
    load_and_merge(args.sem, "semantic")
    load_and_merge(args.df, "dataflow")
    load_and_merge(args.arch, "architecture")

    # 2. Calculate composite scores and write output
    print(f"\nMerging and calculating curriculum tiers for {len(merged_data)} pairs...")
    
    with open(args.output, 'w') as outfile:
        for pid, data in merged_data.items():
            # Only process pairs that successfully passed through all 5 evaluators
            if len(data["metrics"]) == 5:
                # Compute the final tier
                scoring = compute_curriculum_tier(data["metrics"])
                data.update(scoring)
                
                outfile.write(json.dumps(data) + "\n")
            else:
                print(f"Warning: Pair {pid} is missing metrics and was skipped.")

    print(f"Done! Merged dataset saved to: {args.output}")

if __name__ == "__main__":
    main()