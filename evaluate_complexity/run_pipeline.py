import os
import sys
import time
import argparse
import subprocess
from pathlib import Path

def run_step(step_name: str, command: list):
    """
    Executes a shell command via subprocess, monitoring for errors.
    """
    print(f"   ▶ {step_name}")
    try:
        subprocess.run(command, check=True, text=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ ERROR: [{step_name}] failed with return code {e.returncode}.")
        print(f"--- Standard Error Output ---\n{e.stderr}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"\n❌ ERROR: Could not find the script for [{step_name}].")
        sys.exit(1)

def process_dataset(input_jsonl: Path, output_root: Path):
    """
    Runs the 6-step pipeline for a single target dataset.
    Generates dynamic output paths nested safely inside the specified output_root.
    """
    # Extract project name from the parent directory (e.g., "activemq-sim0.7")
    project_name = input_jsonl.parent.name
    base_filename = input_jsonl.name

    print(f"\n{'='*60}")
    print(f"🚀 PROCESSING PROJECT: {project_name}")
    print(f"📄 Input File: {base_filename}")
    print(f"{'='*60}")

    # 1. Define Dynamic Output Paths
    clean_tail = "_filtered_with_func_id.jsonl"
    if clean_tail in base_filename:
        base_out_name = base_filename.replace(clean_tail, "")
    else:
        base_out_name = base_filename.replace(".jsonl", "")

    # Route the output strictly into the requested output_root directory
    out_base_dir = output_root / project_name
    
    out_cf   = out_base_dir / "controlflow" / f"{base_out_name}_eval_cf.jsonl"
    out_sim  = out_base_dir / "similarity" / f"{base_out_name}_eval_sim.jsonl"
    out_sem  = out_base_dir / "semantic" / f"{base_out_name}_eval_sem.jsonl"
    out_df   = out_base_dir / "dataflow" / f"{base_out_name}_eval_df.jsonl"
    out_arch = out_base_dir / "architecture" / f"{base_out_name}_eval_arch.jsonl"
    
    out_merged = out_base_dir / f"{base_out_name}_complexity_merge.jsonl"

    # 2. Define the Pipeline Steps
    pipeline_steps = [
        ("1. Evaluate Control-Flow", ["python", "evaluate_controlflow_complexity.py", "--input", str(input_jsonl), "--output", str(out_cf)]),
        ("2. Evaluate Similarity", ["python", "evaluate_similarity_thresholds.py", "--input", str(input_jsonl), "--output", str(out_sim)]),
        ("3. Evaluate Semantic Divergence", ["python", "evaluate_semantic_divergence.py", "--input", str(input_jsonl), "--output", str(out_sem)]),
        ("4. Evaluate Data-Flow Coupling", ["python", "evaluate_dataflow_coupling.py", "--input", str(input_jsonl), "--output", str(out_df)]),
        ("5. Evaluate Architecture", ["python", "evaluate_architectural_distance.py", "--input", str(input_jsonl), "--output", str(out_arch)]),
        ("6. Merge & Compute Tier", [
            "python", "merge_evaluations.py",
            "--cf", str(out_cf),
            "--sim", str(out_sim),
            "--sem", str(out_sem),
            "--df", str(out_df),
            "--arch", str(out_arch),
            "--output", str(out_merged)
        ])
    ]

    # 3. Execute Steps
    start_time = time.time()
    for step_name, command in pipeline_steps:
        run_step(step_name, command)
        
    elapsed = time.time() - start_time
    print(f"✅ {project_name} completed in {elapsed:.2f} seconds.")
    print(f"📦 Final merged dataset: {out_merged}")

def main():
    parser = argparse.ArgumentParser(description="Batch process code clone datasets through the multi-dimensional complexity matrix.")
    parser.add_argument("--input", type=str, required=True, help="Path to a specific JSONL file OR a directory containing datasets (e.g., ./data)")
    parser.add_argument("--output", type=str, required=True, help="Root directory to save the evaluated outputs (e.g., ./output/complexity/)")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_root = Path(args.output)
    
    # Smart routing: Handle both a single file or a root directory
    if input_path.is_file():
        target_files = [input_path]
    elif input_path.is_dir():
        target_files = list(input_path.rglob("step4_*.jsonl"))
    else:
        print(f"❌ ERROR: Input path '{args.input}' does not exist.")
        sys.exit(1)
        
    if not target_files:
        print(f"❌ No matching 'step4_*.jsonl' datasets found in {input_path.absolute()}")
        sys.exit(1)
        
    print(f"Found {len(target_files)} dataset(s) to process.")
    
    total_start = time.time()
    
    for input_file in target_files:
        process_dataset(input_file, output_root)
        
    total_elapsed = time.time() - total_start
    print(f"\n🎉 ALL DATASETS PROCESSED SUCCESSFULLY in {total_elapsed:.2f} seconds!")
    print(f"All merged curriculum files are located within: {output_root.absolute()}")

if __name__ == "__main__":
    main()