import os
import json
import argparse
import itertools
from pathlib import Path

def calculate_tree_distance(path1: str, path2: str) -> dict:
    """
    Calculates the structural distance between two files in a repository tree.
    It counts the number of edge traversals (up to a common ancestor, then down)
    required to navigate from path1 to path2.
    """
    p1 = Path(path1)
    p2 = Path(path2)
    
    # If they are the exact same file
    if p1 == p2:
        return {
            "distance_metric": 0,
            "common_ancestor": str(p1.parent)
        }
    
    # Find the deepest common ancestor directory
    parts1 = p1.parts
    parts2 = p2.parts
    
    common_prefix_length = 0
    for part1, part2 in zip(parts1, parts2):
        if part1 == part2:
            common_prefix_length += 1
        else:
            break
            
    common_ancestor = Path(*parts1[:common_prefix_length])
    
    # Distance is the number of steps UP to the common ancestor, 
    # plus the number of steps DOWN to the target file.
    steps_up = len(parts1) - common_prefix_length
    steps_down = len(parts2) - common_prefix_length
    total_distance = steps_up + steps_down
    
    return {
        "distance_metric": total_distance,
        "common_ancestor": str(common_ancestor)
    }

def evaluate_pair_architectural_distance(file1_path: str, file2_path: str) -> dict:
    """
    Evaluates the architectural distance of a clone pair and maps it to a 
    curriculum difficulty tier.
    """
    distance_data = calculate_tree_distance(file1_path, file2_path)
    dist = distance_data["distance_metric"]
    
    # Tier Categorization based on directory traversal distance
    if dist == 0:
        tier = "Intra-file (Distance 0)"
    elif dist <= 2:
        tier = "Sibling/Adjacent (Distance 1-2)"
    elif dist <= 4:
        tier = "Distant Relatives (Distance 3-4)"
    else:
        tier = "Globally Dispersed (Distance > 4)"
        
    return {
        "c1_path": file1_path,
        "c2_path": file2_path,
        "common_ancestor": distance_data["common_ancestor"],
        "architectural_distance": dist,
        "distance_tier": tier
    }

# ==========================================
# CLI Execution Block
# ==========================================
if __name__ == "__main__":
    cli_parser = argparse.ArgumentParser(description="Evaluate architectural distance of Python clone pairs in a repository.")
    cli_parser.add_argument("--input", type=str, required=True, help="Path to the input JSONL file")
    cli_parser.add_argument("--output", type=str, required=True, help="Path to save the evaluated JSONL output")
    
    args = cli_parser.parse_args()
    
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    print(f"Reading from : {args.input}")
    print(f"Writing to   : {args.output}\n")
    
    processed_pairs_count = 0
    
    try:
        with open(args.input, 'r') as infile, open(args.output, 'w') as outfile:
            for line in infile:
                if not line.strip():
                    continue
                    
                clone_class = json.loads(line.strip())
                sources = clone_class.get("sources", [])
                
                if len(sources) < 2:
                    continue
                    
                for inst_1, inst_2 in itertools.combinations(sources, 2):
                    # Note: We pass the 'file' key here, not the 'code' key
                    arch_metrics = evaluate_pair_architectural_distance(inst_1["file"], inst_2["file"])
                    
                    result_record = {
                        "pair_id": f"{inst_1.get('func_id', 'unknown')}__AND__{inst_2.get('func_id', 'unknown')}",
                        "clone_class_id": clone_class.get("classid"),
                        "architectural_evaluation": arch_metrics
                    }
                    
                    outfile.write(json.dumps(result_record) + "\n")
                    processed_pairs_count += 1
                    
        print(f"Done! Successfully evaluated architectural distance for {processed_pairs_count} Python pairs.")
        
    except FileNotFoundError:
        print(f"Error: The input file '{args.input}' was not found.")
    except json.JSONDecodeError:
        print(f"Error: The file '{args.input}' contains invalid JSON formatting.")