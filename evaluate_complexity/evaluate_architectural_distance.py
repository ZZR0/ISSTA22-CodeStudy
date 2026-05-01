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

def evaluate_pair_architectural_distance(clone_1_filepath: str, clone_2_filepath: str) -> dict:
    """
    Categorizes the clone pair into a 1-4 curriculum tier based on repository topology.
    """
    tree_metrics = calculate_tree_distance(clone_1_filepath, clone_2_filepath)
    distance = tree_metrics["distance_metric"]
    
    # Tier 1: Intra-file proximity (Distance 0)
    if distance == 0:
        tier = 1
        description = "Intra-file proximity. Clones share identical namespace and imports."
        
    # Tier 2: Intra-package proximity (Distance 1-2)
    # E.g., Up 1 to parent dir, Down 1 to sibling file
    elif distance <= 2:
        tier = 2
        description = "Intra-package proximity. Clones reside in the same directory/module."
        
    # Tier 3: Cross-package proximity (Distance 3-4)
    # E.g., Up 2 directories, Down 2 directories
    elif distance <= 4:
        tier = 3
        description = "Cross-package proximity. Clones share a recent common ancestor directory."
        
    # Tier 4: Globally dispersed (Distance > 4)
    else:
        tier = 4
        description = "Inter-package dispersion. Clones are architecturally isolated."

    return {
        "tree_distance": distance,
        "architectural_tier": tier,
        "description": description,
        "common_ancestor": tree_metrics["common_ancestor"]
    }

# ==========================================
# CLI Execution Block
# ==========================================
if __name__ == "__main__":
    cli_parser = argparse.ArgumentParser(description="Evaluate architectural/file distance of clone pairs.")
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
                    
        print(f"Done! Successfully evaluated architectural distance for {processed_pairs_count} pairs.")
        
    except FileNotFoundError:
        print(f"Error: The input file '{args.input}' was not found.")
    except json.JSONDecodeError:
        print(f"Error: The file '{args.input}' contains invalid JSON formatting.")