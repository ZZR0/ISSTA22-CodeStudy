import os
import json
import argparse
import itertools
import math
from tree_sitter import Language, Parser
import tree_sitter_python as tspython

# Initialize the Tree-sitter Python Language and Parser
PYTHON_LANGUAGE = Language(tspython.language())
parser = Parser(PYTHON_LANGUAGE)

# Common Python built-ins, keywords, and magic methods to ignore during external reference counting
PYTHON_BUILTINS = {
    'print', 'len', 'range', 'int', 'str', 'float', 'bool', 'list', 'dict', 'set', 'tuple',
    'Exception', 'ValueError', 'TypeError', 'self', 'cls', 'True', 'False', 'None', 
    'open', 'enumerate', 'zip', 'map', 'filter', 'isinstance', 'type', 'super', 'append'
}

def calculate_coupling_density(source_code: str) -> dict:
    """
    Traverses the Tree-sitter AST to calculate data-flow coupling.
    It identifies locally declared variables and isolates identifiers that 
    must be external dependencies (In/Out sets for refactoring).
    """
    try:
        tree = parser.parse(bytes(source_code, "utf8"))
        if tree.root_node.has_error and len(tree.root_node.children) == 0:
            return {"density_score": -1, "external_variables": []}
    except Exception:
        return {"density_score": -1, "external_variables": []}

    local_defs = set()
    used_identifiers = set()

    def traverse(node):
        # Track all identifier usages
        if node.type == 'identifier':
            used_identifiers.add(node.text.decode('utf8'))

        # Track Local Definitions via Assignments
        if node.type == 'assignment':
            left_node = node.child_by_field_name('left')
            if left_node:
                if left_node.type == 'identifier':
                    local_defs.add(left_node.text.decode('utf8'))
                elif left_node.type in ('pattern_list', 'tuple', 'list'):
                    # Handle destructuring assignment (e.g., x, y = 1, 2)
                    for child in left_node.children:
                        if child.type == 'identifier':
                            local_defs.add(child.text.decode('utf8'))
                            
        # Track Local Definitions via For-loops
        if node.type == 'for_statement':
            left_node = node.child_by_field_name('left')
            if left_node and left_node.type == 'identifier':
                local_defs.add(left_node.text.decode('utf8'))
                
        # Track Local Definitions via Function Parameters
        if node.type == 'parameters':
            for child in node.children:
                if child.type == 'identifier':
                    local_defs.add(child.text.decode('utf8'))

        for child in node.children:
            traverse(child)

    traverse(tree.root_node)

    # External dependencies = All used identifiers minus local definitions minus built-ins
    external_dependencies = used_identifiers - local_defs - PYTHON_BUILTINS

    return {
        "density_score": len(external_dependencies),
        "external_variables": sorted(list(external_dependencies))
    }

def evaluate_pair_dataflow_coupling(clone_1_code: str, clone_2_code: str) -> dict:
    """
    Computes the Euclidean Coupling Magnitude for a clone pair.
    Score_DF = sqrt(d1^2 + d2^2)
    """
    data1 = calculate_coupling_density(clone_1_code)
    data2 = calculate_coupling_density(clone_2_code)
    
    d1 = data1["density_score"]
    d2 = data2["density_score"]
    
    # Handle parsing errors
    if d1 == -1 or d2 == -1:
        return {
            "c1_external_vars": [],
            "c1_density": -1,
            "c2_external_vars": [],
            "c2_density": -1,
            "df_score": -1.0
        }
        
    # Euclidean calculation for dataflow coupling magnitude
    score_df = math.sqrt((d1 ** 2) + (d2 ** 2))
    
    return {
        "c1_external_vars": data1["external_variables"],
        "c1_density": d1,
        "c2_external_vars": data2["external_variables"],
        "c2_density": d2,
        "df_score": round(score_df, 4)
    }

# ==========================================
# CLI Execution Block
# ==========================================
if __name__ == "__main__":
    cli_parser = argparse.ArgumentParser(description="Evaluate data-flow coupling density of Python clone pairs.")
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
                    
                # Generate combinatorial pairs
                for inst_1, inst_2 in itertools.combinations(sources, 2):
                    coupling_metrics = evaluate_pair_dataflow_coupling(inst_1["code"], inst_2["code"])
                    
                    result_record = {
                        "pair_id": f"{inst_1.get('func_id', 'unknown')}__AND__{inst_2.get('func_id', 'unknown')}",
                        "clone_class_id": clone_class.get("classid"),
                        "dataflow_evaluation": coupling_metrics
                    }
                    
                    outfile.write(json.dumps(result_record) + "\n")
                    processed_pairs_count += 1
                    
        print(f"Done! Successfully evaluated coupling density for {processed_pairs_count} pairs.")
        
    except FileNotFoundError:
        print(f"Error: The input file '{args.input}' was not found.")
    except json.JSONDecodeError:
        print(f"Error: The file '{args.input}' contains invalid JSON formatting.")