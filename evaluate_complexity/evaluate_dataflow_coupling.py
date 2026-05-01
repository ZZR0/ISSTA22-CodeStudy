import os
import json
import argparse
import itertools
from tree_sitter import Language, Parser
import tree_sitter_java as tsjava

# Initialize the Tree-sitter Java Language and Parser
JAVA_LANGUAGE = Language(tsjava.language())
parser = Parser(JAVA_LANGUAGE)

# Common Java built-ins and keywords to ignore during external reference counting
JAVA_BUILTINS = {
    'System', 'out', 'err', 'in', 'String', 'Object', 'Math', 'Integer', 
    'Boolean', 'Double', 'Float', 'Long', 'Exception', 'RuntimeException',
    'this', 'super', 'class', 'true', 'false', 'null'
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
        # 1. Track Local Definitions
        # variable_declarator generally contains the identifier name being defined
        if node.type == 'variable_declarator':
            name_node = node.child_by_field_name('name')
            if name_node:
                local_defs.add(name_node.text.decode('utf8'))
                
        # formal_parameter (e.g., inside a catch block or loop)
        elif node.type == 'formal_parameter' or node.type == 'catch_formal_parameter':
            name_node = node.child_by_field_name('name')
            if name_node:
                local_defs.add(name_node.text.decode('utf8'))
                
        # 2. Track Usages (Identifiers)
        # We want to capture identifiers, but ignore method invocation names (which are structure, not data)
        elif node.type == 'identifier':
            parent_type = node.parent.type if node.parent else ""
            if parent_type != 'method_invocation':
                ident_name = node.text.decode('utf8')
                if ident_name not in JAVA_BUILTINS:
                    used_identifiers.add(ident_name)

        # Recursively visit children
        for child in node.children:
            traverse(child)

    traverse(tree.root_node)
    
    # External references are identifiers used in the block but NEVER defined in the block
    external_refs = used_identifiers - local_defs

    return {
        "density_score": len(external_refs),
        "external_variables": list(external_refs)
    }

import math

def evaluate_pair_dataflow_coupling(clone_1_code: str, clone_2_code: str) -> dict:
    """
    Evaluates the data-flow coupling density for a pair of Java clone candidates.
    
    Mathematical Computation: Euclidean Vector Magnitude
        Score = sqrt(c1^2 + c2^2)
        
    Rationale:
        A simple `max()` function fails to capture the cumulative cognitive load when 
        BOTH clones possess high coupling. By modeling the external variable dependencies 
        as coordinates in a 2D coupling space, the Euclidean magnitude scales non-linearly.
        A highly entangled pair (e.g., 4 and 4 external vars) yields a higher magnitude (5.66) 
        than an asymmetrical pair (4 and 0 vars -> 4.0), accurately reflecting the compounded 
        difficulty of resolving multiple extraction boundaries simultaneously.
    """
    c1_metrics = calculate_coupling_density(clone_1_code)
    c2_metrics = calculate_coupling_density(clone_2_code)
    
    c1_score = c1_metrics["density_score"]
    c2_score = c2_metrics["density_score"]
    
    if c1_score == -1 or c2_score == -1:
        pair_score = -1.0
    else:
        # Calculate the Euclidean magnitude of the coupling vector
        pair_score = round(math.sqrt((c1_score ** 2) + (c2_score ** 2)), 2)

    return {
        "clone_1_coupling": c1_metrics,
        "clone_2_coupling": c2_metrics,
        "pair_coupling_score": pair_score,
        "description": f"Euclidean coupling magnitude of {pair_score} based on external dependencies."
    }

# ==========================================
# CLI Execution Block
# ==========================================
if __name__ == "__main__":
    cli_parser = argparse.ArgumentParser(description="Evaluate data-flow coupling (external variable dependencies) of Java clone pairs.")
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