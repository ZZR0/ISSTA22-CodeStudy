import os
import json
import argparse
import itertools
import difflib
from tree_sitter import Language, Parser
import tree_sitter_python as tspython

# Initialize the Tree-sitter Python Language and Parser
PYTHON_LANGUAGE = Language(tspython.language())
parser = Parser(PYTHON_LANGUAGE)

def extract_normalized_text_and_structure(source_code: str):
    """
    Traverses the Tree-sitter AST to extract:
    1. A normalized string of code (no comments, single spaces) for Type-1 checking.
    2. An ordered list of AST node types for Type-2 and Type-3 checking.
    """
    try:
        tree = parser.parse(bytes(source_code, "utf8"))
        if tree.root_node.has_error and len(tree.root_node.children) == 0:
            return "", []
    except Exception:
        return "", []

    normalized_tokens = []
    ast_node_types = []

    def traverse(node):
        # Ignore comments to ensure formatting doesn't disrupt Type-1 checks.
        # Python tree-sitter uses a single 'comment' type for all comments.
        if node.type == 'comment':
            return
            
        ast_node_types.append(node.type)
        
        # If it's a leaf node, capture its text for the normalized string
        if len(node.children) == 0:
            text = node.text.decode('utf8').strip()
            if text:
                normalized_tokens.append(text)
                
        for child in node.children:
            traverse(child)

    traverse(tree.root_node)
    
    # Rebuild a clean, single-spaced string of the code
    normalized_string = " ".join(normalized_tokens)
    return normalized_string, ast_node_types

def evaluate_pair_semantic_divergence(clone_1_code: str, clone_2_code: str, type_3_threshold: float = 0.70) -> dict:
    """
    Evaluates the semantic divergence of a Python clone pair and maps it to a curriculum difficulty tier (1-4).

    Classification Logic:
        - Tier 1 (Type-1): Exact textual matches (ignoring whitespace and comments). 
          Requires minimal LLM alignment effort.
        - Tier 2 (Type-2): Structurally identical (AST matches perfectly) but identifiers/literals differ.
        - Tier 3 (Type-3): High structural overlap (AST similarity >= threshold), meaning statements 
          were added, removed, or modified.
        - Tier 4 (Type-4): Functionally equivalent but syntactically divergent (AST similarity < threshold).
          Requires pure semantic reasoning from the LLM.

    Args:
        clone_1_code (str): Raw source code for the first clone instance.
        clone_2_code (str): Raw source code for the second clone instance.
        type_3_threshold (float): The minimum AST sequence similarity required to be considered Type-3 
                                  rather than falling through to Type-4. Default is 0.70.

    Returns:
        dict: A dictionary containing the clone type, structural similarity, and the 1-4 divergence score.
    """
    norm_1, ast_1 = extract_normalized_text_and_structure(clone_1_code)
    norm_2, ast_2 = extract_normalized_text_and_structure(clone_2_code)
    
    # Fallback for unparseable snippets: Default to highest difficulty
    if not ast_1 or not ast_2:
        return {
            "clone_type": "Unknown/Unparseable",
            "divergence_score": 4,
            "structural_similarity": 0.0,
            "description": "Syntax errors present; relies entirely on LLM semantic embedding."
        }

    # 1. Type-1 Check
    if norm_1 == norm_2:
        return {
            "clone_type": "Type-1",
            "divergence_score": 1,
            "structural_similarity": 1.0,
            "description": "Exact textual match (ignoring whitespace and comments)."
        }

    # 2. Type-2 Check
    if ast_1 == ast_2:
        return {
            "clone_type": "Type-2",
            "divergence_score": 2,
            "structural_similarity": 1.0,
            "description": "Structurally identical; divergent identifiers or literals."
        }

    # 3. Type-3 Check
    matcher = difflib.SequenceMatcher(None, ast_1, ast_2)
    structural_similarity = matcher.ratio()
    
    if structural_similarity >= type_3_threshold:
        return {
            "clone_type": "Type-3",
            "divergence_score": 3,
            "structural_similarity": round(structural_similarity, 3),
            "description": "High structural overlap, but statements were added, removed, or modified."
        }

    # 4. Type-4 Check (Fallback)
    return {
        "clone_type": "Type-4",
        "divergence_score": 4,
        "structural_similarity": round(structural_similarity, 3),
        "description": "Functionally equivalent but syntactically divergent."
    }

# ==========================================
# CLI Execution Block
# ==========================================
if __name__ == "__main__":
    cli_parser = argparse.ArgumentParser(description="Evaluate semantic divergence (Type 1-4) of Python clone pairs.")
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
                    divergence_metrics = evaluate_pair_semantic_divergence(inst_1["code"], inst_2["code"])
                    
                    result_record = {
                        "pair_id": f"{inst_1.get('func_id', 'unknown')}__AND__{inst_2.get('func_id', 'unknown')}",
                        "clone_class_id": clone_class.get("classid"),
                        "semantic_evaluation": divergence_metrics
                    }
                    
                    outfile.write(json.dumps(result_record) + "\n")
                    processed_pairs_count += 1
                    
        print(f"Done! Successfully evaluated and classified {processed_pairs_count} pairs.")
        
    except FileNotFoundError:
        print(f"Error: The input file '{args.input}' was not found.")
    except json.JSONDecodeError:
        print(f"Error: The file '{args.input}' contains invalid JSON formatting.")