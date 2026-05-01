import os
import json
import argparse
import itertools
import difflib
from tree_sitter import Language, Parser
import tree_sitter_java as tsjava

# Initialize the Tree-sitter Java Language and Parser
JAVA_LANGUAGE = Language(tsjava.language())
parser = Parser(JAVA_LANGUAGE)

def extract_tokens_and_structure(source_code: str):
    """
    Traverses the Tree-sitter AST to extract both lexical tokens (leaf node text)
    and the structural skeleton (node types).
    Comments are explicitly ignored to prevent formatting biases.
    """
    try:
        tree = parser.parse(bytes(source_code, "utf8"))
        if tree.root_node.has_error and len(tree.root_node.children) == 0:
            return [], []
    except Exception:
        return [], []

    lexical_tokens = []
    ast_node_types = []

    def traverse(node):
        # Ignore comments to ensure we only evaluate semantic code content
        if node.type in {'line_comment', 'block_comment'}:
            return
            
        # Record the structural sequence
        ast_node_types.append(node.type)
        
        # If it's a leaf node, record its lexical text
        if len(node.children) == 0:
            text = node.text.decode('utf8').strip()
            if text:
                lexical_tokens.append(text)
                
        # Recursively visit children
        for child in node.children:
            traverse(child)

    traverse(tree.root_node)
    return lexical_tokens, ast_node_types

def calculate_jaccard_similarity(list1: list, list2: list) -> float:
    """Calculates Jaccard similarity between two lists (surface lexical overlap)."""
    set1, set2 = set(list1), set(list2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0

def calculate_sequence_similarity(seq1: list, seq2: list) -> float:
    """Calculates Ratcliff/Obershelp similarity (structural sequence overlap)."""
    if not seq1 or not seq2:
        return 0.0
    matcher = difflib.SequenceMatcher(None, seq1, seq2)
    return matcher.ratio()

def evaluate_similarity_thresholds(clone_1_code: str, clone_2_code: str) -> dict:
    """
    Evaluates both the lexical (surface text) and structural (AST skeleton) 
    similarity between two Java clone candidates.
    
    The composite score is calculated using the Harmonic Mean to heavily 
    penalize asymmetrical similarities, mirroring the mechanics of an F1-score.
    """
    lexical_1, structure_1 = extract_tokens_and_structure(clone_1_code)
    lexical_2, structure_2 = extract_tokens_and_structure(clone_2_code)
    
    lexical_sim = calculate_jaccard_similarity(lexical_1, lexical_2)
    structural_sim = calculate_sequence_similarity(structure_1, structure_2)
    
    # Advanced Computation: Harmonic Mean
    # We add a tiny epsilon (1e-9) to the denominator to prevent ZeroDivisionError 
    # in the rare case that both similarities evaluate to absolute 0.0.
    denominator = (lexical_sim + structural_sim) + 1e-9
    composite_score = (2.0 * lexical_sim * structural_sim) / denominator

    return {
        "lexical_similarity": round(lexical_sim, 4),
        "structural_similarity": round(structural_sim, 4),
        "composite_similarity_score": round(composite_score, 4)
    }

# ==========================================
# CLI Execution Block
# ==========================================
if __name__ == "__main__":
    cli_parser = argparse.ArgumentParser(description="Evaluate lexical and structural similarity of Java clone pairs.")
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
                    similarity_metrics = evaluate_similarity_thresholds(inst_1["code"], inst_2["code"])
                    
                    result_record = {
                        "pair_id": f"{inst_1.get('func_id', 'unknown')}__AND__{inst_2.get('func_id', 'unknown')}",
                        "clone_class_id": clone_class.get("classid"),
                        "similarity_evaluation": similarity_metrics
                    }
                    
                    outfile.write(json.dumps(result_record) + "\n")
                    processed_pairs_count += 1
                    
        print(f"Done! Successfully evaluated and saved {processed_pairs_count} pairs.")
        
    except FileNotFoundError:
        print(f"Error: The input file '{args.input}' was not found.")
    except json.JSONDecodeError:
        print(f"Error: The file '{args.input}' contains invalid JSON formatting.")