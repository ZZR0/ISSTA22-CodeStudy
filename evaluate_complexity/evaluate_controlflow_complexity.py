import os
import json
import argparse
import itertools
from tree_sitter import Language, Parser
import tree_sitter_java as tsjava

# Initialize the Tree-sitter Java Language and Parser
JAVA_LANGUAGE = Language(tsjava.language())
parser = Parser(JAVA_LANGUAGE)

def count_branching_nodes(node) -> int:
    """
    Recursively traverses the Tree-sitter AST to count Java control-flow branches.
    """
    complexity_addition = 0
    
    branch_node_types = {
        'if_statement', 'for_statement', 'enhanced_for_statement', 
        'while_statement', 'do_statement', 'catch_clause', 
        'switch_label', 'ternary_expression'
    }
    
    if node.type in branch_node_types:
        complexity_addition += 1
        
    elif node.type == 'binary_expression':
        operator_node = node.child_by_field_name('operator')
        if operator_node and operator_node.type in {'&&', '||'}:
            complexity_addition += 1

    for child in node.children:
        complexity_addition += count_branching_nodes(child)
        
    return complexity_addition

def calculate_snippet_complexity(source_code: str) -> int:
    """
    Parses the Java source code into a Tree-sitter AST and computes its cyclomatic complexity.
    """
    try:
        tree = parser.parse(bytes(source_code, "utf8"))
        if tree.root_node.has_error and len(tree.root_node.children) == 0:
            return -1
        return 1 + count_branching_nodes(tree.root_node)
    except Exception:
        return -1

def evaluate_pair_complexity(clone_1_code: str, clone_2_code: str, gamma: float = 0.5) -> dict:
    """
    Evaluates the composite control-flow complexity of a clone pair using an 
    Asymmetry-Penalized Alignment Gap model.

    Mathematical Computation:
        Score = max(c1, c2) + (gamma * |c1 - c2|)

    Rationale:
        A simple `max(c1, c2)` ignores structural asymmetry. For an LLM attempting 
        to map representations (or an engine attempting to extract them), a clone 
        pair with complexities (2, 8) is fundamentally harder to process than a 
        symmetrical pair of (8, 8). The (2, 8) pair contains a massive structural 
        gap (e.g., mapping straight-line logic to heavily nested try/catch blocks).
        
        This formula accounts for both dimensions of difficulty:
        1. The Cognitive Ceiling: `max(c1, c2)` ensures the baseline difficulty 
           matches the hardest snippet the LLM must process.
        2. The Alignment Penalty: `gamma * |c1 - c2|` strictly penalizes the 
           structural divergence characteristic of complex Type-3 and Type-4 clones.

    Args:
        clone_1_code (str): Raw source code for the first clone instance.
        clone_2_code (str): Raw source code for the second clone instance.
        gamma (float): The weighting factor for the structural gap penalty. 
                       Default is 0.5.

    Returns:
        dict: A dictionary containing individual scores, the calculated gap, 
              and the final composite pair score.
    """
    c1_score = calculate_snippet_complexity(clone_1_code)
    c2_score = calculate_snippet_complexity(clone_2_code)
    
    # If either snippet fails to parse (SyntaxError / Tree-sitter error)
    if c1_score == -1 or c2_score == -1:
        pair_score = -1 
        structural_gap = -1
    else:
        base_ceiling = max(c1_score, c2_score)
        structural_gap = abs(c1_score - c2_score)
        
        # Calculate the final penalized score
        pair_score = round(base_ceiling + (gamma * structural_gap), 2)

    return {
        "clone_1_complexity": c1_score,
        "clone_2_complexity": c2_score,
        "structural_gap": structural_gap,
        "pair_complexity_score": pair_score
    }

# ==========================================
# CLI Execution Block
# ==========================================
if __name__ == "__main__":
    # 1. Set up the argument parser
    cli_parser = argparse.ArgumentParser(description="Evaluate control-flow complexity of Java clone pairs.")
    cli_parser.add_argument(
        "--input", 
        type=str, 
        required=True, 
        help="Path to the input JSONL file (e.g., ./data/activemq-sim0.7/step1_nicad_activemq_sim0.7_raw.jsonl)"
    )
    cli_parser.add_argument(
        "--output", 
        type=str, 
        required=True, 
        help="Path to save the evaluated JSONL output (e.g., ./output/step1_nicad_activemq_sim0.7_evaluated.jsonl)"
    )
    
    args = cli_parser.parse_args()
    input_filepath = args.input
    output_filepath = args.output
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_filepath)), exist_ok=True)
    
    print(f"Reading from : {input_filepath}")
    print(f"Writing to   : {output_filepath}\n")
    
    processed_pairs_count = 0
    
    # 2. Process files
    try:
        with open(input_filepath, 'r') as infile, open(output_filepath, 'w') as outfile:
            for line in infile:
                if not line.strip():
                    continue
                    
                clone_class = json.loads(line.strip())
                sources = clone_class.get("sources", [])
                
                if len(sources) < 2:
                    continue
                    
                # Generate all unique combinations within this clone class
                for inst_1, inst_2 in itertools.combinations(sources, 2):
                    
                    # Evaluate complexity
                    complexity_metrics = evaluate_pair_complexity(inst_1["code"], inst_2["code"])
                    
                    # Construct output record
                    result_record = {
                        "pair_id": f"{inst_1.get('func_id', 'unknown')}__AND__{inst_2.get('func_id', 'unknown')}",
                        "clone_class_id": clone_class.get("classid"),
                        "complexity_evaluation": complexity_metrics
                    }
                    
                    # Write the result to the output file
                    outfile.write(json.dumps(result_record) + "\n")
                    processed_pairs_count += 1
                    
        print(f"Done! Successfully evaluated and saved {processed_pairs_count} pairs.")
        
    except FileNotFoundError:
        print(f"Error: The input file '{input_filepath}' was not found.")
    except json.JSONDecodeError:
        print(f"Error: The file '{input_filepath}' contains invalid JSON formatting.")