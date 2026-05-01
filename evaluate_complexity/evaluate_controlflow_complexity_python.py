import os
import json
import argparse
import itertools
from tree_sitter import Language, Parser
import tree_sitter_python as tspython

# Initialize the Tree-sitter Python Language and Parser
PYTHON_LANGUAGE = Language(tspython.language())
parser = Parser(PYTHON_LANGUAGE)

def count_branching_nodes(node) -> int:
    """
    Recursively traverses the Tree-sitter AST to count Python control-flow branches.
    """
    complexity_addition = 0
    
    # Python-specific AST nodes that create control-flow branches
    branch_node_types = {
        'if_statement', 
        'elif_clause', 
        'for_statement', 
        'while_statement', 
        'except_clause', 
        'case_clause',             # Python 3.10+ match/case
        'conditional_expression'   # Python's ternary: a if b else c
    }
    
    if node.type in branch_node_types:
        complexity_addition += 1
        
    # In tree-sitter-python, 'and' and 'or' are categorized as 'boolean_operator'
    elif node.type == 'boolean_operator':
        complexity_addition += 1

    for child in node.children:
        complexity_addition += count_branching_nodes(child)
        
    return complexity_addition

def calculate_snippet_complexity(source_code: str) -> int:
    """
    Parses the Python source code into a Tree-sitter AST and computes its cyclomatic complexity.
    """
    try:
        tree = parser.parse(bytes(source_code, "utf8"))
        # Base complexity is 1 for a single execution path
        return 1 + count_branching_nodes(tree.root_node)
    except Exception:
        return 1

def evaluate_pair_complexity(code1: str, code2: str, gamma: float = 0.5) -> dict:
    """
    Computes the Asymmetry-Penalized Alignment Gap for a code clone pair.
    Score_CF = max(c1, c2) + gamma * |c1 - c2|
    """
    c1 = calculate_snippet_complexity(code1)
    c2 = calculate_snippet_complexity(code2)
    score_cf = max(c1, c2) + gamma * abs(c1 - c2)
    
    return {
        "c1_complexity": c1,
        "c2_complexity": c2,
        "cf_score": score_cf
    }

def main():
    parser_arg = argparse.ArgumentParser(description="Evaluate control-flow complexity for Python clone pairs.")
    parser_arg.add_argument("--input", required=True, help="Path to the input JSON lines file")
    parser_arg.add_argument("--output", required=True, help="Path to the output JSON lines file")
    parser_arg.add_argument("--gamma", type=float, default=0.5, help="Gamma penalty parameter (default: 0.5)")
    args = parser_arg.parse_args()

    processed_pairs_count = 0

    try:
        with open(args.input, 'r', encoding='utf-8') as infile, \
             open(args.output, 'w', encoding='utf-8') as outfile:
             
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
                    complexity_metrics = evaluate_pair_complexity(
                        inst_1.get("code", ""), 
                        inst_2.get("code", ""),
                        gamma=args.gamma
                    )
                    
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
        print(f"Error: The input file '{args.input}' was not found.")
    except json.JSONDecodeError:
        print(f"Error: The file '{args.input}' contains invalid JSON formatting.")

if __name__ == "__main__":
    main()