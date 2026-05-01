import json
import os
import sys
import html
import re

DEFAULT_INPUT = "data/nicad_camel_clone_func.jsonl"
OUTPUT_FILE = "min_tokens_standard.html"

def java_tokenize_standard(code):
    """
    A standard lexical tokenizer for Java.
    Counts: Keywords, Identifiers, Literals, Operators.
    Ignores: Whitespace, Comments.
    """
    token_pattern = re.compile(
        r'''
        "(?:\\.|[^\\"])*"         |  # String Literal
        '(?:\\.|[^\\'])*'         |  # Char Literal
        //.*?$                    |  # Line Comment (matches but we ignore)
        /\*.*?\*/                 |  # Block Comment (matches but we ignore)
        \b0[xX][0-9a-fA-F]+\b     |  # Hex Number
        \b[0-9]+\.?[0-9]*(?:[eE][-+]?[0-9]+)?\b | # Decimal Number (FIXED: Non-capturing)
        @[a-zA-Z_$][a-zA-Z0-9_$]* |  # Annotation
        [a-zA-Z_$][a-zA-Z0-9_$]* |  # Identifier / Keyword
        [(){}\[\],.;:?!~+\-*/%&|^=<>]+ # Operators & Punctuation
        ''',
        re.VERBOSE | re.MULTILINE | re.DOTALL
    )
    
    # Find all matches
    raw_matches = token_pattern.findall(code)
    
    # Filter out comments (start with // or /*)
    clean_tokens = [t for t in raw_matches if not (t.startswith('//') or t.startswith('/*'))]
    
    return clean_tokens

def get_min_token_functions(file_path, top_n=10):
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return

    print(f"Reading data from: {file_path}")
    functions = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    data = json.loads(line)
                    sources = data.get("sources", [])
                    
                    for src in sources:
                        code = src.get("code", "")
                        # Use qualified name if available, else file path
                        name = src.get("qualified_name", src.get("file", "Unknown"))
                        
                        # Use Standard Tokenizer
                        tokens = java_tokenize_standard(code)
                        count = len(tokens)
                        
                        functions.append((count, name, code))
                        
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"An error occurred: {e}")
        return

    # Sort by tokens (ascending) to find the minimums
    functions.sort(key=lambda x: x[0])
    
    # Take the top N
    top_functions = functions[:top_n]

    # Generate HTML Output
    html_output = []
    html_output.append("<html><body>")
    html_output.append("<h3>Top 10 Functions with Minimum Tokens (Standard Counter)</h3>")
    html_output.append("<p><i>* This count ignores whitespace and comments. It represents logical code size.</i></p>")
    html_output.append("<table border='1' style='border-collapse: collapse; width: 100%; text-align: left; font-family: Arial, sans-serif;'>")
    html_output.append("<tr style='background-color: #e0f7fa;'>")
    html_output.append("<th style='padding: 8px;'>Rank</th>")
    html_output.append("<th style='padding: 8px;'>Standard Tokens</th>")
    html_output.append("<th style='padding: 8px;'>Function Name / File</th>")
    html_output.append("<th style='padding: 8px;'>Code Snippet</th>")
    html_output.append("</tr>")
    
    for i, (tokens, name, code) in enumerate(top_functions, 1):
        escaped_code = html.escape(code)
        escaped_name = html.escape(name)
        
        html_output.append("<tr>")
        html_output.append(f"<td style='padding: 8px;'>{i}</td>")
        html_output.append(f"<td style='padding: 8px; font-weight: bold;'>{tokens}</td>")
        html_output.append(f"<td style='padding: 8px; font-size: 0.9em;'>{escaped_name}</td>")
        html_output.append(f"<td style='padding: 8px;'><pre style='background-color: #f8f8f8; padding: 5px; margin: 0; white-space: pre-wrap; font-size: 0.85em;'>{escaped_code}</pre></td>")
        html_output.append("</tr>")
    
    html_output.append("</table>")
    html_output.append("</body></html>")
    
    # Save to file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(html_output))
    
    print(f"\nSuccessfully generated HTML report: {OUTPUT_FILE}")
    print("You can open this file in your browser.")

if __name__ == "__main__":
    input_file = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT
    get_min_token_functions(input_file)