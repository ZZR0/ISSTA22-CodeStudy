import json
import os
import sys
import html
import re

# Try to import transformers for accurate tokenization
try:
    from transformers import AutoTokenizer
    # Use the same model as in your run.py
    TOKENIZER_MODEL = "microsoft/CodeGPT-small-java-adaptedGPT2"
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)
    print(f"Using tokenizer: {TOKENIZER_MODEL}")
except ImportError:
    tokenizer = None
    print("Transformers not found. Using simple whitespace/symbol tokenizer.")

DEFAULT_INPUT = "data/nicad_camel_clone_func.jsonl"

def simple_tokenize(text):
    """Fallback tokenizer if transformers is not installed."""
    return re.findall(r'\w+|[^\w\s]', text)

def count_tokens(text):
    if tokenizer:
        return len(tokenizer.tokenize(text))
    else:
        return len(simple_tokenize(text))

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
                        
                        tokens = count_tokens(code)
                        functions.append((tokens, name, code))
                        
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
    html_output.append("<h3>Top 10 Functions with Minimum Tokens</h3>")
    html_output.append("<table border='1' style='border-collapse: collapse; width: 100%; text-align: left; font-family: Arial, sans-serif;'>")
    html_output.append("<tr style='background-color: #f2f2f2;'>")
    html_output.append("<th style='padding: 8px;'>Rank</th>")
    html_output.append("<th style='padding: 8px;'>Tokens</th>")
    html_output.append("<th style='padding: 8px;'>Function Name / File</th>")
    html_output.append("<th style='padding: 8px;'>Code Snippet</th>")
    html_output.append("</tr>")
    
    for i, (tokens, name, code) in enumerate(top_functions, 1):
        escaped_code = html.escape(code)
        escaped_name = html.escape(name)
        
        html_output.append("<tr>")
        html_output.append(f"<td style='padding: 8px;'>{i}</td>")
        html_output.append(f"<td style='padding: 8px;'>{tokens}</td>")
        html_output.append(f"<td style='padding: 8px; font-size: 0.9em;'>{escaped_name}</td>")
        html_output.append(f"<td style='padding: 8px;'><pre style='background-color: #f8f8f8; padding: 5px; margin: 0; white-space: pre-wrap; font-size: 0.85em;'>{escaped_code}</pre></td>")
        html_output.append("</tr>")
    
    html_output.append("</table>")
    html_output.append("</body></html>")
    
    # Save to file
    output_filename = "min_tokens.html"
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write("\n".join(html_output))
    
    print(f"\nSuccessfully generated HTML report: {output_filename}")
    print("You can open this file in your browser to view the functions.")

if __name__ == "__main__":
    input_file = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT
    get_min_token_functions(input_file)