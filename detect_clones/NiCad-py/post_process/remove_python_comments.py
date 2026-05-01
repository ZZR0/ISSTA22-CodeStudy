import json
import ast
import argparse
import sys
import tokenize
import io

# -----------------------------------------------------------------------------
# Method 1: AST (Abstract Syntax Tree)
# Pros: Guarantees syntactically valid output.
# Cons: Completely reformats the code (removes original whitespace/newlines).
# -----------------------------------------------------------------------------
def remove_python_comments_ast(source_code):
    try:
        parsed = ast.parse(source_code)
    except SyntaxError:
        return None

    for node in ast.walk(parsed):
        # Docstrings are found in Modules, Functions, AsyncFunctions, and Classes
        if not isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef, ast.ClassDef, ast.Module)):
            continue

        if not node.body:
            continue

        # A docstring is a lone string expression as the first statement in the body
        if isinstance(node.body[0], ast.Expr):
            value = node.body[0].value
            is_docstring = False
            
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                is_docstring = True
            elif hasattr(ast, 'Str') and isinstance(value, ast.Str):
                is_docstring = True

            if is_docstring:
                node.body.pop(0)

    if hasattr(ast, 'unparse'):
        return ast.unparse(parsed)
    return None

# -----------------------------------------------------------------------------
# Method 2: Tokenizer
# Pros: Preserves original formatting (newlines, indentation).
# Cons: Can fail on partial code or complex multi-line strings.
# -----------------------------------------------------------------------------
def remove_python_comments_tokenizer(code: str) -> str:
    if not code:
        return ""

    src = code if code.endswith("\n") else code + "\n"

    try:
        toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return None

    kept = []
    at_module_start = True     
    at_suite_start = False     

    i = 0
    n = len(toks)
    while i < n:
        tok = toks[i]

        # 1) Remove comments
        if tok.type == tokenize.COMMENT:
            i += 1
            continue

        # Track suite starts
        if tok.type == tokenize.INDENT:
            at_suite_start = True
            kept.append(tok)
            i += 1
            continue

        if tok.type == tokenize.DEDENT:
            at_suite_start = False
            kept.append(tok)
            i += 1
            continue

        if tok.type == tokenize.NL:
            kept.append(tok)
            i += 1
            continue

        # 2) Remove docstring if STRING appears as first statement
        if tok.type == tokenize.STRING and (at_module_start or at_suite_start):
            # Check previous token to ensure it's not an assignment (e.g. x = "doc")
            # This is a basic heuristic; AST is safer, but this works for most standard docstrings
            i += 1
            if i < n and toks[i].type == tokenize.NEWLINE:
                i += 1
            at_module_start = False
            at_suite_start = False
            continue

        if tok.type not in (tokenize.ENCODING, tokenize.NL, tokenize.NEWLINE):
            at_module_start = False
            at_suite_start = False

        kept.append(tok)
        i += 1

    out = tokenize.untokenize(kept)

    # Post-clean:
    lines = out.splitlines()
    cleaned_lines = []
    for ln in lines:
        if ln.strip() in ("\\", "\\\\"):
            continue
        ln = ln.rstrip()
        if ln.strip() == "":
            continue
        cleaned_lines.append(ln)

    return "\n".join(cleaned_lines)

# -----------------------------------------------------------------------------
# Selection Logic
# -----------------------------------------------------------------------------
def clean_code_smart(original_code):
    # Try both methods
    res_ast = remove_python_comments_ast(original_code)
    res_tok = remove_python_comments_tokenizer(original_code)

    # Case 1: Both failed (e.g. invalid syntax in original)
    if res_ast is None and res_tok is None:
        return original_code

    # Case 2: Only one succeeded
    if res_ast is not None and res_tok is None:
        return res_ast
    if res_tok is not None and res_ast is None:
        # Verify tokenizer output is valid python before returning
        try:
            ast.parse(res_tok)
            return res_tok
        except SyntaxError:
            return original_code

    # Case 3: Both succeeded
    # We prefer Tokenizer because it keeps formatting, BUT we must ensure
    # it didn't break the code (which tokenize sometimes does with complex multiline logic).
    try:
        # Check validity of tokenizer output
        ast.parse(res_tok)
        return res_tok # It's valid and formatted well, choose this.
    except SyntaxError:
        # Tokenizer produced broken code, fallback to AST version
        return res_ast

# -----------------------------------------------------------------------------
# Main Processing Loop
# -----------------------------------------------------------------------------
def process_jsonl(input_file, output_file):
    print(f"Processing: {input_file} -> {output_file}")
    count = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line_number, line in enumerate(infile):
            if not line.strip():
                continue
            
            try:
                data = json.loads(line)
                
                if 'sources' in data and isinstance(data['sources'], list):
                    for source_item in data['sources']:
                        if 'code' in source_item:
                            original = source_item['code']
                            # Use the smart selector
                            cleaned = clean_code_smart(original)
                            source_item['code'] = cleaned
                
                outfile.write(json.dumps(data) + '\n')
                count += 1
                
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON on line {line_number + 1}")

    print(f"Success! Processed {count} lines.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove Python comments/docstrings using hybrid AST/Tokenizer strategy.")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file")
    
    args = parser.parse_args()
    process_jsonl(args.input, args.output)