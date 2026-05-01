import json
import re
import os
import sys
import html

# Default input file
DEFAULT_INPUT = "data/nicad_camel_clone_func.jsonl"
OUTPUT_HTML = "check_comments_clone_func.html"

# ---- Regex Pattern for Java Comments ----
# 1. ("..."): Double-quoted strings (handles escapes)
# 2. ('...'): Single-quoted chars
# 3. (/\*.*?\*/): Block comments
# 4. (//[^\r\n]*): Line comments
JAVA_PATTERN = re.compile(
    r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
    re.DOTALL | re.MULTILINE
)

def remove_java_comments(text):
    """
    Removes Java block comments (/* ... */) and line comments (// ...).
    Preserves comment markers inside string literals.
    """
    def replacer(match):
        s = match.group(0)
        # If it starts with '/', it is a comment -> Replace with space
        if s.startswith('/'):
            return " "
        # Otherwise, it is a string literal -> Keep as is
        else:
            return s

    return re.sub(JAVA_PATTERN, replacer, text)

def extract_comments_only(text):
    """
    Extracts purely the comment parts for display purposes.
    Filters out string literals that were matched by the regex.
    """
    matches = JAVA_PATTERN.findall(text)
    # Filter: Only keep matches that start with '/' (actual comments)
    return [m for m in matches if m.startswith('/')]

def generate_html_report(samples, output_file=OUTPUT_HTML):
    """
    Generates an HTML report comparing original code vs. stripped code.
    Adapted for Clone Function format (Class ID, Qualified Name).
    """
    html_content = [
        "<html><head><style>",
        "body { font-family: sans-serif; margin: 20px; }",
        "h2 { color: #333; }",
        "table { border-collapse: collapse; width: 100%; table-layout: fixed; }",
        "th, td { border: 1px solid #ccc; padding: 10px; vertical-align: top; word-wrap: break-word; }",
        "th { background-color: #f4f4f4; }",
        "pre { white-space: pre-wrap; margin: 0; font-size: 0.85em; font-family: Consolas, monospace; }",
        ".col-meta { width: 150px; font-size: 0.85em; color: #555; }",
        ".col-comments { width: 15%; color: #d9534f; font-size: 0.8em; }",
        ".col-code { width: 35%; background-color: #fff0f0; }", # Reddish for original
        ".col-stripped { width: 35%; background-color: #f0fff0; }", # Greenish for result
        "</style></head><body>",
        "<h2>Comment Check & Stripping Verification (Clone Functions)</h2>",
        "<table>",
        "<tr>",
        "<th class='col-meta'>Metadata</th>",
        "<th class='col-comments'>Detected Comments</th>",
        "<th class='col-code'>Original Code</th>",
        "<th class='col-stripped'>Stripped Code</th>",
        "</tr>"
    ]

    for class_id, func_name, found_comments, original_code, stripped_code in samples:
        escaped_original = html.escape(original_code)
        escaped_stripped = html.escape(stripped_code)
        escaped_name = html.escape(func_name)
        
        # Summary of comments found
        comments_html = "<br><hr><br>".join([f"<code>{html.escape(c)}</code>" for c in found_comments[:5]])
        if len(found_comments) > 5:
            comments_html += f"<br>... ({len(found_comments)-5} more)"

        meta_info = f"<b>ClassID:</b> {class_id}<br><br><b>Func:</b><br>{escaped_name}"

        html_content.append("<tr>")
        html_content.append(f"<td class='col-meta'>{meta_info}</td>")
        html_content.append(f"<td>{comments_html}</td>")
        html_content.append(f"<td><pre>{escaped_original}</pre></td>")
        html_content.append(f"<td><pre>{escaped_stripped}</pre></td>")
        html_content.append("</tr>")

    html_content.append("</table></body></html>")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(html_content))
    print(f"\nHTML report generated: {output_file}")

def analyze_comments(input_file):
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    samples = [] # Stores (class_id, qualified_name, comments_list, original, stripped)
    total_funcs = 0
    comment_func_count = 0
    
    print(f"--- Analyzing comments in: {input_file} ---")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line: continue
                
                try:
                    data = json.loads(line)
                    class_id = data.get("classid", "Unknown")
                    sources = data.get("sources", [])
                    
                    for src in sources:
                        code = src.get("code", "")
                        # Try to get qualified name, fallback to PCID or File
                        func_name = src.get("qualified_name", src.get("pcid", "Unknown"))
                        
                        total_funcs += 1
                        
                        # 1. Extract comments
                        comments_found = extract_comments_only(code)
                        
                        if comments_found:
                            comment_func_count += 1
                            
                            # 2. Collect top 10 samples
                            if len(samples) < 10:
                                stripped_code = remove_java_comments(code)
                                samples.append((class_id, func_name, comments_found, code, stripped_code))
                        
                except json.JSONDecodeError:
                    print(f"JSON Error at line {line_no}")
                    continue

        # Print Statistics
        print("\n=== Analysis Result ===")
        print(f"Total Functions Scanned: {total_funcs}")
        print(f"Functions with Comments: {comment_func_count}")
        if total_funcs > 0:
            print(f"Ratio: {(comment_func_count / total_funcs) * 100:.2f}%")

        # Generate HTML
        if samples:
            generate_html_report(samples)
        else:
            print("No comments found in the dataset.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    target_file = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT
    analyze_comments(target_file)