import json
import re
import os
import sys
import html

# ---- Regex Pattern from your reference ----
# 1. ("..."): Double-quoted strings (handles escapes)
# 2. ('...'): Single-quoted chars
# 3. (/\*.*?\*/): Block comments
# 4. (//[^\r\n]*): Line comments
# Combined pattern to distinguish comments from string literals
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

def generate_html_report(samples, output_file="comment_stripping_check.html"):
    """
    Generates an HTML report comparing original code vs. stripped code.
    """
    html_content = [
        "<html><head><style>",
        "body { font-family: sans-serif; margin: 20px; }",
        "h2 { color: #333; }",
        "table { border-collapse: collapse; width: 100%; table-layout: fixed; }",
        "th, td { border: 1px solid #ccc; padding: 10px; vertical-align: top; word-wrap: break-word; }",
        "th { background-color: #f4f4f4; }",
        "pre { white-space: pre-wrap; margin: 0; font-size: 0.85em; font-family: Consolas, monospace; }",
        ".col-id { width: 80px; }",
        ".col-comments { width: 15%; color: #d9534f; font-size: 0.8em; }",
        ".col-code { width: 40%; background-color: #fff0f0; }", # Reddish tint for original
        ".col-stripped { width: 40%; background-color: #f0fff0; }", # Greenish tint for result
        "</style></head><body>",
        "<h2>Comment Stripping Verification (Top 10 Samples)</h2>",
        "<table>",
        "<tr>",
        "<th class='col-id'>ID</th>",
        "<th class='col-comments'>Detected Comments</th>",
        "<th class='col-code'>Original Code (Before)</th>",
        "<th class='col-stripped'>Processed Code (After)</th>",
        "</tr>"
    ]

    for idx, found_comments, original_code, stripped_code in samples:
        escaped_original = html.escape(original_code)
        escaped_stripped = html.escape(stripped_code)
        
        # Summary of comments found
        comments_html = "<br><hr><br>".join([f"<code>{html.escape(c)}</code>" for c in found_comments[:5]])
        if len(found_comments) > 5:
            comments_html += f"<br>... ({len(found_comments)-5} more)"

        html_content.append("<tr>")
        html_content.append(f"<td>{idx}</td>")
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

    samples = [] # Stores (idx, comments_list, original, stripped)
    total_count = 0
    comment_func_count = 0
    
    print(f"--- Analyzing comments in: {input_file} ---")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                
                try:
                    data = json.loads(line)
                    code = data.get("func", "")
                    idx = data.get("idx", "Unknown")
                    
                    total_count += 1
                    
                    # 1. Extract comments using the robust regex
                    comments_found = extract_comments_only(code)
                    
                    if comments_found:
                        comment_func_count += 1
                        
                        # 2. Collect top 10 samples for visualization
                        if len(samples) < 10:
                            # Apply the stripping logic
                            stripped_code = remove_java_comments(code)
                            samples.append((idx, comments_found, code, stripped_code))
                        
                except json.JSONDecodeError:
                    continue

        # Print Statistics
        print("\n=== Analysis Result ===")
        print(f"Total Functions Scanned: {total_count}")
        print(f"Functions with Comments: {comment_func_count}")
        if total_count > 0:
            print(f"Ratio: {(comment_func_count / total_count) * 100:.2f}%")

        # Generate HTML
        if samples:
            generate_html_report(samples)
        else:
            print("No comments found in the dataset.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    target_file = sys.argv[1] if len(sys.argv) > 1 else "data/data.jsonl"
    analyze_comments(target_file)