import re

def remove_java_comments(text):
    """
    Removes block comments (/* ... */) and line comments (// ...) from Java code.
    Comment symbols inside string literals are preserved.
    """
    def replacer(match):
        s = match.group(0)
        if s.startswith('/'):
            return " "  # Replace comments with a whitespace
        else:
            return s  # Keep strings unchanged

    # Pattern description:
    # 1. ("..."): Double-quoted string (including escape sequences)
    # 2. ('...'): Single-quoted character literal
    # 3. (/\*.*?\*/): Block comment (can span multiple lines)
    # 4. (//[^\r\n]*): Line comment
    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE
    )
    
    # Apply the regex
    return re.sub(pattern, replacer, text)

# --- Test ---
code_with_comments = """
public void test() {
    // This is a line comment
    int a = 10; /* This is a block comment */
    String url = "http://example.com"; // This should NOT be removed
}
"""

print(remove_java_comments(code_with_comments))