# Your original version issue:
import re

code = "int x = 123; double y = 3.14e-5;"

# Original pattern (with capturing group)
bad_pattern = re.compile(r'\b[0-9]+\.?[0-9]*([eE][-+]?[0-9]+)?\b')
print("With capturing group:", bad_pattern.findall(code))
# Output: ['', 'e-5']  # Returns the captured group content!

# Fixed pattern (non-capturing group)
good_pattern = re.compile(r'\b[0-9]+\.?[0-9]*(?:[eE][-+]?[0-9]+)?\b')
print("With non-capturing group:", good_pattern.findall(code))
# Output: ['123', '3.14e-5']  # Returns the full match!




java_code = """
public SecureRandomParameters getObject() throws Exception {
    if (this.isSingleton()) {
        if (instance == null) {
            instance = createInstance();
        }
        return instance;
    } else {
        return createInstance();
    }
}
"""

tokens = java_tokenize_standard(java_code)
print(f"Token count: {len(tokens)}")
print(f"Tokens: {tokens}")