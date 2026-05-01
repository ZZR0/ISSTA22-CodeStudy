import io
import tokenize
from remove_python_comments import remove_python_comments

def reproduce1():
    # The specific failing case reported
    failing_code = """
    def __init__(
        self,
        **kwargs
    ):
        \"\"\"
        :keyword description: The asset description text.
        :paramtype description: str
        :keyword properties: The asset property dictionary.
        :paramtype properties: dict[str, str]
        \"\"\"
        super(AzureFileDatastore, self).__init__(**kwargs)
        self.datastore_type = 'AzureFile'  # type: str
    """

    print("--- [Original Code Snippet] ---")
    print(failing_code)
    print("-" * 30)

    # 1. Check if tokenize raises an error on this snippet
    print("--- [Tokenization Check] ---")
    try:
        src = failing_code if failing_code.endswith("\n") else failing_code + "\n"
        tokens = list(tokenize.generate_tokens(io.StringIO(src).readline))
        print(f"SUCCESS: Tokenization complete ({len(tokens)} tokens generated).")
        # Optional: Print first few tokens to debug indentation
        # for t in tokens[:5]: print(t)
    except (tokenize.TokenError, IndentationError, SyntaxError) as e:
        print(f"FAILURE: Tokenization raised exception: {e}")
        print(">> This triggers the fallback logic which SKIPS docstring removal.")

    # 2. Run the actual cleaning function
    cleaned = remove_python_comments(failing_code)

    print("\n--- [Result Code] ---")
    print(cleaned)
    print("-" * 30)

    # 3. Validation
    if ':keyword description:' in cleaned:
        print("❌ FAIL: Docstring was NOT removed.")
    else:
        print("✅ SUCCESS: Docstring was removed.")

def reproduce2(code):
    # The specific failing case reported
    failing_code = code

    print("--- [Original Code Snippet] ---")
    print(failing_code)
    print("-" * 30)

    # 1. Check if tokenize raises an error on this snippet
    print("--- [Tokenization Check] ---")
    try:
        src = failing_code if failing_code.endswith("\n") else failing_code + "\n"
        tokens = list(tokenize.generate_tokens(io.StringIO(src).readline))
        print(f"SUCCESS: Tokenization complete ({len(tokens)} tokens generated).")
        # Optional: Print first few tokens to debug indentation
        # for t in tokens[:5]: print(t)
    except (tokenize.TokenError, IndentationError, SyntaxError) as e:
        print(f"FAILURE: Tokenization raised exception: {e}")
        print(">> This triggers the fallback logic which SKIPS docstring removal.")

    # 2. Run the actual cleaning function
    cleaned = remove_python_comments(failing_code)

    print("\n--- [Result Code] ---")
    print(cleaned)
    print("-" * 30)

    # 3. Validation
    if ':keyword description:' in cleaned:
        print("❌ FAIL: Docstring was NOT removed.")
    else:
        print("✅ SUCCESS: Docstring was removed.")


if __name__ == "__main__":
    reproduce1()
    code = "def _send_matcher_request(matcher: str, headers: Dict, parameters: Optional[Dict] = None) -> None:\n    \"\"\"Sends a POST request to the test proxy endpoint to register the specified matcher.\n\n    If live tests are being run, no request will be sent.\n\n    :param str matcher: The name of the matcher to set.\n    :param dict headers: Any matcher headers, as a dictionary.\n    :param parameters: Any matcher constructor parameters, as a dictionary. Defaults to None.\n    :type parameters: Optional[dict]\n    \"\"\"\n\n    if is_live():\n        return\n\n    headers_to_send = {\"x-abstraction-identifier\": matcher}\n    for key in headers:\n        if headers[key] is not None:\n            headers_to_send[key] = headers[key]\n\n    http_client = get_http_client()\n    http_client.request(\n        method=\"POST\",\n        url=f\"{PROXY_URL}/Admin/SetMatcher\",\n        headers=headers_to_send,\n        body=json.dumps(parameters).encode(\"utf-8\"),\n    )"
    print(code)
    print("*"*30)
    reproduce2(code)
