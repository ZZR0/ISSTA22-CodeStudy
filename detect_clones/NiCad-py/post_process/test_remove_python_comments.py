#!/usr/bin/env python3
import argparse
from remove_python_comments import normalize_python_code


def _print_block(title: str, text: str):
    print(title)
    print(text.rstrip())
    print()


def _run_case(name: str, code: str, *, must_contain=(), must_not_contain=(), verbose=True):
    after = normalize_python_code(code)

    if verbose:
        print(f"\n[CASE] {name}")
        _print_block("--- BEFORE ---", code)
        _print_block("--- AFTER (docstrings + comments removed) ---", after)

    for s in must_contain:
        assert s in after, f"[{name}] expected to contain: {s!r}\n---after---\n{after}"
    for s in must_not_contain:
        assert s not in after, f"[{name}] expected NOT to contain: {s!r}\n---after---\n{after}"

    print(f"[PASS] {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    verbose = not args.quiet

    _run_case(
        "remove_full_and_inline_comments",
        code=(
            "# comment 1\n"
            "x = 1  # inline comment\n"
            "y = 2\n"
        ),
        must_contain=("x = 1", "y = 2"),
        must_not_contain=("comment 1", "inline comment"),
        verbose=verbose,
    )

    _run_case(
        "url_fragment_in_string_kept",
        code=(
            "url = \"https://example.com/path#section-2\"  # trailing\n"
            "print(url)\n"
        ),
        must_contain=("https://example.com/path#section-2", "print(url)"),
        must_not_contain=("trailing",),
        verbose=verbose,
    )

    _run_case(
        "hash_in_strings_kept",
        code=(
            "a = '#not_a_comment'\n"
            "b = \"#still_not_a_comment\"  # remove\n"
            "print(a, b)\n"
        ),
        must_contain=("#not_a_comment", "#still_not_a_comment", "print(a, b)"),
        must_not_contain=("# remove",),
        verbose=verbose,
    )

    _run_case(
        "function_docstring_removed",
        code=(
            "def f():\n"
            "    \"\"\"Docstring with # is fine.\n"
            "    It should be removed.\n"
            "    \"\"\"\n"
            "    return 1  # remove\n"
        ),
        must_contain=("def f():", "return 1"),
        must_not_contain=("Docstring with # is fine.", "It should be removed.", "# remove"),
        verbose=verbose,
    )

    _run_case(
        "class_docstring_removed",
        code=(
            "class C:\n"
            "    \"\"\"Class docstring\"\"\"\n"
            "    def m(self):\n"
            "        return 1  # remove\n"
        ),
        must_contain=("class C:", "def m", "return 1"),
        must_not_contain=("Class docstring", "# remove"),
        verbose=verbose,
    )

    _run_case(
        "module_docstring_removed",
        code=(
            "\"\"\"Module docstring with URL https://x.y/#frag\"\"\"\n"
            "x = 1  # remove\n"
        ),
        must_contain=("x = 1",),
        must_not_contain=("Module docstring", "# remove"),
        verbose=verbose,
    )

    _run_case(
        "triple_quoted_data_string_kept",
        code=(
            "def g():\n"
            "    x = \"\"\"not a docstring # keep\"\"\"\n"
            "    return x  # remove\n"
        ),
        must_contain=("not a docstring # keep", "return x"),
        must_not_contain=("# remove",),
        verbose=verbose,
    )

    _run_case(
        "shebang_and_encoding_removed",
        code=(
            "#!/usr/bin/env python3\n"
            "# -*- coding: utf-8 -*-\n"
            "print('hi')\n"
        ),
        must_contain=("print('hi')",),
        must_not_contain=("coding:", "/usr/bin/env"),
        verbose=verbose,
    )

    _run_case(
        "import_like_text_in_string_kept",
        code=(
            "s = \"import pytest  # not real\"  # remove this comment\n"
            "print(s)\n"
        ),
        must_contain=("import pytest  # not real", "print(s)"),
        must_not_contain=("remove this comment",),
        verbose=verbose,
    )

    print("\nAll normalize (docstrings+comments) test cases passed.")


if __name__ == "__main__":
    main()