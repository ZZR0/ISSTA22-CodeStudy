/**
 * Identifier sanitisation and small naming helpers.
 *
 * Mirrors `sanitize_identifier`, `normalize_type` and
 * `choose_unique_method_name` from `extract_method_refactor_python.py`.
 */

const PYTHON_KEYWORDS: ReadonlySet<string> = new Set([
  "False", "None", "True", "and", "as", "assert", "async", "await",
  "break", "class", "continue", "def", "del", "elif", "else", "except",
  "finally", "for", "from", "global", "if", "import", "in", "is",
  "lambda", "nonlocal", "not", "or", "pass", "raise", "return", "try",
  "while", "with", "yield", "match", "case",
]);

/** True if `name` is reserved by the Python language. */
export function isPythonKeyword(name: string): boolean {
  return PYTHON_KEYWORDS.has(name);
}

/**
 * Coerce an arbitrary string into a syntactically-legal Python identifier.
 *
 *  - Non-`[A-Za-z0-9_]` runs become `_`.
 *  - A leading digit is prefixed with `extracted_` so the result remains a
 *    valid identifier.
 *  - Reserved keywords get a trailing `_` to avoid collisions.
 */
export function sanitizeIdentifier(name: string): string {
  let cleaned = name.replace(/[^A-Za-z0-9_]/g, "_");
  if (!cleaned) {
    cleaned = "extracted";
  }
  if (/^[0-9]/.test(cleaned)) {
    cleaned = `extracted_${cleaned}`;
  }
  if (isPythonKeyword(cleaned)) {
    cleaned = `${cleaned}_`;
  }
  return cleaned;
}

/** Default method name used by the extractor when the user has none in mind. */
export function defaultExtractedMethodName(_classid?: string): string {
  return "extracted";
}

/** Coerce an unspecified type to `"Any"` so callers can join annotations safely. */
export function normalizeType(typeName: string | undefined | null): string {
  const t = (typeName ?? "Any").trim();
  return t.length > 0 ? t : "Any";
}

/**
 * Build a fresh method name not already used inside `existingNames`. The base
 * is sanitised first and a numeric suffix is appended on collision.
 */
export function uniqueMethodName(base: string, existingNames: Iterable<string>): string {
  const taken = new Set(existingNames);
  const root = sanitizeIdentifier(base);
  if (!taken.has(root)) {
    return root;
  }
  let i = 2;
  while (taken.has(`${root}_${i}`)) {
    i++;
  }
  return `${root}_${i}`;
}
