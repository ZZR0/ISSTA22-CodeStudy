/**
 * Indentation helpers shared by the refactor pipeline.
 *
 * Mirrors the small Python helpers in `extract_method_refactor_python.py`
 * (`indent_block`, `expandtabs`, leading-whitespace inspection) but keeps a
 * pure-TypeScript surface so the rest of the code base does not need to know
 * about Python conventions.
 */

const LEADING_WS = /^[ \t]*/;

/**
 * Replace `\t` with the equivalent number of spaces (default 4) so that
 * indent comparisons are byte-stable across files using mixed whitespace.
 */
export function expandTabs(text: string, tabSize: number = 4): string {
  if (text.indexOf("\t") < 0) {
    return text;
  }
  const filler = " ".repeat(tabSize);
  return text.replace(/\t/g, filler);
}

/** Return the whitespace prefix of a single line (after expanding tabs). */
export function getLeadingIndent(line: string, tabSize: number = 4): string {
  const expanded = expandTabs(line, tabSize);
  const match = LEADING_WS.exec(expanded);
  return match ? match[0] : "";
}

/** First non-blank line's leading indent, or "" when all lines are blank. */
export function detectFirstIndent(lines: string[], tabSize: number = 4): string {
  for (const line of lines) {
    if (line.trim().length > 0) {
      return getLeadingIndent(line, tabSize);
    }
  }
  return "";
}

/**
 * Prefix every non-blank line with `indent`. Blank lines are emitted as-is.
 *
 * Equivalent to the Python `indent_block(text, indent)` helper.
 */
export function indentBlock(text: string, indent: string): string {
  if (!indent) {
    return text;
  }
  return text
    .split("\n")
    .map((line) => (line.trim().length > 0 ? indent + line : line))
    .join("\n");
}

/**
 * Strip the most-common leading indent from a block of code (similar to
 * Python's `textwrap.dedent`). Blank lines never contribute to the prefix.
 */
export function dedentBlock(text: string, tabSize: number = 4): string {
  const lines = expandTabs(text, tabSize).split("\n");
  let common: string | null = null;

  for (const line of lines) {
    if (line.trim().length === 0) {
      continue;
    }
    const indent = getLeadingIndent(line, tabSize);
    if (common === null) {
      common = indent;
      continue;
    }
    let i = 0;
    while (i < common.length && i < indent.length && common[i] === indent[i]) {
      i++;
    }
    common = common.slice(0, i);
    if (!common) {
      break;
    }
  }

  if (!common) {
    return lines.join("\n");
  }

  return lines.map((line) => (line.startsWith(common!) ? line.slice(common!.length) : line)).join("\n");
}
