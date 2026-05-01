/**
 * Syntax-validation helpers built on top of tree-sitter.
 *
 * Tree-sitter is error-tolerant: when the input has a syntax error it inserts
 * `ERROR` / `MISSING` nodes rather than failing outright. Refactoring against
 * an unparseable file is dangerous, so we surface those nodes through these
 * helpers and let the caller bail out with a friendly diagnostic.
 */

import Parser from "tree-sitter";
import { iterDescendants } from "./treeUtils";

export interface SyntaxIssue {
  kind: "error" | "missing";
  startLine: number;
  startColumn: number;
  endLine: number;
  endColumn: number;
  text: string;
}

/** True when tree-sitter emitted any ERROR/MISSING nodes inside `node`. */
export function hasSyntaxErrors(node: Parser.SyntaxNode): boolean {
  if (node.hasError) {
    return true;
  }
  for (const desc of iterDescendants(node)) {
    if (desc.isError || desc.isMissing) {
      return true;
    }
  }
  return false;
}

/** All ERROR/MISSING nodes inside `node`, deduped by span. */
export function collectSyntaxIssues(node: Parser.SyntaxNode): SyntaxIssue[] {
  const seen = new Set<string>();
  const issues: SyntaxIssue[] = [];
  for (const desc of iterDescendants(node)) {
    if (!desc.isError && !desc.isMissing) {
      continue;
    }
    const key = `${desc.startIndex}-${desc.endIndex}-${desc.type}`;
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    issues.push({
      kind: desc.isMissing ? "missing" : "error",
      startLine: desc.startPosition.row,
      startColumn: desc.startPosition.column,
      endLine: desc.endPosition.row,
      endColumn: desc.endPosition.column,
      text: desc.text,
    });
  }
  return issues;
}
