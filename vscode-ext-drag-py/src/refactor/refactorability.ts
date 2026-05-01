/**
 * Decide whether a selection is safe to extract.
 *
 * The checks here are the TypeScript counterpart of the runtime guards
 * scattered through `extract_method_refactor_python.py` (no clones, no
 * runaway syntax errors, no `return` mid-block, etc.). They run before any
 * edits are computed so the user gets a clear diagnostic instead of a
 * mangled file when extraction is impossible.
 */

import * as vscode from "vscode";
import Parser from "tree-sitter";
import { findUnsupportedConstructs, UnsupportedFinding } from "../languages/python/pythonUnsupportedNodes";
import { hasSyntaxErrors } from "../parser/syntaxValidation";
import { isAsyncFunction } from "../languages/python/pythonAnalyzer";
import { RefactorabilityIssue, RefactorabilityReport } from "./types";

interface RefactorabilityInput {
  source: string;
  document: vscode.TextDocument;
  fnNode: Parser.SyntaxNode;
  selectionRoot: Parser.SyntaxNode;
  selectionRange: vscode.Range;
  /** Selection's start byte. Used to scope the scan to the user's actual region. */
  selectionStartByte: number;
  /** Selection's end byte (exclusive). */
  selectionEndByte: number;
}

const KIND_TO_MESSAGE: Record<UnsupportedFinding["kind"], string> = {
  "return": "The selection contains a `return` statement; extracting it would change the host function's control flow.",
  "yield": "The selection contains a `yield` expression; generator semantics cannot be preserved by extracting a regular method.",
  "await-in-sync": "The selection contains `await` but the host function is not `async`.",
  "break-outside-loop": "The selection contains a `break` whose target loop is outside the selection.",
  "continue-outside-loop": "The selection contains a `continue` whose target loop is outside the selection.",
  "nonlocal": "The selection contains a `nonlocal` declaration; the rebinding would not survive extraction.",
  "global": "The selection contains a `global` declaration; the rebinding would not survive extraction.",
  "syntax-error": "The selection contains a syntax error; please fix the file before extracting.",
};

export function evaluateRefactorability(input: RefactorabilityInput): RefactorabilityReport {
  const issues: RefactorabilityIssue[] = [];

  if (hasSyntaxErrors(input.fnNode)) {
    issues.push({
      severity: "error",
      code: "syntax-error",
      message: "The enclosing function contains syntax errors; resolve them before refactoring.",
      range: input.selectionRange,
    });
    return { ok: false, issues };
  }

  if (input.selectionRange.isEmpty) {
    issues.push({
      severity: "error",
      code: "empty-selection",
      message: "Select one or more whole statements to extract.",
      range: input.selectionRange,
    });
    return { ok: false, issues };
  }

  const findings = findUnsupportedConstructs(input.selectionRoot, {
    enclosingFunctionIsAsync: isAsyncFunction(input.source, input.fnNode),
    containsLoop: false,
    selectionStartByte: input.selectionStartByte,
    selectionEndByte: input.selectionEndByte,
  });

  for (const finding of findings) {
    const range = new vscode.Range(
      new vscode.Position(finding.startLine, 0),
      new vscode.Position(finding.endLine, Number.MAX_SAFE_INTEGER)
    );
    issues.push({
      severity: "error",
      code: finding.kind,
      message: KIND_TO_MESSAGE[finding.kind],
      range,
    });
  }

  // Selection should not start mid-expression: the smallest enclosing node
  // must begin at or before the selection start AND end at or after.
  if (
    input.selectionRoot.startIndex > input.document.offsetAt(input.selectionRange.start) ||
    input.selectionRoot.endIndex < input.document.offsetAt(input.selectionRange.end)
  ) {
    issues.push({
      severity: "warning",
      code: "partial-statement",
      message: "Selection does not align with statement boundaries; the extractor will widen it to the nearest full statements.",
      range: input.selectionRange,
    });
  }

  return { ok: !issues.some((i) => i.severity === "error"), issues };
}
