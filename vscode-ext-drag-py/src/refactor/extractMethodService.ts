/**
 * Top-level orchestrator for the Python Extract Method refactor.
 *
 * Mirrors `ExtractMethodRefactorer` from `extract_method_refactor_python.py`,
 * but driven by an editor selection instead of a JSONL clone class. The flow:
 *
 *   1. Parse the document with tree-sitter.
 *   2. Locate the enclosing function/class for the selection.
 *   3. Run the refactorability checks.
 *   4. Analyse identifier read/write regions to derive inputs/outputs.
 *   5. Synthesise the new method's signature.
 *   6. Build a `vscode.WorkspaceEdit` that swaps the selection for a call and
 *      appends the new method.
 */

import * as vscode from "vscode";
import {
  PYTHON_FUNCTION_LIKE,
  PYTHON_SCOPE_TYPES,
} from "../languages/python/pythonNodeTypes";
import {
  analyzeSelection,
  findEnclosingFunctionAt,
  functionName,
} from "../languages/python/pythonAnalyzer";
import { detectLanguage, parseDocument } from "../parser/parserFactory";
import {
  enclosingClass,
  methodsDirectlyUnder,
} from "../languages/common/scopeAnalysis";
import { evaluateRefactorability } from "./refactorability";
import { inferSignature } from "./signatureInference";
import { buildExtractMethodEdit } from "./editBuilder";
import { logger } from "../utils/logging";
import {
  ExtractMethodOptions,
  ExtractMethodResult,
  RefactorabilityIssue,
} from "./types";

export class ExtractMethodError extends Error {
  constructor(message: string, public readonly issues: RefactorabilityIssue[] = []) {
    super(message);
    this.name = "ExtractMethodError";
  }
}

const SCOPE = "extractMethodService";

export interface ExtractMethodInput {
  document: vscode.TextDocument;
  selection: vscode.Range;
  options?: ExtractMethodOptions;
}

export async function extractMethod(input: ExtractMethodInput): Promise<ExtractMethodResult> {
  const { document, selection, options } = input;

  const language = detectLanguage(document);
  if (language !== "python") {
    throw new ExtractMethodError(
      `Extract Method (Python) is not supported for documents of type '${document.languageId}'.`
    );
  }

  const logEndLine = selection.end.character === 0 && selection.end.line > selection.start.line
    ? selection.end.line
    : selection.end.line + 1;
  logger.info(SCOPE, "starting extract method", {
    file: document.fileName,
    range: `${selection.start.line + 1}-${logEndLine}`,
  });

  const parsed = parseDocument(document);
  if (!parsed) {
    throw new ExtractMethodError("Unable to parse document with tree-sitter.");
  }

  // 1. Locate enclosing function. We use the byte offset of the selection
  //    start (after expanding to non-whitespace) so trailing whitespace at
  //    the very top of the selection does not push us out of the function.
  const startOffset = document.offsetAt(selection.start);
  const fnNode = findEnclosingFunctionAt(parsed.root, startOffset);
  if (!fnNode) {
    throw new ExtractMethodError(
      "Place the cursor inside a function or method before invoking Extract Method."
    );
  }

  // 2. Compute the byte range of the user's selection. Tree-sitter's
  //    `namedDescendantForIndex(a,b)` may climb above the enclosing function
  //    when the selection straddles sibling statements, so we keep the
  //    user's exact bytes for downstream filters.
  const endOffset = document.offsetAt(selection.end);
  const selStartByte = Math.min(startOffset, endOffset);
  const selEndByte = Math.max(startOffset, endOffset);

  let selectionRoot = parsed.root.namedDescendantForIndex(
    selStartByte,
    Math.max(selStartByte, selEndByte - 1)
  );

  // Clamp: the selection root must live inside the enclosing function. If
  // the user dragged past the function body (e.g. onto trailing whitespace
  // or a sibling def) the descendant query escapes upward — re-anchor on
  // the function so refactorability checks don't see the rest of the file.
  if (
    selectionRoot.startIndex < fnNode.startIndex ||
    selectionRoot.endIndex > fnNode.endIndex
  ) {
    selectionRoot = fnNode;
  }

  // 3. Refactorability gate.
  const report = evaluateRefactorability({
    source: parsed.source,
    document,
    fnNode,
    selectionRoot,
    selectionRange: selection,
    selectionStartByte: selStartByte,
    selectionEndByte: selEndByte,
  });
  const errorIssues = report.issues.filter((i) => i.severity === "error");
  if (!report.ok || errorIssues.length > 0) {
    throw new ExtractMethodError(
      errorIssues.map((i) => i.message).join("\n") || "Selection cannot be extracted.",
      report.issues
    );
  }

  // 4. Selection text + line span (1-indexed for the analyzer).
  const selectionText = document.getText(selection);
  const withinStartLine = selection.start.line + 1;
  const withinEndLine = selection.end.character === 0 && selection.end.line > selection.start.line
    ? selection.end.line
    : selection.end.line + 1;

  const analysis = analyzeSelection({
    source: parsed.source,
    fn: fnNode,
    withinStartLine,
    withinEndLine,
  });

  // 5. Existing method names (for collision avoidance).
  const classNode = enclosingClass(fnNode, PYTHON_SCOPE_TYPES);
  const siblings = classNode
    ? methodsDirectlyUnder(classNode, PYTHON_SCOPE_TYPES)
    : findTopLevelFunctions(parsed.root);
  const siblingNames = new Set(siblings.map((n) => functionName(parsed.source, n)).filter(Boolean));

  const signature = inferSignature({
    analysis,
    options,
    existingNames: siblingNames,
  });

  // 6. Assemble call-site arguments. By default we pass the same identifiers
  //    that the analyzer flagged as inputs — matching the behaviour of the
  //    Python `Rewriter._call_args_for_source`.
  const argumentExprs = signature.parameters.slice();
  const outputAssignments = signature.returnVariables.slice();

  const editOutput = buildExtractMethodEdit({
    document,
    selectionRange: selection,
    selectionText,
    fnNode,
    classNode,
    signature,
    argumentExprs,
    outputAssignments,
  });

  logger.info(SCOPE, "extract method complete", {
    inputs: signature.parameters,
    outputs: signature.returnVariables,
    isStatic: signature.isStatic,
    inClass: signature.inClassScope,
  });

  return {
    edit: editOutput.edit,
    extractedMethodCode: editOutput.extractedMethodCode,
    callSite: editOutput.callSite,
    signature,
    insertedRange: editOutput.insertedRange,
    replacedRange: editOutput.replacedRange,
  };
}

function findTopLevelFunctions(root: import("tree-sitter").SyntaxNode) {
  const result: import("tree-sitter").SyntaxNode[] = [];
  for (const child of root.children) {
    if (PYTHON_FUNCTION_LIKE.has(child.type)) {
      result.push(child);
    } else if (child.type === "decorated_definition") {
      const def = child.namedChildren.find((n) => PYTHON_FUNCTION_LIKE.has(n.type));
      if (def) {
        result.push(def);
      }
    }
  }
  return result;
}

/**
 * Probe whether Extract Method is available for a given selection without
 * actually computing any edits. Used by the CodeActionProvider so the
 * lightbulb only shows up when the user can take the action.
 */
export function canOfferExtractMethod(document: vscode.TextDocument, selection: vscode.Range): boolean {
  if (selection.isEmpty) {
    return false;
  }

  const language = detectLanguage(document);
  if (language !== "python") {
    return false;
  }

  const parsed = parseDocument(document);
  if (!parsed) {
    return false;
  }

  const startOffset = document.offsetAt(selection.start);
  const fn = findEnclosingFunctionAt(parsed.root, startOffset);
  if (!fn) {
    return false;
  }

  const endOffset = document.offsetAt(selection.end);
  const selStartByte = Math.min(startOffset, endOffset);
  const selEndByte = Math.max(startOffset, endOffset);

  let selectionRoot = parsed.root.namedDescendantForIndex(
    selStartByte,
    Math.max(selStartByte, selEndByte - 1)
  );

  if (selectionRoot.startIndex < fn.startIndex || selectionRoot.endIndex > fn.endIndex) {
    selectionRoot = fn;
  }

  const report = evaluateRefactorability({
    source: parsed.source,
    document,
    fnNode: fn,
    selectionRoot,
    selectionRange: selection,
    selectionStartByte: selStartByte,
    selectionEndByte: selEndByte,
  });

  return report.ok;
}
