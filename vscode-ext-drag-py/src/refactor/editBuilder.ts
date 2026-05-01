/**
 * Convert refactor decisions into a `vscode.WorkspaceEdit`.
 *
 * This is the bridge between the language-agnostic refactor pipeline and
 * VSCode's edit infrastructure. Mirrors `Rewriter.rewrite` from
 * `extract_method_refactor_python.py`, but emits a sparse edit (call-site
 * replacement + method insertion) instead of producing a full rewritten file
 * blob.
 */

import * as vscode from "vscode";
import Parser from "tree-sitter";
import { dedentBlock, getLeadingIndent, indentBlock } from "../utils/indentation";
import {
  PythonMethodSignature,
  renderPythonCallSite,
  renderPythonMethod,
} from "../languages/python/pythonCodeGenerator";
import { ExtractedSignature } from "./types";

export interface BuildEditInput {
  document: vscode.TextDocument;
  selectionRange: vscode.Range;
  selectionText: string;
  /** Function node enclosing the selection. */
  fnNode: Parser.SyntaxNode;
  /** Class node enclosing the selection (may be null when at module level). */
  classNode: Parser.SyntaxNode | null;
  signature: ExtractedSignature;
  /** Identifier names provided by the call site, in `signature.parameters` order. */
  argumentExprs: string[];
  /** LHS of the call assignment (typically the same names as `signature.returnVariables`). */
  outputAssignments: string[];
}

export interface BuildEditOutput {
  edit: vscode.WorkspaceEdit;
  callSite: string;
  extractedMethodCode: string;
  insertedRange: vscode.Range;
  replacedRange: vscode.Range;
}

export function buildExtractMethodEdit(input: BuildEditInput): BuildEditOutput {
  const {
    document,
    selectionRange,
    fnNode,
    classNode,
    signature,
    argumentExprs,
    outputAssignments,
  } = input;

  // 1. Compute the indent of the selection's first line so the call site
  //    drops in cleanly.
  const firstLineText = document.lineAt(selectionRange.start.line).text;
  const baseIndent = getLeadingIndent(firstLineText);

  // 2. Render call site & extracted method as plain strings.
  const codegenSig: PythonMethodSignature = {
    name: signature.methodName,
    parameters: signature.parameters,
    typeHints: signature.typeHints,
    returnVariables: signature.returnVariables,
    isStatic: signature.isStatic,
    isAsync: signature.isAsync,
    inClassScope: signature.inClassScope,
    isInstanceMethod: signature.isInstanceMethod,
    returnTypeHint: signature.returnTypeHint,
  };

  const callSite = renderPythonCallSite({
    signature: codegenSig,
    argumentExprs,
    outputAssignments,
    indent: baseIndent,
  });

  const expandedReplaceRange = expandToFullLines(document, selectionRange);
  const selectedFullLines = document.getText(expandedReplaceRange);
  const dedentedBody = dedentBlock(stripTrailingBlankLines(selectedFullLines));
  const extractedMethodCode = renderPythonMethod(codegenSig, dedentedBody);

  // 3. Decide where to insert the new method.
  const insertionTarget = classNode ?? fnNode;
  const insertionStart = document.positionAt(insertionTarget.endIndex);
  const insertionIndent = computeInsertionIndent(document, insertionTarget, classNode);
  const indentedMethod = "\n\n" + indentBlock(extractedMethodCode, insertionIndent) + "\n";

  // 4. Build the WorkspaceEdit: replace selection with call-site, then append
  //    the method after the enclosing scope. VSCode applies edits as if they
  //    were sorted by descending offset, so the order of `replace` vs
  //    `insert` does not need any manual offset arithmetic.
  const edit = new vscode.WorkspaceEdit();
  edit.replace(document.uri, expandedReplaceRange, callSite);
  edit.insert(document.uri, insertionStart, indentedMethod);

  // 5. Compute reporting ranges using the post-edit positions.
  const replacedRange = expandedReplaceRange;
  const insertedStartOffset = document.offsetAt(insertionStart);
  const insertedEndOffset = insertedStartOffset + indentedMethod.length;
  const insertedRange = new vscode.Range(
    document.positionAt(insertedStartOffset),
    document.positionAt(insertedEndOffset)
  );

  return {
    edit,
    callSite,
    extractedMethodCode,
    insertedRange,
    replacedRange,
  };
}

function expandToFullLines(document: vscode.TextDocument, range: vscode.Range): vscode.Range {
  const startLine = range.start.line;
  const endLine = range.end.character === 0 && range.end.line > range.start.line
    ? range.end.line - 1
    : range.end.line;

  const startPos = new vscode.Position(startLine, 0);
  const lastLineLen = document.lineAt(endLine).range.end.character;
  // include the trailing newline so the call site's own `\n` is enough.
  const endPos = endLine + 1 < document.lineCount
    ? new vscode.Position(endLine + 1, 0)
    : new vscode.Position(endLine, lastLineLen);
  return new vscode.Range(startPos, endPos);
}

function computeInsertionIndent(
  document: vscode.TextDocument,
  target: Parser.SyntaxNode,
  classNode: Parser.SyntaxNode | null
): string {
  const startLine = target.startPosition.row;
  if (startLine >= document.lineCount) {
    return "";
  }
  const baseIndent = getLeadingIndent(document.lineAt(startLine).text);
  return classNode ? baseIndent + "    " : baseIndent;
}

function stripTrailingBlankLines(text: string): string {
  return text.replace(/[ \t]*\n+$/g, "\n");
}
