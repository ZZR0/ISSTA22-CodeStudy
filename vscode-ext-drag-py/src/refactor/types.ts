/**
 * Shared types for the refactor pipeline.
 *
 * Kept free of VSCode imports so the lower layers can be unit-tested without
 * pulling in the editor host.
 */

import * as vscode from "vscode";
import Parser from "tree-sitter";

export interface ExtractMethodOptions {
  /** New method name. Defaults to `extracted` if omitted. */
  methodName?: string;
  /** Override the inferred return-type hint. */
  returnTypeHint?: string;
  /** Force the new method to be `@staticmethod` regardless of inference. */
  forceStatic?: boolean;
}

export interface ExtractedSignature {
  methodName: string;
  parameters: string[];
  returnVariables: string[];
  isStatic: boolean;
  isAsync: boolean;
  inClassScope: boolean;
  isInstanceMethod: boolean;
  typeHints: Record<string, string>;
  returnTypeHint?: string;
}

export interface ExtractMethodResult {
  /** Workspace edit ready to be applied. */
  edit: vscode.WorkspaceEdit;
  /** Source of the new method, in case the caller wants to preview it. */
  extractedMethodCode: string;
  /** Replacement string spliced into the original selection range. */
  callSite: string;
  /** Computed signature. */
  signature: ExtractedSignature;
  /** Range where the new method was inserted. */
  insertedRange: vscode.Range;
  /** Range that was replaced with the call site. */
  replacedRange: vscode.Range;
}

export interface RefactorabilityIssue {
  severity: "error" | "warning";
  code: string;
  message: string;
  range?: vscode.Range;
}

export interface RefactorabilityReport {
  ok: boolean;
  issues: RefactorabilityIssue[];
}

export interface FunctionContext {
  fnNode: Parser.SyntaxNode;
  classNode: Parser.SyntaxNode | null;
  fnNameRange: vscode.Range;
  fnRange: vscode.Range;
  /** Sibling-method names already in the enclosing scope. */
  siblingNames: Set<string>;
}
