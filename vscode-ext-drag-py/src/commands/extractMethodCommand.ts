/**
 * VSCode entry points for the Python Extract Method refactor.
 *
 * Two surfaces:
 *
 *   1. A command (`pythonRefactor.extractMethod`) the user can invoke from
 *      the Command Palette or a keybinding.
 *   2. A `CodeActionProvider` that surfaces an "Extract method" lightbulb
 *      similar to VSCode's built-in refactor menu.
 */

import * as vscode from "vscode";
import {
  canOfferExtractMethod,
  extractMethod,
  ExtractMethodError,
} from "../refactor/extractMethodService";
import { logger } from "../utils/logging";

export const EXTRACT_METHOD_COMMAND_ID = "pythonRefactor.extractMethod";

const SCOPE = "extractMethodCommand";
const DUPLICATE_WINDOW_MS = 1000;

let lastExtractRequest: { key: string; at: number } | null = null;

interface CommandArgs {
  documentUri?: string;
  range?: vscode.Range;
  methodName?: string;
}

async function runExtractMethod(args?: CommandArgs): Promise<void> {
  const editor = vscode.window.activeTextEditor;
  if (!editor) {
    vscode.window.showWarningMessage("Open a Python file before running Extract Method.");
    return;
  }

  const document = editor.document;
  const range = args?.range ?? editor.selection;

  if (range.isEmpty) {
    vscode.window.showWarningMessage("Select one or more statements to extract.");
    return;
  }

  let methodName = args?.methodName;
  if (!methodName) {
    methodName = await vscode.window.showInputBox({
      prompt: "Name for the extracted method",
      value: "extracted",
      validateInput: (value) => {
        if (!value) {
          return "Method name cannot be empty";
        }
        if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(value)) {
          return "Method name must be a valid Python identifier";
        }
        return null;
      },
    });
    if (!methodName) {
      return; // user cancelled
    }
  }

  if (isDuplicateExtractRequest(document, range, methodName)) {
    logger.warn(SCOPE, "ignored duplicate extract request", extractRequestKey(document, range, methodName));
    return;
  }

  try {
    const result = await extractMethod({
      document,
      selection: range,
      options: { methodName },
    });
    const applied = await vscode.workspace.applyEdit(result.edit);
    if (!applied) {
      vscode.window.showErrorMessage("Failed to apply Extract Method edits.");
      return;
    }

    // Reveal the freshly inserted method so the user can see it.
    editor.revealRange(result.insertedRange, vscode.TextEditorRevealType.InCenterIfOutsideViewport);
    vscode.window.setStatusBarMessage(
      `Extracted '${result.signature.methodName}' (${result.signature.parameters.length} params, ${result.signature.returnVariables.length} returns).`,
      4000
    );
  } catch (err) {
    if (err instanceof ExtractMethodError) {
      vscode.window.showErrorMessage(err.message);
      logger.warn(SCOPE, "extract method failed", err.message, err.issues);
    } else {
      const message = err instanceof Error ? err.message : String(err);
      vscode.window.showErrorMessage(`Extract Method failed: ${message}`);
      logger.error(SCOPE, "unexpected error", message);
    }
  }
}

function isDuplicateExtractRequest(
  document: vscode.TextDocument,
  range: vscode.Range,
  methodName: string
): boolean {
  const key = extractRequestKey(document, range, methodName);
  const now = Date.now();
  const duplicate = lastExtractRequest !== null &&
    lastExtractRequest.key === key &&
    now - lastExtractRequest.at < DUPLICATE_WINDOW_MS;

  lastExtractRequest = { key, at: now };
  return duplicate;
}

function extractRequestKey(
  document: vscode.TextDocument,
  range: vscode.Range,
  methodName: string
): string {
  return [
    document.uri.toString(),
    range.start.line,
    range.start.character,
    range.end.line,
    range.end.character,
    methodName,
  ].join(":");
}

/** Register the command and CodeActionProvider on the extension context. */
export function registerExtractMethod(context: vscode.ExtensionContext): void {
  context.subscriptions.push(
    vscode.commands.registerCommand(EXTRACT_METHOD_COMMAND_ID, runExtractMethod)
  );

  context.subscriptions.push(
    vscode.languages.registerCodeActionsProvider(
      [{ language: "python" }, { pattern: "**/*.py" }],
      new ExtractMethodCodeActionProvider(),
      { providedCodeActionKinds: ExtractMethodCodeActionProvider.providedCodeActionKinds }
    )
  );
}

class ExtractMethodCodeActionProvider implements vscode.CodeActionProvider {
  static readonly providedCodeActionKinds = [vscode.CodeActionKind.RefactorExtract];

  provideCodeActions(
    document: vscode.TextDocument,
    range: vscode.Range | vscode.Selection,
    _context: vscode.CodeActionContext,
    _token: vscode.CancellationToken
  ): vscode.ProviderResult<(vscode.CodeAction | vscode.Command)[]> {
    if (!canOfferExtractMethod(document, range)) {
      return [];
    }

    const action = new vscode.CodeAction(
      "Extract Method...",
      vscode.CodeActionKind.RefactorExtract.append("function").append("python")
    );
    action.command = {
      command: EXTRACT_METHOD_COMMAND_ID,
      title: "Extract Method...",
      arguments: [{ documentUri: document.uri.toString(), range } satisfies CommandArgs],
    };
    return [action];
  }
}
