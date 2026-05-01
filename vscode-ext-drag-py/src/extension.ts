import * as vscode from "vscode";
import { isCloneVisualizerManagedDocument, registerCloneVisualizer } from "./commands/cloneVisualizer";
import { registerExtractMethod, EXTRACT_METHOD_COMMAND_ID } from "./commands/extractMethodCommand";
import { registerCodeSummarizer } from "./commands/codeSummarizer";
import { registerDropzone } from "./commands/dropzone";
import { logger, setDebugEnabled } from "./utils/logging";

let selectedText = "";
let selectedStartPosition: vscode.Position | null = null;
let selectedEndPosition: vscode.Position | null = null;
let selectedDocumentUri = "";
let selectedTextFingerprint = "";
let isHandlingDrop = false;

const DRAG_SCOPE = "dragDropDetector";

function isSupportedDocument(document: vscode.TextDocument): boolean {
  return (
    document.languageId === "java" ||
    document.languageId === "python" ||
    document.fileName.endsWith(".java") ||
    document.fileName.endsWith(".py")
  );
}

export function activate(context: vscode.ExtensionContext) {
  console.log("Extension activated!");

  // Toggle verbose tracing through the standard VSCode setting
  // (`pythonRefactor.debug`) so noisy logs don't surprise users by default.
  const config = vscode.workspace.getConfiguration("pythonRefactor");
  setDebugEnabled(Boolean(config.get<boolean>("debug")));

  // 1. Register the Extract Method command + CodeActionProvider.
  registerExtractMethod(context);
  registerDropzone(context);
  registerCloneVisualizer(context);
  registerCodeSummarizer(context);

  // 2. Capture selected text and its original position.
  context.subscriptions.push(
    vscode.window.onDidChangeTextEditorSelection((event) => {
      if (isHandlingDrop) {
        return;
      }

      const editor = event.textEditor;
      const document = editor.document;
      const selection = editor.selection;

      if (!isSupportedDocument(document)) {
        return;
      }

      if (!selection.isEmpty) {
        selectedText = document.getText(selection);
        selectedTextFingerprint = fingerprintCode(selectedText);
        selectedStartPosition = selection.start;
        selectedEndPosition = selection.end;
        selectedDocumentUri = document.uri.toString();
      }
    })
  );

  // 3. Detect drag/drop and offer Extract Method as a follow-up.
  context.subscriptions.push(
    vscode.workspace.onDidChangeTextDocument(async (event) => {
      if (isHandlingDrop) {
        return;
      }

      const document = event.document;

      if (!isSupportedDocument(document)) {
        return;
      }

      if (isCloneVisualizerManagedDocument(document)) {
        return;
      }

      if (!selectedText || !selectedStartPosition || !selectedEndPosition) {
        return;
      }

      if (document.uri.toString() !== selectedDocumentUri) {
        return;
      }

      for (const change of event.contentChanges) {
        if (isDroppedSelection(change.text)) {
          logger.info(DRAG_SCOPE, "drop detected; converting original selection to Extract Method");
          await extractMethodAfterDrop(document);
        }
      }
    })
  );
}

async function extractMethodAfterDrop(
  document: vscode.TextDocument
): Promise<void> {
  if (document.languageId !== "python" && !document.fileName.endsWith(".py")) {
    return;
  }

  if (!selectedStartPosition || !selectedEndPosition) {
    return;
  }

  const originalRange = new vscode.Range(selectedStartPosition, selectedEndPosition);

  isHandlingDrop = true;
  try {
    // VS Code applies the editor drag/drop before extensions see the text
    // change. Undo that raw move, then run our refactor on the original
    // selected range so the dropped copy does not remain as unreachable code.
    await vscode.commands.executeCommand("undo");
    await vscode.commands.executeCommand(EXTRACT_METHOD_COMMAND_ID, {
      documentUri: document.uri.toString(),
      range: originalRange,
      methodName: "extracted",
    });
  } finally {
    selectedText = "";
    selectedTextFingerprint = "";
    selectedStartPosition = null;
    selectedEndPosition = null;
    selectedDocumentUri = "";
    isHandlingDrop = false;
  }
}

function isDroppedSelection(text: string): boolean {
  return text.length > 0 && fingerprintCode(text) === selectedTextFingerprint;
}

function fingerprintCode(text: string): string {
  return text
    .split(/\r\n|\r|\n/)
    .map((line) => line.trim())
    .filter((line) => line.length > 0)
    .join("\n");
}

export function deactivate() {}
