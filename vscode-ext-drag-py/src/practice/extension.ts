import * as vscode from "vscode";
import { getEnclosingFunctionWithPosition } from "./utilTreeSitter";

let selectedText = "";
let selectedStartPosition: vscode.Position | null = null;
let selectedDocumentUri = "";

const PYTHON_MODE: vscode.DocumentSelector = [
  { language: "python", scheme: "file" },
  { pattern: "**/*.py", scheme: "file" },
];

function isPythonDocument(document: vscode.TextDocument): boolean {
  return document.languageId === "python" || document.fileName.endsWith(".py");
}

export function activate(context: vscode.ExtensionContext): void {
  console.log("[DBG] Practice extension activated");

  context.subscriptions.push(
    vscode.window.onDidChangeTextEditorSelection((event) => {
      const { document } = event.textEditor;
      const { selection } = event.textEditor;

      if (!isPythonDocument(document) || selection.isEmpty) {
        return;
      }

      selectedText = document.getText(selection);
      selectedStartPosition = selection.start;
      selectedDocumentUri = document.uri.toString();
    })
  );

  context.subscriptions.push(
    vscode.workspace.onDidChangeTextDocument(async (event) => {
      const { document } = event;

      if (
        !isPythonDocument(document) ||
        !selectedText ||
        !selectedStartPosition ||
        document.uri.toString() !== selectedDocumentUri
      ) {
        return;
      }

      for (const change of event.contentChanges) {
        if (change.text.trim() !== selectedText.trim()) {
          continue;
        }

        const droppedRange = rangeForInsertedText(change.range.start, change.text);
        console.log("[DBG] Possible Python drag/drop detected");
        console.log("[DBG] Dropped selected text:");
        console.log(selectedText);

        const enclosingFunction = await getEnclosingFunctionWithPosition(
          document,
          droppedRange.start
        );

        if (!enclosingFunction) {
          console.log("[DBG] <no enclosing Python function found>");
          continue;
        }

        console.log(
          `[DBG] Enclosing Python function starts at line=${enclosingFunction.startPosition.line}, character=${enclosingFunction.startPosition.character}`
        );
        console.log("[DBG] Function text:");
        console.log(enclosingFunction.text);
      }
    })
  );

  const disposable = vscode.languages.registerCompletionItemProvider(
    PYTHON_MODE,
    {
      async provideCompletionItems(
        document: vscode.TextDocument,
        position: vscode.Position,
        _token: vscode.CancellationToken,
        context: vscode.CompletionContext
      ): Promise<vscode.CompletionItem[] | undefined> {
        if (context.triggerKind !== vscode.CompletionTriggerKind.Invoke) {
          return undefined;
        }

        const enclosingFunction = await getEnclosingFunctionWithPosition(
          document,
          position
        );

        if (!enclosingFunction) {
          return [];
        }

        console.log(
          `[DBG] Cursor is inside Python function starting at line=${enclosingFunction.startPosition.line}, character=${enclosingFunction.startPosition.character}`
        );
        console.log(enclosingFunction.text);

        return [];
      },
    }
  );

  context.subscriptions.push(disposable);
}

function rangeForInsertedText(start: vscode.Position, text: string): vscode.Range {
  const lines = text.split(/\r\n|\r|\n/);
  const end = lines.length === 1
    ? start.translate(0, text.length)
    : new vscode.Position(start.line + lines.length - 1, lines[lines.length - 1].length);

  return new vscode.Range(start, end);
}

export function deactivate(): void {}
