import * as path from "path";
import * as vscode from "vscode";
import { spawn } from "child_process";

const SUMMARIZE_COMMAND_ID = "codeSummarizer.summarizeSelection";
const SUMMARY_VIEW_ID = "codeSummarizer.results";
const PYTHON_MODEL = "Dreamuno/llm-codeknowhow-python";
const GENERAL_MODEL = "Dreamuno/llm_codeknowhow_v1";

interface SummaryResult {
  summary: string;
  probability?: number;
}

export function registerCodeSummarizer(context: vscode.ExtensionContext): void {
  const provider = new SummaryResultsProvider();

  context.subscriptions.push(
    vscode.window.registerWebviewViewProvider(SUMMARY_VIEW_ID, provider, {
      webviewOptions: { retainContextWhenHidden: true },
    }),
    vscode.commands.registerCommand(SUMMARIZE_COMMAND_ID, async () => summarizeSelection(context, provider))
  );
}

async function summarizeSelection(
  context: vscode.ExtensionContext,
  provider: SummaryResultsProvider
): Promise<void> {
  const editor = vscode.window.activeTextEditor;
  if (!editor) {
    vscode.window.showErrorMessage("No active editor detected.");
    return;
  }

  const code = editor.document.getText(editor.selection);
  if (!code.trim()) {
    vscode.window.showWarningMessage("Select code before running Code Summarizer.");
    return;
  }

  const config = vscode.workspace.getConfiguration("codeSummarizer");
  const pythonPath = config.get<string>("pythonPath") || "/opt/anaconda3/envs/code-sum311/bin/python";
  const scriptPath = config.get<string>("scriptPath") || path.join(context.extensionPath, "summarizer.py");
  const model = modelForDocument(editor.document, config);

  await vscode.window.withProgress(
    {
      location: vscode.ProgressLocation.Notification,
      title: "Summarizing selected code",
      cancellable: false,
    },
    async () => {
      try {
        const results = await runSummarizer(pythonPath, scriptPath, model, code);
        provider.update(results.slice(0, 10));
        await vscode.commands.executeCommand("workbench.view.explorer");
        vscode.window.setStatusBarMessage("Code Summary updated in the Explorer sidebar.", 3000);
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        vscode.window.showErrorMessage(`Code Summarizer failed: ${message}`);
      }
    }
  );
}

function modelForDocument(document: vscode.TextDocument, config: vscode.WorkspaceConfiguration): string {
  const configuredModel = config.get<string>("model");
  if (configuredModel?.trim()) {
    return configuredModel.trim();
  }

  if (document.languageId === "python" || document.fileName.endsWith(".py")) {
    return PYTHON_MODEL;
  }
  return GENERAL_MODEL;
}

function runSummarizer(
  pythonPath: string,
  scriptPath: string,
  model: string,
  code: string
): Promise<SummaryResult[]> {
  return new Promise((resolve, reject) => {
    const child = spawn(pythonPath, [scriptPath, "--json", "--stdin", "--model", model], {
      stdio: ["pipe", "pipe", "pipe"],
    });

    let stdout = "";
    let stderr = "";

    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    child.stdout.on("data", (chunk: string) => {
      stdout += chunk;
    });
    child.stderr.on("data", (chunk: string) => {
      stderr += chunk;
    });
    child.on("error", (error) => {
      reject(error);
    });
    child.on("close", (codeNumber) => {
      if (codeNumber !== 0) {
        reject(new Error(stderr.trim() || `summarizer.py exited with code ${codeNumber}`));
        return;
      }

      try {
        const parsed = JSON.parse(stdout.trim()) as SummaryResult[];
        resolve(Array.isArray(parsed) ? parsed : []);
      } catch {
        resolve(parseLegacyOutput(stdout));
      }
    });

    child.stdin.end(code, "utf8");
  });
}

function parseLegacyOutput(stdout: string): SummaryResult[] {
  return stdout
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line.length > 0)
    .map((line) => {
      const match = /\(Probability: ([0-9.]+)\)$/.exec(line);
      if (!match) {
        return { summary: line };
      }
      return {
        summary: line.replace(/\(Probability: [0-9.]+\)$/, "").trim(),
        probability: Number(match[1]),
      };
    });
}

class SummaryResultsProvider implements vscode.WebviewViewProvider {
  private view: vscode.WebviewView | undefined;
  private results: SummaryResult[] = [];

  resolveWebviewView(webviewView: vscode.WebviewView): void {
    this.view = webviewView;
    webviewView.webview.options = { enableScripts: true };
    webviewView.webview.onDidReceiveMessage(async (message: { command?: string; index?: number }) => {
      if (message.command !== "copy" || typeof message.index !== "number") {
        return;
      }

      const result = this.results[message.index];
      if (!result) {
        return;
      }

      await vscode.env.clipboard.writeText(cleanSummary(result.summary));
      vscode.window.setStatusBarMessage("Copied selected summary to clipboard.", 3000);
    });
    this.render();
  }

  update(results: SummaryResult[]): void {
    this.results = results;
    this.render();
  }

  private render(): void {
    if (!this.view) {
      return;
    }

    this.view.webview.html = summaryViewHtml(this.view.webview, this.results);
  }
}

function summaryViewHtml(webview: vscode.Webview, results: SummaryResult[]): string {
  const nonce = getNonce();
  const rows = results.length > 0
    ? results.map((result, index) => summaryButton(result, index)).join("")
    : `<div class="empty">Run Code Summarizer on a selection.</div>`;

  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta
    http-equiv="Content-Security-Policy"
    content="default-src 'none'; style-src ${webview.cspSource} 'unsafe-inline'; script-src 'nonce-${nonce}';"
  >
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    body {
      background: var(--vscode-sideBar-background);
      color: var(--vscode-sideBar-foreground);
      font-family: var(--vscode-font-family);
      font-size: var(--vscode-font-size);
      margin: 0;
      padding: 8px;
    }

    .summary-list {
      display: flex;
      flex-direction: column;
      gap: 6px;
    }

    .summary-item {
      background: var(--vscode-list-inactiveSelectionBackground, transparent);
      border: 1px solid var(--vscode-panel-border);
      border-radius: 6px;
      color: inherit;
      cursor: pointer;
      display: grid;
      gap: 6px 8px;
      grid-template-columns: 24px minmax(0, 1fr);
      padding: 8px;
      text-align: left;
      width: 100%;
    }

    .summary-item:hover {
      background: var(--vscode-list-hoverBackground);
    }

    .summary-item:focus {
      outline: 1px solid var(--vscode-focusBorder);
      outline-offset: 1px;
    }

    .rank {
      color: var(--vscode-textLink-foreground);
      font-weight: 700;
      line-height: 1.35;
      text-align: right;
    }

    .summary {
      line-height: 1.35;
      overflow-wrap: anywhere;
      white-space: normal;
    }

    .meta {
      color: var(--vscode-descriptionForeground);
      font-size: 0.9em;
      grid-column: 2;
    }

    .best {
      color: var(--vscode-textLink-foreground);
      font-weight: 700;
      margin-left: 6px;
    }

    .empty {
      color: var(--vscode-descriptionForeground);
      line-height: 1.4;
      padding: 4px;
    }
  </style>
</head>
<body>
  <div class="summary-list">${rows}</div>
  <script nonce="${nonce}">
    const vscode = acquireVsCodeApi();
    document.querySelectorAll('[data-summary-index]').forEach((button) => {
      button.addEventListener('click', () => {
        vscode.postMessage({ command: 'copy', index: Number(button.dataset.summaryIndex) });
      });
    });
  </script>
</body>
</html>`;
}

function summaryButton(result: SummaryResult, index: number): string {
  const probability = typeof result.probability === "number" && Number.isFinite(result.probability)
    ? result.probability.toFixed(4)
    : "n/a";
  const best = index === 0 ? `<span class="best">Best</span>` : "";
  return `<button class="summary-item" type="button" data-summary-index="${index}">
  <span class="rank">${index + 1}</span>
  <span class="summary">${escapeHtml(cleanSummary(result.summary))}${best}</span>
  <span class="meta">Probability: ${escapeHtml(probability)}</span>
</button>`;
}

function cleanSummary(summary: string): string {
  return summary.replace(/\?$/, "");
}

function escapeHtml(value: string): string {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function getNonce(): string {
  const chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
  let nonce = "";
  for (let index = 0; index < 32; index++) {
    nonce += chars.charAt(Math.floor(Math.random() * chars.length));
  }
  return nonce;
}
