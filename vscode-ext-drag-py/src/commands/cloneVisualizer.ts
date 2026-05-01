import * as fs from "fs";
import * as path from "path";
import * as vscode from "vscode";
import { cloneServerUrl, fetchCloneResults } from "../api/cloneApiClient";
import { enclosingClass } from "../languages/common/scopeAnalysis";
import { findEnclosingFunctionAt } from "../languages/python/pythonAnalyzer";
import { PYTHON_SCOPE_TYPES } from "../languages/python/pythonNodeTypes";
import { parseDocument } from "../parser/parserFactory";
import { extractMethod } from "../refactor/extractMethodService";
import { getLeadingIndent, indentBlock } from "../utils/indentation";
import { logger } from "../utils/logging";

const SHOW_CLONES_COMMAND_ID = "clone-visualizer.show_code_clones";
const PYTHON_REFACTOR_RESULTS_FILE = "all_refactor_results_py.json";
const JAVA_REFACTOR_RESULTS_FILE = "all_refactor_results.json";
const cloneVisualizerManagedFiles = new Set<string>();
const sharedCloneRecordMap = new Map<string, CloneRecord>();

export interface CloneSource {
  file: string;
  range: string;
  replacement_code?: string;
  enclosing_function?: {
    fun_range?: string;
  };
}

export interface CloneRecord {
  classid: string;
  sources: CloneSource[];
  extracted_method?: {
    method_name?: string;
    code?: string;
  };
}

interface TreeLeaf {
  name: string;
  file: string;
  range: string;
  parentClassid: string;
}

interface TreeGroup {
  name: string;
  classid: string;
  children: TreeLeaf[];
}

export interface PreparedCloneRefactoring {
  edit: vscode.WorkspaceEdit;
  firstFile: string;
}

export function registerCloneVisualizer(context: vscode.ExtensionContext): void {
  const log = vscode.window.createOutputChannel("Clone Visualizer - Drag Log");
  const lastOpenedByFile = new Map<string, string>();
  const openCloneEditors = new Map<string, vscode.ViewColumn>();
  let ignoreChangeUntil = 0;

  context.subscriptions.push(
    log,
    vscode.commands.registerCommand(SHOW_CLONES_COMMAND_ID, async () => {
      const loaded = await loadCloneRecordsForVisualizer(context, log);
      if (!loaded) {
        return;
      }
      const { records, label } = loaded;
      if (records.length === 0) {
        vscode.window.showWarningMessage("Clone Visualizer: no clone records found.");
        return;
      }

      sharedCloneRecordMap.clear();
      lastOpenedByFile.clear();
      openCloneEditors.clear();
      for (const record of records) {
        sharedCloneRecordMap.set(record.classid, record);
      }

      const panelTitle = label.includes("_py") || records.some((record) => record.sources.some((source) => source.file.endsWith(".py")))
        ? "Code Clone Tree (Python)"
        : "Code Clone Tree";
      const panel = vscode.window.createWebviewPanel(
        "cloneVisualizer",
        panelTitle,
        vscode.ViewColumn.One,
        {
          enableScripts: true,
          localResourceRoots: [vscode.Uri.file(path.join(context.extensionPath, "media"))],
        }
      );

      panel.webview.html = renderCloneTreeHtml(context, panel);
      setTimeout(() => {
        panel.webview.postMessage({ command: "loadData", data: parseCloneData(records) });
      }, 250);
      panel.webview.onDidReceiveMessage(
        async (message) => {
          if (message.command === "openFile") {
            await openCloneFile(message, context, lastOpenedByFile, openCloneEditors, log);
            return;
          }

          if (message.command === "applyExtractMethod") {
            await runApplyExtractMethod(String(message.classid ?? ""), context, sharedCloneRecordMap, log);
          }
        },
        undefined,
        context.subscriptions
      );
    }),
    vscode.workspace.onDidChangeTextDocument(async (event) => {
      if (Date.now() < ignoreChangeUntil) {
        return;
      }
      if (
        event.reason === vscode.TextDocumentChangeReason.Undo ||
        event.reason === vscode.TextDocumentChangeReason.Redo
      ) {
        return;
      }

      const document = event.document;
      if (!isSupportedDocument(document)) {
        return;
      }

      logDocumentChanges(document, event.contentChanges, event.reason, log);
      const dragMatch = findDragDownChange(event.contentChanges, log);
      if (!dragMatch) {
        return;
      }

      const classid = lastOpenedByFile.get(document.uri.fsPath);
      const record = classid ? sharedCloneRecordMap.get(classid) : undefined;
      log.appendLine(record ? `  -> FILE MATCH: ${record.classid}` : `  -> NO MATCH for ${document.uri.fsPath}`);
      if (!record) {
        return;
      }

      ignoreChangeUntil = Date.now() + 3000;
      try {
        await vscode.commands.executeCommand("undo");
        await runApplyExtractMethod(record.classid, context, sharedCloneRecordMap, log);
      } finally {
        setTimeout(() => {
          ignoreChangeUntil = 0;
        }, 250);
      }
    })
  );
}

export function isCloneVisualizerManagedDocument(document: vscode.TextDocument): boolean {
  return document.uri.scheme === "file" && cloneVisualizerManagedFiles.has(document.uri.fsPath);
}

export function findLoadedCloneRecordForRange(
  documentUri: vscode.Uri,
  range: vscode.Range,
  context: vscode.ExtensionContext
): CloneRecord | undefined {
  if (documentUri.scheme !== "file") {
    return undefined;
  }

  ensureSharedCloneRecordsLoaded(context);

  for (const record of sharedCloneRecordMap.values()) {
    for (const source of record.sources) {
      const resolved = resolvePath(source.file, context);
      if (!resolved || resolved !== documentUri.fsPath) {
        continue;
      }

      if (rangesReferToSameLines(range, source.range)) {
        return record;
      }
    }
  }
  return undefined;
}

function ensureSharedCloneRecordsLoaded(context: vscode.ExtensionContext): void {
  if (sharedCloneRecordMap.size > 0) {
    return;
  }

  for (const fileName of [PYTHON_REFACTOR_RESULTS_FILE, JAVA_REFACTOR_RESULTS_FILE]) {
    const dataPath = path.join(context.extensionPath, "media", fileName);
    if (!fs.existsSync(dataPath)) {
      continue;
    }

    for (const record of loadCloneRecords(dataPath)) {
      sharedCloneRecordMap.set(record.classid, record);
    }
  }
}

async function loadCloneRecordsForVisualizer(
  context: vscode.ExtensionContext,
  log: vscode.OutputChannel
): Promise<{ records: CloneRecord[]; label: string } | undefined> {
  const source = await vscode.window.showQuickPick(
    [
      { label: "Bundled Python JSON", description: `media/${PYTHON_REFACTOR_RESULTS_FILE}`, value: "media-python" },
      { label: "Bundled Java JSON", description: `media/${JAVA_REFACTOR_RESULTS_FILE}`, value: "media-java" },
      { label: "REST: all clones", description: "GET /get_clones", value: "rest-all" },
      { label: "REST: active file", description: "GET/POST /get_clone for the current file", value: "rest-active" },
      { label: "REST: choose files", description: "GET/POST /get_clone for selected files", value: "rest-files" },
      { label: "Pick JSON file", description: "Load clone/refactor results from disk", value: "json-file" },
    ],
    { title: "Load clone detection results" }
  );
  if (!source) {
    return undefined;
  }

  if (source.value.startsWith("rest-")) {
    const request = await buildCloneApiRequest(source.value);
    if (!request) {
      return undefined;
    }

    const serverUrl = cloneServerUrl();
    log.appendLine(`[api] requesting ${source.description} from ${serverUrl}`);
    try {
      const raw = await fetchCloneResults(request, { serverUrl });
      const records = normalizeCloneApiResponse(raw);
      return { records, label: `${source.value}:${serverUrl}` };
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      log.appendLine(`[api] request failed: ${message}`);
      vscode.window.showErrorMessage(
        `Clone Visualizer: could not reach clone API at ${serverUrl}. Start it with "npm run clone-api", or choose a bundled JSON option.`
      );
      return undefined;
    }
  }

  if (source.value === "media-python" || source.value === "media-java") {
    const fileName = source.value === "media-python"
      ? PYTHON_REFACTOR_RESULTS_FILE
      : JAVA_REFACTOR_RESULTS_FILE;
    const dataPath = path.join(context.extensionPath, "media", fileName);
    if (!fs.existsSync(dataPath)) {
      vscode.window.showErrorMessage(`Clone Visualizer: bundled clone data not found - ${dataPath}`);
      return undefined;
    }
    return { records: loadCloneRecords(dataPath), label: fileName };
  }

  const dataPath = await chooseRefactorResultsJsonPath();
  if (!dataPath) {
    return undefined;
  }
  if (!fs.existsSync(dataPath)) {
    vscode.window.showErrorMessage(`Clone Visualizer: clone data not found - ${dataPath}`);
    return undefined;
  }
  return { records: loadCloneRecords(dataPath), label: path.basename(dataPath) };
}

async function buildCloneApiRequest(source: string):
  Promise<{ kind: "all" } | { kind: "files"; files: string[] } | undefined> {
  if (source === "rest-all") {
    return { kind: "all" };
  }

  if (source === "rest-active") {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
      vscode.window.showWarningMessage("Open a file before requesting clones for the active file.");
      return undefined;
    }
    return { kind: "files", files: [editor.document.uri.fsPath] };
  }

  const picked = await vscode.window.showOpenDialog({
    canSelectFiles: true,
    canSelectFolders: false,
    canSelectMany: true,
    title: "Select files to request clone information for",
  });
  if (!picked || picked.length === 0) {
    return undefined;
  }
  return { kind: "files", files: picked.map((uri) => uri.fsPath) };
}

async function chooseRefactorResultsJsonPath(): Promise<string | undefined> {
  const picked = await vscode.window.showOpenDialog({
    canSelectFiles: true,
    canSelectFolders: false,
    canSelectMany: false,
    filters: { JSON: ["json"] },
    title: "Select clone/refactor results JSON",
  });
  return picked?.[0]?.fsPath;
}

function loadCloneRecords(dataPath: string): CloneRecord[] {
  const raw = JSON.parse(fs.readFileSync(dataPath, "utf8")) as unknown;
  return normalizeCloneApiResponse(raw);
}

function normalizeCloneApiResponse(raw: unknown): CloneRecord[] {
  const payload = unwrapClonePayload(raw);
  const normalizedRecords = Array.isArray(payload) ? payload : [payload];
  return normalizedRecords.map(normalizeLoadedCloneRecord).filter((record) => record.sources.length > 0);
}

function unwrapClonePayload(raw: unknown): unknown {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) {
    return raw;
  }

  const object = raw as Record<string, unknown>;
  return object.refactorability_analysis ??
    object.refactorabilityAnalysis ??
    object.refactor_results ??
    object.refactorResults ??
    object.clone_detection_results ??
    object.cloneDetectionResults ??
    object.clones ??
    object.records ??
    raw;
}

function normalizeLoadedCloneRecord(raw: unknown, index: number): CloneRecord {
  const record = raw as Partial<CloneRecord> & {
    id?: string;
    class_id?: string;
    clones?: CloneSource[];
  };
  return {
    classid: String(record.classid ?? record.class_id ?? record.id ?? `clone-${index + 1}`),
    sources: Array.isArray(record.sources)
      ? record.sources
      : Array.isArray(record.clones)
        ? record.clones
        : [],
    extracted_method: record.extracted_method,
  };
}

function parseCloneData(records: CloneRecord[]): { name: string; children: TreeGroup[] } {
  return {
    name: "Clone Results",
    children: records.map((record) => ({
      name: record.classid,
      classid: record.classid,
      children: record.sources.map((source) => ({
        name: `${path.basename(source.file)}:${source.range}`,
        file: source.file,
        range: source.range,
        parentClassid: record.classid,
      })),
    })),
  };
}

function renderCloneTreeHtml(context: vscode.ExtensionContext, panel: vscode.WebviewPanel): string {
  const htmlPath = path.join(context.extensionPath, "media", "collapsible-tree.html");
  if (!fs.existsSync(htmlPath)) {
    return fallbackCloneTreeHtml();
  }

  const d3Uri = panel.webview.asWebviewUri(
    vscode.Uri.file(path.join(context.extensionPath, "media", "d3.min.js"))
  );
  return fs.readFileSync(htmlPath, "utf8").replace("{{D3_URI}}", d3Uri.toString());
}

function fallbackCloneTreeHtml(): string {
  return `<!doctype html>
<html><body><h2>Code Clone Tree</h2><p>Missing media/collapsible-tree.html.</p></body></html>`;
}

async function openCloneFile(
  message: { file?: string; range?: string; parentClassid?: string },
  context: vscode.ExtensionContext,
  lastOpenedByFile: Map<string, string>,
  openCloneEditors: Map<string, vscode.ViewColumn>,
  log: vscode.OutputChannel
): Promise<void> {
  const file = message.file;
  if (!file) {
    return;
  }

  const resolved = resolvePath(file, context);
  if (!resolved) {
    vscode.window.showWarningMessage(`Clone Visualizer: file not found - ${file}`);
    return;
  }

  const parentClassid = message.parentClassid ?? "";
  if (parentClassid) {
    lastOpenedByFile.set(resolved, parentClassid);
    cloneVisualizerManagedFiles.add(resolved);
    log.appendLine(`[openFile] ${path.basename(resolved)} -> classid=${parentClassid}`);
  }

  const range = parseLineRange(message.range ?? "1-1");
  const document = await vscode.workspace.openTextDocument(vscode.Uri.file(resolved));
  const cloneKey = `${resolved}::${message.range ?? "1-1"}`;
  const targetColumn = pickCloneColumn(cloneKey, resolved, openCloneEditors);
  const editor = await vscode.window.showTextDocument(document, {
    selection: new vscode.Range(range.startLine, 0, range.endLine, Number.MAX_SAFE_INTEGER),
    viewColumn: targetColumn,
    preserveFocus: false,
  });
  openCloneEditors.set(cloneKey, editor.viewColumn ?? targetColumn);
}

async function runApplyExtractMethod(
  classid: string,
  context: vscode.ExtensionContext,
  recordMap: Map<string, CloneRecord>,
  log: vscode.OutputChannel
): Promise<void> {
  const record = recordMap.get(classid);
  if (!record) {
    vscode.window.showWarningMessage(`Clone Visualizer: clone group not found - ${classid}`);
    return;
  }

  let applied = false;
  try {
    applied = await applyTreeSitterCloneRefactoring(record, context, log);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    log.appendLine(`[refactor] Tree-sitter clone refactor failed: ${message}`);
    vscode.window.showErrorMessage(`Clone Visualizer: Tree-sitter refactor failed: ${message}`);
    return;
  }
  if (applied) {
    vscode.window.showInformationMessage(`Clone Visualizer: extract method applied for ${classid}.`);
    return;
  }

  vscode.window.showWarningMessage(`Clone Visualizer: unable to apply Tree-sitter refactor for ${classid}.`);
}

export async function applyTreeSitterCloneRefactoring(
  record: CloneRecord,
  context: vscode.ExtensionContext,
  log: vscode.OutputChannel
): Promise<boolean> {
  try {
    const prepared = await prepareTreeSitterCloneRefactoring(record, context, log);
    if (!prepared) {
      return false;
    }

    const applied = await vscode.workspace.applyEdit(prepared.edit);
    if (!applied) {
      const errorMsg = `failed to apply Tree-sitter refactor for ${record.classid}.`;
      log.appendLine(`[refactor] ${errorMsg}`);
      vscode.window.showErrorMessage(`Clone Visualizer: ${errorMsg}`);
      return false;
    }

    const firstDocument = await vscode.workspace.openTextDocument(vscode.Uri.file(prepared.firstFile));
    await vscode.window.showTextDocument(firstDocument, { viewColumn: vscode.ViewColumn.Active });
    return true;
  } catch (error) {
    const errorMsg = error instanceof Error ? error.message : String(error);
    log.appendLine(`[refactor] Tree-sitter clone refactor failed: ${errorMsg}`);
    logger.error("refactor", `Tree-sitter clone refactor exception: ${errorMsg}`);
    return false;
  }
}

export async function prepareTreeSitterCloneRefactoring(
  record: CloneRecord,
  context: vscode.ExtensionContext,
  log: vscode.OutputChannel
): Promise<PreparedCloneRefactoring | undefined> {
  const fileGroups = groupSourcesByResolvedFile(record.sources, context);
  if (fileGroups.size === 0) {
    vscode.window.showWarningMessage(`Clone Visualizer: no files found for ${record.classid}.`);
    return undefined;
  }

  log.appendLine(`[refactor] applying Tree-sitter clone Extract Method for ${record.classid}`);
  const workspaceEdit = new vscode.WorkspaceEdit();

  for (const [filePath, sources] of fileGroups) {
    const document = await vscode.workspace.openTextDocument(vscode.Uri.file(filePath));
    const methodName = "extracted";
    const primarySource = sources[0];
    const primaryRange = fullLineRangeFromCloneRange(document, primarySource.range);
    
    let primaryResult;
    try {
      primaryResult = await extractMethod({
        document,
        selection: primaryRange,
        options: { methodName },
      });
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : String(error);
      log.appendLine(`[refactor] Tree-sitter clone refactor failed: ${errorMsg}`);
      logger.error("refactor", `Tree-sitter clone refactor failed for ${record.classid}: ${errorMsg}`);
      throw error;
    }

    const computedCallSites = new Map<CloneSource, string>();
    computedCallSites.set(primarySource, primaryResult.callSite);

    for (const source of sources.slice(1)) {
      const sourceRange = fullLineRangeFromCloneRange(document, source.range);
      try {
        const result = await extractMethod({
          document,
          selection: sourceRange,
          options: { methodName },
        });
        computedCallSites.set(source, result.callSite);
      } catch (error) {
        const errorMsg = error instanceof Error ? error.message : String(error);
        log.appendLine(`[refactor] Tree-sitter clone refactor failed at clone site: ${errorMsg}`);
        logger.error("refactor", `Tree-sitter clone refactor failed at clone site for ${record.classid}: ${errorMsg}`);
        throw error;
      }
    }

    const sortedSources = [...sources].sort((left, right) =>
      parseLineRange(right.range).startLine - parseLineRange(left.range).startLine
    );
    const textEdits: vscode.TextEdit[] = [];
    for (const source of sortedSources) {
      const sourceRange = fullLineRangeFromCloneRange(document, source.range);
      textEdits.push(vscode.TextEdit.replace(sourceRange, computedCallSites.get(source) ?? primaryResult.callSite));
      log.appendLine(`[refactor] replace ${path.basename(filePath)}:${source.range} using Tree-sitter call site`);
    }

    const { position: insertionPosition, indent: insertionIndent } = insertionPositionForExtractedMethod(document, primaryRange);
    const methodText = formatExtractedMethodForInsertion(primaryResult.extractedMethodCode, document, insertionPosition, insertionIndent);
    textEdits.push(vscode.TextEdit.insert(insertionPosition, methodText));
    workspaceEdit.set(document.uri, sortTextEdits(textEdits));
    log.appendLine(`[refactor] insert Tree-sitter primary method at ${path.basename(filePath)}:${insertionPosition.line + 1}`);
  }

  const firstFile = [...fileGroups.keys()][0];
  return { edit: workspaceEdit, firstFile };
}

function sortTextEdits(edits: vscode.TextEdit[]): vscode.TextEdit[] {
  return [...edits].sort((left, right) => {
    const startComparison = left.range.start.compareTo(right.range.start);
    if (startComparison !== 0) {
      return startComparison;
    }
    return left.range.end.compareTo(right.range.end);
  });
}

function groupSourcesByResolvedFile(
  sources: CloneSource[],
  context: vscode.ExtensionContext
): Map<string, CloneSource[]> {
  const groups = new Map<string, CloneSource[]>();
  for (const source of sources) {
    const resolved = resolvePath(source.file, context);
    if (!resolved) {
      continue;
    }

    const group = groups.get(resolved) ?? [];
    group.push(source);
    groups.set(resolved, group);
  }
  return groups;
}

function fullLineRangeFromCloneRange(document: vscode.TextDocument, range: string): vscode.Range {
  const parsed = parseLineRange(range);
  const startLine = Math.min(Math.max(0, parsed.startLine), document.lineCount - 1);
  const endLine = Math.min(Math.max(startLine, parsed.endLine), document.lineCount - 1);
  const endPosition = endLine + 1 < document.lineCount
    ? new vscode.Position(endLine + 1, 0)
    : document.lineAt(endLine).range.end;
  return new vscode.Range(new vscode.Position(startLine, 0), endPosition);
}

function insertionPositionForExtractedMethod(
  document: vscode.TextDocument,
  primaryRange: vscode.Range
): { position: vscode.Position; indent: string } {
  const fallback = { position: document.positionAt(document.getText().length), indent: "" };
  const parsed = parseDocument(document);
  if (!parsed) {
    return fallback;
  }

  const fnNode = findEnclosingFunctionAt(parsed.root, document.offsetAt(primaryRange.start));
  if (!fnNode) {
    return fallback;
  }

  // Mirror editBuilder.ts: when inside a class, append after the class and
  // indent to the class-body level; otherwise stay at the enclosing function's
  // indentation level (handles nested functions like closures).
  const classNode = enclosingClass(fnNode, PYTHON_SCOPE_TYPES);
  const target = classNode ?? fnNode;
  const position = document.positionAt(target.endIndex);

  const startLine = target.startPosition.row;
  const baseIndent = startLine < document.lineCount
    ? getLeadingIndent(document.lineAt(startLine).text)
    : "";
  const indent = classNode ? baseIndent + "    " : baseIndent;

  return { position, indent };
}

function formatExtractedMethodForInsertion(
  extractedCode: string,
  document: vscode.TextDocument,
  insertionPosition: vscode.Position,
  indent: string
): string {
  const normalized = extractedCode.replace(/\s+$/g, "");
  const indented = indent ? indentBlock(normalized, indent) : normalized;
  const prefix = needsLeadingBlankLine(document, insertionPosition) ? "\n\n" : "\n";
  return `${prefix}${indented}\n`;
}

function needsLeadingBlankLine(document: vscode.TextDocument, insertionPosition: vscode.Position): boolean {
  if (insertionPosition.line <= 0 || insertionPosition.line >= document.lineCount) {
    return true;
  }

  const previousLine = document.lineAt(insertionPosition.line - 1).text;
  const currentLine = document.lineAt(insertionPosition.line).text;
  return previousLine.trim().length > 0 || currentLine.trim().length > 0;
}

function pickCloneColumn(
  cloneKey: string,
  resolved: string,
  openCloneEditors: Map<string, vscode.ViewColumn>
): vscode.ViewColumn {
  const previousColumn = openCloneEditors.get(cloneKey);
  const previousEditor = previousColumn !== undefined
    ? vscode.window.visibleTextEditors.find((editor) =>
      editor.viewColumn === previousColumn && editor.document.uri.fsPath === resolved)
    : undefined;
  if (previousEditor && previousColumn !== undefined) {
    return previousColumn;
  }

  const liveColumns = new Set(
    [...openCloneEditors.values()].filter((column) =>
      vscode.window.visibleTextEditors.some((editor) => editor.viewColumn === column))
  );
  let column = vscode.ViewColumn.Two as number;
  while (liveColumns.has(column as vscode.ViewColumn)) {
    column++;
  }
  return column as vscode.ViewColumn;
}

function findDragDownChange(
  changes: readonly vscode.TextDocumentContentChangeEvent[],
  log: vscode.OutputChannel
): { deletion: vscode.TextDocumentContentChangeEvent; insertion: vscode.TextDocumentContentChangeEvent } | undefined {
  if (changes.length !== 2) {
    log.appendLine(`  -> SKIP: expected 2 changes, got ${changes.length}`);
    return undefined;
  }

  const deletion = changes.find((change) => change.text === "" && change.rangeLength > 0);
  const insertion = changes.find((change) => change.text !== "" && change.rangeLength === 0);
  if (!deletion || !insertion) {
    log.appendLine("  -> SKIP: could not identify deletion+insertion pair");
    return undefined;
  }
  if (deletion.rangeLength !== insertion.text.length) {
    log.appendLine(`  -> SKIP: lengths differ (del=${deletion.rangeLength} ins=${insertion.text.length})`);
    return undefined;
  }
  if (insertion.text.replace(/\s+/g, "").length < 25) {
    log.appendLine("  -> SKIP: body too short");
    return undefined;
  }
  if (insertion.rangeOffset <= deletion.rangeOffset + deletion.rangeLength) {
    log.appendLine(
      `  -> SKIP: not drag-down (D=${deletion.rangeOffset} N=${deletion.rangeLength} I=${insertion.rangeOffset})`
    );
    return undefined;
  }

  return { deletion, insertion };
}

function logDocumentChanges(
  document: vscode.TextDocument,
  changes: readonly vscode.TextDocumentContentChangeEvent[],
  reason: vscode.TextDocumentChangeReason | undefined,
  log: vscode.OutputChannel
): void {
  log.appendLine(
    `[drag] ${path.basename(document.fileName)}  langId=${document.languageId}` +
    `  reason=${reason}  nChanges=${changes.length}`
  );
  changes.forEach((change, index) => {
    log.appendLine(
      `  [${index}] rangeOffset=${change.rangeOffset} rangeLen=${change.rangeLength}` +
      `  textLen=${change.text.length}  text=${JSON.stringify(change.text.slice(0, 60))}`
    );
  });
  log.show(true);
}

function isSupportedDocument(document: vscode.TextDocument): boolean {
  return ["java", "python"].includes(document.languageId) ||
    document.fileName.endsWith(".java") ||
    document.fileName.endsWith(".py");
}

function resolvePath(file: string, context?: vscode.ExtensionContext): string | undefined {
  if (path.isAbsolute(file) && fs.existsSync(file)) {
    return file;
  }

  const normalizedFile = file.replace(/\\/g, "/");
  const candidateSuffixes = suffixesForClonePath(normalizedFile);
  const roots = [
    ...(vscode.workspace.workspaceFolders ?? []).map((folder) => folder.uri.fsPath),
    context?.extensionPath,
    context ? path.join(context.extensionPath, "media") : undefined,
    context ? path.join(context.extensionPath, "media", "refactor_out") : undefined,
  ].filter((root): root is string => Boolean(root));

  for (const folder of vscode.workspace.workspaceFolders ?? []) {
    const candidate = path.resolve(folder.uri.fsPath, normalizedFile);
    if (fs.existsSync(candidate)) {
      return candidate;
    }
  }

  for (const root of roots) {
    for (const suffix of candidateSuffixes) {
      const direct = path.resolve(root, suffix);
      if (fs.existsSync(direct)) {
        return direct;
      }

      const found = findFileBySuffix(root, suffix);
      if (found) {
        return found;
      }
    }
  }

  return undefined;
}

function suffixesForClonePath(file: string): string[] {
  const suffixes = new Set<string>([file]);
  if (file.startsWith("systems-py/")) {
    suffixes.add(file.replace(/^systems-py\//, "systems/"));
  }
  if (file.startsWith("systems-java/")) {
    suffixes.add(file.replace(/^systems-java\//, "systems/"));
  }
  if (file.startsWith("output/refactor_out/")) {
    suffixes.add(file.replace(/^output\/refactor_out\//, ""));
  }
  if (file.startsWith("data/refactor_out/")) {
    suffixes.add(file.replace(/^data\/refactor_out\//, ""));
  }
  return [...suffixes];
}

function findFileBySuffix(root: string, suffix: string): string | undefined {
  if (!fs.existsSync(root) || !fs.statSync(root).isDirectory()) {
    return undefined;
  }

  const normalizedSuffix = suffix.replace(/\\/g, "/");
  const stack = [root];
  while (stack.length > 0) {
    const current = stack.pop()!;
    let entries: fs.Dirent[];
    try {
      entries = fs.readdirSync(current, { withFileTypes: true });
    } catch {
      continue;
    }

    for (const entry of entries) {
      const fullPath = path.join(current, entry.name);
      if (entry.isDirectory()) {
        stack.push(fullPath);
      } else if (fullPath.replace(/\\/g, "/").endsWith(normalizedSuffix)) {
        return fullPath;
      }
    }
  }
  return undefined;
}

function parseLineRange(range: string): { startLine: number; endLine: number } {
  const [startRaw, endRaw] = range.split("-");
  const startLine = Math.max(0, parseInt(startRaw, 10) - 1);
  const endLine = Math.max(startLine, parseInt(endRaw ?? startRaw, 10) - 1);
  return { startLine, endLine };
}

function rangesReferToSameLines(range: vscode.Range, cloneRange: string): boolean {
  const parsed = parseLineRange(cloneRange);
  const rangeEndLine = range.end.character === 0 && range.end.line > range.start.line
    ? range.end.line - 1
    : range.end.line;
  const sameRange = range.start.line === parsed.startLine && rangeEndLine === parsed.endLine;
  const selectionContainsClone = range.start.line <= parsed.startLine && rangeEndLine >= parsed.endLine;
  const cloneContainsSelection = parsed.startLine <= range.start.line && parsed.endLine >= rangeEndLine;
  const overlapsClone = range.start.line <= parsed.endLine && rangeEndLine >= parsed.startLine;
  return sameRange || selectionContainsClone || cloneContainsSelection || overlapsClone;
}
