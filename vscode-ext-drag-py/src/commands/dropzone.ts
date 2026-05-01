import * as vscode from "vscode";
import {
  applyTreeSitterCloneRefactoring,
  findLoadedCloneRecordForRange,
  prepareTreeSitterCloneRefactoring,
} from "./cloneVisualizer";
import type { CloneRecord, PreparedCloneRefactoring } from "./cloneVisualizer";
import { extractMethod } from "../refactor/extractMethodService";
import { logger } from "../utils/logging";
import * as path from "path";

const DROPZONE_VIEW_ID = "pythonRefactor.dropzone";
const ADD_SELECTION_COMMAND_ID = "pythonRefactor.dropzone.addSelection";
const CLEAR_COMMAND_ID = "pythonRefactor.dropzone.clear";
const REMOVE_COMMAND_ID = "pythonRefactor.dropzone.removeItem";
const REFACTOR_SELECTED_COMMAND_ID = "pythonRefactor.dropzone.refactorSelected";
const DROPZONE_MIME = "application/vnd.drag.dropzone";

interface DropPayload {
  content: string;
  origin?: {
    documentUri: string;
    range: [number, number, number, number];
    languageId: string;
  };
}

export function registerDropzone(context: vscode.ExtensionContext): void {
  const provider = new DropzoneProvider();
  const log = vscode.window.createOutputChannel("Dropzone Refactor");
  const treeView = vscode.window.createTreeView(DROPZONE_VIEW_ID, {
    treeDataProvider: provider,
    dragAndDropController: provider,
    canSelectMany: true,
  });

  context.subscriptions.push(
    log,
    treeView,
    vscode.commands.registerCommand(ADD_SELECTION_COMMAND_ID, () => addSelectionToDropzone(provider)),
    vscode.commands.registerCommand(REFACTOR_SELECTED_COMMAND_ID, async (_item?: DropItem, selectedItems?: DropItem[]) => {
      const items = selectedItems && selectedItems.length > 0 ? selectedItems : treeView.selection;
      await refactorDropzoneItems([...items], context, log);
    }),
    vscode.commands.registerCommand(CLEAR_COMMAND_ID, () => {
      if (!provider.clear()) {
        vscode.window.showInformationMessage("Dropzone is already empty.");
      }
    }),
    vscode.commands.registerCommand(REMOVE_COMMAND_ID, (item?: DropItem) => {
      if (item && !provider.removeItem(item)) {
        vscode.window.showWarningMessage("Could not remove that Dropzone item.");
      }
    }),
    vscode.languages.registerDocumentDropEditProvider(
      [{ scheme: "file" }, { scheme: "untitled" }],
      new EditorDropProvider(context, log)
    )
  );
}

class DropItem extends vscode.TreeItem {
  constructor(
    public readonly label: string,
    public readonly content: string,
    public readonly origin?: DropPayload["origin"]
  ) {
    super(label, vscode.TreeItemCollapsibleState.None);
    this.tooltip = content;
    this.contextValue = "dropzoneItem";
  }
}

/**
 * Provides the data for the Dropzone TreeView and handles drag/drop
 * interactions originating from the sidebar panel.
 */
class DropzoneProvider implements vscode.TreeDataProvider<DropItem>, vscode.TreeDragAndDropController<DropItem> {
  private readonly dropItems: DropItem[] = [];
  private readonly onDidChangeTreeDataEmitter = new vscode.EventEmitter<DropItem | undefined | void>();

  readonly onDidChangeTreeData = this.onDidChangeTreeDataEmitter.event;

  dropMimeTypes = [
    "text/plain",
    "text/html",
    "text/uri-list",
    "text/x-moz-url",
    "downloadurl",
    "resourceurls",
    "files",
    "public.utf8-plain-text",
    "public.plain-text",
  ];

  dragMimeTypes = ["text/plain", DROPZONE_MIME];

  getTreeItem(element: DropItem): vscode.TreeItem {
    return element;
  }

  getChildren(element?: DropItem): vscode.ProviderResult<DropItem[]> {
    return element ? [] : this.dropItems;
  }

  addSnippet(textContent: string, origin?: DropPayload["origin"]): void {
    const trimmed = textContent.trim();
    const label = trimmed.substring(0, 20).replace(/\n/g, " ") + (trimmed.length > 20 ? "..." : "");
    
    const originInfo = origin 
      ? ` [origin: ${path.basename(vscode.Uri.parse(origin.documentUri).fsPath)} @ L${origin.range[0]+1}:C${origin.range[1]}]`
      : "";
    logger.info("dropzone", `[addSnippet] item="${label}" size=${trimmed.length}${originInfo}`);
    
    this.dropItems.push(new DropItem(label || "Snippet", textContent, origin));
    this.onDidChangeTreeDataEmitter.fire();
  }

  removeItem(item: DropItem): boolean {
    const index = this.dropItems.indexOf(item);
    if (index < 0) {
      logger.warn("dropzone", `[removeItem] Item not found: "${item.label}"`);
      return false;
    }

    const preview = item.content.substring(0, 40).replace(/\n/g, " ");
    logger.info("dropzone", `[removeItem] Removed item: "${preview}..."`);
    this.dropItems.splice(index, 1);
    this.onDidChangeTreeDataEmitter.fire();
    return true;
  }

  clear(): boolean {
    if (this.dropItems.length === 0) {
      logger.info("dropzone", "[clear] Dropzone already empty");
      return false;
    }

    logger.info("dropzone", `[clear] Clearing dropzone (${this.dropItems.length} item(s))`);
    this.dropItems.length = 0;
    this.onDidChangeTreeDataEmitter.fire();
    return true;
  }

  async handleDrop(
    _target: DropItem | undefined,
    dataTransfer: vscode.DataTransfer,
    token: vscode.CancellationToken
  ): Promise<void> {
    if (token.isCancellationRequested) {
      return;
    }

    logger.info("dropzone", "[handleDrop] Attempting to read dropped content from dataTransfer");
    
    let content = await this.readDroppedText(dataTransfer, token);
    let source = "text/plain";
    
    if (!content?.trim()) {
      logger.info("dropzone", "[handleDrop] No text/plain content, trying URI list");
      content = await this.readSnippetFromEditorUriList(dataTransfer, token);
      source = "uri-list";
    }
    
    if (!content?.trim()) {
      logger.info("dropzone", "[handleDrop] No URI list content, trying download URL");
      content = await this.readSnippetFromDownloadUrl(dataTransfer, token);
      source = "downloadurl";
    }
    
    if (!content?.trim()) {
      logger.warn("dropzone", "[handleDrop] Failed to read any content from drop");
      vscode.window.showWarningMessage(
        "Dropzone could not read that drag. Use the Add Selection to Dropzone command or copy text first."
      );
      return;
    }

    const contentPreview = content.trim().substring(0, 60).replace(/\n/g, " ") + (content.trim().length > 60 ? "..." : "");
    logger.info("dropzone", `[handleDrop] Added to dropzone from ${source}: "${contentPreview}"`);
    this.addSnippet(content);
    vscode.window.showInformationMessage("Added to Dropzone.");
  }

  async handleDrag(
    source: readonly DropItem[],
    dataTransfer: vscode.DataTransfer,
    token: vscode.CancellationToken
  ): Promise<void> {
    if (source.length === 0 || token.isCancellationRequested) {
      return;
    }

    const draggedItem = source[0];
    const preview = draggedItem.content.substring(0, 40).replace(/\n/g, " ");
    logger.info("dropzone", `[handleDrag] Dragging item: "${preview}..." (${draggedItem.content.length} chars)`);
    
    const payload: DropPayload = {
      content: draggedItem.content,
      origin: draggedItem.origin,
    };
    
    if (draggedItem.origin) {
      logger.info("dropzone", `  [origin] ${path.basename(vscode.Uri.parse(draggedItem.origin.documentUri).fsPath)} @ L${draggedItem.origin.range[0]+1}:C${draggedItem.origin.range[1]}-L${draggedItem.origin.range[2]+1}:C${draggedItem.origin.range[3]}`);
    }
    
    dataTransfer.set("text/plain", new vscode.DataTransferItem(draggedItem.content));
    dataTransfer.set(DROPZONE_MIME, new vscode.DataTransferItem(JSON.stringify(payload)));
  }

  private async readDroppedText(
    dataTransfer: vscode.DataTransfer,
    token: vscode.CancellationToken
  ): Promise<string | undefined> {
    let best: string | undefined;
    const consider = async (mimeType: string, item: vscode.DataTransferItem) => {
      if (token.isCancellationRequested) {
        return;
      }

      const normalizedMime = mimeType.toLowerCase();
      if (
        normalizedMime.startsWith("application/vnd.code.tree.") ||
        normalizedMime === "text/uri-list" ||
        normalizedMime === "downloadurl"
      ) {
        return;
      }

      let text: string | undefined;
      try {
        text = await item.asString();
      } catch {
        text = typeof item.value === "string" ? item.value : undefined;
      }

      if (!text?.trim()) {
        return;
      }

      if (
        normalizedMime === "text/plain" ||
        normalizedMime === "public.utf8-plain-text" ||
        normalizedMime === "public.plain-text"
      ) {
        best = text;
        return;
      }

      if (!best || text.trim().length > best.trim().length) {
        best = text;
      }
    };

    const plain = dataTransfer.get("text/plain");
    if (plain) {
      await consider("text/plain", plain);
      if (best) {
        return best;
      }
    }

    for (const [mimeType, item] of dataTransfer) {
      if (token.isCancellationRequested) {
        break;
      }
      await consider(mimeType, item);
    }
    return best;
  }

  private async readSnippetFromEditorUriList(
    dataTransfer: vscode.DataTransfer,
    token: vscode.CancellationToken
  ): Promise<string | undefined> {
    const items: vscode.DataTransferItem[] = [];
    for (const [mime, item] of dataTransfer) {
      if (mime.toLowerCase() === "text/uri-list") {
        items.push(item);
      }
    }

    for (const item of items) {
      if (token.isCancellationRequested) {
        return undefined;
      }

      let raw: string | undefined;
      try {
        raw = await item.asString();
      } catch {
        raw = typeof item.value === "string" ? item.value : undefined;
      }
      if (!raw?.trim()) {
        continue;
      }

      const lines = raw
        .split(/\r?\n/)
        .map((line) => line.trim())
        .filter((line) => line.length > 0 && !line.startsWith("#"));
      for (const line of lines) {
        const text = await this.snippetFromUriListLine(line, token);
        if (text?.trim()) {
          return text;
        }
      }
    }
    return undefined;
  }

  private async readSnippetFromDownloadUrl(
    dataTransfer: vscode.DataTransfer,
    token: vscode.CancellationToken
  ): Promise<string | undefined> {
    const item = dataTransfer.get("downloadurl");
    if (!item) {
      return undefined;
    }

    let raw: string | undefined;
    try {
      raw = await item.asString();
    } catch {
      raw = typeof item.value === "string" ? item.value : undefined;
    }
    if (!raw) {
      return undefined;
    }

    const firstColon = raw.indexOf(":");
    const secondColon = firstColon >= 0 ? raw.indexOf(":", firstColon + 1) : -1;
    if (secondColon < 0) {
      return undefined;
    }

    return this.snippetFromUriListLine(raw.slice(secondColon + 1).trim(), token);
  }

  private async snippetFromUriListLine(
    line: string,
    token: vscode.CancellationToken
  ): Promise<string | undefined> {
    let uri: vscode.Uri;
    try {
      uri = vscode.Uri.parse(line, true);
    } catch {
      try {
        uri = vscode.Uri.file(line);
      } catch {
        return undefined;
      }
    }

    if (token.isCancellationRequested) {
      return undefined;
    }

    let fragment = uri.fragment;
    try {
      fragment = decodeURIComponent(fragment);
    } catch {
      // Keep the raw fragment when decoding fails.
    }

    const parsed = DropzoneProvider.parseLinkFragment(fragment);
    const clean = uri.with({ fragment: "" });
    const docFromBuffer = DropzoneProvider.findOpenDocument(clean);

    if (docFromBuffer) {
      if (!parsed) {
        const editor = vscode.window.visibleTextEditors.find((candidate) => candidate.document === docFromBuffer);
        if (editor && !editor.selection.isEmpty) {
          return editor.document.getText(fullLineRangeForSelection(editor.document, editor.selection));
        }
        return DropzoneProvider.textFromEditorSelectionForUri(clean);
      }
      return docFromBuffer.getText(DropzoneProvider.rangeForOpenDocument(docFromBuffer, parsed));
    }

    if (!parsed) {
      return DropzoneProvider.textFromEditorSelectionForUri(clean);
    }

    try {
      const doc = await vscode.workspace.openTextDocument(clean);
      return doc.getText(DropzoneProvider.rangeForOpenDocument(doc, parsed));
    } catch {
      return DropzoneProvider.textFromEditorSelectionForUri(clean);
    }
  }

  private static findOpenDocument(clean: vscode.Uri): vscode.TextDocument | undefined {
    return vscode.workspace.textDocuments.find((document) =>
      document.uri.toString() === clean.toString() ||
      (document.uri.scheme === "file" && clean.scheme === "file" && document.uri.fsPath === clean.fsPath)
    );
  }

  private static textFromEditorSelectionForUri(clean: vscode.Uri): string | undefined {
    const active = vscode.window.activeTextEditor;
    if (!active || active.selection.isEmpty) {
      return undefined;
    }

    const docUri = active.document.uri;
    if (
      docUri.toString() === clean.toString() ||
      (docUri.scheme === "file" && clean.scheme === "file" && docUri.fsPath === clean.fsPath)
    ) {
      return active.document.getText(fullLineRangeForSelection(active.document, active.selection));
    }
    return undefined;
  }

  private static parseLinkFragment(fragment: string):
    | { kind: "wholeLine"; line1: number }
    | { kind: "range"; startLine1: number; startCol1: number; endLine1: number; endCol1?: number }
    | undefined {
    const match = /^L?(\d+)(?:,(\d+))?(-L?(\d+)(?:,(\d+))?)?/.exec(fragment);
    if (!match) {
      return undefined;
    }

    const startLine = parseInt(match[1], 10);
    const startCol = match[2] ? parseInt(match[2], 10) : 1;
    if (!match[4]) {
      return { kind: "wholeLine", line1: startLine };
    }

    const endLine = parseInt(match[4], 10);
    const endCol = match[5] ? parseInt(match[5], 10) : undefined;
    return { kind: "range", startLine1: startLine, startCol1: startCol, endLine1: endLine, endCol1: endCol };
  }

  private static rangeForOpenDocument(
    doc: vscode.TextDocument,
    parsed: Exclude<ReturnType<typeof DropzoneProvider.parseLinkFragment>, undefined>
  ): vscode.Range {
    if (parsed.kind === "wholeLine") {
      const lineIndex = Math.min(Math.max(0, parsed.line1 - 1), doc.lineCount - 1);
      return doc.lineAt(lineIndex).range;
    }

    const startLine = Math.min(Math.max(0, parsed.startLine1 - 1), doc.lineCount - 1);
    const endLine = Math.min(Math.max(0, parsed.endLine1 - 1), doc.lineCount - 1);
    const startLineDoc = doc.lineAt(startLine);
    const endLineDoc = doc.lineAt(endLine);
    const startChar = Math.min(Math.max(0, parsed.startCol1 - 1), startLineDoc.text.length);
    const endChar = parsed.endCol1 !== undefined
      ? Math.min(Math.max(0, parsed.endCol1 - 1), endLineDoc.text.length)
      : endLineDoc.text.length;
    return new vscode.Range(new vscode.Position(startLine, startChar), new vscode.Position(endLine, endChar));
  }
}

class EditorDropProvider implements vscode.DocumentDropEditProvider {
  constructor(
    private readonly context: vscode.ExtensionContext,
    private readonly log: vscode.OutputChannel
  ) {}

  async provideDocumentDropEdits(
    document: vscode.TextDocument,
    position: vscode.Position,
    dataTransfer: vscode.DataTransfer,
    token: vscode.CancellationToken
  ): Promise<vscode.DocumentDropEdit | undefined> {
    const dropzoneItem = dataTransfer.get(DROPZONE_MIME);
    if (!dropzoneItem) {
      return undefined;
    }

    logger.info("dropzone", `[provideDocumentDropEdits] Dropping into ${path.basename(document.fileName)} @ L${position.line + 1}:C${position.character}`);

    const payload = parseDropPayload(await dropzoneItem.asString());
    const content = payload.content;
    if (!content?.trim() || token.isCancellationRequested) {
      logger.warn("dropzone", "[provideDocumentDropEdits] Empty or cancelled drop");
      return undefined;
    }

    if (payload.origin && document.languageId === "python") {
      const originUri = vscode.Uri.parse(payload.origin.documentUri);
      const sameDocument = originUri.toString() === document.uri.toString() ||
        (originUri.scheme === "file" && document.uri.scheme === "file" && originUri.fsPath === document.uri.fsPath);

      if (sameDocument) {
        logger.info("dropzone", `[provideDocumentDropEdits] Same document - origin @ L${payload.origin.range[0]+1}:C${payload.origin.range[1]} to L${payload.origin.range[2]+1}:C${payload.origin.range[3]}`);
        const originRange = rangeFromPayload(payload.origin.range);
        setTimeout(async () => {
          await applyTreeSitterExtractMethod(originUri, originRange, this.context, this.log);
        }, 50);
        logger.info("dropzone", "[provideDocumentDropEdits] Scheduled Tree-sitter extract method");
        return new vscode.DocumentDropEdit("");
      }
    }

    const name = await vscode.window.showInputBox({
      title: "Wrap in Extracted Method",
      prompt: "Name for the method that will wrap the dropped snippet",
      value: document.languageId === "python" ? "extracted" : "extractedMethod",
      validateInput: (value) => {
        const identifier = document.languageId === "python"
          ? /^[A-Za-z_][A-Za-z0-9_]*$/
          : /^[A-Za-z_$][\w$]*$/;
        return identifier.test(value) ? undefined : "Enter a valid identifier";
      },
    });

    if (name === undefined || token.isCancellationRequested) {
      logger.info("dropzone", "[provideDocumentDropEdits] User cancelled method name input");
      return undefined;
    }

    const methodName = name.trim() || (document.languageId === "python" ? "extracted" : "extractedMethod");
    const outerIndent = indentAtDropPosition(document, position);
    const wrapped = wrapInMethod(content, methodName, document.languageId, outerIndent);
    
    logger.info("dropzone", `[provideDocumentDropEdits] Wrapping content (${content.length} chars) in method "${methodName}"`);
    vscode.window.showInformationMessage(`Snippet wrapped in ${methodName}().`);
    return new vscode.DocumentDropEdit(wrapped);
  }
}

async function addSelectionToDropzone(provider: DropzoneProvider): Promise<void> {
  const editor = vscode.window.activeTextEditor;
  if (!editor || editor.selection.isEmpty) {
    vscode.window.showWarningMessage("Select code before adding it to the Dropzone.");
    return;
  }

  const range = fullLineRangeForSelection(editor.document, editor.selection);
  const content = editor.document.getText(range);
  const preview = content.trim().substring(0, 60).replace(/\n/g, " ");
  
  logger.info("dropzone", `[addSelectionToDropzone] Adding selection from ${path.basename(editor.document.fileName)} (${content.length} chars): "${preview}..."`);
  
  provider.addSnippet(content, {
    documentUri: editor.document.uri.toString(),
    range: rangeToPayload(range),
    languageId: editor.document.languageId,
  });
  vscode.window.showInformationMessage("Added selection to Dropzone.");
}

async function refactorDropzoneItems(
  items: DropItem[],
  context: vscode.ExtensionContext,
  log: vscode.OutputChannel
): Promise<void> {
  if (items.length === 0) {
    vscode.window.showWarningMessage("Select one or more Dropzone items to refactor.");
    return;
  }

  logger.info("dropzone", `[refactorDropzoneItems] Processing ${items.length} selected item(s)`);

  const cloneRecords = new Map<string, CloneRecord>();
  for (const item of items) {
    const record = cloneRecordForDropItem(item, context);
    if (record) {
      cloneRecords.set(record.classid, record);
      logger.info("dropzone", `  [${record.classid}] Matched to clone group`);
    } else {
      const preview = item.content.substring(0, 40).replace(/\n/g, " ");
      logger.warn("dropzone", `  No clone group match for: "${preview}..."`);
    }
  }

  if (cloneRecords.size === 0) {
    logger.error("dropzone", "[refactorDropzoneItems] No clone groups matched");
    vscode.window.showWarningMessage("Selected Dropzone items do not match any loaded clone groups.");
    return;
  }

  log.appendLine(`[dropzone] preparing ${cloneRecords.size} clone group refactor(s) in parallel`);
  logger.info("dropzone", `[refactorDropzoneItems] Starting refactor for ${cloneRecords.size} clone group(s)`);
  log.show(true);

  try {
    const prepared = await Promise.all(
      [...cloneRecords.values()].map((record) =>
        prepareTreeSitterCloneRefactoring(record, context, log)
      )
    );
    const validPrepared = prepared.filter((item): item is PreparedCloneRefactoring => Boolean(item));
    
    logger.info("dropzone", `[refactorDropzoneItems] Prepared ${validPrepared.length}/${cloneRecords.size} refactors`);
    
    if (validPrepared.length === 0) {
      logger.error("dropzone", "[refactorDropzoneItems] No refactor edits were prepared");
      vscode.window.showWarningMessage("Dropzone: no refactor edits were prepared.");
      return;
    }

    const mergedEdit = mergeWorkspaceEdits(validPrepared.map((item) => item.edit));
    logger.info("dropzone", `[refactorDropzoneItems] Applying merged workspace edits`);
    
    const applied = await vscode.workspace.applyEdit(mergedEdit);
    if (!applied) {
      logger.error("dropzone", "[refactorDropzoneItems] Failed to apply workspace edits");
      vscode.window.showErrorMessage("Dropzone: failed to apply selected clone refactors.");
      return;
    }

    const firstDocument = await vscode.workspace.openTextDocument(vscode.Uri.file(validPrepared[0].firstFile));
    await vscode.window.showTextDocument(firstDocument, { viewColumn: vscode.ViewColumn.Active });
    
    logger.info("dropzone", `[refactorDropzoneItems] Successfully refactored ${validPrepared.length} clone group(s)`);
    vscode.window.showInformationMessage(`Dropzone: refactored ${validPrepared.length} clone group(s).`);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    logger.error("dropzone", `[refactorDropzoneItems] Exception: ${message}`);
    log.appendLine(`[dropzone] selected refactor failed: ${message}`);
    vscode.window.showErrorMessage(`Dropzone: selected refactor failed: ${message}`);
  }
}

function cloneRecordForDropItem(
  item: DropItem,
  context: vscode.ExtensionContext
): CloneRecord | undefined {
  if (!item.origin) {
    return undefined;
  }

  return findLoadedCloneRecordForRange(
    vscode.Uri.parse(item.origin.documentUri),
    rangeFromPayload(item.origin.range),
    context
  );
}

function mergeWorkspaceEdits(edits: vscode.WorkspaceEdit[]): vscode.WorkspaceEdit {
  const merged = new vscode.WorkspaceEdit();
  const editsByUri = new Map<string, { uri: vscode.Uri; edits: vscode.TextEdit[] }>();

  for (const edit of edits) {
    for (const [uri, textEdits] of edit.entries()) {
      const key = uri.toString();
      const entry = editsByUri.get(key) ?? { uri, edits: [] };
      entry.edits.push(...textEdits);
      editsByUri.set(key, entry);
    }
  }

  for (const { uri, edits: textEdits } of editsByUri.values()) {
    merged.set(uri, textEdits);
  }
  return merged;
}

async function applyTreeSitterExtractMethod(
  documentUri: vscode.Uri,
  range: vscode.Range,
  context: vscode.ExtensionContext,
  log: vscode.OutputChannel
): Promise<void> {
  try {
    logger.info("dropzone", `[applyTreeSitterExtractMethod] Processing extract for ${path.basename(documentUri.fsPath)} @ L${range.start.line + 1}:C${range.start.character}-L${range.end.line + 1}:C${range.end.character}`);
    
    const cloneRecord = findLoadedCloneRecordForRange(documentUri, range, context);
    if (cloneRecord) {
      logger.info("dropzone", `[applyTreeSitterExtractMethod] Found clone group: ${cloneRecord.classid}`);
      log.appendLine(`[dropzone] range belongs to clone group ${cloneRecord.classid}; applying multi-site Tree-sitter refactor`);
      log.show(true);
      const applied = await applyTreeSitterCloneRefactoring(cloneRecord, context, log);
      if (!applied) {
        logger.error("dropzone", `[applyTreeSitterExtractMethod] Failed to apply clone refactor for ${cloneRecord.classid}`);
        vscode.window.showWarningMessage(`Dropzone: unable to apply clone-group refactor for ${cloneRecord.classid}.`);
      } else {
        logger.info("dropzone", `[applyTreeSitterExtractMethod] Successfully applied clone refactor for ${cloneRecord.classid}`);
      }
      return;
    }

    logger.info("dropzone", "[applyTreeSitterExtractMethod] No clone group found - applying single-site extract");
    log.appendLine("[dropzone] no clone group found for saved range; applying single-site Tree-sitter refactor");
    log.show(true);
    
    const document = await vscode.workspace.openTextDocument(documentUri);
    const result = await extractMethod({
      document,
      selection: range,
      options: { methodName: "extracted" },
    });
    
    logger.info("dropzone", `[applyTreeSitterExtractMethod] Extract method created: ${result.signature.methodName}`);
    
    const applied = await vscode.workspace.applyEdit(result.edit);
    if (!applied) {
      logger.error("dropzone", "[applyTreeSitterExtractMethod] Failed to apply extract method edits");
      vscode.window.showErrorMessage("Dropzone: failed to apply Extract Method edits.");
      return;
    }

    const editor = await vscode.window.showTextDocument(document);
    editor.revealRange(result.insertedRange, vscode.TextEditorRevealType.InCenterIfOutsideViewport);
    
    logger.info("dropzone", `[applyTreeSitterExtractMethod] Successfully applied single-site extract: ${result.signature.methodName}`);
    vscode.window.setStatusBarMessage(
      `Dropzone extracted '${result.signature.methodName}' using Tree-sitter.`,
      4000
    );
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    logger.error("dropzone", `[applyTreeSitterExtractMethod] Exception: ${message}`);
    log.appendLine(`[dropzone] Tree-sitter Extract Method failed: ${message}`);
    log.show(true);
    vscode.window.showErrorMessage(`Dropzone: Tree-sitter Extract Method failed: ${message}`);
  }
}

function parseDropPayload(raw: string): DropPayload {
  try {
    const parsed = JSON.parse(raw) as Partial<DropPayload>;
    if (typeof parsed.content === "string") {
      return {
        content: parsed.content,
        origin: parsed.origin,
      };
    }
  } catch {
    // Older Dropzone items used the raw snippet text as the custom payload.
  }

  return { content: raw };
}

function rangeToPayload(range: vscode.Range): [number, number, number, number] {
  return [range.start.line, range.start.character, range.end.line, range.end.character];
}

function rangeFromPayload(payload: [number, number, number, number]): vscode.Range {
  return new vscode.Range(
    new vscode.Position(payload[0], payload[1]),
    new vscode.Position(payload[2], payload[3])
  );
}

function fullLineRangeForSelection(document: vscode.TextDocument, selection: vscode.Selection): vscode.Range {
  const startLine = selection.start.line;
  const endLine = selection.end.character === 0 && selection.end.line > selection.start.line
    ? selection.end.line - 1
    : selection.end.line;
  const endPosition = endLine + 1 < document.lineCount
    ? new vscode.Position(endLine + 1, 0)
    : document.lineAt(endLine).range.end;

  return new vscode.Range(new vscode.Position(startLine, 0), endPosition);
}

function normalizeBodyLines(body: string, bodyIndent: string): string[] {
  const lines = body.split("\n");
  while (lines.length > 0 && lines[lines.length - 1].trim() === "") {
    lines.pop();
  }

  const nonEmptyIndents = lines
    .filter((line) => line.trim().length > 0)
    .map((line) => (/^(\s*)/.exec(line)![1].length));
  const positiveIndents = nonEmptyIndents.filter((indent) => indent > 0);
  const minIndent = positiveIndents.length > 0
    ? Math.min(...positiveIndents)
    : 0;
  return lines.map((line) => bodyIndent + line.slice(minIndent));
}

function wrapInMethod(body: string, methodName: string, languageId: string, outerIndent: string): string {
  const step = "    ";
  const bodyIndent = outerIndent + step;
  const bodyLines = normalizeBodyLines(body, bodyIndent);

  if (languageId === "python") {
    return `${outerIndent}def ${methodName}():\n${bodyLines.join("\n")}`;
  }

  const header = languageId === "java"
    ? `private void ${methodName}()`
    : `function ${methodName}()`;
  return `${outerIndent}${header} {\n${bodyLines.join("\n")}\n${outerIndent}}`;
}

function indentAtDropPosition(document: vscode.TextDocument, position: vscode.Position): string {
  const tryLine = (lineIndex: number): string | undefined => {
    if (lineIndex < 0 || lineIndex >= document.lineCount) {
      return undefined;
    }
    const match = /^(\s*)/.exec(document.lineAt(lineIndex).text);
    return match ? match[1] : "";
  };

  const dropLineIndent = tryLine(position.line) ?? "";
  if (dropLineIndent.length > 0) {
    return dropLineIndent;
  }

  for (let index = position.line - 1; index >= 0; index--) {
    const text = document.lineAt(index).text;
    if (text.trim().length > 0) {
      return /^(\s*)/.exec(text)![1];
    }
  }
  return "";
}
