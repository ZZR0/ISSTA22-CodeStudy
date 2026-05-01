/******/ (() => { // webpackBootstrap
/******/ 	"use strict";
/******/ 	var __webpack_modules__ = ([
/* 0 */
/***/ (function(__unused_webpack_module, exports, __webpack_require__) {


var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", ({ value: true }));
exports.activate = activate;
exports.deactivate = deactivate;
const vscode = __importStar(__webpack_require__(1));
const fs = __importStar(__webpack_require__(2));
const path = __importStar(__webpack_require__(3));
const cp = __importStar(__webpack_require__(4));
// ── Constants ─────────────────────────────────────────────────────────────────
// __dirname at runtime = <ext-root>/dist; one level up is the project root
const EXT_ROOT = path.dirname(__dirname);
const REFACTOR_OUT = path.join(EXT_ROOT, 'media', 'refactor_out');
// ── Parser / transformer ──────────────────────────────────────────────────────
function parseCloneData(records) {
    const projectMap = new Map();
    for (const record of records) {
        if (!projectMap.has(record.project)) {
            projectMap.set(record.project, new Map());
        }
        projectMap.get(record.project).set(record.classid, record);
    }
    const projectNodes = [];
    for (const [project, cloneMap] of projectMap) {
        const cloneNodes = [];
        for (const [classid, record] of cloneMap) {
            const sourceNodes = record.sources.map(src => ({
                name: `${path.basename(src.file)}  (lines ${src.range})`,
                file: src.file,
                range: src.range,
                func_id: src.func_id,
                parentClassid: classid,
            }));
            cloneNodes.push({
                name: `${classid}  [${record.refactoring_type} · ${record.nclones} clones]`,
                classid,
                children: sourceNodes,
            });
        }
        projectNodes.push({ name: project, children: cloneNodes });
    }
    return { name: 'All Code Clones', children: projectNodes };
}
// ── Path helpers ──────────────────────────────────────────────────────────────
function resolvePath(relFile) {
    if (path.isAbsolute(relFile) && fs.existsSync(relFile)) {
        return relFile;
    }
    const folders = vscode.workspace.workspaceFolders ?? [];
    const candidates = [
        path.join(EXT_ROOT, relFile),
        ...folders.map(f => path.join(f.uri.fsPath, relFile)),
    ];
    return candidates.find(c => fs.existsSync(c));
}
// ── Drag helpers ──────────────────────────────────────────────────────────────
/**
 * Reconstructs the pre-drag document text from the post-drag snapshot.
 * Only handles drag-DOWN (insertion offset > deletion end in original coords).
 *
 * In a drag-down: post_drag = orig[0..D] + orig[D+N..I] + body + orig[I..]
 * Inverse:        pre_drag  = post[0..D] + body + post[D..I-N] + post[I..]
 *                           (where I-N = gapStart, I = gapEnd in post-drag coords)
 */
function revertDrag(postDrag, deletionOffset, // D  — where body was deleted (pre-change = post-drag hole position)
gapStart, // I-N — where body was inserted in post-drag coords
gapEnd, // I   — end of inserted body in post-drag coords
body) {
    return (postDrag.slice(0, deletionOffset) +
        body +
        postDrag.slice(deletionOffset, gapStart) +
        postDrag.slice(gapEnd));
}
function wholeDocRange(doc) {
    const last = doc.lineAt(doc.lineCount - 1);
    return new vscode.Range(0, 0, last.lineNumber, last.range.end.character);
}
function resolveSummarizerScript(extensionPath) {
    const cfg = vscode.workspace.getConfiguration('cloneVisualizer');
    const override = (cfg.get('summarizerScriptPath') ?? '').trim();
    if (override && fs.existsSync(override)) {
        return override;
    }
    const bundled = path.join(extensionPath, 'scripts', 'summarizer.py');
    if (fs.existsSync(bundled)) {
        return bundled;
    }
    const sibling = path.join(extensionPath, '..', 'code-summarizer', 'summarizer.py');
    if (fs.existsSync(sibling)) {
        return sibling;
    }
    return undefined;
}
function resolvePythonExecutable(extensionPath) {
    const cfg = vscode.workspace.getConfiguration('cloneVisualizer');
    const configured = (cfg.get('pythonPath') ?? '').trim();
    if (configured) {
        return configured;
    }
    const venvDir = path.join(extensionPath, '..', 'code-summarizer', '.venv');
    const winPy = path.join(venvDir, 'Scripts', 'python.exe');
    const unixPy = path.join(venvDir, 'bin', 'python');
    if (fs.existsSync(winPy)) {
        return winPy;
    }
    if (fs.existsSync(unixPy)) {
        return unixPy;
    }
    return process.platform === 'win32' ? 'python' : 'python3';
}
/** Percent for display, e.g. 12.7 or 9.07 (trims trailing zeros). */
function formatSummaryPercent(probability) {
    const pct = probability * 100;
    let s = pct.toFixed(2);
    s = s.replace(/0+$/, '');
    s = s.replace(/\.$/, '');
    return s || '0';
}
function runSummarizeProcess(python, scriptPath, code) {
    return new Promise((resolve, reject) => {
        const proc = cp.spawn(python, [scriptPath, '--json', '--stdin'], {
            windowsHide: true,
        });
        let out = '';
        let err = '';
        proc.stdout.setEncoding('utf8');
        proc.stderr.setEncoding('utf8');
        proc.stdout.on('data', (c) => { out += c; });
        proc.stderr.on('data', (c) => { err += c; });
        proc.on('error', e => reject(e));
        proc.on('close', (exitCode, _sig) => {
            if (exitCode !== 0) {
                reject(new Error(err.trim() || `Summarizer exited with code ${exitCode}`));
                return;
            }
            try {
                const parsed = JSON.parse(out.trim());
                resolve(parsed);
            }
            catch {
                reject(new Error(out.trim() || err.trim() || 'Empty output from summarizer'));
            }
        });
        proc.stdin.write(code, 'utf8');
        proc.stdin.end();
    });
}
// ── Apply pre-computed WorkspaceEdit ─────────────────────────────────────────
async function applyPrecomputedRefactoring(doc, record) {
    const edit = new vscode.WorkspaceEdit();
    // Sort sources descending by start line so replacements don't shift each other
    const sorted = [...record.sources].sort((a, b) => {
        const aLine = parseInt(a.range.split('-')[0], 10);
        const bLine = parseInt(b.range.split('-')[0], 10);
        return bLine - aLine;
    });
    for (const src of sorted) {
        if (!src.replacement_code) {
            continue;
        }
        const m = /^(\d+)-(\d+)$/.exec(src.range.trim());
        if (!m) {
            continue;
        }
        const start0 = parseInt(m[1], 10) - 1;
        const end0 = parseInt(m[2], 10) - 1;
        if (start0 < 0 || end0 >= doc.lineCount || start0 > end0) {
            continue;
        }
        edit.replace(doc.uri, new vscode.Range(new vscode.Position(start0, 0), new vscode.Position(end0, doc.lineAt(end0).text.length)), src.replacement_code);
    }
    // Insert extracted method after source[0]'s enclosing function closing brace
    const em = record.extracted_method;
    const src0 = record.sources[0];
    if (em?.code && src0?.enclosing_function?.fun_range) {
        const fm = /^(\d+)-(\d+)$/.exec(src0.enclosing_function.fun_range.trim());
        if (fm) {
            const encEnd0 = parseInt(fm[2], 10) - 1;
            if (encEnd0 >= 0 && encEnd0 < doc.lineCount) {
                const closingLine = doc.lineAt(encEnd0).text;
                const indentMatch = /^(\s*)/.exec(closingLine);
                const memberIndent = indentMatch ? indentMatch[1] : '    ';
                const methodCode = em.code
                    .split('\n')
                    .map(l => (l.trim() ? memberIndent + l : l))
                    .join('\n');
                edit.insert(doc.uri, new vscode.Position(encEnd0, closingLine.length), '\n\n' + methodCode);
            }
        }
    }
    return vscode.workspace.applyEdit(edit);
}
// ── Dropzone sidebar ──────────────────────────────────────────────────────────
/**
 * Represents a single snippet item within the Dropzone sidebar.
 */
class DropItem extends vscode.TreeItem {
    label;
    content;
    constructor(label, content) {
        super(label, vscode.TreeItemCollapsibleState.None);
        this.label = label;
        this.content = content;
        this.tooltip = content;
        this.description = '';
        this.contextValue = 'dropzoneSnippet';
    }
}
/**
 * Provides the data for the Dropzone TreeView and handles drag/drop interactions
 * originating from the sidebar panel.
 */
class DropzoneProvider {
    dropItems = [];
    _onDidChangeTreeData = new vscode.EventEmitter();
    onDidChangeTreeData = this._onDidChangeTreeData.event;
    // Accept broad editor/OS types; VS Code normalises keys to lowercase before matching.
    dropMimeTypes = [
        'text/plain',
        'text/html',
        'text/uri-list',
        'text/x-moz-url',
        'downloadurl',
        'resourceurls',
        'files',
        'public.utf8-plain-text',
        'public.plain-text',
    ];
    dragMimeTypes = ['text/plain'];
    getTreeItem(element) { return element; }
    getChildren(element) {
        return element ? [] : this.dropItems;
    }
    addSnippet(textContent) {
        const label = textContent.trim().substring(0, 20).replace(/\n/g, ' ') + '...';
        this.dropItems.push(new DropItem(label, textContent));
        this._onDidChangeTreeData.fire();
    }
    removeItem(item) {
        const i = this.dropItems.indexOf(item);
        if (i < 0) {
            return false;
        }
        this.dropItems.splice(i, 1);
        this._onDidChangeTreeData.fire();
        return true;
    }
    clear() {
        if (this.dropItems.length === 0) {
            return false;
        }
        this.dropItems.length = 0;
        this._onDidChangeTreeData.fire();
        return true;
    }
    async handleDrop(_target, dataTransfer, token) {
        if (token.isCancellationRequested) {
            return;
        }
        let content = await this.readDroppedText(dataTransfer, token);
        if (!content?.trim()) {
            content = await this.readSnippetFromEditorUriList(dataTransfer, token);
        }
        if (!content?.trim()) {
            content = await this.readSnippetFromDownloadUrl(dataTransfer, token);
        }
        if (!content?.trim()) {
            vscode.window.showWarningMessage('Dropzone could not read that drag. Use ⌘⇧R / Ctrl+Shift+R with a selection, or copy text and run "Add to Dropzone".');
            return;
        }
        this.addSnippet(content);
        vscode.window.showInformationMessage('Added to Dropzone (drag).');
    }
    async readDroppedText(dataTransfer, token) {
        let best;
        const consider = async (mimeType, item) => {
            if (token.isCancellationRequested) {
                return;
            }
            const mt = mimeType.toLowerCase();
            if (mt.startsWith('application/vnd.code.tree.') || mt === 'text/uri-list' || mt === 'downloadurl') {
                return;
            }
            let s;
            try {
                s = await item.asString();
            }
            catch {
                s = typeof item.value === 'string' ? item.value : undefined;
            }
            if (!s?.trim()) {
                return;
            }
            if (mt === 'text/plain' || mt === 'public.utf8-plain-text' || mt === 'public.plain-text') {
                best = s;
                return;
            }
            if (!best || s.trim().length > best.trim().length) {
                best = s;
            }
        };
        const plain = dataTransfer.get('text/plain');
        if (plain) {
            await consider('text/plain', plain);
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
    async readSnippetFromEditorUriList(dataTransfer, token) {
        const items = [];
        for (const [mime, item] of dataTransfer) {
            if (mime.toLowerCase() === 'text/uri-list') {
                items.push(item);
            }
        }
        for (const item of items) {
            if (token.isCancellationRequested) {
                return undefined;
            }
            let raw;
            try {
                raw = await item.asString();
            }
            catch {
                raw = typeof item.value === 'string' ? item.value : undefined;
            }
            if (!raw?.trim()) {
                continue;
            }
            const lines = raw.split(/\r?\n/).map(l => l.trim()).filter(l => l.length > 0 && !l.startsWith('#'));
            for (const line of lines) {
                const text = await this.snippetFromUriListLine(line, token);
                if (text?.trim()) {
                    return text;
                }
            }
        }
        return undefined;
    }
    async readSnippetFromDownloadUrl(dataTransfer, token) {
        const item = dataTransfer.get('downloadurl');
        if (!item) {
            return undefined;
        }
        let raw;
        try {
            raw = await item.asString();
        }
        catch {
            raw = typeof item.value === 'string' ? item.value : undefined;
        }
        if (!raw) {
            return undefined;
        }
        const a = raw.indexOf(':');
        const b = a >= 0 ? raw.indexOf(':', a + 1) : -1;
        if (b < 0) {
            return undefined;
        }
        return this.snippetFromUriListLine(raw.slice(b + 1).trim(), token);
    }
    async snippetFromUriListLine(line, token) {
        let uri;
        try {
            uri = vscode.Uri.parse(line, true);
        }
        catch {
            try {
                uri = vscode.Uri.file(line);
            }
            catch {
                return undefined;
            }
        }
        if (token.isCancellationRequested) {
            return undefined;
        }
        const fragRaw = uri.fragment;
        let frag = fragRaw;
        try {
            frag = decodeURIComponent(fragRaw);
        }
        catch { /* keep raw */ }
        const parsed = DropzoneProvider.parseLinkFragment(frag);
        const clean = uri.with({ fragment: '' });
        const docFromBuffer = DropzoneProvider.findOpenDocument(clean);
        if (docFromBuffer) {
            if (!parsed) {
                const ed = vscode.window.visibleTextEditors.find(e => e.document === docFromBuffer);
                if (ed && !ed.selection.isEmpty) {
                    return ed.document.getText(ed.selection);
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
        }
        catch {
            return DropzoneProvider.textFromEditorSelectionForUri(clean);
        }
    }
    static findOpenDocument(clean) {
        return vscode.workspace.textDocuments.find(d => d.uri.toString() === clean.toString() ||
            (d.uri.scheme === 'file' && clean.scheme === 'file' && d.uri.fsPath === clean.fsPath));
    }
    static textFromEditorSelectionForUri(clean) {
        const active = vscode.window.activeTextEditor;
        if (!active || active.selection.isEmpty) {
            return undefined;
        }
        const du = active.document.uri;
        if (du.toString() === clean.toString() || (du.scheme === 'file' && clean.scheme === 'file' && du.fsPath === clean.fsPath)) {
            return active.document.getText(active.selection);
        }
        return undefined;
    }
    static parseLinkFragment(fragment) {
        const match = /^L?(\d+)(?:,(\d+))?(-L?(\d+)(?:,(\d+))?)?/.exec(fragment);
        if (!match) {
            return undefined;
        }
        const startLine = parseInt(match[1], 10);
        const startCol = match[2] ? parseInt(match[2], 10) : 1;
        if (!match[4]) {
            return { kind: 'wholeLine', line1: startLine };
        }
        const endLine = parseInt(match[4], 10);
        const endCol = match[5] ? parseInt(match[5], 10) : undefined;
        return { kind: 'range', startLine1: startLine, startCol1: startCol, endLine1: endLine, endCol1: endCol };
    }
    static rangeForOpenDocument(doc, parsed) {
        if (parsed.kind === 'wholeLine') {
            const lineIdx = Math.min(Math.max(0, parsed.line1 - 1), doc.lineCount - 1);
            return doc.lineAt(lineIdx).range;
        }
        const sl = Math.min(Math.max(0, parsed.startLine1 - 1), doc.lineCount - 1);
        const el = Math.min(Math.max(0, parsed.endLine1 - 1), doc.lineCount - 1);
        const startLineDoc = doc.lineAt(sl);
        const endLineDoc = doc.lineAt(el);
        const startChar = Math.min(Math.max(0, parsed.startCol1 - 1), startLineDoc.text.length);
        const endChar = parsed.endCol1 !== undefined
            ? Math.min(Math.max(0, parsed.endCol1 - 1), endLineDoc.text.length)
            : endLineDoc.text.length;
        return new vscode.Range(new vscode.Position(sl, startChar), new vscode.Position(el, endChar));
    }
    async handleDrag(source, dataTransfer, token) {
        if (source.length === 0 || token.isCancellationRequested) {
            return;
        }
        const draggedItem = source[0];
        dataTransfer.set('text/plain', new vscode.DataTransferItem(draggedItem.content));
        dataTransfer.set('application/vnd.drag.dropzone', new vscode.DataTransferItem(draggedItem.content));
    }
}
// ── Wrap helper ───────────────────────────────────────────────────────────────
/**
 * Strips the common leading whitespace from all non-empty lines of `body`,
 * then re-indents each line with `bodyIndent`.
 */
function normalizeBodyLines(body, bodyIndent) {
    const lines = body.split('\n');
    // Drop trailing blank lines
    while (lines.length > 0 && lines[lines.length - 1].trim() === '') {
        lines.pop();
    }
    const nonEmpty = lines.filter(l => l.trim().length > 0);
    const minIndent = nonEmpty.length === 0
        ? 0
        : Math.min(...nonEmpty.map(l => (/^(\s*)/.exec(l)[1].length)));
    return lines.map(l => bodyIndent + l.slice(minIndent));
}
/**
 * Wraps `body` inside a method/function definition appropriate for the given language.
 * `outerIndent` is the leading whitespace of the line where the user dropped the snippet.
 */
function wrapInMethod(body, methodName, lang, outerIndent) {
    const step = '    ';
    const bodyIndent = outerIndent + step;
    const bodyLines = normalizeBodyLines(body, bodyIndent);
    if (lang === 'python') {
        // Python uses indentation instead of braces
        return `${outerIndent}def ${methodName}():\n${bodyLines.join('\n')}`;
    }
    const header = lang === 'java'
        ? `private void ${methodName}()`
        : `function ${methodName}()`; // TypeScript / JavaScript / fallback
    return `${outerIndent}${header} {\n${bodyLines.join('\n')}\n${outerIndent}}`;
}
/**
 * Returns the leading-whitespace indentation to use for the generated method.
 * Prefers the drop-position line; falls back to the nearest preceding non-blank line.
 */
function indentAtDropPosition(document, position) {
    const tryLine = (lineIdx) => {
        if (lineIdx < 0 || lineIdx >= document.lineCount) {
            return undefined;
        }
        const text = document.lineAt(lineIdx).text;
        const m = /^(\s*)/.exec(text);
        return m ? m[1] : '';
    };
    const dropLineIndent = tryLine(position.line) ?? '';
    if (dropLineIndent.length > 0) {
        return dropLineIndent;
    }
    // Drop line is empty — walk backwards to find a non-blank line
    for (let i = position.line - 1; i >= 0; i--) {
        const text = document.lineAt(i).text;
        if (text.trim().length > 0) {
            return (/^(\s*)/.exec(text)[1]);
        }
    }
    return '';
}
// ── EditorDropProvider ────────────────────────────────────────────────────────
/**
 * Intercepts items dropped from the Dropzone sidebar into a text editor.
 *
 * Two paths:
 *  1. Clone-aware  — file was opened from the clone tree → apply the pre-computed
 *     two-clone "Extract Method" refactoring (both sites updated at once).
 *  2. Generic      — no clone context → prompt for a name and insert the snippet
 *     wrapped in a language-appropriate function definition.
 */
class EditorDropProvider {
    lastOpenedByFile;
    recordMap;
    constructor(lastOpenedByFile, recordMap) {
        this.lastOpenedByFile = lastOpenedByFile;
        this.recordMap = recordMap;
    }
    async provideDocumentDropEdits(document, position, dataTransfer, token) {
        const dropzoneItem = dataTransfer.get('application/vnd.drag.dropzone');
        if (!dropzoneItem) {
            return undefined;
        }
        const content = await dropzoneItem.asString();
        if (!content?.trim()) {
            return undefined;
        }
        // ── Path 1: clone-aware ──────────────────────────────────────────────
        // The file was opened by clicking a source node in the clone tree,
        // so we know which clone group it belongs to.
        const classid = this.lastOpenedByFile.get(document.uri.fsPath);
        const record = classid ? this.recordMap.get(classid) : undefined;
        if (record) {
            const answer = await vscode.window.showWarningMessage(`Apply "Extract Method" for clone group "${record.classid}"?`, {
                modal: true,
                detail: `${record.sources.length} clone site(s) will be updated together.\n` +
                    `Use Ctrl+Z / ⌘Z to undo.`,
            }, 'Apply');
            if (answer !== 'Apply' || token.isCancellationRequested) {
                return undefined;
            }
            // Let VS Code finish applying the no-op drop edit before the workspace edit runs.
            setTimeout(async () => {
                const freshDoc = await vscode.workspace.openTextDocument(document.uri);
                const applied = await applyPrecomputedRefactoring(freshDoc, record);
                if (applied) {
                    vscode.window.showInformationMessage(`Clone Visualizer: extract method applied for ${record.classid} ` +
                        `— ${record.sources.length} clone site(s) updated.`);
                }
            }, 50);
            // Return an empty edit so VS Code has nothing to insert at the drop point.
            return new vscode.DocumentDropEdit('');
        }
        // ── Path 2: generic wrap ─────────────────────────────────────────────
        // No clone context — just wrap the snippet in a new method definition.
        const name = await vscode.window.showInputBox({
            title: 'Wrap in Extracted Method',
            prompt: 'Name for the method that will wrap the dropped snippet',
            value: 'extractedMethod',
            validateInput: (v) => (/^[a-zA-Z_$][\w$]*$/.test(v) ? undefined : 'Valid identifier'),
        });
        if (name === undefined || token.isCancellationRequested) {
            return undefined;
        }
        const methodName = name.trim() || 'extractedMethod';
        const outerIndent = indentAtDropPosition(document, position);
        const wrapped = wrapInMethod(content, methodName, document.languageId, outerIndent);
        vscode.window.showInformationMessage(`Snippet wrapped in ${methodName}() and inserted.`);
        return new vscode.DocumentDropEdit(wrapped);
    }
}
// ── Extension entry point ─────────────────────────────────────────────────────
function activate(context) {
    console.log('Congratulations, your extension "clone-visualizer" is now active!');
    const summaryChannel = vscode.window.createOutputChannel('Clone Visualizer — Code Summary');
    context.subscriptions.push(summaryChannel);
    const summarizeCmd = vscode.commands.registerCommand('clone-visualizer.summarizeSelection', async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor || editor.selection.isEmpty) {
            vscode.window.showWarningMessage('Select code in the editor to summarize.');
            return;
        }
        const code = editor.document.getText(editor.selection);
        if (!code.trim()) {
            vscode.window.showWarningMessage('Selection is empty.');
            return;
        }
        const scriptPath = resolveSummarizerScript(context.extensionPath);
        if (!scriptPath) {
            vscode.window.showErrorMessage('Clone Visualizer: summarizer.py not found. Expected scripts/summarizer.py in the extension folder.');
            return;
        }
        const python = resolvePythonExecutable(context.extensionPath);
        try {
            const items = await vscode.window.withProgress({
                location: vscode.ProgressLocation.Notification,
                title: 'Summarizing code (local model)…',
                cancellable: false,
            }, () => runSummarizeProcess(python, scriptPath, code));
            summaryChannel.clear();
            items.forEach((it, i) => {
                const pct = formatSummaryPercent(it.probability);
                summaryChannel.appendLine(`${i + 1}. (${pct}%) ${it.summary}`);
            });
            summaryChannel.show(true);
            const qpItems = items.map((it, i) => {
                const pct = formatSummaryPercent(it.probability);
                return {
                    label: `${i + 1}. (${pct}%) ${it.summary}`,
                    summary: it.summary,
                };
            });
            const chosen = await vscode.window.showQuickPick(qpItems, {
                placeHolder: 'Top summaries — Enter to copy, Esc to dismiss',
                title: 'Code summaries',
            });
            if (chosen?.summary) {
                await vscode.env.clipboard.writeText(chosen.summary);
                vscode.window.showInformationMessage('Summary copied to clipboard.');
            }
        }
        catch (e) {
            const msg = e instanceof Error ? e.message : String(e);
            vscode.window.showErrorMessage(`Code summarization failed: ${msg}`);
        }
    });
    context.subscriptions.push(summarizeCmd);
    // ── Dropzone sidebar setup ────────────────────────────────────────────────
    const dropzoneProvider = new DropzoneProvider();
    const treeView = vscode.window.createTreeView('dragDropZone', {
        treeDataProvider: dropzoneProvider,
        dragAndDropController: dropzoneProvider,
    });
    const addSnippetCmd = vscode.commands.registerCommand('dragDropZone.addSnippet', () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            return;
        }
        const text = editor.document.getText(editor.selection);
        if (text) {
            dropzoneProvider.addSnippet(text);
            vscode.window.showInformationMessage('Added to Dropzone!');
        }
        else {
            vscode.window.showWarningMessage('Please highlight some text first.');
        }
    });
    const removeSnippetCmd = vscode.commands.registerCommand('dragDropZone.removeSnippet', (item) => {
        if (!(item instanceof DropItem)) {
            return;
        }
        if (dropzoneProvider.removeItem(item)) {
            vscode.window.showInformationMessage('Removed from Dropzone.');
        }
    });
    const removeSelectedCmd = vscode.commands.registerCommand('dragDropZone.removeSelected', () => {
        const selected = treeView.selection;
        let removed = 0;
        for (const node of selected) {
            if (node instanceof DropItem && dropzoneProvider.removeItem(node)) {
                removed++;
            }
        }
        if (removed > 0) {
            vscode.window.showInformationMessage(removed === 1 ? 'Removed from Dropzone.' : `Removed ${removed} snippets from Dropzone.`);
        }
    });
    const clearDropzoneCmd = vscode.commands.registerCommand('dragDropZone.clearDropzone', () => {
        if (dropzoneProvider.clear()) {
            vscode.window.showInformationMessage('Dropzone cleared.');
        }
    });
    // Shared maps: populated by show_code_clones, read by EditorDropProvider.
    // key: absolute file path → classid of the clone group opened from the tree.
    const lastOpenedByFile = new Map();
    // key: classid → CloneRecord (loaded from all_refactor_results.json on panel open).
    const recordMap = new Map();
    // key: "absPath::range" → ViewColumn — each clone instance gets its own editor column.
    const openCloneEditors = new Map();
    const editorDropProvider = vscode.languages.registerDocumentDropEditProvider({ language: '*' }, new EditorDropProvider(lastOpenedByFile, recordMap));
    context.subscriptions.push(treeView, addSnippetCmd, removeSnippetCmd, removeSelectedCmd, clearDropzoneCmd, editorDropProvider);
    // ── End Dropzone setup ────────────────────────────────────────────────────
    const log = vscode.window.createOutputChannel('Clone Visualizer — Drag Log');
    context.subscriptions.push(log);
    let disposable = vscode.commands.registerCommand('clone-visualizer.show_code_clones', () => {
        // 1. Create the Webview Panel
        const panel = vscode.window.createWebviewPanel('cloneVisualizer', 'Code Clone Tree', vscode.ViewColumn.One, {
            enableScripts: true,
            localResourceRoots: [vscode.Uri.file(path.join(context.extensionPath, 'media'))]
        });
        // 2. Inject local D3 URI
        const htmlPath = path.join(context.extensionPath, 'media', 'collapsible-tree.html');
        let htmlContent = fs.readFileSync(htmlPath, 'utf8');
        const d3Uri = panel.webview.asWebviewUri(vscode.Uri.file(path.join(context.extensionPath, 'media', 'd3.min.js')));
        htmlContent = htmlContent.replace('{{D3_URI}}', d3Uri.toString());
        panel.webview.html = htmlContent;
        // 3. Parse all_refactor_results.json → D3 tree + populate shared lookup maps
        const dataPath = path.join(context.extensionPath, 'media', 'all_refactor_results.json');
        const records = JSON.parse(fs.readFileSync(dataPath, 'utf8'));
        // Populate the shared recordMap in-place so EditorDropProvider always sees current data.
        recordMap.clear();
        openCloneEditors.clear();
        for (const r of records) {
            recordMap.set(r.classid, r);
        }
        const treeData = parseCloneData(records);
        // 4. Send tree to webview
        setTimeout(() => {
            panel.webview.postMessage({ command: 'loadData', data: treeData });
        }, 500);
        // 5. Webview message handler (click interactions)
        // Each leaf node (file + range) gets a dedicated editor column.
        // openCloneEditors (keyed by "absPath::range") is the source of truth for which column
        // belongs to which clone; a new leaf always picks the lowest unoccupied column >= 2.
        panel.webview.onDidReceiveMessage(async (message) => {
            if (message.command === 'openFile') {
                const relFile = message.file;
                const range = message.range ?? '1-1';
                const parentClassid = message.parentClassid ?? '';
                const resolved = resolvePath(relFile);
                if (!resolved) {
                    vscode.window.showWarningMessage(`Clone Visualizer: file not found — ${relFile}`);
                    return;
                }
                // Remember which clone group this file belongs to (for drag detection)
                if (parentClassid) {
                    lastOpenedByFile.set(resolved, parentClassid);
                    log.appendLine(`[openFile] ${path.basename(resolved)} → classid=${parentClassid}`);
                }
                const startLine = Math.max(0, parseInt(range.split('-')[0], 10) - 1);
                const endLine = Math.max(startLine, parseInt(range.split('-')[1] ?? range.split('-')[0], 10) - 1);
                const doc = await vscode.workspace.openTextDocument(vscode.Uri.file(resolved));
                // Each clone instance (file + range) gets its own dedicated editor column.
                const cloneKey = `${resolved}::${range}`;
                const prevColumn = openCloneEditors.get(cloneKey);
                // Check if there is still a live editor for this clone in its assigned column.
                const prevEditor = prevColumn !== undefined
                    ? vscode.window.visibleTextEditors.find(e => e.viewColumn === prevColumn &&
                        e.document.uri.fsPath === resolved)
                    : undefined;
                let targetColumn;
                if (prevEditor) {
                    // Re-navigate to the column this clone already occupies.
                    targetColumn = prevColumn;
                }
                else {
                    // Assign the lowest column (>= 2) not yet occupied by any tracked clone.
                    // Collect columns that still have a visible editor (closed groups are freed).
                    const liveColumns = new Set([...openCloneEditors.values()].filter(col => vscode.window.visibleTextEditors.some(e => e.viewColumn === col)));
                    let col = vscode.ViewColumn.Two;
                    while (liveColumns.has(col)) {
                        col++;
                    }
                    targetColumn = col;
                }
                const openedEditor = await vscode.window.showTextDocument(doc, {
                    selection: new vscode.Range(startLine, 0, endLine, Number.MAX_SAFE_INTEGER),
                    viewColumn: targetColumn,
                    preserveFocus: false,
                });
                // Record the column VS Code actually used (fall back to what we requested
                // if viewColumn is undefined — avoids leaving this clone untracked).
                openCloneEditors.set(cloneKey, openedEditor.viewColumn ?? targetColumn);
                return;
            }
            if (message.command === 'applyExtractMethod') {
                await runApplyExtractMethod(message.classid, recordMap);
            }
        }, undefined, context.subscriptions);
        // 6. Editor drag listener — detect same-file drag, match body, apply refactoring
        let ignoreChangeUntil = 0;
        const dragListener = vscode.workspace.onDidChangeTextDocument(async (event) => {
            if (Date.now() < ignoreChangeUntil) {
                return;
            }
            if (event.reason === vscode.TextDocumentChangeReason.Undo ||
                event.reason === vscode.TextDocumentChangeReason.Redo) {
                return;
            }
            const doc = event.document;
            const changes = event.contentChanges;
            // ── Diagnostic log for every supported-language file change ───────
            const isSupportedLang = ['java', 'python'].includes(doc.languageId) ||
                doc.fileName.endsWith('.java') || doc.fileName.endsWith('.py');
            if (isSupportedLang) {
                log.appendLine(`[drag] ${path.basename(doc.fileName)}  langId=${doc.languageId}` +
                    `  reason=${event.reason}  nChanges=${changes.length}`);
                changes.forEach((c, i) => {
                    log.appendLine(`  [${i}] rangeOffset=${c.rangeOffset} rangeLen=${c.rangeLength}` +
                        `  textLen=${c.text.length}  text=${JSON.stringify(c.text.slice(0, 60))}`);
                });
                log.show(true);
            }
            if (!isSupportedLang) {
                return;
            }
            if (changes.length !== 2) {
                log.appendLine(`  → SKIP: expected 2 changes, got ${changes.length}`);
                return;
            }
            const deletion = changes.find(c => c.text === '' && c.rangeLength > 0);
            const insertion = changes.find(c => c.text !== '' && c.rangeLength === 0);
            if (!deletion || !insertion) {
                log.appendLine(`  → SKIP: could not identify deletion+insertion pair`);
                return;
            }
            if (deletion.rangeLength !== insertion.text.length) {
                log.appendLine(`  → SKIP: lengths differ (del=${deletion.rangeLength} ins=${insertion.text.length})`);
                return;
            }
            const body = insertion.text;
            if (body.replace(/\s+/g, '').length < 25) {
                log.appendLine(`  → SKIP: body too short`);
                return;
            }
            // Look up clone group by the file that was opened from the tree
            const classidForFile = lastOpenedByFile.get(doc.uri.fsPath);
            const matched = classidForFile ? recordMap.get(classidForFile) : undefined;
            log.appendLine(matched
                ? `  → FILE MATCH: ${matched.classid}`
                : `  → NO MATCH for ${doc.uri.fsPath}`);
            if (!matched) {
                return;
            }
            // Only handle drag-down (most common: drop to safe space below method)
            const D = deletion.rangeOffset;
            const N = deletion.rangeLength;
            const I = insertion.rangeOffset; // pre-change coordinate
            if (I <= D + N) {
                log.appendLine(`  → SKIP: not drag-down (D=${D} N=${N} I=${I})`);
                return;
            }
            // ── Revert the drag ────────────────────────────────────────────
            //   post_drag = orig[0..D] + orig[D+N..I] + body + orig[I..]
            //   pre_drag  = post[0..D] + body + post[D..I-N] + post[I..]
            const gapStart = I - N;
            const gapEnd = I;
            const preDragText = revertDrag(doc.getText(), D, gapStart, gapEnd, body);
            ignoreChangeUntil = Date.now() + 3000;
            const revertEdit = new vscode.WorkspaceEdit();
            revertEdit.replace(doc.uri, wholeDocRange(doc), preDragText);
            const reverted = await vscode.workspace.applyEdit(revertEdit);
            if (!reverted) {
                return;
            }
            // Re-fetch the document (now in pre-drag state) and apply refactoring
            const freshDoc = await vscode.workspace.openTextDocument(doc.uri);
            const applied = await applyPrecomputedRefactoring(freshDoc, matched);
            if (applied) {
                vscode.window.showInformationMessage(`Clone Visualizer: extract method applied for ${matched.classid} — ${matched.sources.length} clone site(s) updated.`);
            }
        });
        context.subscriptions.push(dragListener);
    });
    context.subscriptions.push(disposable);
}
// ── Shared apply helper (used by both click and future callers) ────────────────
async function runApplyExtractMethod(classid, recordMap) {
    const record = recordMap.get(classid);
    if (!record) {
        return;
    }
    const filesToWrite = record.updated_files
        .map(uf => ({
        src: (() => {
            const rel = uf.rewritten_file_path.replace(/^data\/refactor_out\//, '');
            const abs = path.join(REFACTOR_OUT, rel);
            return fs.existsSync(abs) ? abs : undefined;
        })(),
        dst: resolvePath(uf.file),
        label: path.basename(uf.file),
    }))
        .filter(({ src, dst }) => !!src && !!dst);
    if (filesToWrite.length === 0) {
        vscode.window.showWarningMessage(`Clone Visualizer: no rewritten files found for ${classid}.`);
        return;
    }
    const labels = filesToWrite.map(f => f.label).join(', ');
    const answer = await vscode.window.showWarningMessage(`Apply "Extract Method" for clone group "${classid}"?`, {
        modal: true,
        detail: `${filesToWrite.length} file(s) will be modified:\n${labels}\n\nYou can use Ctrl+Z to undo after applying.`,
    }, 'Apply');
    if (answer !== 'Apply') {
        return;
    }
    // Use WorkspaceEdit so the change lands on VS Code's undo stack (Ctrl+Z works)
    const edit = new vscode.WorkspaceEdit();
    for (const { src, dst } of filesToWrite) {
        const newContent = fs.readFileSync(src, 'utf8');
        const dstUri = vscode.Uri.file(dst);
        const doc = await vscode.workspace.openTextDocument(dstUri);
        const fullRange = doc.validateRange(new vscode.Range(0, 0, doc.lineCount, 0));
        edit.replace(dstUri, fullRange, newContent);
    }
    const ok = await vscode.workspace.applyEdit(edit);
    if (!ok) {
        vscode.window.showErrorMessage(`Clone Visualizer: failed to apply edits for ${classid}.`);
        return;
    }
    // Open each modified file so the user can review (and Ctrl+Z to undo)
    for (const { dst } of filesToWrite) {
        const doc = await vscode.workspace.openTextDocument(vscode.Uri.file(dst));
        await vscode.window.showTextDocument(doc, {
            viewColumn: vscode.ViewColumn.Beside,
            preserveFocus: true,
        });
    }
    vscode.window.showInformationMessage(`✓ Extract method applied for ${classid}. Use Ctrl+Z to undo.`);
}
function deactivate() { }


/***/ }),
/* 1 */
/***/ ((module) => {

module.exports = require("vscode");

/***/ }),
/* 2 */
/***/ ((module) => {

module.exports = require("fs");

/***/ }),
/* 3 */
/***/ ((module) => {

module.exports = require("path");

/***/ }),
/* 4 */
/***/ ((module) => {

module.exports = require("child_process");

/***/ })
/******/ 	]);
/************************************************************************/
/******/ 	// The module cache
/******/ 	var __webpack_module_cache__ = {};
/******/ 	
/******/ 	// The require function
/******/ 	function __webpack_require__(moduleId) {
/******/ 		// Check if module is in cache
/******/ 		var cachedModule = __webpack_module_cache__[moduleId];
/******/ 		if (cachedModule !== undefined) {
/******/ 			return cachedModule.exports;
/******/ 		}
/******/ 		// Create a new module (and put it into the cache)
/******/ 		var module = __webpack_module_cache__[moduleId] = {
/******/ 			// no module.id needed
/******/ 			// no module.loaded needed
/******/ 			exports: {}
/******/ 		};
/******/ 	
/******/ 		// Execute the module function
/******/ 		__webpack_modules__[moduleId].call(module.exports, module, module.exports, __webpack_require__);
/******/ 	
/******/ 		// Return the exports of the module
/******/ 		return module.exports;
/******/ 	}
/******/ 	
/************************************************************************/
/******/ 	
/******/ 	// startup
/******/ 	// Load entry module and return exports
/******/ 	// This entry module is referenced by other modules so it can't be inlined
/******/ 	var __webpack_exports__ = __webpack_require__(0);
/******/ 	module.exports = __webpack_exports__;
/******/ 	
/******/ })()
;
//# sourceMappingURL=extension_clone_viz.js.map