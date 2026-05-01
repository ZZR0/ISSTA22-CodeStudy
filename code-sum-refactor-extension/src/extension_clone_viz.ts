import * as vscode from 'vscode';
import * as fs from 'fs';
import * as path from 'path';
import * as cp from 'child_process';
import * as http from 'http';
import * as https from 'https';

// ── Constants ─────────────────────────────────────────────────────────────────

// __dirname at runtime = <ext-root>/dist; one level up is the project root
const EXT_ROOT     = path.dirname(__dirname);
const REFACTOR_OUT = path.join(EXT_ROOT, 'media', 'refactor_out');

// ── Types ─────────────────────────────────────────────────────────────────────

interface EnclosingFunction {
    qualified_name: string;
    fun_range: string;
    fun_nlines: number;
    func_code: string;
}

interface ExtractedMethod {
    method_name: string;
    code: string;
}

interface CloneSource {
    func_id: string;
    file: string;
    range: string;
    nlines: number;
    code: string;
    replacement_code: string;
    enclosing_function: EnclosingFunction;
}

interface UpdatedFile {
    file: string;
    inserted_extracted_method: boolean;
    rewritten_file_path: string;
}

/** Loaded from JSON; optional fields are filled by {@link normalizeLoadedCloneRecord}. */
interface CloneRecord {
    classid: string;
    project?: string;
    inspection_case?: string;
    refactoring_type?: string;
    nclones?: number;
    same_file?: number;
    Refactorable?: number;
    sources: CloneSource[];
    updated_files?: UpdatedFile[];
    extracted_method?: ExtractedMethod;
}

interface TreeNode {
    name: string;
    classid?: string;       // set on clone-group nodes
    parentClassid?: string; // set on leaf nodes — the enclosing clone group
    file?: string;
    range?: string;
    func_id?: string;
    children?: TreeNode[];
}

// ── Parser / transformer ──────────────────────────────────────────────────────

/** Infer NiCad-style project name from classid when `project` is omitted (e.g. Python JSON). */
function inferProjectFromClassid(classid: string): string {
    const head = classid.split('_vs_')[0] ?? classid;
    return head.replace(/_\d+.*$/, '') || 'clones';
}

/** Fill defaults so Java, Python, and mixed precomputed JSON all work in the tree and maps. */
function normalizeLoadedCloneRecord(r: CloneRecord): CloneRecord {
    const sources = r.sources ?? [];
    return {
        ...r,
        project: r.project ?? inferProjectFromClassid(r.classid),
        inspection_case: r.inspection_case ?? '',
        refactoring_type: r.refactoring_type ?? 'extract_method',
        nclones: r.nclones ?? sources.length,
        same_file: r.same_file ?? 0,
        Refactorable: r.Refactorable ?? 1,
        sources,
        updated_files: r.updated_files ?? [],
        extracted_method: r.extracted_method ?? { method_name: 'extracted', code: '' },
    };
}

function parseCloneData(records: CloneRecord[]): TreeNode {
    const projectMap = new Map<string, Map<string, CloneRecord>>();
    for (const record of records) {
        const project = record.project ?? inferProjectFromClassid(record.classid);
        if (!projectMap.has(project)) {
            projectMap.set(project, new Map());
        }
        projectMap.get(project)!.set(record.classid, record);
    }

    const projectNodes: TreeNode[] = [];
    for (const [project, cloneMap] of projectMap) {
        const cloneNodes: TreeNode[] = [];
        for (const [classid, record] of cloneMap) {
            const sourceNodes: TreeNode[] = record.sources.map(src => ({
                name: `${path.basename(src.file)}  (lines ${src.range})`,
                file: src.file,
                range: src.range,
                func_id: src.func_id,
                parentClassid: classid,
            }));
            cloneNodes.push({
                name: `${classid}  [${record.refactoring_type ?? 'extract_method'} · ${record.nclones ?? record.sources.length} clones]`,
                classid,
                children: sourceNodes,
            });
        }
        projectNodes.push({ name: project, children: cloneNodes });
    }
    return { name: 'All Code Clones', children: projectNodes };
}

// ── Path helpers ──────────────────────────────────────────────────────────────

function resolvePath(relFile: string): string | undefined {
    if (path.isAbsolute(relFile) && fs.existsSync(relFile)) { return relFile; }
    const folders = vscode.workspace.workspaceFolders ?? [];
    const candidates = [
        path.join(EXT_ROOT, relFile),
        ...folders.map(f => path.join(f.uri.fsPath, relFile)),
    ];
    return candidates.find(c => fs.existsSync(c));
}

/** Precomputed bundle: `media/all_refactor_results.json` or `all_refactor_results_py.json`, or an absolute path. */
function resolveRefactorResultsDataPath(extensionPath: string): string {
    const cfg = vscode.workspace.getConfiguration('cloneVisualizer');
    const override = (cfg.get<string>('refactorResultsPath') ?? '').trim();
    if (override) {
        if (path.isAbsolute(override) && fs.existsSync(override)) { return override; }
        const inMedia = path.join(extensionPath, 'media', override);
        if (fs.existsSync(inMedia)) { return inMedia; }
        const inExt = path.join(extensionPath, override);
        if (fs.existsSync(inExt)) { return inExt; }
        void vscode.window.showWarningMessage(
            `Clone Visualizer: refactorResultsPath not found (${override}). Using default bundle.`
        );
    }
    return path.join(extensionPath, 'media', 'all_refactor_results.json');
}

type RefactorBundlePick = vscode.QuickPickItem & { fullPath: string };

/** When no `refactorResultsPath` setting: pick dataset if both Java and Python bundles ship under media/. */
async function chooseRefactorResultsJsonPath(extensionPath: string): Promise<string | undefined> {
    const cfg = vscode.workspace.getConfiguration('cloneVisualizer');
    const override = (cfg.get<string>('refactorResultsPath') ?? '').trim();
    if (override) {
        return resolveRefactorResultsDataPath(extensionPath);
    }
    const media = path.join(extensionPath, 'media');
    const javaPath = path.join(media, 'all_refactor_results.json');
    const pyPath = path.join(media, 'all_refactor_results_py.json');
    const hasJava = fs.existsSync(javaPath);
    const hasPy = fs.existsSync(pyPath);
    if (hasJava && hasPy) {
        const chosen = await vscode.window.showQuickPick<RefactorBundlePick>(
            [
                { label: 'Java clones', description: path.basename(javaPath), fullPath: javaPath },
                { label: 'Python clones', description: path.basename(pyPath), fullPath: pyPath },
            ],
            { placeHolder: 'Which clone dataset should the Code Clone Tree load?' }
        );
        return chosen?.fullPath;
    }
    if (hasPy) { return pyPath; }
    if (hasJava) { return javaPath; }
    return path.join(media, 'all_refactor_results.json');
}

/** Strip known prefixes from `updated_files[].rewritten_file_path` (Java uses data/…, Python export uses output/…). */
function relativeUnderBundledRefactorOut(rewrittenFilePath: string): string {
    const n = rewrittenFilePath.replace(/\\/g, '/');
    for (const p of ['data/refactor_out/', 'output/refactor_out/', 'refactor_out/']) {
        if (n.startsWith(p)) { return n.slice(p.length); }
    }
    return n;
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
function revertDrag(
    postDrag: string,
    deletionOffset: number,   // D  — where body was deleted (pre-change = post-drag hole position)
    gapStart: number,         // I-N — where body was inserted in post-drag coords
    gapEnd: number,           // I   — end of inserted body in post-drag coords
    body: string
): string {
    return (
        postDrag.slice(0, deletionOffset) +
        body +
        postDrag.slice(deletionOffset, gapStart) +
        postDrag.slice(gapEnd)
    );
}

function wholeDocRange(doc: vscode.TextDocument): vscode.Range {
    const last = doc.lineAt(doc.lineCount - 1);
    return new vscode.Range(0, 0, last.lineNumber, last.range.end.character);
}

// ── Code summarizer (Python / Dreamuno model) ───────────────────────────────

interface SummaryItem {
    summary: string;
    probability: number;
}

function resolveRefactorScript(extensionPath: string): string | undefined {
    const bundled = path.join(extensionPath, 'scripts', 'extract_method_refactor_python.py');
    if (fs.existsSync(bundled)) { return bundled; }
    // Development layout: sibling directory
    const sibling = path.join(extensionPath, '..', 'automate_extract_method_refactoring_py', 'extract_method_refactor_python.py');
    if (fs.existsSync(sibling)) { return sibling; }
    return undefined;
}

/**
 * Call the tree-sitter refactor script in ``direct`` mode.
 *
 * The function body is piped via stdin; all other parameters are CLI flags.
 * Returns the rewritten source string, or null on failure.
 */
function runTreeSitterDirect(
    python: string,
    scriptPath: string,
    absFilePath: string,
    signatureLine: string,
    body: string,
    callRanges: string[],
    insertLine: number,
    replacementCodes: string[],
): Promise<string | null> {
    return new Promise((resolve, reject) => {
        const args = [
            scriptPath, 'direct',
            '--file', absFilePath,
            '--signature', signatureLine,
            '--call-ranges', callRanges.join(','),
            '--insert-line', String(insertLine),
            '--replacement-codes', JSON.stringify(replacementCodes),
        ];
        const proc = cp.spawn(python, args, {
            cwd: path.dirname(scriptPath),  // sibling modules (python_treesitter_parser, util_ast_python) resolve from here
            windowsHide: true,
        });
        let out = '';
        let err = '';
        proc.stdout.setEncoding('utf8');
        proc.stderr.setEncoding('utf8');
        proc.stdout.on('data', (c: string) => { out += c; });
        proc.stderr.on('data', (c: string) => { err += c; });
        proc.on('error', e => reject(e));
        proc.on('close', (exitCode: number | null) => {
            if (exitCode !== 0) {
                reject(new Error(err.trim() || `Refactor script exited with code ${exitCode}`));
                return;
            }
            try {
                const result = JSON.parse(out.trim()) as { rewritten_source: string };
                resolve(result.rewritten_source ?? null);
            } catch {
                reject(new Error(`Invalid JSON from refactor script: ${out.trim().slice(0, 200)}`));
            }
        });
        proc.stdin.write(body, 'utf8');
        proc.stdin.end();
    });
}

function resolveSummarizerScript(extensionPath: string): string | undefined {
    const cfg = vscode.workspace.getConfiguration('cloneVisualizer');
    const override = (cfg.get<string>('summarizerScriptPath') ?? '').trim();
    if (override && fs.existsSync(override)) { return override; }
    const bundled = path.join(extensionPath, 'scripts', 'summarizer.py');
    if (fs.existsSync(bundled)) { return bundled; }
    const sibling = path.join(extensionPath, '..', 'code-summarizer', 'summarizer.py');
    if (fs.existsSync(sibling)) { return sibling; }
    return undefined;
}

function resolvePythonExecutable(extensionPath: string): string {
    const cfg = vscode.workspace.getConfiguration('cloneVisualizer');
    const configured = (cfg.get<string>('pythonPath') ?? '').trim();
    if (configured) { return configured; }
    const venvDir = path.join(extensionPath, '..', 'code-summarizer', '.venv');
    const winPy  = path.join(venvDir, 'Scripts', 'python.exe');
    const unixPy = path.join(venvDir, 'bin', 'python');
    if (fs.existsSync(winPy)) { return winPy; }
    if (fs.existsSync(unixPy)) { return unixPy; }
    return process.platform === 'win32' ? 'python' : 'python3';
}

const HF_SUMMARY_JAVA = 'Dreamuno/llm_codeknowhow_v1';
const HF_SUMMARY_PYTHON = 'Dreamuno/llm-codeknowhow-python';

function isPythonDocument(doc: vscode.TextDocument): boolean {
    return doc.languageId === 'python' || doc.fileName.toLowerCase().endsWith('.py');
}

type SummarizerRoute = 'java' | 'python';

/**
 * Python model if the buffer is Python or the selection looks like Python; otherwise Java model.
 * (No manual pick — fully automatic.)
 */
function selectionLooksLikePython(code: string): boolean {
    const s = code.trim();
    if (s.length < 6) { return false; }
    if (/\bfrom\s+[\w.]+\s+import\s+/.test(s)) { return true; }
    if (/\bdef\s+[A-Za-z_]\w*\s*\(/.test(s)) { return true; }
    if (/\basync\s+def\s+[A-Za-z_]\w*\s*\(/.test(s)) { return true; }
    if (/\belif\b/.test(s)) { return true; }
    if (/\bif\s+__name__\s*==\s*['"]__main__['"]/.test(s)) { return true; }
    if (/\bexcept\s+[\w.]+\s+as\s+\w+/.test(s)) { return true; }
    return false;
}

function resolveSummarizerRoute(doc: vscode.TextDocument, selectedCode: string): SummarizerRoute {
    if (isPythonDocument(doc) || selectionLooksLikePython(selectedCode)) {
        return 'python';
    }
    return 'java';
}

/** Hugging Face model id for the chosen route (both models are configured independently). */
function resolveSummarizerModelId(doc: vscode.TextDocument, selectedCode: string, route?: SummarizerRoute): string {
    const cfg = vscode.workspace.getConfiguration('cloneVisualizer');
    const pyModel = (cfg.get<string>('summarizerModelPython') ?? '').trim();
    const javaModel =
        (cfg.get<string>('summarizerModelJava') ?? '').trim() ||
        (cfg.get<string>('summarizerModelDefault') ?? '').trim(); // legacy setting id
    const r = route ?? resolveSummarizerRoute(doc, selectedCode);
    if (r === 'python') {
        return pyModel || HF_SUMMARY_PYTHON;
    }
    return javaModel || HF_SUMMARY_JAVA;
}

function summarizerProgressTitle(route: SummarizerRoute, modelId: string): string {
    if (route === 'python') {
        return `Summarizing Python (${modelId})…`;
    }
    return `Summarizing Java (${modelId})…`;
}

/** Percent for display, e.g. 12.7 or 9.07 (trims trailing zeros). */
function formatSummaryPercent(probability: number): string {
    const pct = probability * 100;
    let s = pct.toFixed(2);
    s = s.replace(/0+$/, '');
    s = s.replace(/\.$/, '');
    return s || '0';
}

function runSummarizeProcess(python: string, scriptPath: string, code: string, modelId: string): Promise<SummaryItem[]> {
    return new Promise((resolve, reject) => {
        const proc = cp.spawn(python, [scriptPath, '--json', '--stdin', '--model', modelId], {
            windowsHide: true,
        });
        let out = '';
        let err = '';
        proc.stdout.setEncoding('utf8');
        proc.stderr.setEncoding('utf8');
        proc.stdout.on('data', (c: string) => { out += c; });
        proc.stderr.on('data', (c: string) => { err += c; });
        proc.on('error', e => reject(e));
        proc.on('close', (exitCode, _sig) => {
            if (exitCode !== 0) {
                reject(new Error(err.trim() || `Summarizer exited with code ${exitCode}`));
                return;
            }
            try {
                const parsed = JSON.parse(out.trim()) as SummaryItem[];
                resolve(parsed);
            } catch {
                reject(new Error(out.trim() || err.trim() || 'Empty output from summarizer'));
            }
        });
        proc.stdin.write(code, 'utf8');
        proc.stdin.end();
    });
}

// ── REST client helpers ───────────────────────────────────────────────────────

function resolveServerUrl(): string {
    const cfg = vscode.workspace.getConfiguration('cloneVisualizer');
    return (cfg.get<string>('serverUrl') ?? 'http://127.0.0.1:5000').replace(/\/+$/, '');
}

/**
 * Generic HTTP/HTTPS POST that sends JSON and resolves with the parsed response body.
 * Rejects on network error, non-2xx status, or JSON parse failure.
 */
function httpPost<T>(urlStr: string, body: unknown, timeoutMs = 30_000): Promise<T> {
    return new Promise((resolve, reject) => {
        let url: URL;
        try { url = new URL(urlStr); } catch (e) { reject(e); return; }

        const payload = JSON.stringify(body);
        const options: http.RequestOptions = {
            hostname: url.hostname,
            port: url.port || (url.protocol === 'https:' ? 443 : 80),
            path: url.pathname + url.search,
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Content-Length': Buffer.byteLength(payload),
            },
            timeout: timeoutMs,
        };

        const lib = url.protocol === 'https:' ? https : http;
        const req = lib.request(options, (res: http.IncomingMessage) => {
            let data = '';
            res.setEncoding('utf8');
            res.on('data', (c: string) => { data += c; });
            res.on('end', () => {
                const status = res.statusCode ?? 0;
                if (status < 200 || status >= 300) {
                    reject(new Error(`HTTP ${status}: ${data.slice(0, 200)}`));
                    return;
                }
                try { resolve(JSON.parse(data) as T); }
                catch { reject(new Error(`Invalid JSON from server: ${data.slice(0, 200)}`)); }
            });
        });
        req.on('error', reject);
        req.on('timeout', () => { req.destroy(); reject(new Error('Request timed out')); });
        req.write(payload);
        req.end();
    });
}

/** Returns true if the server is reachable (GET /health returns 200). */
function checkServerHealth(baseUrl: string): Promise<boolean> {
    return new Promise((resolve) => {
        let url: URL;
        try { url = new URL(baseUrl + '/health'); } catch { resolve(false); return; }

        const lib = url.protocol === 'https:' ? https : http;
        const options: http.RequestOptions = {
            hostname: url.hostname,
            port: url.port || (url.protocol === 'https:' ? 443 : 80),
            path: url.pathname,
            method: 'GET',
            timeout: 2_000,
        };
        const req = lib.request(options, (res: http.IncomingMessage) => {
            res.resume();
            resolve((res.statusCode ?? 0) === 200);
        });
        req.on('error', () => resolve(false));
        req.on('timeout', () => { req.destroy(); resolve(false); });
        req.end();
    });
}

/** POST /summarize — sorted by probability descending. */
async function apiSummarize(baseUrl: string, code: string, modelId: string): Promise<SummaryItem[]> {
    const results = await httpPost<SummaryItem[]>(
        baseUrl + '/summarize',
        { code, model: modelId },
        60_000,
    );
    return results.sort((a, b) => b.probability - a.probability);
}

/** POST /refactor/direct — returns the rewritten source, or null on failure. */
async function apiRefactorDirect(
    baseUrl: string,
    absFilePath: string,
    signatureLine: string,
    body: string,
    callRanges: string[],
    insertLine: number,
    replacementCodes: string[],
): Promise<string | null> {
    const resp = await httpPost<{ rewritten_source: string }>(
        baseUrl + '/refactor/direct',
        {
            file: absFilePath,
            signature: signatureLine,
            body,
            call_ranges: callRanges,
            insert_line: insertLine,
            replacement_codes: replacementCodes,
        },
        60_000,
    );
    return resp.rewritten_source ?? null;
}

// ── Apply pre-computed WorkspaceEdit ─────────────────────────────────────────

/** Re-indent a block (e.g. precomputed `def extracted...`) to `outerIndent` while keeping internal relative indents. */
function reindentBlockPreservingRelativeStructure(code: string, outerIndent: string): string {
    const lines = code.split('\n');
    let minLead = Infinity;
    for (const line of lines) {
        if (!line.trim()) { continue; }
        const m = /^(\s*)/.exec(line);
        const n = m ? m[1].length : 0;
        if (n < minLead) { minLead = n; }
    }
    if (minLead === Infinity) { minLead = 0; }
    return lines
        .map(line => {
            if (!line.trim()) { return ''; }
            const m = /^(\s*)/.exec(line)!;
            const lead = m[1].length;
            const rel = Math.max(0, lead - minLead);
            return outerIndent + ' '.repeat(rel) + line.slice(lead);
        })
        .join('\n');
}

async function applyPrecomputedRefactoring(
    doc: vscode.TextDocument,
    record: CloneRecord
): Promise<boolean> {
    const edit = new vscode.WorkspaceEdit();

    // Sort sources descending by start line so replacements don't shift each other
    const sorted = [...record.sources].sort((a, b) => {
        const aLine = parseInt(a.range.split('-')[0], 10);
        const bLine = parseInt(b.range.split('-')[0], 10);
        return bLine - aLine;
    });

    for (const src of sorted) {
        if (!src.replacement_code) { continue; }
        const m = /^(\d+)-(\d+)$/.exec(src.range.trim());
        if (!m) { continue; }
        const start0 = parseInt(m[1], 10) - 1;
        const end0   = parseInt(m[2], 10) - 1;
        if (start0 < 0 || end0 >= doc.lineCount || start0 > end0) { continue; }

        edit.replace(
            doc.uri,
            new vscode.Range(
                new vscode.Position(start0, 0),
                new vscode.Position(end0, doc.lineAt(end0).text.length)
            ),
            src.replacement_code
        );
    }

    // Insert extracted method after source[0]'s enclosing function:
    // Java: after closing `}` — use that line's indent (class member level).
    // Python: after last body line — use the enclosing `def` line's indent so the new `def` is a sibling, not nested after `return`.
    const em  = record.extracted_method;
    const src0 = record.sources[0];
    if (em?.code && src0?.enclosing_function?.fun_range) {
        const fm = /^(\d+)-(\d+)$/.exec(src0.enclosing_function.fun_range.trim());
        if (fm) {
            const encStart0 = parseInt(fm[1], 10) - 1;
            const encEnd0 = parseInt(fm[2], 10) - 1;
            if (encEnd0 >= 0 && encEnd0 < doc.lineCount && encStart0 >= 0 && encStart0 < doc.lineCount) {
                const closingLine = doc.lineAt(encEnd0).text;
                const defLine = doc.lineAt(encStart0).text;
                const defIndentM = /^(\s*)/.exec(defLine);
                const closingIndentM = /^(\s*)/.exec(closingLine);
                const defLineIndent = defIndentM ? defIndentM[1] : '';
                const closingLineIndent = closingIndentM ? closingIndentM[1] : '    ';

                const methodCode = isPythonDocument(doc)
                    ? reindentBlockPreservingRelativeStructure(em.code, defLineIndent)
                    : em.code
                        .split('\n')
                        .map(l => (l.trim() ? closingLineIndent + l : l))
                        .join('\n');

                edit.insert(
                    doc.uri,
                    new vscode.Position(encEnd0, closingLine.length),
                    '\n\n' + methodCode
                );
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
    constructor(public readonly label: string, public readonly content: string) {
        super(label, vscode.TreeItemCollapsibleState.None);
        this.tooltip = content;
        this.description = '';
        this.contextValue = 'dropzoneSnippet';
    }
}

/**
 * Provides the data for the Dropzone TreeView and handles drag/drop interactions
 * originating from the sidebar panel.
 */
class DropzoneProvider implements vscode.TreeDataProvider<DropItem>, vscode.TreeDragAndDropController<DropItem> {
    private dropItems: DropItem[] = [];

    private _onDidChangeTreeData = new vscode.EventEmitter<DropItem | undefined | void>();
    readonly onDidChangeTreeData = this._onDidChangeTreeData.event;

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

    getTreeItem(element: DropItem): vscode.TreeItem { return element; }

    getChildren(element?: DropItem): vscode.ProviderResult<DropItem[]> {
        return element ? [] : this.dropItems;
    }

    public addSnippet(textContent: string): void {
        const label = textContent.trim().substring(0, 20).replace(/\n/g, ' ') + '...';
        this.dropItems.push(new DropItem(label, textContent));
        this._onDidChangeTreeData.fire();
    }

    public removeItem(item: DropItem): boolean {
        const i = this.dropItems.indexOf(item);
        if (i < 0) { return false; }
        this.dropItems.splice(i, 1);
        this._onDidChangeTreeData.fire();
        return true;
    }

    public clear(): boolean {
        if (this.dropItems.length === 0) { return false; }
        this.dropItems.length = 0;
        this._onDidChangeTreeData.fire();
        return true;
    }

    async handleDrop(_target: DropItem | undefined, dataTransfer: vscode.DataTransfer, token: vscode.CancellationToken): Promise<void> {
        if (token.isCancellationRequested) { return; }

        let content = await this.readDroppedText(dataTransfer, token);
        if (!content?.trim()) { content = await this.readSnippetFromEditorUriList(dataTransfer, token); }
        if (!content?.trim()) { content = await this.readSnippetFromDownloadUrl(dataTransfer, token); }
        if (!content?.trim()) {
            vscode.window.showWarningMessage(
                'Dropzone could not read that drag. Use ⌘⇧R / Ctrl+Shift+R with a selection, or copy text and run "Add to Dropzone".'
            );
            return;
        }
        this.addSnippet(content);
        vscode.window.showInformationMessage('Added to Dropzone (drag).');
    }

    private async readDroppedText(dataTransfer: vscode.DataTransfer, token: vscode.CancellationToken): Promise<string | undefined> {
        let best: string | undefined;
        const consider = async (mimeType: string, item: vscode.DataTransferItem) => {
            if (token.isCancellationRequested) { return; }
            const mt = mimeType.toLowerCase();
            if (mt.startsWith('application/vnd.code.tree.') || mt === 'text/uri-list' || mt === 'downloadurl') { return; }
            let s: string | undefined;
            try { s = await item.asString(); } catch { s = typeof item.value === 'string' ? item.value : undefined; }
            if (!s?.trim()) { return; }
            if (mt === 'text/plain' || mt === 'public.utf8-plain-text' || mt === 'public.plain-text') { best = s; return; }
            if (!best || s.trim().length > best.trim().length) { best = s; }
        };

        const plain = dataTransfer.get('text/plain');
        if (plain) {
            await consider('text/plain', plain);
            if (best) { return best; }
        }
        for (const [mimeType, item] of dataTransfer) {
            if (token.isCancellationRequested) { break; }
            await consider(mimeType, item);
        }
        return best;
    }

    private async readSnippetFromEditorUriList(dataTransfer: vscode.DataTransfer, token: vscode.CancellationToken): Promise<string | undefined> {
        const items: vscode.DataTransferItem[] = [];
        for (const [mime, item] of dataTransfer) {
            if (mime.toLowerCase() === 'text/uri-list') { items.push(item); }
        }
        for (const item of items) {
            if (token.isCancellationRequested) { return undefined; }
            let raw: string | undefined;
            try { raw = await item.asString(); } catch { raw = typeof item.value === 'string' ? item.value : undefined; }
            if (!raw?.trim()) { continue; }
            const lines = raw.split(/\r?\n/).map(l => l.trim()).filter(l => l.length > 0 && !l.startsWith('#'));
            for (const line of lines) {
                const text = await this.snippetFromUriListLine(line, token);
                if (text?.trim()) { return text; }
            }
        }
        return undefined;
    }

    private async readSnippetFromDownloadUrl(dataTransfer: vscode.DataTransfer, token: vscode.CancellationToken): Promise<string | undefined> {
        const item = dataTransfer.get('downloadurl');
        if (!item) { return undefined; }
        let raw: string | undefined;
        try { raw = await item.asString(); } catch { raw = typeof item.value === 'string' ? item.value : undefined; }
        if (!raw) { return undefined; }
        const a = raw.indexOf(':');
        const b = a >= 0 ? raw.indexOf(':', a + 1) : -1;
        if (b < 0) { return undefined; }
        return this.snippetFromUriListLine(raw.slice(b + 1).trim(), token);
    }

    private async snippetFromUriListLine(line: string, token: vscode.CancellationToken): Promise<string | undefined> {
        let uri: vscode.Uri;
        try { uri = vscode.Uri.parse(line, true); } catch {
            try { uri = vscode.Uri.file(line); } catch { return undefined; }
        }
        if (token.isCancellationRequested) { return undefined; }

        const fragRaw = uri.fragment;
        let frag = fragRaw;
        try { frag = decodeURIComponent(fragRaw); } catch { /* keep raw */ }

        const parsed = DropzoneProvider.parseLinkFragment(frag);
        const clean = uri.with({ fragment: '' });
        const docFromBuffer = DropzoneProvider.findOpenDocument(clean);

        if (docFromBuffer) {
            if (!parsed) {
                const ed = vscode.window.visibleTextEditors.find(e => e.document === docFromBuffer);
                if (ed && !ed.selection.isEmpty) { return ed.document.getText(ed.selection); }
                return DropzoneProvider.textFromEditorSelectionForUri(clean);
            }
            return docFromBuffer.getText(DropzoneProvider.rangeForOpenDocument(docFromBuffer, parsed));
        }
        if (!parsed) { return DropzoneProvider.textFromEditorSelectionForUri(clean); }
        try {
            const doc = await vscode.workspace.openTextDocument(clean);
            return doc.getText(DropzoneProvider.rangeForOpenDocument(doc, parsed));
        } catch { return DropzoneProvider.textFromEditorSelectionForUri(clean); }
    }

    private static findOpenDocument(clean: vscode.Uri): vscode.TextDocument | undefined {
        return vscode.workspace.textDocuments.find(d =>
            d.uri.toString() === clean.toString() ||
            (d.uri.scheme === 'file' && clean.scheme === 'file' && d.uri.fsPath === clean.fsPath)
        );
    }

    private static textFromEditorSelectionForUri(clean: vscode.Uri): string | undefined {
        const active = vscode.window.activeTextEditor;
        if (!active || active.selection.isEmpty) { return undefined; }
        const du = active.document.uri;
        if (du.toString() === clean.toString() || (du.scheme === 'file' && clean.scheme === 'file' && du.fsPath === clean.fsPath)) {
            return active.document.getText(active.selection);
        }
        return undefined;
    }

    private static parseLinkFragment(fragment: string):
        | { kind: 'wholeLine'; line1: number }
        | { kind: 'range'; startLine1: number; startCol1: number; endLine1: number; endCol1?: number }
        | undefined {
        const match = /^L?(\d+)(?:,(\d+))?(-L?(\d+)(?:,(\d+))?)?/.exec(fragment);
        if (!match) { return undefined; }
        const startLine = parseInt(match[1], 10);
        const startCol  = match[2] ? parseInt(match[2], 10) : 1;
        if (!match[4]) { return { kind: 'wholeLine', line1: startLine }; }
        const endLine = parseInt(match[4], 10);
        const endCol  = match[5] ? parseInt(match[5], 10) : undefined;
        return { kind: 'range', startLine1: startLine, startCol1: startCol, endLine1: endLine, endCol1: endCol };
    }

    private static rangeForOpenDocument(
        doc: vscode.TextDocument,
        parsed: Exclude<ReturnType<typeof DropzoneProvider.parseLinkFragment>, undefined>
    ): vscode.Range {
        if (parsed.kind === 'wholeLine') {
            const lineIdx = Math.min(Math.max(0, parsed.line1 - 1), doc.lineCount - 1);
            return doc.lineAt(lineIdx).range;
        }
        const sl = Math.min(Math.max(0, parsed.startLine1 - 1), doc.lineCount - 1);
        const el = Math.min(Math.max(0, parsed.endLine1 - 1), doc.lineCount - 1);
        const startLineDoc = doc.lineAt(sl);
        const endLineDoc   = doc.lineAt(el);
        const startChar = Math.min(Math.max(0, parsed.startCol1 - 1), startLineDoc.text.length);
        const endChar   = parsed.endCol1 !== undefined
            ? Math.min(Math.max(0, parsed.endCol1 - 1), endLineDoc.text.length)
            : endLineDoc.text.length;
        return new vscode.Range(new vscode.Position(sl, startChar), new vscode.Position(el, endChar));
    }

    async handleDrag(source: readonly DropItem[], dataTransfer: vscode.DataTransfer, token: vscode.CancellationToken): Promise<void> {
        if (source.length === 0 || token.isCancellationRequested) { return; }
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
function normalizeBodyLines(body: string, bodyIndent: string): string[] {
    const lines = body.split('\n');
    // Drop trailing blank lines
    while (lines.length > 0 && lines[lines.length - 1].trim() === '') { lines.pop(); }
    const nonEmpty = lines.filter(l => l.trim().length > 0);
    const minIndent = nonEmpty.length === 0
        ? 0
        : Math.min(...nonEmpty.map(l => (/^(\s*)/.exec(l)![1].length)));
    return lines.map(l => bodyIndent + l.slice(minIndent));
}

/**
 * Wraps `body` inside a method/function definition appropriate for the given language.
 * `outerIndent` is the leading whitespace of the line where the user dropped the snippet.
 */
function wrapInMethod(body: string, methodName: string, lang: string, outerIndent: string): string {
    const step = '    ';
    const bodyIndent = outerIndent + step;
    const bodyLines = normalizeBodyLines(body, bodyIndent);

    if (lang === 'python') {
        // Python uses indentation instead of braces
        return `${outerIndent}def ${methodName}():\n${bodyLines.join('\n')}`;
    }

    const header = lang === 'java'
        ? `private void ${methodName}()`
        : `function ${methodName}()`;   // TypeScript / JavaScript / fallback

    return `${outerIndent}${header} {\n${bodyLines.join('\n')}\n${outerIndent}}`;
}

/**
 * Returns the leading-whitespace indentation to use for the generated method.
 * Prefers the drop-position line; falls back to the nearest preceding non-blank line.
 */
function indentAtDropPosition(document: vscode.TextDocument, position: vscode.Position): string {
    const tryLine = (lineIdx: number): string | undefined => {
        if (lineIdx < 0 || lineIdx >= document.lineCount) { return undefined; }
        const text = document.lineAt(lineIdx).text;
        const m = /^(\s*)/.exec(text);
        return m ? m[1] : '';
    };

    const dropLineIndent = tryLine(position.line) ?? '';
    if (dropLineIndent.length > 0) { return dropLineIndent; }

    // Drop line is empty — walk backwards to find a non-blank line
    for (let i = position.line - 1; i >= 0; i--) {
        const text = document.lineAt(i).text;
        if (text.trim().length > 0) {
            return (/^(\s*)/.exec(text)![1]);
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
class EditorDropProvider implements vscode.DocumentDropEditProvider {
    constructor(
        private readonly lastOpenedByFile: Map<string, string>,
        private readonly recordMap: Map<string, CloneRecord>
    ) {}

    async provideDocumentDropEdits(
        document: vscode.TextDocument,
        position: vscode.Position,
        dataTransfer: vscode.DataTransfer,
        token: vscode.CancellationToken
    ): Promise<vscode.DocumentDropEdit | undefined> {
        const dropzoneItem = dataTransfer.get('application/vnd.drag.dropzone');
        if (!dropzoneItem) { return undefined; }

        const content = await dropzoneItem.asString();
        if (!content?.trim()) { return undefined; }

        // ── Path 1: clone-aware ──────────────────────────────────────────────
        // The file was opened by clicking a source node in the clone tree,
        // so we know which clone group it belongs to.
        const classid = this.lastOpenedByFile.get(document.uri.fsPath);
        const record  = classid ? this.recordMap.get(classid) : undefined;

        if (record) {
            const answer = await vscode.window.showWarningMessage(
                `Apply "Extract Method" for clone group "${record.classid}"?`,
                {
                    modal: true,
                    detail:
                        `${record.sources.length} clone site(s) will be updated together.\n` +
                        `Use Ctrl+Z / ⌘Z to undo.`,
                },
                'Apply'
            );
            if (answer !== 'Apply' || token.isCancellationRequested) { return undefined; }

            // Let VS Code finish applying the no-op drop edit before the workspace edit runs.
            setTimeout(async () => {
                const freshDoc = await vscode.workspace.openTextDocument(document.uri);
                const applied  = await applyPrecomputedRefactoring(freshDoc, record);
                if (applied) {
                    vscode.window.showInformationMessage(
                        `Clone Visualizer: extract method applied for ${record.classid} ` +
                        `— ${record.sources.length} clone site(s) updated.`
                    );
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

        if (name === undefined || token.isCancellationRequested) { return undefined; }

        const methodName  = name.trim() || 'extractedMethod';
        const outerIndent = indentAtDropPosition(document, position);
        const wrapped     = wrapInMethod(content, methodName, document.languageId, outerIndent);

        vscode.window.showInformationMessage(`Snippet wrapped in ${methodName}() and inserted.`);
        return new vscode.DocumentDropEdit(wrapped);
    }
}

// ── Extension entry point ─────────────────────────────────────────────────────

export function activate(context: vscode.ExtensionContext) {
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
        const python = resolvePythonExecutable(context.extensionPath);
        const route = resolveSummarizerRoute(editor.document, code);
        const modelId = resolveSummarizerModelId(editor.document, code, route);
        const serverUrl = resolveServerUrl();

        try {
            const items = await vscode.window.withProgress(
                {
                    location: vscode.ProgressLocation.Notification,
                    title: summarizerProgressTitle(route, modelId),
                    cancellable: false,
                },
                async () => {
                    const serverUp = await checkServerHealth(serverUrl);
                    if (serverUp) {
                        log.appendLine(`[summarize] using server ${serverUrl}`);
                        return apiSummarize(serverUrl, code, modelId);
                    }
                    if (!scriptPath) {
                        throw new Error(
                            'Server is unreachable and summarizer.py not found. ' +
                            'Start scripts/server.py or check scripts/summarizer.py.'
                        );
                    }
                    log.appendLine('[summarize] server unreachable — using subprocess');
                    return runSummarizeProcess(python, scriptPath, code, modelId);
                },
            );

            summaryChannel.clear();
            summaryChannel.appendLine(`Route: ${route}  |  Model: ${modelId}`);
            items.forEach((it, i) => {
                const pct = formatSummaryPercent(it.probability);
                summaryChannel.appendLine(`${i + 1}. (${pct}%) ${it.summary}`);
            });
            summaryChannel.show(true);

            const qpItems: (vscode.QuickPickItem & { summary: string })[] = items.map((it, i) => {
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
        } catch (e) {
            const msg = e instanceof Error ? e.message : String(e);
            if (/protobuf/i.test(msg)) {
                summaryChannel.appendLine(msg);
                summaryChannel.show(true);
                void vscode.window.showErrorMessage(
                    'Code summarization failed: the Python env is missing protobuf (needed to load the model). ' +
                    'Run: python3 -m pip install protobuf   — or: pip install -r scripts/requirements-summarizer.txt'
                );
            } else {
                void vscode.window.showErrorMessage(`Code summarization failed: ${msg}`);
            }
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
        if (!editor) { return; }
        const text = editor.document.getText(editor.selection);
        if (text) {
            dropzoneProvider.addSnippet(text);
            vscode.window.showInformationMessage('Added to Dropzone!');
        } else {
            vscode.window.showWarningMessage('Please highlight some text first.');
        }
    });

    const removeSnippetCmd = vscode.commands.registerCommand('dragDropZone.removeSnippet', (item: DropItem) => {
        if (!(item instanceof DropItem)) { return; }
        if (dropzoneProvider.removeItem(item)) {
            vscode.window.showInformationMessage('Removed from Dropzone.');
        }
    });

    const removeSelectedCmd = vscode.commands.registerCommand('dragDropZone.removeSelected', () => {
        const selected = treeView.selection;
        let removed = 0;
        for (const node of selected) {
            if (node instanceof DropItem && dropzoneProvider.removeItem(node)) { removed++; }
        }
        if (removed > 0) {
            vscode.window.showInformationMessage(
                removed === 1 ? 'Removed from Dropzone.' : `Removed ${removed} snippets from Dropzone.`
            );
        }
    });

    const clearDropzoneCmd = vscode.commands.registerCommand('dragDropZone.clearDropzone', () => {
        if (dropzoneProvider.clear()) { vscode.window.showInformationMessage('Dropzone cleared.'); }
    });

    // Shared maps: populated by show_code_clones, read by EditorDropProvider.
    // key: absolute file path → classid of the clone group opened from the tree.
    const lastOpenedByFile = new Map<string, string>();
    // key: classid → CloneRecord (loaded from all_refactor_results.json on panel open).
    const recordMap        = new Map<string, CloneRecord>();
    // key: "absPath::range" → ViewColumn — each clone instance gets its own editor column.
    const openCloneEditors = new Map<string, vscode.ViewColumn>();

    const editorDropProvider = vscode.languages.registerDocumentDropEditProvider(
        { language: '*' },
        new EditorDropProvider(lastOpenedByFile, recordMap)
    );

    context.subscriptions.push(treeView, addSnippetCmd, removeSnippetCmd, removeSelectedCmd, clearDropzoneCmd, editorDropProvider);
    // ── End Dropzone setup ────────────────────────────────────────────────────

    const log = vscode.window.createOutputChannel('Clone Visualizer — Drag Log');
    context.subscriptions.push(log);

    let disposable = vscode.commands.registerCommand('clone-visualizer.show_code_clones', async () => {

        const dataPath = await chooseRefactorResultsJsonPath(context.extensionPath);
        if (!dataPath) {
            return;
        }
        if (!fs.existsSync(dataPath)) {
            void vscode.window.showErrorMessage(`Clone Visualizer: clone data not found — ${dataPath}`);
            return;
        }

        // 1. Create the Webview Panel
        const panelTitle =
            path.basename(dataPath).includes('_py') ? 'Code Clone Tree (Python)' : 'Code Clone Tree';
        const panel = vscode.window.createWebviewPanel(
            'cloneVisualizer', panelTitle, vscode.ViewColumn.One,
            {
                enableScripts: true,
                localResourceRoots: [vscode.Uri.file(path.join(context.extensionPath, 'media'))]
            }
        );

        // 2. Inject local D3 URI
        const htmlPath = path.join(context.extensionPath, 'media', 'collapsible-tree.html');
        let htmlContent = fs.readFileSync(htmlPath, 'utf8');
        const d3Uri = panel.webview.asWebviewUri(
            vscode.Uri.file(path.join(context.extensionPath, 'media', 'd3.min.js'))
        );
        htmlContent = htmlContent.replace('{{D3_URI}}', d3Uri.toString());
        panel.webview.html = htmlContent;

        // 3. Parse refactor results JSON → D3 tree + populate shared lookup maps
        const rawRecords = JSON.parse(fs.readFileSync(dataPath, 'utf8')) as CloneRecord[];
        const records = rawRecords.map(normalizeLoadedCloneRecord);
        // Populate the shared recordMap in-place so EditorDropProvider always sees current data.
        recordMap.clear();
        openCloneEditors.clear();
        for (const r of records) { recordMap.set(r.classid, r); }
        const treeData = parseCloneData(records);

        // 4. Send tree to webview
        setTimeout(() => {
            panel.webview.postMessage({ command: 'loadData', data: treeData });
        }, 500);

        // 5. Webview message handler (click interactions)
        // Each leaf node (file + range) gets a dedicated editor column.
        // openCloneEditors (keyed by "absPath::range") is the source of truth for which column
        // belongs to which clone; a new leaf always picks the lowest unoccupied column >= 2.

        panel.webview.onDidReceiveMessage(
            async message => {

                if (message.command === 'openFile') {
                    const relFile: string       = message.file;
                    const range: string         = message.range ?? '1-1';
                    const parentClassid: string = message.parentClassid ?? '';
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
                    const endLine   = Math.max(startLine, parseInt(range.split('-')[1] ?? range.split('-')[0], 10) - 1);
                    const doc = await vscode.workspace.openTextDocument(vscode.Uri.file(resolved));

                    // Each clone instance (file + range) gets its own dedicated editor column.
                    const cloneKey   = `${resolved}::${range}`;
                    const prevColumn = openCloneEditors.get(cloneKey);

                    // Check if there is still a live editor for this clone in its assigned column.
                    const prevEditor = prevColumn !== undefined
                        ? vscode.window.visibleTextEditors.find(
                            e => (e.viewColumn as number) === (prevColumn as number) &&
                                 e.document.uri.fsPath === resolved)
                        : undefined;

                    let targetColumn: vscode.ViewColumn;
                    if (prevEditor) {
                        // Re-navigate to the column this clone already occupies.
                        targetColumn = prevColumn!;
                    } else {
                        // Assign the lowest column (>= 2) not yet occupied by any tracked clone.
                        // Collect columns that still have a visible editor (closed groups are freed).
                        const liveColumns = new Set(
                            [...openCloneEditors.values()].filter(col =>
                                vscode.window.visibleTextEditors.some(
                                    e => (e.viewColumn as number) === (col as number)))
                        );
                        let col = vscode.ViewColumn.Two as number;
                        while (liveColumns.has(col as vscode.ViewColumn)) { col++; }
                        targetColumn = col as vscode.ViewColumn;
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
            },
            undefined,
            context.subscriptions
        );

        // 6. Editor drag listener — detect same-file drag, match body, apply refactoring
        let ignoreChangeUntil = 0;

        const dragListener = vscode.workspace.onDidChangeTextDocument(async event => {
            if (Date.now() < ignoreChangeUntil) { return; }
            if (
                event.reason === vscode.TextDocumentChangeReason.Undo ||
                event.reason === vscode.TextDocumentChangeReason.Redo
            ) { return; }

            const doc = event.document;
            const changes = event.contentChanges;

            // ── Diagnostic log for every supported-language file change ───────
            const isSupportedLang = ['java', 'python'].includes(doc.languageId) ||
                doc.fileName.endsWith('.java') || doc.fileName.endsWith('.py');
            if (isSupportedLang) {
                log.appendLine(
                    `[drag] ${path.basename(doc.fileName)}  langId=${doc.languageId}` +
                    `  reason=${event.reason}  nChanges=${changes.length}`
                );
                changes.forEach((c, i) => {
                    log.appendLine(
                        `  [${i}] rangeOffset=${c.rangeOffset} rangeLen=${c.rangeLength}` +
                        `  textLen=${c.text.length}  text=${JSON.stringify(c.text.slice(0, 60))}`
                    );
                });
                log.show(true);
            }

            if (!isSupportedLang) { return; }

            if (changes.length !== 2) {
                log.appendLine(`  → SKIP: expected 2 changes, got ${changes.length}`);
                return;
            }

            const deletion  = changes.find(c => c.text === '' && c.rangeLength > 0);
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
            log.appendLine(
                matched
                    ? `  → FILE MATCH: ${matched.classid}`
                    : `  → NO MATCH for ${doc.uri.fsPath}`
            );
            if (!matched) { return; }

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
            const gapStart   = I - N;
            const gapEnd     = I;
            const preDragText = revertDrag(doc.getText(), D, gapStart, gapEnd, body);

            ignoreChangeUntil = Date.now() + 3000;

            const revertEdit = new vscode.WorkspaceEdit();
            revertEdit.replace(doc.uri, wholeDocRange(doc), preDragText);
            const reverted = await vscode.workspace.applyEdit(revertEdit);
            if (!reverted) { return; }

            // Re-fetch the document (now in pre-drag state) and apply refactoring.
            // Prefer the tree-sitter dynamic path; fall back to pre-computed on failure.
            const freshDoc = await vscode.workspace.openTextDocument(doc.uri);

            const python     = resolvePythonExecutable(context.extensionPath);
            const scriptPath = resolveRefactorScript(context.extensionPath);

            // 1-indexed drop line in the reverted (original) document.
            // I is the character offset in the original document where the body was dropped.
            const insertLine = freshDoc.positionAt(I).line + 1;

            // Sources that belong to this file (the file being dragged in).
            const fileSources = matched.sources.filter(s => {
                const abs = resolvePath(s.file);
                return abs === freshDoc.uri.fsPath;
            });
            const sources = fileSources.length > 0 ? fileSources : matched.sources;

            // Signature = first line of the pre-computed extracted method code.
            const signatureLine = (matched.extracted_method?.code ?? '').split('\n')[0].trim();
            const serverUrl     = resolveServerUrl();
            const callRanges    = sources.map(s => s.range);
            const replacements  = sources.map(s => s.replacement_code ?? '');

            let applied = false;

            if (signatureLine) {
                // 1. Try REST server.
                try {
                    const serverUp = await checkServerHealth(serverUrl);
                    if (serverUp) {
                        log.appendLine(`[refactor] using server ${serverUrl}`);
                        const rewritten = await apiRefactorDirect(
                            serverUrl,
                            freshDoc.uri.fsPath,
                            signatureLine,
                            body,
                            callRanges,
                            insertLine,
                            replacements,
                        );
                        if (rewritten !== null) {
                            const tsEdit = new vscode.WorkspaceEdit();
                            tsEdit.replace(freshDoc.uri, wholeDocRange(freshDoc), rewritten);
                            applied = await vscode.workspace.applyEdit(tsEdit);
                        }
                    }
                } catch (err) {
                    log.appendLine(`[refactor] server failed: ${String(err)} — trying subprocess`);
                }

                // 2. Fall back to local tree-sitter subprocess.
                if (!applied && scriptPath) {
                    try {
                        log.appendLine('[refactor] using subprocess');
                        const rewritten = await runTreeSitterDirect(
                            python,
                            scriptPath,
                            freshDoc.uri.fsPath,
                            signatureLine,
                            body,
                            callRanges,
                            insertLine,
                            replacements,
                        );
                        if (rewritten !== null) {
                            const tsEdit = new vscode.WorkspaceEdit();
                            tsEdit.replace(freshDoc.uri, wholeDocRange(freshDoc), rewritten);
                            applied = await vscode.workspace.applyEdit(tsEdit);
                        }
                    } catch (err) {
                        log.appendLine(`[refactor] subprocess failed: ${String(err)} — falling back to pre-computed`);
                    }
                }
            }

            // 3. Final fallback: pre-computed refactoring from JSON data.
            if (!applied) {
                applied = await applyPrecomputedRefactoring(freshDoc, matched);
            }

            if (applied) {
                vscode.window.showInformationMessage(
                    `Clone Visualizer: extract method applied for ${matched.classid} — ${sources.length} clone site(s) updated.`
                );
            }
        });

        context.subscriptions.push(dragListener);
    });

    context.subscriptions.push(disposable);
}

// ── Shared apply helper (used by both click and future callers) ────────────────

async function runApplyExtractMethod(
    classid: string,
    recordMap: Map<string, CloneRecord>
): Promise<void> {
    const record = recordMap.get(classid);
    if (!record) { return; }

    const filesToWrite = (record.updated_files ?? [])
        .map(uf => ({
            src: (() => {
                const rel = relativeUnderBundledRefactorOut(uf.rewritten_file_path);
                const abs = path.join(REFACTOR_OUT, rel);
                return fs.existsSync(abs) ? abs : undefined;
            })(),
            dst: resolvePath(uf.file),
            label: path.basename(uf.file),
        }))
        .filter(({ src, dst }) => !!src && !!dst) as { src: string; dst: string; label: string }[];

    if (filesToWrite.length === 0) {
        vscode.window.showWarningMessage(`Clone Visualizer: no rewritten files found for ${classid}.`);
        return;
    }

    const labels = filesToWrite.map(f => f.label).join(', ');
    const answer = await vscode.window.showWarningMessage(
        `Apply "Extract Method" for clone group "${classid}"?`,
        {
            modal: true,
            detail: `${filesToWrite.length} file(s) will be modified:\n${labels}\n\nYou can use Ctrl+Z to undo after applying.`,
        },
        'Apply'
    );
    if (answer !== 'Apply') { return; }

    // Use WorkspaceEdit so the change lands on VS Code's undo stack (Ctrl+Z works)
    const edit = new vscode.WorkspaceEdit();
    for (const { src, dst } of filesToWrite) {
        const newContent = fs.readFileSync(src, 'utf8');
        const dstUri     = vscode.Uri.file(dst);
        const doc        = await vscode.workspace.openTextDocument(dstUri);
        const fullRange  = doc.validateRange(new vscode.Range(0, 0, doc.lineCount, 0));
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
    vscode.window.showInformationMessage(
        `✓ Extract method applied for ${classid}. Use Ctrl+Z to undo.`
    );
}

export function deactivate() {}
