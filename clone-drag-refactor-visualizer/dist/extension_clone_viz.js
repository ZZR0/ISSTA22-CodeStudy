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
// ── Extension entry point ─────────────────────────────────────────────────────
function activate(context) {
    console.log('Congratulations, your extension "clone-visualizer" is now active!');
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
        // 3. Parse all_refactor_results.json → D3 tree + lookup maps
        const dataPath = path.join(context.extensionPath, 'media', 'all_refactor_results.json');
        const records = JSON.parse(fs.readFileSync(dataPath, 'utf8'));
        const recordMap = new Map(records.map(r => [r.classid, r]));
        const treeData = parseCloneData(records);
        // 4. Send tree to webview
        setTimeout(() => {
            panel.webview.postMessage({ command: 'loadData', data: treeData });
        }, 500);
        // 5. Webview message handler (click interactions)
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
                await vscode.window.showTextDocument(doc, {
                    selection: new vscode.Range(startLine, 0, endLine, Number.MAX_SAFE_INTEGER),
                    viewColumn: vscode.ViewColumn.Beside,
                    preserveFocus: false,
                });
                return;
            }
            if (message.command === 'applyExtractMethod') {
                await runApplyExtractMethod(message.classid, recordMap);
            }
        }, undefined, context.subscriptions);
        // Tracks the last leaf node opened via the tree — used by drag listener
        // key: absolute file path  value: classid of the clone group
        const lastOpenedByFile = new Map();
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
            // ── Diagnostic log for every Java-file change ──────────────────
            if (doc.languageId === 'java' || doc.fileName.endsWith('.java')) {
                log.appendLine(`[drag] ${path.basename(doc.fileName)}  langId=${doc.languageId}` +
                    `  reason=${event.reason}  nChanges=${changes.length}`);
                changes.forEach((c, i) => {
                    log.appendLine(`  [${i}] rangeOffset=${c.rangeOffset} rangeLen=${c.rangeLength}` +
                        `  textLen=${c.text.length}  text=${JSON.stringify(c.text.slice(0, 60))}`);
                });
                log.show(true);
            }
            if (doc.languageId !== 'java' && !doc.fileName.endsWith('.java')) {
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