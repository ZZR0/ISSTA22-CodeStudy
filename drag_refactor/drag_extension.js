"use strict";
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
Object.defineProperty(exports, "__esModule", { value: true });
exports.activate = activate;
exports.deactivate = deactivate;
const vscode = __importStar(require("vscode"));
class DropItem extends vscode.TreeItem {
    label;
    content;
    constructor(label, content) {
        super(label, vscode.TreeItemCollapsibleState.None);
        this.label = label;
        this.content = content;
        this.tooltip = content;
        this.description = "Code Snippet";
    }
}
class DropzoneProvider {
    dropItems = [];
    _onDidChangeTreeData = new vscode.EventEmitter();
    onDidChangeTreeData = this._onDidChangeTreeData.event;
    dropMimeTypes = ['text/plain'];
    dragMimeTypes = ['text/plain'];
    getTreeItem(element) {
        return element;
    }
    getChildren(element) {
        if (element) {
            return [];
        }
        return this.dropItems;
    }
    async handleDrop(target, dataTransfer, token) {
        const textItem = dataTransfer.get('text/plain');
        if (!textItem) {
            return;
        }
        const textContent = await textItem.asString();
        const label = textContent.trim().substring(0, 20).replace(/\n/g, ' ') + '...';
        const newItem = new DropItem(label, textContent);
        this.dropItems.push(newItem);
        this._onDidChangeTreeData.fire();
        vscode.window.showInformationMessage(`Saved to Dropzone!`);
    }
    async handleDrag(source, dataTransfer, token) {
    }
}
function activate(context) {
    console.log('Dropzone Extension is now active!');
    const dropzoneProvider = new DropzoneProvider();
    const treeView = vscode.window.createTreeView('dragDropZone', {
        treeDataProvider: dropzoneProvider,
        dragAndDropController: dropzoneProvider
    });
    context.subscriptions.push(treeView);
}
function deactivate() { }
//# sourceMappingURL=drag_extension.js.map