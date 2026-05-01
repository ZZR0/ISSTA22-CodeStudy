/**
 * @file drag_extension.ts
 * @description A VS Code extension that creates a custom "Dropzone" in the Explorer sidebar.
 * Users can save selected code snippets to this sidebar (via command/shortcut) and later 
 * drag-and-drop them directly into any open text editor across different files.
 */

import * as vscode from 'vscode';

/**
 * Represents a single snippet item within the Dropzone sidebar.
 */
class DropItem extends vscode.TreeItem {
    constructor(public readonly label: string, public readonly content: string) {
        super(label, vscode.TreeItemCollapsibleState.None);
        // Show the full code snippet in a popup when the user hovers over the item
        this.tooltip = content;
        this.description = "";
    }
}

/**
 * Provides the data for the Dropzone TreeView and handles the drag/drop interactions
 * originating from the sidebar panel.
 */
class DropzoneProvider implements vscode.TreeDataProvider<DropItem>, vscode.TreeDragAndDropController<DropItem> {
    // Internal array holding all the saved code snippets
    private dropItems: DropItem[] = [];

    // Emitters to signal VS Code to refresh the UI when items are added/removed
    private _onDidChangeTreeData: vscode.EventEmitter<DropItem | undefined | void> = new vscode.EventEmitter<DropItem | undefined | void>();
    readonly onDidChangeTreeData: vscode.Event<DropItem | undefined | void> = this._onDidChangeTreeData.event;

    // Define the MIME types this controller accepts and provides
    dropMimeTypes = ['text/plain'];
    dragMimeTypes = ['text/plain']; 

    // --- TreeDataProvider Methods ---

    getTreeItem(element: DropItem): vscode.TreeItem {
        return element;
    }

    getChildren(element?: DropItem): vscode.ProviderResult<DropItem[]> {
        if (element) {
            // Return an empty array since our snippets don't have nested children
            return [];
        }
        return this.dropItems; 
    }

    /**
     * Programmatically adds a new snippet to the Dropzone.
     * Triggered via the custom context menu or keyboard shortcut.
     */
    public addSnippet(textContent: string) {
        // Create a short label from the text (first 20 chars, forced to a single line)
        const label = textContent.trim().substring(0, 20).replace(/\n/g, ' ') + '...';
        const newItem = new DropItem(label, textContent);
        
        this.dropItems.push(newItem);
        
        // Fire the event to refresh the sidebar UI with the new item
        this._onDidChangeTreeData.fire();
    }

    // --- TreeDragAndDropController Methods ---

    async handleDrop(target: DropItem | undefined, dataTransfer: vscode.DataTransfer, token: vscode.CancellationToken): Promise<void> {
        // Empty: We currently handle adding items via the right-click menu and shortcut instead
    }

    /**
     * Fired when the user clicks and drags an item OUT of the Dropzone sidebar.
     */
    async handleDrag(source: readonly DropItem[], dataTransfer: vscode.DataTransfer, token: vscode.CancellationToken): Promise<void> {
        if (source.length === 0) {
            return;
        }
        const draggedItem = source[0];
        
        // Standard text payload so the Monaco editor knows how to handle it natively
        dataTransfer.set('text/plain', new vscode.DataTransferItem(draggedItem.content));
        
        // Custom "Secret Handshake" payload so our EditorDropProvider can identify it
        dataTransfer.set('application/vnd.drag.dropzone', new vscode.DataTransferItem(draggedItem.content));
    }
}

/**
 * Intercepts items dropped directly into the text editor canvas.
 */
class EditorDropProvider implements vscode.DocumentDropEditProvider {
    async provideDocumentDropEdits(
        _document: vscode.TextDocument,
        _position: vscode.Position,
        dataTransfer: vscode.DataTransfer,
        _token: vscode.CancellationToken
    ): Promise<vscode.DocumentDropEdit | undefined> {
        
        // Look for the custom MIME type attached during handleDrag
        const dropzoneItem = dataTransfer.get('application/vnd.drag.dropzone');
        
        if (dropzoneItem) {
            const content = await dropzoneItem.asString();
            
            console.log('Captured drop from Sidebar to Editor!');
            vscode.window.showInformationMessage('Successfully dropped snippet from the Dropzone!');
            
            // Return the text edit so VS Code inserts the text exactly at the drop location
            return new vscode.DocumentDropEdit(content);
        }
        
        // Return undefined to let VS Code natively handle any other drops (like normal files)
        return undefined;
    }
}

/**
 * Called when the extension is activated. 
 * Wires all providers, commands, views, and event listeners together.
 */
export function activate(context: vscode.ExtensionContext) {
    console.log('Dropzone Extension is now active!');

    // Initialize the Dropzone data provider
    const dropzoneProvider = new DropzoneProvider();

    // Register the TreeView for the Explorer panel
    const treeView = vscode.window.createTreeView('dragDropZone', {
        treeDataProvider: dropzoneProvider,
        dragAndDropController: dropzoneProvider
    });

    // Register the command used by the right-click menu and keyboard shortcut
    const addSnippetCmd = vscode.commands.registerCommand('dragDropZone.addSnippet', () => {
        const editor = vscode.window.activeTextEditor;
        if (editor) {
            const selection = editor.selection;
            const text = editor.document.getText(selection);
            
            if (text) {
                dropzoneProvider.addSnippet(text);
                vscode.window.showInformationMessage('Added to Dropzone!');
            } else {
                vscode.window.showWarningMessage('Please highlight some text first.');
            }
        }
    });

    // Register the drop provider to catch snippets dropped into the text editor from the sidebar
    const dropProvider = vscode.languages.registerDocumentDropEditProvider(
        { language: '*' }, 
        new EditorDropProvider()
    );

    // --- Listen for native, same-file drag-and-drop events ---
    const textChangeListener = vscode.workspace.onDidChangeTextDocument(event => {
        const changes = event.contentChanges;

        // A native drag-and-drop move creates exactly two changes in a SINGLE event: 
        // A deletion of the highlighted text at the source, and an insertion at the destination.
        if (changes.length === 2) {
            
            // Find which change represents the deletion and which is the insertion
            const deletion = changes.find(c => c.text === '' && c.rangeLength > 0);
            const insertion = changes.find(c => c.text !== '' && c.rangeLength === 0);

            if (deletion && insertion) {
                // Verify that the amount of deleted text matches the amount inserted
                if (deletion.rangeLength === insertion.text.length) {
                    
                    console.log('Internal SAME-FILE Drag & Drop Detected!');
                    console.log(`Moved text: "${insertion.text}"`);
                    
                    // Show a pop-up to confirm we caught it
                    vscode.window.showInformationMessage(`Internal move captured: "${insertion.text}"`);
                }
            }
        }
    });

    // Add all disposables to the context subscriptions for proper memory cleanup
    context.subscriptions.push(dropProvider);    
    context.subscriptions.push(treeView);
    context.subscriptions.push(addSnippetCmd);
    context.subscriptions.push(textChangeListener);
}

export function deactivate() {}