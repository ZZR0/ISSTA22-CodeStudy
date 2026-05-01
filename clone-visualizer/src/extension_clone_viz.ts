import * as vscode from 'vscode';
import * as fs from 'fs';
import * as path from 'path';

export function activate(context: vscode.ExtensionContext) {
    console.log('Congratulations, your extension "clone-visualizer" is now active!');

    let disposable = vscode.commands.registerCommand('clone-visualizer.show_code_clones', () => {
        // 1. Create the Webview Panel
        const panel = vscode.window.createWebviewPanel(
            'cloneVisualizer', // Internal ID
            'Code Clone Tree', // Title of the tab
            vscode.ViewColumn.One,
            {
                enableScripts: true, // Allow D3.js to run
                localResourceRoots: [vscode.Uri.file(path.join(context.extensionPath, 'media'))]
            }
        );

        // 2. Read the HTML file from the media folder
        const htmlPath = path.join(context.extensionPath, 'media', 'collapsible-tree.html');
        let htmlContent = fs.readFileSync(htmlPath, 'utf8');
        
        // Assign HTML to the webview
        panel.webview.html = htmlContent;

        // 3. Read the data.json file
        const dataPath = path.join(context.extensionPath, 'media', 'data.json');
        const rawData = fs.readFileSync(dataPath, 'utf8');
        const jsonData = JSON.parse(rawData);

        // 4. Send the data to the webview once it's ready
        // We use a slight timeout to ensure the webview DOM has loaded before sending
        setTimeout(() => {
            panel.webview.postMessage({ command: 'loadData', data: jsonData });
        }, 500);
    });

    context.subscriptions.push(disposable);
}

export function deactivate() {}