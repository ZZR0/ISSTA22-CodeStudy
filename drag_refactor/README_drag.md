# Drag & Drop Refactoring Extension

Welcome to the project! This VS Code extension serves as the foundation for an experimental drag-and-drop code refactoring tool. 

Currently, the extension captures text movement within the editor and provides a custom Sidebar "Dropzone" for cross-file code snippets. Your primary mission is to build upon this foundation to implement a **Drag and Drop Extract Method Refactoring** feature.

---

## Getting Started

### Prerequisites
* **Node.js**: Ensure Node.js is installed on your machine.
* **VS Code**: You will need the standard VS Code desktop application.

### Installation & Running
1. Clone this repository to your local machine.
2. Open the project folder in VS Code.
3. Open your terminal and run `npm install` to install the required dependencies.
4. Run `npm run compile` to build the TypeScript files.
5. Press `F5` on your keyboard. This will launch a new "Extension Development Host" window with the extension loaded and ready to test.

### Testing the Current Features
Before writing new code, familiarize yourself with what is already working in the Extension Development Host window:
* **Same-File Move Detection:** Open a source file, highlight a block of code, and drag it to a new line. You should see a popup saying `Internal move captured: "..."`.
* **The Dropzone:** Highlight code, press `Cmd + Shift + R` (Mac) or `Ctrl + Shift + R` (Windows) to add it to the "Dropzone Snippets" sidebar. Drag it from the sidebar back into the editor.

---

## Current Architecture Overview

The core logic is located in `src/drag_extension.ts`. It relies on three main VS Code APIs:

* **`TreeDataProvider` & `TreeDragAndDropController`:** Manages the visual list of saved snippets in the Explorer sidebar and allows them to be dragged out.
* **`DocumentDropEditProvider`:** Intercepts drops *into* the text editor. We use a custom MIME type (`application/vnd.drag.dropzone`) as a "secret handshake" to identify when text is coming from our sidebar rather than an external file.
* **`workspace.onDidChangeTextDocument`:** A listener that acts as a detective. Because VS Code hides raw mouse events, this listener detects native, same-file drag-and-drop actions by looking for a specific pattern: exactly one deletion and one insertion of the same length occurring in a single event.

---

## Your Mission: Drag & Drop Extract Method

Currently, when a user drags code from inside a method and drops it into an empty space in the class, it simply moves the text. We want to intercept this action and turn it into a formal **Clone-Aware Extract Method Refactoring**.

**The Target Workflow:**
1. The user highlights code clone inside an existing method.
2. The user drags and drops the code outside of the method.
3. The extension intercepts the move.
4. A popup asks the user: *"Enter new method name:"*
5. The dragged text becomes the body of a newly created method at the drop location.
6. A method call replaces the original text at the source location.
7. Sibling clone instances are automatically updated as well.

### Implementation Roadmap

Here is a step-by-step guide on how to approach this feature using the existing `onDidChangeTextDocument` listener.

#### Step 1: Pause the Default Move
In our `onDidChangeTextDocument` listener, we already successfully detect when the move happens (the 1 deletion + 1 insertion pattern). Your first task is to figure out how to programmatically undo that raw text move so we can replace it with our formatted refactoring.
* *Hint:* Look into the `vscode.WorkspaceEdit` API.

#### Step 2: Prompt the User (Optional)
Once the move is detected and the code is captured in memory, use a default name `extract` (or ask the user for a new method name).
* *Hint:* Use `vscode.window.showInputBox()`.

#### Step 3: Format and Apply the Refactoring
Using the default method name (or user's input) and the captured text, construct the new strings.
* **Destination String:** (e.g., `private void [extract]() { \n [capturedText] \n }`)
* **Source String:** (e.g., `[extract]();`)
Apply these changes back to the document using `WorkspaceEdit`.

#### Step 4: Parameter Handling
Connect your existing refactoring engine to the VS Code extension. Pass the captured text and the active document's context to your engine so it can parse the AST, resolve local variables, and return the correct method signature parameters. Then, dynamically inject those parameters into your Destination and Source strings from Step 3.

#### Step 5: Sibling Clone Detection and Replacement
Feed the entire active document's text to your existing clone detection engine. Have your engine return the starting and ending line/character positions of all identified sibling clones. Iterate through these positions and use `WorkspaceEdit.replace()` to automatically swap each sibling instance with the new method call.

---
