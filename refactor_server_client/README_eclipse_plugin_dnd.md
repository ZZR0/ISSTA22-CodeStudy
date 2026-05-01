# Eclipse Plug-in V0.2: Drag-and-Drop (DND) API Client

**Note:** Please see [`README_eclipse_pluginV01.md`](README_eclipse_pluginV01.md) first for the initial setup, basic architecture, and prerequisites. 

This repository expands on V0.1 by implementing an **Early Startup Hook** and a **DropTargetListener**. Instead of relying on a menu click, this version silently attaches a listener to Eclipse's Java editors. When a user highlights code and drags-and-drops it within the editor, the plug-in intercepts the dropped text snippet, fetches data from the local Django API, and displays the result alongside the dropped code.

## Prerequisites
* Completion of the V0.1 setup (Project structure and basic `plugin.xml` configuration).
* Eclipse IDE with Plug-in Development Environment (PDE).
* The `refactor_server` Django backend running locally on port `8000`.

---

## Step 1: Add New Dependencies (`MANIFEST.MF`)

To interact with the Eclipse workbench lifecycle and text editors, additional dependencies are required.

1. Open `META-INF/MANIFEST.MF`.
2. Navigate to the **Dependencies** tab.
3. Under **Required Plug-ins**, add the following:
   * `org.eclipse.ui.editors`
   * `org.eclipse.jface.text`
   * `org.eclipse.core.runtime`

*![Screenshot: MANIFEST.MF Dependencies Tab with new additions](./images/MANIFEST.MF%20Dependencies%20Tab%20with%20new%20additions.jpg)*

---

## Step 2: Register the Startup Hook (`plugin.xml`)

We must tell Eclipse to execute our listener in the background as soon as the workbench starts.

1. Open `plugin.xml` and switch to the source code tab.
2. Add the `org.eclipse.ui.startup` extension point directly above the closing `</plugin>` tag:

```xml
   <extension point="org.eclipse.ui.startup">
      <startup class="refactor_plugin.listeners.EditorDropStartup"/>
   </extension>
```

---

## Step 3: Implement the Listener (`EditorDropStartup.java`)

This class waits for editors to open, locates their underlying text widget (`StyledText`), and attaches the drop logic to intercept code snippets.

1. In the `src` folder, create a new package named `refactor_plugin.listeners`.
2. Create a new Java class named `EditorDropStartup.java`.

```java
package refactor_plugin.listeners;

public class EditorDropStartup implements IStartup {
    ...
}
```

---

## Step 4: Launch and Test

1. Run the project as an **Eclipse Application**.
2. Open a Java Project and open a `.java` source file.
3. Highlight a block of code (e.g., `System.out.println("Hello World!");`).
4. **Drag** the highlighted block to an empty line and **drop** it.

### Expected Output
An information dialog will immediately appear containing:
1. A preview of the dropped code snippet.
2. The `200` status code and JSON response from the Django backend.

*![Screenshot Placeholder: Eclipse Java Editor showing the Drag-and-Drop Action and resulting 'Code Dropped & API Fetched' Popup](./images/Drag-and-Drop%20Action.jpg)*
