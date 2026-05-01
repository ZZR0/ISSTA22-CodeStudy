# Eclipse Plug-in V0.1: Django API Client

This repository contains Version 0.1 of the `refactor_plugin`. It is a foundational Eclipse plug-in built using the standard Eclipse 3.x API. It adds a custom menu item to the Eclipse IDE that, when clicked, uses a native Java 11+ `HttpClient` to fetch JSON data from a local Django backend and displays the result in a native Eclipse popup dialog.

This serves as the initial bridge for sending and receiving code analysis data (like code clone detection or refactoring metrics) between the Eclipse frontend and the Python backend.

## Prerequisites

-   **Eclipse IDE** with the Plug-in Development Environment (PDE) installed.
    -   Ver 2025 12R
-   **Java Execution Environment:** Java 11 or higher (configured for JavaSE-21 in this project) to support `java.net.http.HttpClient`.
-   **Backend Server:** The `refactor_server` Django app must be running locally on port 8000, exposing the `/getJSonValue?hello` endpoint.

---

## Step 1: Create the Plug-in Project

1. In Eclipse, go to **File > New > Project...**
2. Select **Plug-in Project** under Plug-in Development and click **Next**.
3. Name the project `refactor_plugin`.
4. On the Content page, ensure "Generate an activator" is checked.
5. Uncheck "Create a plug-in using one of the templates" to create a clean, manual configuration. Click **Finish**.

![Screenshot: New Plug-in Project Wizard](./images/New%20Plug-in%20Project%20Wizard.jpg)

---

## Step 2: Configure Dependencies (`MANIFEST.MF`)

To access the Eclipse UI components and commands, we must add specific dependencies to the plug-in manifest.

1. Open `META-INF/MANIFEST.MF`.
2. Navigate to the **Dependencies** tab.
3. Under **Required Plug-ins**, add the following:
    - `org.eclipse.ui`
    - `org.eclipse.core.commands`
4. Navigate to the **Overview** tab.
5. Under **Execution Environments**, ensure it is set to `JavaSE-11` or higher (e.g., `JavaSE-21`).

_![Screenshot: MANIFEST.MF Dependencies Tab](./images/MANIFEST.MF%20Dependencies%20Tab.jpg)_

---

## Step 3: Wire the UI Components (`plugin.xml`)

Eclipse uses `plugin.xml` to define UI elements like menus and commands without hardcoding them into the Java source.

1. Open `plugin.xml` and switch to the source code tab at the bottom.
2. Replace the contents inside the `<plugin>` tags with the following XML:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<?eclipse version="3.4"?>
<plugin>

   <extension point="org.eclipse.ui.commands">
      <category id="refactor_plugin.category" name="Refactor Plugin Category"/>
      <command
            categoryId="refactor_plugin.category"
            id="refactor_plugin.command"
            name="Fetch Django JSON">
      </command>
   </extension>

   <extension point="org.eclipse.ui.handlers">
      <handler
            class="refactor_plugin.handlers.FetchDjangoHandler"
            commandId="refactor_plugin.command">
      </handler>
   </extension>

   <extension point="org.eclipse.ui.menus">
      <menuContribution locationURI="menu:org.eclipse.ui.main.menu?after=additions">
         <menu id="refactor_plugin.menu" label="Django API">
            <command
                  commandId="refactor_plugin.command"
                  id="refactor_plugin.menu.command">
            </command>
         </menu>
      </menuContribution>
   </extension>

</plugin>
```

_![Screenshot: plugin.xml Source View](./images/plugin.xml%20Source%20View.jpg)_

---

## Step 4: Implement the Handler (`FetchDjangoHandler.java`)

The handler defines the behavior that occurs when the user clicks the menu item.

1. Create a new package in the `src` folder named `refactor_plugin.handlers`.
2. Create a new class named `FetchDjangoHandler.java` and implement the `AbstractHandler` class:

```java
package refactor_plugin.handlers;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;

import org.eclipse.core.commands.AbstractHandler;
import org.eclipse.core.commands.ExecutionEvent;
import org.eclipse.core.commands.ExecutionException;
import org.eclipse.jface.dialogs.MessageDialog;
import org.eclipse.ui.IWorkbenchWindow;
import org.eclipse.ui.handlers.HandlerUtil;

public class FetchDjangoHandler extends AbstractHandler {

    @Override
    public Object execute(ExecutionEvent event) throws ExecutionException {
        // Get the active Eclipse window
        IWorkbenchWindow window = HandlerUtil.getActiveWorkbenchWindowChecked(event);

        // Fetch the JSON from your local Django server
        String jsonResponse = fetchJsonFromDjango();

        // Display the result in a native Eclipse popup window
        MessageDialog.openInformation(
                window.getShell(),
                "Django API Response",
                jsonResponse);

        return null;
    }

    private String fetchJsonFromDjango() {
        try {
            HttpClient client = HttpClient.newHttpClient();
            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create("http://localhost:8000/getJSonValue?hello"))
                    .GET()
                    .build();

            HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString());
            return "Status Code: " + response.statusCode() + "\n\n" + response.body();

        } catch (Exception e) {
            return "Connection Failed. Is your Django server running on port 8000?\n\nError: " + e.getMessage();
        }
    }
}
```

_![Screenshot: FetchDjangoHandler.java code](./images/FetchDjangoHandler.java%20code.jpg)_

---

## Step 5: Launch and Test

To test the plug-in inside a sandboxed Eclipse environment:

1. Right-click the `refactor_plugin` project.
2. Select **Run As > Eclipse Application**.
3. In the new Eclipse window, locate the **Django API** menu in the top menu bar.
4. Click **Fetch Django JSON**.

_![Screenshot: Eclipse Run Configurations Dialog](./images/Eclipse%20Run%20Configurations%20Dialog.jpg)_

### Expected Output

If the Django server is running correctly, a popup dialog will appear displaying a `200` status code and the successful JSON payload.

_![Screenshot: Final Output - The Native Eclipse Popup displaying the JSON](./images/Final%20Output%20-%20The%20Native%20Eclipse%20Popup%20displaying%20the%20JSON1.jpg)_

_![Screenshot: Final Output - The Native Eclipse Popup displaying the JSON](./images/Final%20Output%20-%20The%20Native%20Eclipse%20Popup%20displaying%20the%20JSON2.jpg)_

