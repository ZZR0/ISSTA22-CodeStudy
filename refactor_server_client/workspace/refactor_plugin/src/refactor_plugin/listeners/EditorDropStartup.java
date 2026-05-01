package refactor_plugin.listeners;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;

import org.eclipse.jface.dialogs.MessageDialog;
import org.eclipse.swt.custom.StyledText;
import org.eclipse.swt.dnd.DND;
import org.eclipse.swt.dnd.DropTarget;
import org.eclipse.swt.dnd.DropTargetAdapter;
import org.eclipse.swt.dnd.DropTargetEvent;
import org.eclipse.swt.dnd.TextTransfer;
import org.eclipse.swt.widgets.Control;
import org.eclipse.swt.widgets.Display;
import org.eclipse.swt.widgets.Shell;
import org.eclipse.ui.IPartListener2;
import org.eclipse.ui.IStartup;
import org.eclipse.ui.IWorkbenchPart;
import org.eclipse.ui.IWorkbenchPartReference;
import org.eclipse.ui.IWorkbenchWindow;
import org.eclipse.ui.PlatformUI;
import org.eclipse.ui.texteditor.ITextEditor;

public class EditorDropStartup implements IStartup {

   @Override
   public void earlyStartup() {
      // Must run UI code on the UI thread
      Display.getDefault().asyncExec(() -> {
         IWorkbenchWindow window = PlatformUI.getWorkbench().getActiveWorkbenchWindow();
         if (window != null) {
            hookEditorListener(window);
         }
      });
   }

   private void hookEditorListener(IWorkbenchWindow window) {
      window.getPartService().addPartListener(new IPartListener2() {
         @Override
         public void partOpened(IWorkbenchPartReference partRef) {
            IWorkbenchPart part = partRef.getPart(false);
            // Check if the opened part is a Text/Java Editor
            if (part instanceof ITextEditor) {
               attachDropListener((ITextEditor) part);
            }
         }

         // Unused interface methods
         @Override
         public void partActivated(IWorkbenchPartReference partRef) {
         }

         @Override
         public void partBroughtToTop(IWorkbenchPartReference partRef) {
         }

         @Override
         public void partClosed(IWorkbenchPartReference partRef) {
         }

         @Override
         public void partDeactivated(IWorkbenchPartReference partRef) {
         }

         @Override
         public void partHidden(IWorkbenchPartReference partRef) {
         }

         @Override
         public void partVisible(IWorkbenchPartReference partRef) {
         }

         @Override
         public void partInputChanged(IWorkbenchPartReference partRef) {
         }
      });
   }

   private void attachDropListener(ITextEditor editor) {
      StyledText textWidget = (StyledText) editor.getAdapter(Control.class);

      if (textWidget != null && !textWidget.isDisposed()) {
         // Eclipse editors usually already have a DropTarget, so we append our listener to it
         DropTarget target = (DropTarget) textWidget.getData(DND.DROP_TARGET_KEY);
         if (target != null) {
            target.addDropListener(new DropTargetAdapter() {
               @Override
               public void drop(DropTargetEvent event) {
                  // Check if the dropped item is text (e.g., a code snippet)
                  if (TextTransfer.getInstance().isSupportedType(event.currentDataType)) {
                     String droppedSnippet = (String) event.data;
                     if (droppedSnippet != null && !droppedSnippet.isEmpty()) {
                        handleDropEvent(droppedSnippet, textWidget.getShell());
                     }
                  }
               }
            });
         }
      }
   }

   private void handleDropEvent(String snippet, Shell shell) {
      // 1. Fetch JSON from Django (Currently hitting ?hello)
      String jsonResponse = fetchJsonFromDjango();

      // 2. We truncate the snippet for the popup so it doesn't overflow the screen
      String preview = snippet.length() > 50 ? snippet.substring(0, 50) + "..." : snippet;

      // 3. Display the Native Popup
      MessageDialog.openInformation(shell, "Code Dropped & API Fetched", "Dropped Snippet:\n" + preview + "\n\nDjango Response:\n" + jsonResponse);
   }

   private String fetchJsonFromDjango() {
      try {
         HttpClient client = HttpClient.newHttpClient();
         HttpRequest request = HttpRequest.newBuilder().uri(URI.create("http://localhost:8000/getJSonValue?hello")).GET().build();

         HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString());
         return "Status Code: " + response.statusCode() + "\n" + response.body();

      } catch (Exception e) {
         return "Connection Failed. \nError: " + e.getMessage();
      }
   }
}