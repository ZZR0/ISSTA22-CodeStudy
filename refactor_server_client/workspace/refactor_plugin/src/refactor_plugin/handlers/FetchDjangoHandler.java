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
      MessageDialog.openInformation(window.getShell(), "Django API Response", jsonResponse);

      return null;
   }

   private String fetchJsonFromDjango() {
      try {
         HttpClient client = HttpClient.newHttpClient();
         HttpRequest request = HttpRequest.newBuilder().uri(URI.create("http://localhost:8000/getJSonValue?hello")).GET().build();

         HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString());
         return "Status Code: " + response.statusCode() + "\n\n" + response.body();

      } catch (Exception e) {
         return "Connection Failed. Is your Django server running on port 8000?\n\nError: " + e.getMessage();
      }
   }
}