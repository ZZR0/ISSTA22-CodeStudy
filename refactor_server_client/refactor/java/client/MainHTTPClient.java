package refactor.java.client;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;

public class MainHTTPClient {
    public static void main(String[] args) {
        String url = "http://localhost:8000/getJSonValue?hello";

        HttpClient client = HttpClient.newHttpClient();

        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(url))
                .GET()
                .build();

        try {
            HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString());
            System.out.println("Status Code: " + response.statusCode());
            System.out.println("JSON Response: " + response.body());

        } catch (Exception e) {
            System.err.println("An error occurred: " + e.getMessage());
        }
    }
}