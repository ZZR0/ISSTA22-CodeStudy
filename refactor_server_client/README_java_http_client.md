# Java HTTP Client for `refactor_server` Django API

This directory contains a simple Java-based HTTP client designed to interact with the local `refactor_server` Django backend. It uses the standard `java.net.http.HttpClient` (available in Java 11+) to send a GET request and retrieve a JSON response, requiring no external dependencies.

## Prerequisites & Server Setup

Before running the Java client, ensure your Django backend ([server](https://github.com/unoselab/ManageCodeClones/tree/master/refactor_server)) is actively running.

1. **Start the Django Server:**
   Open a terminal, activate your Python environment, navigate to the `refactor_server` directory, and start the server on port 8000:
    ```bash
    conda activate django_env
    cd path/to/refactor_server
    python manage.py runserver 0.0.0.0:8000
    ```
2. **Verify the Endpoint:**
   Ensure the server is successfully listening at `http://localhost:8000/getJSonValue?hello`.
3. **Java Environment:**
   Java 11 or higher must be installed and available in your system's PATH.

## Project Structure

-   `refactor/java/client/MainHTTPClient.java`: The Java source code containing the HTTP client logic.
-   `run-java-client.sh`: A bash automation script that compiles the Java source file and executes the compiled class.

## Usage

To build and run the client, simply execute the provided bash script from the root of this client directory (e.g., `refactor_server_client`):

```bash
./run-java-client.sh
```

_(Note: If you receive a "Permission denied" error, ensure the script is executable by running `chmod +x run-java-client.sh` first)._

## Expected Output

Upon successful execution, the console will display the compilation status, the HTTP response code, and the raw JSON response pulled directly from your `refactor_server` backend:

```text
Compiling refactor/java/client/MainHTTPClient.java...
Compilation successful. Running the Java client...
--------------------------------------------------
Status Code: 200
JSON Response: {"status": "success", "message": "Hello, World! You passed the 'hello' parameter."}
--------------------------------------------------
```
