#!/bin/bash

# Define the source file and the fully qualified class name
SRC_FILE="refactor/java/client/MainHTTPClient.java"
MAIN_CLASS="refactor.java.client.MainHTTPClient"

echo "Compiling $SRC_FILE..."
javac $SRC_FILE

# The $? variable holds the exit status of the last command (0 means success)
if [ $? -eq 0 ]; then
    echo "Compilation successful. Running the Java client..."
    echo "--------------------------------------------------"
    java $MAIN_CLASS
    echo "--------------------------------------------------"
else
    echo "Compilation failed. Please check your Java syntax."
fi