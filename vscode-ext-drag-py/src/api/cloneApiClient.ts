import * as path from "path";
import * as vscode from "vscode";

export interface CloneApiOptions {
  serverUrl: string;
}

export type CloneRequest =
  | { kind: "all" }
  | { kind: "files"; files: string[] };

export async function fetchCloneResults(request: CloneRequest, options: CloneApiOptions): Promise<unknown> {
  const baseUrl = options.serverUrl.replace(/\/+$/g, "");

  if (request.kind === "all") {
    return requestJson(`${baseUrl}/get_clones`, {
      method: "GET",
    });
  }

  if (request.files.length === 1) {
    return fetchSingleFileCloneResults(baseUrl, request.files[0]);
  }

  return fetchMultipleFileCloneResults(baseUrl, request.files);
}

async function fetchSingleFileCloneResults(baseUrl: string, file: string): Promise<unknown> {
  const displayName = path.basename(file);
  try {
    return await requestJson(`${baseUrl}/get_clone?req_clone_file=${encodeURIComponent(file)}`, {
      method: "GET",
    });
  } catch (getError) {
    try {
      return await requestJson(`${baseUrl}/get_clone`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ req_clone_file: file }),
      });
    } catch {
      throw getError instanceof Error
        ? getError
        : new Error(`Failed to fetch clone results for ${displayName}`);
    }
  }
}

async function fetchMultipleFileCloneResults(baseUrl: string, files: string[]): Promise<unknown> {
  try {
    return await requestJson(`${baseUrl}/get_clone`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ req_clone_files: files }),
    });
  } catch (postError) {
    const params = new URLSearchParams();
    for (const file of files) {
      params.append("req_clone_files", file);
    }
    try {
      return await requestJson(`${baseUrl}/get_clone?${params.toString()}`, {
        method: "GET",
      });
    } catch {
      throw postError instanceof Error
        ? postError
        : new Error("Failed to fetch clone results for selected files");
    }
  }
}

async function requestJson(url: string, init: RequestInit): Promise<unknown> {
  const response = await fetch(url, init);
  if (!response.ok) {
    throw new Error(`HTTP ${response.status} ${response.statusText} from ${url}`);
  }

  const text = await response.text();
  if (!text.trim()) {
    return [];
  }

  try {
    return JSON.parse(text) as unknown;
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`Invalid JSON from ${url}: ${message}`);
  }
}

export function cloneServerUrl(): string {
  return vscode.workspace
    .getConfiguration("cloneVisualizer")
    .get<string>("serverUrl", "http://localhost:8000");
}
