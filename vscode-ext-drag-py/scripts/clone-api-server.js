#!/usr/bin/env node

const http = require("http");
const fs = require("fs");
const path = require("path");

const root = path.resolve(__dirname, "..");
const port = Number(process.env.PORT || 8000);
const dataPath = process.env.CLONE_DATA ||
  path.join(root, "media", "all_refactor_results_py.json");

function loadRecords() {
  return JSON.parse(fs.readFileSync(dataPath, "utf8"));
}

function sendJson(res, status, payload) {
  const body = JSON.stringify(payload, null, 2);
  res.writeHead(status, {
    "content-type": "application/json; charset=utf-8",
    "access-control-allow-origin": "*",
    "access-control-allow-methods": "GET,POST,OPTIONS",
    "access-control-allow-headers": "content-type",
  });
  res.end(body);
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    let body = "";
    req.on("data", (chunk) => {
      body += chunk;
    });
    req.on("end", () => {
      if (!body.trim()) {
        resolve({});
        return;
      }
      try {
        resolve(JSON.parse(body));
      } catch (error) {
        reject(error);
      }
    });
    req.on("error", reject);
  });
}

function matchesFile(sourceFile, requestedFile) {
  const source = normalizePath(sourceFile);
  const requested = normalizePath(requestedFile);
  return source === requested ||
    source.endsWith(requested) ||
    requested.endsWith(source) ||
    source.endsWith(stripDatasetPrefix(requested)) ||
    requested.endsWith(stripDatasetPrefix(source));
}

function normalizePath(value) {
  return String(value || "").replace(/\\/g, "/");
}

function stripDatasetPrefix(value) {
  return value
    .replace(/^.*?systems-py\//, "systems-py/")
    .replace(/^.*?systems-java\//, "systems-java/")
    .replace(/^.*?systems\//, "systems/");
}

function filterRecordsByFiles(records, files) {
  const requested = files.filter(Boolean).map(normalizePath);
  if (requested.length === 0) {
    return [];
  }

  return records.filter((record) =>
    Array.isArray(record.sources) &&
    record.sources.some((source) =>
      requested.some((file) => matchesFile(source.file, file))
    )
  );
}

function responsePayload(records) {
  return {
    clone_detection_results: records,
    refactorability_analysis: records,
  };
}

const server = http.createServer(async (req, res) => {
  try {
    if (req.method === "OPTIONS") {
      sendJson(res, 204, {});
      return;
    }

    const url = new URL(req.url || "/", `http://${req.headers.host}`);
    const records = loadRecords();

    if (req.method === "GET" && url.pathname === "/") {
      sendJson(res, 200, {
        ok: true,
        message: "Clone API server is running.",
        records: records.length,
        dataPath,
        endpoints: {
          health: "GET /health",
          allClones: "GET /get_clones",
          oneFile: "GET /get_clone?req_clone_file=<path>",
          files: "POST /get_clone with { req_clone_file } or { req_clone_files }",
        },
      });
      return;
    }

    if (req.method === "GET" && url.pathname === "/health") {
      sendJson(res, 200, { ok: true, records: records.length, dataPath });
      return;
    }

    if (req.method === "GET" && url.pathname === "/get_clones") {
      sendJson(res, 200, responsePayload(records));
      return;
    }

    if (url.pathname === "/get_clone") {
      if (req.method === "GET") {
        const files = [
          ...url.searchParams.getAll("req_clone_files"),
          ...url.searchParams.getAll("req_clone_file"),
        ];
        sendJson(res, 200, responsePayload(filterRecordsByFiles(records, files)));
        return;
      }

      if (req.method === "POST") {
        const body = await readBody(req);
        const files = Array.isArray(body.req_clone_files)
          ? body.req_clone_files
          : [body.req_clone_file];
        sendJson(res, 200, responsePayload(filterRecordsByFiles(records, files)));
        return;
      }
    }

    sendJson(res, 404, { error: `No route for ${req.method} ${url.pathname}` });
  } catch (error) {
    sendJson(res, 500, { error: error instanceof Error ? error.message : String(error) });
  }
});

server.listen(port, "127.0.0.1", () => {
  console.log(`Clone API server listening on http://localhost:${port}`);
  console.log(`Serving ${dataPath}`);
  console.log("Endpoints: GET /health, GET /get_clones, GET/POST /get_clone");
});
