"""
Clone Visualizer REST server.

Exposes two endpoints used by the VS Code extension:
  POST /summarize        – code summarization (Dreamuno HF models)
  POST /refactor/direct  – tree-sitter extract-method refactoring

Start:
  python scripts/server.py [--host 127.0.0.1] [--port 5000]
  python scripts/server.py --reload          # hot-reload (development)

Both features are optional: the server starts even if one set of
dependencies is missing, and returns 503 for that endpoint only.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

# ── FastAPI / uvicorn ────────────────────────────────────────────────────────
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import uvicorn
except ImportError:
    sys.exit(
        "Missing server dependencies.\n"
        "  pip install fastapi uvicorn[standard] pydantic\n"
        "Or:  pip install -r scripts/requirements-summarizer.txt"
    )

# ── Sibling modules live in the same scripts/ directory ──────────────────────
_SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(_SCRIPTS_DIR))

# Summarizer (needs torch / transformers — optional)
try:
    from summarizer import summarize as _summarize        # type: ignore[import]
    _SUMMARIZE_OK = True
    _SUMMARIZE_ERR = ""
except Exception as _e:
    _SUMMARIZE_OK = False
    _SUMMARIZE_ERR = str(_e)

# Tree-sitter refactorer (needs tree-sitter-python — optional)
try:
    from extract_method_refactor_python import direct_rewrite as _direct_rewrite  # type: ignore[import]
    _REFACTOR_OK = True
    _REFACTOR_ERR = ""
except Exception as _e:
    _REFACTOR_OK = False
    _REFACTOR_ERR = str(_e)

# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Clone Visualizer API",
    description=(
        "REST back-end for the Clone Visualizer VS Code extension. "
        "Provides code summarization and tree-sitter based extract-method refactoring."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# ── Pydantic models ──────────────────────────────────────────────────────────

class SummarizeRequest(BaseModel):
    code: str
    model: str

class SummaryItem(BaseModel):
    summary: str
    probability: float

class RefactorDirectRequest(BaseModel):
    file: str                                      # absolute path to the source file
    signature: str                                 # "def extracted(a, b, c):"
    body: str                                      # clone body (the dragged code)
    call_ranges: List[str]                         # ["412-425", "355-368"]
    insert_line: int                               # 1-indexed line where user dropped
    replacement_codes: Optional[List[str]] = None  # pre-computed call strings per range

class RefactorDirectResponse(BaseModel):
    rewritten_source: str

class StatusResponse(BaseModel):
    status: str
    summarize: bool
    refactor: bool
    summarize_error: str
    refactor_error: str

# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Lightweight ping used by the extension to check if the server is running."""
    return {"status": "ok"}


@app.get("/status", response_model=StatusResponse)
def status():
    """Reports which features are available based on installed dependencies."""
    return StatusResponse(
        status="ok",
        summarize=_SUMMARIZE_OK,
        refactor=_REFACTOR_OK,
        summarize_error=_SUMMARIZE_ERR,
        refactor_error=_REFACTOR_ERR,
    )


@app.post("/summarize", response_model=List[SummaryItem])
def summarize_endpoint(req: SummarizeRequest):
    """
    Summarize a code snippet using a Dreamuno Hugging Face model.

    Request body:
      { "code": "<source code>", "model": "<hf_model_id>" }

    Response:
      [ { "summary": "...", "probability": 0.87 }, ... ]   (sorted by probability desc)
    """
    if not _SUMMARIZE_OK:
        raise HTTPException(
            status_code=503,
            detail=f"Summarizer unavailable: {_SUMMARIZE_ERR}",
        )
    try:
        results = _summarize(req.code, req.model)
        results = sorted(results, key=lambda x: x["probability"], reverse=True)
        return results
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/refactor/direct", response_model=RefactorDirectResponse)
def refactor_direct(req: RefactorDirectRequest):
    """
    Apply extract-method refactoring using tree-sitter at the user's drop location.

    Uses tree-sitter to:
      1. Find the enclosing function/class at *insert_line*.
      2. Compute the correct insertion byte (end of that scope).
      3. Determine proper indentation from the AST context.
      4. Replace each call-range with a function call.
      5. Splice the extracted method into the source.

    Request body:
      {
        "file":             "/abs/path/to/file.py",
        "signature":        "def extracted(a, b, c):",
        "body":             "<clone code>",
        "call_ranges":      ["412-425", "355-368"],
        "insert_line":      440,
        "replacement_codes": ["        ret = extracted(a, b, c)\\n", ...]  // optional
      }

    Response:
      { "rewritten_source": "<full rewritten file content>" }
    """
    if not _REFACTOR_OK:
        raise HTTPException(
            status_code=503,
            detail=f"Refactoring unavailable: {_REFACTOR_ERR}",
        )
    file_path = Path(req.file)
    if not file_path.is_absolute():
        raise HTTPException(status_code=400, detail="'file' must be an absolute path.")
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {req.file}")
    try:
        file_text = file_path.read_text(encoding="utf-8", errors="replace")
        rewritten = _direct_rewrite(
            file_text=file_text,
            signature_str=req.signature,
            body=req.body,
            call_ranges=req.call_ranges,
            insert_line=req.insert_line,
            replacement_codes=req.replacement_codes,
        )
        return RefactorDirectResponse(rewritten_source=rewritten)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Clone Visualizer REST server (FastAPI + uvicorn).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--host", default="127.0.0.1", help="Bind address.")
    ap.add_argument("--port", type=int, default=5000, help="TCP port.")
    ap.add_argument("--reload", action="store_true", help="Hot-reload on file changes (development).")
    args = ap.parse_args()

    print(f"Starting Clone Visualizer server on http://{args.host}:{args.port}")
    print(f"  Summarize  : {'available' if _SUMMARIZE_OK else 'UNAVAILABLE — ' + _SUMMARIZE_ERR}")
    print(f"  Refactoring: {'available' if _REFACTOR_OK  else 'UNAVAILABLE — ' + _REFACTOR_ERR}")
    print(f"  API docs   : http://{args.host}:{args.port}/docs")

    uvicorn.run(
        "server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        app_dir=str(_SCRIPTS_DIR),
    )


if __name__ == "__main__":
    main()
