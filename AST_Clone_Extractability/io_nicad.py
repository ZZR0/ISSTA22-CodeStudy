# AST_Clone_Extractability/io_nicad.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Union


def load_nicad(path: str) -> List[Dict[str, Any]]:
    """
    Accepts:
      - single JSON object
      - JSON array
      - JSONL (one object per line)
    Returns list of clone-class objects.
    """
    raw = Path(path).read_text(encoding="utf-8", errors="replace").strip()
    if not raw:
        return []

    lines = [ln for ln in raw.splitlines() if ln.strip()]
    if len(lines) > 1 and all(ln.lstrip().startswith("{") for ln in lines):
        return [json.loads(ln) for ln in lines]

    obj = json.loads(raw)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        return [obj]
    raise ValueError("Unsupported JSON input format.")


def write_output(path: str, payload: Union[Dict[str, Any], List[Dict[str, Any]]], jsonl: bool = False) -> None:
    out = Path(path)
    if jsonl:
        if isinstance(payload, dict):
            payload = [payload]
        out.write_text("\n".join(json.dumps(o, ensure_ascii=False) for o in payload) + "\n", encoding="utf-8")
    else:
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
