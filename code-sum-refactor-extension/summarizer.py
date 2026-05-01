"""
Code summarization using Dreamuno Hugging Face checkpoints (two models).

The VS Code extension picks exactly one per run via --model (automatic):
  - Python file / Python-like selection → Dreamuno/llm-codeknowhow-python
  - Otherwise → Dreamuno/llm_codeknowhow_v1

CLI: python summarizer.py --json --stdin --model <hf_model_id>
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from typing import Dict, Tuple

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, RobertaTokenizer, T5ForConditionalGeneration


def _require_protobuf_or_exit() -> None:
    """Hugging Face / tokenizers need `protobuf` in the same env as this script."""
    try:
        import google.protobuf  # noqa: F401
    except ImportError:
        exe = sys.executable
        print(
            "\n=== Missing Python package: protobuf ===\n"
            "Install it into the interpreter VS Code uses for summarization (see Clone Visualizer → Python Path).\n\n"
            f'  "{exe}" -m pip install protobuf\n\n'
            "Or install all summarizer dependencies from the extension repo:\n\n"
            f'  "{exe}" -m pip install -r scripts/requirements-summarizer.txt\n',
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(1) from None


# CLI default when --model omitted (Java / general code; extension passes --model explicitly).
DEFAULT_MODEL = "Dreamuno/llm_codeknowhow_v1"
# Java checkpoint has a stable RobertaTokenizer on the Hub. The Python repo's tokenizer_config
# can trigger tokenizers.AddedToken errors; vocab matches the same T5 CodeKnowHow family.
DREAMUNO_JAVA_TOKENIZER_HUB = "Dreamuno/llm_codeknowhow_v1"

_model_cache: Dict[str, Tuple[object, object]] = {}


def _is_dreamuno_codeknowhow(model_name: str) -> bool:
    n = model_name.lower()
    return "dreamuno" in n and ("codeknowhow" in n or "llm_codeknowhow" in n or "llm-codeknowhow" in n)


def _python_model_needs_java_tokenizer(model_name: str) -> bool:
    return "llm-codeknowhow-python" in model_name.lower()


def _load_model(model_name: str) -> Tuple[object, object]:
    if model_name not in _model_cache:
        print(f"Loading {model_name} …", file=sys.stderr, flush=True)
        if _python_model_needs_java_tokenizer(model_name):
            print(
                f"Using tokenizer from {DREAMUNO_JAVA_TOKENIZER_HUB} (Python repo tokenizer_config breaks Rust path).",
                file=sys.stderr,
                flush=True,
            )
            tokenizer = RobertaTokenizer.from_pretrained(DREAMUNO_JAVA_TOKENIZER_HUB)
            model = T5ForConditionalGeneration.from_pretrained(model_name)
        elif _is_dreamuno_codeknowhow(model_name):
            tokenizer = RobertaTokenizer.from_pretrained(model_name)
            model = T5ForConditionalGeneration.from_pretrained(model_name)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        _model_cache[model_name] = (model, tokenizer)
    return _model_cache[model_name]


def summarize(code: str, model_name: str):
    model, tokenizer = _load_model(model_name)
    enc = tokenizer(
        code,
        return_tensors="pt",
        max_length=512,
        truncation=True,
    )
    source_ids = enc["input_ids"]
    attn = enc.get("attention_mask")
    source_mask = attn if attn is not None else source_ids.ne(tokenizer.pad_token_id)

    with torch.no_grad():
        preds = model.generate(
            source_ids,
            attention_mask=source_mask,
            max_length=80,
            min_length=10,
            length_penalty=1.0,
            num_beams=10,
            num_return_sequences=10,
            output_scores=True,
            return_dict_in_generate=True,
            early_stopping=True,
        )
    pred_nls = [
        tokenizer.decode(seq, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        for seq in preds.sequences
    ]
    probabilities = torch.softmax(preds.sequences_scores, dim=0).tolist()
    return [{"summary": s, "probability": p} for s, p in zip(pred_nls, probabilities)]


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize source code with a Dreamuno seq2seq model.")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print a JSON array of {summary, probability} to stdout (sorted by probability).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Hugging Face model id (default: {DEFAULT_MODEL}).",
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--stdin",
        action="store_true",
        help="Read UTF-8 source code from standard input.",
    )
    src.add_argument(
        "base64_input",
        nargs="?",
        help="Base64-encoded UTF-8 source (legacy CLI).",
    )
    args = parser.parse_args()
    _require_protobuf_or_exit()

    try:
        if args.stdin:
            code = sys.stdin.read()
        else:
            code = base64.b64decode(args.base64_input).decode("utf-8")

        results = summarize(code, args.model)
        results = sorted(results, key=lambda x: x["probability"], reverse=True)

        if args.json:
            cleaned = [
                {"summary": item["summary"].replace("\n", " ").strip(), "probability": item["probability"]}
                for item in results
            ]
            print(json.dumps(cleaned))
        else:
            for idx, item in enumerate(results, 1):
                summary = item["summary"].replace("\n", " ").strip()
                prob = item["probability"]
                print(f"{summary} (Probability: {prob:.4f})")
    except Exception as e:
        print(f"Error: {e!s}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
