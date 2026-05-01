"""
Code summarization using Dreamuno/llm_codeknowhow_v1 (T5 + Roberta tokenizer).
Invoked by the clone-visualizer extension via: python summarizer.py --json --stdin
"""
import argparse
import base64
import json
import sys

import torch
from transformers import T5ForConditionalGeneration, RobertaTokenizer

MODEL_NAME = "Dreamuno/llm_codeknowhow_v1"
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)


def summarize(code: str):
    source_ids = tokenizer.encode(
        code,
        return_tensors="pt",
        max_length=512,
        truncation=True,
    )
    source_mask = source_ids.ne(tokenizer.pad_token_id)

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
    parser = argparse.ArgumentParser(description="Summarize source code with the bundled T5 model.")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print a JSON array of {summary, probability} to stdout (sorted by probability).",
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

    try:
        if args.stdin:
            code = sys.stdin.read()
        else:
            code = base64.b64decode(args.base64_input).decode("utf-8")

        results = summarize(code)
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
