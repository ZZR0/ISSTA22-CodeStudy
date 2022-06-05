# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.
import logging
import argparse
from bleu import _bleu
import json

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser(description='Evaluate leaderboard predictions for code completion (line level).')
    parser.add_argument('--expected', '-a', required=True, help="filename of the labels, in test format.")
    parser.add_argument('--predicted', '-p', required=True,
                        help="filename of the leaderboard predictions, in txt format.")
    args = parser.parse_args()

    preds = open(args.predicted, "r").readlines()
    gts = open(args.expected, "r").readlines()
    args.expected = "./concode/grandtrue.txt"
    expected = open("./concode/grandtrue.txt","w")
    assert len(preds) == len(gts), f"Samples of predictions and answers are not equal, {len(preds)}: {len(gts)}"

    total = len(gts)
    EM = 0.0
    for pred, gt in zip(preds, gts):
        pred = pred.strip()
        gt = json.loads(gt.strip())['code']
        pred = ' '.join([tok.strip() for tok in pred.split()])
        gt = ' '.join([tok.strip() for tok in gt.split()])
        expected.write(gt+"\n")

        if pred == gt:
            EM += 1

    print(args.expected, args.predicted)
    bleu_score = round(_bleu(args.expected, args.predicted), 2)
    print(f"BLEU: {bleu_score}, EM: {round(EM / total * 100, 2)}")


if __name__ == "__main__":
    main()
