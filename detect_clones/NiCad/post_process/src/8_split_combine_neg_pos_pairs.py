#!/usr/bin/env python3
"""
Split NiCad positive/negative sample files into train/valid/test and then
combine splits (neg+pos), shuffle, and write final datasets.

Input format: one sample per line (blank lines ignored).

Example:
  python split_combine_nicad.py \
    --neg /path/to/nicad_camel_neg_samples.txt \
    --pos /path/to/nicad_camel_pos_samples.txt \
    --out-dir ../data/java/camel \
    --seed 42
"""
import argparse
import random
from pathlib import Path
from typing import List, Tuple


def read_lines(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        return [line.rstrip("\n") for line in f if line.strip()]


def write_lines(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def split_50_25_25(lines: List[str], rng: random.Random) -> Tuple[List[str], List[str], List[str]]:
    rng.shuffle(lines)
    n = len(lines)

    n_train = n // 2
    n_valid = (n - n_train) // 2

    train = lines[:n_train]
    valid = lines[n_train:n_train + n_valid]
    test = lines[n_train + n_valid:]

    return train, valid, test


def main():
    parser = argparse.ArgumentParser(description="Split NiCad pos/neg samples and combine them.")
    parser.add_argument("--neg", required=True, type=Path, help="Path to negative samples txt")
    parser.add_argument("--pos", required=True, type=Path, help="Path to positive samples txt")
    parser.add_argument("--out_dir", required=True, type=Path, help="Output dir, e.g. ../data/java/camel")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    args = parser.parse_args()

    rng = random.Random(args.seed)

    neg_lines = read_lines(args.neg)
    pos_lines = read_lines(args.pos)

    neg_train, neg_valid, neg_test = split_50_25_25(neg_lines, rng)
    pos_train, pos_valid, pos_test = split_50_25_25(pos_lines, rng)

    # Combine + shuffle
    def combine(a: List[str], b: List[str], seed_offset: int) -> List[str]:
        combined = a + b
        random.Random(args.seed + seed_offset).shuffle(combined)
        return combined

    train = combine(neg_train, pos_train, 1)
    valid = combine(neg_valid, pos_valid, 2)
    test  = combine(neg_test,  pos_test,  3)

    write_lines(args.out_dir / "train.txt", train)
    write_lines(args.out_dir / "valid.txt", valid)
    write_lines(args.out_dir / "test.txt", test)

    print("✅ Done")
    print(f"Neg: {len(neg_lines)} → train={len(neg_train)}, valid={len(neg_valid)}, test={len(neg_test)}")
    print(f"Pos: {len(pos_lines)} → train={len(pos_train)}, valid={len(pos_valid)}, test={len(pos_test)}")
    print(f"Combined saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
