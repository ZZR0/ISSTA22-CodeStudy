#!/usr/bin/env python3
from pathlib import Path
import argparse
import sys
import random

def read_lines(p: Path):
    if not p.exists():
        print(f"[WARN] missing: {p}", file=sys.stderr)
        return []
    return [ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]

def write_lines(p: Path, lines):
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", required=True,
                    help="repo directory containing repo subfolders (e.g., dataset/python)")
    ap.add_argument("--org_dir", required=True,
                    help="org directory (e.g., dataset/org)")
    ap.add_argument("--out_dir", required=True,
                    help="output directory")
    ap.add_argument("--seed", type=int, default=3,
                    help="shuffle seed")
    args = ap.parse_args()

    rnd = random.Random(args.seed)

    base = Path(args.base_dir)
    org  = Path(args.org_dir)
    out  = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    repos = sorted(p.name for p in base.iterdir() if p.is_dir())

    if not org.exists():
        print(f"[ERROR] org_dir not found: {org}", file=sys.stderr)
        sys.exit(1)

    # -------- train --------
    train_lines = []
    train_lines += read_lines(org / "train_10percent.txt")
    for r in repos:
        train_lines += read_lines(base / r / "train.txt")

    rnd.shuffle(train_lines)
    write_lines(out / "train_mix.txt", train_lines)

    # -------- valid --------
    valid_lines = []
    valid_lines += read_lines(org / "valid_10percent.txt")
    for r in repos:
        valid_lines += read_lines(base / r / "valid.txt")

    rnd.shuffle(valid_lines)
    write_lines(out / "valid_mix.txt", valid_lines)

    # -------- data.jsonl (NO shuffle) --------
    data_lines = []
    data_lines += read_lines(org / "data.jsonl")
    for r in repos:
        data_lines += read_lines(base / r / "data.jsonl")

    write_lines(out / "data_mix.jsonl", data_lines)

    print("[OK] Combined + shuffled dataset created")
    print(f"  train_mix.txt lines  : {len(train_lines)}")
    print(f"  valid_mix.txt lines  : {len(valid_lines)}")
    print(f"  data_mix.jsonl lines : {len(data_lines)}")
    print(f"  shuffle seed         : {args.seed}")
    print(f"  repos included       : {len(repos)} (org + {len(repos)} others)")

if __name__ == "__main__":
    main()
